#define _POSIX_C_SOURCE 200809L

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <sys/wait.h>
#include <unistd.h>

static int parse_port(const char *text) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 1 || value > 65535) {
        return -1;
    }
    return (int)value;
}

static int64_t monotonic_ms(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        _exit(2);
    }
    return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

static int connect_target(const char *host, int port, int idle_ms) {
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &address.sin_addr) != 1) {
        return -1;
    }
    int candidate = socket(AF_INET, SOCK_STREAM, 0);
    if (candidate < 0) {
        return -1;
    }
    int flags = fcntl(candidate, F_GETFL, 0);
    if (idle_ms > 0 && (flags < 0 || fcntl(candidate, F_SETFL, flags | O_NONBLOCK) < 0)) {
        close(candidate);
        return -1;
    }
    if (connect(candidate, (struct sockaddr *)&address, sizeof(address)) != 0) {
        if (idle_ms == 0 || errno != EINPROGRESS) {
            close(candidate);
            return -1;
        }
        struct pollfd target = {.fd = candidate, .events = POLLOUT};
        int64_t deadline = monotonic_ms() + idle_ms;
        int ready;
        do {
            int64_t remaining = deadline - monotonic_ms();
            ready = poll(&target, 1, remaining > 0 ? (int)remaining : 0);
        } while (ready < 0 && errno == EINTR);
        int error = 0;
        socklen_t length = sizeof(error);
        if (ready <= 0 || getsockopt(candidate, SOL_SOCKET, SO_ERROR, &error, &length) != 0 || error) {
            close(candidate);
            return -1;
        }
    }
    if (idle_ms > 0 && fcntl(candidate, F_SETFL, flags) < 0) {
        close(candidate);
        return -1;
    }
    return candidate;
}

static int send_all(int destination, const char *buffer, size_t length) {
    size_t sent = 0;
    while (sent < length) {
        ssize_t count = send(destination, buffer + sent, length - sent, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return -1;
        }
        sent += (size_t)count;
    }
    return 0;
}

static void relay_connection(int client, const char *host, int port, int idle_ms) {
    int upstream = connect_target(host, port, idle_ms);
    if (upstream < 0) {
        close(client);
        _exit(2);
    }

    if (idle_ms > 0) {
        struct timeval timeout = {.tv_sec = idle_ms / 1000, .tv_usec = 0};
        if (setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0 ||
            setsockopt(upstream, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
            close(client);
            close(upstream);
            _exit(2);
        }
    }
    const int sockets[2] = {client, upstream};
    struct pollfd streams[2] = {
        {.fd = client, .events = POLLIN},
        {.fd = upstream, .events = POLLIN},
    };
    char buffer[65536];
    int open_streams = 2;
    int64_t last_activity = monotonic_ms();

    while (open_streams > 0) {
        int remaining = -1;
        if (idle_ms > 0) {
            int64_t delta = idle_ms - (monotonic_ms() - last_activity);
            remaining = delta > 0 ? (int)delta : 0;
        }
        int ready = poll(streams, 2, remaining);
        if (ready == 0) {
            fprintf(stderr, "relay: closing connection after %d seconds without application bytes\n", idle_ms / 1000);
            break;
        }
        if (ready < 0 && errno == EINTR) {
            continue;
        }
        if (ready < 0) {
            break;
        }

        for (int index = 0; index < 2; index++) {
            if (streams[index].fd < 0 || !(streams[index].revents & (POLLIN | POLLHUP | POLLERR))) {
                continue;
            }
            int source = streams[index].fd;
            int destination = sockets[1 - index];
            ssize_t count = recv(source, buffer, sizeof(buffer), 0);
            if (count > 0 && destination >= 0) {
                if (send_all(destination, buffer, (size_t)count) == 0) {
                    last_activity = monotonic_ms();
                    continue;
                }
                goto finished;
            }
            if (destination >= 0) {
                shutdown(destination, SHUT_WR);
            }
            // EOF closes only this read direction; the peer may still reply.
            streams[index].fd = -1;
            open_streams--;
        }
    }

finished:
    close(client);
    close(upstream);
    _exit(0);
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: relay TARGET_HOST TARGET_PORT LISTEN_PORT [IDLE_TIMEOUT_SECONDS]\n");
        return 64;
    }
    int target_port = parse_port(argv[2]);
    int listen_port = parse_port(argv[3]);
    if (target_port < 0 || listen_port < 0) {
        fprintf(stderr, "ports must be integers in 1..65535\n");
        return 64;
    }

    int idle_ms = 0;
    if (argc == 5) {
        char *end = NULL;
        errno = 0;
        long seconds = strtol(argv[4], &end, 10);
        if (errno != 0 || end == argv[4] || *end != '\0' || seconds < 1 || seconds > 86400) {
            fprintf(stderr, "idle timeout must be an integer in 1..86400 seconds\n");
            return 64;
        }
        idle_ms = (int)seconds * 1000;
    }

    signal(SIGPIPE, SIG_IGN);
    signal(SIGCHLD, SIG_IGN);

    int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) {
        perror("socket");
        return 1;
    }
    int enabled = 1;
    if (setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)) != 0) {
        perror("setsockopt");
        return 1;
    }

    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons((uint16_t)listen_port);
    if (bind(listener, (struct sockaddr *)&address, sizeof(address)) != 0) {
        perror("bind");
        return 1;
    }
    if (listen(listener, 128) != 0) {
        perror("listen");
        return 1;
    }

    for (;;) {
        int client = accept(listener, NULL, NULL);
        if (client < 0 && errno == EINTR) {
            continue;
        }
        if (client < 0) {
            perror("accept");
            return 1;
        }
        pid_t child = fork();
        if (child < 0) {
            close(client);
            continue;
        }
        if (child == 0) {
            close(listener);
            relay_connection(client, argv[1], target_port, idle_ms);
        }
        close(client);
    }
}
