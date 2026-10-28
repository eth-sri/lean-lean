"""Exercise the compiled relay against loopback sockets; no model requests."""
from contextlib import contextmanager
from pathlib import Path
import os
import signal
import socket
import subprocess
import time

import pytest


@pytest.fixture(scope="module")
def relay_binary(tmp_path_factory):
    binary = tmp_path_factory.mktemp("relay") / "relay"
    source = Path(__file__).resolve().parents[1] / "docker/model-relay/relay.c"
    subprocess.run(["cc", "-O2", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)], check=True)
    return binary


@contextmanager
def relay_pair(binary, timeout=1):
    with socket.socket() as server, socket.socket() as reservation:
        server.bind(("127.0.0.1", 0))
        server.listen(8)
        server.settimeout(4)
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
        reservation.close()
        args = [str(binary), "127.0.0.1", str(server.getsockname()[1]), str(port)]
        if timeout is not None:
            args.append(str(timeout))
        process = subprocess.Popen(args, start_new_session=True, stderr=subprocess.PIPE)
        client = None
        upstream = None
        try:
            deadline = time.monotonic() + 3
            while client is None:
                try:
                    client = socket.create_connection(("127.0.0.1", port), timeout=3)
                except ConnectionRefusedError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
            upstream, _ = server.accept()
            upstream.settimeout(3)
            yield client, upstream, process, server, port
        finally:
            if client:
                client.close()
            if upstream:
                upstream.close()
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
            process.stderr.close()


def recv_exact(sock, size):
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        assert chunk, "unexpected EOF"
        data += chunk
    return data


def test_silent_partial_stream_closes_both_sides_and_listener_survives(relay_binary):
    with relay_pair(relay_binary) as (client, upstream, process, server, port):
        partial = b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial"
        upstream.sendall(partial)
        assert recv_exact(client, len(partial)) == partial
        started = time.monotonic()
        assert client.recv(1) == b""
        assert .65 <= time.monotonic() - started < 2.5
        assert upstream.recv(1) == b""
        assert process.poll() is None
        with socket.create_connection(("127.0.0.1", port), timeout=3) as second:
            other, _ = server.accept()
            with other:
                second.sendall(b"new request")
                assert recv_exact(other, 11) == b"new request"


def test_stream_activity_resets_timeout_and_preserves_bytes(relay_binary):
    with relay_pair(relay_binary) as (client, upstream, _, _, _):
        for index in range(6):
            data = (str(index) * 10000).encode()
            client.sendall(data)
            assert recv_exact(upstream, len(data)) == data
            upstream.sendall(data)
            assert recv_exact(client, len(data)) == data
            time.sleep(.25)
        assert client.recv(1) == b""


def test_legacy_caller_has_no_inactivity_timeout(relay_binary):
    with relay_pair(relay_binary, timeout=None) as (client, upstream, _, _, _):
        time.sleep(1.25)
        client.sendall(b"still open")
        assert recv_exact(upstream, 10) == b"still open"
        upstream.sendall(b"ok")
        assert recv_exact(client, 2) == b"ok"


def test_request_half_close_still_delivers_response(relay_binary):
    with relay_pair(relay_binary) as (client, upstream, _, _, _):
        client.sendall(b"request")
        client.shutdown(socket.SHUT_WR)
        assert recv_exact(upstream, 7) == b"request"
        assert upstream.recv(1) == b""
        upstream.sendall(b"response")
        upstream.shutdown(socket.SHUT_WR)
        assert recv_exact(client, 8) == b"response"
        assert client.recv(1) == b""


@pytest.mark.parametrize("value", ["0", "-1", "86401", "1x", "", "9999999999999999999999"])
def test_rejects_invalid_timeout(relay_binary, value):
    result = subprocess.run([str(relay_binary), "127.0.0.1", "1", "2", value], capture_output=True)
    assert result.returncode == 64
