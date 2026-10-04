import time
import functools


def retry(n_attempts):
    """
    Decorator factory: returns a decorator that retries a function
    up to n_attempts times.
    """
    if n_attempts < 1:
        raise ValueError("n_attempts must be at least 1")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(n_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(
                        f"Attempt {attempt + 1}/{n_attempts} failed: {type(e).__name__}: {e}"
                    )

                    if attempt < n_attempts - 1:
                        print("Retrying in 1 second...")
                        time.sleep(1)

            raise last_exception

        return wrapper

    return decorator

