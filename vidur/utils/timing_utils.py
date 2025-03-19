import time
from contextlib import contextmanager


@contextmanager
def timeit(name='Unnamed code block'):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f'[{name}] execution time: {elapsed_time}')
