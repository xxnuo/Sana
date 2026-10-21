"""Release cold checkpoint pages before the next Thor CUDA allocation."""
from contextlib import contextmanager
import os


def discard_file_cache(filename):
    with open(filename, "rb") as file:
        os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


@contextmanager
def pread_open(safe_open, filename, *args, **kwargs):
    try:
        with safe_open(filename, *args, **dict(kwargs, backend="pread")) as reader:
            yield reader
    finally:
        # pread tensors own their storage; no tensor depends on this file cache.
        discard_file_cache(filename)
