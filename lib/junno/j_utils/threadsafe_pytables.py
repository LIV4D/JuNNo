import threading
import functools

import tables
import tables.file as _tables_file


class ThreadsafeFileRegistry(_tables_file._FileRegistry):
    lock = threading.RLock()

    @property
    def handlers(self):
        return self._handlers.copy()

    def add(self, handler):
        with self.lock:
            return super().add(handler)

    def remove(self, handler):
        with self.lock:
            return super().remove(handler)

    def close_all(self):
        with self.lock:
            return super().close_all()


class ThreadsafeFile(_tables_file.File):
    def __init__(self, *args, **kargs):
        with ThreadsafeFileRegistry.lock:
            super().__init__(*args, **kargs)

    def close(self):
        with ThreadsafeFileRegistry.lock:
            super().close()


@functools.wraps(tables.open_file)
def synchronized_open_file(*args, **kwargs):
    with ThreadsafeFileRegistry.lock:
        return _tables_file._original_open_file(*args, **kwargs)


# monkey patch the tables package
_tables_file._original_open_file = _tables_file.open_file
_tables_file.open_file = synchronized_open_file
tables.open_file = synchronized_open_file

_tables_file._original_File = _tables_file.File
_tables_file.File = ThreadsafeFile
tables.File = ThreadsafeFile

_tables_file._open_files = ThreadsafeFileRegistry()


# -------------------------------------------------------------------------------------
_pytables_files = {}


def open_pytable(path):
    import os
    from os.path import abspath, dirname
    path = abspath(path)
    f = _pytables_files.get(path, None)
    if f is not None and f() is not None:
        return f()

    import weakref
    os.makedirs(dirname(path), exist_ok=True)

    f = tables.open_file(path, mode='a')
    _pytables_files[path] = weakref.ref(f)

    return f
