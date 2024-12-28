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


def open_pytable(path, mode='r'):
    from os.path import abspath
    import time
    from ..j_utils import log
    try:
        f = tables.open_file(abspath(path), mode)
    except ValueError as e:
        if mode == 'r':
            log.error('File %s is already opened in write mode. Close it to proceed. (15s before retry...)' % path)
        else:
            log.error('File %s is already opened. Close it to proceed. (15s before retry...)' % path)
        time.sleep(15)
        f = tables.open_file(abspath(path), mode)

    return f
