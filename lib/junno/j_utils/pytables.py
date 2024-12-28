import tables
from os.path import abspath
from .. import log
import time


class H5PyTable:
    def __init__(self):
        pass

    def populate_from_numpy(self, h5_path, table_fullname, data_dict):
        table_path, table_name = table_fullname.rsplit('/', maxsplit=1)
        f = open_h5(h5_path, mode='a')

        if not table_path.startswith('/'):
            table_path = '/' + table_path
        f.create_table(table_path, table_name, obj=data_dict, createparents=True)
        f.flush()
        f.close()
        return H5PyTable(h5_path, table_fullname)

    def __enter__(self):
        pass

    def __exit__(self):
        pass


class H5PyTableFile(H5PyTable):
    def __init__(self, hdf_path, table_path, mode='r'):
        super(H5PyTableFile, self).__init__()
        if table_path.startswith('/'):
            table_path, table_name = table_path.rsplit('/', maxsplit=1)
            table_path += '/'
        else:
            table_name = table_path
            table_path = '/'

        self.table_path = table_path
        self.table_name = table_name
        self.h5_path = hdf_path

        self.h5_file = None
        self._table = None
        self._mode = mode

    @property
    def mode(self):
        return self._mode

    def open(self, mode=None):
        if mode is None:
            mode = self.mode
        if self.h5_file is not None:
            if mode != self._mode:
                raise RuntimeError('File %s is currently opened in %s mode. Impossible to open it in %s mode.' % (
                    self.h5_path, self._mode, mode))
            else:
                return self._table
        self._mode = mode

        # --- Open file ---
        self.h5_file = open_h5(self.h5_path)
        try:
            self._table = self.h5_file.get_node(self.table_path, self.table_name, 'Table')
        except tables.NoSuchNodeError:
            self.close()
            raise ValueError('Table %s not found in %s.' % (self.table_path + self.table_name, self.h5_path))

    def close(self):
        if self.h5_file is None:
            return
        self.h5_file.close()
        self.h5_file = None
        if self._table is not None:
            del self._table
            self._table = None

    def __enter__(self):
        if self.h5_file is not None:
            raise RuntimeError('File %s is currently opened.' % self.h5_path)
        return self.open(self._mode)

    def __exit__(self):
        self.close()


def open_h5(path, mode='r'):
    i=10
    error = ""
    while i > 0:
        try:
            hdf = tables.open_file(abspath(path), mode)
            break
        except ValueError as e:
            if mode == 'r':
                error = 'File %s is already opened in write mode. Close it to proceed.' % path
            else:
                log.error('File %s is already opened. Close it to proceed. (6s before retry, %i trìes left)' % (
                    path, i))
            log.error(error + ' (6s before retry, %i attempts left)' % i)
            i -= 1
            time.sleep(6)
    else:
        raise RuntimeError(error)
    return hdf