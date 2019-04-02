import os
from os.path import dirname, join, abspath
import tempfile
import fcntl
import shutil
from zipfile import ZipFile, ZIP_DEFLATED


def abs_path(path, f=None):
    """
    Return path in an absolute path. If path is relative, f will be used as the origin path.
    """
    if f is None:
        return abspath(path)
    dir = dirname(abspath(f))
    path = join(dir, path)
    if ':' in path:  # Handling Windows path style
        path = '///' + path.replace('\\', "/")
    return path


_pytables_files = {}


def format_filepath(path, extension=None, origin_path=None, exists=True):
    if path.endswith('/'):
        raise ValueError('%s is not a valid file name' % path)
    if extension and not path.endswith('.'+extension):
        path += '.'+extension

    apath = abs_path(path, f=origin_path)
    if (exists and not os.path.exists(apath)) or os.path.isdir(apath):
        raise ValueError('%s not found.' % path)
    return apath


def open_pytable(path):
    path = abspath(path)
    f = _pytables_files.get(path, None)
    if f is not None and f() is not None:
        return f()

    import tables
    import weakref
    os.makedirs(dirname(path), exist_ok=True)
    f = tables.open_file(path, mode='a')
    _pytables_files[path] = weakref.ref(f)

    return f


class ZippedProject:
    def __init__(self, path, tmp_suffix, auto_recover=True, overwrite=False):
        self._path = path
        self._tmp_path = os.path.join(tempfile.gettempdir(), tmp_suffix)

        self._opened = False
        if auto_recover:
            self._opened = os.path.exists(self.tmp_path)
        self._keep_opened = []

        if overwrite or not os.path.exists(path):
            if self._opened:
                self.save_copy(path)
            else:
                with ZipFile(file=path, mode='w'):
                    pass

    #   --- Temporary folder ---
    @property
    def path(self):
        return self._path

    @property
    def tmp_path(self):
        return self._tmp_path

    def open_tmp(self):
        """
        Open the zip archive in a temporary folder. In general its better to use the with syntax.
        ZipFile.close() should be called after usage.
        """
        if self._opened:
            return

        # Test lock
        if os.path.exists(self.tmp_path):
            raise RuntimeError('Temporary folder is already in use. (path is: %s)\n'
                               'This happen after crash. Please use recover() to clear it.'
                               % self.tmp_path)

        # Extract to tmp
        if not os.path.exists(self.tmp_path):
            os.mkdir(self.tmp_path)

        zipf = ZipFile(self.path, 'r')
        zipf.extractall(self.tmp_path)
        zipf.close()

        self._opened = True

    def close(self, save=True):
        """
        Save the content of the temporary folder to the zipped archive and delete the folder.
        :param save:
        :return:
        """
        if not self._opened:
            return

        # Save from tmp
        if save:
            self._save_tmp()
        self._close_tmp()
        self._opened = False

    def recover(self, path=None):
        if self._opened:
            return
        if not os.path.exists(self.tmp_path):
            raise RuntimeError('No left temporary folder was found to recover.')
        if path is not None:
            self._opened = True
            self.save_copy(path)
        self._close_tmp()

    def _save_tmp(self):
        self.save_copy(self.path)

    def _close_tmp(self):
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def save_copy(self, path):
        """
        Save a copy of the archive to a given path.
        :param path: Path where to save the archive.
        """
        if self._opened:
            zipf = ZipFile(path, 'w', ZIP_DEFLATED)
            for root, dirs, files in os.walk(self.tmp_path):
                for file in files:
                    path = join(root, file)
                    arcname = os.path.relpath(path, start=self.tmp_path)
                    zipf.write(path, arcname=arcname)
            zipf.close()
        else:
            from shutil import copyfile
            copyfile(self.path, path)

    def __enter__(self):
        self._keep_opened.append(self._opened)
        self.open_tmp()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._keep_opened.pop():
            self.close()

    #   --- Access data ---
    def __getitem__(self, item):
        return self.read(item)

    def __setitem__(self, key, value):
        self.write(key, str(value), append=False)

    def __delitem__(self, key):
        self.delete(key)

    def _format_arcname(self, arcname, exists=False):
        if isinstance(arcname, tuple):
            arcname = join(*arcname)
        if not isinstance(arcname, str):
            raise ValueError('Invalid archive name: %s' % repr(arcname))

        if exists:
            if arcname not in self.namelist():
                raise ValueError('%s does not exist in archive.' % arcname)

        return arcname

    def read(self, arcname):
        """
        Read a file from the archive
        :param arcname: File name in the archive of the file to read
        :return: ByteStream as a string
        """
        arcname = self._format_arcname(arcname)
        if self._opened:
            path = join(self.tmp_path, arcname)
            with open(path, 'r') as f:
                d = f.read()
            return d
        else:
            with ZipFile(self.path, 'r') as f:
                d = f.read(arcname)
            return d

    def write(self, arcname, data, append=True):
        """
        Write data to a file in the archive
        :param arcname: File name in the archive of the file to write in
        :param data: Data to write
        :param append: If true the data will be append to the existing file content
        """
        arcname = self._format_arcname(arcname)
        if self._opened:
            path = join(self.tmp_path, arcname)
            with open(path, 'a' if append else 'w') as file:
                file.write(data)
        else:
            with ZipFile(self.path, 'w') as f:
                f.writestr(arcname, data)

    def open(self, arcname, mode='r'):
        """
        Open a file-like IOStream to read or write a file from the archive.
        The return file should be closed after usage (with file.close())
        :param arcname: File name in the archive of the file to access
        :param mode: Access Mode
        :return: File-like IOStream
        """
        arcname = self._format_arcname(arcname)
        if self._opened:
            path = join(self.tmp_path, arcname)
            return open(path, mode=mode)
        else:
            fzip = ZipFile(self.path, mode=mode)
            return fzip.open(arcname, mode=mode)

    def delete(self, arcname):
        """
        Delete a file or a folder from the archive
        """
        arcname = self._format_arcname(arcname)
        with self:
            path = join(self.tmp_path, arcname)
            os.remove(path)

    def extract(self, arcname, path):
        """
        Extract a file from the archive to the disk
        :param arcname: File name in the archive
        :param path: Path where the file should be extracted
        """
        arcname = self._format_arcname(arcname)
        if self._opened:
            from shutil import copyfile
            copyfile(join(self.tmp_path, arcname), path)
        else:
            with ZipFile(self.path) as fzip:
                fzip.extract(arcname, path=path)

    #   --- List content ---
    def namelist(self):
        if self._opened:
            r = []

            def recursive_namelist(addr, d):
                for k, v in d.items():
                    k_addr = join(addr, k)
                    if v is None:
                        r.append(k_addr)
                    else:
                        recursive_namelist(k_addr, v)

            recursive_namelist('', self.namedict())
            return r
        else:
            with ZipFile(self.path, 'r') as fzip:
                r = fzip.namelist()
            return r

    def namedict(self):
        if self._opened:
            def recursive_listdir(path):
                dictionary = {}
                for fpath in os.listdir(path):
                    fname = os.path.basename(fpath)
                    if os.path.isdir(fpath):
                        dictionary[fname] = recursive_listdir(fpath)
                    else:
                        dictionary[fname] = None
                return dictionary

            return recursive_listdir(self.tmp_path)
        else:
            dictionary = {}
            for fname in self.namelist():
                fpath = os.path.split(fname)
                d = dictionary
                while fpath:
                    f = fpath.pop(0)
                    if not fpath:
                        d[f] = None
                    elif f not in d:
                        next_d = {}
                        d[f] = next_d
                        d = next_d
                    else:
                        d = d[f]
            return dictionary

    def nametree(self):
        from .collections import Tree
        return Tree.create_from_dict(self.namedict())