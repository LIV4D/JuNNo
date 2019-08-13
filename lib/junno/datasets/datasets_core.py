import numpy as np
import time

from os.path import basename
from .dataset import AbstractDataSet, DSColumn, DSColumnFormat
from ..j_utils.j_log import log
from ..j_utils.function import match_params, not_optional_args
from ..j_utils.math import interval
from ..j_utils.collections import if_else, if_none, OrderedDict, stack


########################################################################################################################
class NumPyDataSet(AbstractDataSet):
    def __init__(self, data_dict, name='NumPyDataSet', pk=None):
        super(NumPyDataSet, self).__init__(name=name, pk_type=np.int32 if pk is None else pk.dtype)

        size = -1
        for name, array in data_dict.items():
            if size != -1:
                assert size == array.shape[0]
            else:
                size = array.shape[0]
            self.add_column(name, array.shape[1:], array.dtype)

        self._size = size
        self._offset = 0
        self.data = data_dict
        self._pk_data = pk

    @property
    def size(self):
        return self._size

    def _generator(self, gen_context):
        while not gen_context.ended():
            i, n, weakref = gen_context.create_result()
            r = weakref()
            r[:] = {_: self.data[_][i:i+n] for _ in r.columns_name()}
            r[:, 'pk'] = np.arange(start=i, stop=n+i) if self._pk_data is None else self._pk_data[i:i+n]
            r = None
            yield weakref

    def as_cached(self):
        return self


def from_numpy(**data_dict):
    return NumPyDataSet(data_dict, name='NumpyCache')


########################################################################################################################
class PyTableDataSet(AbstractDataSet):
    def __init__(self, pytable, rows=None, hdf_file=None, name='PyTableDataset'):
        if pytable.description._v_is_nested:
            raise NotImplementedError('PyTable with nested columns are not supported.')

        col_descr = {}
        pk_type = None
        for col_name, col in pytable.description._v_colobjects.items():
            if col_name=='pk':
                pk_type = col.dtype.base
            else:
                col_descr[col._v_pos] = col_name, col.dtype.shape, col.dtype.base

        super(PyTableDataSet, self).__init__(name=name, pk_type=if_none(pk_type, np.uint32))
        self.pytable = pytable
        self.hdf_file = hdf_file

        for i in sorted(col_descr.keys()):
            col_name, col_shape, col_dtype = col_descr[i]
            col_dtype = np.dtype(str(col_dtype).replace('S', 'U'))
            self.add_column(col_name, col_shape, col_dtype)

        self._rows = rows

        self.sortby = None
        self.where = None
        self._sorted_rows = None
        self._filtered_rows = None

    def __del__(self):
        if self.hdf_file is not None:
            self.hdf_file.flush()
            self.hdf_file.close()

    @staticmethod
    def from_file(path, table="dataset", name='PyTableDataset'):
        from ..j_utils.threadsafe_pytables import open_pytable
        f = open_pytable(path)
        table_path, table_name = table.rsplit('/', 1)
        if not table_path.startswith('/'):
            table_path = '/'+table_path
        table = f.get_node(table_path, table_name, classname='Table')
        return PyTableDataSet(table, name=name)

    @staticmethod
    def from_numpy(data_dict, cache_path=None, name='PyTableDataset'):
        from ..j_utils.threadsafe_pytables import open_pytable, tables
        from ..j_utils.collections import is_dict

        if is_dict(data_dict):
            cols_data, cols_name = zip(data_dict.items())
            data_dict = np.rec.fromarrays(cols_data, names=cols_name)
        elif not isinstance(data_dict, np.recarray):
            raise ValueError('data_dict should either be a dictionnary of numpy arrays or a numpy recarray.')

        if cache_path is not None:
            cache_path = cache_path.split(':')
            if len(cache_path) == 1:
                path = cache_path[0]
                table_name = 'dataset'
            elif len(cache_path) == 2:
                path, table_name = cache_path
            else:
                raise ValueError('cache_path should be formated as "PATH:TABLE_NAME"')

            f = open_pytable(path)
            table_path, table_name = table_name.rsplit('/', maxsplit=1)
            if not table_path.startswith('/'):
                table_path = '/' + table_path
            table = f.create_table(table_path, table_name, obj=data_dict, createparents=True)
        else:
            f = tables.open_file("/tmp/empty.h5", "a", driver="H5FD_CORE", driver_core_backing_store=0)
            table = f.create_array(f.root, "dataset", obj=data_dict)
        return PyTableDataSet(table, name=name)

    @property
    def size(self):
        return len(self._rows) if self._rows is not None else len(self.pytable)

    def _generator(self, gen_context):
        columns = gen_context.columns
        start = gen_context.start_id
        stop = gen_context.stop_id

        if self._rows is not None:
            hd5_gen = self.pytable.itersequence(self._rows[start:stop])
        else:
            hd5_gen = self.pytable.iterrows(start, stop)

        while not gen_context.ended():
            i, n, weakref = gen_context.create_result()
            r = weakref()

            for j in range(n):
                row = next(hd5_gen)
                for c in columns:
                    r[j, c] = row[c]

            r = None
            yield weakref

    def select(self, where):
        """
        See PyTables expressions syntax: https://www.pytables.org/usersguide/condition_syntax.html.
        """
        d = PyTableDataSet(self.pytable, name=self.dataset_name)
        d.sortby = self.sortby
        d._sorted_rows = self._sorted_rows
        d.where = where
        d._solve_where()
        for c_name, c in self.columns.items():
            d.col[c_name].format = c.format
        return d

    def sort(self, sortby, reverse=False):
        if reverse:
            sortby = '!'+sortby
        d = PyTableDataSet(self.pytable, name=self.dataset_name)
        d.where = self.where
        d._filtered_rows = self._filtered_rows
        d.sortby = sortby
        d._solve_sortby()
        for c_name, c in self.columns.items():
            d.col[c_name].format = c.format
        return d

    def __call__(self, where=None, sortby=None):
        if where is None:
            if sortby is None:
                return self
            return self.sort(sortby)
        if sortby is None:
            return self.select(where)
        d = PyTableDataSet(self.pytable, name=self.dataset_name)
        d.sortby = sortby
        d.where = where
        d._solve_sortby()
        d._solve_where()
        for c_name, c in self.columns.items():
            d.col[c_name].format = c.format
        return d

    def _solve_where(self):
        if self.where is None:
            self._filtered_rows = None
            if self._sorted_rows is not None:
                self._rows = self._sorted_rows
            else:
                self._rows = None
        else:
            self._filtered_rows = self.pytable.get_where_list(self.where, condvars=None)
            if self._sorted_rows is not None:
                rows = np.zeros(shape=(self.pytable.nrows,), dtype=np.bool)
                rows[self._filtered_rows] = 1
                self._rows = self._sorted_rows[rows[self._sorted_rows]]
            else:
                self._rows = self._filtered_rows

    def _solve_sortby(self):
        if self.sortby is None:
            self._sorted_rows = None
            if self._filtered_rows is not None:
                self._rows = self._filtered_rows
            else:
                self._rows = None
        else:
            reversed_sort = self.sortby.startswith('!')
            sortby = self.sortby[1:] if reversed_sort else self.sortby
            if not getattr(self.pytable.cols, sortby).is_indexed:
                # Create a full index
                getattr(self.pytable.cols, sortby).create_csindex()

            self._sorted_rows = self.pytable._check_sortby_csi(sortby=sortby, checkCSI=False).read_indices()
            if reversed_sort:
                self._sorted_rows = self._sorted_rows[::-1]
            if self._filtered_rows is not None:
                rows = np.zeros(shape=(self.pytable.nrows,), dtype=np.bool)
                rows[self._filtered_rows] = 1
                self._rows = self._sorted_rows[rows[self._sorted_rows]]
            else:
                self._rows = self._sorted_rows

    def has_custom_rows(self):
        return self._rows is not None and self._filtered_rows is None and self._sorted_rows is None


def load_hdf(path, name=None):
    if isinstance(path, str):
        path_split = path.split(':')
        if len(path_split) == 1:
            path = path_split[0]
            table_name = 'dataset'
        elif len(path_split) == 2:
            path, table_name = path_split
        else:
            raise ValueError('path should be formated as "PATH:TABLE_NAME"')
    else:
        raise NotImplementedError

    if name is None:
        name = table_name

    from ..j_utils.path import format_filepath
    from ..j_utils.threadsafe_pytables import open_pytable
    path = format_filepath(path, 'cache', exists=True)
    hdf_f = open_pytable(path)
    if not table_name.startswith('/'):
        table_name = '/' + table_name
    table_path, table_name = table_name.rsplit('/', maxsplit=1)
    if not table_path:
        table_path = '/'

    hdf_t = hdf_f.get_node(table_path, table_name, 'Table')
    return PyTableDataSet(hdf_t, name=name)


########################################################################################################################
class RandomVersionPyTableDataSet(AbstractDataSet):
    def __init__(self, pytables, hdf_file=None, name='PyTableDataset'):

        col_descr = {}
        pk_type = None
        exception = NotImplementedError
        for i, pytable in enumerate(pytables):
            if pytable.description._v_is_nested:
                raise NotImplementedError('PyTable with nested columns are not supported.')


            if i==0:
                nrows = len(pytable)
            elif nrows != len(pytable):
                raise exception

            for col_name, col in pytable.description._v_colobjects.items():
                if i==0:
                    if col_name=='pk':
                        pk_type = col.dtype.base
                    else:
                        col_descr[col._v_pos] = col_name, col.dtype.shape, col.dtype.base
                else:
                    if col_name=='pk':
                        if pk_type != col.dtype.base:
                            raise exception
                    elif any(_1!=_2 for _1, _2 in
                     zip((col_name, col.dtype.shape, col.dtype.base),col_descr.get(col._v_pos, (None,)*3))):
                        raise exception

        super(RandomVersionPyTableDataSet, self).__init__(name=name, pk_type=if_none(pk_type, np.uint32))
        self.pytables = pytables
        self.hdf_file = hdf_file

        for i in sorted(col_descr.keys()):
            col_name, col_shape, col_dtype = col_descr[i]
            col_dtype = np.dtype(str(col_dtype).replace('S', 'U'))
            self.add_column(col_name, col_shape, col_dtype)

    def __del__(self):
        if self.hdf_file is not None:
            self.hdf_file.flush()
            self.hdf_file.close()

    @property
    def size(self):
        return len(self.pytables[0])

    def _generator(self, gen_context):
        columns = gen_context.columns
        start = gen_context.start_id
        stop = gen_context.stop_id

        table_count = len(self.pytables)
        N = stop-start
        if gen_context.determinist:
            hd5_gens = [iter(t.iterrows(start+i, stop, step=table_count)) for i, t in enumerate(self.pytables)]
            table_seq = np.repeat([np.arange(table_count, dtype=np.int)], N//table_count, axis=0).flatten()
            table_seq = np.concatenate((table_seq, np.arange(N % table_count, dtype=np.int)))
        else:
            table_seq = np.concatenate(list([i]*(N//table_count + round(i/table_count)) for i in range(table_count)))
            self.rng.shuffle(table_seq)
            hd5_gens = [iter(t.itersequence(np.argwhere(table_seq == i).flatten()+start))
                        for i, t in enumerate(self.pytables)]

        while not gen_context.ended():
            i, n, weakref = gen_context.create_result()
            r = weakref()
            for j in range(n):
                row = next(hd5_gens[int(table_seq[i+j-start])])
                for c in columns:
                    r[j, c] = row[c]

            r = None
            yield weakref


def load_random_version_hdf(path, name=None):
    import tables
    if isinstance(path, str):
        path_split = path.split(':')
        if len(path_split) == 1:
            path = path_split[0]
            table_name = 'dataset'
        elif len(path_split) == 2:
            path, table_name = path_split
        else:
            raise ValueError('path should be formated as "PATH:TABLE_NAME"')
    else:
        raise NotImplementedError

    if name is None:
        name = basename(path)

    from ..j_utils.path import format_filepath
    from ..j_utils.threadsafe_pytables import open_pytable
    path = format_filepath(path, 'cache', exists=True)
    hdf_f = open_pytable(path)
    
    if not table_name.startswith('/'):
        table_name = '/' + table_name
    table_path, table_name = table_name.rsplit('/', maxsplit=1)
    if not table_path:
        table_path = '/'

    hdf_tables = []
    i = 0

    while True:
        try:
            hdf_tables.append(hdf_f.get_node(table_path, table_name+'_'+str(i), 'Table'))
        except tables.NoSuchNodeError:
            break
        i += 1

    return RandomVersionPyTableDataSet(hdf_tables, name=name)


########################################################################################################################
class DataSetSubset(AbstractDataSet):
    def __init__(self, dataset, start=0, stop=None, name='subset'):
        """
        :type dataset: AbstractDataSets
        """
        super(DataSetSubset, self).__init__(name, dataset, pk_type=dataset.pk.dtype)
        self._columns = dataset.copy_columns(self)
        self.start, self.stop = interval(dataset.size, start, stop)

    def _generator(self, gen_context):
        first_id = gen_context.start_id + self.start
        last_id = gen_context.stop_id + self.start
        gen = gen_context.generator(self._parent, start=first_id, stop=last_id, columns=gen_context.columns)
        while not gen_context.ended():
            i, n, weakref = gen_context.create_result()
            gen.next(copy={c: weakref()[c] for c in gen_context.columns}, seek=self.start+i, limit=n)
            yield weakref

    @property
    def size(self):
        return self.stop - self.start

    def subset(self, *args, start=0, stop=None, name='subset'):
        if len(args) == 1:
            start = 0
            stop = args[0]
        elif len(args) == 2:
            start = args[0]
            stop = args[1]
        return DataSetSubset(self._parent, start + self.start, min(self.start + stop, self.stop), name=name)


########################################################################################################################
class DataSetSubgen(AbstractDataSet):
    def __init__(self, dataset, n=1, name='subgen'):
        """
        :type dataset: AbstractDataSets
        """
        super(DataSetSubgen, self).__init__(name, dataset, pk_type=dataset.pk.dtype)
        self._columns = dataset.copy_columns(self)
        self.n = n

    def _generator(self, gen_context):
        max_n = min(self.n, gen_context.n)
        parent_gen = gen_context.generator(self._parent, n=max_n)

        while not gen_context.ended():
            i_global, N, weakref = gen_context.create_result()
            r = weakref()

            for i in range(0, N, max_n):
                n = min(N-i, max_n)
                parent_gen.next(copy={c: r[i:i+n, c] for c in r.columns_name() + ['pk']}, seek=i_global+i, limit=n)

            r = None
            yield weakref

    @property
    def size(self):
        return self.parent_dataset.size


########################################################################################################################
class DataSetMap(AbstractDataSet):
    def __init__(self, dataset, mapping, keep_all_columns=True, name='mapping'):
        """
        :type dataset: AbstractDataSets
        """
        super(DataSetMap, self).__init__(name, dataset, pk_type=dataset.pk.dtype)

        mapped_cols = set()
        for column, parent_columns in mapping.items():
            if isinstance(parent_columns, str):
                parent_columns = dataset.interpret_columns(parent_columns)
                mapping[column] = parent_columns
            if isinstance(parent_columns, (list, tuple)):
                mapped_cols.update(parent_columns)
            else:
                raise ValueError('Invalid mapping value: %s.\n '
                                 'Valid value are column name for mapping and list or tuple for concatenation.'
                                 % parent_columns)

        if keep_all_columns:
            for c in set(dataset.columns_name())-mapped_cols:
                mapping[c] = [c]

        self.concatenation_map = mapping
        for column, parent_columns in mapping.items():
            c = dataset.column_by_name(parent_columns[0])
            shape = c.shape
            for real_column in parent_columns[1:]:
                s = dataset.column_by_name(real_column).shape
                assert s[1:] == shape[1:]
                shape = tuple([shape[0]+s[0]] + list(s[1:]))

            self.add_column(name=column, shape=shape, dtype=c.dtype, format=c.format)

    def _generator(self, gen_context):
        columns = gen_context.columns

        gen_columns = set()
        copy_columns = {}
        duplicate_columns = {}
        for c_name in columns:
            if c_name == "pk":
                continue
            c_parents = self.concatenation_map[c_name]
            gen_columns.update(set(c_parents))
            i = 0
            for c_parent in c_parents:
                column_parent = self.parent_dataset.column_by_name(c_parent)
                n = column_parent.shape[0] if column_parent.shape else 0
                if c_parent not in copy_columns:
                    copy_columns[c_parent] = (c_name, i, n)
                else:
                    d = duplicate_columns.get(c_parent, None)
                    if d is None:
                        d = []
                        duplicate_columns[c_parent] = d
                    d.append((c_name, i, n))
                i += n
        copy_columns['pk'] = ('pk', 0, 0)
        gen_columns = list(gen_columns)
        gen = gen_context.generator(self._parent, columns=gen_columns)

        while not gen_context.ended():
            global_i, N, weakref = gen_context.create_result()
            r = weakref()
            result = gen.next(copy={c_parent: r[c_name][:, i:i+n] if n > 0 else r[:, c_name]
                              for c_parent, (c_name, i, n) in copy_columns.items()}, limit=N, r=r, seek=global_i)
            for c_parent, duplicates in duplicate_columns.items():
                for c_name, i, n in duplicates:
                    r[c_name][:, i:i+n] = result[c_parent]
            result = None
            r = None
            yield weakref

    @property
    def size(self):
        return self._parent.size


########################################################################################################################
class DataSetShuffle(AbstractDataSet):
    def __init__(self, dataset, indices=None, subgen=1, name='shuffle', rng=None):
        """
        :type dataset: AbstractDataSets        if args
        """
        super(DataSetShuffle, self).__init__(name=name, parent_datasets=dataset, pk_type=dataset.pk.dtype, rng=rng)
        self._columns = dataset.copy_columns(self)

        self.indices = None if indices is None else np.asarray(indices, dtype=np.uint32)
        self.random_indices = indices is None

        self.subgen = max(subgen, 0) if isinstance(subgen, int) else 0
        self.subgen_range = None
        self.subgen_index = None

        if subgen and subgen != 1 and not self.random_indices:
            if isinstance(subgen, list):
                self.subgen = len(subgen)
                self.subgen_range = subgen
            else:
                log.warn("With not random indicies, subgenerators range must be specified. "
                         "Falling back to no subgenerators...")
                self.subgen = 0

    def _generate_random_sequence(self):
        if self.subgen > 1:
            # Compute (start, stop) for each subgen
            subgen_range = [(int(round(_ * self.size / self.subgen)), int(round((_ + 1) * self.size / self.subgen)))
                        for _ in range(self.subgen)]

            # Create one-hot
            rand_seq = np.zeros((self.size, self.subgen), dtype=np.uint32)
            for i, (start, stop) in enumerate(subgen_range):
                n = stop-start
                rand_seq[start:start + n, :] = [1 if i == _ else 0 for _ in range(self.subgen)]
            # Shuffle one-hot
            self.rng.shuffle(rand_seq)

            # Compute the table of subgenerator own indexes by cumsum one-hot
            rand_seq_id = rand_seq*rand_seq.cumsum(axis=0)

            # Replace every one in the one-hot table by the starting index of its sub-generator
            rand_seq_start = np.multiply(rand_seq, [start for (start, stop) in subgen_range])

            # Compute final random sequence by adding the sub-generators own indexes and starting index
            indices = np.asarray((rand_seq_start+rand_seq_id).sum(axis=1), dtype=np.uint32)-1

            # Compute subgenerator indexes
            np.multiply(rand_seq, np.arange(len(subgen_range), dtype=np.uint32), out=rand_seq)
            rand_seq = rand_seq.sum(axis=1)

            return indices, subgen_range, rand_seq
        else:
            rand_seq = np.arange(self.parent_dataset.size, dtype=int)
            self.rng.shuffle(rand_seq)
            return rand_seq

    def _setup_determinist(self):
        if self.random_indices:
            if self.indices is None:
                seq = self._generate_random_sequence()
                if isinstance(seq, tuple):
                    self.indices, self.subgen_range, self.subgen_index = seq
                else:
                    self.indices = seq
        else:
            if self.subgen > 1:
                if self.subgen_index is None:
                    indices_map = np.zeros(dtype=np.int32, shape=(self.parent_dataset.size,))-1
                    for i, s in enumerate(self.subgen_range):
                        indices_map[s[0]:s[1]] = i
                    self.subgen_index = indices_map[self.indices]

    def _generator(self, gen_context):
        columns = gen_context.columns

        if gen_context.determinist or not self.random_indices:
            indices = self.indices
            subgen_index = self.subgen_index
            subgen_range = self.subgen_range
        else:
            if self.subgen > 1:
                indices, subgen_range, subgen_index = self._generate_random_sequence()
            else:
                indices = self._generate_random_sequence()
                subgen_range = None

        # indices = indices[gen_context.start_id:gen_context.stop_id]

        # Setup subgenerators
        subgen = []
        async_subgen = None
        if self.subgen > 1:
            valid_indices = indices[gen_context.start_id:gen_context.stop_id+1]
            start = valid_indices.min()
            stop = valid_indices.max()+1

            valid_subgen = []
            valid_ranges = []
            for i, (gen_start, gen_stop) in enumerate(subgen_range):
                if gen_start < stop and stop > start:
                    gen_start = max(gen_start, start)
                    gen_stop = min(gen_stop, stop)
                    valid_ranges.append((gen_start, gen_stop))
                    valid_subgen.append(i)
            subgen_range = valid_ranges
            valid_subgen = np.array(valid_subgen)
            # valid_subgen: [-1 if _ not in valid_subgen else valid_subgen.index(_) for _ in range(self.subgen)]
            valid_subgen = np.array([if_else(np.where(valid_subgen == _)[0], lambda x: len(x) > 0, [-1])[0]
                                     for _ in range(self.subgen)])

            if not 1 < gen_context.ncore <= len(subgen_range):
                async_subgen = gen_context.ncore > len(subgen_range)
                # Setup subgenerator
                for i, (gen_start, gen_stop) in enumerate(subgen_range):
                    is_last = i == len(subgen_range)-1
                    gen = gen_context.generator(self._parent, start=gen_start, stop=gen_stop, n=1,
                                                parallel=async_subgen and not is_last,
                                                ncore=round(i*gen_context.ncore/len(subgen_range)))
                    if async_subgen:
                        gen.setup()     # Begin computation if parallel
                    subgen.append(gen)

            else:
                async_subgen = True

                # Split subgen
                s_split = []
                indices_map = np.zeros(dtype=np.int32, shape=(self.parent_dataset.size,))-1
                for i in range(gen_context.ncore):  # For each core select the subset of valid subgen
                    i0 = round(i*len(subgen_range)/gen_context.ncore)
                    i1 = round((i+1)*len(subgen_range)/gen_context.ncore)
                    for start, stop in valid_ranges[i0:i1]:
                        indices_map[start:stop] = i
                    s_split.append((i0, i1))
                subgen_ids = indices_map[valid_indices]
                del indices_map
                for i, (i0, i1) in enumerate(s_split):  # For each core setup dataset and generator
                    is_last = i == len(s_split) - 1
                    s_indices = indices[subgen_ids == i]
                    s_dataset = DataSetShuffle(self._parent, indices=s_indices, subgen=valid_ranges[i0:i1])
                    s_dataset.subgen_index = valid_subgen[subgen_index[subgen_ids == i]]-i0
                    gen = gen_context.generator(s_dataset, n=1, ncore=1, parallel='thread' if is_last else 'process')
                    gen.setup()
                    subgen.append(gen)
                subgen_index = subgen_ids
        else:
            if gen_context.ncore > 1:
                for i in range(gen_context.ncore):
                    gen = gen_context.generator(self._parent, n=1, start=indices[gen_context.start_id],
                                                ncore=1, parallel=True)
                    gen.seq_id = None
                    subgen.append(gen)
            else:
                subgen.append(gen_context.generator(self._parent, n=1, start=0, stop=self._parent.size, ncore=1))

        while not gen_context.ended():
            i_global, n, weakref = gen_context.create_result()
            r = weakref()

            if self.subgen <= 1:
                seq = indices[i_global:i_global+n]
                if len(subgen) > 1:  # -- Mutliple core, No subgen --
                    waiting_seq = len(seq)
                    while waiting_seq:
                        waiting = True
                        for s in subgen:
                            if s.seq_id is not None and s.poll(copy=r[s.seq_id:s.seq_id+1], r=r, ask_next=False):
                                waiting = False
                                s.seq_id = None
                                waiting_seq -= 1

                            if len(seq) and s.seq_id is None:     # Ask next
                                s.seq_id = n - len(seq)
                                s.ask(seq.pop(0))

                        if waiting:
                            time.sleep(1e-6)

                else:       # -- Single core, No subgen --
                    for i, seq_id in enumerate(seq):
                        try:
                            subgen[0].next(copy=r[i:i+1], r=r, seek=seq_id)
                        except StopIteration:
                            raise RuntimeError('%s generator stopped early at id=%i (parent id=%i).'
                                               % (self._name, i+i_global, seq_id))
            else:
                seq_subgens = subgen_index[i_global:i_global+n]
                if async_subgen:    # -- Async subgen --
                    for i, sub_id in enumerate(seq_subgens):
                        subgen[sub_id].next(copy=r[i:i+1], r=r)
                else:               # -- Sync subgen --
                    seq_indexes = indices[i_global:i_global + n]
                    for i, (sub_id, seq_id) in enumerate(zip(seq_subgens, seq_indexes)):
                        try:
                            subgen[sub_id].next(copy=r[i:i+1], r=r, seek=seq_id)
                        except StopIteration as e:
                            import traceback
                            log.error("Subgen %i (stop_id=%i) of dataset %s raised StopIteration exception at index %i."
                                      % (sub_id, subgen[sub_id].stop_id, self.dataset_fullname, seq_id))
                            raise e
            r = None
            yield weakref

        for s in subgen:
            s.clean()

    @property
    def size(self):
        return self._parent.size if self.indices is None else len(self.indices)


########################################################################################################################
class DataSetJoin(AbstractDataSet):
    def __init__(self, datasets, verbose=False, **kwargs):
        """
        :param datasets: A list of dataset to join. Each element of the list should be either:
                          - a dataset (the primary key column is used for the join)
                          - a tuple containing a dataset and the name of the column where the join should be performed
                          - a dataset column: the dataset will be join on this column

        :param kwargs: For each elements of kwargs, a column will be created in this dataset.
                       kwargs elements should be either  tuple containing the index of the dataset and the name of
                       its column on which the join should be processed, or simply this column.
                       If kwargs is empty, all columns of the datasets are inserted in this dataset
                       as DATASETNAME_COLUMNNAME.

                       A 'pk' parameters can be passed to specify which column should be used as primary key
                       (by default the primary key of the first dataset is used)

        WARNING: FOR THE TIME BEING, THIS DATASET WILL PERFOM A NAIVE JOIN WHEN THE DATASET IS CREATED.
                 THIS PROCESS CAN BE LONG AND MEMORY CONSUMING, DON'T USE IT ON LARGE DATABASES!!!!!!
                 (all the columns on which the join is performed are read and ordered,
                 indexes are stored permanently in RAM...)

        NOTE:    All dataset will be accessed in ascending order of their join column
        """

        #  ---  REFORMATING PARAMETERS  ---
        dataset_tuples = []
        datasets_list = []
        for dataset in datasets:
            if isinstance(dataset, DSColumn):
                join_column = dataset.name
                dataset = dataset.dataset
            elif isinstance(dataset, tuple):
                join_column = dataset[1]
                dataset = dataset[0]
            elif isinstance(dataset, AbstractDataSet):
                join_column = 'pk'
            else:
                raise NotImplementedError
            dataset_tuples.append((dataset, join_column))
            datasets_list.append(dataset)
        datasets = dataset_tuples

        if 'pk' in kwargs:
            pk = getattr(datasets_list[kwargs['pk'][0]], kwargs['pk'][1])
            del kwargs['pk']
        else:
            pk = datasets_list[0].pk

        for name, column in kwargs.items():
            if isinstance(column, DSColumn):
                kwargs[name] = (datasets_list.index(column.dataset), column.name)

        #  ---  READING AND SIMPLIFYING DATASETS  ---
        if len(kwargs):
            used_datasets = {_[0] for n, _ in kwargs.items()}
        else:
            used_datasets = {_ for _ in range(len(datasets))}

        datasets_name = []
        simp_dataset_map = [-1]*len(datasets)
        simplified_datasets = []
        join_columns = []
        for dataset_id in used_datasets:
            dataset, join_column = datasets[dataset_id]
            root_name = dataset.dataset_name
            name = root_name
            i = 0
            while name in datasets_name:
                i += 1
                name = root_name + '(%i)' % i
            datasets_name.append(name)
            simp_dataset_map[dataset_id] = len(simplified_datasets)
            simplified_datasets.append(dataset)
            join_columns.append((dataset, join_column))
        datasets = simplified_datasets

        super(DataSetJoin, self).__init__('Join', datasets, pk_type=pk.dtype)

        #  ---  DEFINING COLUMNS ---
        self._pk_foreign_col = (datasets.index(pk.dataset), pk.name)
        self._columns_map = {}
        if len(kwargs):     # Only specific columns of joined datasets must be used
            for column_name, column in kwargs.items():
                dataset_id, foreign_name = column
                dataset_id = simp_dataset_map[dataset_id]
                if not isinstance(column, tuple) or len(column) != 2:
                    continue
                remote_c = datasets[dataset_id].column_by_name(foreign_name)
                self.add_column(column_name, remote_c.shape, remote_c.dtype, remote_c.format)
                self._columns_map[column_name] = (dataset_id, foreign_name)
        else:               # All columns of joined datasets will be used
            for dataset_id, (dataset, dataset_name) in enumerate(zip(datasets, datasets_name)):
                for column in dataset.columns:
                    dataset_id = simp_dataset_map[dataset_id]
                    column_name = '%s_%s' % (dataset_name, column.name)
                    self.add_column(column_name, column.shape, column.dtype, column.format)
                    self._columns_map[column_name] = (dataset_id, column.name)

        #  ---  JOIN DATASETS  ---
        # Read and ordering join columns
        join_datas = []
        ordered_join = []
        min_size = datasets[0].size
        for dataset, column in join_columns:
            min_size = min(min_size, dataset.size)

            gen = dataset.generator(n=1, columns=[column])
            data = next(gen)[column]
            join_data = np.zeros((dataset.size,),  dtype=data.dtype)
            join_data[0] = data[0]
            for i in range(1, dataset.size):
                join_data[i] = next(gen)[column][0]

            join_datas.append(join_data)
            ordered_join.append(np.argsort(join_data))

        self.ordered_join = ordered_join

        # Iterating through join columns to find matches
        join = np.zeros((min_size, len(join_datas)), dtype=int)
        n = 0
        all_it = np.zeros((len(join_columns)-1,), dtype=int)
        for it1 in range(join_columns[0][0].size):
            ref = join_datas[0][ordered_join[0][it1]]
            for id_it in range(len(all_it)):
                join_data = join_datas[id_it+1]
                ds_ordered_join = ordered_join[id_it+1]
                skipped = False
                while all_it[id_it]+1 < len(ds_ordered_join) and (join_data[ds_ordered_join[all_it[id_it]+1]] <= ref
                      or join_data[ds_ordered_join[all_it[id_it]+1]] == join_data[ds_ordered_join[all_it[id_it]]]):
                    if skipped and verbose:
                        log.info("%s (dataset: %s)  will be skipped"
                              % (join_data[ds_ordered_join[all_it[id_it]]], datasets[id_it+1].dataset_name))
                    all_it[id_it] += 1
                    skipped = True

            is_matched = True
            for id_it in range(len(all_it)):
                if ref != join_datas[id_it+1][ordered_join[id_it+1][all_it[id_it]]]:
                    is_matched = False
                    break

            if is_matched:
                join[n, 0] = ordered_join[0][it1]
                join[n, 1:] = [order[it] for order, it in zip(ordered_join[1:], all_it)]
                n += 1
            elif verbose:
                log.info("%s (dataset: %s)  will be skipped"
                      % (ref, datasets[0].dataset_name))

            end = False
            for id_it in range(len(all_it)):
                if all_it[id_it]+1 == len(ordered_join[id_it+1]):
                    end = True
                    break
            if end:
                break

        # Storing joined index for each dataset
        self._join = join[:n, :]

    @property
    def size(self):
        return self._join.shape[0]

    def _generator(self, gen_context):
        columns = gen_context.columns

        # Initialise to match asked columns
        datasets_columns = [[] for _ in self.parent_datasets]
        reverse_columns_map = [{} for _ in self.parent_datasets]
        for c_name, (dataset_id, foreign_name) in self._columns_map.items():
            if c_name in columns:
                datasets_columns[dataset_id].append(foreign_name)
                reverse_columns_map[dataset_id][foreign_name] = c_name

        datasets_columns[self._pk_foreign_col[0]].append(self._pk_foreign_col[1])
        reverse_columns_map[self._pk_foreign_col[0]][self._pk_foreign_col[1]] = 'pk'
        ngen = len(datasets_columns)
        generators = [[None, -1] for _ in range(ngen)]

        intime_gens = [False] + [i < gen_context.ncore for i in reversed(range(1, ngen))]
        if gen_context.ncore <= ngen:
            ncore_gens = [1]*ngen
        else:
            free_core = gen_context.ncore-ngen
            mean_ncore = free_core//ngen
            ncore_gens = [mean_ncore + (1 if i < free_core % ngen else 0) for i in range(ngen)]

        while not gen_context.ended():
            global_i, n, weakref = gen_context.create_result()
            r = weakref()

            for i in range(n):
                # Setup generators
                for dataset_id, gen in enumerate(generators):
                    needed_index = self._join[global_i + i, dataset_id]
                    if gen[1] != needed_index or gen[0] is None:
                        dataset = self.parent_datasets[dataset_id]
                        generators[dataset_id][0] = gen_context.generator(dataset, start=needed_index, stop=dataset.size,
                                                                          n=1, columns=datasets_columns[dataset_id],
                                                                          parallel=intime_gens[dataset_id],
                                                                          ncore=ncore_gens[dataset_id])
                        generators[dataset_id][1] = needed_index

                # Reading generators
                for gen_id, gen in enumerate(generators):
                    gen[0].next(copy={c: r[i:i+1, reverse_columns_map[gen_id][c]] for c in gen[0].columns
                                      if c in reverse_columns_map[gen_id]}, r=r)

                # Updating generator index
                for gen in generators:
                    gen[1] += 1
            r = None
            yield weakref

    def subset(self, *args, start=0, stop=None, name=None):
        from copy import deepcopy
        start, stop = interval(self.size, start, stop, args)
        sub = deepcopy(self)
        if name is None:
            sub._name += '_Subset'
        else:
            sub._name = name
        sub._join = self._join[start:stop, :]
        return sub


def join(*join_columns, **map_columns):
    if not join_columns:
        if not map_columns:
            raise ValueError("You seriously didn't provide any argument to join()? You know it's not gone work, right?")
        join_columns = set()
        for v in map_columns.values():
            if not isinstance(v, DSColumn):
                raise ValueError('Invalid column value: %s.' % repr(v))
            join_columns.add(v.dataset)
    if len(join_columns) == 1:
        if isinstance(join_columns[0], str):
            dataset_set = set()
            for c in map_columns.values():
                if isinstance(c, DSColumn):
                    if c.dataset is None:
                        raise ValueError('Error %s has no parent dataset. (c._dataset=%s)' % (c, c._dataset))
                    dataset_set.add(c.dataset)
                else:
                    raise ValueError('Error when joining on %s. %s is not a DSColumn.' % (join_columns, repr(c)))
            join_columns = [_.column_by_name(join_columns) for _ in dataset_set]
        elif isinstance(join_columns[0], (list, tuple, set)) and join_columns[0]:
            join_columns = join_columns[0]
        else:
            raise ValueError('Invalid datasets: %s.' % repr(join_columns[0]))

    infer_join = isinstance(next(iter(join_columns)), AbstractDataSet)
    if infer_join:
        if len(join_columns) == 1:
            raise ValueError('You definitely want to provide more than one dataset to perform a join...')
        common_col = set(join_columns[0].columns_name())

        for dataset in join_columns[1:]:
            if not isinstance(dataset, AbstractDataSet):
                raise ValueError('If the first positional argument of join() is a dataset, all of them should be, but'
                                 '%s is not.' % repr(dataset))
            common_col.intersection_update(dataset.columns_name())
            if not common_col:
                break

        if len(common_col) > 1:
            raise ValueError('Provided datasets have more than one columns in common, '
                             'please specify which one should be used to join of: %s' % common_col)
        if common_col:
            col = next(iter(common_col))
            join_columns = [d.column_by_name(col) for d in join_columns]
        else:
            join_columns = [d.column_by_name('pk') for d in join_columns]

    if not map_columns:
        map_columns = OrderedDict()
        for join_col in join_columns:
            dataset = join_col.dataset
            for c in dataset.col:
                if c.name in map_columns and c is not join_col:
                    raise ValueError('Column named %s appears in at least 2 datasets, you must specify a mapping.' % c)
                map_columns[c.name] = c

    return DataSetJoin(join_columns, **map_columns)


########################################################################################################################
class DataSetConcatenate(AbstractDataSet):
    def __init__(self, datasets, columns=None, name='DataSetConcatenate'):
        """
        Concatenate multiple datasets into one.
        (The resulting dataset length equals the sum of the provided datasets length)
        :param datasets: A list of dataset to concatenate.
        :param columns: columns determines which columns should be included in this dataset.
                    - None: Only columns included in every concatenated datasets are included
                    - List of column name: Every name of this list must refer to a column of at least one dataset.
                                           If a columns is not present in a dataset its data will be filled with 0.
                    - Dictionary: keys must be a column name included in at least one dataset,
                                  values are the default value used when the column is not included in a dataset.
        """
        if not len(datasets):
            raise ValueError('datasets list to concatenate is empty!')

        # -- Check columns --
        if columns is None:
            columns = [_ for _ in datasets[0].columns_name()]
            for d in datasets[1:]:
                columns = [_ for _ in d.columns_name() if _ in columns]
            if not columns:
                raise ValueError('Provided datasets for concatenations have no columns in common')
        if isinstance(columns, (list, tuple)):
            columns = {_: 0 for _ in columns}
        elif isinstance(columns, str):
            columns = {columns: 0}
        elif not isinstance(columns, dict):
            raise ValueError('Columns must either be of type None, str, list, tuple or dict (not: %s)' % type(columns))

        # Check presence and compatiblity
        columns_default = columns
        columns = {}
        for col_name, col_default in columns_default.items():
            col = None
            for d in datasets:
                if col_name in d.columns_name():
                    d_column = d.column_by_name(col_name)
                    if col is None:
                        col = d_column
                        # Check default value compatibility
                        if col.dtype == str and col_default == 0:
                            col_default = ''
                            columns_default[col_name] = col_default
                        if not isinstance(col_default, np.ndarray):
                            columns_default[col_name] = np.full(shape=col.shape, fill_value=col_default, dtype=col.dtype)
                        else:
                            if col_default.shape != col.shape:
                                raise ValueError('Default value shape must match columns shape.\n'
                                                 'Column %s expected shape: %s, but the given default value shape is: %s'
                                                 % (repr(col_name), col.shape, col_default.shape))
                            if col_default.dtype != col.dtype:
                                raise ValueError('Default value dtype must match columns dtype.\n'
                                                 'Column %s expected dtype: %s, but the given default value dtype is: %s'
                                                 % (repr(col_name), col.dtype, col_default.dtype))
                    else:
                        if col.shape != d_column.shape:
                            raise ValueError('Columns shape must the same across datasets.\n'
                                             'Column %s expected shape: %s, but shape from dataset %s is: %s'
                                             % (repr(col_name), col.shape, d.dataset_name, d_column.shape))
                        if col.dtype != d_column.dtype:
                            raise ValueError('Columns dtype must the same across datasets.\n'
                                             'Column %s expected type: %s, but type from dataset %s is: %s'
                                             % (repr(col_name), col.dtype, d.dataset_name, d_column.dtype))

            if col is None:
                raise ValueError('Column %s is not included in any concatenated datasets.' % col_name)
            columns[col_name] = col.shape, col.dtype, col.format

        # -- Setup dataset --
        super(DataSetConcatenate, self).__init__(name=name, parent_datasets=datasets, pk_type=str)
        self._columns = [DSColumn(name, shape, dtype, self, format) for name, (shape, dtype, format) in columns.items()]
        self._columns_default = columns_default

        self._datasets_start_index = []
        start = 0
        for d in self.parent_datasets:
            self._datasets_start_index.append(start)
            start += d.size

    def _generator(self, gen_context):
        from bisect import bisect_right

        columns = gen_context.columns

        copy_cols = []
        default_cols = []
        gen_stop_index = 0
        parent_gen = None

        def read_from_parent(result, global_i, n):
            r = parent_gen.next(copy={c: result[global_i:global_i + n, c] for c in copy_cols}, limit=n, r=result)
            for i in range(n):
                result[global_i+i, 'pk'] = parent_gen.dataset.dataset_name + '|' + str(r[i, 'pk'])
                for c in default_cols:
                    result[global_i+i, c] = self._columns_default[c]

        while not gen_context.ended():
            i_global, N, weakref = gen_context.create_result()
            r = weakref()

            i = 0

            while i < N:
                if parent_gen is None:
                    # Setup next generator
                    dataset_id = bisect_right(self._datasets_start_index, i_global+i)-1
                    dataset = self.parent_datasets[dataset_id]
                    gen_start_index = self._datasets_start_index[dataset_id]
                    gen_stop_index = min(gen_start_index + dataset.size, gen_context.stop_id)

                    copy_cols = [_ for _ in dataset.columns_name() if _ in columns]
                    default_cols = [_ for _ in columns if _ not in copy_cols]
                    parent_gen = gen_context.generator(dataset, n=N, columns=copy_cols,
                                                       start=i_global + i - gen_start_index,
                                                       stop=min(gen_context.stop_id-gen_start_index, dataset.size))

                n = min(N - i, gen_stop_index - i_global - i)
                try:
                    result = parent_gen.next(copy={c: r[i:i + n, c] for c in copy_cols}, limit=n, r=r)
                except StopIteration:
                    parent_gen = None
                    continue

                for i_pk in range(n):
                    r[i + i_pk, 'pk'] = parent_gen.dataset.dataset_name + '|' + str(result[i_pk, 'pk'])
                    for c in default_cols:
                        r[i + i_pk, c] = self._columns_default[c]

                if parent_gen.ended():
                    parent_gen = None

                i += n

            r = None
            yield weakref

    @property
    def size(self):
        return sum(_.size for _ in self.parent_datasets)


def concatenate(*args):
    return DataSetConcatenate(args)


########################################################################################################################
class DataSetApply(AbstractDataSet):
    """
    Apply a function to some columns of a dataset, all other columns are copied
    """
    def __init__(self, dataset, function, columns=None, remove_parent_columns=True, format=None,
                 before_apply=None, after_apply=None,
                 n_factor=None, batchwise=False, name='apply'):
        """
        :param dataset: Dataset on which the function should be applied
        :param function: the function to apply to the dataset. The function can be apply element-wise or batch-wise.
                         Parameters should be named after **dataset** columns or with generic names if the function
                         should be applied several times to the dataset (see **columns** description).
        :param columns: Describe to which columns the function should be applied and which column should be created.
                        Columns can be either:
                            - A column name: The function will be applied to the specified column or will create a
                                column if no column existed with that name. In the latter case, the function parameters
                                must correspond to columns of **dataset**.
                            - A list of column names: The function will be applied to each column individually.
                            - A tuple of column names: Should be used when the function return several values.
                                The outputs of the function will be stored into the specified columns. Any columns not
                                present in the dataset will be created. The function parameters must correspond to
                                correspond to columns of **dataset**.

                            - A dictionary describing the mapping: The keys of the dictionary are the name of the
                                    new columns (any existing columns will be replaced, the other will be copied).
                                    Should the function return several values, the keys must be tuples of column names.
                                    The values specify the column on which the function is applied (which columns should
                                    be passed as a parameter of the function). It must be a **dataset** column name or a
                                    tuple of those or None. If the number of columns specified is fewer than the number
                                    of not-optional arguments of the function, the left-over arguments must be named as
                                    **dataset** columns.

        :param format: A dictionary mapping columns names to a tuple specifying (dtype, shape [, format]). For
                                every column not described here, its type and shape will be read after applying the
                                function to the first row of the dataset. If a column name is given instead of a tuple,
                                the type, shape and format will be copied from the **dataset** column. If None is given,
                                a similar behaviour will be attempted based on the column name.
                            Finally, this parameters could be set to 'same'. In this case, all column type, shape and
                            format will be copied from **dataset**.

        :param before_apply: A function called before **function** to apply a preprocessing on its arguments.
                             This function will receive all the arguments of **function** as **kwargs and should return
                             a dictionary with the preprocessed values.
                             ::Exemple::
                             def preprocessing(**kwargs):
                                return {k: do_stuff(v) for k, v in kwargs.items()}

        :param after_apply: A function called after **function** to apply a postprocesing on its results.
                            Please remember that a **function** may return a single value or an array,
                            a tuple or a list of those and a list of tuple of those...

        :param n_factor: The number of rows generated for each row given to the function. If not specified,
                            the value will be read by passing the **dataset** first row to the function.

        :param batchwise: Should be set to True if the function support batch execution (better performance are
                            expected).
        """
        super(DataSetApply, self).__init__(name, dataset, pk_type=dataset.pk.dtype if n_factor==1 else str)
        self.f_params = tuple(not_optional_args(function))

        # ---  HANDLE COLUMNS  ---
        parent_columns_name = dataset.columns_name()
        parent_copied_columns = dataset.copy_columns(self)
        own_columns = []
        self._single_col_mapping = {}
        self._columns_mapping = {}

        def str_tuple(c):
            if isinstance(c, str):
                c = tuple(c.split(','))
            if isinstance(c, tuple):
                if all(_ in dataset.col for _ in c):
                    return {c:c[:len(self.f_params)]}
                else:
                    return {c: ()}
            return None

        if columns is None:
            columns = self.columns_name()
        elif isinstance(columns, (str, tuple)):
            columns = str_tuple(columns)
        elif isinstance(columns, list):
            col_dict = {}
            for c in columns:
                c = str_tuple(c)
                if c is None:
                    raise ValueError('Invalid column: %s' % c)
                col_dict.update(c)
            columns = col_dict
        if not isinstance(columns, dict):
            raise ValueError('Columns should be of the following type str, tuple, list, dict.'
                             '(type provided: %s)' % type(columns).__name__)

        for own_c, parent_c in columns.items():
            own_c = tuple(self.interpret_columns(own_c, exists=False))
            if parent_c is None or not parent_c:
                parent_c = []
            else:
                parent_c = dataset.interpret_columns(parent_c)

            # Check explicit parent columns
            if len(parent_c) > len(self.f_params):
                raise ValueError('Too many parent columns: function expect %i parameters, %i was given.'
                                 % (len(self.f_params), len(parent_c)))
            for i, c in enumerate(parent_c):
                if isinstance(c, DSColumn):
                    if c.dataset is not dataset:
                        raise ValueError('%s is not a columns of %s!' % (c.name, dataset.dataset_name))
                    parent_c[i] = c.name
                elif c not in parent_columns_name:
                    raise ValueError('%s is not a columns of %s!' % (c, dataset.dataset_name))
            # Removing explicit parent columns from copied parent columns
            if remove_parent_columns is None:
                parent_copied_columns = [_ for _ in parent_copied_columns if _.name not in parent_c]

            # Solving implicit parent columns
            for p in self.f_params[len(parent_c):]:
                if p not in parent_columns_name:
                    raise ValueError('Could not find any correspondence to the not optional parameter: %s.' % (p,))
                parent_c.append(p)
            # Removing implicit and explicit parent columns from copied parent columns
            if remove_parent_columns:
                parent_copied_columns = [_ for _ in parent_copied_columns if _.name not in parent_c]

            # Check own columns
            for c_id, c in enumerate(own_c):
                if c in self._single_col_mapping:
                    raise ValueError('%s is already a column of %s!' % (c, self.dataset_name))

                # Apply column
                own_columns.append(DSColumn(c, None, None, self, None))
                self._single_col_mapping[c] = parent_c
            self._columns_mapping[own_c] = parent_c

        # Removing doublons from parent_copied_columns
        parent_copied_columns = [_ for _ in parent_copied_columns if _.name not in self._columns_mapping]

        self._columns = own_columns + parent_copied_columns

        # ---  HANDLE FUNCTION  ---
        self._f = function
        self._before_apply = before_apply
        self._after_apply = after_apply
        self._batchwise = batchwise
        self._n_factor = n_factor

        # ---  INFER COLUMN SHAPE, TYPE and FORMAT ---
        if format is None:
            format = {}
        elif isinstance(format, list):
            own_c_sample = next(iter(self._columns_mapping.keys()))
            if len(own_c_sample) != len(format):
                raise ValueError('%i format was expected but only %i was provided.' % (len(own_c_sample), len(format)))
            dict_format = {}
            for own_c in self._columns_mapping.keys():
                for f, c in zip(format, own_c):
                    dict_format[c] = f
            format = dict_format
        elif isinstance(format, (tuple, str, DSColumnFormat.Base, DSColumn)):
            format = {c_own: format for c_own in self._single_col_mapping}
        elif not isinstance(format, (dict, OrderedDict)):
            raise ValueError("columns_type_shape should be of type dict (provided type: %s)"
                             % type(format).__name__)

        # Try to infer
        unknown_columns_format = []
        for c in self._single_col_mapping:
            if c in format:
                # Check if format was given explicitly
                col_format = format[c]
                if col_format is None or (col_format == 'same' and 'same' not in dataset.columns_name()):
                    # Infer format from first parent
                    parent_column = dataset.column_by_name(self._single_col_mapping[c][0])
                    format[c] = (parent_column.dtype, parent_column.shape, parent_column.format)
                elif isinstance(col_format, str) and col_format in dataset.columns_name():
                    parent_column = dataset.column_by_name(col_format)
                    format[c] = (parent_column.dtype, parent_column.shape, parent_column.format)
                elif isinstance(col_format, DSColumn):
                    format[c] = (col_format.dtype, col_format.shape, col_format.format)
                elif isinstance(col_format, DSColumnFormat.Base):
                    format[c] = (col_format.dtype, col_format.shape, col_format)
                elif not isinstance(col_format, tuple) or len(col_format) not in (1, 2, 3):
                    unknown_columns_format.append(c)
            else:
                unknown_columns_format.append(c)

        # Read format from function
        if self._n_factor is None and len(unknown_columns_format) == 0:
            force_c = list(self._single_col_mapping.keys())[0]
            unknown_columns_format.append(force_c)
            f = format.pop(force_c)
            if len(f) == 3:
                format[force_c] = f[2]     # Set the format to DSColumnFormat to be consistent with real unknown columns.

        if unknown_columns_format:
            sample = dataset.read_one(0, columns=self.col_parents(unknown_columns_format), extract=False)
            while unknown_columns_format:
                unkown_col = unknown_columns_format[0]

                # Find arguments and return definition
                c_parent = self._single_col_mapping[unkown_col]
                c_own = self.col_sibling(unkown_col)

                # Call the function
                if self._batchwise:
                    c_samples = self.compute_f(args=sample[c_parent], rkeys=c_own, cols_name=c_parent)
                else:
                    c_samples = self.compute_f(args=sample[0, c_parent], rkeys=c_own, cols_name=c_parent)

                    # Read format
                for c_id, (c_name, c_sample) in enumerate(c_samples.items()):
                    parent_col = dataset.col[c_parent[c_id]] if len(c_parent) == len(c_samples) \
                                                             else dataset.col[c_parent[0]]
                    if self._n_factor is None:
                        self._n_factor = c_sample.shape[0]
                        if self._n_factor == 1:
                            self.pk._dtype = dataset.pk.dtype
                    elif self._n_factor != c_sample.shape[0]:
                        raise ValueError('The function returned %i rows for columns %s, but %i was expected.'
                                         % (c_sample.shape[0], c_name, self._n_factor))

                    c_sample = c_sample[0]
                    format_info = format.get(c_name, None)
                    if isinstance(c_sample, np.ndarray):
                        col_format = (c_sample.dtype, c_sample.shape)
                        if format_info:
                            format[c_name] = col_format + (format_info,)
                        elif c_sample.dtype == parent_col.dtype and c_sample.shape == parent_col.shape:
                            format[c_name] = col_format + (parent_col.format,)
                        else:
                            format[c_name] = col_format
                    else:
                        if type(c_sample) == str:
                            format[c_name] = 'str'
                        else:
                            dtype = np.dtype(type(c_sample))
                            if format_info:
                                format[c_name] = (dtype, (), format_info)
                            elif c_sample.dtype == parent_col.dtype and parent_col.shape == ():
                                format[c_name] = (dtype, (), parent_col.format,)
                            else:
                                format[c_name] = (dtype,)
                    if c_name in unknown_columns_format:
                        unknown_columns_format.remove(c_name)

        # Apply the format
        for c_name, c_format in format.items():
            col = self.column_by_name(c_name)
            if c_format == "str" or 'U' in str(c_format):
                col._dtype = np.dtype('O')
                col._is_text = True
                col.format = DSColumnFormat.Text()
                col._shape = ()
            else:
                col._dtype = c_format[0]
                col._shape = c_format[1] if len(c_format) > 1 else ()
                if len(c_format) > 2:
                    col.format = c_format[2]
                    col.format = c_format[2]
                else:
                    col.format = None

    def _generator(self, gen_context):
        i_global = gen_context.start_id
        n = gen_context.n
        columns = gen_context.columns

        copy_columns = [c for c in columns if c not in self._single_col_mapping]
        columns_mapping = {}
        for c_own, c_parent in self._columns_mapping.items():
            for c in c_own:
                if c in columns:
                    columns_mapping[c_own] = c_parent
                    break

        parent_n = int(np.ceil(n / self._n_factor))
        parent_gen = gen_context.generator(self._parent, columns=self.col_parents(columns), n=parent_n,
                                           start=gen_context.start_id//self._n_factor, stop=gen_context.stop_id // self._n_factor)


        result = None
        f_results = {}
        i_f = i_global % self._n_factor
        while not gen_context.ended():
            i_global, n, weakref = gen_context.create_result()
            r = weakref()
            for i in range(n):
                # Retrieve data, store f results in f_results
                if result is None:
                    parent_n = int(np.ceil((n-i) / self._n_factor))
                    if self._n_factor==1:
                        result = parent_gen.next(copy={c: r[i:i+parent_n, c] for c in copy_columns}, limit=parent_n, r=r)
                    else:
                        result = parent_gen.next(limit=parent_n, r=r)
                    f_results = {c: [] for c in stack(columns_mapping.keys())}
                    for c_own, c_parent in columns_mapping.items():
                        if self._batchwise:
                            f_results.update(self.compute_f(args=result[c_parent], rkeys=c_own, cols_name=c_parent))
                        else:
                            for j in range(len(result)):
                                for k, v in self.compute_f(args=result[j, c_parent], rkeys=c_own, cols_name=c_parent).items():
                                    f_results[k].append(v)

                            f_results = {k: np.concatenate(v) for k, v in f_results.items()}

                for c in r.columns_name():  # Store f results
                    if c in self._single_col_mapping:
                        r[i, c] = f_results[c][i_f]

                if self._n_factor != 1:     # Manual copy
                    for c in copy_columns:
                        r[i, c] = result[i_f//self._n_factor, c]

                if 'pk' not in self._single_col_mapping:    # Handle primary key
                    if self._n_factor > 1:
                        r[i, 'pk'] = str(result[i_f//self._n_factor, 'pk'])+str((i+i_global) % self._n_factor)
                    else:
                        r[i, 'pk'] = result[i_f, 'pk']

                # Clean
                i_f += 1
                if i_f == parent_n*self._n_factor:
                    for f in list(f_results.keys()):
                        del f_results[f]
                    del result
                    result = None
                    i_f = 0
            r = None
            yield weakref

    @property
    def size(self):
        return self._parent.size * self._n_factor

    def col_parents(self, columns=None):
        r = set()
        if columns is None:
            columns = self._single_col_mapping.keys()
        for c in columns:
            if c in self._single_col_mapping:
                r.update(self._single_col_mapping[c])
            else:
                r.add(c)
        return list(r)

    def col_sibling(self, column):
        for c_own in self._columns_mapping:
            if column in c_own:
                return c_own
        return None

    def compute_f(self, args, rkeys, cols_name):

        if not self._batchwise:
            r = {_: None for _ in rkeys}
            # for i in range(args_n):
            kwargs = {name: arg for name, arg in zip(self.f_params, args)}
            if self._before_apply:
                kwargs = self._before_apply(**kwargs)
            f_result = self._f(**kwargs)
            del kwargs

            if self._after_apply:
                f_result = self._after_apply(f_result)

            if not isinstance(f_result, list):
                if self._n_factor == 1 or self._n_factor is None:
                    f_result = [f_result]
                else:
                    raise ValueError('Function return a single row but %i was expected.' % self._n_factor)
            else:
                if self._n_factor is not None and len(f_result) != self._n_factor:
                    raise ValueError('The function returned %i rows but %i was expected.'
                                     % (len(f_result), self._n_factor))
            if not isinstance(f_result[0], tuple):
                if len(rkeys) != 1:
                    raise ValueError('The function returned a single column but %i was expected.'
                                     % len(rkeys))
                else:
                    r[rkeys[0]] = np.stack(f_result)                 # return [row1, row2, row3, ...]
            else:
                f_result = list(zip(*f_result))
                if len(f_result) != len(rkeys):
                    raise ValueError('The function returned %i columns but %i was expected.'
                                     % (len(f_result), len(rkeys)))
                for c_name, c_data in zip(rkeys, f_result):
                    r[c_name] = np.stack(c_data)      # return [(r1c1, r1c2, ...), (r2c1, r2c2, ...)]
            # print({k: v.shape for k, v in r.items()})
            return r
        else:
            args_n = args[0].shape[0]
            kwargs = {arg: c for arg, c in zip(self.f_params, args)}
            if self._before_apply:
                kwargs = self._before_apply(**kwargs)
            f_result = self._f(**kwargs)
            del kwargs

            if self._after_apply:
                f_result = self._after_apply(f_result)
            r = {}

            if not isinstance(f_result, tuple):
                if len(rkeys) != 1:
                    raise ValueError('The function returned a single column but %i was expected.'
                                     % len(rkeys))
                else:
                    f_result = (f_result,)
            else:
                if len(rkeys) != len(f_result):
                    raise ValueError('The function returned %i columns but %i was expected.'
                                     % (len(f_result), len(rkeys)))

            for c_name, c_data in zip(rkeys, f_result):
                if c_data.shape == ():
                    raise ValueError('The function was applied batchwise but returned an array with shape (). '
                                     '(Please note that even if the batchsize is 1, the array shape should be (1,))')
                if self._n_factor is not None and c_data.shape[0] != self._n_factor*args_n:
                    raise ValueError('The function returned %i rows for columns %s, but %i was expected.\n'
                                     '(The function is currently applied batchwise thus the data shape: %s should be '
                                     'match (batch_length, sample_dim1, sample_dim1, ...). If it\'s not the '
                                     'case you might consider setting batchwise=False).'
                                     % (c_data.shape[0], c_name, self._n_factor*args_n, c_data.shape))
                else:
                    r[c_name] = c_data
            return r


class DataSetApplyCV(DataSetApply):
    def __init__(self, dataset, function, columns=None, remove_parent_columns=True, format=None, n_factor=None,
                 converted_cols=None, force_mono=False, name='apply'):
        self.force_mono = force_mono
        self.converted_cols = converted_cols

        super(DataSetApplyCV, self).__init__(dataset=dataset, function=function, columns=columns, batchwise=False,
                                             remove_parent_columns=remove_parent_columns, format=format,
                                             n_factor=n_factor, name=name)

    def compute_f(self, args, rkeys, cols_name):
        if self.converted_cols is None:
            col_parents = [self.parent_dataset.column_by_name(c) for c in self.col_parents()]

            self.converted_cols = [c.name for c in col_parents if c.ndim in (2, 3)]

        split_n = None
        first_split_col_name = None
        split_kwargs = []
        final_dtype = None

        kwargs = {}
        for name, col_name, a in zip(self.f_params, cols_name, args):
            if col_name in self.converted_cols:
                if 'float' in str(a.dtype) or 'bool' in str(a.dtype):
                    final_dtype = a.dtype
                    a = a.astype(np.float, copy=False)
                    a = (a * 255.).astype(np.uint8)

                if a.ndim == 3:
                    a = a.transpose((1, 2, 0))
                    n = a.shape[2]

                    if not split_n:
                        split_n = n
                        first_split_col_name = col_name
                    elif split_n != n:
                        raise ValueError("Columns %s and %s don't have the same channel count "
                                         "(respectively %s and %s)"
                                         % (name, first_split_col_name, n, split_n))
                    split_kwargs.append(name)

                    if self.force_mono:
                        a = [a[:, :, _] for _ in range(split_n)]
                    else:
                        N = split_n // 3
                        split = [a[:, :, _ * 3:_ * 3 + 3] for _ in range(N)]
                        a = split + [a[:, :, N * 3 + _:N * 3 + _ + 1] for _ in range(split_n % 3)]
            kwargs[name] = a

        if split_n is None:
            split_n = 1
        elif not self.force_mono:
            split_n = split_n // 3 + split_n % 3

        kwargs = {k: v if k in split_kwargs else [v]*split_n for k, v in kwargs.items()}

        r = {_: [] for _ in rkeys}

        for i in range(split_n):
            f_kwargs = {k: v[i] for k, v in kwargs.items()}
            f_result = self._f(**f_kwargs)
            del f_kwargs

            if not isinstance(f_result, list):
                if self._n_factor == 1 or self._n_factor is None:
                    f_result = [f_result]
                else:
                    raise ValueError('Function return a single row but %i was expected.' % self._n_factor)
            else:
                if self._n_factor is not None and len(f_result) != self._n_factor:
                    raise ValueError('The function returned %i rows but %i was expected.'
                                     % (len(f_result), self._n_factor))
            if not isinstance(f_result[0], tuple):
                if len(rkeys) != 1:
                    raise ValueError('The function returned a single column but %i was expected.'
                                     % len(rkeys))
                else:
                    r[rkeys[0]].append(np.stack(f_result))  # return [row1, row2, row3, ...]
            else:
                f_result = list(zip(*f_result))
                if len(f_result) != len(rkeys):
                    raise ValueError('The function returned %i columns but %i was expected.'
                                     % (len(f_result), len(rkeys)))
                for c_name, c_data in zip(rkeys, f_result):
                    r[c_name].append(np.stack(c_data))  # return [(r1c1, r1c2, ...), (r2c1, r2c2, ...)]

        del kwargs
        f_result = {n: np.stack(d) for n, d in r.items()}
        # At this point, the shape of cv cols is (split_n, row, h, w, c)
        #       where row is usually 1 apart when f return several rows.
        #       It should be converted to (row, split_n*c, h, w)
        # The shape of not cv cols is (split_n, row, s1, s2, ...)
        #       and should be converted to (row, split_n, s1, s2, ...)

        r = {}
        for k, a in f_result.items():
            # Type conversion if a is an uint8 image
            if a.dtype == np.uint8 and final_dtype and (a.ndim==4 or (a.ndim==5 and a.shape[4] in (1,3))):
                a = (a.astype(np.float)/255).astype(final_dtype, copy=False)

            if a.ndim == 5 and a.shape[4] in (1, 3):
                a = np.concatenate(a, axis=-1).transpose((0, 3, 1, 2))
            else:
                a = a.swapaxes(0, 1)
            if split_n == 1 and a.shape[1] == 1:
                a.squeeze(1)

            r[k] = a

        return r
