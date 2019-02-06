import numpy as np
import time

from .dataset import AbstractDataSet, DSColumn, DSColumnFormat
from ..j_utils.j_log import log
from ..j_utils.function import match_params, not_optional_args
from ..j_utils.math import interval
from ..j_utils.collections import if_else, if_none


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


########################################################################################################################
class PyTableDataSet(AbstractDataSet):
    def __init__(self, pytable, where=None, sortby=None, name='PyTableDataset'):
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

        for i in sorted(col_descr.keys()):
            col_name, col_shape, col_dtype = col_descr[i]
            self.add_column(col_name, col_shape, col_dtype)

        self._sorted_rows = None
        self._filtered_rows = None
        self._rows = None

        self.sortby = sortby
        self._solve_sortby()
        self.where = where
        self._solve_where()

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

            self._sorted_rows = self.pytable._check_sortby_csi(sortby=sortby, checkCSI=False)
            if reversed_sort is not None:
                self._sorted_rows = self._sorted_rows[::-1]
            if self._filtered_rows is not None:
                rows = np.zeros(shape=(self.pytable.nrows,), dtype=np.bool)
                rows[self._filtered_rows] = 1
                self._rows = self._sorted_rows[rows[self._sorted_rows]]
            else:
                self._rows = self._sorted_rows

    @staticmethod
    def from_file(path, table="dataset", name='PyTableDataset'):
        from ..j_utils.path import open_pytable
        f = open_pytable(path)
        table_path, table_name = table.rsplit('/', 1)
        if not table_path.startswith('/'):
            table_path = '/'+table_path
        table = f.get_node(table_path, table_name, classname='Table')
        return PyTableDataSet(table, name=name)

    @staticmethod
    def from_numpy(data_dict, cache_path=None, name='PyTableDataset'):
        import tables
        from ..j_utils.path import open_pytable
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
                    r[c] = row[c]

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
            return self.where(sortby)
        d = PyTableDataSet(self.pytable, where=where, sortby=sortby, name=self.dataset_name)
        return d


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
            yield gen.next(copy={c: weakref()[c] for c in gen_context.columns})

    @property
    def size(self):
        return self.stop - self.start

    def subset(self, start=0, stop=None, *args):
        if len(args) == 1:
            start = 0
            stop = args[0]
        elif len(args) == 2:
            start = args[0]
            stop = args[1]
        return DataSetSubset(self._parent, start + self.start, min(self.start + stop, self.stop))


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
                parent_gen.next(copy={c: r[i:i+n, c] for c in r.columns_name() + ['pk']})

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
        for column, real_columns in mapping.items():
            if isinstance(real_columns, str):
                mapping[column] = [real_columns]
                mapped_cols.add(real_columns)
            elif isinstance(real_columns, (list, tuple)):
                mapped_cols.update(real_columns)
            else:
                raise ValueError('Invalid mapping value: %s.\n '
                                 'Valid value are column name for mapping and list or tuple for concatenation.'
                                 % real_columns)

        if keep_all_columns:
            for c in set(dataset.columns_name())-mapped_cols:
                mapping[c] = [c]

        self.concatenation_map = mapping
        for column, real_columns in mapping.items():
            c = dataset.column_by_name(real_columns[0])
            shape = c.shape
            for real_column in real_columns[1:]:
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
            _, N, weakref = gen_context.create_result()
            r = weakref()
            result = gen.next(copy={c_parent: r[c_name][:, i:i+n] if n > 0 else r[:, c_name]
                              for c_parent, (c_name, i, n) in copy_columns.items()}, limit=N, r=r)
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
    def __init__(self, dataset, indices=None, subgen=1, name='shuffle', rnd=None):
        """
        :type dataset: AbstractDataSets        if args
        """
        super(DataSetShuffle, self).__init__(name, dataset, pk_type=dataset.pk.dtype)
        self._columns = dataset.copy_columns(self)

        self.rnd = rnd

        self.indices = None if indices is None else np.asarray(indices, dtype=np.uint32)
        self.random_indices = indices is None

        self.subgen = max(subgen,0) if isinstance(subgen, int) else 0
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
            self.rnd.shuffle(rand_seq)

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
            self.rnd.shuffle(rand_seq)
            return rand_seq

    def _setup_determinist(self):
        if self.random_indices:
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

        #indices = indices[gen_context.start_id:gen_context.stop_id]

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
                seq = list(indices[i_global:i_global+n])
                if len(subgen) > 1:  # -- Mutliple core, No subgen --
                    waiting_seq = len(seq)
                    while waiting_seq:
                        waiting = True
                        for s in subgen:
                            if s.seq_id is not None and s.poll(copy=r[s.seq_id:s.seq_id+1], r=r, ask_next=False):
                                waiting = False
                                s.seq_id = None
                                waiting_seq -= 1

                            if seq and s.seq_id is None:     # Ask next
                                s.seq_id = n - len(seq)
                                s.ask(seq.pop(0))

                        if waiting:
                            time.sleep(1e-3)

                else:       # -- Single core, No subgen --
                    for i, seq_id in enumerate(seq):
                        subgen[0].next(copy=r[i:i+1], r=r, seek=seq_id)
                    if gen_context.is_last() or indices[i_global+n] != indices[i_global+n-1]+1:
                        subgen[0].clean()
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

    def subset(self, start=0, stop=None, *args):
        from copy import deepcopy
        if len(args) == 1:
            start = 0
            stop = args[0]
        elif len(args) == 2:
            start = args[0]
            stop = args[1]
        if not 0 <= start < self.size:
            start = 0
        if not start <= stop < self.size:
            stop = self.size

        sub = deepcopy(self)
        sub._name += '_Subset'
        sub._join = self._join[start:stop, :]
        return sub


def join(datasets, **kwargs):
    return DataSetJoin(datasets, **kwargs)


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
            columns[col_name] = col.shape, col.dtype

        # -- Setup dataset --
        super(DataSetConcatenate, self).__init__(name=name, parent_datasets=datasets, pk_type=str)
        self._columns = [DSColumn(name, shape, dtype, self) for name, (shape, dtype) in columns.items()]
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

                if parent_gen.ended():
                    parent_gen = None

                for i_pk in range(n):
                    r[i + i_pk, 'pk'] = parent_gen.dataset.dataset_name + '|' + str(result[i_pk, 'pk'])
                    for c in default_cols:
                        r[i + i_pk, c] = self._columns_default[c]
                i += n

            r = None
            yield weakref

    @property
    def size(self):
        return sum(_.size for _ in self.parent_datasets)


########################################################################################################################
class DataSetApply(AbstractDataSet):
    """
    Apply a function to some columns of a dataset, all other columns are copied
    """
    def __init__(self, dataset, function, columns=None, remove_parent_columns=True, cols_format=None, format=None,
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

        :param cols_format: A dictionary mapping columns names to a tuple specifying (dtype, shape [, format]). For
                                every column not described here, its type and shape will be read after applying the
                                function to the first row of the dataset. If a column name is given instead of a tuple,
                                the type, shape and format will be copied from the **dataset** column. If None is given,
                                a similar behaviour will be attempted based on the column name.
                            Finnally, this parameters could be set to 'same'. In this case, all column type, shape and
                            format will be copied from **dataset**.

        :param format: the format of columns returned by the function

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

        if columns is None:
            columns = self.columns_name()
        elif isinstance(columns, str):
            if columns in parent_columns_name:
                columns = {columns: columns}
            else:
                columns = (columns,)
        if isinstance(columns, list):
            columns = {_: _ if _ in parent_columns_name else None for _ in columns}
        elif isinstance(columns, tuple):
            if all(_ in dataset.col for _ in columns):
                columns = {columns: columns[:len(self.f_params)]}
            else:
                columns = {columns: ()}
        if not isinstance(columns, dict):
            raise ValueError('Columns should be of the following type str, tuple, list, dict.'
                             '(type provided: %s)' % type(columns).__name__)

        for own_c, parent_c in columns.items():
            if isinstance(own_c, str):
                own_c = (own_c,)
            if parent_c is None or not parent_c:
                parent_c = []
            elif isinstance(parent_c, (str, DSColumn)):
                parent_c = [parent_c]
            elif isinstance(parent_c, tuple):
                parent_c = list(parent_c)
            else:
                raise ValueError('Invalid columns description values: %s.' % parent_c)

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
            # Removing explicit parent columns from
            if remove_parent_columns:
                parent_copied_columns = [_ for _ in parent_copied_columns if _.name not in parent_c]

            # Solving implicit parent columns
            for p in self.f_params[len(parent_c):]:
                if p not in parent_columns_name:
                    raise ValueError('Could not find any correspondence to the not optional parameter: %s.' % (p,))
                parent_c.append(p)

            # Check own columns
            for c_id, c in enumerate(own_c):
                if c in self._single_col_mapping:
                    raise ValueError('%s is already a column of %s!' % (c, self.dataset_name))

                # Apply column
                own_columns.append(DSColumn(c, None, None, self, None))
                self._single_col_mapping[c] = parent_c
            self._columns_mapping[own_c] = parent_c

        self._columns = own_columns + parent_copied_columns

        # ---  HANDLE FUNCTION  ---
        self._f = function
        self._batchwise = batchwise
        self._n_factor = n_factor

        # ---  INFER COLUMN SHAPE, TYPE and FORMAT ---
        if cols_format == 'same':
            cols_format = {c_own: None for c_own in self._single_col_mapping}
        elif cols_format is None:
            cols_format = {}
        if not isinstance(cols_format, dict):
            raise ValueError("columns_type_shape should be of type dict (provided type: %s)"
                             % type(cols_format).__name__)

        if not isinstance(format, dict):
            format = {_: format for _ in self._single_col_mapping.keys()}
        elif not all(_ in self._single_col_mapping for _ in format.keys()):
            format = {_: format for _ in self._single_col_mapping.keys()}


        # Try to infer
        unknown_columns_format = []
        for c in self._single_col_mapping:
            if c in cols_format:
                # Check if format was given explicitly
                col_format = cols_format[c]
                if col_format is None or (col_format == 'same' and 'same' not in dataset.columns_name):
                    # Try to infer format from homonym
                    parent_column = dataset.column_by_name(c, False)
                    if parent_column is not None:
                        cols_format[c] = (parent_column.dtype, parent_column.shape, parent_column.format)
                    # Infer format from first parent
                    parent_column = dataset.column_by_name(self._single_col_mapping[c][0])
                    col_format = format.get(c, parent_column.format)
                    cols_format[c] = (parent_column.dtype, parent_column.shape, col_format)
                elif isinstance(col_format, str) and col_format in dataset.columns_name:
                    parent_column = dataset.column_by_name(col_format)
                    cols_format[c] = (parent_column.dtype, parent_column.shape, parent_column.format)
                elif not isinstance(col_format, tuple) or len(col_format) not in (1, 2, 3):
                    unknown_columns_format.append(c)
            else:
                unknown_columns_format.append(c)

        # Read format from function
        if self._n_factor is None and len(unknown_columns_format) == 0:
            unknown_columns_format.append(list(self._single_col_mapping.keys())[0])

        if unknown_columns_format:
            sample = dataset.read_one(0, columns=self.col_parents(unknown_columns_format), extract=False)
            while unknown_columns_format:
                unkown_col = unknown_columns_format[0]

                # Find arguments and return definition
                c_parent = self._single_col_mapping[unkown_col]
                c_own = self.col_sibling(unkown_col)

                # Call the function
                c_samples = self.compute_f(args=sample[c_parent], rkeys=c_own)

                # Read format
                for c_name, c_sample in c_samples.items():
                    if self._n_factor is None:
                        self._n_factor = c_sample.shape[0]
                        if self._n_factor == 1:
                            self.pk._dtype = dataset.pk.dtype
                    elif self._n_factor != c_sample.shape[0]:
                        raise ValueError('The function returned %i rows for columns %s, but %i was expected.'
                                         % (c_sample.shape[0], c_name, self._n_factor))

                    c_sample = c_sample[0]
                    if isinstance(c_sample, np.ndarray):
                        cols_format[c_name] = (c_sample.dtype, c_sample.shape)
                    else:
                        cols_format[c_name] = (np.dtype(type(c_sample)),)
                    if c_name in unknown_columns_format:
                        unknown_columns_format.remove(c_name)

        # Apply the format
        for c_name, c_format in cols_format.items():
            col = self.column_by_name(c_name)
            col._dtype = c_format[0]
            col._shape = c_format[1] if len(c_format) > 1 else ()
            col.format = c_format[2] if len(c_format) > 2 else format.get(c_name, None)

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
                    f_results = {}
                    for c_own, c_parent in columns_mapping.items():
                        f_results.update(self.compute_f(args=result[c_parent], rkeys=c_own))

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

    def col_parents(self, columns):
        r = set()
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

    def compute_f(self, args, rkeys):
        args_n = args[0].shape[0]

        if not self._batchwise:
            r = {_: [] for _ in rkeys}
            for i in range(args_n):
                kwargs = {name: arg[i] for name, arg in zip(self.f_params, args)}
                f_result = self._f(**kwargs)
                del kwargs

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
                        r[rkeys[0]] += f_result                 # return [row1, row2, row3, ...]
                else:
                    f_result = list(zip(*f_result))
                    if len(f_result) != len(rkeys):
                        raise ValueError('The function returned %i columns but %i was expected.'
                                         % (len(f_result), len(rkeys)))
                    for c_name, c_data in zip(rkeys, f_result):
                        r[c_name] += c_data      # return [(r1c1, r1c2, ...), (r2c1, r2c2, ...)]

            return {n: np.stack(d) for n, d in r.items()}
        else:
            kwargs = {arg: c for arg, c in zip(self.f_params, args)}
            f_result = self._f(**kwargs)
            del kwargs

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
                if self._n_factor is not None and c_data.shape[0] != self._n_factor*args_n:
                    raise ValueError('The function returned %i rows for columns %s, but %i was expected.'
                                     % (c_data.shape[0], c_name, self._n_factor*args_n))
                else:
                    r[c_name] = c_data
            return r


class DataSetApplyCV(DataSetApply):
    def __init__(self, dataset, function, columns=None, remove_parent_columns=True, cols_format=None, n_factor=None,
                 name='apply'):
        super(DataSetApplyCV, self).__init__(dataset=dataset, function=function, columns=columns, batchwise=False,
                                             remove_parent_columns=remove_parent_columns, cols_format=cols_format,
                                             n_factor=n_factor, name=name)

    def compute_f(self, args, rkeys):

        args_n = args[0].shape[0]

        r = {_: [] for _ in rkeys}
        for i in range(args_n):
            kwargs = {}
            for name, a in zip(self.f_params, args):
                a = a[i]
                if isinstance(a, np.ndarray):
                    if a.ndim == 3 and a.shape[0] in (1, 3):
                        a = a.transpose((1, 2, 0))
                    if 'float' in str(a.dtype):
                        a = (a * 255).astype(np.uint8)
                kwargs[name] = a
            f_result = self._f(**kwargs)
            del kwargs

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
                    r[rkeys[0]] += f_result  # return [row1, row2, row3, ...]
            else:
                f_result = list(zip(f_result))
                if len(f_result) != len(rkeys):
                    raise ValueError('The function returned %i columns but %i was expected.'
                                     % (len(f_result), len(rkeys)))
                for c_name, c_data in zip(rkeys, f_result):
                    r[c_name].append(np.stack(c_data))  # return [(r1c1, r1c2, ...), (r2c1, r2c2, ...)]

        f_result = {n: np.stack(d) for n, d in r.items()}
        r = {}
        for k, a in f_result.items():
            if isinstance(a, np.ndarray) and a.ndim == 4 and a.shape[3] in (1, 3):
                a = a.transpose((0, 3, 1, 2))
            if a.dtype == np.uint8:
                a = a.astype(np.float)/255
            r[k] = a

        return r
