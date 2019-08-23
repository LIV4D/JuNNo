"""This module is used to build **Dataset**
which are a convenient way to deal with database. Each dataset contains a primary key identifying a row,
and one or many columns. A column is an instance of :class:`DataSetColumn`.

In the general idea, a dataset never loads in memory the whole database, only a subset of it.
When the data-processing of the subset is over, the internal structure of the dataset object handles the loading of the
following adjacent subset.
Therefore, a memorization of the last index covered is stored: the new subset can be retrieved from there.
The underlying idea is to use generator to optimize memory consumption.

The main strength of the **Dataset** system is to propose a great flexibility on the processing of the data loaded,
subset by subset. For example, it is possible to decompose a **Dataset** of image in a **Dataset** of patches
and apply data augmentation on them.
Technically speaking, the first **Dataset** will transmit a reference to itself to the **DataSetPatches**,
which in its turn will transmit its reference to the **DataSetAugmentedData**. During those exchanges,
not a single operation is made, until a call to the last **Dataset**'s genetator is made
(in this case, **DataSetAugmentedData**).
Indeed, the exchanges only transfer reference of object's parent. This reference allows accessing to the generator
object from a parent **Dataset**. When calling the last **Dataset**'s generator (to access real data),
the generator will call in chain its parent's generator thanks to the reference it has of it.

The generator logic allows therefore to see the construction of **Dataset** as a list of
instruction to apply to the data from a database before using them in a actual implementation.

.. note::
    In a deep learning framework, this is useful as the training part usually takes place on the GPU and the loading is
    handled by the CPU. Using two threads, those operations can therefore be computed in parallel.

:Example:

>>> img_dataset = ImagesCollection(DATA_PATH, crop=(160, 450)) # A dataset composed of images. The dataset has the columns 'data' and 'name'. No actual loading has been made yet.
>>> labels_dataset = ImagesCollection(DATA_PATH+'labels', crop=(160, 450)) # Another dataset composed of images.
>>> ds_join = DataSetJoin([labels_dataset.name, img_dataset.name], name=labels_dataset.name, labels=labels_dataset.model, img=img_dataset.model)

The last operation joins the two datasets according to their name, and create a new dataset with columns *name*, *labels* and *img*
No real operation has been made here. The operation will eventually be made when calling the generator from ds_join.

:Example:

>>> def labelisation(x):
>>>     _, h, w = x.shape
>>>     return h - np.sum(np.round(x[0, :, :]), axis=0)
>>> train = ds_join.patches(['img', 'labels'], patch_shape=(128, 64), stride=(10, 16)).apply(labelisation, 'labels').shuffle()

A train dataset is then created, which is composed of patches having a size of (128, 64),
separated with a horizontal stride of 10 and a vertical stride of 16.
Then, a custom function labelisation is applied on the columns *labels*.
Finally, the dataset is shuffle.

:Example:

>>> train.sql_write('data.db', 'train', compress_img=True)

The dataset is then written in a SQLite database, in a file called 'data.db' and a table called 'train'.

The module is composed of the classes:
    * :class:`AbstractDataSet`
    * :class:`DataSetColumn`
    * :class:`DataSetResult`
    * :class:`NumPyDataSet`
    * :class:`DataSetSubset`
    * :class:`DataSetConcatMap`
    * :class:`DataSetRandomize`
    * :class:`DataSetJoin`
    * :class:`DataSetApply`
    * :class:`DataSetAugmentedData`
    * :class:`DataSetPatches`

"""
from abc import ABCMeta, abstractmethod
import base64
from collections import OrderedDict
from functools import partial
import multiprocessing as mp
import numpy as np
import os
from os.path import dirname, exists, join, basename, abspath
from os import makedirs
import weakref

from ..j_utils.j_log import log, Process

from ..j_utils.function import not_optional_args, to_callable, optional_args
from ..j_utils.parallelism import N_CORE
from ..j_utils.math import interval, apply_scale
from ..j_utils.collections import AttributeDict, Interval, if_none, is_dict


class AbstractDataSet(metaclass=ABCMeta):
    """Abstract class of a dataset. Each dataset must inherited of this class.

    Store information describing a DataSet. Data from it can be retrieved using generator().

    To define a new dataset, one should specify 3 characteristics::
        * Dataset columns: self._columns should contains the list of columns of this dataset. \
                           A DataSetColumn is defined by the name of the column, its parent dataset, \
                           its shape (without the batch dimension, for ex. a scalar's shape is () ), \
                           and its data type (similar to np.array.dtype)
        * Dataset size: The number of row inside this dataset. It should be invariant once the dataset is constructed.
        * Dataset data: the generator(n, from_id) method should be overloaded to read batches of data from the dataset.

    """

    def __init__(self, name='dataset', parent_datasets=None, pk_type=np.int, rng=None):
        """ Constructor of the class

        The constructor instanciates the specific attribute of the dataset (name and list of parents datasets) and
        builds the instance of the :class:`DataSetColumn` that corresponds to the primary key.

        :param name: The name of the dataset
        :param parent_datasets: The list of parent of the dataset. If None is passed, it will be interpretated as an empty list.
        :param pk_type: The type of the primary key.
        :type parent_datasets: None or list
        :type name: str
        """
        self._columns = []
        self._name = name

        if parent_datasets is None:
            parent_datasets = []
        elif isinstance(parent_datasets, AbstractDataSet):
            parent_datasets = [parent_datasets]
        elif not isinstance(parent_datasets, list):
            parent_datasets = list(parent_datasets)

        self._parents = parent_datasets
        self._pk = DSColumn('pk', (), pk_type, self)

        self._sample = None
        if rng is None and self.parent_dataset is not None:
            self.rng = self.parent_dataset.rng
        else:
            self.rng = rng

    def __getstate__(self):
        d = self.__dict__
        d['_sample'] = None
        return d

    #   ---   Dataset properties   ---
    def __len__(self):
        return self.size

    @property
    def size(self):
        """
        Return the size of the dataset (value of the first dimension of every columns)
        """
        return -1

    def __getitem__(self, item):
        if isinstance(item, slice) or isinstance(item, int):
            item = (item, None)
        if isinstance(item, tuple):
            if isinstance(item[0], int):
                return self.read_one(row=item[0], columns=item[1], extract=True)
            elif isinstance(item[0], slice):
                return self.read(item[0].start, item[0].stop, columns=item[1], extract=True)
        raise NotImplementedError

    @property
    def rng(self):
        return self._rnd

    @rng.setter
    def rng(self, rnd):
        if rnd is None:
            rnd = np.random.RandomState(1234+os.getpid())
        elif isinstance(rnd, int):
            rnd = np.random.RandomState(rnd)
        self._rnd = rnd

    def step_rng(self):
        rng_states = []
        for d in self.walk_parents():
            if all(d.rng is not _ for _ in rng_states):
                d.rng.uniform()  # Step
                rng_states.insert(0, d.rng)

    @property
    def dataset_name(self):
        return self._name

    @property
    def dataset_fullname(self):
        fullname = self._name
        n = self
        while len(n._parents) == 1:
            fullname = n._parent.dataset_name + '_' + fullname
            n = n._parent
        return fullname

    #   ---   Data direct access   ---
    def read(self, start: int = None, stop: int = None, columns=None, extract=False, n=None, determinist=True):
        if start is None:
            start = 0
        if stop is None:
            stop = self.size
        elif stop < 0:
            stop -= self.size
        elif stop < start:
            stop = start + 1
        if stop > self.size:
            raise ValueError('%i is not a valid index: dataset size is %i' % (stop, self.size))

        d = self
        if n is not None:
            d = d.subgen(n)

        gen = d.generator(stop - start, start=start, columns=columns, determinist=determinist)
        r = next(gen)

        if not extract:
            return r
        if columns is None:
            return {c_name: c_data for c_name, c_data in r.items()}
        elif isinstance(columns, list):
            return [r[_] for _ in self.interpret_columns(columns)]
        else:
            return r[columns]

    def gen_read(self, id, n=1, columns=None, gen_context=None, clear_weakref=False):
        if gen_context is not None:
            gen = gen_context.generator(self, start=id, stop=id+n, n=n, columns=columns)
        else:
            gen = self.generator(start=id, stop=id+n, n=n, columns=columns)
        r = gen.next()
        gen.clean()
        if clear_weakref:
            r.clear_dataset()
            return r

        return r, gen

    def read_one(self, row=0, columns=None, extract=False, determinist=True):
        """
        Read a specific element of a dataset. If extract is True, the result will depend on the form of columns.
        Thus, if columns is None, read_one(i) will return a dictionnary a the i-th value of all the dataset's columns,
              if columns is a list, read_one will organize the values in a list in the same order as in columns
        finally if columns is a string, only the element at this position and column will be returned.
        :param row: Row of the wanted elements
        :param columns: Columns of the wanted elements (None mean all of them)
        :param extract: If true, the data is extracted from the DataSetResult.
        """
        r = self.read(start=row, stop=row + 1, columns=columns, extract=False, determinist=determinist)

        if not extract:
            return r
        if columns is None:
            return {c_name: c_data[0] for c_name, c_data in r.items()}
        elif isinstance(columns, list):
            return [r[_][0] for _ in self.interpret_columns(columns)]
        else:
            return r[columns][0]

    @property
    def at(self):
        return AbstractDataSet.DatasetReader(self)

    class DatasetReader:
        def __init__(self, dataset):
            self._dataset = dataset

        @property
        def dataset(self):
            return self._dataset

        def __getitem__(self, item):
            if isinstance(item, slice) or isinstance(item, int):
                item = (item, None)
            if isinstance(item, tuple):
                if isinstance(item[0], int):
                    return self.dataset.read_one(row=item[0], columns=item[1], extract=False)
                elif isinstance(item[0], slice):
                    return self.dataset.read(item[0].start, item[0].stop, columns=item[1], extract=False)
            raise NotImplementedError

        def __call__(self, row, columns=None, determinist=True):
            return self.dataset.read_one(row=row, columns=columns, extract=False, determinist=True)

    #   ---   Generators   ---
    def generator(self, n=1, start=None, stop=None, columns=None, determinist=False, intime=False, ncore=0):
        """Creates a generator which iterate through data.

        :param n:  Number of element to return (maximum) by iteration
        :param start: index from which the generator will start reading data
        :param columns: list of columns to read
        :type columns: None or list of str or list of :class:`DSColumn`
        :type start: int
        :type n: int

        :return The generator which loops start at from_id and does n iterations.
        :rtype: generator
        """
        self.clear_sample()
        from .dataset_generator import DataSetSmartGenerator
        return DataSetSmartGenerator(dataset=self, n=n, start_id=start, stop_id=stop, columns=columns,
                                     determinist=determinist, intime=intime, ncore=ncore)

    def __iter__(self):
        return self.generator(determinist=True, intime=False, ncore=0)

    @abstractmethod
    def _generator(self, gen_context):
        pass

    def _setup_determinist(self):
        pass

    #   ---   Columns   ---
    def interpret_columns(self, columns, to_column_name=True, exists=True):
        """Returns a list of columns name

        :param columns: None, list of :class:`DataSetColumn` or str, str or :class:`DataSetColumn`. \
        If None, the method returns the columns' name of the current dataset.
        :return: list of columns name
        :rtype: list
        """
        assert to_column_name or exists
        if columns is None:
            columns = self._columns.copy()
        elif isinstance(columns, (tuple, set)):
            columns = list(columns)
        elif isinstance(columns, str):
            columns = [_.strip() for _ in columns.split(',')]

        elif not isinstance(columns, list):
            columns = [columns]
        else:
            columns = columns.copy()

        for c_id, c in enumerate(columns):
            if exists:
                if isinstance(c, DSColumn):
                    if c.dataset is not self:
                        raise ValueError('%s is not a column of %s' % (c, self.dataset_name))
                    if to_column_name:
                        columns[c_id] = c.name
                elif isinstance(c, str):
                    if c != 'pk' and not c in self.columns_name():
                        raise ValueError('%s is not a column of %s' % (c, self.dataset_name))
                    if not to_column_name:
                        columns[c_id] = self.column_by_name(c)
            else:
                if isinstance(c, DSColumn):
                    columns[c_id] = c.name

        return columns

    def column_by_name(self, name, raise_on_unknown=True):
        if name == 'pk':
            return self.pk
        for c in self._columns:
            if c.name == name:
                return c
        if raise_on_unknown:
            raise (ValueError('Unknown column %s in dataset %s' % (name, self._name)))
        return None

    def column_index(self, name):
        for c_id, c in enumerate(self._columns):
            if c.name == name:
                return c_id
        return -1

    def columns_name(self):
        return [_.name for _ in self._columns]

    @property
    def columns(self):
        r = AttributeDict()
        for c in self._columns:
            r[c.name] = c
        return r

    @property
    def col(self):
        return self.columns

    @property
    def pk(self):
        return self._pk

    def copy_columns(self, dataset=None):
        if dataset is None:
            dataset = self
        return [DSColumn(_.name, _.shape, _.dtype, dataset, _.format) for _ in self._columns]

    def add_column(self, name, shape, dtype, format=None):
        self._columns.append(DSColumn(name=name, shape=shape, dtype=dtype, dataset=self, format=format))

    def format(self, columns, format):
        if not isinstance(columns, (list, tuple, set)):
            columns = [columns]
        columns = self.interpret_columns(columns, to_column_name=False)

        if isinstance(format, list):
            if len(format) != len(columns):
                raise ValueError('%i format were expected, %i received...' % (len(columns), len(format)))
            format = {c: f for c, f in zip(columns, format)}
        elif not is_dict(format):
            format = {c: format for c in columns}

        for c in columns:
            c.format = format[c]

        return self

    #   ---   Dataset Hierarchy   ---
    @property
    def parent_datasets(self):
        return self._parents

    @property
    def _parent(self):
        return self._parents[0] if len(self._parents) else None

    @property
    def parent_dataset(self):
        return self._parent

    @property
    def root_dataset(self):
        root = self
        while root.parent_dataset is not None:
            root = root.parent_dataset
        return root

    def walk_parents(self):
        """
        Walk through all parent datasets
        /!\ The same datasets may be listed multiple times if two or more of its child have been joined!
        :return:
        """
        active = self.parent_datasets[:]
        while active:
            d = active.pop()
            yield d
            active += d.parent_datasets

    #   ---   Representation functions   ---
    def __str__(self):
        return self.dataset_fullname + ': ' + self.__class__.__name__ + '()'

    def __repr__(self):
        s = str(self)
        for column in self._columns:
            s += '\n\t'+str(column)
        return s

    def _ipython_display_(self):
        from ..j_utils.ipython import import_display
        import_display(self.ipywidget())

    def ipywidget(self):
        from ..j_utils.ipython.databaseview import DatabaseView, SimpleDatabaseView

        datasets = [self]
        while len(datasets[-1].parent_datasets) == 1:
            datasets.append(datasets[-1].parent_dataset)
        datasets = datasets[::-1]

        w = DatabaseView([d.dataset_fullname.split('_')[-1] for d in datasets])

        def dataset_changed(id):
            dataset = datasets[id]
            w.length = dataset.size

            context = {'generator': None, 'gen_id': -1, 'result': None}

            columns = []
            columns_description = []
            for c in dataset._columns:
                columns.append(c)
                c_name = c.name + ';' + c.format.dtype_name
                if len(c.shape) > 0:
                    c_name += ' [' + ('x'.join([str(_) for _ in c.shape])) + ']'

                columns_description.append(c_name)

            def retreive_data(row, col, fullscreen_id=None):
                if context['generator'] is not None and context['gen_id']+1 == row:
                    context['gen_id'] += 1
                    context['result'] = next(context['generator'])
                elif context['gen_id'] != row:
                    context['generator'] = dataset.generator(start=row, determinist=True)
                    context['gen_id'] = row
                    context['result'] = next(context['generator'])

                c = columns[col]
                d = context['result'][0, c.name]
                if fullscreen_id is None:
                    return c.format.export_html(d), c.format.html_fullscreen
                else:
                    return c.format.export_html(d, fullscreen_id)

            w.columns_name = columns_description
            w.db_view.retreive_data = retreive_data
            w.db_view.retreive_fullscreen = retreive_data
            w.db_view.reset()

        dataset_changed(len(datasets)-1)
        w.hierarchy_bar.on_id_changed(dataset_changed)

        return w

    #   ---   Sample   ---
    @property
    def sample(self):
        if self._sample is None:
            self._sample = self.read_one(extract=True)
        return self._sample

    def clear_sample(self):
        del self._sample
        self._sample = None

    #   ---   Export   ---
    def export(self, cb, n=1, start=0, stop=None, columns=None, determinist=True, ncore=None):
        columns = self.interpret_columns(columns)
        start, stop = interval(self.size, start, stop)

        size = stop-start
        if size <= 0:
            return

        if ncore is None:
            ncore = N_CORE
        ncore = min(ncore, size)

        if ncore > 1:
            # Setup generators
            gens = []
            for id_gen in range(ncore):
                start_id = start + np.round(size / ncore * id_gen)
                stop_id = start + np.round(size / ncore * (id_gen + 1))

                g = self.generator(n=n, start=start_id, stop=stop_id, columns=columns, intime=True,
                                   determinist=determinist)
                g.setup()
                gens.append(g)

            # Read from generators
            while gens:
                # Poll from active generators
                remove_gen = []
                for gen_id, gen in enumerate(gens):
                    try:
                        r = gen.poll()
                    except StopIteration:
                        remove_gen.append(gen_id)
                        continue
                    if r is not None:
                        cb(r)

                # Remove ended generators
                for gen_id in reversed(remove_gen):
                    gens[gen_id].clean()
                    del gens[gen_id]

        else:
            for r in self.generator(n=n, start=start, stop=stop, columns=columns, determinist=determinist):
                cb(r)

    def sql_write(self, database_path, table_name, start=0, stop=0, n=10,
                  compress_img=True, include_pk=False, replace_table=False, show_progression=True, compress_format='.png'):
        """Write the current dataset in a SQLite file.

        The method retranscribes the internal relation of the current dataset (rows and columns) in a SQLite logic. If the filepath
        given does not correspond to a existing file, a file is created. Otherwise, the existing file is only modified, by creating
        a new table or updating the previous one.

        :Example:

        >>> train.sql_write('data.db', 'train', start=-5, compress_img=True, include_pk=True)


        :param database_path: The path to the database file that will be created.
        :param table_name: The table name in the database
        :param start: The starting point in the current dataset's generator. This can be a negative number, \
        in which case, it works as in list (starting from the end). If negative or null, will be replaced by start+dataset.length
        :param end: The ending point in the current dataset's generator. If negative or null, will be replaced by end+dataset.length
        :param n: Number of rows yielded by the generator.
        :param compress_img: Choose whether or not images should be compressed.
        :param include_pk: Choose whether or not the primary key should be include in the SQLite database (and mark as it in the SQLite syntax)
        :param replace_table: Choose whether replace or append data in pre-existing table in the SQLite database (if the table already exists).
        :param show_progression: Display² a progression bar in the terminal
        :type database_path: str
        :type table_name: str
        :type end: int
        :type start: int
        :type n: int
        :type compress_img: bool
        :type include_pk: bool
        :type replace_table: bool
        :type show_progression: bool
        """
        from ..j_utils.sql_tools import SQLAccessor, CreateTableQuery, InsertQuery

        compress = []
        if compress_img:
            compress = [c.name for c in self._columns if len(c.shape) == 3 and c.shape[0] in (1, 3)]

        columns = self.columns_name()

        if start < 0:
            start += self.size
        if stop <= 0:
            stop += self.size

        data_gen = self.generator(start=start)
        i_global = start

        def array2bytes(array, format):
            array = array.transpose((1, 2, 0))
            if array.shape[2] == 1:
                array = array.reshape(array.shape[:-1])
            if not str(array.dtype).startswith('uint') and np.max(array) <= 1. and np.min(array) >= 0.:
                array *= 255.

            import cv2
            return cv2.imencode(format, array)[1]

        def format_type(r, c):
            if c in compress:
                return array2bytes(r[0, c], compress_format)
            else:
                column = self.column_by_name(c)
                d = r[c][0]
                if column.sql_type == 'INTEGER':
                    return int(d)
                elif column.sql_type == 'TEXT':
                    return str(d)
                else:
                    return d

        def format_gen(p):
            for i in range(min(n, stop-i_global)):
                try:
                    r = next(data_gen)
                except StopIteration:
                    break
                p.update(1)
                pk = format_type(r, 'pk') if include_pk else i_global+i
                yield [pk] + [format_type(r, _) for _ in columns]

        database_path_dir = dirname(database_path)
        if database_path_dir and not exists(database_path_dir):
            makedirs(database_path_dir)

        access = SQLAccessor(database_path=database_path)
        column_constraint = [self.column_by_name(_).sql_type if _ not in compress else 'BLOB'
                             for _ in (['pk'] if include_pk else []) + columns]
        access.write_db('PRAGMA journal_mode=wal')
        if replace_table:
            if show_progression:
                log.info('Dropping table %s if exists...' % table_name)
            access.write_db('DROP TABLE IF EXISTS %s' % table_name)
        access.execute_query(CreateTableQuery(if_not_exists=True,
                                              table_or_subquery=table_name,
                                              column_name=(['pk'] if include_pk else []) + columns,
                                              column_constraint=column_constraint,
                                              table_constraint="PRIMARY KEY (pk)" if include_pk else None))
        with Process('Saving %s' % self._name, stop-start) as p:
            for i in range(start, stop, n):
                access.execute_query(InsertQuery(table_or_subquery=table_name,
                                                 replace=True,
                                                 column_name=['pk' if include_pk else 'rowid'] + columns,
                                                 update_value='?' + (',?'*(len(columns)))),
                                     format_gen(p))
                i_global += n
        access.write_db('PRAGMA journal_mode=DELETE')

    def export_files(self, path, columns=None, stop=None,  start=0, filename_column=None, metadata_file='.xlsx',
                      determinist=True, ncore=1, overwrite=True):
        import pandas
        #   ---  HANDLE PARAMETERS ---

        # Create path
        os.makedirs(path, exist_ok=True)

        # Handle columns
        exported_columns = set()
        columns_mapping = OrderedDict()
        single_column = False

        if isinstance(columns, (list, tuple, set)):
            columns = OrderedDict()
            for _ in columns:
                columns[_] = _

        if isinstance(columns, str):
            if columns not in self.columns_name():
                raise ValueError('Unknown column %s' % columns)
            columns_mapping = {'': columns}
            exported_columns = {columns}
            single_column = True
        elif isinstance(columns, DSColumn):
            if columns.dataset is not self:
                raise ValueError('%s is not a column of %s' % (columns.name, self.dataset_name))
            columns_mapping = {'': columns.name}
            exported_columns = {columns.name}
            single_column = True
        elif is_dict(columns):
            for c_name, c in columns.items():
                if not isinstance(c_name, str):
                    raise ValueError('columns key should be str (received: %s)' % type(c_name).__name__)
                if isinstance(c, DSColumn):
                    if c.dataset is not self:
                        raise ValueError('%s is not a column of %s' % (c.name, self.dataset_name))
                    c = c.name
                if isinstance(c, str):
                    if c not in self.columns_name():
                        raise ValueError('Unknown column %s' % c)
                    exported_columns.add(c)
                    columns_mapping[c_name] = c
                else:
                    raise ValueError('Invalid columns value. Expected type is str or DSColumn, received %s.'
                                     % type(c).__name__)

        # Handle filename_column
        if filename_column is None:
            for n in ('name', 'filename'):
                if n in self.columns_name():
                    filename_column = n
                    break
            if filename_column is None:
                filename_column = 'pk'
        exported_columns.add(filename_column)

        # Handle metadata_file
        metadata_sheet = self.dataset_name
        if ':' in metadata_file:
            metadata_file, metadata_sheet = metadata_file.rsplit(':', 1)
        if metadata_file.startswith('.'):
            f = list(exported_columns)[0] if single_column else self.dataset_name
            metadata_file = f + metadata_file
        metadata_file = join(path, metadata_file)

        if exists(metadata_file):
            if overwrite:
                os.remove(metadata_file)
            else:
                raise RuntimeError('%s already exists.' % metadata_file)

        meta_name_column = filename_column if filename_column not in columns_mapping else None

        # Handle start stop
        start, stop = interval(size=self.size, start=start, stop=stop)

        with Process('Exporting '+self.dataset_name, total=stop-start, verbose=False) as p:
            from .dataset_generator import DataSetResult
            def write_cb(r: DataSetResult):
                metadata = OrderedDict()
                filename = r[0, filename_column]
                for c_to, c in columns_mapping.items():
                    c_data = r[0, c]
                    col = self.column_by_name(c)
                    c_path = path if single_column else join(path, c_to)

                    meta = col.format.export_file(data=c_data, path=c_path, filename=filename, overwrite=overwrite)
                    if meta is not None:
                        metadata[c_to] = meta
                if metadata:
                    meta_exists = exists(metadata_file)
                    meta_columns = [] if meta_name_column is None else [meta_name_column]
                    meta_columns += list(metadata.keys())

                    if metadata_file.endswith('xlsx'):
                        if meta_exists:
                            initial_df = pandas.read_excel(metadata_file)
                        else:
                            initial_df = pandas.DataFrame(columns=meta_columns)
                        writer = pandas.ExcelWriter(metadata_file)
                        initial_df.to_excel(writer, sheet_name=metadata_sheet, index=False)

                        if meta_name_column is not None:
                            metadata[meta_name_column] = filename
                        df = pandas.DataFrame(metadata)
                        df.to_excel(writer, sheet_name=metadata_sheet, index=False, startrow=r.start_id)
                        writer.save()

                    elif metadata_file.endswith('csv'):
                        if not meta_exists:
                            pandas.DataFrame(columns=meta_columns).to_csv(metadata_file, header=True, mode='w')

                        if meta_name_column is not None:
                            metadata[meta_name_column] = filename
                        pandas.DataFrame(metadata).to_csv(metadata_file, header=False, mode='a')

                p.update(1)

            self.export(cb=write_cb, start=start, stop=stop, columns=exported_columns, n=1,
                        determinist=determinist, ncore=ncore)

    def to_pytorch_dataset(self, ncore=1, intime='process'):
        from torch.utils.data import Dataset

        class CustomDataLoader(Dataset):
            def __init__(self, dataset):
                self._dataset = dataset
                self._gen = None

            def __len__(self):
                return self._dataset.size

            def __getitem__(self, item):
                if self._gen is None:
                    self._gen = self._dataset.generator(ncore=ncore, intime=intime)
                try:
                    return self._gen.next()
                except StopIteration:
                    raise IndexError

        return CustomDataLoader(self)

    def cache(self, start=0, stop=None, columns=None, ncore=1, ondisk=None, name=None, random_version=None,
              overwrite='auto', compression='auto'):
        from collections import OrderedDict
        from ..j_utils.path import format_filepath
        from ..j_utils.threadsafe_pytables import open_pytable, tables

        start, stop = interval(self.size, start, stop)
        if columns is None:
            columns = self.columns_name()
        elif isinstance(columns, str):
            columns = [columns]
        else:
            columns = list(columns)

        if name is None:
            label = self._name
            name = 'cache'
        else:
            label = name

        if compression == 'auto':
            if ondisk:
                compression = 5 if any(np.prod(c.shape) > 100 for c in self._columns) else 0
            else:
                compression = 0
        comp_filters = tables.Filters(complevel=compression) if compression else None

        hdf_tables = [None for _ in range(random_version if random_version else 1)]
        if ondisk:
            if isinstance(ondisk, bool):
                import tempfile
                ondisk = join(tempfile.gettempdir(), 'dataset_cache.hd5') + ':dataset'
            if isinstance(ondisk, str):
                ondisk_split = ondisk.split(':')
                if len(ondisk_split) == 1:
                    path = ondisk_split[0]
                    table_name = 'dataset'
                elif len(ondisk_split) == 2:
                    path, table_name = ondisk_split
                else:
                    raise ValueError('cache_path should be formated as "PATH:TABLE_NAME"')

                path = format_filepath(path, 'cache', exists=False)
                hdf_f = open_pytable(path)
                if not table_name.startswith('/'):
                    table_name = '/' + table_name
                table_path, table_name = table_name.rsplit('/', maxsplit=1)
                if not table_path:
                    table_path = '/'
            else:
                raise ValueError('cache_path should be formated as "PATH:TABLE_NAME"')

            if random_version:
                table_names = [table_name+'_'+str(i) for i in range(random_version)]
            else:
                table_names = [table_name]

            for i, table_name in enumerate(table_names):
                try:
                    hdf_t = hdf_f.get_node(table_path, table_name, 'Table')

                    erase_table = False
                    if overwrite is True:
                        erase_table = True
                    elif overwrite == 'auto':
                        if len(hdf_t) != self.size:
                            erase_table = True
                            #print('Wrong size', len(hdf_t), self.size)
                        else:
                            erase_table = True
                            for col_name, hdf_col in hdf_t.description._v_colobjects.items():
                                col = self.column_by_name(col_name, raise_on_unknown=False)
                                if col is None or hdf_col.dtype.shape != col.shape:
                                    #print(col, 'Mismatch shape', hdf_col.dtype.shape, col.shape)
                                    break
                                else:
                                    if col.is_text or col.dtype in ('O', object):
                                        if str(hdf_col.dtype.base)[1] != 'S':
                                            #print(col, 'Not STR', str(hdf_col.dtype.base), str(hdf_col.dtype.base)[1])
                                            break
                                    elif hdf_col.dtype.base != col.dtype:
                                        #print(col, 'Mismatch type', hdf_col.dtype.base, col.dtype)
                                        break
                            else:
                                erase_table = False
                    if erase_table:
                        hdf_f.remove_node(table_path, table_name)
                        hdf_t = None
                    hdf_tables[i] = hdf_t
                except tables.NoSuchNodeError:
                    pass
        else:
            hdf_f = None
            hdf_i = 0
            while hdf_f is None:
                try:
                    hdf_f = tables.open_file("/tmp/tmpEmptyHDF_%i.h5" % hdf_i, "a", driver="H5FD_CORE", driver_core_backing_store=0)
                except tables.HDF5ExtError:
                    hdf_i += 1

            table_path = '/'
            table_names = [str(i)+'dataset' for i in range(random_version)] if random_version else ['dataset']

        for i_table, (hdf_t, table_name) in enumerate(zip(hdf_tables, table_names)):
            if hdf_t is None:
                desc = OrderedDict()
                for i, c in enumerate(['pk']+columns):
                    col = self.column_by_name(c)
                    if col.shape == ():
                        if col.is_text or col.dtype in ('O', object):
                            desc[col.name] = tables.StringCol(1024, pos=i)
                        else:
                            desc[col.name] = tables.Col.from_dtype(col.dtype, pos=i)
                    else:
                        desc[col.name] = tables.Col.from_sctype(col.dtype.type, col.shape, pos=i)
                hdf_t = hdf_f.create_table(table_path, table_name, description=desc, expectedrows=self.size,
                                         createparents=True, track_times=False, filters=comp_filters)
                chunck_size = min(stop - start, hdf_t.chunkshape[0])

                if ncore > 1:
                    with Process('Allocating %s' % label, stop - start + 1, verbose=False) as p:
                        empty_row = hdf_t.row
                        for i in range(0, stop - start):
                            empty_row.append()
                            p.step = i
                        hdf_t.flush()

                    with Process('Caching %s' % label, stop - start, verbose=False) as p:
                        from .dataset_generator import DataSetResult
                        def write_back(r: DataSetResult):
                            hdf_t.modify_rows(start=r.start_id - start, stop=r.stop_id - start, rows=r.to_row_list())
                            p.update(r.size)

                        self.export(write_back, n=chunck_size, start=start, stop=stop, columns=columns, ncore=ncore,
                                    determinist=not random_version)
                else:
                    with Process('Caching %s' % label, stop - start, verbose=False) as p:
                        for r in self.generator(n=chunck_size, start=start, stop=stop, determinist=not random_version,
                                                columns=columns):
                            hdf_t.append(r.to_row_list())
                            p.update(r.size)
                hdf_f.flush()
                hdf_tables[i_table] = hdf_t

        from .datasets_core import PyTableDataSet, RandomVersionPyTableDataSet
        if random_version:
            hdfDataset = RandomVersionPyTableDataSet(hdf_tables, name=name, hdf_file=None if ondisk else hdf_f)
        else:
            hdfDataset = PyTableDataSet(hdf_tables[0], name=name, hdf_file=None if ondisk else hdf_f)
        for c in columns:
            hdfDataset.col[c].format = self.col[c].format
            hdfDataset.col[c]._is_text = self.col[c].is_text     # Bof...
        return hdfDataset

    #   --- Global operator ---
    def sum(self, columns=None, start=0, stop=None, ncore=1, n=1, determinist=True):
        single_column = (isinstance(columns, str) and ',' not in columns) or isinstance(columns, DSColumn)
        columns = self.interpret_columns(columns)
        for c in columns:
            c = self.column_by_name(c)
            if not np.issubdtype(c.dtype, np.number):
                raise ValueError('Only numeric columns can be summed. (%s is not numeric, dtype: %s).'
                                 % (c.name, c.dtype))

        start, stop = interval(self.size, start, stop)

        from .dataset_generator import DataSetResult
        result = DataSetResult.create_empty(n=1, dataset=self, columns=columns)

        with Process('Summing ' + self.dataset_name, total=stop - start, verbose=False) as p:
            def write_cb(r):
                p.update(r.size)
                for c in r.keys():
                    if c == 'pk':
                        continue
                    result[0, c] += r[:, c].sum(axis=0)
                result.trace.affiliate_parent_trace(r.trace)

            self.export(write_cb, columns=columns, n=n, start=start, stop=stop, ncore=ncore, determinist=determinist)

        return result[columns[0]] if single_column else result

    def mean(self, columns=None, start=0, stop=None, std=False, ncore=1, n=1, determinist=True):
        single_column = (isinstance(columns, str) and ',' not in columns) or isinstance(columns, DSColumn)
        columns = self.interpret_columns(columns)
        for c in columns:
            c = self.column_by_name(c)
            if not np.issubdtype(c.dtype, np.number):
                raise ValueError('Only numeric columns can be averaged. (%s is not numeric, dtype: %s).'
                                 % (c.name, c.dtype))

        start, stop = interval(self.size, start, stop)
        from .dataset_generator import DataSetResult
        result = DataSetResult.create_from_data({c: np.zeros((2,)+self.col[c].shape, np.float)
                                                 for c in columns})

        with Process('Averaging ' + self.dataset_name, total=stop - start, verbose=False) as p:
            if std:
                def write_cb(r):
                    p.update(r.size)
                    for c in columns:
                        result[0, c] += r[:, c].sum(axis=0) / (stop - start)
                        result[1, c] += np.square(r[:, c]).sum(axis=0) / (stop - start)
                    result.trace.affiliate_parent_trace(r.trace)
            else:
                def write_cb(r):
                    p.update(r.size)
                    for c in columns:
                        result[0, c] += r[:, c].sum(axis=0)/(stop-start)
                    result.trace.affiliate_parent_trace(r.trace)

            self.export(write_cb, columns=columns, n=n, start=start, stop=stop, ncore=ncore, determinist=determinist)

        if std:
            for c in result.keys():
                result[1, c] = np.sqrt(result[1, c]-np.square(result[0, c]))
            return result[:, columns[0]] if single_column else result
        return result[0, columns[0]] if single_column else result

    def std(self, columns=None, start=0, stop=None, ncore=1, n=1, determinist=True):
        single_column = (isinstance(columns, str) and ',' not in columns) or isinstance(columns, DSColumn)
        columns = self.interpret_columns(columns)
        result = self.mean(columns=columns, start=start, stop=stop, ncore=ncore, n=n, determinist=determinist, std=True)
        return result[1, columns[0]] if single_column else result.truncate(start=1)

    def confusion_matrix(self, pred, true, weight=None, label=None, rowwise=False,
                         start=0, stop=None, ncore=1, n=1, determinist=True):
        from ..j_utils.math import ConfMatrix

        # Handle pred, true and weight
        if isinstance(pred, DSColumn):
            if pred.dataset is not self:
                raise ValueError("%s is not a column of %s." % (pred.name, self.dataset_name))
        elif isinstance(pred, str):
            pred = self.column_by_name(pred)
        else:
            raise ValueError("Invalid type for pred. (Expected type: str or DSColumn, received: %s)" % type(pred))
        pred_shape = pred.shape

        if isinstance(true, DSColumn):
            if true.dataset is not self:
                raise ValueError("%s is not a column of %s." % (true.name, self.dataset_name))
        elif isinstance(true, str):
            true = self.column_by_name(true)
        elif isinstance(true, np.ndarray):
            pass
        else:
            raise ValueError("Invalid type for true. (Expected type: str or DSColumn, received: %s)" % type(true))
        true_shape = true.shape

        if weight is not None:
            if isinstance(weight, DSColumn):
                if weight.dataset is not self:
                    raise ValueError("%s is not a column of %s." % (weight.name, self.dataset_name))
            elif isinstance(weight, str):
                weight = self.column_by_name(weight)
            elif isinstance(weight, np.ndarray):
                pass
            else:
                raise ValueError("Invalid type for weight. (Expected type: str or DSColumn, received: %s)" % type(weight))
            weight_shape = weight.shape
        else:
            weight_shape = None

        one_hot = False
        if np.prod(pred_shape) != np.prod(true_shape):
            if np.prod(pred_shape[1:]) == np.prod(true_shape):
                one_hot = True
                pred_shape = pred.shape[1:]
            else:
                raise ValueError("Error when computing the confusion matrix of %s and %s:\n"
                                 "Shape mismatch: %s.shape=%s, %s.shape=%s"
                                 % (pred.name, true.name, pred.name, pred.shape, true.name, true.shape))
        if weight is not None and np.prod(true_shape) != np.prod(weight_shape):
            raise ValueError("Error when computing the confusion matrix of %s and %s:\n"
                             "Shape mismatch: %s.shape=%s, %s.shape=%s"
                             % (true.name, weight.name, true.name, true.shape, weight.name, weight.shape))

        start, stop = interval(self.size, start, stop)

        if label is None:
            if not self.col[pred].format.is_label or self.col[pred].format.mapping is None:
                raise ValueError('%s is not a label columns. confusion_matrix(label=...) should be specified.')
            label = self.col[pred].format.mapping
        elif isinstance(label, int):
            label = list(range(label))
        conf_labels = label
        n_class = len(conf_labels)

        confmat_name = pred.name+'_confmat' if not isinstance(rowwise, str) else rowwise
        kwargs = dict(name=confmat_name, keep_parent=True, format=DSColumnFormat.ConfMatrix(n_class), n_factor=1)

        def conf_mat(pred, true, weight):
            if one_hot:
                pred = np.argmax(pred, axis=0)
            c = ConfMatrix.confusion_matrix(y_pred=pred, y_true=true, sample_weight=weight, labels=conf_labels)
            return c

        if isinstance(true, DSColumn):
            if isinstance(weight, DSColumn):
                conf_D = self.apply({confmat_name: (pred, true, weight)}, conf_mat, **kwargs)
            else:
                conf_D = self.apply({confmat_name: (pred, true)}, **kwargs,
                                    function=lambda pred, true: conf_mat(pred, true, weight))
        else:
            if isinstance(weight, DSColumn):
                conf_D = self.apply({confmat_name: (pred, weight)}, **kwargs,
                                    function=lambda pred, weight: conf_mat(pred, true, weight))
            else:
                conf_D = self.apply({confmat_name: pred}, **kwargs,
                                    function=lambda pred: conf_mat(pred, true, weight))
        if rowwise:
            return conf_D
        else:
            confmat = ConfMatrix.zeros(labels=label)

            with Process('Confusion Matrix: %s' % label, stop - start, verbose=False) as p:
                def write_cb(r):
                    confmat[:] += r[confmat_name].sum(axis=0)
                    p.update(r.size)

                conf_D.export(write_cb, n=n, start=start, stop=stop, columns=confmat_name, determinist=determinist,
                              ncore=ncore)

            return confmat

    def roc_curve(self, pred, true, weight=None, rowwise=False, negative_label=0,
                  start=0, stop=None, ncore=1, n=1, determinist=True):
        from ..j_utils.math import ROCCurve

        # Handle pred, true and weight
        if isinstance(pred, DSColumn):
            if pred.dataset is not self:
                raise ValueError("%s is not a column of %s." % (pred.name, self.dataset_name))
        elif isinstance(pred, str):
            pred = self.column_by_name(pred)
        else:
            raise ValueError("Invalid type for pred. (Expected type: str or DSColumn, received: %s)" % type(pred))
        pred_shape = pred.shape

        if isinstance(true, DSColumn):
            if true.dataset is not self:
                raise ValueError("%s is not a column of %s." % (true.name, self.dataset_name))
        elif isinstance(true, str):
            true = self.column_by_name(true)
        elif isinstance(true, np.ndarray):
            pass
        else:
            raise ValueError("Invalid type for true. (Expected type: str or DSColumn, received: %s)" % type(true))
        true_shape = true.shape

        if weight is not None:
            if isinstance(weight, DSColumn):
                if weight.dataset is not self:
                    raise ValueError("%s is not a column of %s." % (weight.name, self.dataset_name))
            elif isinstance(weight, str):
                weight = self.column_by_name(weight)
            elif isinstance(weight, np.ndarray):
                pass
            else:
                raise ValueError(
                    "Invalid type for weight. (Expected type: str or DSColumn, received: %s)" % type(weight))
            weight_shape = weight.shape
        else:
            weight_shape = None

        if weight is not None and np.prod(true_shape) != np.prod(weight_shape):
            raise ValueError("Error when computing the confusion matrix of %s and %s:\n"
                             "Shape mismatch: %s.shape=%s, %s.shape=%s"
                             % (true.name, weight.name, true.name, true.shape, weight.name, weight.shape))

        start, stop = interval(self.size, start, stop)

        if rowwise:
            roc_name = pred.name + '_roc' if not isinstance(rowwise, str) else rowwise
            kwargs = dict(name=roc_name, keep_parent=True, format=DSColumnFormat.ROCCurve(), n_factor=1)

            def roc_curve(pred, true, weight):
                return ROCCurve.roc_curve(score=pred.flatten(), true=true.flatten(), sample_weight=weight.flatten(),
                                          negative_label=negative_label)

            if isinstance(true, DSColumn):
                if isinstance(weight, DSColumn):
                    roc_D = self.apply({roc_name: (pred, true, weight)}, roc_curve, **kwargs)
                else:
                    roc_D = self.apply({roc_name: (pred, true)}, **kwargs,
                                        function=lambda pred, true: roc_curve(pred, true, weight))
            else:
                if isinstance(weight, DSColumn):
                    roc_D = self.apply({roc_name: (pred, weight)}, **kwargs,
                                        function=lambda pred, weight: roc_curve(pred, true, weight))
                else:
                    roc_D = self.apply({roc_name: pred}, **kwargs,
                                        function=lambda pred: roc_curve(pred, true, weight))
            return roc_D
        else:
            if isinstance(true, DSColumn):
                if isinstance(weight, DSColumn):
                    r = self.read(start=start, stop=stop, columns=(pred, true, weight))
                    pred, true, weight = r[pred], r[true], r[weight]
                else:
                    r = self.read(start=start, stop=stop, columns=(pred, true))
                    pred, true = r[pred], r[true]
            else:
                if isinstance(weight, DSColumn):
                    r = self.read(start=start, stop=stop, columns=(pred, weight))
                    pred, weight = r[pred], r[weight]
                else:
                    r = self.read(start=start, stop=stop, columns=pred)
                    pred = r[pred]

            return ROCCurve.roc_curve(score=pred, true=true, sample_weight=weight, negative_label=negative_label)

    #   ---   Operations   ---
    @classmethod
    def operation(cls, func):
        if hasattr(cls, func.__name__):
            raise AttributeError('%s method name already exist in AbstractDataset.' % func.__name__)
        setattr(cls, func.__name__, func)

    def subset(self, *args, start=0, stop=None, name='subset'):
        from .datasets_core import DataSetSubset
        start, stop = interval(self.size, start, stop, args)
        return DataSetSubset(self, start, stop, name=name)

    def subgen(self, n=1, name='subgen'):
        from .datasets_core import DataSetSubgen
        return DataSetSubgen(self, n=n, name=name)

    def split_sets(self, *args, **kwargs):
        """Split sets into a dictionary named after the parameters keys.

        :Example:

        >>> dict_of_sets = dataset.split_sets(train=-1, validation=1000)

        In this example, a **train** key and a **validation** key are created in the dictionary.
        The validation key is associated with 1000 instances. The training key is associated with the rest of the dataset.


        :param kwargs: pair of key:value, where value are whether a ratio of usage (range between 0 and 1) \
         or a number of instances of the dataset to use for the associated key. \
        If the value associated to the key is -1, then the associated dataset will be filled until completion \
        of the dataset. If many keys has a -1 value, then the free instances (not associated with any other keys) \
        of the dataset are equally shared between them.
        :return A dictionary of set, each set being defined by its ratio of usage.
        :rtype: dict
        """

        d = {}
        offset = 0
        cummulative_ratio = 0
        eq_ratio = []

        for i, arg in enumerate(args):
            kwargs[i] = arg

        for name, ratio in kwargs.items():
            if ratio != -1:
                if 0 < ratio < 1:
                    l = round(self.size * ratio)
                elif ratio >= 1:
                    l = round(ratio)
                    ratio = l / self.size
                else:
                    raise NotImplementedError('Datasets ratio must be a positive number')
                d[name] = self.subset(offset, offset+l, name='split%i'%name if isinstance(name, int) else name)
                offset += l
                cummulative_ratio += ratio
                if cummulative_ratio > 1:
                    raise ValueError('Total of amount given to split database is larger than the database...')
            else:
                eq_ratio.append(name)

        if eq_ratio:
            ratio = (1-cummulative_ratio)/len(eq_ratio)
            for name in eq_ratio:
                l = round(self.size * ratio)
                d[name] = self.subset(offset, offset + l, name='split%i'%name if isinstance(name, int) else name)
                offset += l

        r_args = [d.pop(i) for i in range(len(args))]
        if r_args:
            if d:
                return (*r_args, d)
            return tuple(r_args)
        return d

    def repeat(self, n=None, rows=None, name="repeat"):
        if rows is not None:
            if rows == self.size:
                return self
            elif rows < self.size:
                return self.subset(stop=rows, name=name)
            elif rows > self.size:
                l = [self] * (rows//self.size)
                l.append(self.subset(rows % self.size))
                from .datasets_core import DataSetConcatenate
                return DataSetConcatenate(l, name=name)
        elif n is not None:
            if n == 1:
                return self
            elif n < 1:
                return self.subset(stop=int(self.size*n), name=name)
            elif n > 1:
                l = [self] * int(np.floor(n))
                l.append(self.subset(n % 1))
                from .datasets_core import DataSetConcatenate
                return DataSetConcatenate(l, name=name)

    def concat(self, **kwargs):
        """"Map columns name or concatenate two columns according to kwargs (keys are created columns,
        and values are either names of column to map or lists of column names to concatenate).

        If a column is not mentioned kwargs values, it is kept with the same name (unlike ```map()```).
        """
        from .datasets_core import DataSetMap
        return DataSetMap(self, kwargs, keep_all_columns=True)

    def map(self, *args, **kwargs):
        """"Map columns name or concatenate two columns according to kwargs (keys are created columns,
        and values are either names of column to map or lists of column names to concatenate).

        If a column is not mentioned kwargs values, it is discarded (unlike ```concat()```).
        To keep a column, give its name as a positional argument.
        """
        from .datasets_core import DataSetMap
        for a in args:
            a = self.interpret_columns(a)
            for _ in a:
                kwargs[_] = _
        return DataSetMap(self, kwargs, keep_all_columns=False)

    def shuffle(self, indices=None, subgen=0, rng=None, name='shuffle'):
        from .datasets_core import DataSetShuffle
        return DataSetShuffle(dataset=self, subgen=subgen, indices=indices, rng=rng, name=name)

    def apply(self, columns, function, format=None, n_factor='auto', batchwise=False, keep_parent=None,
              name=None):
        if name is None:
            name = getattr(function, '__name__', 'apply')
            if name == '<lambda>':
                name = "apply"
        if n_factor == 'auto':
            n_factor = 1 if format is not None else None

        from .datasets_core import DataSetApply
        remove_parent_columns = None if keep_parent is None else not keep_parent
        return DataSetApply(self, function=function, columns=columns, name=name, format=format, n_factor=n_factor,
                            remove_parent_columns=remove_parent_columns, batchwise=batchwise)

    def apply_cv(self, columns, function, format=None, converted_cols=None, force_mono=False,
                 n_factor=1, keep_parent=None, name=None):
        if name is None:
            name = getattr(function, '__name__', 'apply')
            if name == '<lambda>':
                name = "apply"
        from .datasets_core import DataSetApplyCV
        remove_parent_columns = None if keep_parent is None else not keep_parent
        return DataSetApplyCV(self, function=function, columns=columns, name=name, n_factor=n_factor,
                              converted_cols=converted_cols, force_mono=force_mono,
                              format=format, remove_parent_columns=remove_parent_columns)

    def apply_torch(self, columns, f, format=None, device=None, eval=True, forward_hooks=None, backward_hooks=None,
                    requires_grad=False,
                    n_factor=1, batchwise=True, keep_parent=None, name=None):
        import torch
        from .datasets_core import DataSetApply
        if name is None:
            name = 'torch'

        if not requires_grad:
            requires_grad = []

        def before_apply(**kwargs):
            kwargs = {k: torch.tensor(np.array(v) * 1,
                  requires_grad=requires_grad if not isinstance(requires_grad, (list, tuple)) else k in requires_grad
                                      ) for k, v in kwargs.items()}
            if device:
                return {k: v.to(device) for k, v in kwargs.items()}
            else:
                return kwargs

        def after_apply(r):
            if isinstance(r, torch.Size):
                return np.array(r)
            if isinstance(r, torch.Tensor):
                return r.detach().cpu().numpy()
            elif isinstance(r, tuple):
                return tuple(after_apply(_) for _ in r)
            elif isinstance(r, list):
                return list(after_apply(_) for _ in r)

            return r

        if isinstance(f, torch.nn.Module):
            net = f

            if (forward_hooks or backward_hooks) and (isinstance(columns, (dict, list)) and len(columns) > 1):
                raise ValueError('Forward and backward hooks are not supported with multiple source columns.')

            source_cols = None
            if isinstance(columns, dict):
                k, source_cols = next(columns.items())
                if isinstance(k, str):
                    cols_name = [k]
                else:
                    cols_name = list(k)
            elif isinstance(columns, str):
                cols_name = [columns]
            else:
                cols_name = list(columns)

            f_hooks = []
            if forward_hooks:
                for cols, module in forward_hooks.items():
                    if isinstance(cols, str):
                        cols = (cols,)
                    else:
                        cols = tuple(cols)
                    if not isinstance(module, torch.nn.Module):
                        raise ValueError('Wrong forward_hooks arguments: %s is not a torch module.' % repr(module))
                    f_hooks.append((module, cols))
                    cols_name += list(cols)

            b_hooks = []
            if backward_hooks:
                for cols, module in backward_hooks.items():
                    if isinstance(cols, str):
                        cols = (cols,)
                    else:
                        cols = tuple(cols)
                    if not isinstance(module, torch.nn.Module):
                        raise ValueError('Wrong forward_hooks arguments: %s is not a torch module.' % repr(module))
                    b_hooks.append((module, cols))
                    cols_name += list(cols)

            def f(x):
                # Create hooks
                hooks_handle = []
                tensor_store = [None for _ in range(len(f_hooks) + len(b_hooks))]

                def hook(i, forward):
                    if forward:
                        def _hook(module, input, output):
                            tensor_store[i] = output
                    else:
                        def _hook(module, grad_input, grad_output):
                            tensor_store[i] = grad_input
                    return _hook

                i = 0
                for module, cols in f_hooks:
                    hooks_handle.append(module.register_forward_hook(hook(i, forward=True)))
                    i += 1

                for module, cols in b_hooks:
                    hooks_handle.append(module.register_backward_hook(hook(i, forward=False)))
                    i += 1

                # Process data
                reset_to_train = False
                if eval and getattr(net, 'training', False):
                    reset_to_train = True
                    net.eval()

                y = net(x)

                if reset_to_train:
                    net.train()

                #
                if isinstance(y, tuple):
                    r = list(y)
                else:
                    r = [y]

                for handle, tensors, (module, cols) in zip(hooks_handle, tensor_store, f_hooks+b_hooks):
                    handle.remove()
                    if tensors is None:
                        r.append(None)
                    else:
                        if not isinstance(tensors, tuple):
                            tensors = (tensors,)
                        if len(tensors) != len(cols):
                            raise ValueError('The torch module hooked to columns %s returned %i tensors (%i was expected)'
                                             % (repr(cols), len(tensors), len(cols)))
                        r += list(tensors)

                return tuple(r)

            columns = {tuple(cols_name): source_cols} if source_cols else tuple(cols_name)

        else:
            if backward_hooks or forward_hooks:
                raise ValueError('Forward and backward hooks are not supported when f is not a torch module.')

        remove_parent_columns = None if keep_parent is None else not keep_parent

        return DataSetApply(dataset=self, function=f, columns=columns, n_factor=n_factor, batchwise=batchwise,
                            remove_parent_columns=remove_parent_columns, name=name, format=format,
                            before_apply=before_apply, after_apply=after_apply)

    def map_values(self, columns, mapping, default=None, sampling=None, name='map_value'):
        from .datasets_core import DataSetApply
        from ..j_utils.image import prepare_lut
        if default is None:
            default = mapping.pop('default', None)
        f_lut = prepare_lut(mapping, default=default, sampling=sampling)
        return DataSetApply(self, function=f_lut, columns=columns, name=name,
                            format=None, batchwise=True, n_factor=1)

    def scale_values(self, columns, range=(0,1), domain=(0,1), clip=None, name='scale_value'):
        from .datasets_core import DataSetApply
        from ..j_utils.math import apply_scale
        def f_apply_scale(x):
            return apply_scale(x, range=range, domain=domain, clip=clip)
        return DataSetApply(self, function=f_apply_scale, columns=columns, name=name,
                            format='same', batchwise=True, n_factor=1)

    def as_label(self, columns, mapping, format=None, sampling=None, name="label"):
        f_mapping = None
        infered_format = {}
        if isinstance(mapping, dict):
            from ..j_utils.image import prepare_lut
            map = {}
            for k, v in mapping.items():
                k = np.uint32(k)
                if isinstance(v, (tuple, list)):
                    for _ in v:
                        map[_] = k
                    infered_format[k] = v[0]
                else:
                    map[v] = k
                    infered_format[k] = v
            f_mapping = prepare_lut(map=map, sampling=sampling)
        elif callable(mapping):
            f_mapping = mapping
        else:
            raise NotImplementedError
        dataset = self.apply(columns, f_mapping, name=name)

        if isinstance(format, int):
            infered_format = (0, format)
        elif isinstance(format, dict) or (isinstance(format, tuple) and len(format)==2):
            infered_format = format
        for c in dataset._single_col_mapping.keys():
            col = dataset.column_by_name(c)
            col.format = infered_format

        return dataset

    def reshape(self, columns, shape, keep_parent=False, name='reshape'):
        from .datasets_core2d import DataSetReshape
        cols_shape = OrderedDict()
        if is_dict(shape):
            columns = self.interpret_columns(columns)
            if len(columns) != 1:
                raise ValueError('When shape is a dictionary of shape, columns should be the name of a specific column')
            columns = columns[0]
            for c, s in shape.items():
                cols_shape[c] = (columns, s)
        elif is_dict(columns):
            for new_c, parent_c in columns.items():
                parent_c = self.interpret_columns(parent_c)
                if len(parent_c) != 1:
                    raise ValueError('Multiple parent for a single column is not supported.')
                parent_c = parent_c[0]
                new_c = self.interpret_columns(new_c, exists=False)
                for c in new_c:
                    cols_shape[c] = (parent_c, shape)
        else:
            columns = self.interpret_columns(columns)
            suffix = '_reshaped' if keep_parent else ''
            for c in columns:
                cols_shape[c+suffix] = (c, shape)
        return DataSetReshape(self, cols_shape=cols_shape, keep_parent=keep_parent, name=name)

    def augment(self, data_augment, columns=None, N=1, original=False, name='augment'):
        from .datasets_augment import DataSetAugment
        if columns is None:
            columns = [col.name for col in self.columns if col.ndim >= 2]
        return DataSetAugment(dataset=self, data_augment=data_augment, columns=columns, N=N, original=original,
                              name=name)

    def patches(self, columns=None, patch_shape=(128, 128), stride=(0, 0), ignore_borders=True, mask=None,
                center_pos=False, name='patch'):
        """Generate patches from specific columns.

            :param columns: Columns from which patches will be extracted.
            :type columns: [str, list]
            :param patch_shape: Shape of the patches extracted.
                                If columns is None, patches should be a dictionary where the keys are the new column names,
                                 and the values are a tuple of the form (patch_shape) or (parent_column_name, (patch_shape)).
            :type patch_shape: [tuple, dict]
            :param stride:
            :type stride: [tuple, int, float]
            :param ignore_borders:
            :type ignore_borders: bool
            :param mask:
            :type mask: np.ndarray
            :rtype: DataSetPatches
        """
        from ..j_utils.image import compute_regular_patch_centers

        # Initialize patches
        patches_def = {}
        if isinstance(patch_shape, dict) and columns is not None:
            raise ValueError("If columns is defined, patch_shape can't be a dictionary of patch shapes.")
        if columns is not None:
            columns = self.interpret_columns(columns)
            if not isinstance(patch_shape, tuple):
                patch_shape = (patch_shape, patch_shape)
            for c in columns:
                patches_def[c] = (c, patch_shape)
        else:
            if not isinstance(patch_shape, dict):
                raise ValueError('If columns is not defined, patch_shape should be a dictionary of patch shapes.')
            for c, patch_def in patch_shape.items():
                if isinstance(c, DSColumn):
                    c = c.name
                parent_c = c
                if isinstance(patch_def, tuple) and not isinstance(patch_def[0], int):
                    parent_c = patch_def[0].name if isinstance(patch_def[0], DSColumn) else patch_def[0]
                    patch_def = patch_def[1]
                if isinstance(patch_def, int):
                    patch_def = (patch_def, patch_def)
                patches_def[c] = (parent_c, patch_def)

        # Reading img_shape
        img_shape = self.column_by_name(list(patches_def.values())[0][0]).shape[-2:]

        # Initialize stride
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        stride = list(stride)
        for i in range(2):
            if stride[i] <= 0:
                if isinstance(patch_shape, dict):
                    min_s = None
                    for patch_def in patch_shape.values():
                        if isinstance(patch_def, tuple) and len(patch_def)==2 and isinstance(patch_def[0], str):
                            n, s = patch_def
                        else:
                            s = patch_def
                        if isinstance(s, tuple) and len(s) == 2:
                            s = s[i]
                        min_s = s if min_s is None or min_s > s else min_s
                    stride[i] = min_s - stride[i]
                else:
                    stride[i] = patch_shape[i] - stride[i]
                if stride[i] <= 0:
                    raise ValueError('Invalid stride (when negative, stride should not be greater than patch_shape...)')
            elif stride[i] < 1:
                stride[i] *= img_shape[i]
        stride = tuple(stride)

        # Initialize patches_function
        patchify = partial(compute_regular_patch_centers, ignore_borders=ignore_borders, mask=mask, stride=stride)

        from .datasets_core2d import DataSetPatches
        d = DataSetPatches(self, patch_shape=patches_def, patches_function=patchify,
                           center_pos=center_pos, name=name)
        d.stride = stride
        d.ignore_borders = ignore_borders
        return d

    def random_patches(self, columns=None, patch_shape=(128, 128), n=10, proba_map=None, rng=None,
                       center_pos=False, name='randomPatch'):
        """Generate patches randomly from specific columns.

        Generate ``n`` patches from each images of ``columns``. Patches centers are chosen using a probability map.

        :param columns: Columns from which patches will be extracted.
        :type columns: [str, list]
        :param patch_shape: Shape of the patches extracted.
                                If columns is None, patches should be a dictionary where the keys are the new column names,
                                 and the values are a tuple of the form (patch_shape) or (parent_column_name, (patch_shape)).
            :type patch_shape: [tuple, dict]
        :param n: Number of patches extracted from 1 image.
        :type n: int
        :param proba_map: This parameters can either be a probabilty map or a function which generates such map.
                            - Probability map: Its shape should be the same as ``columns`` shape.\
                              Each pixel of the map give its probability to being selected as a patch's center.
                            - Function: the function can have all parameters passed to DataSetPatches.patches_function.\
                                       Such function should return a probability map.
                          By default ``proba_map`` is an uniform map
        :param rng: The random state used to select centers.
        :type rng: np.random.RandomState
        :param determinist: If true, ``rng`` is reset after each patches selection, thus the selection is determinist.
        :type determinist: bool
        :rtype: DataSetPatches
        """
        from ..j_utils.image import compute_proba_patch_centers

        if rng is None:
            rng = np.random.RandomState(1234)

        # Initialize patches definition
        patches_def = {}
        if isinstance(patch_shape, dict) and columns is not None:
            raise ValueError("If columns is defined, patch_shape can't be a dictionary of patch shapes.")
        if columns is not None:
            columns = self.interpret_columns(columns)
            if not isinstance(patch_shape, tuple):
                patch_shape = (patch_shape, patch_shape)

            for c_id, c in enumerate(columns):
                patches_def[c] = (c, patch_shape)
        else:
            if not isinstance(patch_shape, dict):
                raise ValueError('If columns is not defined, patch_shape should be a dictionary of patch shapes.')
            for c, patch_def in patch_shape.items():
                if isinstance(c, DSColumn):
                    c = c.name
                parent_c = c
                if isinstance(patch_def, tuple) and not isinstance(patch_def[0], int):
                    parent_c = patch_def[0].name if isinstance(patch_def[0], DSColumn) else patch_def[0]
                    patch_def = patch_def[1]
                if isinstance(patch_def, int):
                    patch_def = (patch_def, patch_def)
                patches_def[c] = (parent_c, patch_def)

        # Reading img_shape
        img_shape = self.column_by_name(list(patches_def.values())[0][0]).shape[-2:]
        h, w = img_shape
        if proba_map is None:
            proba_map = np.ones(img_shape)/np.prod(img_shape)

        patchify = partial(compute_proba_patch_centers, n=n, rng=rng)

        if isinstance(proba_map, np.ndarray):
            if proba_map.shape != img_shape:
                raise ValueError('Wrong shape for probability map (map shape is %s but should be %s)'
                                 % (proba_map.shape, img_shape))
            proba_map = (proba_map*1.0) / proba_map.sum()
            proba_map = to_callable(proba_map)
        elif isinstance(proba_map, (str, DSColumn)):
            if isinstance(proba_map, DSColumn):
                if proba_map.dataset is not self:
                    raise ValueError("%s is not a column of %s." % (proba_map.name, self.dataset_name))
                proba_map = proba_map.name
            else:
                self.column_by_name(proba_map)
        elif not callable(proba_map):
            raise NotImplementedError

        from .datasets_core2d import DataSetPatches
        return DataSetPatches(self, patch_shape=patches_def, n=n, center_pos=center_pos, name=name,
                              patches_function=proba_map, post_process=patchify)

    def unpatch(self, columns=None, patch_mix='replace', restore_columns=None, columns_shape=None, n_patches=None,
                stride_factor=None):
        from .datasets_core2d import DataSetUnPatch
        return DataSetUnPatch(self, patch_mix=patch_mix, columns=columns, n_patches=n_patches,
                              stride_factor=stride_factor,
                              restore_columns=restore_columns, columns_shape=columns_shape)


########################################################################################################################
class DSColumn:
    """
    Store information of a column of a DataSet
    """
    def __init__(self, name, shape, dtype, dataset=None, format=None):
        """

        :param name:
        :param shape:
        :param dtype:
        :param dataset:
        :param format:
        Those dimensions are always the first ones in the list
        """
        self._name = name
        self._shape = shape
        self._is_text = 'U' in str(dtype)
        if dtype == str or (isinstance(dtype, str) and dtype in ('str', 'string')):
            self._dtype = np.dtype('O')
            self._is_text = True
        else:
            self._dtype = np.dtype(dtype)

        self._format = None
        self._dataset = None if dataset is None else weakref.ref(dataset)
        self.format = format

    def __getstate__(self):
        return self._name, self._shape, self._dtype, self._is_text

    def __setstate__(self, state):
        self._name, self._shape, self._dtype, self._is_text = state
        self._dataset = None
        self._format = None

    @property
    def shape(self):
        """
        Shape of the column, not including the varying dimensions size
        :rtype: tuple 
        """
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def dtype(self):
        """
        Type of the data in the column
        :rtype: type
        """
        return self._dtype

    @property
    def sql_type(self):
        # TODO: Remove this property
        if self.shape != ():
            return 'array'
        if 'int' in str(self.dtype):
            return 'INTEGER'
        elif 'float' in str(self.dtype):
            return 'FLOAT'
        elif self._is_text or str(self.dtype) in ('str', 'string', 'U'):
            return 'TEXT'
        return 'UNKNOWN'

    @property
    def name(self):
        """
        Name of the column
        :rtype: str
        """
        return self._name

    @property
    def dataset(self):
        """
        Parent dataset
        :rtype: AbstractDataSet
        """
        if self._dataset is not None:
            return self._dataset()
        return None

    @property
    def is_text(self):
        return self._is_text or isinstance(self.format, DSColumnFormat.Text)

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, f):
        if self._dtype is None or self._shape is None:
            self._format = f
        else:
            self._format = DSColumnFormat.auto_format(self.dtype, self.shape, f)

    def __repr__(self):
        return '%s: %s %s' % (self.name, str(self.shape), str(self.dtype))

    def __str__(self):
        return self.name

    def prepare_hdf5_col(self, table):
        f = repr(self.format)
        if self.is_text:
            f += ' TEXT'
        setattr(table.attrs, 'COLFORMAT_'+self.name, f)

    @staticmethod
    def from_hdf5_col(table, name, dataset=None):
        col = table.description._v_colobjects[name]
        column = DSColumn(name=name, dtype=col.dtype.base, shape=col.dtype.shape, dataset=dataset)

        f = getattr(table.attrs, 'COLFORMAT_'+name, None)
        if f is not None:
            f = f.split(' ')
            args = f[1:]
            f = f[0]
            if 'TEXT' in args:
                column._is_text = True
            column.format = f
        return column


class DSColumnFormat:
    class Base:
        def __init__(self, dtype, shape, is_label=False):
            self.__dtype = dtype
            self.__shape = shape
            self._is_label = is_label
            self.check_type(dtype, shape)
            self.html_fullscreen = False

        def __repr__(self):
            return "Base()"

        def check_type(self, dtype, shape):
            return True

        def copy(self, dtype, shape):
            from copy import deepcopy
            self.check_type(dtype, shape)
            f = deepcopy(self)
            f._dtype = dtype
            f._shape = shape
            return f

        @property
        def dtype(self):
            return self.__dtype

        @property
        def dtype_name(self):
            if self.shape != ():
                return 'array(%s)' % str(self.dtype)
            if np.issubdtype(self.dtype, np.integer):
                return 'INTEGER'
            elif np.issubdtype(self.dtype, np.inexact):
                return 'FLOAT'
            elif str(self.dtype) in ('str', 'U') or np.issubdtype(self.dtype, np.character):
                return 'TEXT'
            return 'UNKNOWN'

        @property
        def shape(self):
            return self.__shape

        @property
        def format_name(self):
            return self.__class__.__name__

        @property
        def is_label(self):
            return self._is_label

        def preformat(self, data):
            preformat = None
            for C in reversed(type(self).mro()):
                p = getattr(C, '_preformat', None)
                if p and p is not preformat:
                    preformat = p
                    data = preformat(self, data)
            return data

        def _preformat(self, data):
            return data

        def format_html(self, data, raw_data, fullscreen=None):
            return "<p> %s </p>" % str(data)

        def format_data(self, data):
            return data

        def write_file(self, data, path, filename, overwrite=True):
            return self.format_data(data)

        def export_html(self, data, fullscreen=None):
            html = self.format_html(self.preformat(data), data, fullscreen=fullscreen)
            return html

        def export_data(self, data):
            return self.format_data(self.preformat(data))

        def export_file(self, data, path, filename, overwrite=True):
            return self.write_file(self.preformat(data), path, filename, overwrite)

    class Number(Base):
        def __init__(self, dtype=np.float, shape=()):
            super(DSColumnFormat.Number, self).__init__(dtype, shape)

        def __repr__(self):
            return "Number()"

        def check_type(self, dtype, shape):
            if not np.issubdtype(dtype, np.number):
                raise ValueError('Number format must be applied to number columns.')
            if len(shape):
                raise ValueError('Number format can only be applied to columns with an empty shape.')
            return True

        def export_html(self, data, fullscreen=None):
            return ("<p>%i</p>" if 'int' in str(self.dtype) else "<p>%.3f</p>") % data

    class Text(Base):
        def __init__(self, dtype='O', shape=()):
            super(DSColumnFormat.Text, self).__init__(dtype, shape)

        def __repr__(self):
            if 'U' in str(self.dtype):
                return "Text(%i)" % np.dtype(self.dtype).itemsize
            return "Text()"

        @property
        def dtype_name(self):
            return "TEXT"

        def check_type(self, dtype, shape):
            if len(shape):
                raise ValueError('String format can only be applied to columns with an empty shape.')
            return True

    class Label(Base):
        def __init__(self, dtype, shape, mapping=None, default=None):
            super(DSColumnFormat.Label, self).__init__(dtype, shape, is_label=True)
            self.mapping = if_none(mapping, dict())
            self.default = if_none(default, mapping['default'] if 'default' in mapping else None)

        def __repr__(self):
            from json import dumps
            from copy import copy
            m = copy(self.mapping)
            m['default'] = self.default
            json_mapping = dumps(m, indent=0).replace('\n', '')
            return 'Label(%s)' % json_mapping

        def format_html(self, data, raw_data, fullscreen=None):
            try:
                d = self.mapping[data]
            except KeyError:
                d = self.default
            return "<p> <i> %s </i> </p>" % str(d)

    class Matrix(Base):
        def __init__(self, dtype, shape, is_label=False):
            super(DSColumnFormat.Matrix, self).__init__(dtype, shape, is_label=is_label)
            self._domain = Interval()
            self._range = Interval()
            self._clip = Interval()
            self.no_scaling = False

        def __repr__(self):
            return 'Matrix(%s,%s,%s)' % (self.domain, self.range, self.clip)

        def check_type(self, dtype, shape):
            if not (np.issubdtype(dtype, np.number) or dtype == np.bool):
                raise ValueError('Matrix format must be applied to number columns (not %s).' % repr(dtype))
            if not len(shape):
                raise ValueError('Matrix format can only be applied to columns with a non empty shape.')
            return True

        def _preformat(self, data):
            if self.no_scaling:
                return data
            return apply_scale(data, self.range, self.domain, self.clip)

        def format_data(self, data):
            return str(data)

        def write_file(self, data, path, filename, overwrite=True):
            if not overwrite and exists(join(path, filename+'.npy')):
                return None
            np.save(join(path, filename+'.npy'), data)

        @property
        def domain(self):
            return self._domain

        @domain.setter
        def domain(self, d):
            self._domain = Interval(d)

        @property
        def range(self):
            return self._range

        @range.setter
        def range(self, d):
            self._range = Interval(d)

        @property
        def clip(self):
            return self._clip

        @clip.setter
        def clip(self, d):
            self._clip = Interval(d)

    class LabelMatrix(Matrix):
        def __init__(self, dtype, shape, mapping=None, default=None):
            super(DSColumnFormat.LabelMatrix, self).__init__(dtype, shape, is_label=True)
            self.default = default
            self._lut = None
            self.mapping = if_none(mapping, dict())

        def _preformat(self, data):
            if len(self.shape) == 2:
                data = data[0]
            if self._lut is not None:
                data = self._lut(data)
            if data.ndim == 2:
                return data.reshape((1,)+data.shape)
            return data

        def __repr__(self):
            from json import dumps
            from copy import copy
            m = copy(self.mapping)
            m['default'] = self.default
            json_mapping = dumps(m, indent=0).replace('\n', '')
            return 'LabelMatrix(%s,%s,%s,%s)' % (json_mapping, self.domain, self.range, self.clip)

        @property
        def mapping(self):
            return self._mapping

        @mapping.setter
        def mapping(self, m):
            if not m:
                self._mapping = m
                self._lut = None
                self.no_scaling = False
                return

            mapping = {}
            for k, v in m.items():
                if isinstance(v, str):
                    from ..j_utils.image import str2color
                    v = str2color(v, bgr=True, uint8=True)
                if k == "default":
                    self.default = v
                else:
                    if len(self.shape) == 3 and np.array(k).ndim == 0:
                        k = (k,)
                    mapping[k] = v
            self._mapping = mapping

            from ..j_utils.image import prepare_lut
            self._lut = prepare_lut(mapping, source_dtype=self.dtype, default=self.default)
            self.no_scaling = True

    class ConfMatrix(Base):
        def __init__(self, shape, dtype='int64'):
            if isinstance(shape, int):
                self.n_class = shape
                shape = (shape, shape)
            else:
                self.n_class = shape[-1]
            super(DSColumnFormat.ConfMatrix, self).__init__(dtype, shape)

        def __repr__(self):
            return 'ConfMatrix()'

        def format_html(self, data, raw_data, fullscreen=None):
            table_tmp = """
            <table style="font-size: 10px;
                          text-align: center;
                          margin: auto;
                          border-collapse: separate;
                          border-spacing: 2px 2px;"> 
                {} 
            </table>
            """
            table = ""
            s = data.sum()

            def color_scale(f, rgb):
                r,g,b = rgb
                return (255 - f * (255 - r),
                        255 - f * (255 - g),
                        255 - f * (255 - b))

            for i, row in enumerate(data):
                row_html = ""
                for j, c in enumerate(row):
                    f = c/s
                    r,g,b = color_scale(f, (163,209,76) if i==j else (179,39,39))
                    row_html += """
                    <td style="padding: 5px 10px 1px 10px;
                               background-color: rgb(%i,%i,%i)">%i</td>
                    """ % (r, g, b, c)

                table += '<tr style="border: none;">' + row_html + "</tr>"

            return table_tmp.format(table)

    class ROCCurve(Base):
        def __init__(self, shape, dtype='int64'):
            super(DSColumnFormat.ROCCurve, self).__init__(dtype, shape)

        def __repr__(self):
            return 'ROCCurve()'

    class Image(Matrix):
        def __init__(self, dtype, shape, is_label=False):
            from ..j_utils.math import dimensional_split
            super(DSColumnFormat.Image, self).__init__(dtype, shape, is_label=is_label)
            self.html_fullscreen = True
            self.clip = 0, 255
            self.range = 0, 255
            if dtype == np.uint8:
                self.domain = 0, 255
            elif 'float' in str(dtype):
                self.domain = 0, 1.0
            else:
                self.domain = None, None

            self.html_height = lambda h: int(np.round(256*(1-np.exp(-h/128))))
            self.html_columns = lambda n: dimensional_split(n)[1]

        def __repr__(self):
            return 'Image()'

        def check_type(self, dtype, shape):
            if not (np.issubdtype(dtype, np.number) or str(dtype) == 'bool'):
                raise ValueError('Image format must be applied to number columns (not %s).' % repr(dtype))
            if len(shape) not in (2, 3):
                raise ValueError('Image format can only be applied to columns with a non empty shape.')

        @property
        def channels_count(self):
            return int(np.prod(self.shape[:-2]))

        def _preformat(self, data):
            if data.ndim > 3:
                return data.reshape((np.prod(data.shape[:-2]),)+data.shape[-2:])
            if data.ndim == 2:
                return data.reshape((1,)+data.shape)
            return data

        def format_html(self, data, raw_data, fullscreen=None):
            THUMBNAIL_SIZE = (128, 128)
            MAX_FULL_SIZE = (1024, 1024)

            import cv2
            c, h, w = data.shape
            ratio = h / w

            if fullscreen is None:
                if c % 3 == 0:
                    gen_channels = tuple(data[_:_+3].transpose((1, 2, 0)) for _ in range(0, c, 3))
                else:
                    gen_channels = tuple(data[_] for _ in range(0, c))

                n = len(gen_channels)
                nw = self.html_columns(n) if callable(self.html_columns) else self.html_columns
                nh = int(np.ceil(n/nw))

                d = []
                for channel_data in gen_channels:
                    html_height = self.html_height(h) if callable(self.html_height) else self.html_height
                    th = html_height//nh
                    tw = int(np.round(th/ratio))
                    thumbnail = cv2.resize(channel_data, (th, tw), interpolation=cv2.INTER_AREA)
                    png = str(base64.b64encode(cv2.imencode('.png', thumbnail)[1]))[2:-1]
                    html = '<img src="data:image/png;base64, %s" style="height: %ipx; min-width: %ipx" />' % (png, th, tw)
                    d.append(html)

                return '#%i,%i|' % (nh, nw) + ' '.join(d)
            else:
                if c % 3 == 0:
                    d = data[3*fullscreen:3*fullscreen+3].transpose((1, 2, 0))
                else:
                    d = data[fullscreen]
                if data.shape[0] > MAX_FULL_SIZE[0] or data.shape[1] > MAX_FULL_SIZE[1]:
                    fulldim = min(MAX_FULL_SIZE[0]*ratio, MAX_FULL_SIZE[1])
                    full_size = (round(fulldim/ratio), round(fulldim))
                    d = cv2.resize(d, full_size, interpolation=cv2.INTER_AREA)
                png = str(base64.b64encode(cv2.imencode('.png', d)[1]))[2:-1]

                mindim = min(THUMBNAIL_SIZE[0] * ratio, THUMBNAIL_SIZE[1])
                thumbnail_size = (round(mindim / ratio), round(mindim))

                interp = cv2.INTER_AREA if not self.is_label else cv2.INTER_NEAREST
                thumbnail = cv2.resize(d, thumbnail_size, interpolation=interp)
                t_png = str(base64.b64encode(cv2.imencode('.png', thumbnail)[1]))[2:-1]

                legend = '<p> min: %f<br/> max: %f<br /> mean: %f <br /> std: %f</p>' \
                          % (raw_data.min(), raw_data.max(), raw_data.mean(), raw_data.std())
                # -- Send data --
                # HACKISH: message interpreted by FulllscreenView.js IPython.FullscreenView.setContent()
                msg = 'I data:image/png;base64, %s||data:image/png;base64, %s||%s' % \
                      (t_png, png, legend)
                return msg

        def format_data(self, data):
            import cv2
            return [cv2.imencode('png', d)[0] for d in data]

        def write_file(self, data, path, filename, overwrite=True):
            import cv2

            split_data = []

            for i in range(0, data.shape[0], 3):
                d = data[i:i+3]
                if d.shape[0] in (1, 3):
                    split_data.append(d.transpose((1, 2, 0)))
                elif d.shape[0] == 2:
                    split_data.append(d[0])
                    split_data.append(d[1])

            if len(split_data) == 1:
                f = join(path, filename + '.png')
                if not overwrite and exists(f):
                    return
                cv2.imwrite(f, split_data[0])
            else:
                for i, d in enumerate(split_data):
                    f = join(path, filename+str(i)+'.png')
                    if not overwrite and exists(f):
                        continue
                    cv2.imwrite(f, d)

    class LabelImage(Image):
        def __init__(self, dtype, shape, mapping=None, default=None):
            super(DSColumnFormat.LabelImage, self).__init__(dtype, shape, is_label=True)
            self.default = default
            self._lut = None
            self.mapping = if_none(mapping, dict())

        def __repr__(self):
            from json import dumps
            from copy import copy
            m = copy(self.mapping)
            m['default'] = self.default
            return 'LabelImage()'

        def _preformat(self, data):
            if len(self.shape) == 2:
                data = data[0]
            if self._lut is not None:
                data = self._lut(data)
            if data.ndim == 2:
                return data.reshape((1,)+data.shape)
            return data

        @property
        def mapping(self):
            return self._mapping

        @mapping.setter
        def mapping(self, m):
            if not m:
                self._mapping = m
                self._lut = None
                self.no_scaling = False
                return

            mapping = {}
            for k, v in m.items():
                if isinstance(v, str):
                    from ..j_utils.image import str2color
                    v = str2color(v, bgr=True, uint8=True)
                if k == "default":
                    self.default = v
                else:
                    if len(self.shape) == 3 and np.array(k).ndim == 0:
                        k = (k,)
                    mapping[k] = v
            self._mapping = mapping

            from ..j_utils.image import prepare_lut
            self._lut = prepare_lut(mapping, source_dtype=self.dtype, default=self.default)
            self.no_scaling = True

    @staticmethod
    def auto_format(dtype, shape, info=None):
        if isinstance(info, DSColumnFormat.Base):
            return info.copy(dtype, shape)
        elif isinstance(info, DSColumn):
            return info.format.copy(dtype, shape)
        elif isinstance(info, str):
            # Parse
            info_split = info.split('(', 1)

            if len(info_split) == 2:
                format_name = info_split[0].strip()
                t = info_split[1].rsplit(')', 1)[0]
                args = [_.strip() for _ in t.split(',')]
            else:
                format_name = info.strip()
                args = []

            # Return format
            if format_name == 'Number':
                return DSColumnFormat.Number(dtype, shape)
            elif format_name == 'Text':
                return DSColumnFormat.Text(dtype, shape)
            elif format_name == 'Label':
                return DSColumnFormat.Label(dtype, shape, mapping=args[0] if args else None)
            elif format_name == 'Matrix':
                f = DSColumnFormat.Matrix(dtype, shape)
                if len(args) > 0:
                    f.domain = args[0]
                if len(args) > 1:
                    f.range = args[1]
                if len(args) > 2:
                    f.clip = args[2]
                return f
            elif format_name == 'Matrix':
                f = DSColumnFormat.LabelMatrix(dtype, shape, mapping=args[0] if args else None)
                if len(args) > 1:
                    f.domain = args[1]
                if len(args) > 2:
                    f.range = args[2]
                if len(args) > 3:
                    f.clip = args[3]
                return f
            elif format_name == 'ConfMatrix':
                return DSColumnFormat.ConfMatrix(shape[0])
            elif format_name == 'Image':
                return DSColumnFormat.Image(dtype, shape)
            elif format_name == 'ImageLabel':
                return DSColumnFormat.LabelImage(dtype, shape, mapping=args[0] if args else None)

        if shape == ():
            if np.issubdtype(dtype, np.number):
                if isinstance(info, dict):
                    return DSColumnFormat.Label(dtype, shape, mapping=info)
                else:
                    return DSColumnFormat.Number(dtype, shape)
            elif dtype == 'str' or 'str' in str(dtype) or 'S' in str(dtype) or 'U' in str(dtype):
                return DSColumnFormat.Text(dtype, shape)
        elif np.issubdtype(dtype, np.number) or dtype == np.bool:
            if isinstance(info, str) and info.lower() == 'image':
                is_img = True
            elif isinstance(info, str) and info.lower() == 'matrix':
                is_img = False
            else:
                is_img = len(shape) in (2, 3) and np.min(shape[-2:]) >= 3

            if isinstance(info, dict):
                if is_img:
                    return DSColumnFormat.LabelImage(dtype, shape, mapping=info)
                else:
                    return DSColumnFormat.LabelMatrix(dtype, shape, mapping=info)
            elif is_img and isinstance(info, tuple) and len(info) == 2:
                f = DSColumnFormat.Image(dtype, shape)
                f.domain = info
                return f
            elif dtype == 'bool':
                if is_img:
                    return DSColumnFormat.LabelImage(dtype, shape)
                else:
                    return DSColumnFormat.LabelMatrix(dtype, shape)
            else:
                if is_img:
                    return DSColumnFormat.Image(dtype, shape)
                else:
                    return DSColumnFormat.Matrix(dtype, shape)
        return DSColumnFormat.Base(dtype, shape)
