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
import numpy as np
import base64
import enum
from functools import partial
from copy import copy
from os.path import dirname, exists, join, splitext, basename
from os import makedirs
import pandas as pd
import time
import multiprocessing as mp
import queue
import threading

from ..j_utils.j_log import log, Process, float_to_str

from ..j_utils.function import not_optional_args, to_callable, optional_args
from ..j_utils.parallelism import parallel_exec, intime_generator
from ..j_utils.math import interval
from ..j_utils.collections import if_none, AttributeDict
import weakref


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

    def __init__(self, name='dataset', parent_datasets=None, pk_type=np.int):
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
        if isinstance(parent_datasets, tuple):
            parent_datasets = list(parent_datasets)
        if not isinstance(parent_datasets, list):
            parent_datasets = [parent_datasets]

        self._parents = parent_datasets
        self._pk = DSColumn('pk', (), pk_type, self)

        self._sample = None

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
                return self.read_one(row=item[0], columns=item[1])
            elif isinstance(item[0], slice):
                return self.read(item[0].start, item[0].stop, columns=item[1])
        raise NotImplementedError

    @property
    def rnd(self):
        return self._rnd

    @rnd.setter
    def rnd(self, rnd):
        if rnd is None:
            rnd = np.random.RandomState(1234+os.getpid())
        elif isinstance(rnd, int):
            rnd = np.random.RandomState(rnd)
        self._rnd = rnd

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
    def read(self, start: int = None, end: int = None, columns=None, extract=True, n=None):
        if start is None:
            start = 0
        if end is None:
            end = self.size
        elif end < 0:
            end -= self.size
        elif end < start:
            end = start + 1
        if end > self.size:
            raise ValueError('%i is not a valid index: dataset size is %i' % (end, self.size))

        d = self
        if n is not None:
            d = d.subgen(n)

        gen = d.generator(end - start, start=start, columns=columns)
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
            gen = gen_context.generator(self, start=id, end=id+n, n=n, columns=columns)
        else:
            gen = self.generator(start=id, end=id+n, n=n, columns=columns)
        r = gen.next()
        gen.clean()
        if clear_weakref:
            r.clear_dataset()
            return r

        return r, gen

    def read_one(self, row=0, columns=None, extract=True):
        """
        Read a specific element of a dataset. If extract is True, the result will depend on the form of columns.
        Thus, if columns is None, read_one(i) will return a dictionnary a the i-th value of all the dataset's columns,
              if columns is a list, read_one will organize the values in a list in the same order as in columns
        finally if columns is a string, only the element at this position and column will be returned.
        :param row: Row of the wanted elements
        :param columns: Columns of the wanted elements (None mean all of them)
        :param extract: If true, the data is extracted from the DataSetResult.
        """
        r = self.read(start=row, end=row + 1, columns=columns, extract=False)

        if not extract:
            return r
        if columns is None:
            return {c_name: c_data[0] for c_name, c_data in r.items()}
        elif isinstance(columns, list):
            return [r[_][0] for _ in self.interpret_columns(columns)]
        else:
            return r[columns][0]

    #   ---   Generators   ---
    def generator(self, n=1, start=None, end=None, columns=None, determinist=True, intime=False, ncore=1):
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
        return DataSetSmartGenerator(dataset=self, n=n, start_id=start, end_id=end, columns=columns,
                                     determinist=determinist, intime=intime, ncore=ncore)

    @abstractmethod
    def _generator(self, gen_context):
        pass

    def _setup_determinist(self):
        pass

    #   ---   Columns   ---
    def interpret_columns(self, columns, to_column_name=True):
        """Returns a list of columns name

        :param columns: None, list of :class:`DataSetColumn` or str, str or :class:`DataSetColumn`. \
        If None, the method returns the columns' name of the current dataset.
        :return: list of columns name
        :rtype: list
        """

        if columns is None:
            columns = self._columns.copy()
        elif not isinstance(columns, list):
            columns = [columns]
        else:
            columns = columns.copy()

        for c_id, c in enumerate(columns):
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
    def pk(self):
        return self._pk

    def copy_columns(self, dataset=None):
        if dataset is None:
            dataset = self
        return [DSColumn(_.name, _.shape, _.dtype, dataset) for _ in self._columns]

    def add_column(self, name, shape, dtype):
        self._columns.append(DSColumn(name=name, shape=shape, dtype=dtype, dataset=self))

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

    #   ---   Representation functions   ---
    def __str__(self):
        return self.__class__.__name__ +'(name="' + self._name + '")'

    def __repr__(self):
        s = 'Dataset: %s' % str(self)
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

            columns_name = []
            columns_description = []
            for c in dataset._columns:
                columns_name.append(c.name)
                c_name = c.name + ';' + c.sql_type
                if len(c.shape) > 0:
                    c_name += ' [' + ('x'.join([str(_) for _ in c.shape])) + ']'
                columns_description.append(c_name)

            def retreive_data(row, col):
                if context['generator'] is not None and context['gen_id']+1 == row:
                    context['gen_id'] += 1
                    context['result'] = next(context['generator'])
                elif context['gen_id'] != row:
                    context['generator'] = dataset.generator(start=row)
                    context['gen_id'] = row
                    context['result'] = next(context['generator'])

                return context['result'][0, columns_name[col]]

            w.columns_name = columns_description
            w.db_view.retreive_data = retreive_data
            w.db_view.reset()

        dataset_changed(len(datasets)-1)
        w.hierarchy_bar.on_id_changed(dataset_changed)

        return w

    #   ---   Sample   ---
    @property
    def sample(self):
        if self._sample is None:
            self._sample = self.read_one()
        return self._sample

    def clear_sample(self):
        del self._sample
        self._sample = None

    #   ---   Export   ---
    def export(self, cb, n=1, start=0, end=None, columns=None, determinist=True, ncore=None):
        if columns is None:
            columns = self.columns_name()
        start %= self.size
        if end is None:
            end = self.size
        elif end != 0:
            end %= self.size
            if end == 0:
                end = self.size

        size = end-start
        if size <= 0:
            return

        if ncore is None:
            ncore = N_THREAD
        ncore = min(ncore, size)

        if ncore > 1:
            # Setup generators
            gens = []
            for id_gen in range(ncore):
                start_id = start + np.round(size / ncore * id_gen)
                end_id = start + np.round(size / ncore * (id_gen + 1))

                g = self.generator(n=n, start=start_id, end=end_id, columns=columns, intime=True,
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
            for r in self.generator(n=n, start=start, end=end, columns=columns, determinist=determinist):
                cb(r)

    def sql_write(self, database_path, table_name, start=0, end=0, n=10,
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
        if end <= 0:
            end += self.size

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
            for i in range(min(n, end-i_global)):
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
        with Process('Saving %s' % self._name, end-start) as p:
            for i in range(start, end, n):
                access.execute_query(InsertQuery(table_or_subquery=table_name,
                                                 replace=True,
                                                 column_name=['pk' if include_pk else 'rowid'] + columns,
                                                 update_value='?' + (',?'*(len(columns)))),
                                     format_gen(p))
                i_global += n
        access.write_db('PRAGMA journal_mode=DELETE')

    def folder_write(self, folder_path, start=0, end=0, columns=None, determinist=True,
                     compress_format='png', metadata_format='csv', compress_img=True,
                     column_for_filename=None):

        if column_for_filename is None:
            if 'name' in self.columns_name():
                column_for_filename = 'name'
            else:
                column_for_filename = 'pk'

        single_column = False
        if columns is None:
            columns = self.columns_name()
            if column_for_filename in columns:
                columns.remove(column_for_filename)
        elif isinstance(columns, str) or isinstance(columns, DSColumn):
            columns = [columns]
            single_column = True
        if isinstance(columns, list):
            for c_id, c in enumerate(columns):
                if isinstance(columns, str):
                    if columns not in self.columns_name():
                        raise ValueError('Unknown column %s.' % columns)
                elif isinstance(columns, DSColumn):
                    if columns.dataset is not self:
                        raise ValueError('%s is not a column of this dataset.' % columns.name)
                    columns[c_id] = c.name
            columns = list(set(columns))
        else:
            raise NotImplementedError

        # img_columns = [c.name for c in self.columns if len(c.shape) == 3 and c.shape[0] in (1, 3)] This won't work in case of images of format (h, w) or (h, w, c).
        # It's the default format when reading a SQLDataset with compressed images.
        img_columns = [c for c in columns if len(self.column_by_name(c).shape) > 1]
        metadata_columns = [c for c in columns if c not in img_columns]

        if start < 0:
            start += self.size
        if end <= 0:
            end += self.size
        gen_columns = columns.copy()
        if column_for_filename is not 'pk' and column_for_filename not in gen_columns:
            gen_columns.append(column_for_filename)
        data_gen = self.generator(start=start, columns=gen_columns)

        if folder_path[-1] != '/':
            folder_path += '/'
        database_path_dir = dirname(folder_path)
        if database_path_dir and not exists(database_path_dir):
            makedirs(database_path_dir)

        if not single_column:
            for _ in img_columns:
                path = join(folder_path, _)
                if not exists(path):
                    makedirs(path)

        def format_type(r, c):
            column = self.column_by_name(c)
            d = r[c][0]
            if column.sql_type == 'INTEGER':
                return int(d)
            elif column.sql_type == 'TEXT':
                return str(d)
            else:
                return d

        dataframe = pd.DataFrame(columns=metadata_columns)
        with Process('Saving %s' % self._name, end - start) as p:
            for i in range(start, end):
                # Read data
                try:
                    r = next(data_gen)
                except StopIteration:
                    break
                p.update(0.5)

                # Format and store data
                data = [format_type(r, _) for _ in img_columns]
                filename = basename(str(r[column_for_filename][0])).replace(".", "_")

                for img_data, column_name in zip(data, img_columns):
                    array = img_data
                    if array.ndim > 2:
                        if array.shape[0] < array.shape[2]:
                            array = array.transpose((1, 2, 0))
                    else:
                        array = np.expand_dims(array, axis=2)
                    if array.shape[2] == 1:
                        array = array.reshape(array.shape[:-1])
                    if np.max(array) <= 1. and np.min(array) >= 0.:
                        array *= 255

                    img_folder_path = join(folder_path, column_name) if not single_column else folder_path

                    import cv2
                    if compress_img:
                        path = join(img_folder_path, filename+'.'+compress_format)
                        cv2.imwrite(path, array)
                    else:
                        path = join(img_folder_path, filename + '.ppm')
                        cv2.imwrite(path, array)

                metadata = [format_type(r, _) for _ in metadata_columns]
                if column_for_filename in metadata_columns and len(metadata) == 1:
                    dataframe = None
                else:
                    serie = pd.Series(metadata, metadata_columns)
                    dataframe = dataframe.append(serie, ignore_index=True)
                p.update(0.5)
        if dataframe is None or not metadata_columns:
            return

        if metadata_format is 'csv':
            dataframe.to_csv(folder_path+'meta.csv')
        elif metadata_format is 'excel' or metadata_format is 'xlsx':
            dataframe.to_csv(folder_path+'meta.xlsx')
        elif metadata_format is 'json':
            dataframe.to_json(folder_path+'meta.json')

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
                    self._gen = self._dataset.generator(ncore=ncore, mp=mp)
                try:
                    return self._gen.next()
                except StopIteration:
                    raise IndexError

        return CustomDataLoader(self)

    #   ---   Operations   ---
    def subset(self, start=0, end=None, name='subset', *args):
        from .datasets_core import DataSetSubset
        start, end = interval(self.size, start, end, args)
        return DataSetSubset(self, start, end, name=name)

    def subgen(self, n=1, name='subgen'):
        from .datasets_core import DataSetSubgen
        return DataSetSubgen(self, n=n, name=name)

    def as_cache(self, n=1, start=0, end=None, columns=None, ncore=1, name=None):
        start, end = interval(self.size, start, end)
        if columns is None:
            columns = self.columns_name()
        data = DataSetResult.create_empty(dataset=self, n=end - start, start_id=start, columns=columns)

        if name is None:
            label = self._name
            name = 'cache'
        else:
            label = name

        with Process('Caching %s' % label, end - start, verbose=False) as p:
            def write_back(r):
                data[r.start_id-start:r.end_id-start] = r
                p.update(r.size)
            self.export(write_back, n=n, start=start, end=end, columns=columns, ncore=ncore)

        dataset = NumPyDataSet(data, name=name)
        dataset._parents = [self]
        return dataset

    def split_sets(self, **kwargs):
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

        for name, ratio in kwargs.items():
            if ratio != -1:
                if 0 < ratio < 1:
                    l = round(self.size * ratio)
                elif ratio >= 1:
                    l = round(ratio)
                    ratio = l / self.size
                else:
                    raise NotImplementedError('Datasets ratio must be a positive number')
                d[name] = self.subset(offset, offset+l, name=name)
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
                d[name] = self.subset(offset, offset + l)
                offset += l

        return d

    def concat(self, **kwargs):
        """"Map columns name or concatenate two columns according to kwargs (keys are created columns,
        and values are either names of column to map or lists of column names to concatenate).

        If a column is not mentioned kwargs values, it is kept with the same name (unlike ```map()```).
        """
        from .datasets_core import DataSetMap
        return DataSetMap(self, kwargs, keep_all_columns=True)

    def map(self, **kwargs):
        """"Map columns name or concatenate two columns according to kwargs (keys are created columns,
        and values are either names of column to map or lists of column names to concatenate).

        If a column is not mentioned kwargs values, it is discarded (unlike ```concat()```).
        """
        from .datasets_core import DataSetMap
        return DataSetMap(self, kwargs, keep_all_columns=False)

    def shuffle(self, indices=None, subgen=0, rnd=None, name='shuffle'):
        from .datasets_core import DataSetShuffle
        return DataSetShuffle(self, subgen=subgen, indices=indices, rnd=rnd, name=name)

    def apply(self, function, columns=None, same_size_type=False, columns_type_shape=None, name='apply'):
        from .datasets_core import DataSetApply
        return DataSetApply(self, function=function, columns=columns, name=name,
                            columns_type_shape=columns_type_shape, same_size_type=same_size_type)

    def reshape(self, columns, shape, label_columns=None, keep_original=False, name='reshape'):
        from .datasets_core2d import DataSetReshape
        return DataSetReshape(self, columns=columns, shape=shape, label_columns=label_columns,
                              keep_original=keep_original, name=name)

    def join(self, datasets, verbose=False, parallel=False, **kwargs):
        for c in self._columns:
            if c.name in kwargs:
                kwargs[c.name+'1'] = kwargs[c.name]
            kwargs[c.name] = c

        if not isinstance(datasets, list):
            datasets = [datasets]
        found_self = False
        for id, _ in enumerate(datasets):
            if isinstance(_, tuple):
                if _[0] is self:
                    found_self = True
                    break
            elif isinstance(_, DSColumn):
                if _.dataset is self:
                    found_self = True
                    break
            elif isinstance(_, str):
                if _ != 'pk' and _ not in self.columns_name():
                    raise ValueError('%s is not a column of dataset %s' % (_, self.dataset_name))
                datasets[id] = self.column_by_name(_)
                found_self = True,
                break
        if not found_self:
            datasets.insert(0, self.pk)

        from .datasets_core import DataSetJoin
        return DataSetJoin(datasets=datasets, verbose=verbose, parallel=parallel, **kwargs)

    def augment_data(self, columns,
                     N_augmented,
                     geom_trans=False,
                     color_trans=False,
                     function2avoid=None,
                     keep_original=True,
                     custom_function=None,
                     column_transform = False,
                     **kwargs):
        """Proceeds with data augmentation

        This method returns a :class:`DataSetAugmentedData` object that will generate new transformed instances.

        :param columns: columns in the dataset that will be affected by the data augmentation engine.
        :param N_augmented: number of new instances generated from one single instance
        :param custom_function: A reference to a method that will be applied on a instance. The method definition must as least have \
        an ``input`` parameter, that will be automatically filled. Others parameters can be passed as a pair of key-value within this function.
        :param geom_trans: determines whether standard geometric transformation should be randomly proceeded. Others parameters \
        can be passed as a pair of key-value within this function.
        :param color_trans: determines whether standard color transformation should be randomly proceeded. Others parameters \
        can be passed as a pair of key-value within this function.
        :param function2avoid:
        :param keep_original:
        :param kwargs:
        :return:
        """
        from .data_augmentation import DataAugmentation

        dict_params = kwargs
        dict_params['use_geom_func'] = geom_trans
        dict_params['use_color_func'] = color_trans
        dict_params['custom_function'] = custom_function
        dict_params['function2avoid'] = function2avoid

        # Initialize columns
        if not isinstance(columns, list):
            columns = [columns]
        for c_id, c in enumerate(columns):
            if isinstance(c, str):
                if c not in self.columns_name():
                    raise ValueError('%s is not a column of %s' % (c, self.dataset_name))
                columns[c_id] = self.column_by_name(c)

        img_shape = columns[0].shape[-2:]
        for c in columns[1:]:
            if c.shape[-2:] != img_shape:
                raise ValueError('All data-augmented columns should have the same shape!')

        da_engine = DataAugmentation(**dict_params)

        from .data_augmentation import DataSetAugmentedData
        return DataSetAugmentedData(self, columns=columns, n=N_augmented,
                                    da_engine=da_engine, keep_original=keep_original, column_transform=column_transform)

    def patches(self, columns=None, patch_shape=(128, 128), stride=(0, 0), ignore_borders=False, mask=None,
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
            if not isinstance(patch_shape, tuple):
                patch_shape = (patch_shape, patch_shape)

            if not isinstance(columns, list):
                if isinstance(columns, DSColumn):
                    columns = columns.name
                patches_def[columns] = (columns, patch_shape)
            else:
                for c_id, c in enumerate(columns):
                    if isinstance(c, DSColumn):
                        c = c.name
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
        return DataSetPatches(self, patch_shape=patches_def, patches_function=patchify,
                              center_pos=center_pos, name=name)

    def random_patches(self, columns=None, patch_shape=(128,128), n=10, proba_map=None, rng=None,
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
            if not isinstance(patch_shape, tuple):
                patch_shape = (patch_shape, patch_shape)

            if not isinstance(columns, list):
                if isinstance(columns, DSColumn):
                    columns = columns.name
                patches_def[columns] = (columns, patch_shape)
            for c_id, c in enumerate(columns):
                if isinstance(c, DSColumn):
                    c = c.name
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
            proba_map = proba_map / proba_map.sum()
            proba_map = to_callable(proba_map)
        elif not callable(proba_map):
            raise NotImplementedError

        from .datasets_core2d import DataSetPatches
        return DataSetPatches(self, patch_shape=patches_def, n=n, center_pos=center_pos, name=name,
                              patches_function=proba_map, post_process=patchify)

    def unpatch(self, columns=None, patch_mix='replace', restore_columns=None, columns_shape=None, n_patches=None):
        from .datasets_core2d import DataSetUnPatch
        return DataSetUnPatch(self, patch_mix=patch_mix, columns=columns, n_patches=n_patches,
                              restore_columns=restore_columns, columns_shape=columns_shape)


########################################################################################################################
class DSColumn:
    """
    Store information of a column of a DataSet
    """
    def __init__(self, name, shape, dtype, dataset=None, datatype=None):
        self._name = name
        self._shape = shape
        self._dtype = dtype
        if dtype == str:
            self._dtype = 'O'

        self._format_export = None
        self._format_display = None
        if datatype is not None:
            self._datatype = datatype
        elif dataset is None:
            self._datatype = DSColumn.DataType.UNKNOWN
        else:
            self._datatype = DSColumn.DataType.infer_type(self._dataset.sample[name])
        self.fullscreen_display = DSColumn.DataType.default_fullscreen_display(self._datatype)

    def __getstate__(self):
        return self._name, self._shape, self._dtype

    def __setstate__(self, state):
        self._name, self._shape, self._dtype = state
        self._dataset = None

    @property
    def shape(self):
        """
        Shape of the column
        :rtype: tuple 
        """
        return self._shape

    @property
    def dtype(self):
        """
        Type of the data in the column
        :rtype: type
        """
        return self._dtype

    @property
    def datatype(self):
        return self._datatype

    @property
    def sql_type(self):
        # TODO: Remove this property
        if self.shape != ():
            return 'array'
        if 'int' in str(self.dtype):
            return 'INTEGER'
        elif 'float' in str(self.dtype):
            return 'FLOAT'
        elif str(self.dtype) in ('str', 'O'):
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

    def __repr__(self):
        return '%s: %s %s' % (self.name, str(self.shape), str(self.dtype))

    @property
    def format_display(self):
        if self._format_display is None:
            return self.default_display_export
        else:
            return self._format_export

    @format_display.setter
    def format_display(self, f):
        if f is None or callable(f):
            self._format_display = f

    def format_html(self, x, thumbnail):
        datatype = self.datatype
        DT = DSColumn.DataType

        if datatype is DT.IMAGE:
            import cv2
            if x.ndim == 3:
                x = x.transpose((1, 2, 0))
            if thumbnail:
                h, w = x.shape[:2]
                ratio = h / w
                mindim = min(thumbnail[0] * ratio, thumbnail[1])
                thumbnail_size = (round(mindim / ratio), round(mindim))
                x = cv2.resize(x, thumbnail_size, interpolation=cv2.INTER_AREA)

            png = base64.b64encode(cv2.imencode('.png', x)[1])[2:-1]

        elif datatype is DT.INTEGER:
            return '<p> %i </p>' % x[0]
        elif datatype is DT.FLOAT:
            return '<p> %.4f </p>' % x[0]
        elif datatype is DT.TEXT:
            return '<p> %s </p>' % x[0]
        else:
            return '<p> %s </p>' % repr(x)

    @property
    def format_export(self):
        if self._format_export is None:
            return DSColumn.default_format_export
        return self._format_export

    @format_export.setter
    def format_export(self, f):
        if f is None or callable(f):
            self._format_export = f

    def default_format_export(self, x):
        DT = DSColumn.DataType
        if self.datatype is DT.IMAGE:
            if 'int' in x.dtype:
                if x.min() < 0 or x.max() > 255:
                    x -= x.min()
                    x /= 255*x.max()
            elif 'float' in x.dtype:
                if x.min() < 0 or x.max() > 1:
                    x -= x.min()
                    x /= x.max()
        return x

    def default_display_export(self, x):
        return self.format_export(x)

    @enum.unique
    class DataType(enum):
        CUSTOM = -1
        UNKNOWN = 0
        ARRAY = 1       # Array of Numbers
        INTEGER = 2     # Single integer
        NUMBER = 3      # Single float number
        TEXT = 4        # Single text
        IMAGE = 5       # Single or mutliple images

        @staticmethod
        def infer_type(x):
            if x.shape == ():
                if 'int' in x.dtype:
                    return DSColumn.DataType.INTEGER
                elif 'float' in x.dtype:
                    return DSColumn.DataType.NUMBER
                elif x.dtype == 'str':
                    return DSColumn.DataType.TEXT
                elif x.dtype == 'O':
                    if isinstance(x[0], str):
                        return DSColumn.DataType.TEXT
                    else:
                        return DSColumn.DataType.UNKNOWN
            elif 'int' in x.dtype or 'float' in x.dtype or x.dtype in ('bool',):
                if x.ndim in (2, 3) and x.shape[-2:].min() >= 3:
                    return DSColumn.DataType.IMAGE
                else:
                    return DSColumn.DataType.ARRAY
            return DSColumn.DataType.UNKNOWN

        @staticmethod
        def default_fullscreen_display(datatype):
            DT = DSColumn.DataType
            return datatype in (DT.IMAGE,)


########################################################################################################################
class DataSetResult:
    """
    Store the result of an iteration of a generator from DataSet
    """

    def __init__(self, data_dict, columns, start_id, size, dataset=None):
        """
        :type data_dict: dict
        :type dataset: AbstractDataSet
        """
        self._data_dict = data_dict
        self._columns = columns
        self._start_id = start_id
        self._size = size
        self._dataset = None

        self.affiliate_dataset(dataset)
        self._trace = DataSetResult.Trace(self)

        self._ipywidget = None

    def __getstate__(self):
        return self._data_dict, self._columns, self._start_id, self._size, self._trace

    def __setstate__(self, state):
        self._data_dict, self._columns, self._start_id, self._size, self._trace = state
        self._dataset = None
        self._ipywidget = None

    @staticmethod
    def create_empty(n, start_id, columns=None, dataset=None, assign=None):
        """

        :param dataset:
        :param n:
        :param start_id:
        :param columns:
        :rtype: DataSetResult
        """
        if dataset is None:
            if columns is None:
                raise ValueError('Either the dataset or the columns list must be specified '
                                 'when creating an empty dataset result.')
            if not isinstance(columns, list):
                columns = [columns]
            else:
                columns = columns.copy()
            for c in columns:
                if not isinstance(c, DSColumn):
                    raise ValueError(
                        'When creating a dataset result without specifying a dataset, columns list should only'
                        'contains valid columns.')
        else:
            columns = dataset.interpret_columns(columns, to_column_name=False)
            if 'pk' not in [_.name for _ in columns]:
                columns.insert(0, dataset.column_by_name('pk'))

        if assign is None:
            assign = {}

        data_dict = {}
        for c in columns:
            if c.name in assign:
                a = assign[c.name]
                if a.shape != (n,) + tuple(c.shape):
                    raise ValueError('The shape of the assigned value for column %s is %s but should be %s.'
                                     % (c.name, repr(a.shape), repr((n,)+c.shape)))
                data_dict[c.name] = a
            else:
                data_dict[c.name] = np.zeros(tuple([n]+list(c.shape)), dtype=c.dtype)

        return DataSetResult(data_dict=data_dict, columns=columns, start_id=start_id, size=n, dataset=dataset)

    @staticmethod
    def create_from_data(data):
        """
        Create a generic DataSetResult using data
        :param data: a dictionary where each column uses a different key
        :return:
        """
        columns = [DSColumn(name=name, shape=d.shape[1:], dtype=d.dtype) for name, d in data.items()]

        n = None
        for d in data.values():
            if n is None:
                n = d.shape[0]
            elif n != d.shape[0]:
                raise ValueError("Error when creating DataSetResult from data: some columns don't share the same length.")

        if 'pk' not in [_.name for _ in columns]:
            columns.insert(0, DSColumn(name='pk', shape=(), dtype=np.uint16))
            data['pk'] = np.arange(n, dtype=np.uint16)

        return DataSetResult(data_dict=data, columns=columns, start_id=0, size=n)

    def _ipython_display_(self):
        from ..j_utils.ipython import import_display
        import_display(self.ipywidget)

    @property
    def ipywidget(self):
        from ..j_utils.ipython.databaseview import SimpleDatabaseView

        if self._ipywidget is not None:
            return self._ipywidget

        w = SimpleDatabaseView()

        columns_name = []
        columns_description = []
        for c in self.columns:
            columns_name.append(c.name)
            c_name = c.name + ';' + c.sql_type
            if len(c.shape) > 0:
                c_name += ' [' + ('x'.join([str(_) for _ in c.shape])) + ']'
            columns_description.append(c_name)

        def retreive_data(row, col):
            return self[row, columns_name[col]]

        w.columns_name = '|'.join(columns_description)
        w.retreive_data = retreive_data
        w.length = self.size

        self._ipywidget = w
        return w

    @property
    def dataset(self):
        """
        Parent dataset
        :rtype: AbstractDataSet
        """
        if self._dataset is not None:
            return self._dataset()
        return None

    def affiliate_dataset(self, dataset):
        self._dataset = weakref.ref(dataset)
        for c in self._columns:
            c._dataset = weakref.ref(dataset)

    def clear_dataset(self):
        self._dataset = None
        for c in self._columns:
            c._dataset = None

    def __del__(self):
        self.clean()

    def clean(self):
        for data in self._data_dict.values():
            if data.flags['OWNDATA']:
                del data
        self._data_dict.clear()
        self._columns.clear()
        self._dataset = None
        self._size = 0
        self._start_id = 0
        self._trace = DataSetResult.Trace(self)

    @property
    def start_id(self):
        """
        Index of the beginning of this subset in the parent dataset
        :rtype: int
        """
        return self._start_id

    @property
    def end_id(self):
        """
        Index of the ending of this subset in the parent dataset
        :rtype: int
        """
        return self._start_id + self._size

    @property
    def size(self):
        """
        Size of this subset (first dimension of every columns of this subset)
        :rtype: int
        """
        return self._size

    @property
    def mem_size(self):
        """
        Amount of memory used to store this DataSetResult in bytes. (Does not include shared memory)
        :rtype: int
        """
        b = 0
        for data in self._data_dict.values():
            if data.flags['OWNDATA']:
                b += data.nbytes
        return b

    @property
    def total_mem_size(self):
        """
        Total amount of memory used to store this DataSetResult in bytes. (Includes shared memory)
        :rtype: int
        """
        b = 0
        for data in self._data_dict.values():
            b += data.nbytes
        return b

    @property
    def trace(self):
        return self._trace

    @property
    def columns(self):
        return self._columns

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size != 0

    def __setitem__(self, key, value):
        simple_id = isinstance(key, int)
        if simple_id:
            key = slice(key, key+1, 1)
        if isinstance(key, slice):
            if not key.step is None and key.step != 1:
                raise IndexError('Not continuous slice index is not supported'
                                 '(step should be 1 and not %i)' % key.step)
            start = 0 if key.start is None else key.start
            stop = self.size if key.stop is None else key.stop
            length = stop-start

            if isinstance(value, DataSetResult):
                assert value.size >= length
                value = value._data_dict
            if isinstance(value, dict):
                for v_name, v in value.items():
                    if v_name in self._data_dict:
                        d = v if simple_id else v[:length]

                        from scipy.sparse import spmatrix
                        if isinstance(d, spmatrix):
                            d = d.toarray()

                        self._data_dict[v_name][start:stop] = d
            elif isinstance(value, list):
                try:
                    self._data_dict['pk'][start:stop] = value[0] if simple_id else value[0][:length]
                    for c_id, c in enumerate(self.columns):
                        self._data_dict[c.name][start:stop] = value[c_id+1] if simple_id else value[c_id+1][:length]
                except:
                    raise ValueError('%s is not a valid value to assign to a dataset row...' % value)

        elif isinstance(key, tuple):
            def istypeof_or_listof(o, t):
                return isinstance(o, t) or (isinstance(o, t) and isinstance(o[0], t))

            if (isinstance(key[0], slice) or istypeof_or_listof(key[0], int)) and (
                istypeof_or_listof(key[1], str) or istypeof_or_listof(key[1], list)):
                indexes = key[0]
                columns = key[1]
            else:
                raise NotImplementedError
            if isinstance(indexes, int):
                indexes = slice(indexes, indexes+1)
            if columns in self:
                if isinstance(value, np.ndarray):
                    np.copyto(self._data_dict[columns][indexes], value)
                else:
                    self._data_dict[columns][indexes] = value
        elif isinstance(key, str):
            self._data_dict[key][:] = value
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._data_dict[item]
        elif isinstance(item, (int, slice)):
            return self[item, set(self.columns_name() + ['pk'])]
        elif isinstance(item, list):
            return [self[_] for _ in item]
        elif isinstance(item, tuple):
            if (isinstance(item[0], slice) or istypeof_or_listof(item[0], int)) and istypeof_or_collectionof(item[1], str, (list, tuple, set)):
                indexes = item[0]
                columns = item[1]
            else:
                raise NotImplementedError('First index should be row index and second should be columns index\n'
                                          '(here provided type is [%s, %s])' % (str(type(item[0])), str(type(item[1]))))

            if isinstance(columns, (list, tuple)):
                return [self._data_dict[_][indexes] for _ in columns]
            elif isinstance(columns, set):
                return {c: self._data_dict[c][indexes] for c in columns}
            else:
                return self._data_dict[columns][indexes]
        else:
            raise NotImplementedError

    def __contains__(self, item):
        return item in self._data_dict

    def columns_name(self):
        """
        :rtype: List[str]
        """
        return [_ for _ in self.keys() if _ != 'pk']

    def keys(self):
        return [_.name for _ in self._columns]

    def items(self):
        for c_name in self.keys():
            yield c_name, self._data_dict[c_name]

    def truncate(self, n):
        for _, data in self._data_dict.items():
            self._data_dict[_] = data[:n]
        self._size = n
        return self

    class Trace:
        def __init__(self, r):
            dataset = r.dataset
            if dataset is not None:
                self._dataset_name = dataset.dataset_name[:]
                self._n = dataset.size
            else:
                self._dataset_name = 'unknown'
                self._n = -1

            self._parent_traces = {}

            self._start_date = time.time()
            self._end_date = None


            self._mem_size = r.mem_size
            self._total_mem_size = r.total_mem_size

        def computation_ended(self):
            self._end_date = time.time()

        def affiliate_parent_trace(self, trace):
            if trace.dataset_name not in self._parent_traces:
                self._parent_traces[trace.dataset_name] = []
            self._parent_traces[trace.dataset_name].append(trace)

        def __repr__(self):
            return 'Trace(%s)' % self.dataset_name

        def __str__(self):
            return self.trace2str()

        def __call__(self, *args, **kwargs):
            print(self.trace2str())

        def print(self):
            print(self.trace2str())

        def trace2str(self, total_time=None):
            s = ''
            if total_time is None:
                total_time = self.total_exec_time
            for t in self.all_parent_traces():
                s += t.trace2str(total_time)
            s += '-- %s --\n' % self.dataset_name

            if total_time:
                s += '\t exec time: %s s, %s%% (total: %s s, %s%%)\n' \
                     % (float_to_str(self.exec_time, '.2f'),
                        float_to_str(self.exec_time / total_time * 100, '.1f'),
                        float_to_str(self.total_exec_time, '.2f'),
                        float_to_str(self.total_exec_time / total_time * 100, '.1f'))
                s += '\t mem usage: %s kB\n' % float_to_str(self.memory_allocated / 1000, '.1f')
                s += '\t mem shared: %s kB\n' % float_to_str(self.shared_memory / 1000, '.1f')
            else:
                s += 'Not executed yet'
            s += '\n'
            return s

        @property
        def dataset_name(self):
            return self._dataset_name

        @property
        def start_date(self):
            return self._start_date

        @property
        def end_date(self):
            return self._end_date

        @property
        def total_exec_time(self):
            if self._end_date is None:
                return None
            return self._end_date - self._start_date

        @property
        def exec_time(self):
            if self._parent_traces:
                return self.total_exec_time - self.traces_total_time(self.all_parent_traces())
            return self.total_exec_time

        @property
        def memory_allocated(self):
            return self._mem_size

        @property
        def shared_memory(self):
            return self._total_mem_size - self._mem_size

        @property
        def total_memory(self):
            return self._total_mem_size

        @property
        def parent_traces(self):
            return self._parent_traces

        def all_parent_traces(self, recursive=False):
            traces = []

            if recursive:
                def recursive_search(trace):
                    for trace_list in trace._parent_traces.values():
                        for t in trace_list:
                            traces.append(t)
                            recursive_search(t)
                recursive_search(self)
            else:
                for t in self._parent_traces.values():
                    traces += t
            return traces

        def dataset_traces(self, dataset_name, recursive=False):
            if dataset_name not in self._parent_traces:
                return 0
            if recursive:
                traces = []

                def recursive_search(trace):
                    for trace_list in trace._parent_traces.values():
                        if trace_list.dataset_name == dataset_name:
                            for t in trace_list:
                                traces.append(t)
                        else:
                            for t in trace_list:
                                recursive_search(t)

                recursive_search(self)
            else:
                traces = self._parent_traces[dataset_name]
            return traces

        @staticmethod
        def traces_total_time(traces, parallel_time=True):
            if not traces:
                return 0
            if parallel_time:
                #Todo: check not continuous execution
                return max([_.end_date for _ in traces]) - min([_.start_date for _ in traces])
            else:
                return sum(_.total_exec_time for _ in traces)


########################################################################################################################
class DataSetSmartGenerator:
    """
    Handy interface for generator's computation
    """

    class Context:
        def __init__(self, start_id, end_id, n, columns, dataset, determinist, ncore, parallelism):
            self.start_n = n
            self.start_id = start_id
            self.end_id = end_id
            self.columns = columns
            self.dataset = dataset
            self.determinist = determinist
            self.ncore = ncore
            self.parallelism = parallelism

            self._copy = {}
            self._r = None
            self.n = n
            self.id = start_id

        def __getstate__(self):
            return self.start_n, self.start_id, self.end_id, self.columns, self.dataset, self.determinist, \
                   self.ncore, self.parallelism, self.n, self.id

        def __setstate__(self, state):
            self.start_n, self.start_id, self.end_id, self.columns, self.dataset, self.determinist, \
            self.ncore, self.parallelism, self.n, self.id = state
            self._copy = {}
            self._r = None

        def generator(self, dataset, start=None, end=None, n=None, columns=None, parallel=False, ncore=None):
            if start is None:
                start = self.id
            if end is None:
                end = min(self.end_id, dataset.size)
            if n is None:
                n = self.n
            if columns is None:
                columns = self.columns
            if ncore is None:
                ncore = self.ncore

            gen = dataset.generator(start=start, end=end, n=n, columns=columns,
                                    determinist=self.determinist, intime=parallel, ncore=ncore)

            return gen

        def create_result(self, result_only=False, as_weakref=True):
            n = min(self.n, self.dataset.size - self.id)
            c = if_none(self._copy, {})
            self._r = DataSetResult.create_empty(dataset=self.dataset, columns=self.columns,
                                                 n=n, start_id=self.id, assign=c)
            self._mem = self._r.mem_size
            self._mem_shared = self._r.total_mem_size - self._mem
            if as_weakref:
                r = weakref.ref(self._r)
            else:
                r = self._r
            if result_only:
                return r
            else:
                return self.id, n, r

        def is_last(self):
            return self.id + self.n >= self.end_id

        def ended(self):
            return self.id >= self.end_id

        def lite_copy(self):
            r = copy(self)
            r._r = None
            return r

        @property
        def id_length(self):
            return self.end_id-self.start_id

    def __init__(self, dataset: AbstractDataSet, n, start_id, end_id, columns=None, determinist=True,
                 ncore=None, intime=False):
        if columns is None:
            columns = dataset.columns_name()
        else:
            columns = dataset.interpret_columns(columns)

        if ncore is None:
            ncore=N_THREAD

        if start_id is None:
            start_id = 0
        else:
            start_id %= dataset.size

        if end_id is None:
            end_id = dataset.size
        elif end_id != 0:
            end_id %= dataset.size
            if end_id == 0:
                end_id = dataset.size

        if not intime:
            intime = False
        elif isinstance(intime, bool):
            intime = 'process'
        elif intime not in ('process', 'thread'):
            raise ValueError('Invalid intime argument: %s \n(should be either "process" or "thread")'
                             % repr(intime))

        self._context = DataSetSmartGenerator.Context(start_id=int(start_id), end_id=int(end_id), n=int(n),
                                                      columns=columns, dataset=dataset,
                                                      determinist=determinist, ncore=ncore, parallelism=intime)

        self._generator_setup = False
        self._generator = None

        self._generator_conn = None
        self._generator_process = None
        self.__warn_limit_copy = False

    def __del__(self):
        self.clean()

    def __next__(self):
        return self.next()

    def next(self, copy=None, seek=None, limit=None, r=None):
        affiliated_result = r
        r = None

        if self.ended():
            raise StopIteration

        if not self._generator_setup or (seek is not None and seek!=self.current_id):
            self.reset(start=seek)
            self.setup()

        if limit is None:
            limit = self._context.start_n
        self._context.n = min(self.end_id - self.current_id, limit)

        if self._asynchronous_exec:
            if (limit is not None) and not self.__warn_limit_copy:
                log.warn("%s's generator is executed asynchronously, data sharing and size limit will be ignored."
                         % self.dataset.dataset_name)

            return self.poll(timeout=None, copy=copy, r=affiliated_result)

        else:
            # Setup context
            self._context._copy = copy

            # Compute next sample
            r = self.next_core(self._generator, self._context)
            self._context._copy = None

            if affiliated_result:
                affiliated_result.trace.affiliate_parent_trace(r.trace)

            if self.ended():
                self.clean()

            return r

    @staticmethod
    def next_core(generator, context):
        r = next(generator)
        if isinstance(r, weakref.ref):
            r = r()
        r.trace.computation_ended()

        # Clean context
        context._r = None
        context.id += r.size
        return r

    def poll(self, timeout=0, copy=None, r=None, ask_next=True):
        """
        Wait for data to be generated and returns it or returns None if timeout is reached.
        :param timeout: Number of second to wait (must be a positive number). If None, wait indefinitely.
        :param r: If specified, affiliate the generated result to r.
        :return : The generated DataSetResult or None
        """
        affiliated_result = r
        r = None
        if self.ended():
            raise StopIteration
        self.setup()

        if not self._asynchronous_exec:
            return self.next(r=r)

        if not self._generator_conn.poll(timeout=timeout):
            return None         # No data available yet

        r = self._generator_conn.recv()

        if isinstance(r, tuple) and len(r)==2 and isinstance(r[0], Exception):
            self._send_to_subprocess(False)
            raise RuntimeError('An error occurred when executing  %s on a sub process:\n| %s'
                               % (str(self.dataset), r[1].replace('\n', '\n| '))) from r[0]
        elif r is None:
            self._context.id = self.end_id
            raise StopIteration

        if ask_next:
            self.ask()

        self._context.n = min(self.end_id - self.current_id, self._context.start_n)

        if copy is not None:
            for c, d in copy.items():
                np.copyto(d, r[c])

        r.affiliate_dataset(self.dataset)
        if affiliated_result:
            affiliated_result.trace.affiliate_parent_trace(r.trace)

        self.context.id += self.n
        if self.ended():
            self.clean()
            if self.n > r.size:
                r.truncate(self.n)

        return r

    def reset(self, start=None, end=None, columns=None, n=None, determinist=None):
        if columns is None:
            columns = self.columns

        if determinist is None:
            determinist = self._context.determinist

        if n is None:
            n = self.n
        if start is None:
            start = self.current_id
        else:
            start %= self.dataset.size
        if end is None:
            end = self.end_id
        elif end != 0:
            end %= self.dataset.size
            if end == 0:
                end = self.dataset.size
        start = int(start); end = int(end); n = int(n)
        self._context.start_id = start

        if start == self.current_id and end == self.end_id and columns == self.columns and n == self.n \
                and determinist == self._context.determinist:
            return

        if self._asynchronous_exec:
            while self._generator_conn.poll(timeout=0):
                self._generator_conn.recv()     # Discard previous samples
            self._send_to_subprocess(dict(start=start, end=end, columns=columns, n=n, determinist=determinist))
        else:
            self.clean()

        self._context.end_id = end
        self._context.columns = columns
        self._context.start_n = n
        self._context.determinist = determinist
        self._context.id = start
        self._context.n = n

    def ask(self, seek=None):
        if seek is not None and seek!=self.current_id:
            self.reset(start=seek)
            self.setup()
        elif self._asynchronous_exec:
            self._send_to_subprocess(True)

    def _send_to_subprocess(self, d, log_broken_pipe=True):
        if self._generator_conn is None:
            if log_broken_pipe:
                log.warn("WARNING: %s generator's pipe is closed.")
            return
        try:
            self._generator_conn.send(d)
        except IOError as e:
            self.context.id = self.end_id
            self._generator_conn = None
            if log_broken_pipe:
                log.warn("WARNING: %s generator's pipe was closed unexpectedly.")

    def __iter__(self):
        while True:
            yield self.next()

    def setup(self):
        if self._generator_setup:
            return
        if self._context.determinist:
            self.dataset._setup_determinist()
        self._generator_setup = True

        if self.intime:
            pipe = mp.Pipe()
            self._generator_conn = pipe[0]

            mpArg = dict(target=parallel_generator_exec,
                         args=(self.context, pipe[1]),
                         name='%s.generator' % self.dataset.dataset_name)

            if self.intime == 'process':
                process = mp.Process(**mpArg)
            else:
                process = threading.Thread(**mpArg)

            process.start()

            self._generator_process = process
        else:
            self._generator = self.dataset._generator(self._context)

    def clean(self):
        self._generator = None
        self._context._r = None
        if self._generator_process:
            while self._generator_process.is_alive():
                self._send_to_subprocess(False, log_broken_pipe=False)
                time.sleep(1e-3)
            del self._generator_process
            self._generator_process = None
        if self._generator_conn:
                del self._generator_conn
                self._generator_conn = None
        self._generator_setup = False

    def is_last(self):
        return self.context.is_last()

    def ended(self):
        return self.context.ended()

    @property
    def columns(self):
        return self._context.columns

    @property
    def dataset(self):
        return self._context.dataset

    @property
    def n(self):
        return self._context.n

    @property
    def current_id(self):
        return self._context.id

    @property
    def start_id(self):
        return self._context.start_id

    @property
    def end_id(self):
        return self._context.end_id

    @property
    def intime(self):
        return self._context.parallelism

    def __len__(self):
        return np.ceil((self.end_id-self.current_id)/self.n)

    @property
    def ncore(self):
        return self._context.ncore

    @property
    def _asynchronous_exec(self):
        # When a generator is executed in parallel, an exact copy of it is pickled and is executed synchronously in an
        # other thread or process. This property will return True for the original generator and False for its copy.
        return self._generator_process is not None

    @property
    def context(self):
        return self._context


def parallel_generator_exec(gen_context, conn):
    SmartGen = DataSetSmartGenerator

    def reset_gen(start, end, n, columns, determinist):
        new_context = SmartGen.Context(start_id=start, end_id=end, n=n,
                                       columns=columns, dataset=gen_context.dataset,
                                       determinist=determinist,
                                       ncore=gen_context.ncore, parallelism=gen_context.parallelism)
        gen = new_context.dataset._generator(new_context)
        return gen, new_context

    # -- Setup generator --
    gen = gen_context.dataset._generator(gen_context)

    # -- Run generator --
    try:
        r = None
        end = False
        wait_response = False
        while not end:
            while "Messages":
                # Checking for messages
                if wait_response or conn.poll(0):
                    rcv = conn.recv()
                    if not rcv:
                        end = True
                        break
                    elif isinstance(rcv, dict):
                        del gen
                        gen, gen_context = reset_gen(**rcv)
                        wait_response = False
                        r.clean()
                        break
                    elif wait_response and rcv is True:
                        wait_response = False
                        break
                    continue

                if r:   # Sending the result in the connection
                    conn.send(r)
                    r.clean()
                    wait_response = True
                else:
                    break

            # Compute next sample
            if not end:
                r = SmartGen.next_core(gen, gen_context)

    except StopIteration:
        pass
    except Exception as e:
        conn.send((e, traceback.format_exc()))

    # Sending end notification
    conn.send(None)
    conn.close()

