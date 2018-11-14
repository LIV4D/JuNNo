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
import cv2
from functools import partial
from copy import deepcopy, copy
from os.path import dirname, exists, join, splitext, basename
from os import makedirs
from sys import getsizeof
import PIL.Image as Image
import io
import pandas as pd
import time
from scipy.sparse import spmatrix
import multiprocessing as mp
import queue
import threading

from ..j_utils.j_log import log, Process, float_to_str

from ..j_utils.function import not_optional_args, to_callable, identity_function, match_params
from ..j_utils.parallelism import parallel_exec, intime_generator
from ..j_utils.math import interval, cartesian
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
        self._pk = DataSetColumn('pk', (), pk_type, self)

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
    def generator(self, n=1, start=None, end=None, columns=None, determinist=True, parallel_exec=False):
        """Creates a generator which iterate through data.

        :param n:  Number of element to return (maximum) by iteration
        :param start: index from which the generator will start reading data
        :param columns: list of columns to read
        :type columns: None or list of str or list of :class:`DataSetColumn`
        :type start: int
        :type n: int

        :return The generator which loops start at from_id and does n iterations.
        :rtype: generator
        """
        return DataSetSmartGenerator(dataset=self, n=n, start_id=start, end_id=end, columns=columns,
                                     determinist=determinist, parallel_exec=parallel_exec)

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
            if isinstance(c, DataSetColumn):
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
        return [DataSetColumn(_.name, _.shape, _.dtype, dataset) for _ in self._columns]

    def add_column(self, name, shape, dtype):
        self._columns.append(DataSetColumn(name=name, shape=shape, dtype=dtype, dataset=self))

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
        return self._name

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

    #   ---   Export   ---
    def sql_write(self, database_path, table_name, start=0, end=0, n=10,
                  compress_img=True, include_pk=False, replace_table=False, show_progression=True, compress_format='png'):
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
            output = io.BytesIO()
            img = Image.fromarray(array.astype(dtype=np.uint8))
            img.save(output, format)
            contents = output.getvalue()
            return contents

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

    def folder_write(self, folder_path, start=0, end=0, compress_img=True, columns=None,
                     compress_format='png', metadata_format='csv',
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
        elif isinstance(columns, str) or isinstance(columns, DataSetColumn):
            columns = [columns]
            single_column = True
        if isinstance(columns, list):
            for c_id, c in enumerate(columns):
                if isinstance(columns, str):
                    if columns not in self.columns_name():
                        raise ValueError('Unknown column %s.' % columns)
                elif isinstance(columns, DataSetColumn):
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

    #   ---   Operations   ---
    def subset(self, start=0, end=None, name='subset', *args):
        start, end = interval(self.size, start, end, args)
        return DataSetSubset(self, start, end, name=name)

    def subgen(self, n=1, name='subgen'):
        return DataSetSubgen(self, n=n, name=name)

    def as_cache(self, n=1, start=0, end=None, name=None):
        start, end = interval(self.size, start, end)
        data = DataSetResult.create_empty(dataset=self, n=end - start, start_id=start, columns=self.columns_name())
        gen = self.generator(n=n, start=start)

        if name is None:
            label = self._name
            name = 'cache'
        else:
            label = name

        with Process('Caching %s' % label, end - start, verbose=False) as p:
            for i in range(0, end-start, n):
                l = min(n, end-start-i)
                gen.next(copy=data[i:i+l], limit=l)
                p.update(n)

        dataset = NumPyDataSet(name=name, **data)
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

    def shuffle(self, nb_subgenerator=0, parallel=True, rnd=None, name='shuffle'):
        from .datasets_core import DataSetShuffle
        return DataSetShuffle(self, nb_subgenerator=nb_subgenerator, parallel=parallel, rnd=rnd, name=name)

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
            elif isinstance(_, DataSetColumn):
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
                if isinstance(columns, DataSetColumn):
                    columns = columns.name
                patches_def[columns] = (columns, patch_shape)
            else:
                for c_id, c in enumerate(columns):
                    if isinstance(c, DataSetColumn):
                        c = c.name
                    patches_def[c] = (c, patch_shape)
        else:
            if not isinstance(patch_shape, dict):
                raise ValueError('If columns is not defined, patch_shape should be a dictionary of patch shapes.')
            for c, patch_def in patch_shape.items():
                if isinstance(c, DataSetColumn):
                    c = c.name
                parent_c = c
                if isinstance(patch_def, tuple) and not isinstance(patch_def[0], int):
                    parent_c = patch_def[0].name if isinstance(patch_def[0], DataSetColumn) else patch_def[0]
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
                if isinstance(columns, DataSetColumn):
                    columns = columns.name
                patches_def[columns] = (columns, patch_shape)
            for c_id, c in enumerate(columns):
                if isinstance(c, DataSetColumn):
                    c = c.name
                patches_def[c] = (c, patch_shape)
        else:
            if not isinstance(patch_shape, dict):
                raise ValueError('If columns is not defined, patch_shape should be a dictionary of patch shapes.')
            for c, patch_def in patch_shape.items():
                if isinstance(c, DataSetColumn):
                    c = c.name
                parent_c = c
                if isinstance(patch_def, tuple) and not isinstance(patch_def[0], int):
                    parent_c = patch_def[0].name if isinstance(patch_def[0], DataSetColumn) else patch_def[0]
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
class DataSetColumn:
    """
    Store information of a column of a DataSet
    """
    def __init__(self, name, shape, dtype, dataset=None):
        self._name = name
        self._shape = shape
        self._dataset = None
        if dataset is not None:
            self._dataset = weakref.ref(dataset)
        self._dtype = dtype
        if dtype == str:
            self._dtype = 'O'

    def __getstate__(self):
        return self._name, self._shape, self._dtype

    def __setstate__(self, state):
        self._name, self._shape, self._dtype = state

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
    def sql_type(self):
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
        self._dataset = None
        self._data_dict = data_dict
        self._columns = columns
        self._start_id = start_id
        self._size = size

        self.affiliate_dataset(dataset)
        self._trace = DataSetResult.Trace(self)

        self._ipywidget = None

    def __getstate__(self):
        return self._data_dict, self._columns, self._start_id, self.size, self._trace

    def __setstate__(self, state):
        self._data_dict, self._columns, self._start_id, self._size, self._trace = state

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
                if not isinstance(c, DataSetColumn):
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
        columns = [DataSetColumn(name=name, shape=d.shape[1:], dtype=d.dtype) for name, d in data.items()]

        n = None
        for d in data.values():
            if n is None:
                n = d.shape[0]
            elif n != d.shape[0]:
                raise ValueError("Error when creating DataSetResult from data: some columns don't share the same length.")

        if 'pk' not in [_.name for _ in columns]:
            columns.insert(0, DataSetColumn(name='pk', shape=(), dtype=np.uint16))
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
        for data in self._data_dict.values():
            if data.flags['OWNDATA']:
                del data

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
        elif isinstance(item, int):
            return (self[item, 'pk'],) + tuple([self[item, _.name] for _ in self.columns])
        elif isinstance(item, list):
            return [self[_] for _ in item]
        elif isinstance(item, tuple):
            def istypeof_or_listof(o, t):
                return isinstance(o, t) or (isinstance(o, list) and isinstance(o[0], t))
            if (isinstance(item[0], slice) or istypeof_or_listof(item[0], int))and (istypeof_or_listof(item[1], str) or istypeof_or_listof(item[1], list)):
                indexes = item[0]
                columns = item[1]
            else:
                raise NotImplementedError('First index should be row index and second should be columns index\n'
                                          '(here provided type is [%s, %s])' % (str(type(item[0])), str(type(item[1]))))

            if isinstance(columns, list):
                return [self._data_dict[_][indexes] for _ in columns]
            else:
                return self._data_dict[columns][indexes]
        elif isinstance(item, slice):
            return {c: self[item, c] for c in self.columns_name()}
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
            self._dataset_name = r.dataset.dataset_name[:]
            self._parent_traces = {}

            self._start_date = time.time()
            self._end_date = None
            self._n = r.dataset.size

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
        def __init__(self, n, id, end_id, columns, dataset, determinist, parallel):
            self.start_n = n
            self.start_id = id
            self.end_id = end_id
            self.columns = columns
            self.dataset = dataset
            self.determinist = determinist
            self.parallel = parallel

            self._copy = {}
            self._r = None
            self.n = n
            self.id = id

        def generator(self, dataset, start=None, end=None, n=None, columns=None, parallel=False):
            if start is None:
                start = self.id
            if end is None:
                end = min(self.end_id, dataset.size)
            if n is None:
                n = self.n
            if columns is None:
                columns = self.columns

            gen = dataset.generator(start=start, end=end, n=n, columns=columns,
                                    determinist=self.determinist, parallel_exec=parallel)

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

    def __init__(self, dataset: AbstractDataSet, n, start_id, end_id, columns=None, determinist=True, parallel_exec=False):
        if columns is None:
            columns = dataset.columns_name()
        else:
            columns = dataset.interpret_columns(columns)

        if parallel_exec and isinstance(parallel_exec, bool):
            parallel_exec = 'process'

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

        self._context = DataSetSmartGenerator.Context(n=n, id=start_id, end_id=end_id, columns=columns,
                                                      dataset=dataset,
                                                      determinist=determinist,
                                                      parallel=parallel_exec)

        self._generator_setup = False
        self._generator = None

        self._generator_queue = None
        self._generator_process = None
        self.__warn_limit_copy = False

    def __del__(self):
        self.clean()

    def __next__(self):
        return self.next()

    def next(self, copy=None, limit=None, r=None):
        affiliated_result = r
        r = None

        if self.ended():
            raise StopIteration
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
            r = next(self._generator)
            if isinstance(r, weakref.ref):
                r = r()
            r.trace.computation_ended()

            # Clean context
            self._context._copy = None
            self._context._r = None
            self.context.id += self.n

            if affiliated_result:
                affiliated_result.trace.affiliate_parent_trace(r.trace)

            if self.ended():
                self.clean()

            return r

    def poll(self, timeout=0, copy=None, r=None):
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

        if timeout == 0:
            try:
                r = self._generator_queue.get_nowait()
            except queue.Empty:
                return None
        else:
            try:
                r = self._generator_queue.get(block=True, timeout=timeout)
            except queue.Empty:
                return None

        if r is None:
            self._context.id = self.end_id
            raise StopIteration

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

    def __iter__(self):
        while True:
            yield self.next()

    def setup(self):
        if self._generator_setup:
            return
        if self._context.determinist:
            self.dataset._setup_determinist()
        self._generator_setup = True

        if self.parallel_exec:
            self._generator_queue = mp.Queue(1)

            if self.parallel_exec=='process':
                process = mp.Process(target=parallel_generator_exec, args=(self,),
                                                     name='%s.generator' % self.dataset.dataset_name)
            else:
                process = threading.Thread(target=parallel_generator_exec, args=(self,),
                                                           name='%s.generator' % self.dataset.dataset_name)

            process.start()

            self._generator_process = process
        else:
            self.init_generator()

    def init_generator(self):
        self._generator = self.dataset._generator(self._context)

    def clean(self):
        self._generator = None
        self._context._r = None
        if self._generator_process:
            if self._generator_process.is_alive():
                self._generator_process.terminate()
            self._generator_process = None
        if self._generator_queue:
            del self._generator_queue
            self._generator_queue = None

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
    def parallel_exec(self):
        return self._context.parallel

    @property
    def _asynchronous_exec(self):
        # When a generator is executed in parallel, an exact copy of it is pickled and is executed synchronously in an
        # other thread or process. This property will return True for the original generator and False for its copy.
        return self._generator_process is not None

    @property
    def context(self):
        return self._context


def parallel_generator_exec(smart_generator):
    try:
        # Setup generator
        smart_generator.init_generator()
        queue = smart_generator._generator_queue

        # Run generator
        for r in smart_generator:
            # Sending the result in the queue
            queue.put(r)
    except Exception as e:
        queue.put(e)

    # Sending end notification
    queue.put(None)
    queue.close()


########################################################################################################################

########################################################################################################################
class NumPyDataSet(AbstractDataSet):
    def __init__(self, name='NumPyDataSet', pk=None, **kwargs):
        super(NumPyDataSet, self).__init__(name=name, pk_type=np.int32 if pk is None else pk.dtype)

        size = -1
        for name, array in kwargs.items():
            if size != -1:
                assert size == array.shape[0]
            else:
                size = array.shape[0]
            self.add_column(name, array.shape[1:], array.dtype)

        self._size = size
        self._offset = 0
        self.data = kwargs
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
class DataSetSubset(AbstractDataSet):
    def __init__(self, dataset, start=0, end=None, name='subset'):
        """
        :type dataset: AbstractDataSets
        """
        super(DataSetSubset, self).__init__(name, dataset, pk_type=dataset.pk.dtype)
        self._columns = dataset.copy_columns(self)
        self.start, self.end = interval(dataset.size, start, end)

    def _generator(self, gen_context):
        first_id = gen_context.start_id + self.start
        last_id = gen_context.end_id + self.start
        gen = gen_context.generator(self._parent, start=first_id, end=last_id)
        while not gen_context.ended():
            i, n, r = gen_context.create_result()
            yield gen.next(copy={c: r()[c] for c in gen_context.columns})

    @property
    def size(self):
        return self.end - self.start

    def subset(self, start=0, end=None, *args):
        if len(args) == 1:
            start = 0
            end = args[0]
        elif len(args) == 2:
            start = args[0]
            end = args[1]
        return DataSetSubset(self._parent, start+self.start, min(self.start+end, self.end))


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
