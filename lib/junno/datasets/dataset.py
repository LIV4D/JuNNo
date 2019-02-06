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
                return self.read_one(row=item[0], columns=item[1], extract=True)
            elif isinstance(item[0], slice):
                return self.read(item[0].start, item[0].stop, columns=item[1], extract=True)
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
    def read(self, start: int = None, stop: int = None, columns=None, extract=False, n=None):
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

        gen = d.generator(stop - start, start=start, columns=columns)
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

    def read_one(self, row=0, columns=None, extract=False):
        """
        Read a specific element of a dataset. If extract is True, the result will depend on the form of columns.
        Thus, if columns is None, read_one(i) will return a dictionnary a the i-th value of all the dataset's columns,
              if columns is a list, read_one will organize the values in a list in the same order as in columns
        finally if columns is a string, only the element at this position and column will be returned.
        :param row: Row of the wanted elements
        :param columns: Columns of the wanted elements (None mean all of them)
        :param extract: If true, the data is extracted from the DataSetResult.
        """
        r = self.read(start=row, stop=row + 1, columns=columns, extract=False)

        if not extract:
            return r
        if columns is None:
            return {c_name: c_data[0] for c_name, c_data in r.items()}
        elif isinstance(columns, list):
            return [r[_][0] for _ in self.interpret_columns(columns)]
        else:
            return r[columns][0]

    def at(self, row, columns=None):
        return self.read_one(row=row, columns=columns, extract=False)

    #   ---   Generators   ---
    def generator(self, n=1, start=None, stop=None, columns=None, determinist=True, intime=False, ncore=0):
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
        elif isinstance(columns, (tuple, set)):
            columns = list(columns)
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

    def folder_write(self, folder_path, start=0, stop=0, columns=None, determinist=True,
                     compress_format='png', metadata_format='csv', compress_img=True,
                     column_for_filename=None):

        import pandas as pd

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
        if stop <= 0:
            stop += self.size
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
        with Process('Saving %s' % self._name, stop - start) as p:
            for i in range(start, stop):
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

    def export_files(self, path, start=0, stop=None, columns=None, filename_column=None, metadata_file='.xlsx',
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
                    self._gen = self._dataset.generator(ncore=ncore, mp=mp)
                try:
                    return self._gen.next()
                except StopIteration:
                    raise IndexError

        return CustomDataLoader(self)

    def as_cache(self, n=1, start=0, stop=None, columns=None, ncore=1, name=None):
        start, stop = interval(self.size, start, stop)
        if columns is None:
            columns = self.columns_name()

        from .dataset_generator import DataSetResult
        data = DataSetResult.create_empty(dataset=self, n=stop - start, start_id=start, columns=columns)

        if name is None:
            label = self._name
            name = 'cache'
        else:
            label = name

        with Process('Caching %s' % label, stop - start, verbose=False) as p:
            def write_back(r):
                data[r.start_id-start:r.stop_id-start] = r
                p.update(r.size)
            self.export(write_back, n=n, start=start, stop=stop, columns=columns, ncore=ncore)

        from .datasets_core import NumPyDataSet
        dataset = NumPyDataSet(data, name=name)
        dataset._parents = [self]
        return dataset

    def cache(self, start=0, stop=None, columns=None, ncore=1, ondisk=None, name=None,
              overwrite='auto'):
        import tables
        from .dataset_generator import DataSetResult
        from collections import OrderedDict
        from ..j_utils.path import open_pytable

        start, stop = interval(self.size, start, stop)
        if columns is None:
            columns = self.columns_name()

        if name is None:
            label = self._name
            name = 'cache'
        else:
            label = name

        hdf_t = None
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

                hdf_f = open_pytable(path)
                if not table_name.startswith('/'):
                    table_name = '/' + table_name
                table_path, table_name = table_name.rsplit('/', maxsplit=1)
                if not table_path:
                    table_path = '/'
            else:
                raise ValueError('cache_path should be formated as "PATH:TABLE_NAME"')

            try:
                hdf_t = hdf_f.get_node(table_path, table_name, 'Table')

                erase_table = False
                if overwrite and isinstance(overwrite, bool):
                    erase_table = True
                elif overwrite == 'auto':
                    if hdf_t.nrows != self.size:
                        erase_table = True
                    else:
                        for col_name, hdf_col in hdf_t.description._v_colobjects.items():
                            col = self.column_by_name(col_name, raise_on_unknown=False)
                            if col is None or hdf_col.dtype.shape != col.shape or hdf_col.dtype.base != col.dtype:
                                erase_table = True
                                break
                if erase_table:
                    hdf_f.remove_node(table_path, table_name)
                    hdf_t = None
            except tables.NoSuchNodeError:
                pass
        else:
            hdf_f = tables.open_file("/tmp/empty.h5", "a", driver="H5FD_CORE", driver_core_backing_store=0)
            table_path = '/'
            table_name = 'dataset'

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
                                     createparents=True, track_times=False)
            chunck_size = min(stop - start, hdf_t.chunkshape[0])

            if ncore > 1:
                with Process('Allocating %s' % label, stop-start, verbose=False) as p:
                    empty = DataSetResult.create_empty(dataset=self, n=1).to_row_list()[0]

                    for i in range(0, stop-start):
                        hdf_t.append([empty]*min(chunck_size, stop-i))
                        p.step = i

                with Process('Caching %s' % label, stop-start, verbose=False) as p:
                    from .dataset_generator import DataSetResult
                    def write_back(r: DataSetResult):
                        hdf_t.modify_rows(start=r.start_id-start, stop=r.stop_id-start, rows=r.to_row_list())
                        p.update(r.size)
                    self.export(write_back, n=chunck_size, start=start, stop=stop, columns=columns, ncore=ncore)
            else:
                with Process('Caching %s' % label, stop-start, verbose=False) as p:
                    for r in self.generator(n=chunck_size, start=start, stop=stop, determinist=True, columns=columns):
                        hdf_t.append(r.to_row_list())
                        p.update(r.size)

        from .datasets_core import PyTableDataSet
        hdfDataset = PyTableDataSet(hdf_t, name=name)
        for c in columns:
            hdfDataset.col[c].format = self.col[c].format
        return hdfDataset

    #   --- Global operator ---
    def sum(self, columns=None, start=0, stop=None, ncore=1, n=1, determinist=True):
        single_column = isinstance(columns, (str, DSColumn))
        columns = self.interpret_columns(columns)
        for c in columns:
            c = self.column_by_name(c)
            if not np.issubdtype(c.dtype, np.number):
                raise ValueError('Only numeric columns can be summed. (%s is not numeric, dtype: %s).'
                                 % (c.name, c.dtype))

        from .dataset_generator import DataSetResult
        result = DataSetResult.create_empty(n=1, dataset=self, columns=columns)

        def write_cb(r):
            for c in r.keys():
                result[0, 0, c] += r[:, c].sum(axis=0)

        self.export(write_cb, n=n, start=start, stop=stop, ncore=ncore, determinist=determinist)
        return result[columns[0]] if single_column else result

    def mean(self, columns=None, start=0, stop=None, std=False, ncore=1, n=1, determinist=True):
        single_column = isinstance(columns, (str, DSColumn))
        columns = self.interpret_columns(columns)
        for c in columns:
            c = self.column_by_name(c)
            if not np.issubdtype(c.dtype, np.number):
                raise ValueError('Only numeric columns can be averaged. (%s is not numeric, dtype: %s).'
                                 % (c.name, c.dtype))

        start, stop = interval(self.size, start, stop)
        from .dataset_generator import DataSetResult
        result = DataSetResult.create_empty(n=2 if std else 1, dataset=self, columns=columns)

        if std:
            def write_cb(r):
                for c in columns:
                    result[0, c] += r[:, c].sum(axis=0) / (stop - start)
                    result[1, c] += np.square(r[:, c]).sum(axis=0) / (stop - start)
        else:
            def write_cb(r):
                for c in columns:
                    result[0, c] += r[:, c].sum(axis=0)/(stop-start)

        self.export(write_cb, columns=columns, n=n, start=start, stop=stop, ncore=ncore, determinist=determinist)

        if std:
            for c in result.keys():
                result[1, c] = np.sqrt(result[1, c]-np.square(result[1, c]))
            return result[:, columns[0]] if single_column else result
        return result[0, columns[0]] if single_column else result

    def std(self, columns=None, start=0, stop=None, ncore=1, n=1, determinist=True):
        single_column = isinstance(columns, (str, DSColumn))
        columns = self.interpret_columns(columns)
        result = self.mean(columns=columns, start=start, stop=stop, ncore=ncore, n=n, determinist=determinist, std=True)
        return result[1, columns[0]] if single_column else result.truncate(start=1)

    def confusion_matrix(self, pred, true, weight=None, label=None, rowwise=False, start=0, stop=None, ncore=1, n=1, determinist=True):
        from sklearn.metrics import confusion_matrix

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

        conf_labels = label
        n_class = len(conf_labels)

        if rowwise:
            confmat_name = pred.name+'_confmat'
            kwargs = dict(name=confmat_name,
                          cols_format=(np.uint32, (n_class, n_class),
                                       DSColumnFormat.ConfMatrix(n_class)) )

            def conf_mat(pred, true, weight):
                if one_hot:
                    pred = np.argmax(pred, axis=0)
                return confusion_matrix(y_pred=pred.flatten(), y_true=true.flatten(), sample_weight=weight,
                                        labels=conf_labels)
            if isinstance(true, DSColumn):
                if isinstance(weight, DSColumn):
                    return self.apply({confmat_name, (pred, true, weight)}, conf_mat, **kwargs)
                else:
                    return self.apply({confmat_name, (pred, true)}, **kwargs,
                                      function=lambda pred, true: conf_mat(pred, true, weight))
            else:
                if isinstance(weight, DSColumn):
                    return self.apply({confmat_name, (pred, weight)}, **kwargs,
                                      function=lambda pred, weight: conf_mat(pred, true, weight))
                else:
                    return self.apply({confmat_name, (pred)}, **kwargs,
                                      function=lambda pred: conf_mat(pred, true, weight))
        else:
            confmat = np.zeros((n_class, n_class), dtype=np.uint)

            with Process('Confustion Matrix: %s' % label, stop - start, verbose=False) as p:
                def write_cb(r):
                    if one_hot:
                        y_pred = np.argmax(r[pred.name], axis=1).flatten()
                    else:
                        y_pred = r[pred.name].flatten()

                    if isinstance(true, DSColumn):
                        y_true = r[true.name].flatten()
                    else:
                        y_true = true.flatten()

                    if isinstance(weight, DSColumn):
                        sample_weight = r[weight.name].flatten()
                    else:
                        sample_weight = weight.flatten()

                    confmat[:] += confusion_matrix(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
                                                   labels=conf_labels)

                    p.update(r.size)

                columns = [pred.name]
                if isinstance(true, DSColumn):
                    columns.append(true.name)
                if isinstance(weight, DSColumn):
                    columns.append(weight.name)

                self.export(write_cb, n=n, start=start, stop=stop, columns=columns, determinist=determinist,
                            ncore=ncore)

            return confmat

    #   ---   Operations   ---
    @classmethod
    def operation(cls, func):
        if hasattr(cls, func.__name__):
            raise AttributeError('%s method name already exist in AbstractDataset.' % func.__name__)
        setattr(cls, func.__name__, func)

    def subset(self, start=0, stop=None, name='subset', *args):
        from .datasets_core import DataSetSubset
        start, stop = interval(self.size, start, stop, args)
        return DataSetSubset(self, start, stop, name=name)

    def subgen(self, n=1, name='subgen'):
        from .datasets_core import DataSetSubgen
        return DataSetSubgen(self, n=n, name=name)

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

    def apply(self, columns, function, cols_format=None, format=None, n_factor=None, batchwise=False, keep_parent=False,
              name=None):
        if name is None:
            name = getattr(function, '__name__', 'apply')
            if name == '<lambda>':
                name = "apply"
        if n_factor is None and cols_format == 'auto':
            n_factor = 1
        from .datasets_core import DataSetApply
        return DataSetApply(self, function=function, columns=columns, name=name, format=format, n_factor=n_factor,
                            remove_parent_columns=not keep_parent, cols_format=cols_format, batchwise=batchwise)

    def cv_apply(self, columns, function, cols_format=None, n_factor=1, keep_parent=False, name=None):
        if name is None:
            name = getattr(function, '__name__', 'apply')
            if name == '<lambda>':
                name = "apply"
        from .datasets_core import DataSetApplyCV
        return DataSetApplyCV(self, function=function, columns=columns, name=name, n_factor=n_factor,
                              cols_format=cols_format, remove_parent_columns=not keep_parent)

    def apply_map_values(self, columns, mapping, default=None, sampling=None, name='map_value'):
        from .datasets_core import DataSetApply
        from ..j_utils.image import prepare_lut
        if default is None:
            default = mapping.pop('default', None)
        f_lut = prepare_lut(mapping, default=default, sampling=sampling)
        return DataSetApply(self, function=f_lut, columns=columns, name=name,
                            cols_format=None, batchwise=True, n_factor=1)

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
        c_tmp = []
        if isinstance(columns, str):
            columns = [columns]
        for c_id, c in enumerate(columns):
            if isinstance(c, str):
                if c not in self.columns_name():
                    raise ValueError('%s is not a column of %s' % (c, self.dataset_name))
                c_tmp.append(self.column_by_name(c))
        columns = c_tmp

        img_shape = columns[0].shape[-2:]
        for c in columns[1:]:
            if c.shape[-2:] != img_shape:
                raise ValueError('All data-augmented columns should have the same shape!')

        da_engine = DataAugmentation(**dict_params)

        from .data_augmentation import DataSetAugmentedData
        return DataSetAugmentedData(self, columns=columns, n=N_augmented,
                                    da_engine=da_engine, keep_original=keep_original, column_transform=column_transform)

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
            proba_map = proba_map / proba_map.sum()
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

    def unpatch(self, columns=None, patch_mix='replace', restore_columns=None, columns_shape=None, n_patches=None):
        from .datasets_core2d import DataSetUnPatch
        return DataSetUnPatch(self, patch_mix=patch_mix, columns=columns, n_patches=n_patches,
                              restore_columns=restore_columns, columns_shape=columns_shape)


########################################################################################################################
class DSColumn:
    """
    Store information of a column of a DataSet
    """
    def __init__(self, name, shape, dtype, dataset=None, format=None):
        self._name = name
        self._shape = shape
        self._is_text = False
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
        return self._is_text

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, f):
        if self._dtype is None or self._shape is None:
            self._format = f
        else:
            self._format = DSColumnFormat.auto_format(self._dtype if not self._is_text else 'str', self._shape, f)

    def __repr__(self):
        return '%s: %s %s' % (self.name, str(self.shape), str(self.dtype))


class DSColumnFormat:
    class Base:
        def __init__(self, dtype, shape, is_label=False):
            self.__dtype = dtype
            self.__shape = shape
            self._is_label = is_label
            self.check_type(dtype, shape)
            self.html_fullscreen = False

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
            if 'int' in str(self.dtype):
                return 'INTEGER'
            elif 'float' in str(self.dtype):
                return 'FLOAT'
            elif str(self.dtype) in ('str', 'U') or self.dtype == np.str_:
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
            return self.format_html(self.preformat(data), data, fullscreen=fullscreen)

        def export_data(self, data):
            return self.format_data(self.preformat(data))

        def export_file(self, data, path, filename, overwrite=True):
            return self.write_file(self.preformat(data), path, filename, overwrite)

    class Number(Base):
        def __init__(self, dtype, shape):
            super(DSColumnFormat.Number, self).__init__(dtype, shape)

        def check_type(self, dtype, shape):
            if not np.issubdtype(dtype, np.number):
                raise ValueError('Number format must be applied to number columns.')
            if len(shape):
                raise ValueError('Number format can only be applied to columns with an empty shape.')
            return True

        def export_html(self, data, fullscreen=None):
            return ("<p>%i</p>" if 'int' in str(self.dtype) else "<p>%f.3</p>") % data

    class Text(Base):
        def __init__(self, dtype, shape):
            super(DSColumnFormat.Text, self).__init__(dtype, shape)

        def check_type(self, dtype, shape):
            if len(shape):
                raise ValueError('String format can only be applied to columns with an empty shape.')
            return True

    class Label(Base):
        def __init__(self, dtype, shape, mapping=None, default=None):
            super(DSColumnFormat.Label, self).__init__(dtype, shape, is_label=True)
            self.mapping = if_none(mapping, dict())
            self.default = if_none(default, mapping['default'] if 'default' in mapping else None)

        def format_html(self, data, raw_data, fullscreen=None):
            try:
                d = self.mapping[data]
            except KeyError:
                d = self.default
            return "<p> <i> %s </i> </p>" % str(d)

        def __repr__(self):
            return 'DSColumnFormat.Label(%s, %s):\n %s' % (str(self.dtype), str(self.shape), repr(self.mapping))

    class Matrix(Base):
        def __init__(self, dtype, shape, is_label=False):
            super(DSColumnFormat.Matrix, self).__init__(dtype, shape, is_label=is_label)
            self._domain = Interval()
            self._range = Interval()
            self._clip = Interval()

        def check_type(self, dtype, shape):
            if not np.issubdtype(dtype, np.number):
                raise ValueError('Matrix format must be applied to number columns (not %s).' % repr(dtype))
            if not len(shape):
                raise ValueError('Matrix format can only be applied to columns with a non empty shape.')
            return True

        def _preformat(self, data):
            d = apply_scale(data, self.range, self.domain, self.clip)
            return d

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
            return 'DSColumnFormat.LabelMatrix(%s, %s):\n %s' % (str(self.dtype), str(self.shape), repr(self.mapping))

        @property
        def mapping(self):
            return self._mapping

        @mapping.setter
        def mapping(self, m):
            if not m:
                self._mapping = m
                self._lut = None
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

    class ConfMatrix(Matrix):
        def __init__(self, n_class):
            super(DSColumnFormat.ConfMatrix, self).__init__(np.uint32, (n_class, n_class))

    class Image(Matrix):
        def __init__(self, dtype, shape, is_label=False):
            from ..j_utils.math import dimensional_split
            super(DSColumnFormat.Image, self).__init__(dtype, shape, is_label=is_label)
            self.html_fullscreen = True
            self.clip = 0, 255
            self.range = 0, 255
            if dtype == np.uint8:
                self.domain = 0, 255
            else:
                self.domain = 0, 1.0

            self.html_height = lambda h: int(np.round(256*(1-np.exp(-h/128))))
            self.html_columns = lambda n: dimensional_split(n)[1]

        def check_type(self, dtype, shape):
            if not np.issubdtype(dtype, np.number):
                raise ValueError('Image format must be applied to number columns (not %s).' % repr(dtype))
            if len(shape) not in (2, 3):
                raise ValueError('Image format can only be applied to columns with a non empty shape.')

        @property
        def channels_count(self):
            return int(np.prod(self.shape[:-2]))

        def _preformat(self, data):
            if data.ndim > 3:
                return data.reshape((self.channels_count,)+data.shape[-2:])
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

                thumbnail = cv2.resize(d, thumbnail_size, interpolation=cv2.INTER_AREA)
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

        def _preformat(self, data):
            if len(self.shape) == 2:
                data = data[0]
            if self._lut is not None:
                data = self._lut(data)
            if data.ndim == 2:
                return data.reshape((1,)+data.shape)
            return data

        def __repr__(self):
            return 'DSColumnFormat.LabelImage(%s, %s):\n %s' % (str(self.dtype), str(self.shape), repr(self.mapping))

        @property
        def mapping(self):
            return self._mapping

        @mapping.setter
        def mapping(self, m):
            if not m:
                self._mapping = m
                self._lut = None
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

    @staticmethod
    def auto_format(dtype, shape, info=None):
        if isinstance(info, DSColumnFormat.Base):
            return info.copy(dtype, shape)
        if shape == ():
            if 'int' in repr(dtype):
                if isinstance(info, dict):
                    return DSColumnFormat.Label(dtype, shape, mapping=info)
                else:
                    return DSColumnFormat.Number(dtype, shape)
            elif 'float' in repr(dtype):
                return DSColumnFormat.Number(dtype, shape)
            elif dtype == 'str':
                return DSColumnFormat.Text(dtype, shape)
            elif dtype == 'O':
                return DSColumnFormat.Text(dtype, shape)
        elif 'int' in repr(dtype) or 'float' in repr(dtype) or dtype in ('bool',):
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
            else:
                if is_img:
                    return DSColumnFormat.Image(dtype, shape)
                else:
                    return DSColumnFormat.Matrix(dtype, shape)
        return DSColumnFormat.Base(dtype, shape)
