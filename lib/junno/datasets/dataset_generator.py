from copy import copy
import multiprocessing as mp
import numpy as np
import traceback
import threading
import time
import weakref

from ..j_utils.parallelism import N_CORE
from ..j_utils.collections import istypeof_or_listof, istypeof_or_collectionof, if_none, AttributeDict
from ..j_utils.j_log import log, float_to_str

from .dataset import AbstractDataSet, DSColumn


########################################################################################################################
class DataSetResult:
    """
    Store the result of an iteration of a generator from DataSet
    """

    def __init__(self, data_dict, columns, start_id, size, dataset=None, undef_dims=None):
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
        self._undef_dims = undef_dims
        self._trace = DataSetResult.Trace(self)

        self._ipywidget = None

    def __copy__(self):
        r = self.create_empty(n=self.size, start_id=self.start_id, columns=self._columns, dataset=self.dataset)
        r[:] = self
        return r

    def copy(self):
        return self.__copy__()

    def __getstate__(self):
        return self._data_dict, self._columns, self._start_id, self._size, self._trace

    def __setstate__(self, state):
        self._data_dict, self._columns, self._start_id, self._size, self._trace = state
        self._dataset = None
        self._ipywidget = None

    @staticmethod
    def create_empty(n, start_id=0, columns=None, dataset=None, assign=None):
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
            if isinstance(c, str):
                if dataset is not None:
                    c = dataset.column_by_name(c)
                else:
                    raise ValueError('No dataset was specified, refering column by their name is disabled.')
            if c.name in assign:
                a = assign[c.name]
                if c.is_seq:
                    a_shape = (a.shape[0],) +tuple(a.shape[c.undef_dims+1:])
                else:
                    a_shape = a.shape
                if a_shape != (n,) + tuple(c.shape):
                    raise ValueError('The shape of the assigned value for column %s is %s but should be %s.'
                                     % (c.name, repr(a_shape), repr((n,)+c.shape)))
                data_dict[c.name] = a
            else:
                if c.is_seq:
                    new_dimensions = c.undef_dims*[1]
                    data_dict[c.name] = np.empty(tuple([n]+new_dimensions+list(c.shape)), dtype=c.dtype)
                else:
                    data_dict[c.name] = np.empty(tuple([n]+list(c.shape)), dtype=c.dtype)

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
            c_name = c.name + ';' + c.format.dtype_name
            if len(c.shape) > 0:
                c_name += ' [' + ('x'.join([str(_) for _ in c.shape])) + ']'
            columns_description.append(c_name)

        def retreive_data(row, col, fullscreen_id=None):
            c = self.columns[col]
            d = self[row, columns_name[col]]
            if fullscreen_id is None:
                return c.format.export_html(d), c.format.html_fullscreen
            else:
                return c.format.export_html(d, fullscreen_id)

        w.columns_name = '|'.join(columns_description)
        w.retreive_data = retreive_data
        w.retreive_fullscreen = retreive_data
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
    def stop_id(self):
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
        a = AttributeDict()
        for c in self._columns:
            a[c.name] = c
        return a

    @property
    def col(self):
        return self.columns

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
            if (isinstance(key[0], slice) or istypeof_or_listof(key[0], int)) and (
               isinstance(key[1], (str, DSColumn)) or istypeof_or_listof(key[1], list)):
                indexes = key[0]
                columns = key[1] if isinstance(key[1], str) else key[1].name
            else:
                raise NotImplementedError
            if columns in self:
                if isinstance(value, np.ndarray):
                    col = self.columns[columns]

                    if col.is_seq:
                        undef_dims = value.shape[:col.undef_dims]
                        previous_size = self._data_dict[columns].shape
                        shape = list(previous_size)
                        for i, dim in enumerate(undef_dims):
                            shape[i+1] = dim
                        self._data_dict[columns] = np.resize(self._data_dict[columns], shape)
                    np.copyto(self._data_dict[columns][indexes], value)
                else:
                    self._data_dict[columns][indexes] = value
            else:
                raise KeyError('Unknown column: %s' % columns)
        elif isinstance(key, str):
            self._data_dict[key][:] = value
        elif isinstance(key, DSColumn):
            self._data_dict[key.name][:] = value
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._data_dict[item]
        elif isinstance(item, DSColumn):
            return self._data_dict[item.name]
        elif isinstance(item, (int, slice)):
            return self[item, set(self.columns_name() + ['pk'])]
        elif isinstance(item, list):
            return [self[_] for _ in item]
        elif isinstance(item, tuple):
            if (isinstance(item[0], slice) or istypeof_or_listof(item[0], int)) and \
               istypeof_or_collectionof(item[1], (str, DSColumn), (list, tuple, set)):
                indexes = item[0]
                columns = item[1]
            else:
                raise NotImplementedError('First index should be row index and second should be columns index\n'
                                          '(here provided type is [%s, %s])' % (str(type(item[0])), str(type(item[1]))))

            if isinstance(columns, (list, tuple)):
                return [self._data_dict[_ if isinstance(_, str) else _.name][indexes] for _ in columns]
            elif isinstance(columns, set):
                return {c: self._data_dict[c if isinstance(c, str) else c.name][indexes] for c in columns}
            else:
                return self._data_dict[columns if isinstance(columns, str) else columns.name][indexes]
        else:
            raise NotImplementedError

    def __contains__(self, item):
        return item in self._data_dict

    def columns_name(self):
        """
        :rtype: List[str]
        """
        return [_ for _ in self.keys() if _ != 'pk']

    def column_by_name(self, name, raise_on_unknown=True):
        for c in self._columns:
            if c.name == name:
                return c
        if raise_on_unknown:
            raise (ValueError('Unknown column %s' % name))
        return None

    def keys(self):
        return [_.name for _ in self._columns]

    def items(self):
        for c_name in self.keys():
            yield c_name, self._data_dict[c_name]

    def truncate(self, *args, start=None, stop=None):
        from ..j_utils.math import interval
        start, stop = interval(self.size, start, stop, args)
        for _, data in self._data_dict.items():
            self._data_dict[_] = data[start:stop]
        self._size = stop-start
        return self

    def __iter__(self):
        for i in range(self.size):
            yield tuple(np.array(self._data_dict[_][i]) for _ in self.keys())

    def to_row_list(self):
        return list(self)

    def to_array_rec(self, hdf_compatible=False):
        array_list = []
        names = []
        dtypes = []
        for n, a in self._data_dict.items():
            names.append(n)
            array_list.append(a)

            col = self.column_by_name(n)
            dtypes.append(np.dtype(col.dtype, col.shape))

        return np.core.records.fromarrays(array_list)
    
    def to_torch(self, *args, device=None):
        import torch
        r = []

        if isinstance(device, torch.nn.Module):
            if hasattr(device, "device"):
                device = device.device
            else:
                device = "cpu"

        for c in args:
            if c not in self:
                raise ValueError("Unknown column %s" % str(c))
            t = torch.from_numpy(self[c])
            if device:
                t = t.to(device)
            r.append(t)

        return tuple(r)

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
        def __init__(self, start_id, stop_id, n, columns, dataset, determinist, ncore, parallelism):
            self.start_n = n
            self.start_id = start_id
            self.stop_id = stop_id
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
            return self.start_n, self.start_id, self.stop_id, self.columns, self.dataset, self.determinist, \
                   self.ncore, self.parallelism, self.n, self.id

        def __setstate__(self, state):
            self.start_n, self.start_id, self.stop_id, self.columns, self.dataset, self.determinist, \
            self.ncore, self.parallelism, self.n, self.id = state
            self._copy = {}
            self._r = None

        def generator(self, dataset, start=None, stop=None, n=None, columns=None, parallel=False, ncore=None):
            if start is None:
                start = self.id
            if stop is None:
                stop = min(self.stop_id, dataset.size)
            if n is None:
                n = self.n
            if columns is None:
                columns = self.columns
            if ncore is None:
                ncore = self.ncore

            gen = dataset.generator(start=start, stop=stop, n=n, columns=columns,
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
            return self.id + self.n >= self.stop_id

        def ended(self):
            return self.id >= self.stop_id

        def lite_copy(self):
            r = copy(self)
            r._r = None
            return r

        @property
        def id_length(self):
            return self.stop_id-self.start_id

    def __init__(self, dataset: AbstractDataSet, n, start_id, stop_id, columns=None, determinist=True,
                 ncore=None, intime=False):
        if columns is None:
            columns = dataset.columns_name()
        else:
            columns = dataset.interpret_columns(columns)

        if ncore is None:
            ncore=N_CORE

        if start_id is None:
            start_id = 0
        else:
            start_id %= dataset.size

        if stop_id is None:
            stop_id = dataset.size
        elif stop_id != 0:
            stop_id %= dataset.size
            if stop_id == 0:
                stop_id = dataset.size

        if not intime:
            intime = False
        elif isinstance(intime, bool):
            intime = 'process'
        elif intime not in ('process', 'thread'):
            raise ValueError('Invalid intime argument: %s \n(should be either "process" or "thread")'
                             % repr(intime))

        self._context = DataSetSmartGenerator.Context(start_id=int(start_id), stop_id=int(stop_id), n=int(n),
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
        self._context.n = min(self.stop_id - self.current_id, limit)

        if self._asynchronous_exec:
            if (limit is not None) and not self.__warn_limit_copy:
                log.warn("%s's generator is executed asynchronously, data sharing and size limit will be ignored."
                         % self.dataset.dataset_name)
                self.__warn_limit_copy = True

            return self.poll(timeout=None, copy=copy, r=affiliated_result)

        else:
            # Setup context
            self._context._copy = copy

            # Compute next sample
            r = self.next_core(self._generator, self._context)

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
            self._context.id = self.stop_id
            raise StopIteration

        if ask_next:
            self.ask()

        self._context.n = min(self.stop_id - self.current_id, self._context.start_n)

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

        # Reapply column format
        for c in r._columns:
            c.format = self.dataset.column_by_name(c.name).format

        return r

    def reset(self, start=None, stop=None, columns=None, n=None, determinist=None):
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
        if stop is None:
            stop = self.stop_id
        elif stop != 0:
            stop %= self.dataset.size
            if stop == 0:
                stop = self.dataset.size
        start = int(start); stop = int(stop); n = int(n)
        self._context.start_id = start

        if start == self.current_id and stop == self.stop_id and columns == self.columns and n == self.n \
                and determinist == self._context.determinist:
            return

        if self._asynchronous_exec:
            while self._generator_conn.poll(timeout=0):
                self._generator_conn.recv()     # Discard previous samples
            self._send_to_subprocess(dict(start=start, stop=stop, columns=columns, n=n, determinist=determinist))
        else:
            self.clean()

        self._context.stop_id = stop
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
            self.context.id = self.stop_id
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
    def stop_id(self):
        return self._context.stop_id

    @property
    def intime(self):
        return self._context.parallelism

    def __len__(self):
        return np.ceil((self.stop_id-self.current_id)/self.n)

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

    def reset_gen(start, stop, n, columns, determinist):
        new_context = SmartGen.Context(start_id=start, stop_id=stop, n=n,
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

