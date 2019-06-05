import pandas
import numpy as np
from copy import copy
from os.path import exists
from os import remove
from ..j_utils.string import str2time, time2str
from ..j_utils.path import format_filepath
from ..j_utils.threadsafe_pytables import open_pytable, tables
from ..j_utils.collections import OrderedDict, AttributeDict, df_empty, if_none
from ..j_utils.math import pad_forward_na
from ..datasets.dataset import DSColumnFormat


class History:
    """
    Store dataseries by iteration and epoch.
    Data are index through timestamp: the number of iteration since the first iteration of the first epoch.
    """
    def __init__(self, path, overwrite=False):
        from tables.nodes import filenode

        path = format_filepath(path, 'h5log', exists=False)

        if overwrite and exists(path):
            remove(path)
        self._hdf_file = open_pytable(path)

        # --- CREATE EPOCH_INFO Table ---
        try:
            self._epoch_info_hdf = self._hdf_file.get_node('/', 'epoch_info')
        except tables.NoSuchNodeError:
            class DefaultEpochInfoRow(tables.IsDescription):
                start_iter = tables.Int64Col()
                max_e_iter = tables.Int32Col()
            self._epoch_info_hdf = self._hdf_file.create_table('/', 'epoch_info', DefaultEpochInfoRow)
            r = self._epoch_info_hdf.row
            r['start_iter'] = 0
            r['max_e_iter'] = 0
            r.append()
            self._epoch_info_hdf.flush()

        self._epoch_info = np.empty((self._epoch_info_hdf.nrows, 2), dtype=np.int64)

        for i, r in enumerate(self._epoch_info_hdf.iterrows()):
            self._epoch_info[i] = (r['start_iter'], r['max_e_iter'])

        # --- CREATE TIMELINE ---
        try:
            self._timeline_hdf = self._hdf_file.get_node('/', 'timeline')
        except tables.NoSuchNodeError:
            class DefaultTimelineRow(tables.IsDescription):
                epoch = tables.Int32Col()
                e_iter = tables.Int32Col()
                date = tables.Time32Col()
                time = tables.Float32Col()
            self._timeline_hdf = self._hdf_file.create_table('/', 'timeline', DefaultTimelineRow, 'History Timeline')

        self._timeline = df_empty(columns=['epoch', 'e_iter', 'date', 'time'],
                                  dtypes=['int64', 'int64', 'datetime64[ns]', 'timedelta64[ns]'],
                                  index=pandas.Index([], dtype='int64'))

        n = len(self._timeline_hdf)
        step = 500
        for i0 in range(0, n, step):
            i1 = min(i0+step, n)
            if i1 > i0:
                r = self._timeline_hdf.read(i0, i1)
                df = pandas.DataFrame(index=self._epoch_info[r['epoch']-1][:, 0]+r['e_iter']-1,
                                data=dict(epoch=r['epoch'], e_iter=r['e_iter'], date=pandas.to_datetime(r['date']*1e9),
                                          time=r['time']*1e9))
                self._timeline = self._timeline.append(df)

        # --- CREATE EVENTS FILE ---
        try:
            events_node = self._hdf_file.get_node('/', 'events')
            self._events_file = filenode.open_node(events_node, 'a+')
        except tables.NoSuchNodeError:
            self._events_file = filenode.new_node(self._hdf_file, where='/', name='events')

        # --- CREATE KEYS  GROUP ---
        try:
            self.g_keys = self._hdf_file.get_node('/', 'KEYS')
        except tables.NoSuchNodeError:
            self.g_keys = self._hdf_file.create_group('/', 'KEYS')

        timeline_desc = self._timeline_hdf.description._v_colobjects
        self._keys = list(_[1:] for _ in timeline_desc.keys() if _.startswith('_'))

        # --- SETUP SERIES FORMAT ---
        self.empty_row = self._timeline_hdf.row
        self.format = {}
        for k in self._keys:
            self.empty_row['_' + k] = 0

            table = self._table_by_key(k)
            if table is not None:
                col = table.description._v_colobjects['value']
                self.format[k] = DSColumnFormat.auto_format(col.dtype.base, col.dtype.shape)

        # --- SET CURRENT TIMESTAMPS ---
        if len(self._timeline_hdf) > 0:
            last_row = self._timeline_hdf[-1]
            self._current_timestamp = TimeStamp.from_timeline_row(last_row)
            self._current_timeline_id = len(self._timeline_hdf) - 1
        else:
            self._current_timestamp = TimeStamp.now(0, 1)
            self._current_timeline_id = None

    def __del__(self):
        self._hdf_file.close()

    def add_keys(self, keys):
        if isinstance(keys, str):
            if keys in self.keys():
                return True
            keys = (keys,)
        else:
            keys = tuple(_ for _ in keys if _ not in self.keys())
            if not keys:
                return True

        import tables
        old_timeline = self._timeline_hdf
        old_timeline._f_rename('timeline_old', overwrite=True)

        timeline_desc = old_timeline.description._v_colobjects.copy()
        for k in keys:
            timeline_desc['_'+k] = tables.Int64Col()
            self._keys.append(k)
        new_timeline = tables.Table(old_timeline._v_parent, 'timeline', timeline_desc)

        # Copy the user attributes
        old_timeline.attrs._f_copy(new_timeline)

        # Fill the rows of new table with default values
        empty_row = new_timeline.row
        for k in keys:
            empty_row['_'+k] = np.int64(0)
        self.empty_row = empty_row
        for i in range(old_timeline.nrows):
            empty_row.append()
        new_timeline.flush()

        # Copy the columns of source table to destination
        for col in timeline_desc:
            if not col.startswith('_') or col[1:] not in keys:
                getattr(new_timeline.cols, col)[:] = getattr(old_timeline.cols, col)[:]

        self._timeline_hdf = new_timeline
        old_timeline.remove()
        return True

    def set_key(self, key, value, flush=True):
        import tables
        if key not in self.keys():
            self.add_keys(key)

        try:
            table = self._hdf_file.get_node(self.g_keys, key, 'Table')
        except tables.NoSuchNodeError:
            desc = OrderedDict()
            desc['timeline_id'] = tables.Int64Col(pos=0)
            if isinstance(value, np.ndarray):
                desc['value'] = tables.Col.from_sctype(value.dtype.type, value.shape, pos=1)
                self.format[key] = DSColumnFormat.auto_format(value.dtype, value.shape)
            else:
                v = np.array(value)
                self.format[key] = DSColumnFormat.auto_format(v.dtype, v.shape)
                desc['value'] = tables.Col.from_dtype(v.dtype, pos=1)
            table = self._hdf_file.create_table(self.g_keys, key, description=desc)

        if self._current_timeline_id is None:
            r = self.empty_row
            if self._timeline_hdf.nrows > 0:
                last_row = self._timeline_hdf[-1]
            else:
                last_row = {'_'+k: 0 for k in self.keys()}
            t = self._current_timestamp
            r['epoch'] = t.epoch
            r['e_iter'] = t.iteration
            r['time'] = t.time
            r['date'] = t.date
            for k in self.keys():
                if key == k:
                    r['_' + k] = table.nrows + 1
                else:
                    r['_' + k] = -abs(last_row['_'+k])
            r.append()

            self._timeline.loc[self._epoch_info[t.epoch-1,0] + t.iteration - 1] = [t.epoch, t.iteration,
                                                                                pandas.to_datetime(t.date * 1e9),
                                                                                t.time * 1e9]
            self._current_timeline_id = len(self._timeline) - 1
        else:
            for r in self._timeline_hdf.iterrows(start=self._current_timeline_id):
                r['_' + key] = table.nrows + 1
                r.update()

        table_row = table.row
        table_row['timeline_id'] = self._current_timeline_id
        table_row['value'] = value
        table_row.append()
        if flush:
            self._timeline_hdf.flush()
            table.flush()

    def _table_by_key(self, key):
        import tables
        if key not in self.keys():
            raise ValueError('Unknown key: %s.' % key)
        try:
            return self._hdf_file.get_node(self.g_keys, key, 'Table')
        except tables.NoSuchNodeError:
            return None

    # ---  Current Iteration  ---
    @property
    def timestamp(self):
        return self._current_timestamp

    @property
    def epoch(self):
        return self._current_timestamp.epoch

    @property
    def e_iter(self):
        return self._current_timestamp.iteration

    @property
    def iter(self):
        last_e_info = self._epoch_info[-1]
        return last_e_info['start_iter'] + self._current_timestamp.iteration

    @property
    def iteration(self):
        return self.e_iter

    def __len__(self):
        last_e_info = self._epoch_info[-1]
        return last_e_info[0] + last_e_info[1] + 1

    def step(self, iteration=0, time=None, epoch=0, date=None):
        if date is None:
            import time as python_time
            date = python_time.time()
        if epoch:
            self._epoch_info = np.concatenate((self._epoch_info,
                                         np.array([[self._epoch_info[-1, 0]+self._epoch_info[-1, 1], iteration+1]],
                                                  dtype=self._epoch_info.dtype)))
            r = self._epoch_info_hdf.row
            r['start_iter'] = self._epoch_info[-1, 0]
            r['max_e_iter'] = self._epoch_info[-1, 1]
            r.append()
        elif iteration > 0:
            self._epoch_info[-1][1] += iteration
            for r in self._epoch_info_hdf.iterrows(start=-1):
                r['max_e_iter'] = self._epoch_info[-1,1]
                r.update()
        self._epoch_info_hdf.flush()

        self._current_timeline_id = None
        self._current_timestamp = TimeStamp(len(self._epoch_info), self._epoch_info[-1,1], time, date)

    def step_iteration(self, time, date=None):
        self.step(iteration=1, time=time, date=date)

    def step_epoch(self, time, date=None):
        self.step(epoch=1, time=time, date=date)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError('History key should be a serie name not (%s, type:%s).'
                           % (str(key), type(key)))
        if type(value).__name__ == "Tensor":
            value = value.detach().cpu().numpy()
        self.set_key(key, value)

    # ---  Store/Read Data  ---
    def keys(self):
        return self._keys

    def series(self, only_number=False, no_std=False):
        keys = list(self.keys())
        if only_number:
            return [k for k in keys
                    if np.issubdtype(self.format[k].dtype, np.number)
                    and self.format[k].shape == ()]
        if no_std:
            keys = [k for k in keys if not (k.endswith('_std') and k[:-4] in keys)]
        return keys

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.keys():
                table = self._table_by_key(item)
                return table[-1]['value'] if table.nrows else None
            if ':' in item:
                item_split = item.split(':')
                if len(item_split) in (2, 3):
                    item = slice(*(_ if _ != "" else None for _ in item_split))
            else:
                item = self.interpret_timestamp(item)
        if isinstance(item, tuple):
            if len(item) != 2 or item[0] not in self.keys():
                raise KeyError("Invalid history index: %s\n"
                               "Index should follow the form: ['series name', time_index]" % repr(item))
            series = item[0]
            iterid = item[1]

            if isinstance(iterid, str):
                if ':' in iterid:
                    iterid_split = iterid.split(':')
                    if len(iterid_split) in (2, 3):
                        iterid = slice(*(_ if _ != "" else None for _ in iterid_split))
            if isinstance(iterid, slice):
                df = self.read(series=series, start=iterid.start, stop=iterid.stop, step=iterid.step,
                               interpolation='previous', averaged=True, std=False)
                return df[series].values
            else:
                return self.get(series=series, iterid=iterid, previous=True)
        elif isinstance(item, TimeStamp):
            return self.get(series=None, iterid=item)
        elif isinstance(item, slice):
            return self.read(start=item.start, stop=item.stop, step=item.step, interpolation='previous',
                             timestamp=['epoch', 'e_iter'])
        raise IndexError('Invalid index: unable to read from history series')

    def get(self, series, iterid=-1, previous=True, default='raise exception'):
        try:
            if series not in self.keys():
                raise KeyError('%s is an unknown serie name.' % series)
        except LookupError as e:
            if default != 'raise exception':
                return default
            raise e from None
        table = self._table_by_key(series)

        t = self.interpret_iterid(iterid)
        if t is None:
            if default != 'raise exception':
                return default
            raise IndexError('Invalid iterid: %s.' % str(iterid))

        if previous:
            t = self._timeline.index.searchsorted(t, side='right')
            if t == 0:
                return default if default != 'raise exception' else None
            t -= 1
        else:
            t = self._timeline.index.get_loc(t)

        series_id = self._timeline_hdf[t]['_'+series]
        if series_id == 0:
            return default if default != 'raise exception' else None

        if series_id < 0:
            if previous:
                series_id = -series_id
            else:
                if default != 'raise exception':
                    return default
                raise IndexError('No data for series %s at iterid %s. \n'
                                 '(You might consider set previous=True to get the last stored value.)'
                                 % (series, str(iterid)))

        return table[series_id-1]['value']

    def read(self, series=None, start=0, stop=0, step=1, timestamp=('epoch', 'e_iter'),
             interpolation='previous', smooth=None, averaged=True, std=False):
        """
        Read data from the specified range and return it as a pandas Dataframe. The data can be averaged over ``step``
        elements, interpolated, and smoothed (in this order). When averaged, the standard deviation may be computed.
        :param series: Names of the series to be read
        :type series: str or tuple or set
        :param start: timestamp from which data should be read
        :type start: int, TimeStamp, ...
        :param stop: timestamp until which data should be read
        :type stop: int, TimeStamp, ...
        :param step: Interval between to sample
        :type step: int, TimeStamp, ...
        :param timestamp: Additional timestamp related columns. Acceptable values are:
                    - epoch
                    - iteration
                    - time
                    - date
        :param interpolation: Specify which number series should be interpolated and how.
        NaN in number series can automatically be replaced by interpolated values using pandas interpolation algorithms.
        This parameter most be one of those:
            - True: All numbers series are interpolated linearly
            - False: No interpolation is applied (NaN are not replaced)
            - List of series name: The specified series are interpolated linearly
            - Dictionary associating an interpolation method to a series name.
            Valid interpolation method are: 'previous', 'nearest', 'linear', 'zero', 'slinear', 'quadratic', 'cubic'.
        :param smooth: Specify which number series should be smoothed and how much.
        Specified series are Savitzky-Golay filter of order 3.
        This parameter most be one of those:
            - None or False: Series are not smoothed (default).
            - True: All numbers series are smoothed with a window of 15.
            - List of series name: The specified series are smoothed with a window of 15.
            - Dictionary: associating series names with smoothing window size.
        :param averaged: Names of the time series whose values should be averaged along each step
        instead of being naively down-sampled. Can only be applied on number series.
        True means that all number series are be averaged and False means no series are.
        :param std: Names of the averaged time series whose standard deviation should be computed.
        A new columns is created for every of these time series with the name 'STD columnName'.
        :return: time series
        :rtype: pandas.DataFrame
        """
        if stop is None:
            stop = len(self)

        if step == 1:
            averaged = False

        indexes = np.array(list(self.iterate_iterid(start=start, stop=stop, step=step, last=True)), dtype=np.uint32)
        start_timeid = max(self._timeline.index.searchsorted(indexes[0], side='left')-1, 0)
        end_timeid = min(self._timeline.index.searchsorted(indexes[-1], side='right')+1, len(self._timeline)-1)
        if averaged:
            time_indexes = self._timeline.index.searchsorted(indexes, side='right') - 1
            intervals = []
            last_timeid = time_indexes[0]
            if last_timeid < 0:
                last_timeid = 0
            for timeid in time_indexes[1:]:
                if timeid < 0:
                    timeid = 0
                if last_timeid is not None and last_timeid != timeid:
                    intervals.append((last_timeid, timeid))
                    last_timeid = timeid
                else:
                    intervals.append(None)
        indexes = indexes[:-1]

        series_name = self.interpret_series_name(series)
        number_series_name = self.series(only_number=True)
        number_series_name = [_ for _ in series_name if _ in number_series_name]

        if isinstance(averaged, bool):
            averaged = number_series_name if averaged else []
        else:
            averaged = self.interpret_series_name(averaged, only_number=True)

        if isinstance(std, bool):
            std = averaged if std else []
        if isinstance(std, (list, tuple)):
            std = self.interpret_series_name(std, only_number=True)
            std = {_: _+'_std' if _+'_std' in self.series(only_number=True) else None for _ in std}
        if isinstance(std, dict):
            not_averaged_series = set(std).difference(averaged)
            if not_averaged_series:
                raise ValueError("Can't compute standard deviation of: %s.\n"
                                 "Those series are not averaged." % repr(not_averaged_series))
        else:
            raise ValueError('Invalid value for argument **std**: %s.\n '
                             'Valid types are bool, list/tuple, dict. Given is %s.' % (repr(std), type(std)))


        if not interpolation:
            interpolation = {}
        elif interpolation is True:
            interpolation = {_: 'linear' if _ in number_series_name else 'previous' for _ in series_name}
        elif isinstance(interpolation, str):
            if interpolation in number_series_name:
                interpolation = {interpolation: 'linear'}
            elif interpolation in series_name:
                interpolation = {interpolation: 'previous'}
            else:
                interpolation = {_: interpolation if _ in number_series_name else 'previous' for _ in series_name}
        elif isinstance(interpolation, (dict, OrderedDict)):
            unknown_keys = set(interpolation.keys()).difference(number_series_name)
            if unknown_keys:
                raise ValueError("Can't interpolate series: %s.\n"
                                 "Those series are either unknown or don't contain numbers!" % repr(unknown_keys))
        else:
            unknown_keys = set(interpolation).difference(number_series_name)
            if unknown_keys:
                raise ValueError("Can't interpolate series: %s.\n"
                                 "Those series are either unknown or don't contain numbers!" % repr(unknown_keys))
            interpolation = {_: 'linear' for _ in interpolation}

        if not smooth:
            smooth = {}
        elif isinstance(smooth, bool):
            smooth = {_: 15 for _ in number_series_name}
        elif isinstance(smooth, int):
            smooth = {_: smooth for _ in number_series_name}
        elif isinstance(smooth, str):
            if smooth not in number_series_name:
                raise ValueError("Can't smooth series %s. It is either unknown or doesn't contain number!"
                                 % smooth)
            smooth = {smooth: 15}
        elif isinstance(smooth, (dict, OrderedDict)):
            unknown_keys = set(smooth.keys()).difference(number_series_name)
            if unknown_keys:
                raise ValueError("Can't smooth series: %s.\n"
                                 "Those series are either unknown or don't contain numbers!" % repr(unknown_keys))
        else:
            unknown_keys = set(smooth).difference(number_series_name)
            if unknown_keys:
                raise ValueError("Can't smooth series: %s.\n"
                                 "Those series are either unknown or don't contain numbers!" % repr(unknown_keys))
            smooth = {_: 15 for _ in smooth}

        def interpolate(x, y, kind, start_i, end_i):
            import scipy.interpolate
            interpolator = scipy.interpolate.interp1d(x, y, kind=kind, copy=False, assume_sorted=True)
            r = np.empty(shape=(len(indexes),), dtype=y.dtype)
            r[:start_i] = np.nan
            r[start_i:end_i+1] = interpolator(indexes[start_i:end_i+1])
            r[end_i+1:] = np.nan
            return r

        def perform_smooth(x, factor):
            import  scipy.signal
            return scipy.signal.savgol_filter(x, factor, 3, mode='nearest')

        series = OrderedDict()
        mask_na = OrderedDict()

        for k in series_name:
            table = self._table_by_key(k)
            serie_std = None

            # Sample
            if k in averaged:
                # timeidx = np.empty(shape=(len(intervals),), dtype=np.int64)
                serie = np.empty(shape=(len(intervals),))
                mask = np.empty(shape=(len(intervals),), dtype=np.bool)
                mask.fill(True)

                table_std = None
                if k in std:
                    serie_std = np.empty(shape=(len(intervals),)) if k in std else None
                    if isinstance(std[k], str):
                        table_std = self._table_by_key(std[k])

                last_row_read = -1
                first_not_nan_i = None
                last_not_nan_i = None
                for i, interval in enumerate(intervals):
                    if interval and last_row_read < table.nrows:
                        start_id, end_id = interval
                        ids = table.get_where_list("(start_id <= timeline_id) & (timeline_id < end_id)",
                                                   condvars=dict(start_id=start_id, end_id=end_id))

                        if len(ids):
                            s = table.read_coordinates(ids)
                            last_row_read += len(s)
                            # timeidx[i] = self._timeline.index[s['timeline_id'][0]]
                            if s.shape[0]:
                                serie[i] = np.mean(s['value'])
                                if serie_std is not None:
                                    if table_std:  # Aggregate std
                                        std_ids = table_std.get_where_list(
                                            "(start_id <= timeline_id) & (timeline_id < end_id)",
                                            condvars=dict(start_id=start_id, end_id=end_id))
                                        if not len(std_ids):
                                            serie_std[i] = np.nan
                                            continue

                                        # Retreive std values and index
                                        s_std = table_std.read_coordinates(std_ids)
                                        s0_sigma = self._timeline2iter(s_std['timeline_id'])
                                        s0_sigma[1:] -= s0_sigma[:-1]
                                        sigma = s_std['value']
                                        if std_ids[0] > 0:
                                            s0_sigma[0] -= self._timeline2iter(table_std[std_ids[0]-1]['timeline_id'])

                                        # Retreive mean index
                                        s0_mu = self._timeline2iter(s['timeline_id'])
                                        s0_mu[1:] -= s0_mu[:-1]
                                        mu = s['value']
                                        if ids[0] > 0:
                                            s0_mu[0] -= self._timeline2iter(table[ids[0]-1]['timeline_id'])

                                        s1 = mu * s0_mu
                                        s2 = s0_sigma*sigma*sigma + s0_mu*mu*mu
                                        s0_mu = s0_mu.sum()
                                        s0_sigma = s0_sigma.sum()
                                        s1 = s1.sum()
                                        s2 = s2.sum()
                                        s0 = np.sqrt(s0_mu*s0_sigma)

                                        serie_std[i] = np.sqrt(s2/s0 - np.power(s1/s0_mu, 2))
                                    else:
                                        serie_std[i] = np.std(s['value'])
                                if first_not_nan_i is None:
                                    first_not_nan_i = i
                                last_not_nan_i = i
                                continue

                    serie[i] = np.nan
                    mask[i] = False
                    if serie_std is not None:
                        serie_std[i] = np.nan

                # Interpolate
                if k in interpolation and last_not_nan_i is not None:
                    if mask[0] or mask.argmax() != 0:
                        if interpolation[k] == 'previous':
                            pad_forward_na(serie, inplace=True, mask=mask)
                            if serie_std is not None:
                                pad_forward_na(serie_std, inplace=True, mask=mask)
                        else:
                            serie = interpolate(x=indexes[mask], y=serie[mask], kind=interpolation[k],
                                                start_i=first_not_nan_i, end_i=last_not_nan_i)
                            if serie_std is not None:
                                serie_std = interpolate(x=indexes[mask], y=serie_std[mask], kind=interpolation[k],
                                                        start_i=first_not_nan_i, end_i=last_not_nan_i)
                mask_na[k] = mask
                if serie_std is not None:
                    mask_na[k+'_std'] = mask

                # Smooth
                if k in smooth:
                    serie = perform_smooth(serie, smooth[k])
                    if serie_std is not None:
                        serie_std = perform_smooth(serie_std, smooth[k])

                series[k] = serie
                if serie_std is not None:
                    series[k+'_std'] = serie_std

            else:

                start_table = max(abs(self._timeline_hdf[start_timeid]['_'+k])-1, 0)
                end_table = max(abs(self._timeline_hdf[end_timeid]['_' + k]), 0)
                s = table.read(start=start_table, stop=end_table)

                s_id = self._timeline2iter(s['timeline_id'])
                s_value = s['value']

                interp_k = interpolation.get(k, None)
                if interp_k and len(s_id) < len(indexes):

                    first_not_nan_i = np.argmax(indexes >= s_id[0])
                    last_not_nan_i = np.argmax(indexes >= s_id[-1])

                    if interp_k != 'previous':
                        serie = interpolate(s_id, s_value, kind=interpolation[k],
                                            start_i=first_not_nan_i, end_i=last_not_nan_i)
                        mask_na[k] = np.ones(serie.shape[0], dtype=np.bool)
                    else:
                        import sortednp as snp
                        indexes_k = snp.merge(s_id, indexes.astype(s_id.dtype, copy=False))
                        # Remove doubles
                        indexes_k = indexes_k[np.concatenate(([True], indexes_k[1:] != indexes_k[:-1]))]

                        serie = np.empty(shape=(len(indexes_k),)+s_value.shape[1:], dtype=s_value.dtype)
                        serie.fill(np.nan)
                        mask = np.isin(indexes_k, s_id, assume_unique=True)
                        mask_in = np.isin(s_id, indexes_k, assume_unique=True)
                        serie[mask] = s_value[mask_in]

                        pad_forward_na(serie, inplace=True, mask=mask)
                        mask_na[k] = np.ones(serie.shape[0], dtype=np.bool)
                        serie = serie[np.isin(indexes_k, indexes, assume_unique=True)]
                else:
                    import sortednp as snp
                    indexes_k = snp.merge(s_id, indexes.astype(s_id.dtype, copy=False))
                    # Remove doubles
                    indexes_k = indexes_k[np.concatenate(([True], indexes_k[1:] != indexes_k[:-1]))]
                    serie = np.empty(shape=(len(indexes_k),) + s_value.shape[1:], dtype=s_value.dtype)
                    serie.fill(np.nan)
                    mask = np.isin(indexes_k, s_id, assume_unique=True)
                    mask_in = np.isin(s_id, indexes_k, assume_unique=True)
                    serie[mask] = s_value[mask_in]
                    mask_na[k] = mask

                if k in smooth:
                    serie = perform_smooth(serie, smooth[k])
                series[k] = serie

        df_series = OrderedDict()
        for k, v in series.items():
            if v.ndim > 1:
                m = mask_na.get(k, None)
                if m is None:
                    m = np.invert(np.isnan(v.sum(axis=tuple(range(1, v.ndim)))))
                df_series[k] = list(_v if _m else None for _v, _m in zip(v, m))
            else:
                df_series[k] = v

        df = pandas.DataFrame(index=indexes, data=df_series, copy=False)
        if timestamp:
            timestamp_df = self.export_timeline(timestamp, indexes)
            return pandas.concat([timestamp_df, df], axis=1)
        return df

    def plot(self, series=None, start=0, stop=None, step=None, interpolation='quadratic', smooth=None, averaged=True):
        from ..j_utils.math import vega_graph
        from IPython.display import display
        if series is None:
            series = self.series(only_number=True, )
            series = [s for s in series if not (s.endswith('_std') and s[:-4] in series)]

        if stop is None:
            stop = len(self)

        if step is None:
            step = (stop-start) // 300 + 1
            print(step)

        df = self.read(series=series, std=True, start=start, stop=stop, step=step, interpolation=interpolation,
                       smooth=smooth, averaged=averaged)
        df['id'] = df.index

        def is_percent(serie_name):
            return 'accuracy' in serie_name or 'kappa' in serie_name or\
                   'sensitivity' in serie_name or 'specificity' in serie_name

        map = {s: {'x': 'id', 'y': s, 'std': s+'_std', 'scale': 'y_ref' if is_percent(s) else None} for s in series}
        opt = {'scales':
                   [ {'name': "xepoch",
                      'domain': {'fields': [{'data': "table", 'field': "epoch"}]},
                      'range': "width"
                   }],
               'axes': [{'name': "XAxis", 'scale': "xepoch", 'title': "Epoch"}]
               }
        graph = vega_graph(df,  map, opt, shape=(800, 400))
        display(graph)

    #   --- Export ---
    def export_dataframe(self, series=None, start=0, stop=0, timestamp=None):
        """
        Export time series as a pandas DataFrame
        :param series: Name of the series to export. None means all series.
        :param start: Minimum time index of exported data
        :param stop: Maximum time index of exported data
        :param timestamp: Additional exported columns. Acceptable values are:
                    - epoch
                    - iteration
                    - time
                    - date
        :rtype: pandas.DataFrame
        """
        start = self.interpret_iterid(start)
        stop = self.interpret_iterid(stop, stop_index=True)
        series_name = self.interpret_series_name(series)

        series = []
        for k in series_name:
            serie = self._timeline_series[k].loc[start:stop]
            series.append(serie)

        df = pandas.DataFrame(series).transpose()

        if timestamp:
            timestamp_df = self.export_timeline(timestamp, df.index)
            df = pandas.concat([timestamp_df, df], axis=1)
        return df

    def export_csv(self, path, series=None, start=0, stop=0, timestamp=('epoch', 'iteration')):
        df = self.export_dataframe(series=series, start=start, stop=stop, timestamp=timestamp)
        df.to_csv(path_or_buf=path)

    def export_CURView(self, path, series=None, start=0, stop=0):
        def minibatch_count(e):
            return self._nb_iterations_by_epoch[e-1]

        df = self.export_dataframe(series=series, start=start, stop=stop, timestamp=['epoch', 'iteration'])
        mini_count = df['epoch'].map(minibatch_count)
        mini_count.name = 'number_of_minibatches'
        df = pandas.concat((df, mini_count), axis=1, copy=False)
        df.rename({'iteration': 'minibatch'}, axis='columns', inplace=True)

        df.to_csv(path_or_buf=path)

    # ---  Timestamp Conversion ---
    def epoch_to_iterid(self, epoch, iteration=1):
        # Check
        if epoch > self.epoch:
            raise IndexError('Invalid time stamp: %ie%i. (Current iteration is %ie%i)'
                             % (epoch, iteration, self.epoch, self.e_iter))
        if iteration > self._epoch_info[epoch-1,1]:
            raise IndexError('Invalid time stamp: %ie%i. (Epoch %i only has %i iterations)'
                             % (epoch, iteration, epoch, self._epoch_info[epoch-1,1]))
        # Sum
        return iteration + self._epoch_info[epoch-1,0] - 1

    def iterid_to_timestamp(self, time_id):
        if not 0 <= time_id < len(self):
            raise ValueError('%i is not a valid timestamp (min:0, max:%i)' % (time_id, len(self)-1))
        e = 0
        while e < self.epoch-1 and time_id > self._epoch_info[e+1,0]:
            e += 1
        i = time_id-self._epoch_info[e,0]

        time = self._timeline.loc[time_id, 'time']
        date = self._timeline.loc[time_id, 'date']

        return TimeStamp(epoch=e+1, iteration=i+1, time=time, date=date)

    def interpret_iterid(self, timestamp, stop_index=False):
        if isinstance(timestamp, TimeStamp):
            return self.epoch_to_iterid(epoch=timestamp.epoch, iteration=timestamp.iteration)
        if isinstance(timestamp, int):
            length = len(self)
            if not -length < timestamp < length+(1 if stop_index else 0):
                raise IndexError('%i is not a valid timestamp (min:-%i, max:%i)' % (timestamp, length, length+(1 if stop_index else 0)))
            if timestamp < 0:
                timestamp += length
            if timestamp == 0 and stop_index:
                timestamp = length
            return timestamp
        else:
            timestamp = TimeStamp.interpret(timestamp)
            return self.epoch_to_iterid(epoch=timestamp.epoch, iteration=timestamp.iteration)

    def interpret_timestamp(self, timestamp):
        return self.iterid_to_timestamp(self.interpret_iterid(timestamp))

    def iterate_iterid(self, start=0, stop=None, step=1, last=False):
        start = 0 if start is None else self.interpret_iterid(start)
        stop = len(self) if stop is None else self.interpret_iterid(stop, stop_index=True)
        if step is None:
            step = 1

        if isinstance(step, int):
            i = start
            for i in range(start, stop, step):
                yield i
            if last and i+step < len(self):
                yield i+step
            return

        start_timestamp = self.iterid_to_timestamp(start)
        step = TimeStamp.interpret(step)
        i = start
        e = start_timestamp.epoch
        e_i = start_timestamp.iteration
        while i < stop:
            yield i

            e += step.epoch
            e_i += step.iteration

            while e < self.epoch and e_i > self._epoch_info[e,1]:
                e_i -= self._epoch_info[e,1]
                e += 1

            try:
                i = self.epoch_to_iterid(e, e_i)
            except IndexError:
                i = len(self)
        if i < len(self) and last:
            yield i

    def iterate_timeid(self, start=0, stop=None, step=1, last=False):
        for iterid in self.iterate_iterid(start=start, stop=stop, step=step, last=last):
            yield self._timeline.index.searchsorted(iterid, side='right')-1

    def iterate_timeids_list(self, start=0, stop=None, step=1, last=False):
        iterator = self.iterate_iterid(start=start, stop=stop, step=step, last=last)
        start_timeid = self._timeline.index.searchsorted(next(iterator), side='right')
        for iterid in iterator:
            end_timeid = self._timeline.index.searchsorted(iterid, side='left')
            yield tuple(range(start_timeid, end_timeid))
            start_timeid = end_timeid+1

    def interpret_series_name(self, series, only_number=False):
        if series is None:
            return self.series(only_number=only_number, no_std=True)
        if isinstance(series, str):
            series = [series]
        elif not isinstance(series, list):
                series = list(series)
        unknown_keys = set(series).difference(self.series(only_number=only_number))
        if unknown_keys:
            raise KeyError('%s are not known or valid serie name.' % repr(unknown_keys))

        if only_number:
            not_number_keys = set(series).difference(self.series(only_number=True))
            if not_number_keys:
                raise KeyError('%s are not number series.' % repr(not_number_keys))

        return series

    def export_timeline(self, timestamp=('date', 'time', 'epoch', 'e_iter'), iter_ids=None):
        if isinstance(timestamp, str):
            timestamp = (timestamp,)
        if iter_ids is None:
            iter_ids = list(range(len(self)))
        timeline = self._timeline[list(timestamp)].reindex(iter_ids, copy=False)
        nan_idx = timeline.index[timeline['epoch'].isnull()]

        if not len(nan_idx):
            return timeline

        if 'epoch' in timestamp or 'e_iter' in timestamp:
            epoch_starts = self._epoch_info[:, 0]
            epoch = np.searchsorted(epoch_starts, nan_idx, side='right', sorter=np.arange(len(epoch_starts)))
            if 'e_iter' in timestamp:
                e_iter = nan_idx - epoch_starts[epoch-1]
                timeline.loc[nan_idx, 'e_iter'] = e_iter
                timeline['e_iter'] = timeline['e_iter'].astype('int64')
            if 'epoch' in timestamp:
                timeline.loc[nan_idx, 'epoch'] = epoch
                timeline['epoch'] = timeline['epoch'].astype('int64')
        if 'date' in timestamp:
            timeline['date'].interpolate(method='pad', inplace=True)
        if 'time' in timestamp:
            timeline['time'].interpolate(method='pad', inplace=True)
        return timeline

    def _timeline2iter(self, t):
        r = np.array(self._timeline.index[t], dtype=np.int64)
        return r
        return self._timeline.iloc[t].index.values


class TimeStamp:
    def __init__(self, epoch, iteration, time=None, date=None):
        self._epoch = epoch
        self._iteration = iteration
        self._time = time
        self._date = date

    @property
    def epoch(self):
        return self._epoch

    @property
    def iteration(self):
        return self._iteration

    @property
    def time(self):
        return self._time

    @property
    def date(self):
        return self._date

    def __str__(self):
        return '%ie%i' % (self.epoch, self.iteration)

    def __repr__(self):
        r = 'E%i I%i' % (self.epoch, self.iteration)
        if self.time is not None:
            r += ' (%s)' % time2str(self.time)
        return r

    @staticmethod
    def now(iteration=0, epoch=1):
        import time as python_time
        date = python_time.time()
        return TimeStamp(epoch=epoch, iteration=iteration, time=0, date=date)

    @staticmethod
    def from_timeline_row(timeline_row):
        r = timeline_row
        return TimeStamp(r['epoch'], r['e_iter'], r['time'], r['date'])

    @staticmethod
    def interpret(timestamp):
        if isinstance(timestamp, TimeStamp):
            return timestamp
        elif isinstance(timestamp, tuple):
            if len(timestamp) != 2:
                raise ValueError('%s is not a invalid timestamp\n'
                                 'tuple size should be 2 to be interpreted as TimeStamp'
                                 % (repr(timestamp)))
            if isinstance(timestamp[0], int) and isinstance(timestamp[1], int):
                raise ValueError('%s is not a valid timestamp' % (repr(timestamp)))
            return TimeStamp(epoch=timestamp[0], iteration=timestamp[1])
        elif isinstance(timestamp, str):
            error = ValueError('%s is not a valid timestamp\n'
                               'timestamp string should be formatted: #Ee#I where #E is the epoch and #I the iteration'
                               % timestamp)
            timestamp = timestamp.replace(' ', '')
            try:
                if 'e' in timestamp:
                    timestamp = [int(_) for _ in timestamp.split('e') if _]
                    if len(timestamp) not in (1, 2):
                        raise error
                    if len(timestamp) == 1:
                        return TimeStamp(epoch=timestamp[0], iteration=0)
                    else:
                        return TimeStamp(epoch=timestamp[0], iteration=timestamp[1])
                elif timestamp.endswith('i'):
                    return TimeStamp(epoch=0, iteration=int(timestamp[:-1]))
            except TypeError:
                raise error from None

        raise TypeError('%s is not a valid timestamp.\n Invalid timestamp type: %s'
                        % (repr(timestamp), type(timestamp)))
