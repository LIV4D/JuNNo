import pandas
import scipy.interpolate
import numpy as np
from ..j_utils.string import str2time, time2str
from ..j_utils.path import format_filepath
from collections import OrderedDict


class History:
    """
    Store dataseries by iteration and epoch.
    Data are index through timestamp: the number of iteration since the first iteration of the first epoch.
    """
    def __init__(self):
        self._timeline_series = OrderedDict()
        self._timestamps = pandas.DataFrame(columns=['date', 'time'])
        self._events = []
        self._nb_iterations_by_epoch = [0]

        self._current_epoch = 1
        self._current_epoch_iteration = -1

    def save(self, path):
        path = format_filepath(path)
        df = self.export_dataframe()

    def load(self, path):
        path = format_filepath(path)

    # ---  Current Iteration  ---
    @property
    def epoch(self):
        return self._current_epoch

    @property
    def iteration(self):
        return self._current_epoch_iteration

    @property
    def last_timeid(self):
        return sum(self._nb_iterations_by_epoch)

    def __len__(self):
        return self.last_timeid + 1

    def next_iteration(self, time, date=None):
        self._current_epoch_iteration += 1
        self._nb_iterations_by_epoch[-1] = self._current_epoch_iteration
        self._update_timestamp(time, date)

    def next_epoch(self, time, date=None):
        self._current_epoch += 1
        self._current_epoch_iteration = 0
        self._nb_iterations_by_epoch.append(0)
        self._update_timestamp(time, date)

    def _update_timestamp(self, time, date):
        if date is None:
            date = pandas.Timestamp.now()
        date = pandas.to_datetime(date)
        df = pandas.DataFrame([[time, date]], index=[self.last_timeid], columns=['time', 'date'])
        self._timestamps = self._timestamps.append(df)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError('History key should be a serie name not (%s, type:%s).'
                           % (str(key), type(key)))

        if key not in self._timeline_series:
            serie = pandas.Series(data=[value], index=[self.last_timeid], name=key)
            self._timeline_series[key] = serie
        else:
            self._timeline_series[key][self.last_timeid] = value

    # ---  Store/Read Data  ---
    def keys(self):
        return self._timeline_series.keys()

    def series(self, only_number=False):
        keys = list(self.keys())
        if only_number:
            return [k for k in keys if self._timeline_series[k].dtype != 'O']
        return keys

    def __getitem__(self, item):
        if isinstance(item, str):
            if item not in self.keys():
                raise KeyError('%s is an unknown serie name.' % item)
            return self._timeline_series[item].iloc[-1]
        elif isinstance(item, tuple):
            if len(item) != 2 or item[0] not in self.keys():
                raise KeyError("Invalid history index: %s\n"
                               "Index should follow the form: ['series name', time_index]" % repr(item))
            series = item[0]
            timeid = item[1]

            if isinstance(timeid, slice):
                df = self.read(series=series, start=timeid.start, stop=timeid.stop, step=timeid.step,
                               interpolation='previous', averaged=True, std=False)
                return df[series].values
            else:
                return self.get(series=series, timeid=timeid, interpolation='previous')
        raise IndexError('Invalid index: unable to read from history series')

    def get(self, series, timeid=-1, interpolation='previous', default='raise exception'):
        try:
            t = self.interpret_timeid(timeid)
            if series not in self.keys():
                raise KeyError('%s is an unknown serie name.' % series)
        except LookupError as e:
            if default != 'raise exception':
                return default
            raise e from None

        serie = self._timeline_series[series]
        if interpolation is None:
            try:
                return serie.loc[t]
            except KeyError:
                if default != 'raise exception':
                    return default
                raise IndexError("Serie %s doesn't store any data at time: %s.\n"
                                 "The interpolation parameter may be use to remove this exception."
                                 % (series, repr(timeid)))
        else:
            serie = scipy.interpolate.interp1d(x=serie.index, y=serie.values,
                                               kind=interpolation, fill_value='extrapolate',
                                               assume_sorted=True, copy=False)
            return serie(timeid)

    def read(self, series=None, start=0, stop=0, step=1, timestamp=None,
             interpolation='previous', smooth=None, averaged=True, std=False):
        """
        Interpolate or average
        :param series: Keys of the variables to read
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
        :param interpolation: Specify which number serie should be interpolated and how.
        NaN in number series can automatically be replaced by interpolated values using pandas interpolation algorithms.
        This parameter most be one of those:
            - True: All numbers series are interpolated linearly
            - False: No interpolation is applied (NaN are not replaced)
            - List of series name: The specified series are interpolated linearly
            - Dictionary associating an interpolation method to a series name.
        :param smooth: Specify which number series should be smoothed and how much.
        Specified series are Savitzky-Golay filter of order 3. The window size may be chosen (default is 15).
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
        indexes = np.array(list(self.timeid_iterator(start=start, stop=stop, step=step)), dtype=np.uint32)
        intervals = np.stack((indexes, np.concatenate((indexes[1:], [stop]))), axis=1)

        series_name = self.interpret_series_name(series)

        if isinstance(averaged, bool):
            averaged = self.series(only_number=True) if averaged else []
        else:
            averaged = self.interpret_series_name(averaged, only_number=True)

        if isinstance(std, bool):
            std = averaged if std else []
        else:
            if isinstance(std, str):
                std = [std]
            not_averaged_series = set(std).difference(averaged)
            if not_averaged_series:
                raise ValueError("Can't compute standard deviation of: %s.\n"
                                 "Those series are not averaged." % repr(not_averaged_series))

        if not interpolation:
            interpolation = {}
        elif isinstance(interpolation, bool):
            interpolation = {_: 'linear' for _ in self.series(only_number=True)}
        elif isinstance(interpolation, str):
            if interpolation in self.series(only_number=True):
                interpolation = {interpolation: 'linear'}
            else:
                interpolation = {_: interpolation for _ in self.series(only_number=True)}
        elif isinstance(interpolation, (dict, OrderedDict)):
            unknown_keys = set(interpolation.keys()).difference(self.series(only_number=True))
            if unknown_keys:
                raise ValueError("Can't interpolate series: %s.\n"
                                 "Those series are either unknown or don't contain numbers!" % repr(unknown_keys))
        else:
            unknown_keys = set(interpolation).difference(self.series(only_number=True))
            if unknown_keys:
                raise ValueError("Can't interpolate series: %s.\n"
                                 "Those series are either unknown or don't contain numbers!" % repr(unknown_keys))
            interpolation = {_: 'linear' for _ in interpolation}

        if not smooth:
            smooth = {}
        elif isinstance(smooth, bool):
            smooth = {_: 15 for _ in self.series(only_number=True)}
        elif isinstance(smooth, str):
            if smooth not in self.series(only_number=True):
                raise ValueError("Can't smooth series %s. It is either unknown or doesn't contain number!"
                                 % smooth)
            smooth = {smooth: 15}
        elif isinstance(smooth, (dict, OrderedDict)):
            unknown_keys = set(smooth.keys()).difference(self.series(only_number=True))
            if unknown_keys:
                raise ValueError("Can't smooth series: %s.\n"
                                 "Those series are either unknown or don't contain numbers!" % repr(unknown_keys))
        else:
            unknown_keys = set(smooth).difference(self.series(only_number=True))
            if unknown_keys:
                raise ValueError("Can't smooth series: %s.\n"
                                 "Those series are either unknown or don't contain numbers!" % repr(unknown_keys))
            smooth = {_: 15 for _ in smooth}
        if smooth:
            import scipy.signal

        df = []
        for k in series_name:
            series = self._timeline_series[k]
            std_series = None

            # Sample
            if k in self.series(only_number=True):
                if k not in averaged:
                    series = series.reindex(indexes, copy=False)
                else:
                    mean_series = np.zeros(shape=(intervals.shape[0],))
                    std_series = np.zeros(shape=(intervals.shape[0],)) if k in std else None
                    for i, (start_id, end_id) in enumerate(intervals):
                        s = series.loc[start_id:end_id-1]
                        mean_series[i] = np.nanmean(s) if len(s) else np.nan
                        if std_series is not None:
                            std_series[i] = np.nanvar(s) if len(s) else np.nan
                    series = pandas.Series(index=indexes, data=mean_series, name=series.name)
                    if std_series is not None:
                        std_series = pandas.Series(index=indexes, data=std_series, name='STD '+series.name)

                # Interpolate
                if k in interpolation:
                    if interpolation[k] == 'previous':
                        series.fillna(method='pad', inplace=True)
                        if std_series is not None:
                            std_series.fillna(method='pad', inplace=True)
                    else:
                        series.interpolate(method=interpolation[k], inplace=True)
                        if std_series is not None:
                            std_series.interpolate(method=interpolation[k], inplace=True)
                # Smooth
                if k in smooth:
                    s = series.values
                    s = scipy.signal.savgol_filter(s, smooth[k], 3, mode='constant')
                    series = pandas.Series(index=indexes, data=s, dtype=series.dtype, name=series.name)
            else:
                series = series.reindex(indexes, copy=False, method='pad')

            # Store
            df.append(series)
            if std_series is not None:
                df.append(std_series)

        if timestamp:
            df = self.timestamp_dataframe(timestamp, indexes, series_list=True) + df
        return pandas.DataFrame(df).transpose()

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
        start = self.interpret_timeid(start)
        stop = self.interpret_timeid(stop, stop_index=True)
        series_name = self.interpret_series_name(series)

        series = []
        for k in series_name:
            serie = self._timeline_series[k].loc[start:stop]
            series.append(serie)

        df = pandas.DataFrame(series).transpose()

        if timestamp:
            timestamp_df = self.timestamp_dataframe(timestamp, df.index)
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
    def epoch_to_timeid(self, epoch, iteration=1):
        # Check
        if epoch > self.epoch:
            raise IndexError('Invalid time stamp: %ie%i. (Current iteration is %ie%i)'
                             % (epoch, iteration, self.epoch, self.last_timeid))
        if iteration > self._nb_iterations_by_epoch[epoch]:
            raise IndexError('Invalid time stamp: %ie%i. (Epoch %i only has %i iterations)'
                             % (epoch, iteration, epoch, self._nb_iterations_by_epoch[epoch]))
        # Sum
        return iteration + sum(nb_it for e, nb_it in enumerate(self._nb_iterations_by_epoch) if e + 1 < epoch) - 1

    def timeid_to_timestamp(self, time_id):
        if not 0 <= time_id < len(self):
            raise ValueError('%i is not a valid timestamp (min:0, max:%i)' % (time_id, len(self)-1))
        e = 1
        epoch_iteration = self._nb_iterations_by_epoch[0]
        while e <= self.epoch and time_id > epoch_iteration:
            epoch_iteration += self._nb_iterations_by_epoch[e]
            e += 1
        i = time_id-epoch_iteration

        time = self._timestamps['time'][time_id]
        date = self._timestamps['date'][time_id]

        return TimeStamp(epoch=e, iteration=i, time=time, date=date)

    def interpret_timeid(self, timestamp, stop_index=False):
        if isinstance(timestamp, TimeStamp):
            return self.epoch_to_timeid(epoch=timestamp.epoch,
                                        iteration=timestamp.iteration)
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
            return self.epoch_to_timeid(epoch=timestamp.epoch, iteration=timestamp.iteration)

    def interpret_timestamp(self, timestamp):
        return self.timeid_to_timestamp(self.interpret_timeid(timestamp))

    def timeid_iterator(self, start=0, stop=0, step=1, last=False):
        start = 0 if start is None else self.interpret_timeid(start)
        stop = len(self) if stop is None else self.interpret_timeid(stop, stop_index=True)
        if step is None:
            step = 1

        if isinstance(step, int):
            i = start
            for i in range(start, stop, step):
                yield i
            if last and i+step < len(self):
                yield i+step
            return

        start_timestamp = self.timeid_to_timestamp(start)
        step = TimeStamp.interpret(step)
        i = start
        e = start_timestamp.epoch; e_i = start_timestamp.iteration
        while i < stop:
            yield i
            e += step.epoch
            e_i += step.iteration
            while e < self.epoch and e_i > self._nb_iterations_by_epoch[e]:
                e_i -= self._nb_iterations_by_epoch[e]
                e += 1
            i = self.epoch_to_timeid(e, e_i)
        if i < len(self) and last:
            yield i

    def interpret_series_name(self, series, only_number=False):
        if series is None:
            return self.series(only_number=only_number)
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

    def timestamp_dataframe(self, timestamp=('date', 'time', 'epoch', 'iteration'), indexes=None, series_list=False):
        if isinstance(timestamp, str):
            timestamp = (timestamp,)
        if indexes is None:
            indexes = self._timestamps.index

        from bisect import bisect_left
        cumul_epoch = np.concatenate(([0], np.cumsum(self._nb_iterations_by_epoch)))

        df = []
        for k in timestamp:
            if k == 'epoch':
                series = pandas.Series(index=indexes, name='epoch',
                                       data=indexes.map(lambda timeid: bisect_left(cumul_epoch, timeid+1)))
                df.append(series)
            elif k == 'iteration':
                series = pandas.Series(index=indexes, name='iteration',
                                       data=indexes.map(
                                           lambda timeid: timeid - cumul_epoch[bisect_left(cumul_epoch, timeid+1)-1]))
                df.append(series)
            elif k == 'time':
                df.append(self._timestamps['time'].reindex(indexes, copy=False))
            elif k == 'date':
                df.append(self._timestamps['date'].reindex(indexes, copy=False))

        if series_list:
            return df
        return pandas.DataFrame(df).transpose()


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
            try:
                timestamp = [int(_) for _ in timestamp.split('e')]
            except TypeError:
                raise error
            if len(timestamp) not in (1,2):
                raise error
            if len(timestamp) == 1:
                return TimeStamp(epoch=timestamp[0], iteration=0)
            else:
                return TimeStamp(epoch=timestamp[0], iteration=timestamp[1])

        raise TypeError('%s is not a valid timestamp.\n Invalid timestamp type: %s'
                        % (repr(timestamp), type(timestamp)))
