import os
import psutil
from time import clock, ctime


def ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def print_ram_usage(msg=''):
    if msg:
        msg = ' (%s)' % msg
    print('RAM: %ikB%s' % (ram_usage()//1000, msg))


def format_duration(t):
    if t < 1e-9:
        return '%.3fns' % (t*1e9)
    elif t < 1e-6:
        return '%.1fns' % (t*1e9)
    elif t < 1e-3:
        return '%.1fµs' % (t*1e6)
    elif t < 1:
        return '%.1fms' % (t*1e3)
    elif t < 60:
        return '%.1fs' % t
    elif t < 3600:
        return '%02.0f:%02.1f' % (t/60, t % 60)
    elif t < 3600*24:
        return '%02.0f:%02.0f:%02.1f' % (t / 3600, (t % 3600) / 60, t % 60)
    d = t // (3600*24)
    t = t % (3600*24)
    return '%id %02.0f:%02.0f:%02.1f' % (d, t / 3600, (t % 3600) / 60, t % 60)


def tic(name):
    tictoc = TicToc()
    tictoc.tic(name)
    return tictoc


class TicToc:
    def __init__(self):
        self.t0 = None
        self.current_tic = None
        self.t = []

    def tic(self, name):
        if self.current_tic:
            print('')
        self.current_tic = TicToc.Tic(name, clock())
        print(self.current_tic)

    def toc(self, name):
        t = clock()
        toc = TicToc.Toc(name, t-self.current_tic.clock, t, t-self.t[-1].clock if self.t else None)
        self.t.append(toc)
        print('\t', toc)

    class Tic:
        def __init__(self, name, clock):
            self.name = name
            self.clock = clock
            self.date = ctime()

        def __str__(self):
            return '[%s] -  %s' % (self.date[-13:-5], self.name)

    class Toc:
        def __init__(self, name, time, clock, duration=None):
            self.time = time
            self.duration = duration if duration is not None else time
            self.clock = clock
            self.name = name

        def __str__(self):
            return '+ %s : %s  (%s)' % (format_duration(self.time), self.name, format_duration(self.duration))

