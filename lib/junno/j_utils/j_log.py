import time
from ipywidgets import HBox, VBox, Layout, HTML, Accordion
from .ipython.customwidgets import TinyLoading, RichLabel, TimerLabel, LogView, LogToolBar, VSpace, HSpace
from .ipython.import_js import import_display
from .collections import Tree
from tqdm import tqdm
import threading
from threading import Thread
import sys
import os


def is_ipython():
    try:
        get_ipython()
        return True
    except:
        return False


def is_pycharm():
    return "PYCHARM_HOSTED" in os.environ


def second_to_str(time_s):
    time_s = int(round(time_s))
    d = time_s // (3600*24)
    h = time_s % (3600*24) // 3600
    m = time_s % 3600 // 60
    s = time_s % 60

    h = str(h) if h > 9 else '0' + str(h)
    m = str(m) if m > 9 else '0' + str(m)
    s = str(s) if s > 9 else '0' + str(s)

    if d > 1:
        return '%i jours %s:%s:%s' % (d, h, m, s)
    elif d:
        return '1 jour %s:%s:%s' % (h, m, s)
    else:
        return '%s:%s:%s' % (h, m, s)


def float_to_str(number, format='f'):
    s = (('%'+format) % number).split('.')
    n = ''
    for i, _ in enumerate(reversed(s[0])):
        n = _ + n
        if i % 3 == 2:
            n = ' '+n
    n = n+','
    for i, _ in enumerate(s[1]):
        n = n+_
        if i % 3 == 2:
            n = n+' '
    return n

########################################################################################################################
real_stdout = sys.stdout
real_stderr = sys.stderr


class StdOut(object):
    def __init__(self, err=False):
        self._buffer = ''
        self.err = err

    def write(self, string):
        self._buffer += string
        if self._buffer.endswith('\n'):
            self._buffer = self._buffer[:-1]
            self.flush()

    def flush(self):
        if not self.err:
            log.global_log.debug(self._buffer)
            log.flush()
        else:
            log.global_log.error(self._buffer)
        self._buffer = ''


class LogPrinter(Thread):
    def __init__(self):
        self.filter = []
        self._process_stack = []
        self.pure_ascii = False

        self.current_process = None
        self._erase_line = False
        self.ncols = 130
        self.print_stack = []
        self.anim_step = 0

        self.wake_event = threading.Event()
        super(LogPrinter, self).__init__(daemon=True)
        self.start()

    @property
    def catch_print(self):
        return sys.stdout is not real_stdout

    @catch_print.setter
    def catch_print(self, b):
        if b != self.catch_print:
            if b:
                sys.stdout = StdOut()
                sys.stderr = StdOut(err=True)
            else:
                sys.stdout = real_stdout
                sys.stderr = real_stderr

    def run(self):
        while 'program is running':
            self.wake_event.clear()
            self.do_print()
            self.wake_event.wait(1)

    def do_print(self):
        print_stack = self.print_stack[:]
        self.print_stack = []
        self.anim_step = (self.anim_step + 1) % 60

        if print_stack:  # Print messages
            if self._erase_line:
                real_stdout.write('\r' + ' ' * self.ncols + '\r')
                self._erase_line = False
            real_stdout.write('\n'.join(print_stack))
            real_stdout.write('\n')

        if self.current_process:  # Print progress bar
            p = self.current_process
            status = p.name
            status_info = []
            if p.status:
                status_info.append(p.status)
            if not p.max_step and p.step:
                status_info.append(str(p.step))
            if status_info:
                status += ' [%s]' % ', '.join(status_info)
            status += ': '
            if p.max_step:
                try:
                    str_bar = tqdm.format_meter(n=p.step, total=p.max_step, elapsed=p.elapsed_time(), rate=p.rate(),
                                                prefix=status, ncols=self.ncols, ascii=self.pure_ascii)
                except UnicodeEncodeError:
                    self.pure_ascii = True
                    str_bar = tqdm.format_meter(n=p.step, total=p.max_step, elapsed=p.elapsed_time(), rate=p.rate(),
                                                prefix=status, ncols=self.ncols, ascii=True)

                if self._erase_line:
                    real_stdout.write('\r' + ' ' * self.ncols + '\r')
                real_stdout.write(str_bar)

            else:
                r = p.formated_rate()
                if r:
                    time = '[%s, %s]' % (second_to_str(p.elapsed_time()), r)
                else:
                    time = '[%s]' % second_to_str(p.elapsed_time())
                status += '.' * (self.anim_step % 3 + 1)

                if self._erase_line:
                    real_stdout.write('\r' + ' ' * self.ncols + '\r')
                real_stdout.write(status + ' ' * (self.ncols - len(status) - len(time)) + time)

            self._erase_line = True
        real_stdout.flush()

    def print_msg(self, msg, log=None):
        if msg.type in self.filter:
            return
        color = {'warning': 'yellow', 'debug': 'grey', 'info': -1, 'error': 'red'}[msg.type]
        txt = msg.msg
        self.print(txt, {'color': color}, log=log)

    def print(self, txt, option=None, log=None, indent=0):
        if txt == '':
            return
        if not isinstance(txt, str):
            txt = str(txt)
        if not indent and log is not None and isinstance(log, Process):
            try:
                indent = self._process_stack.index(log) + 1
            except ValueError:
                pass

        if indent:
            indent = ' ' * indent * 3
            txt = indent+txt.replace('\n', '\n' + indent)

        if option is not None:
            if 'color' in option:
                color = option['color']
                light = option.get('light', False)
                txt = LogPrinter.color(txt, color, light)
            if 'background_color' in option:
                color = option['background_color']
                light = option.get('background_light', False)
                txt = LogPrinter.background_color(txt, color, light)
            if option.get('bold', False):
                txt = LogPrinter.bold(txt)
            if option.get('underlined', False):
                txt = LogPrinter.underlined(txt)
            if option.get('blink', False):
                txt = LogPrinter.blink(txt)
            if option.get('dim', False):
                txt = LogPrinter.dim(txt)

        self.print_stack.append(txt)
        self.wake_event.set()

    def process_started(self, process):
        if process not in self._process_stack:
            if process.verbose:
                self.print('\n'+process.name.upper(), {'bold': True, 'color': 'white'}, indent=len(self._process_stack))
            self._process_stack.append(process)

        self.process_changed(process)

    def process_end(self, process):
        indent = 0
        if process in self._process_stack:
            indent = self._process_stack.index(process)
            self._process_stack.remove(process)

        if process.verbose:
            time_info = 'in %s' % second_to_str(process.elapsed_time())
            if process.rate():
                time_info += ' (%.2fit/s)' % process.rate()

            if process.state == 'success':
                self.print('%s ended %s\n' % (process.name, time_info),
                           {'color': 'white', 'dim': True}, indent=indent)
            elif process.state == 'failed':
                self.print('%s ended with error %s\n' % (process.name, time_info),
                           {'color': 'red', 'dim': True}, indent=indent)

        self.process_changed()

    def process_changed(self, process=None):
        if process is None and self._process_stack:
            process = self._process_stack[-1]
        self.current_process = process
        self.wake_event.set()

    @staticmethod
    def decode_color(color, light):
        if isinstance(color, int):
            if color < 0 or color > 9:
                color = 9
                light = True
        elif isinstance(color, str):
            color = color.lower()
            if color.startswith('light '):
                color = color[6:]
                light = True
            else:
                light = light

            if color == 'white' or color == 'w':
                if is_pycharm():
                    color = 0
                    light = False
                else:
                    color = 7
                    light = True
            elif color == 'black':
                if is_pycharm():
                    color = 0
                    light = True
                else:
                    color = 0
                    light = False
            elif color == 'red' or color == 'r':
                color = 1
            elif color == 'green' or color == 'g':
                color = 2
            elif color == 'yellow' or color == 'y':
                color = 3
            elif color == 'blue' or color == 'b':
                color = 4
            elif color == 'magenta' or color == 'm':
                color = 5
            elif color == 'cyan' or color == 'c':
                color = 6
            elif color == 'gray':
                color = 7
            else:
                color = 9
                light = True

        return color, light

    @staticmethod
    def color(txt, color=-1, light=False):
        color, light = LogPrinter.decode_color(color, light)
        return '\033[%i%im%s\033[39m' % (9 if light else 3, color, txt)

    @staticmethod
    def background_color(txt, color, light):
        color, light = LogPrinter.decode_color(color, light)
        return '\033[%i%im%s\033[49m' % (10 if light else 4, color, txt)

    @staticmethod
    def bold(txt):
        return '\033[1m%s\033[21m' % txt

    @staticmethod
    def underlined(txt):
        return ' \033[4m%s\033[24m' % txt

    @staticmethod
    def blink(txt):
        return '\033[5m%s\033[25m' % txt

    @staticmethod
    def dim(txt):
        return '\033[2m%s\033[22m' % txt


if not is_ipython() or is_pycharm():
    log_out = LogPrinter()
else:
    log_out = None


########################################################################################################################
class LogsHandler:
    global log_out

    def __init__(self):
        self._process_stack = {}
        self._process_tree = Tree()
        self.global_log = None
        self.default_log = self.global_log
        self._ipywidget = None

    @property
    def log(self):
        id = threading.get_ident()
        if id in self._process_stack and self._process_stack[id]:
            return self._process_stack[id][-1]
        return self.default_log

    def warn(self, *args, msg=None):
        return self.log.warn(*args, msg=msg)

    def debug(self, *args, msg=None):
        return self.log.debug(*args, msg=msg)

    def info(self, *args, msg=None):
        return self.log.info(*args, msg=msg)

    def error(self, *args, msg=None):
        return self.log.error(*args, msg=msg)

    def __lshift__(self, other):
        return self.log.__lshift__(other)

    def flush(self):
        log_out.do_print()

    def register_process(self, process):
        if process.thread_id not in self._process_stack:
            stack = []
            self._process_stack[process.thread_id] = stack
        else:
            stack = self._process_stack[process.thread_id]

        if process in stack:
            raise ValueError('Process %s was already registered') % process.name

        if process.parent == 'auto':
            if stack:
                process.parent = stack[-1]
            else:
                process.parent = None

        if process.verbose:
            if process.parent is None:
                self._process_tree.root[process.name] = process
                self._process_tree.root[process.name].clear_children()
            else:
                parent_node = self._process_tree.root.search(process.parent)
                if parent_node is not None:
                    parent_node.add_node(process.name, process)

        stack.append(process)

        if log_out is not None and isinstance(process, Process):
            log_out.process_started(process)

        if self._ipywidget:
            self.update_ipywidget()

    def release_process(self, process):
        if process.thread_id in self._process_stack and process in self._process_stack[process.thread_id]:
            self._process_stack[process.thread_id].remove(process)

        if self._ipywidget:
            tree_node = self._process_tree.root.search(process)
            if tree_node:
                address = tree_node.address
                self.accordion_by_id(address[:-1]).set_title(address[-1], process.name + '    (ended in %s)'
                                                             % second_to_str(process.elapsed_time()))
            if process.state == 'success':
                process.debug('%s ended in %s' % (process.name, second_to_str(process.elapsed_time())))
            elif process.state == 'failed':
                process.debug('%s ended with eror in %s' % (process.name, second_to_str(process.elapsed_time())))

        if log_out is not None and isinstance(process, Process):
            log_out.process_end(process)

    def update_process(self, process):
        if log_out is not None:
            log_out.process_changed(process)

        if self._ipywidget:
            self._ipywidget.header.children = (process.status_ipywidget, self._ipywidget.header.children[1])

    def accordion_by_id(self, address, root=None):
        root_accordion = self._ipywidget.accordion
        if root is None:
            root = root_accordion
        if len(address) == 0:
            return root
        current = root.children[address[0]].children[0]
        if len(address) == 1:
            return current
        return self.accordion_by_id(address[1:], current)

    def update_ipywidget(self):
        w = []
        root_accordion = self._ipywidget.accordion
        toolbar = self._ipywidget.toolbar

        for n in self._process_tree.walk():
            address = n.address
            if len(address) > 1:
                parent_accordion = self.accordion_by_id(address[:-1])
            else:
                parent_accordion = root_accordion

            child = list(parent_accordion.children)
            if len(child) <= address[-1] or child[address[-1]].children[2].source is not n.data:
                node_w = n.data.log_ipywidget(toolbar=False)
                acc = Accordion(layout=Layout(width='100%', max_height='500px'))
                #acc.selected_index = -1

                new_item = VBox([acc, VSpace(), node_w], layout=Layout(overflow_y='auto'))
                if len(child) <= address[-1]:
                    child.append(new_item)
                else:
                    child[address[-1]] = new_item

                name = n.data.name
                if isinstance(n.data, Process) and n.data.state in ('success', 'failed'):
                    name += ' (ended in %s)' % second_to_str(n.data.elapsed_time())
                parent_accordion.set_title(address[-1], name)
                parent_accordion.children = tuple(child)
            else:
                node_w = parent_accordion.children[address[-1]].children[2]
            w.append(node_w)

        toolbar.logviews = w

    def __getitem__(self, item):
        if isinstance(item, str):
            if item == 'global':
                return self.global_log
            if item in self._process_stack:
                return self._process_stack[item][-1]
            for stack in self._process_stack.values():
                for log in stack:
                    if log.name == item:
                        return log
        return None

    @property
    def ipywidget(self):
        if self._ipywidget is None:
            accordion = Accordion()
            #accordion.selected_index = -1

            toolbar = LogToolBar()
            header = HBox([HSpace(400), toolbar], layout=Layout(justify_content='space-between', align_items='flex-end'))

            self._ipywidget = VBox([header, accordion])
            self._ipywidget.header = self._ipywidget.children[0]
            self._ipywidget.accordion = accordion
            self._ipywidget.toolbar = toolbar
            self.update_ipywidget()
        return self._ipywidget

    def _ipython_display_(self):
        import_display(self.ipywidget)

log = LogsHandler()


########################################################################################################################
class Logs:
    global log

    class Message:
        def __init__(self, msg, type, log_time):
            self.msg = msg
            self.type = type
            self.time = time.time()
            self.log_time = log_time

    def __init__(self, name):
        self.msg_list = []
        self._name = name
        self.thread_id = threading.get_ident()
        self.default_type = 'info'
        self.verbose = []
        self._log_ipywidgets = []
        self._ipywidget = None

    @property
    def name(self):
        return self._name

    def new_msg(self, type, msg=None):
        if msg is None:
            self.default_type = type
            return self
        msg = Logs.Message(msg, type, self.elapsed_time())
        self.msg_list.append(msg)
        self.notify_change()
        if log_out is not None:
            log_out.print_msg(msg, log=self)
        return self

    def clear_log(self):
        self.msg_list.clear()
        self.notify_change()

    def notify_change(self):
        for log in self._log_ipywidgets:
            log.value = self._log_txt()

    def warn(self, *args, msg=None):
        if args and msg is None:
            msg = ' '.join(str(_) for _ in args)
        return self.new_msg('warning', msg)

    def debug(self, *args, msg=None):
        if args and msg is None:
            msg = ' '.join(str(_) for _ in args)
        return self.new_msg('debug', msg)

    def info(self, *args, msg=None):
        if args and msg is None:
            msg = ' '.join(str(_) for _ in args)
        return self.new_msg('info', msg)

    def error(self, *args, msg=None):
        if args and msg is None:
            msg = ' '.join(str(_) for _ in args)
        return self.new_msg('error', msg)

    def __lshift__(self, other):
        return self.new_msg(other, self.default_type)

    def elapsed_time(self):
        return -1

    def log_ipywidget(self, toolbar=False, height=200):
        l = LogView(layout=Layout(height='%ipx' % height, overflow_y='auto'),
                    value=self._log_txt())
        l.source = self
        self._log_ipywidgets.append(l)

        if not toolbar:
            return l

        t = LogToolBar(l)
        layout = VBox([t, l])
        layout.log_view = l
        layout.toolbar = t
        return layout

    def _log_txt(self):
        txt = ''
        for msg in self.msg_list:
            txt += '\\\\%s|%s @ %s' % (msg.type, msg.msg.replace('\n', '<br />'), second_to_str(msg.log_time))
        return txt

    @property
    def ipywidget(self):
        if self._ipywidget is None:
            self._ipywidget = self.log_ipywidget(toolbar=True)
        return self._ipywidget

    def _ipython_display_(self):
        import_display(self.ipywidget)

log.global_log = Logs('global')
log.default_log = log.global_log
log._process_tree.root.add_node('global', log.global_log)
if not is_ipython() and log_out is not None:
    log_out.catch_print = True


class Process(Logs):
    def __init__(self, name, total=None, initial=0, start=False, start_time=None, raise_on_fail=True, verbose=True, parent='auto'):
        super(Process, self).__init__(name)
        self._step = initial
        self._status = ""
        self._max_step = total
        self._avg_rate = 0
        self.smoothing = 0.8
        self._last_elapsed_time = 0

        self.start_time = -1 if start_time is None else start_time
        self._paused_time = 0
        self._last_time = self.start_time
        self._state = 'paused'
        self._status_ipywidget = None
        self.raise_on_fail = raise_on_fail
        self.verbose = verbose
        self.parent = parent

        if start:
            self.start()

    def elapsed_time(self, as_str=False):
        if as_str:
            return second_to_str(self.elapsed_time())

        if self.start_time < 0:
            return 0
        if self.state == 'running':
            return time.time() - self.start_time - self._paused_time
        else:
            return self._last_time - self.start_time - self._paused_time

    def pause(self):
        if self.state == 'running':
            self._last_time = time.time()
            self.status_ipywidget.timer.pause()
        self.status_ipywidget.timer.time = self.elapsed_time()

        self._state = 'paused'
        self._update_views()

    def fail(self, msg=None):
        if self.state == 'running':
            self._last_time = time.time()
            self.status_ipywidget.timer.pause()
        self.status_ipywidget.timer.time = self.elapsed_time()

        if msg is not None:
            self.error(msg)
        self._state = 'failed'
        self._update_views()

        log.release_process(self)

    def succeed(self):
        if self.state == 'running':
            self._last_time = time.time()
            self.status_ipywidget.timer.pause()
        self.status_ipywidget.timer.time = self.elapsed_time()

        self._state = 'success'
        self._update_views()

        log.release_process(self)

    def start(self):
        if self.state != 'running':
            if self.start_time > 0:
                self._paused_time += time.time() - self._last_time
                self._last_time = time.time()
            else:
                self.start_time = time.time()
                self._last_time = self.start_time
                log.register_process(self)
        self.status_ipywidget.timer.time = self.elapsed_time()
        self.status_ipywidget.timer.start()

        self._state = 'running'
        self._update_views()

    def exec(self, method, *arg, **kwargs):
        self.start()
        try:
            method(*arg, **kwargs)
        except Exception as e:
            self.fail(str(e))
            if self.raise_on_fail:
                raise e
        self.succeed()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            self.succeed()
            return True
        if hasattr(exc_val, 'consumed'):
            self.fail()
        else:
            exc_val.consumed = True
            self.fail(str(exc_val))
        return not self.raise_on_fail

    @property
    def status_ipywidget(self):
        if self._status_ipywidget is None:
            title = RichLabel(self.name)
            title.font = 'bold 25px Helvetica'
            title.color = '#111'

            status = RichLabel(self.status, align='right', font='14px Helvetica')
            rate = RichLabel(self.formated_rate(), align='right', font='11px Helvetica')
            timer = TimerLabel(self.elapsed_time(), align='right', font='11px helvetica')
            progress = TinyLoading(self.progress(), layout=Layout(width='400px'))

            tbox = HBox([rate, HSpace(10), timer])

            self._status_ipywidget = VBox([HBox([title,
                                                 VBox([status, tbox], layout=Layout(justify_content='flex-end'))]
                                                , layout=Layout(justify_content='space-between')),
                                           progress])
            self._status_ipywidget.title = title
            self._status_ipywidget.rate = rate
            self._status_ipywidget.status = status
            self._status_ipywidget.timer = timer
            self._status_ipywidget.progress = progress
            self._update_views()

        return self._status_ipywidget

    def _update_views(self):
        if self._status_ipywidget:
            p = self.status_ipywidget.progress
            if self.state == 'running':
                p.value = self.progress()
                self.status_ipywidget.rate.value = self.formated_rate()
                p.color = '#2facff'
            elif self.state == 'paused':
                p.value = max(0, self.progress())
                p.color = 'orange'
            elif self.state == 'failed':
                p.value = max(0, self.progress())
                p.color = 'red'
            elif self.state == 'success':
                p.value = 1
                p.color = 'green'

        log.update_process(self)

    @property
    def ipywidget(self):
        if self._ipywidget is None:
            logs = self.log_ipywidget()
            logs_toolbar = LogToolBar(logs)
            status = self.status_ipywidget
            space = HTML(layout=Layout(height='7px'))
            self._ipywidget = VBox([HBox([status, logs_toolbar],
                                         layout=Layout(justify_content='space-between', align_items='flex-end')),
                                    space, logs])
            self._ipywidget.logs = logs
            self._ipywidget.logs_toolbar = logs_toolbar
            self._ipywidget.status = status
        return self._ipywidget

    def _ipython_display_(self):
        import_display(self.ipywidget)

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, s):
        if self.max_step is None or s <= self.max_step:
            t = self.elapsed_time()
            delta_t = t-self._last_elapsed_time
            if self._step >= s or not delta_t:
                self._avg_rate = 0
            else:
                rate = (s-self._step) / delta_t
                self._avg_rate = (rate+self.smoothing*self._avg_rate)/(self.smoothing+1) if self._avg_rate else rate
            self._step = s
            self._last_elapsed_time = t
        self._update_views()

    def update(self, i):
        self.step += i

    @property
    def max_step(self):
        return self._max_step

    @max_step.setter
    def max_step(self, s):
        if s > 0:
            self._max_step = s
        self._update_views()

    @property
    def state(self):
        return self._state

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, s):
        self._status = s
        self.status_ipywidget.status.value = s

    def progress(self):
        if self.max_step is not None and self.max_step>0:
            return self.step / self.max_step
        return -1

    def rate(self):
        if not self.step or not self._avg_rate or self.max_step <= 1:
            return None
        global_rate = self.step / self.elapsed_time()
        if self.state in ('success', 'failed'):
            return global_rate
        else:
            return self._avg_rate

    def formated_rate(self):
        r = self.rate()
        if r is None:
            r = ''
        elif r >= 1:
            r = '%.2fit/s' % r
        else:
            r = '%.2fs/it' % (1/r)

        if self.max_step is not None and self.max_step > 1:
            step = '%i/%i' % (self.step, self.max_step)
        elif self.max_step == 1:
            step = '%.2f%%' % self.step*100
            print(step)
        else:
            step = '%i' % self.step

        if r:
            return step + '|' + r
        else:
            return step

log.Process = Process