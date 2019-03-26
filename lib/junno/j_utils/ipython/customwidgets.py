from ipywidgets.widgets import HTML, DOMWidget, VBox, HBox, Text, Layout, jsdlink, ToggleButton
from traitlets import Unicode, Float, Bool, Int, validate, observe, List, TraitType
from IPython.display import Javascript, display
from time import time
from ..function import match_params
from .import_js import import_js, AutoImportDOMWidget


class TinyLoading(AutoImportDOMWidget):
    _view_name = Unicode('TinyLoading').tag(sync=True)
    _view_module = Unicode('gcustom').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    value = Float(0).tag(sync=True)
    color = Unicode('#2facff').tag(sync=True)
    background_color = Unicode('#CCCCCC').tag(sync=True)
    bar_height = Unicode('6px').tag(sync=True)

    def __init__(self, value=0, **kwargs):
        super(TinyLoading, self).__init__(value=value, dependencies=('CustomWidgets',), **kwargs)


class RichLabel(AutoImportDOMWidget):
    _view_name = Unicode('RichLabel').tag(sync=True)
    _view_module = Unicode('gcustom').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    value = Unicode("").tag(sync=True)
    color = Unicode('#444').tag(sync=True)
    font = Unicode('15px helvetica').tag(sync=True)
    align = Unicode('right').tag(sync=True)

    def __init__(self, value='', **kwargs):
        super(RichLabel, self).__init__(value=value, dependencies=('CustomWidgets',), **kwargs)

    @property
    def bold(self):
        return 'bold' in self.font

    @bold.setter
    def bold(self, isBold):
        if isBold != self.bold:
            if isBold:
                self.font = 'bold ' + self.font
            else:
                self.font = self.font.replace('bold ', '')


    @property
    def italic(self):
        return 'italic' in self.font

    @italic.setter
    def italic(self, isItalic):
        if isItalic != self.italic:
            if isItalic:
                self.font = 'italic ' + self.font
            else:
                self.font = self.font.replace('italic ', '')

    @property
    def font_size(self):
        return float(self.font.split('px')[0].split(' ')[-1])

    @font_size.setter
    def font_size(self, px):
        if px != self.font_size:
            font_info = self.font.split('px')
            font_info[0] = ' '.join(font_info[0].split(' ')[:-1])
            if font_info[0]:
                font_info[0] += ' '
            self.font = font_info[0] + '%.1fpx' % px + font_info[1]

    @property
    def font_family(self):
        return self.font.split('px ')[1]

    @font_family.setter
    def font_family(self, ff):
        if ff != self.font_family:
            self.font = self.font.split('px ')[0] + 'px %s' % ff


class TimerLabel(RichLabel):
    _view_name = Unicode('TimerLabel').tag(sync=True)
    _view_module = Unicode('gcustom').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    _time = Int(0).tag(sync=True)
    running = Bool(False).tag(sync=True)
    step = Int(1).tag(sync=True)

    def __init__(self, time=0, color='#444', font='15px helvetica', **kwargs):
        super(TimerLabel, self).__init__(time=time, font=font, color=color, **kwargs)
        self._start_time = 0
        self._initial_timer = 0

    def start(self):
        if not self.running:
            self._start_time = int(time())
            self._initial_timer = self.time
            self.running = True

    def pause(self):
        if self.running:
            self._time = self.time
            self.running = False

    def reset(self):
        if self.running:
            self._time = int(time())
        else:
            self._time = 0

    @property
    def time(self):
        if self.running:
            return int((time() - self._start_time) * self.step + self._initial_timer)
        else:
            return self._time

    @time.setter
    def time(self, t):
        if self.running:
            self._start_time = int(time())
            self._initial_timer = int(t)
        self._time = int(t)


class LogView(AutoImportDOMWidget):
    _view_name = Unicode('LogView').tag(sync=True)
    _view_module = Unicode('gcustom').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    value = Unicode("").tag(sync=True)
    filter_info = Bool(True).tag(sync=True)
    filter_error = Bool(True).tag(sync=True)
    filter_debug = Bool(True).tag(sync=True)
    filter_warning = Bool(True).tag(sync=True)
    case_sensitive = Bool(True).tag(sync=True)
    search_filter = Unicode('').tag(sync=True)

    def __init__(self, **kwargs):
        super(LogView, self).__init__(dependencies=('CustomWidgets',), **kwargs)


class LogToolBar(HBox):
    def __init__(self, logviews=None):
        if logviews is None:
            logviews = []

        line = Text(
            value='',
            placeholder='log filters',
            layout=Layout(width='200px'),
            disabled=False
        )

        icon_info = ToolButton(value=True, toggleable=True, button_color='#bdbdbd', pressed_color='#eee',
                               icon='info-circle', color='#2176ff')
        def info_color(value):
            icon_info.color = '#2176ff' if value else '#333'
        icon_info.on_value_changed(info_color)

        icon_warn = ToolButton(value=True, toggleable=True, button_color='#bdbdbd', pressed_color='#eee',
                               icon='exclamation-triangle', color='#eba427')
        def warn_color(value):
            icon_warn.color = '#eba427' if value else '#333'
        icon_warn.on_value_changed(warn_color)

        icon_error = ToolButton(value=True, toggleable=True, button_color='#bdbdbd', pressed_color='#eee',
                                icon='exclamation-circle', color='#c71e1e')

        def error_color(value):
            icon_error.color = '#c71e1e' if value else '#333'
        icon_error.on_value_changed(error_color)

        icon_debug = ToolButton(value=True, toggleable=True, button_color='#bdbdbd', pressed_color='#eee',
                                icon='code', color='#111')

        def debug_color(value):
            icon_debug.color = '#111' if value else '#333'
        icon_debug.on_value_changed(debug_color)

        space = HTML(layout=Layout(width='25px'))
        case_sensitive = HTMLButton(value=True, html='Aa', size=40, button_color='#bdbdbd', pressed_color='#eee',
                                    toggleable=True)

        super(LogToolBar, self).__init__([icon_info, icon_warn, icon_error, icon_debug, space, line, case_sensitive])
        self.info = icon_info
        self.warn = icon_warn
        self.error = icon_error
        self.debug = icon_debug
        self.filter = line
        self.case = case_sensitive
        if not isinstance(logviews, list):
            logviews = [logviews]

        self._logviews = []
        for logview in logviews:
            self.link_to(logview)

    def link_to(self, logview):
        jsdlink((self.filter, 'value'), (logview, 'search_filter'))
        jsdlink((self.info, 'value'), (logview, 'filter_info'))
        jsdlink((self.warn, 'value'), (logview, 'filter_warning'))
        jsdlink((self.error, 'value'), (logview, 'filter_error'))
        jsdlink((self.debug, 'value'), (logview, 'filter_debug'))
        jsdlink((self.case, 'value'), (logview, 'case_sensitive'))
        self._logviews.append(logview)

    @property
    def logviews(self):
        return self._logviews

    @logviews.setter
    def logviews(self, views):
        self._logviews = []
        for v in views:
            self.link_to(v)


class HTMLButton(AutoImportDOMWidget):
    _view_name = Unicode('HTMLButton').tag(sync=True)
    _view_module = Unicode('gcustom').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)

    value = Bool(False).tag(sync=True)
    toggleable = Bool(False).tag(sync=True)
    resettable = Bool(True).tag(sync=True)
    html = Unicode("").tag(sync=True)
    button_color = Unicode('#eee').tag(sync=True)
    type = Unicode('normal').tag(sync=True)
    pressed_color = Unicode('#bdbdbd').tag(sync=True)
    has_focus = Bool(False).tag(sync=True)
    _clicked = Bool(False).tag(sync=True)

    def __init__(self, size=None, layout=None,  **kwargs):

        self._callbacks = []
        self._value_callbacks = []
        if layout is None and size is not None:
            layout = Layout(width='%ipx' % size, height='%ipx'%size)
        super(HTMLButton, self).__init__(layout=layout, dependencies=('CustomWidgets',), **kwargs)

    def on_click(self, callback, append=False, **kwargs):
        if append:
            self._callbacks.append((callback, kwargs))
        else:
            self._callbacks = [(callback, kwargs)]

    def on_value_changed(self, callback, append=False, **kwargs):
        if append:
            self._value_callbacks.append((callback, kwargs))
        else:
            self._value_callbacks = [(callback, kwargs)]

    @observe('_clicked')
    def _button_clicked(self, change):
        if self._clicked:
            for cb in self._callbacks:
                match_params(cb[0], state=self.value, **cb[1])
            self._clicked = False

    @observe('value')
    def _value_changed(self, change):
        for cb in self._value_callbacks:
            match_params(cb[0], value=self.value, **cb[1])


class ToolButton(HTMLButton):
    def __init__(self, icon='question', color='#333', icon_size=20, size=40, layout=None, **kwargs):
        self._icon = icon
        self._color = color
        self._size = icon_size
        if layout is None:
            layout = Layout(width='%ipx'%size, height='%ipx'%size)
        super(ToolButton, self).__init__(html=self._html(), layout=layout, **kwargs)

    def _html(self):
        return '<i class="fa fa-%s" style="font-size: %ipx; color: %s;"></i>' \
               % (self.icon, self.icon_size, self.color)

    @property
    def icon(self):
        return self._icon

    @icon.setter
    def icon(self, icon):
        self._icon = icon
        self.html = self._html()

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, c):
        self._color = c
        self.html = self._html()

    @property
    def icon_size(self):
        return self._size

    @icon_size.setter
    def icon_size(self, size):
        self._size = size
        self.html = self._html()


class ToolBar(HBox):
    def __init__(self, icons, exclusive=False, current=-1, size=40, **kwargs):
        self.buttons = []
        for i in range(len(icons)):
            b = ToolButton(icon=icons[i], size=size, toggleable=True, resettable=current==-1)
            b.on_click(self.button_clicked, button_id=i)
            if i == 0:
                b.type = 'left'
            elif i == len(icons)-1:
                b.type = 'right'
            else:
                b.type = 'middle'
            if i == current:
                b.value = True
            self.buttons.append(b)

        self._button_callbacks = []
        self.exclusive = exclusive
        self.current_id = current

        super(ToolBar, self).__init__(self.buttons, **kwargs)

    def button_clicked(self, button_id):
        if self.exclusive:
            for i in range(len(self.buttons)):
                if i != button_id:
                    self.buttons[i].value = False
        for cb in self._button_callbacks:
            match_params(cb[0], button_id=button_id, **cb[1])

    def on_button_clicked(self, callback, append=False, **kwargs):
        if append:
            self._button_callbacks.append((callback, kwargs))
        else:
            self._button_callbacks = [(callback, kwargs)]


class HierarchyBar(DOMWidget):
    _view_name = Unicode('HierarchyBar').tag(sync=True)
    _view_module = Unicode('gcustom').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)

    value = List(default_value=[]).tag(sync=True)
    current_id = Int(-1).tag(sync=True)

    def __init__(self, list=None, **kwargs):
        if list is None:
            list = []
        self._current_callbacks = []
        super(HierarchyBar, self).__init__(value=list, **kwargs)

    def on_id_changed(self, callback, append=False, **kwargs):
        if append:
            self._current_callbacks.append((callback, kwargs))
        else:
            self._current_callbacks = [(callback, kwargs)]

    @observe('current_id')
    def _current_id_changed(self, change):
        for cb in self._current_callbacks:
            match_params(cb[0], id=self.current_id, **cb[1])


class HSpace(HTML):
    def __init__(self, space='25px'):
        if isinstance(space, int):
            space = '%ipx' % space
        super(HSpace, self).__init__(layout=Layout(width=space))

    @property
    def space(self):
        return self.layout.width

    @space.setter
    def space(self, s):
        if not isinstance(s, str):
            s = "%ipx" % s
        if s != self.space:
            self.layout.width = s


class VSpace(HTML):
    def __init__(self, space='25px'):
        if isinstance(space, int):
            space = '%ipx' % space
        super(VSpace, self).__init__(layout=Layout(height=space))

    @property
    def space(self):
        return self.layout.height

    @space.setter
    def space(self, s):
        if not isinstance(s, str):
            s = "%ipx" % s
        if s != self.space:
            self.layout.height = s
