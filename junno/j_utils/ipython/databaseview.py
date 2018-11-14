from traitlets import Unicode, Float, Bool, Int, validate, observe, List
from ipywidgets import DOMWidget, Layout, VBox, HBox, jsdlink, HTML, BoundedIntText
from IPython import get_ipython
import traceback

from ..math import dimensional_split
from ..j_log import log
from .customwidgets import ToolBar, RichLabel, HierarchyBar, VSpace, TinyLoading, HSpace
from .import_js import import_js, AutoImportDOMWidget

from io import BytesIO
from PIL import Image
import numpy as np
import base64


class SimpleDatabaseView(AutoImportDOMWidget):
    _view_name = Unicode('DatabaseView').tag(sync=True)
    _view_module = Unicode('gdatabaseview').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    _database_last_id = 0

    name = Unicode('nom').tag(sync=True)
    length = Int(0).tag(sync=True)
    visible = Bool(True).tag(sync=True)
    columns_name = Unicode('').tag(sync=True)
    columns_width = List().tag(sync=True)
    limit = Int(20).tag(sync=True)
    offset = Int(0).tag(sync=True)
    cache_progress = Float(0).tag(sync=True)
    clear_cache_data = Bool(False).tag(sync=True)
    _database_id = Unicode('000').tag(sync=True)

    def __init__(self, **kwargs):
        SimpleDatabaseView._database_last_id += 1
        db_id = str(SimpleDatabaseView._database_last_id)
        get_ipython().kernel.comm_manager.register_target('database_comm'+db_id, self.register_target)
        super(SimpleDatabaseView, self).__init__(layout=Layout(width='100%'), dependencies=('DatabaseView',),
                                                 _database_id=db_id, **kwargs)
        self.retreive_data = None
        self.retreive_row_name = None

    def register_target(self, comm, msg):
        self.db_comm = comm

        @comm.on_msg
        def _recv(msg):
            if self.retreive_data is None:
                return
            try:
                msg = msg['content']['data']
                request = msg[0]
                pos = msg[1:].split(',')
                if len(pos) == 2:
                    row, col = [int(_) for _ in pos]
                    channel = 0
                elif len(pos) == 3:
                    row, col, channel = [int(_) for _ in pos]
                else:
                    return

                data = self.retreive_data(row, col)
                if request == 'm':
                    # --- CREATE MINIATURE ---
                    if isinstance(data, str):
                        # --- String ---
                        data = '<p> %s </p>' % data
                    elif 'int' in str(type(data)):
                        # --- Integer ---
                        data = '<p> %i </p>' % data
                    elif 'float' in str(type(data)):
                        # --- Float ---
                        data = '<p> %f </p>' % data
                    elif isinstance(data, np.ndarray):
                        # --- Numpy Array ---
                        if len(data.shape) == 3 or (len(data.shape) == 2 and data.shape[0] > 5):
                            # -- Images --
                            img_temp = '<img src="data:image/png;base64, %s" style="max-width: none"/>'
                            if len(data.shape) == 2 or data.shape[0] == 1 or data.shape[0] == 3:
                                data = img_temp % str(SimpleDatabaseView._img2png(data, thumbnail=(128, 128)))[2:-1]
                            elif data.shape[0] % 3 == 0:
                                n = data.shape[0] // 3
                                nw, nh = dimensional_split(n)
                                img = data
                                data = '#%i,%i|' % (nw, nh)
                                for i in range(n):
                                    data += img_temp % str(SimpleDatabaseView._img2png(img[i*3:(i+1)*3], thumbnail=(128, 128)))[2:-1]

                            else:
                                nw, nh = dimensional_split(data.shape[0])
                                img = data
                                data = '#%i,%i|' % (nw, nh)
                                for i in range(img.shape[0]):
                                    data += img_temp % str(SimpleDatabaseView._img2png(img[i], thumbnail=(128, 128)))[2:-1]

                        else:
                            # -- Plot --
                            # log.debug('\tunknown: %s' % repr(data))
                            data = '<p> %s </p>' % data

                    comm.send(data)

                elif request == 'f' and isinstance(data, np.ndarray):
                    # --- SHOW IN FULLSCREEN ---
                    if len(data.shape) == 3 or (len(data.shape) == 2 and data.shape[0]>5):
                        # -- Images --
                        if len(data.shape) == 2 or data.shape[0] == 1 or data.shape[0] == 3:
                            pass
                        elif data.shape[0] % 3 == 0:
                            data = data[channel*3:(channel+1)*3]
                        else:
                            data = data[channel]
                        # -- Legend --
                        legend = ''
                        if self.retreive_row_name:
                            legend = '<h3>%s</h3>'
                        legend += '<p> min: %f<br/> max: %f<br /> mean: %f <br /> std: %f</p>'\
                                   % (data.min(), data.max(), data.mean(), data.std())
                        # -- Send data --
                        data = 'I data:image/png;base64, %s||data:image/png;base64, %s||%s' % \
                               (str(SimpleDatabaseView._img2png(data, thumbnail=(128, 128)))[2:-1],
                                str(SimpleDatabaseView._img2png(data, thumbnail=(1024, 1024)))[2:-1],
                                legend)
                        comm.send('$'+data)
            except:
                error_msg = 'Error when retreiving [%i,%i]...\n' % (row, col)
                error_msg += traceback.format_exc()
                log.error(error_msg)


    @staticmethod
    def _img2png(img, normalize=False, normalize_img=None, thumbnail=None, keep_ratio=True):
        if len(img.shape) == 3:
            img = img.transpose((1, 2, 0))
            if img.shape[2] == 1:
                img = img[:, :, 0]
            elif img.shape[2] == 3:
                img = img[:, :, ::-1]

        if normalize_img is None:
            normalize_img = img

        if np.max(normalize_img) <= 1. and np.min(normalize_img) >= 0 and not normalize:
            img = img * 255.
        elif np.max(normalize_img) < 20:
            normalize = True

        if normalize:
            img = img - np.min(normalize_img)
            img = img / (np.max(normalize_img) - np.min(normalize_img)) * 255.

        output = BytesIO()
        if thumbnail is not None:
            if keep_ratio:
                if len(img.shape) == 2:
                    h, w = img.shape
                else:
                    h, w, l = img.shape
                ratio = h / w
                mindim = min(thumbnail[0] * ratio, thumbnail[1])
                thumbnail = (round(mindim / ratio), round(mindim))

            img = Image.fromarray(img.astype(dtype=np.uint8))
            img.thumbnail(thumbnail, Image.ANTIALIAS)
        else:
            img = Image.fromarray(img.astype(dtype=np.uint8))
        img.save(output, 'png')
        return base64.b64encode(output.getvalue())

    def reset(self):
        self.clear_cache_data = True


class DatabaseView(VBox):
    def __init__(self, path):
        self.db_view = SimpleDatabaseView()

        toolbar = ToolBar(['eye-slash', 'list', 'image'], exclusive=True, current=0)
        self.toolbar = toolbar
        self.hierarchy_bar = HierarchyBar(path, layout=Layout(width='70%'))
        self.length_label = RichLabel('[0]')
        self.header = HBox([toolbar,  HTML(layout=Layout(width='25px')),
                            self.hierarchy_bar, HTML(layout=Layout(width='40px')),
                            self.length_label], layout=Layout(align_content='flex-end'))

        self.offset_edit = BoundedIntText(description='offset: ', value=self.db_view.offset, min=0,
                                          layout=Layout(width='150px'))
        self.limit_edit = BoundedIntText(description='limit: ', value=self.db_view.limit, min=0, max=self.db_view.limit,
                                          layout=Layout(width='150px'))
        self.progress = TinyLoading(layout=Layout(width='150px'))
        self.foot = HBox([self.offset_edit, self.limit_edit, HSpace(), self.progress],
                         layout=Layout(align_items='center'))

        jsdlink((self.offset_edit, 'value'), (self.db_view, 'offset'))
        jsdlink((self.limit_edit, 'value'), (self.db_view, 'limit'))
        jsdlink((self.db_view, 'cache_progress'), (self.progress, 'value'))

        def toolbar_handling(button_id):
            self.db_view.visible = button_id != 0
            self.foot.layout.display = 'none' if button_id == 0 else 'flex'
        self.toolbar.on_button_clicked(toolbar_handling)
        toolbar_handling(0)

        super(DatabaseView, self).__init__([self.header, VSpace(), self.db_view, self.foot])

    @property
    def length(self):
        return self.db_view.length

    @length.setter
    def length(self, l):
        self.db_view.length = l
        self.length_label.value = '[%i rows]' % l
        self.limit_edit.max = l
        self.offset_edit.max = l-1

    @property
    def name(self):
        return self.name_label.value

    @property
    def columns_name(self):
        return self.db_view.columns_name.split('|')

    @columns_name.setter
    def columns_name(self, names):
        self.db_view.columns_name = str('|').join(names)

