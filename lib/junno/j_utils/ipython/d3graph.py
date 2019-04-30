from .import_js import import_js, AutoImportDOMWidget
from ipywidgets.widgets import HTML, DOMWidget, VBox, HBox, Text, Layout, jsdlink, ToggleButton
from traitlets import Unicode, Float, Bool, Int, validate, observe, List, TraitType


class D3LineChart(AutoImportDOMWidget):
    _view_name = Unicode('D3LineChart').tag(sync=True)
    _view_module = Unicode('gcustom').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    data = Unicode("").tag(sync=True)

    def __init__(self):
        super(D3LineChart, self).__init__(layout=Layout(width='100%'), dependencies=('D3Graph',))