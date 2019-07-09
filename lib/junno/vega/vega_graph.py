from ..j_utils.json_template import JSONClass, JSONAttr
from traitlets import Unicode
from ..j_utils.ipython.import_js import AutoImportDOMWidget


class VegaGraph(AutoImportDOMWidget):
    _view_name = Unicode('VegaGraph').tag(sync=True)
    _view_module = Unicode('vega_graph').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    model = Unicode('').tag(sync=True)

    def __init__(self):
        super(VegaGraph, self).__init__()
