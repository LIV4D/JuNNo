from traitlets import Unicode, Int
from os.path import dirname
from IPython import get_ipython

from ..j_utils.json_template import JSONClass, JSONAttr
from ..j_utils.ipython.import_js import AutoImportDOMWidget


class VegaGraph(AutoImportDOMWidget):
    _view_name = Unicode('VegaGraph').tag(sync=True)
    _view_module = Unicode('vega_graph').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    _vega_last_uuid = 0

    _spec = Unicode('').tag(sync=True)
    _width = Int(500).tag(sync=True)
    _height = Int(300).tag(sync=True)
    _uuid = Unicode('000').tag(sync=True)

    def __init__(self):
        VegaGraph._vega_last_uuid += 1
        uuid = str(VegaGraph._vega_last_uuid)
        # get_ipython().kernel.comm_manager.register_target('vegagraph_comm'+uuid, self._register_comm)
        super(VegaGraph, self).__init__(dependencies=('vega_graph'), path=dirname(__file__)+'/js_lib', _uuid=uuid)

    def _register_comm(self, comm, msg):
        self._comm = comm


class VegaSpec(JSONClass):
    description = JSONAttr.String()
    width = JSONAttr.Int(500)
    height = JSONAttr.Int(200)
    padding = JSONAttr.Int(5)
    autosize = JSONAttr.String()
