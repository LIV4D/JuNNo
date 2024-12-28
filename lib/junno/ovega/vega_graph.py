from traitlets import Unicode, Int
from os.path import dirname
from IPython import get_ipython
from weakref import ref

from ..j_utils.json_template import JSONClass, JSONAttr
from ..j_utils.ipython.import_js import AutoImportDOMWidget

from .vega_data import Data
from .vega_mark import Mark
from .vega_chart_area import Scale, Axes, Legend
from .vega_interactive import Signal


class VegaGraph(AutoImportDOMWidget):
    _view_name = Unicode('VegaGraph').tag(sync=True)
    _view_module = Unicode('vega_graph').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    _vega_last_uuid = 0

    _spec_str = Unicode('').tag(sync=True)
    _width = Int(500).tag(sync=True)
    _height = Int(300).tag(sync=True)
    _uuid = Unicode('000').tag(sync=True)

    def __init__(self, interactive=True):
        VegaGraph._vega_last_uuid += 1
        uuid = str(VegaGraph._vega_last_uuid)

        self.interactive = interactive
        if interactive:
            get_ipython().kernel.comm_manager.register_target('vegagraph_comm'+uuid, self._register_comm)
        super(VegaGraph, self).__init__(dependencies='vega_graph', path=dirname(__file__) + '/js_lib', _uuid=uuid)

        self._spec = None
        self._upload_spec_timeout = None

    def __del__(self):
        self._spec.unassign_chart(self)

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, s):
        if self._spec is not None:
            self._spec.unassign_chart(self)
        self._spec = s
        self.update_spec()
        s.assign_chart(self)

    def update_spec(self, timeout=True):
        def upload_spec():
            import json
            self._upload_spec_timeout = None
            spec = self._spec.to_json()
            spec = json.dumps(spec)
            self._spec_str = spec
        if timeout:
            if self._upload_spec_timeout is None:
                from threading import Timer
                self._upload_spec_timeout = Timer(.5, upload_spec)
        else:
            upload_spec()

    def _register_comm(self, comm, msg):
        self._comm = comm


    def signal(self, name, value=None):
        self._comm.send('S')


class VegaSpec(JSONClass):
    description = JSONAttr.String()
    width = JSONAttr.Int(500)
    height = JSONAttr.Int(300)
    padding = JSONAttr.Int(5)
    autosize = JSONAttr.String('fit')
    title = JSONAttr.String(optional=True)
    signals = JSONAttr.List(Signal, key='name')
    data = JSONAttr.List(Data, key='name')
    scales = JSONAttr.List(Scale, key='name')
    axes = JSONAttr.List(Axes, key='name')
    legends = JSONAttr.List(Legend, key='name')
    marks = JSONAttr.List(Mark, key='name')

    def __init__(self):
        super(VegaSpec, self).__init__()
        self._graphs = []

    def assign_chart(self, chart):
        self._graphs.append(ref(chart))

    def unassign_chart(self, chart):
        for i, r in self._graphs:
            if r() is chart:
                del self._graphs[i]

    def apply_change(self):
        for graph in self.graphs:
            graph.update_spec()

    @property
    def graphs(self):
        graphs = []
        weakref_graphs = []
        for i, r in enumerate(self._graphs):
            g = r()
            if g is not None:
                graphs.append(g)
                weakref_graphs.append(r)
        self._graphs = weakref_graphs
        return graphs

