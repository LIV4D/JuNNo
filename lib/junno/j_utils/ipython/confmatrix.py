from traitlets import Unicode

from ..math import ConfMatrix
from .import_js import AutoImportDOMWidget


class ConfMatrixView(AutoImportDOMWidget):
    _view_name = Unicode('ConfMatrix').tag(sync=True)
    _view_module = Unicode('gconfmatrix').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)

    labelsStr = Unicode('').tag(sync=True)
    dataStr = Unicode('').tag(sync=True)
    normed = Unicode('none').tag(sync=True)

    def __init__(self, data, labels=None, normed='none'):
        if isinstance(data, ConfMatrix):
            normed = data.norm
            data = data.no_normed()
            if labels is None:
                labels = data.labels
        if labels is None:
            labels = list(range(len(data)))
        labelsStr = '|'.join(str(_) for _ in labels)
        dataStr = ';'.join(','.join(str(_) for _ in r) for r in data)

        super(ConfMatrixView, self).__init__(dependencies=('ConfMatrixView',), labelsStr=labelsStr, dataStr=dataStr,
                                             normed=normed)

    @property
    def labels(self):
        return self.labelsStr.split('|')

    @labels.setter
    def labels(self, l):
        self.labelsStr = '|'.join(l)

    @property
    def data(self):
        rows = self.dataStr.split(';')
        return ConfMatrix((int(_.split(',')) for _ in rows), self.labels.split('|'))

    @data.setter
    def data(self, d):
        if isinstance(d, ConfMatrix):
            self.labels = '|'.join(d.labels)
            self.normed = d.norm
            d = d.no_normed()
        self.dataStr = ';'.join(','.join(str(_)) for _ in d)
