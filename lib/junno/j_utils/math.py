import functools
from ..j_utils.function import bind_args
import math
import numpy as np


def prod(factor):
    """
    Equivalent to sum(list) with multiplication.
    Return the multiplication of all the elements in the list.
    """
    return functools.reduce(lambda x, y: x*y, factor, 1)


def quadratic_kappa(conf_matrix):
    """
    Compute the kappa factor from a confusion matrix
    :param conf_matrix: Each row should correspond to true values, and each column to predicted values
    """
    import numpy as np

    nb_class = conf_matrix.shape[0]
    ratings_mat = np.tile(np.arange(0, nb_class)[:, None], reps=(1, nb_class))
    ratings_squared = (ratings_mat - ratings_mat.T) ** 2

    weights = (ratings_squared / (nb_class - 1) ** 2)

    hist_rater_a = conf_matrix.sum(axis=1)  # Predicated totals
    hist_rater_b = conf_matrix.sum(axis=0)  # True value totals
    n = hist_rater_a.sum()

    # The nominator.
    nom = np.sum(weights * conf_matrix)
    expected_probs = np.dot(hist_rater_a[:, None], hist_rater_b[None, :])
    # The denominator.
    denom = np.sum(weights * expected_probs / n)

    return 1. - nom / denom


def kappa(conf_matrix):
    """
    Compute the kappa factor from a confusion matrix
    :param conf_matrix: Each row should correspond to true values, and each column to predicted values
    """
    import numpy as np

    nb_class = conf_matrix.shape[0]

    hist_rater_a = conf_matrix.sum(axis=1)  # Predicated totals
    hist_rater_b = conf_matrix.sum(axis=0)  # True value totals
    n = hist_rater_a.sum()

    effectiveAgreement = np.sum(np.eye(nb_class) * conf_matrix) / n
    theoricAgreement = np.sum((hist_rater_a * hist_rater_b)) / (n ** 2)
    return (effectiveAgreement - theoricAgreement) / (1 - theoricAgreement)


def dimensional_split(n, dim=2, perfect=True):
    from sympy.ntheory import factorint
    if not perfect:
        dim_n = math.pow(n, 1/dim)
        return [math.ceil(dim_n)] * (dim-1) + [math.ceil(n/math.pow(dim_n,dim-1))]
    else:
        pfactor_count = factorint(n)
        dim_list = []
        pfactor = list(pfactor_count.keys())
        pfactor.sort()
        for p in pfactor:
            dim_list += [p] * pfactor_count[p]

        if len(dim_list) < dim:
            return ([1]*(dim-len(dim_list))) + dim_list

        def symetric_reduce(l):
            r = l[:math.ceil(len(l)/2)]
            for i in range(int(len(l)/2)):
                r[i] *= l[-i-1]
            return r

        while len(dim_list) != dim:
            if len(dim_list) / 2 < dim:
                id = 2*(len(dim_list) - dim)
                dim_list = symetric_reduce(dim_list[:id]) + dim_list[id:]
            else:
                dim_list = symetric_reduce(dim_list)
            dim_list.sort()

    return dim_list


def index_decomposition(index, base):
    r = []
    for b in reversed(base):
        q = index % b
        index = (index-q) // b
        r.insert(0, q)
    return tuple(r)


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    import numpy as np

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def interval(size, start=None, stop=None, args=()):
    if len(args)==1:
        stop = args[0]
        start = 0
    elif len(args)==2:
        start = args[0]
        stop = args[1]

    if start is None:
        start = 0
    if stop is None:
        stop = size
    elif stop < 0:
        stop += size
    elif stop < start:
        stop = start + 1
    if stop > size:
        raise ValueError('%i is not a valid index: size is %i' % (stop, size))
    return start, stop


def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    import numpy as np

    x = np.swapaxes(x, 0, axis)
    e_x = np.exp(x - np.max(x, axis=0))
    r = e_x / e_x.sum(axis=0)
    return np.swapaxes(r, 0, axis)


def apply_scale(a, range=None, domain=None, clip=None):
    from .collections import Interval
    import numpy as np

    range = Interval(range)
    domain = Interval(domain)
    clip = Interval(clip)

    if range:
        if domain.min is None:
            domain.min = np.min(a)
        if domain.max is None:
            domain.max = np.max(a)

        if range.min is None:
            range.min = domain.min
        elif range.max is None:
            range.max = domain.max

        a = range.min + (a-domain.min) * range.length / domain.length

    if clip.min is not None:
        a[a < clip.min] = clip.min
    if clip.max is not None:
        a[a > clip.max] = clip.max
    return a

########################################################################################################################
#              -------- THEANO IMPROVMENTS -------
# def merge_axis(a, keep_axis=None, shape=None):
#     import numpy as np
#
#     if keep_axis is not None and keep_axis < 0:
#         keep_axis += a.dim
#
#     if keep_axis is None or keep_axis == 0:
#         a = a.flatten(ndim=1)
#     elif keep_axis == 1 and a.dim == 2:
#         pass
#     else:
#         a = np.move_axis(a, keep_axis, 0)
#         a = a.flatten(ndim=2).dimshuffle((1, 0))
#
#     if shape is None:
#         return a
#     else:
#         info = {'axis': keep_axis, 'shape': np.move_axis(list(shape), keep_axis, 0)}
#         return a, info
#
#
# def unmerge_axis(a, info, output_shape=False):
#     import numpy as np
#
#     axis = info['axis']
#     shape = tuple(info['shape'])
#     if axis is not None and axis > 0:
#         if axis > 1 or len(shape)!=2:
#             a = a.dimshuffle((1,0)).reshape(shape)
#             a = np.move_axis(a, 0, axis)
#     else:
#         a = a.reshape(shape)
#
#     if output_shape:
#         return a, move_axis(shape, 0, axis)
#     else:
#         return a
#
#


########################################################################################################################
class ROCCurve(np.ndarray):
    def __new__(cls, input_array, labels=None):
        a = np.asarray(input_array).astype(np.float).view(cls)
        return a

    @property
    def tpr(self):
        return self[..., 0]

    @property
    def fpr(self):
        return self[..., 1]

    @property
    def thresholds(self):
        return self[..., 2]

    def __array_finalize__(self, obj):
        if obj is None: return
        if obj.ndim < 2:
            print(obj, obj.shape)
            raise ValueError('ROCCurve must have at least 2 axis!')
        if obj.shape[-1] != 3:
            raise ValueError("The length of ROCCurve's last axis must be 3 and not %i." % obj.shape[-1])
        from sklearn.metrics import auc
        if hasattr(obj, 'auc'):
            self.auc = obj.auc
        else:
            self.auc = auc(self.fpr, self.tpr)

    @staticmethod
    def create(tpr, fpr, thresholds):
        return np.stack([tpr, fpr, thresholds], axis=-1).view(ROCCurve)

    @staticmethod
    def roc_curve(score, true, sample_weight=None, negative_label=0):
        if np.prod(score.shape) != np.prod(true.shape):
            raise ValueError("Proba and True doesn't have the same shape (proba: %s, true: %s)." %
                             (score.shape, true.shape))
        if true.dtype != np.bool:
            true = true != negative_label

        score = score.flatten()
        true = true.flatten()

        # Sort proba
        idx_sorted = np.argsort(score, kind="quicksort")[::-1]
        true = true[idx_sorted]
        score = score[idx_sorted]
        if sample_weight is not None:
            sample_weight = sample_weight.flatten()[idx_sorted]
        else:
            sample_weight = 1.

        idx_threshold = np.where(np.diff(score))[0]
        idx_threshold = np.r_[idx_threshold, true.size - 1]

        tpr = np.cumsum(true*sample_weight)[idx_threshold]
        if isinstance(sample_weight, float) and sample_weight == 1.:
            fpr = 1 + idx_threshold - tpr
        else:
            fpr = np.cumsum(np.invert(true)*sample_weight)[idx_threshold]

        tpr = tpr / tpr[-1]
        fpr = fpr / fpr[-1]

        thresholds = score[idx_threshold]
        return np.stack([tpr, fpr, thresholds], axis=-1).view(ROCCurve)

    def __getitem__(self, item):
        return self.view(np.ndarray)[item]

    def _ipython_display_(self):
        from IPython.display import display
        display(self.vegalite_graph())

    def vegalite_graph(self, shape=(400, 300), simple=False):
        from vega import VegaLite
        import pandas
        idx = np.linspace(0, self.fpr.size - 1, 500, dtype=np.uint)
        return VegaLite({
            "title": {
                "text": "AUC = %.3f       " % self.auc,
                "anchor": "end"
            },
            "width": shape[0],
            "height": shape[1],
            "mark": {"type": "area", "line": True, "point": True},
            "encoding": {
                "y": {"type": "quantitative", "field": "tpr", "title": "True Positive Rate"},
                "x": {"type": "quantitative", "field": "fpr", "title": "False Positive Rate"},
                "tooltip": [{"type": "quantitative", "field": "tpr", "title": "True Positive"},
                            {"type": "quantitative", "field": "fpr", "title": "False Positive"},
                            {"type": "quantitative", "field": "thresholds", "title": "Threshold"}]
            }
        }, pandas.DataFrame(dict(fpr=self.fpr[idx], tpr=self.tpr[idx], thresholds=self.thresholds[idx])))


########################################################################################################################
def metric(alias=None):
    def decorator(func):
        @functools.wraps(func)
        def compute_metric(self, *params, **kwargs):
            if isinstance(self, ConfMatrix):
                m = self.no_normed().view(np.ndarray)
            else:
                m = np.ndarray(self)
            params = bind_args(func, *params, **kwargs)
            return func(m=m, **params)

        _metrics[func.__name__] = compute_metric
        if alias is not None:
            if isinstance(alias, str):
                _metrics[alias] = compute_metric
            else:
                for a in alias:
                    _metrics[a] = compute_metric
        return compute_metric
    return decorator


_metrics = {}


class ConfMatrix(np.ndarray):
    """
    [..., pred, true]
    """

    @staticmethod
    def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None):
        if isinstance(labels, int):
            labels = list(range(labels))
        if np.prod(y_true.shape)<=100 or labels is None or len(labels) <= 3:
            # For small or huge computation use memory optimized method
            from sklearn.metrics import confusion_matrix
            conf = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels, sample_weight=sample_weight)
            conf = conf.view(ConfMatrix)
            conf.labels = labels
            return conf
        else:
            # Else use computation-time optimized method
            conf = ConfMatrix.zeros(labels)
            labels = conf.labels
            if sample_weight is None:
                sample_weight = 1

            a = np.stack([y_true, y_pred], axis=0)

            for i in labels:
                for j in labels:
                    conf[i, j] = np.sum(((a[0] == i) & (a[1] == j))*sample_weight)

            return conf

    @staticmethod
    def zeros(labels):
        if isinstance(labels, int):
            labels = list(range(labels))
        elif not isinstance(labels, (list, tuple)):
            raise ValueError('ConfMatrix.zeros(labels=...) should be a list. (labels=%s)' % repr(labels))
        conf = np.zeros((len(labels), len(labels)), np.int)
        conf = conf.view(ConfMatrix)
        conf.labels = labels
        return conf

    def __new__(cls, input_array, labels=None):
        a = np.asarray(input_array).astype(np.int).view(cls)
        a.labels = labels
        a.total = None
        a.total_dim = None
        return a

    def __array_finalize__(self, obj):
        if obj is None: return
        if obj.ndim < 2:
            raise ValueError('ConfMatrix must have at least 2 axis!')
        if obj.shape[-2] != obj.shape[-1]:
            raise ValueError('ConfMatrix must be a square matrix. (shape=%s)' % obj.shape)
        self.labels = getattr(obj, 'labels', None)
        self.total = getattr(obj, 'total', None)
        self.total_dim = getattr(obj, 'total_dim', None)

    def _ipython_display_(self):
        from .ipython import import_display
        from .ipython.confmatrix import ConfMatrixView
        import_display(ConfMatrixView(self, labels=self.labels, normed=self.norm))

    def __call__(self, *labels, **labels_map):
        from collections import OrderedDict
        labels_dict = OrderedDict()
        for l in labels:
            labels_dict[l] = l
        for k, v in labels_map.items():
            if isinstance(v, str):
                labels_dict[k] = [self.labels.index(v)]
            elif isinstance(v, tuple):
                labels_dict[k] = tuple(self.labels.index(_) if isinstance(_, str) else _ for _ in v)
            else:
                labels_dict[k] = [v]

        d = self.no_normed()
        r = ConfMatrix.zeros([str(_) for _ in labels_dict.keys()])
        for i, y in enumerate(labels_dict.values()):
            for j, x in enumerate(labels_dict.values()):
                ids = cartesian((y, x))
                r[i, j] = np.sum(d[ids[:, 0], ids[:, 1]])
        return r

    def __getitem__(self, item):
        from .collections import istypeof_or_collectionof
        if self.labels is not None:
            if istypeof_or_collectionof(item, (str,), recursive=True):
                if isinstance(item, str):
                    item = (item,)
                labels = []
                for label in item:
                    if label not in self.labels:
                        raise KeyError('Unkown label: %s.' % label)
                    labels.append(self.labels.index(label))
                x_labels, y_labels = np.meshgrid(labels, labels)
                a = self[..., y_labels.astype(np.uint32), x_labels.astype(np.uint32)]
                if isinstance(a, ConfMatrix):
                    a.labels = list(item)
                return a

            elif isinstance(item, set) and all(isinstance(_, int) for _ in item):
                item = list(item)
                x_labels, y_labels = np.meshgrid(item, item)
                a = self[..., y_labels.astype(np.int32), x_labels.astype(np.int32)]
                a.labels = [self.labels[_] for _ in item]
                return a
            elif isinstance(item, tuple) and self.ndim > 1:
                if not item:
                    return self.copy()
                elif item[0] is Ellipsis:
                    if len(item) == 3:
                        l_pred = set(item[1].flatten())
                        l_true = set(item[2].flatten())
                        if l_pred == l_true:
                            a = super(ConfMatrix, self).__getitem__(item)
                            if isinstance(a, ConfMatrix):
                                a.labels = list(l_pred)
                            return a
                elif len(item) == self.ndim-2:
                    return super(ConfMatrix, self).__getitem__(item)
                elif len(item) == self.ndim:
                    try:
                        l_pred = [self.labels[_] for _ in item[-2]]
                    except TypeError:
                        l_pred = [self.labels[item[-2]]]
                    try:
                        l_true = [self.labels[_] for _ in item[-1]]
                    except TypeError:
                        l_true = [self.labels[item[-1]]]

                    if l_pred == l_true:
                        a = super(ConfMatrix, self).__getitem__(item)
                        if isinstance(a, ConfMatrix):
                            a.labels = l_pred
                        return a
            elif self.ndim > 2:
                return super(ConfMatrix, self).__getitem__(item)
        return self.view(np.ndarray)[item]

    def sum_true(self):
        return np.sum(self, axis=-2)

    def sum_pred(self):
        return np.sum(self, axis=-1)

    def normed_pred(self):
        dim = (slice(None),)*(self.ndim-2) + (np.newaxis, slice(None))
        a = self.no_normed()
        t = a.sum_true()
        a = a / t[dim]
        a.total = t
        a.total_dim = 'pred'
        return a

    def normed_true(self):
        dim = (slice(None),)*(self.ndim-2) + (slice(None), np.newaxis)
        a = self.no_normed()
        t = a.sum_pred()
        a = a / t[dim]
        a.total = t
        a.total_dim = 'true'
        return a

    def no_normed(self):
        if self.total_dim == "pred":
            dim = (slice(None),) * (self.ndim - 2) + (np.newaxis, slice(None))
            a = self * self.total[dim]
            a.total = None
            a.total_dim = None
            return a
        elif self.total_dim == "true":
            dim = (slice(None),) * (self.ndim - 2) + (slice(None), np.newaxis)
            a = self * self.total[dim]
            a.total = None
            a.total_dim = None
            return a
        return self

    @property
    def norm(self):
        if self.total_dim in ("true", "pred"):
            return self.total_dim
        else:
            return "none"

    def _metric_flatten(self):
        return self.view(np.ndarray)

    @staticmethod
    def metrics(name):
        m = _metrics.get(name, None)
        if m is None:
            raise ValueError('Unkown metric %s.' % name)
        return _metrics[name]

    @staticmethod
    def metrics_name():
        return list(_metrics.keys())

    def to_metric(self, metric_name, **kwargs):
        return self.metrics(metric_name)(self, **kwargs)

    @metric()
    def accuracy(m):
        return _true(m) / _total(m)

    @metric(alias=('sensitivity', 'recall'))
    def true_positive_rate(m, negative_axis=0, average=None):
        if average == 'micro' or average is True:
            negative_axis = None
        if average == 'macro':
            l = m.shape[-1]
            return np.mean(ConfMatrix.true_positive_rate(m, i) for i in range(l))
        else:
            tp = _true_positive(m, negative_axis)
            fn = _false_negative(m, negative_axis)
            return tp / (tp+fn)
    sensitivity = true_positive_rate
    recall = true_positive_rate

    @metric()
    def false_negative_rate(m, negative_axis=0, average=None):
        if average == 'micro' or average is True:
            negative_axis = None
        if average == 'macro':
            l = m.shape[-1]
            return np.mean(ConfMatrix.false_negative_rate(m, i) for i in range(l))
        else:
            tp = _true_positive(m, negative_axis)
            fn = _false_negative(m, negative_axis)
        return fn / (tp + fn)

    @metric(alias='specificity')
    def true_negative_rate(m, negative_axis=0, average=None):
        if average == 'micro' or average is True:
            negative_axis = None
        if average == 'macro':
            l = m.shape[-1]
            return np.mean(ConfMatrix.true_negative_rate(m, i) for i in range(l))
        else:
            tn = _true_negative(m, negative_axis)
            fp = _false_positive(m, negative_axis)
        return tn / (tn+fp)
    specifity = true_negative_rate

    @metric()
    def false_positive_rate(m, negative_axis=0, average=None):
        if average == 'micro' or average is True:
            negative_axis = None
        if average == 'macro':
            l = m.shape[-1]
            return np.mean(ConfMatrix.false_positive_rate(m, i) for i in range(l))
        else:
            tn = _true_negative(m, negative_axis)
            fp = _false_positive(m, negative_axis)
        return fp / (tn + fp)

    @metric(alias='precision')
    def positive_predictive_value(m, negative_axis=0, average=None):
        if average == 'micro' or average is True:
            negative_axis = None
        if average == 'macro':
            l = m.shape[-1]
            return np.mean(ConfMatrix.positive_predictive_value(m, i) for i in range(l))
        else:
            tp = _true_positive(m, negative_axis)
            fp = _false_positive(m, negative_axis)
            return tp / (tp+fp)
    precision = positive_predictive_value

    @metric()
    def false_discovery_rate(m, negative_axis=0, average=None):
        if average == 'micro' or average is True:
            negative_axis = None
        if average == 'macro':
            l = m.shape[-1]
            return np.mean(ConfMatrix.false_discovery_rate(m, i) for i in range(l))
        else:
            tp = _true_positive(m, negative_axis)
            fp = _false_positive(m, negative_axis)
            return fp / (tp + fp)

    @metric()
    def negative_predictive_value(m, negative_axis=0, average=None):
        if average == 'micro' or average is True:
            negative_axis = None
        if average == 'macro':
            l = m.shape[-1]
            return np.mean(ConfMatrix.negative_predictive_value(m, i) for i in range(l))
        else:
            tn = _true_negative(m, negative_axis)
            fn = _false_negative(m, negative_axis)
            return tn / (tn+fn)

    @metric()
    def false_omission_rate(m, negative_axis=0, average=None):
        if average == 'micro' or average is True:
            negative_axis = None
        if average == 'macro':
            l = m.shape[-1]
            return np.mean(ConfMatrix.false_omission_rate(m, i) for i in range(l))
        else:
            tn = _true_negative(m, negative_axis)
            fn = _false_negative(m, negative_axis)
            return fn / (tn+fn)

    @metric()
    def diagnostic_odd_ratio(m, negative_axis=0, average=None):
        if average == 'micro' or average is True:
            negative_axis = None
        if average == 'macro':
            l = m.shape[-1]
            return np.mean(ConfMatrix.diagnostic_odd_ratio(m, i) for i in range(l))
        else:
            tn = _true_negative(m, negative_axis)
            fn = _false_negative(m, negative_axis)
            tp = _true_positive(m, negative_axis)
            fp = _false_positive(m, negative_axis)
            return (tp*tn) / (fp*fn)

    @metric()
    def TP(m, negative_axis=0):
        return _true_positive(m, negative_axis)

    @metric()
    def FP(m, negative_axis=0):
        return _false_positive(m, negative_axis)

    @metric()
    def TN(m, negative_axis=0):
        return _true_negative(m, negative_axis)

    @metric()
    def FN(m, negative_axis=0):
        return _false_negative(m, negative_axis)

    @metric()
    def kappa(m):
        total_preds = m.sum(axis=-1)  # Predicated totals
        total_trues = m.sum(axis=-2)  # True value totals
        n = total_preds.sum(axis=-1)

        effectiveAgreement = np.trace(m, axis1=-1, axis2=-2) / n
        theoricAgreement = (total_preds * total_trues).sum(axis=-1) / (n ** 2)
        return (effectiveAgreement - theoricAgreement) / (1 - theoricAgreement)

    @metric()
    def kappa_quadratic(m):
        nb_class = m.shape[-1]
        ratings_mat = np.tile(np.arange(0, nb_class)[:, None], reps=(1, nb_class))
        ratings_squared = (ratings_mat - ratings_mat.T) ** 2

        weights = (ratings_squared / (nb_class - 1) ** 2)

        total_trues = m.sum(axis=-2)  # True value totals
        total_preds = m.sum(axis=-1)  # Predicated totals
        n = total_preds.sum(axis=-1)

        # The nominator.
        nom = np.sum(m * weights, axis=(-1, -2))
        expected_probs = np.einsum('...i,...j->...ij', total_trues, total_preds)
        # The denominator.
        denom = np.sum(weights * expected_probs / n, axis=(-1, -2))

        return 1. - nom / denom

    @metric()
    def dice(m, smooth=0):
        total_trues = m.sum(axis=-2)  # True value totals
        total_preds = m.sum(axis=-1)  # Predicated totals
        intersection = np.trace(m, axis1=-1, axis2=-2)
        double_union = total_trues + total_preds
        return (2. * intersection + smooth) / (double_union + smooth)

    @metric()
    def F1(m, negative_axis=0, average=None):
        pre = ConfMatrix.precision(m, negative_axis=negative_axis, average=average)
        rec = ConfMatrix.recall(m, negative_axis=negative_axis, average=average)
        return 2*pre*rec / (pre+rec)


def _true(m):
    return np.trace(m, axis1=-2, axis2=-1)


def _true_positive(m, negative_axis=0):
    if negative_axis is None:
        return np.trace(m, axis1=-2, axis2=-1).sum()
    if not isinstance(negative_axis, tuple):
        negative_axis = (negative_axis,)
    positive_axis = [_ for _ in range(m.shape[-1]) if _ not in negative_axis]
    ids = cartesian((positive_axis, positive_axis))
    return m[..., ids[:, 0], ids[:, 1]].sum(axis=-1)


def _true_negative(m, negative_axis=0):
    if negative_axis is None:
        return np.trace(m, axis1=-2, axis2=-1).sum() + m.sum() * (m.shape[-1]-2)
    if not isinstance(negative_axis, tuple):
        negative_axis = (negative_axis,)
    ids = cartesian((negative_axis, negative_axis))
    return m[..., ids[:, 0], ids[:, 1]].sum(axis=-1)


def _false_positive(m, negative_axis=0):
    if negative_axis is None:
        return m.sum() - np.trace(m, axis1=-2, axis2=-1).sum()
    if not isinstance(negative_axis, tuple):
        negative_axis = (negative_axis,)
    positive_axis = [_ for _ in range(m.shape[-1]) if _ not in negative_axis]
    ids = cartesian((positive_axis, negative_axis))
    return m[..., ids[:, 0], ids[:, 1]].sum(axis=-1)


def _false_negative(m, negative_axis=0):
    if negative_axis is None:
        return m.sum() - np.trace(m, axis1=-2, axis2=-1).sum()
    if not isinstance(negative_axis, tuple):
        negative_axis = (negative_axis,)
    positive_axis = [_ for _ in range(m.shape[-1]) if _ not in negative_axis]
    ids = cartesian((negative_axis, positive_axis))
    return m[..., ids[:, 0], ids[:, 1]].sum(axis=-1)


def _total(m):
    return np.sum(m, axis=(-2, -1))
