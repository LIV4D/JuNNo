from functools import reduce
import math
import numpy as np


def prod(factor):
    """
    Equivalent to sum(list) with multiplication.
    Return the multiplication of all the elements in the list.
    """
    return reduce(lambda x, y: x*y, factor, 1)

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
class ConfMatrix(np.ndarray):
    """
    [..., true, pred]
    """
    def __new__(cls, input_array, labels=None):
        a = np.asarray(input_array).astype(np.uint).view(cls)
        a.labels = labels
        return a

    def __array_finalize__(self, obj):
        if obj is None: return
        print(obj)
        if obj.ndim < 2:
            raise ValueError('ConfMatrix must have at least 2 axis!')
        if obj.shape[-2] != obj.shape[-1]:
            raise ValueError('ConfMatrix must be a square matrix. (shape=%s)' % obj.shape)
        self.labels = getattr(obj, 'labels', None)

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
        return np.sum(self, axis=-1)

    def sum_pred(self):
        return np.sum(self, axis=-2)

    @classmethod
    def metric(cls, f):
        return f

    # @metric
    # def accuracy(self, m):
    #     return _true_positive(m) / _total(m)
    #
    # @metric
    # def TP(self, m):
    #     return _true_positive(m)


def _true_positive(m):
    return np.trace(m, axis1=-2, axis2=-1)


def _false_positive(m, false_axis=0):
    if not isinstance(false_axis, tuple):
        false_axis = (false_axis,)
    x_axis, y_axis = np.meshgrid(false_axis, false_axis)
    return m[..., y_axis.flatten(), x_axis.flatten()].sum(axis=-1)
    

def _total(m):
    return np.sum(m, axis=(-2, -1))
