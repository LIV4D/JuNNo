import cv2
from copy import copy
import functools
import numpy as np

from .dataset import AbstractDataSet, DSColumn
from .dataset_generator import DataSetResult
from ..j_utils.collections import if_none
from ..j_utils.function import bind_args, bind_args_partial, match_params


_augment_methods = {}
_augment_by_type = {}


def augment_method(augment_type=None, cv=False):
    def decorator(func):
        @functools.wraps(func)
        def register_augment(self, *params, **kwargs):
            params = bind_args(func, *params, **kwargs)
            self._augment_stack.append((func.__name__, params))
            return self

        _augment_methods[func.__name__] = func, cv, augment_type
        _augment_by_type[augment_type] = func.__name__
        return register_augment

    return decorator


class DataAugment:
    def __init__(self, rng=None):
        self._augment_stack = []
        self._rng_stack = [np.random.RandomState(1234)]
        self.rng = rng

    @property
    def rng(self):
        return self._rng_stack[-1]

    @rng.setter
    def rng(self, rng):
        if rng is None:
            self._rng_stack[-1] = np.random.RandomState(1234)
        elif isinstance(rng, int):
            self._rng_stack[-1] = np.random.RandomState(rng)
        elif isinstance(rng, np.random.RandomState):
            self._rng_stack[-1] = copy(rng)
        else:
            raise ValueError('Rng seed is expected to be an int or a RandomState (received %s).' % type(rng))

    def random_seed(self):
        self.rng = np.random.randint(0, 100000)

    def push_rng(self):
        self._rng_stack.append(copy(self._rng_stack[-1]))

    def pop_rng(self):
        self._rng_stack.pop()

    def __call__(self, x, **kwargs):
        f = self.compile(**kwargs)
        f_label = None

        if isinstance(x, DataSetResult):
            r = x.copy()
            for c in r.col:
                if c.ndim >= 2:
                    if c.format.is_label:
                        if f_label is None:
                            k = copy(kwargs)
                            k['interpolation'] = cv2.INTER_NEAREST
                            k['except_type'] = set(k.get('except_type', ())).union({'color'})
                            f_label = self.compile(**k)
                        r[c] = f_label(x[c])
                    else:
                        r[c] = f(x[c])
            return r
        else:
            return f(x)

    def compile(self, only_type=None, except_type=None, **kwargs):
        cv_kwargs = kwargs.pop('cv') if 'cv' in kwargs else False

        if isinstance(only_type, str):
            only_type = (only_type,)
        if isinstance(except_type, str):
            except_type = (except_type,)

        augment_stack = []
        cv = cv_kwargs
        for f_name, f_params in self._augment_stack:
            f, f_cv, a_type = _augment_methods[f_name][:3]

            if a_type is not None\
               and((only_type is not None and a_type not in only_type)
               or (except_type is not None and a_type in except_type)):
                continue

            if cv != f_cv:
                augment_stack.append(DataAugment.split_cv if f_cv else DataAugment.merge_cv)
                cv = f_cv

            params = bind_args_partial(f, **kwargs)
            params.update(f_params)
            augment_stack.append(match_params(f, self=self, **params))

        if cv != cv_kwargs:
            augment_stack.append(DataAugment.split_cv if cv_kwargs else DataAugment.merge_cv)

        def augment(x, rng=None):
            if rng is None:
                rng = copy(self.rng)
            x_shape = x.shape
            x_dtype = x.dtype
            if cv_kwargs:
                if x.ndim == 2:
                    x = x[:, :, np.newaxis]
                elif x.ndim != 3:
                    raise ValueError('Invalid cv image format, shape is: %s' % repr(x.shape))
            else:
                if x.ndim == 2:
                    x = x[np.newaxis, np.newaxis, :, :]
                elif x.ndim > 3:
                    x = x.reshape(-1, *x.shape[-2:])
                elif x.ndim != 3:
                    raise ValueError('Invalid image data, shape is: %s' % repr(x.shape))
            x = [x]

            for f_augment in augment_stack:
                if f_augment is DataAugment.split_cv or f_augment is DataAugment.merge_cv:
                    x = f_augment(x, dtype=x_dtype)
                else:
                    x = [f_augment(_, rng) for _ in x]

            if cv_kwargs:
                return x[0]
            else:
                return np.concatenate(x).reshape(x_shape).astype(x_dtype)
        return augment

    @staticmethod
    def split_cv(x, dtype=None):
        x = np.concatenate(x)
        c = x.shape[0]

        x_cv = []
        for i in range(c//3):
            x_i = x[i*3:(i+1)*3].transpose((1, 2, 0))
            if 'float' in str(dtype):
                x_i = x_i * 255
            x_cv.append(x_i.astype(np.uint8))
        for i in range(c-(c % 3), c):
            x_i = x[i:i+1].transpose((1, 2, 0))
            if 'float' in str(dtype):
                x_i = x_i * 255
            x_cv.append(x_i.astype(np.uint8))
        return x_cv

    @staticmethod
    def merge_cv(x, dtype=None):
        x = np.concatenate([_.transpose((2, 0, 1)) for _ in x])
        if 'float' in str(dtype):
            x = x.astype(np.float32) / 255
        return [x]

    @augment_method('geometric')
    def flip(self, p_horizontal=0.5, p_vertical=0.5):
        h_flip = _RD.binary(p_horizontal)
        v_flip = _RD.binary(p_vertical)

        def augment(x, rng):
            if h_flip(rng):
                x = np.flip(x, axis=-1)
            if v_flip(rng):
                x = np.flip(x, axis=-2)
            return x
        return augment

    def flip_horizontal(self, p=0.5):
        return self.flip(p_horizontal=p, p_vertical=0)

    def flip_vertical(self, p=0.5):
        return self.flip(p_horizontal=0, p_vertical=p)

    @augment_method('geometric', cv=True)
    def warp_affine(self, rotate=None, scale=None, translate=None, translate_direction=None, shear=None,
                    interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
        """ Rotates, scales, transforms and shears the image with random coefficients.
        (Transformations are always applied in this order: the translation length is scaled by scale and its direction
        is rotated by rotate).

        :param angle: Distribution from which the angle of rotation is sampled (in degrees) .
        :param scale: Distribution from which the factor of scale is sampled.
        :param translate: Distribution from which the translation length is sampled (in pixel).
        :param translate_direction: Distribution from which the translation direction is sampled (in degrees).
                    If 0, translate towards the right, if 90 translate towards the top.
        :param shear: Distribution from which the shear angle is sampled (in degrees).
        :param interpolation: INTER_NEAREST,  INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
        :param border_mode: Pixel extrapolation method. When borderMode=BORDER_TRANSPARENT , \
        it means that the pixels in the destination image that corresponds to the “outliers” in the source image are not modified by the function.
        :param border_value: Value used in case of a constant border. By default, it is 0.
        :return: Transformed image
        :rtype: numpy.ndarray
        """
        rotate = _RD.constant(0) if rotate is None else _RD.auto(rotate)
        scale = _RD.constant(1) if scale is None else _RD.auto(scale)
        translate = _RD.constant(0) if translate is None else _RD.auto(translate)
        translate_direction = _RD.uniform(360) if translate_direction is None else _RD.auto(translate_direction)
        shear = _RD.constant(0) if shear is None else _RD.auto(shear)

        deg2rad = np.pi/180

        def augment(x, rng):
            h, w = x.shape[:2]
            theta = rotate(rng)
            gamma = scale(rng)
            t_r = translate(rng)
            t_theta = translate_direction(rng) * deg2rad
            alpha = shear(rng) * deg2rad

            M = np.array([[1.0, -np.sin(alpha),  t_r*np.cos(t_theta)],
                          [0.0,  np.cos(alpha), -t_r*np.sin(t_theta)],
                          [0.0,            0.0,                  1.0]])
            M = np.matmul(cv2.getRotationMatrix2D(center=(w / 2, h / 2), angle=theta, scale=gamma), M)
            dst = cv2.warpAffine(x, M, (w, h), flags=interpolation, borderMode=border_mode, borderValue=border_value)
            return dst.reshape(x.shape)
        return augment

    def rotate(self, angle=(-25, 25),
               interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
        return self.warp_affine(rotate=angle,
                                interpolation=interpolation, border_mode=border_mode, border_value=border_value)

    def scale(self, scale=(0.9, 1.1),
              interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
        return self.warp_affine(scale=scale,
                                interpolation=interpolation, border_mode=border_mode, border_value=border_value)

    def translate(self, distance=15, direction=None,
                  interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
        if isinstance(direction, (int, float)):
            direction = _RD.constant(direction)
        return self.warp_affine(translate=distance, translate_direction=direction,
                                interpolation=interpolation, border_mode=border_mode, border_value=border_value)

    def shear(self, shear=(0.9, 1.1),
              interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
        return self.warp_affine(shear=shear,
                                interpolation=interpolation, border_mode=border_mode, border_value=border_value)

    @augment_method('geometric', cv=True)
    def elastic_distortion(self, dist=10, sigma=2, scale=1, mask=None,
                           interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0.0):
        """Apply an elastic distortion on an image

        :param dist: Distribution from which the displacement map is sampled (if int, the value is interpreted as the
        standard deviation of a gaussian distribution, in pixel * scale).
        :param sigma: Standard deviation of the low-pass filter smoothing the displacement map (in pixel * scale).
        :param scale: Rescaling factor of the displacement map after smoothing.
        :param mask: Mask specifying which part of the image should be displaced
        :param interpolation: INTER_NEAREST,  INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
        :param border_mode: Pixel extrapolation method. When borderMode=BORDER_TRANSPARENT , \
        it means that the pixels in the destination image that corresponds to the “outliers” in the source image are not modified by the function.
        :param border_value: Value used in case of a constant border. By default, it is 0.
        """
        from scipy.interpolate import RectBivariateSpline

        if isinstance(dist, int):
            dist = _RD.normal(mean=0, std=dist)
        else:
            dist = _RD.auto(dist)

        def augment(x, rng):
            h, w = x.shape[:2]
            shape = (h, w)
            dist_shape = (h // scale, w // scale)
            # Random
            dx = dist(rng, shape=dist_shape) * 2 - 1
            dy = dist(rng, shape=dist_shape) * 2 - 1

            # Mask
            if mask is not None:
                 dx = dx * mask
                 dy = dy * mask

            # Low-pass filter
            dx = cv2.GaussianBlur(dx, ksize=None, sigmaX=sigma)
            dy = cv2.GaussianBlur(dy, ksize=None, sigmaX=sigma)

            # Scale displacement map
            if scale != 1:
                x = np.linspace(0, w, dist_shape[1])
                y = np.linspace(0, h, dist_shape[0])
                dx = RectBivariateSpline(y, x, dx)
                dy = RectBivariateSpline(y, x, dy)

            range_x = range(w)
            range_y = range(h)
            x, y = np.meshgrid(range_x, range_y, indexing='xy')
            if scale != 1:
                dx = np.reshape(x + dx(range_y, range_x), shape).astype('float32')
                dy = np.reshape(y + dy(range_y, range_x), shape).astype('float32')
            else:
                dx = np.reshape(x + dx, shape).astype('float32')
                dy = np.reshape(y + dy, shape).astype('float32')

            # Apply displacement map
            dst = cv2.remap(x, map1=dx, map2=dy, interpolation=interpolation, borderMode=border_mode,
                            borderValue=border_value)
            return dst.reshape(x.shape)
        return augment

    @augment_method('color')
    def color(self, brightness=None, contrast=None, gamma=None, r=None, g=None, b=None):
        if brightness is None:
            brightness = _RD.constant(0)
        else:
            brightness = _RD.auto(brightness)

        if contrast is None:
            contrast = _RD.constant(1)
        else:
            contrast = _RD.auto(contrast)

        if gamma is None:
            gamma = _RD.constant(1)
        else:
            gamma = _RD.auto(gamma)

        if r is None:
            r = _RD.constant(0)
        else:
            r = _RD.auto(r)
        if g is None:
            g = _RD.constant(0)
        else:
            g = _RD.auto(g)
        if b is None:
            b = _RD.constant(0)
        else:
            b = _RD.auto(b)

        a_min = np.array([0, 0, 0], np.float32)
        a_max = np.array([1, 1, 1], np.float32)

        def augment(x, rng):
            _bright = brightness(rng)
            _contrast = contrast(rng)
            _gamma = gamma(rng)
            _r = r(rng)
            _b = b(rng)
            _g = g(rng)
            bgr = np.array([_b, _g, _r])

            x = ((x+_bright)*_contrast)**_gamma
            x = x + bgr[np.newaxis, :, np.newaxis, np.newaxis]
            return np.clip(x, a_min=a_min[np.newaxis, :, np.newaxis, np.newaxis],
                              a_max=a_max[np.newaxis, :, np.newaxis, np.newaxis])
        return augment

    def brightness(self, brightness=(-0.1, 0.1)):
        return self.color(brightness=brightness)

    def contrast(self, contrast=(-0.1, 0.1)):
        return self.color(contrast=contrast)

    def gamma(self, gamma=(-0.1, 0.1)):
        return self.color(gamma=gamma)

    @augment_method('color', cv=True)
    def hsv(self, hue=None, saturation=None, value=None):
        if hue is None:
            hue = _RD.constant(0)
        else:
            hue = _RD.auto(hue)
        if saturation is None:
            saturation = _RD.constant(0)
        else:
            saturation = _RD.auto(saturation)
        if value is None:
            value = _RD.constant(0)
        else:
            value = _RD.auto(value)

        a_min = np.array([0, 0, 0], np.uint8)
        a_max = np.array([179, 255, 255], np.uint8)

        def augment(x, rng):
            h = hue(rng)
            s = saturation(rng)
            v = value(rng)
            hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
            hsv = hsv + np.array([h, s, v])
            hsv[:, :, 0] = hsv[:, :, 0] % 179
            hsv = np.clip(hsv, a_min=a_min, a_max=a_max).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return augment

    def hue(self, hue=(-20, 20)):
        return self.hsv(hue=hue)

    def staturation(self, staturation=(-20, 20)):
        return self.hsv(staturation=staturation)


########################################################################################################################
class DataSetAugment(AbstractDataSet):
    def __init__(self, dataset, data_augment, columns, N=1, original=False, augment_kwargs=None, name="augment"):
        self._data_augment = data_augment
        self.N = N
        self.original = original
        pk_type = dataset.pk.dtype if self.n_factor == 1 else str

        super(DataSetAugment, self).__init__(name=name, parent_datasets=dataset, pk_type=pk_type)
        self._columns = dataset.copy_columns(self)

        self._f_augment = None
        self._f_augment_label = None
        if augment_kwargs is None:
            augment_kwargs = {}

        for col in self.interpret_columns(columns, to_column_name=False):
            if len(col.shape) not in (2, 3):
                raise ValueError('%s does not support data augmentation transform (shape: %s)' % (col, str(col.shape)))
            if col.format.is_label:
                if self._f_augment_label is None:
                    k = {}
                    k.update(augment_kwargs)
                    k['interpolation'] = cv2.INTER_NEAREST
                    k['except_type'] = set(k.get('except_type', ())).union({'color'})
                    self._f_augment_label = self.data_augment.compile(**k)
            else:
                if self._f_augment is None:
                    self._f_augment = self.data_augment.compile(**augment_kwargs)

        self.augmented_columns = columns

    def get_augmented(self, x, rng=np.random.RandomState(1234), label=None):
        if isinstance(rng, int):
            rng = np.random.RandomState(rng)
        if isinstance(x, DataSetResult):
            r = x.copy()
            for c in r.col:
                if c.ndim >= 2:
                    if label or (label is None and c.format.is_label):
                        r[c] = self._f_augment_label(x[c], rng=rng)
                    else:
                        r[c] = self._f_augment(x[c], rng=rng)
            return r
        else:
            if label:
                return self._f_augment_label(x, rng=rng)
            else:
                return self._f_augment(x, rng=rng)

    def _generator(self, gen_context):
        i_global = gen_context.start_id
        columns = gen_context.columns
        copy_columns = [c for c in columns if c not in self.augmented_columns]
        gen = gen_context.generator(self._parent, n=1, columns=columns,
                                    start=gen_context.start_id // self.n_factor,
                                    stop=gen_context.stop_id // self.n_factor)
        result = None

        while not gen_context.ended():
            i_global, n, weakref = gen_context.create_result()
            r = weakref()

            for i in range(n):
                # Read appropriate result
                if (i + i_global) % self.n_factor == 0 or result is None:
                    copy = {} if i + i_global % self.n_factor else {c: r[i:i + 1, c] for c in copy_columns}
                    if self.n_factor == 1:
                        copy['pk'] = r[i:i+1, 'pk']
                    result = gen.next(r=r, copy=copy, seek=(i_global+i)//self.n_factor)
                # Compute augmented data
                seed = i+i_global if gen_context.determinist else np.random.randint(0, 100000)
                for c in self.augmented_columns:
                    if c in self.augmented_columns and ((i + i_global) % self.n_factor != 0 or not self.original):
                        r[i, c] = self.get_augmented(result[c][0], label=r.col[c].format.is_label, rng=seed)
                    else:
                        r[i, c] = result[c][0]
                if (i+i_global) % self.n_factor:
                    for c in copy_columns:
                        r[i, c] = result[c][0]

                if self.n_factor > 1:
                    r[i, 'pk'] = str(result[0, 'pk'])+str((i + i_global) % self.n_factor)

            r = None
            yield weakref

    @property
    def data_augment(self):
        return self._data_augment

    @property
    def n_factor(self):
        return self.N + (1 if self.original else 0)

    @property
    def size(self):
        return self.n_factor*self.parent_dataset.size


########################################################################################################################
class RandomDistribution:
    def __init__(self, random_f, **kwargs):
        self._f = random_f
        self._kwargs = kwargs

    def __call__(self, rng, shape=None):
        if isinstance(rng, int):
            rng = np.random.RandomState(seed=rng)
        elif not isinstance(rng, np.random.RandomState):
            raise ValueError('rng is not a valid RandomState or seed')
        return self._f(rng=rng, shape=shape, **self._kwargs)

    def __getattr__(self, item):
        if item in self._kwargs:
            return self._kwargs[item]

    def __setattr__(self, key, value):
        if not key.startswith('_') and key in self._kwargs:
            self._kwargs[key] = value
        else:
            super(RandomDistribution, self).__setattr__(key, value)

    @staticmethod
    def auto(info):
        """
        Generate a RandomDistribution according to the value of an argument
        :rtype: RandomDistribution
        """
        if isinstance(info, tuple):
            if len(info) == 2:
                return RandomDistribution.uniform(*info)
            elif len(info) == 1:
                return RandomDistribution.uniform(low=-info[0], high=+info[0])
        elif isinstance(info, (float, int)):
            return RandomDistribution.uniform(high=info)
        elif isinstance(info, RandomDistribution):
            return info
        raise ValueError('Not interpretable random distribution: %s.' % repr(info))

    @staticmethod
    def uniform(high=1, low=0):
        if high < low:
            low, high = high, low

        def f(rng: np.random.RandomState, shape, low, high):
            return rng.uniform(low=low, high=high, size=shape)
        return RandomDistribution(f, low=low, high=high)

    @staticmethod
    def normal(mean=0, std=1):
        def f(rng: np.random.RandomState, shape, mean, std):
            return rng.normal(loc=mean, scale=std, size=shape)
        return RandomDistribution(f, mean=mean, std=std)

    @staticmethod
    def truncated_normal(mean=0, std=1, truncate_high=1, truncate_low=None):
        if truncate_low is None:
            truncate_low = -truncate_high

        def f(rng, shape, mean, std, truncate_low, truncate_high):
            return np.clip(rng.normal(loc=mean, scale=std, size=shape), a_min=truncate_low, a_max=truncate_high)
        return RandomDistribution(f, mean=mean, std=std, truncate_high=truncate_high, truncate_low=truncate_low)

    @staticmethod
    def binary(p=0.5):
        def f(rng: np.random.RandomState, shape, p):
            return rng.binomial(n=1, p=p, size=shape) > 0
        return RandomDistribution(f, p=p)

    @staticmethod
    def constant(c=0):
        def f(rng, shape, c):
            return np.ones(shape=shape, dtype=type(c))*c
        return RandomDistribution(f, c=c)

    @staticmethod
    def custom(f_dist, **kwargs):
        def f(rng, shape, **kwargs):
            return f_dist(x=rng.uniform(0, 1, size=shape), **kwargs)

        return RandomDistribution(f, **kwargs)

_RD = RandomDistribution