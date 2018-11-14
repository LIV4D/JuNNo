""" This module serves to data augmentation, integrated as part of the dataset framework.

The module is composed of the classes:
    * :class:`DataAugmentation`
    * :class:`GeometricOp`
    * :class:`ColorOp`

The main class :class:`DataAugmentation` is the core engine of the data augmentation \
process. A single instance per dataset of this class should be created. It will handle the call to the geometric \
transformation methods coded in :class:`GeometricOp` and the color transformation methods coded in :class:`ColorOp`.

"""
import inspect
import math
import random

import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline

from ..j_utils.function import match_params
from .dataset import AbstractDataSet, DataSetColumn


########################################################################################################################
class DataSetAugmentedData(AbstractDataSet):
    def __init__(self, dataset, columns, da_engine, n=3, name_column=None, keep_original=True, name='augment', column_transform=False):
        """
        :type dataset: AbstractDataSets

        """
        super(DataSetAugmentedData, self).__init__(name, dataset, pk_type=dataset.pk.dtype)

        # Initialize columns

        self._columns = dataset.copy_columns(self)
        if not isinstance(columns, list):
            columns = [columns]
        for c_id, c in enumerate(columns):
            if isinstance(c, DataSetColumn):
                columns[c_id] = c.name

        self.column_transform = column_transform
        if self.column_transform:
            if 'transformation' not in self.columns_name():
                self.add_column('transformation', (), str)

        self.augmented_columns = columns
        self.name_column = name_column
        self.da_engine = da_engine
        self.keep_original = keep_original

        for c in columns:
            column = self.column_by_name(c)
            if len(column.shape) not in (2, 3):
                raise ValueError('%s does not support data augmentation transform (shape: %s)' % (c, str(c.shape)))

        # n is the number of new images generated from one single image
        self.N_aug = n + (1 if self.keep_original else 0)

    def _generator(self, gen_context):
        i_global = gen_context.start_id
        columns = gen_context.columns

        columns_parent = gen_context.columns[:]
        if self.column_transform:
            if 'transformation' not in self._parent.columns_name() and 'transformation' in columns_parent:
                columns_parent.remove('transformation')
        gen = gen_context.generator(self._parent, n=1, from_id=i_global // self.N_aug, columns=columns_parent)
        result = None

        while not gen_context.ended():
            global_i, n, weakref = gen_context.create_result()
            r = weakref()

            for i in range(n):
                # Read appropriate result
                if (i + i_global) % self.N_aug == 0 or result is None:
                    result = gen.next(r=r)
                # Compute augmented data
                seed = i+i_global
                if not gen_context.determinist:
                    seed += gen_context.generator_id*self.size

                self.da_engine.init_object(seed)
                for c in columns:
                    if c in self.augmented_columns and ((i + i_global) % self.N_aug != 0 or not self.keep_original):
                        r[i, c] = self.da_engine.transform_data(result[c][0])
                        if 'transformation' in columns:
                            if 'transformation' not in self._parent.columns_name():
                                r[i, 'transformation'] = self.da_engine.transform_name
                            else:
                                if isinstance(result['transformation'][0], int):
                                    r[i, 'transformation'] = self.da_engine.transform_name
                                else:
                                    r[i, 'transformation'] = result['transformation'][0]+self.da_engine.transform_name
                    elif c == 'transformation':
                        pass
                    else:
                        r[i, c] = result[c][0]

                r[i, 'pk'] = str(result[0, 'pk'])+str((i + i_global) % self.N_aug)

            r = None
            yield weakref
            i_global += n

    @property
    def size(self):

        return self.N_aug * self.parent_dataset.size


########################################################################################################################
class DataAugmentation:
    """This class is integrated as part of the dataset framework. Its goal is to proceed with data augmentation, by \
    automatizing most of the standards operations.
    Nonetheless, one can still chooses to customize those operations by passing a list of custom function as an \
    of the constructor.

    For automatic transformation, standard geometric and color variations are applied. \
    In order to maximise the diversity of those operations, the class specifically chooses a ramdom list of methods \
    to proceeds with. This random subset (at least one) is built upon the methods contained in :class:`ColorOp`
    and :class:`GeometricOp`

    A general seed is defined in the constructor of the class. As long as this seed is not modified, the sames \
    transformation will be applied through the :func:`~data_augmentation.DataAugmentation.transform_data`
    This allows guaranteeing that same operations are applied to multiple columns of a single row in a dataset,
    which useful when labels and input should be altered in the same way (for image segmentation per example).

    It is possible to specify if one or many of those basics operations should not be executed.
    Inversely, it is also possible to specify which exacts transform should be applied.

    The other way of proceeding with data_augmentation is to pass one or a list of custom function. \
    Those function are going to be called one by one on the same input (transforms are cumulated)
    """

    def __init__(self,
                 use_geom_func=True,
                 use_color_func=False,
                 custom_function=None,
                 function2avoid=None,
                 use_only=None,
                 **kwargs):
        """Constructor of the class.
        This method does nothing a part of instantiating the parameters that will be used.

        :param use_geom_func: A library of standard geometric transforamtion function has been written. \
        If this parameter is True, the transform_data method will check in it and will choose randomly method in it. \
        Nonetheless, you can precise if you want to avoid some specific functions.
        :param use_color_func: A library of standard geometric transforamtion function has been written. \
        If this parameter is True, the transform_data method will check in it and will choose randomly method in it. \
        Nonetheless, you can precise if you want to avoid some specific functions.
        :param custom_function: A list of reference to method that take a single array as an input and returns an array \
        of the same size.
        :param function2avoid: A list of string, representing the name of the standards methods that you might want \
        to skip.
        :param use_only: A list of string, representing the name of the standards methods that might want to apply \
        exclusively
        :param kwargs: Optional parameters that are passed (through a match params method) to any function that is \
        called.
        :type use_only: list of str or str
        :type function2avoid: list of str or str
        :type custom_function: reference to methods
        :type use_color_func: bool
        :type use_geom_func: bool
        :type kwargs: dict

        """
        self.seed = None
        self.use_geom_func = use_geom_func
        self.use_color_func = use_color_func
        self.custom_function = custom_function
        self.std_func_used = []
        self.function2avoid = function2avoid
        self._kwargs = kwargs
        self.transform_name = ''
        if use_only is not None and not isinstance(use_only, list):
            use_only = [use_only]
        self.use_only = use_only

        self.geomOperator = GeometricOp()
        self.colorOperator = ColorOp()

        if self.use_only is not None:
            if not isinstance(self.use_only, list):
                self.use_only = [self.use_only]

        if self.function2avoid is not None:
            if not isinstance(self.function2avoid, list):
                self.function2avoid = [self.function2avoid]

    def init_object(self, seed):
        """Initializes the DA engine instance.

        The goal of this function is to initialize the list of methods used by the engine, according to the \
        specification given to the engine when built.

        :param seed: The seed used for all the  call to a numpy.RandomState function. \
        Controlling this seed inside a loop permits having the same transformation over all the columns
        :type seed: int
        """
        self.define_seed(seed)
        self.std_func_used = []

        if self.use_geom_func:
            if self.use_only is not None:
                for el in inspect.getmembers(self.geomOperator, predicate=inspect.ismethod):
                    if el[0] in self.use_only:
                        self.std_func_used.append(el)
            else:

                if self.function2avoid is not None:

                    for el in inspect.getmembers(self.geomOperator, predicate=inspect.ismethod):
                        if not el[0] in self.function2avoid:
                            self.std_func_used.append(el)
                else:
                    self.std_func_used += [_ for _ in inspect.getmembers(self.geomOperator, predicate=inspect.ismethod)]

        if self.use_color_func:
            if self.use_only is not None:
                for el in inspect.getmembers(self.colorOperator, predicate=inspect.ismethod):
                    if el[0] in self.use_only:
                        self.std_func_used.append(el)
            else:
                if self.function2avoid is not None:

                    for el in inspect.getmembers(self.colorOperator, predicate=inspect.ismethod):
                        if not el[0] in self.function2avoid:
                            self.std_func_used.append(el)
                else:
                    self.std_func_used += [_ for _ in
                                           inspect.getmembers(self.colorOperator, predicate=inspect.ismethod)]

        random.shuffle(self.std_func_used)  # Those functions can be color modification OR geometric modification.
        # We don't want to apply color modification on label!
        count_existing_operation = len(self.std_func_used)
        self.count_operation2use = np.random.randint(1, count_existing_operation + 1)

    def transform_data(self, x):
        """Apply transformation to an image

        This method proceeds with data augmentation on a single image x. If asked in the constructor, \
        it will first proceeds with random standards processing, hardcoded in classes :class:`GeometricOp` and \
        :class:`ColorOp`.
        Then eventually, it will apply custom_function over the image.
        The attribute of the class transform_name is also updated in order to keep a track of the transformation applied.

        The function matches all the params passed in the _kwargs attribute of the class with the different functions called.
        :param x: An numpy representing an image. At this point, only one or three channels images are handled.
        :type x: numpy.ndarray
        :return: An array of the same shape as x
        :rtype: numpy.ndarray

        """
        output = []
        list_arrays = self.decompose_channels(x)
        for x in list_arrays:
            x = np.transpose(x, (1, 2, 0))
            dtype = x.dtype
            # if dtype == np.float32:
            #     max_value = np.max(x)
                # x = ((x/max_value) * 255).astype(np.uint8)

            if self.use_geom_func or self.use_color_func:
                for i in range(self.count_operation2use):
                    if dtype != np.float32:
                        tmp = match_params(self.std_func_used[i][1], input=x, interpolation=cv2.INTER_NEAREST, seed=self.seed, **self._kwargs)
                    else:
                        tmp = match_params(self.std_func_used[i][1], input=x, seed=self.seed, **self._kwargs)
            if self.custom_function is not None:
                if not isinstance(self.custom_function, list): self.custom_function = [self.custom_function]
                for f in self.custom_function:
                    if dtype != np.float32:
                        tmp = match_params(f, input=x, interpolation=cv2.INTER_NEAREST, **self._kwargs)
                    else:
                        tmp = match_params(f, input=x, **self._kwargs)

            # if dtype == np.float32:
                # tmp = (tmp / 255.)*max_value
            self.build_transform_name()
            output.append(np.transpose(tmp, (2, 0, 1)).astype(dtype=dtype))
        return np.concatenate(output, axis=0)

    def decompose_channels(self, x):
        """
        Return a list of arrays, each array being of the form 1*h*w or 3*h*w
        :param x: Input array of form c*h*w
        :return:
        """
        initial_shape = x.shape
        nb_canal = initial_shape[0]
        if nb_canal == 1 or nb_canal == 3:
            return [x]
        list_arrays = []
        while nb_canal % 3:
            list_arrays.append(np.expand_dims(x[nb_canal - 1], axis=0))
            nb_canal -= 1
        return [x[i * 3:(i + 1) * 3] for i in range(nb_canal // 3)] + list_arrays

    def build_transform_name(self):
        """Associates an unique name with the list of transform applied on the image.

        List all the transformations that have affected the original image.
        This builds a transformation name and will be use to update the attribute transform_name.

        .. note::
            The typology of the name follows the convention:
            _seed_ **SEED** _ **transf1** _ **transf2** _...

            where:
                * **SEED** is the seed used (an integer)
                * **transf1** is the first two letters of the method's name of the first transformation applied.
                * **transf2** is the first two letters of the method's name of the second transformation applied.

        """
        self.transform_name = ''
        if self.use_geom_func or self.use_color_func:
            self.transform_name += '_seed_' + str(self.seed)
            for i in range(self.count_operation2use):
                name = self.std_func_used[i][0]
                if '_' in name:
                    name = name.split('_')[0][:2] + name.split('_')[1][:2]
                else:
                    name = name[:2]
                self.transform_name += '_' + name

    def define_seed(self, seed):
        """Define seed.

        Fixing the seed guarantees that all the random call will give the same results.
        This is usefull when the transform_data is called on two images different that should have the same sets of
        transformations (as for example, an image and its associated groundtruth).

        :param seed: The seed used in all the call to random function in this class
        :type seed: int

        """
        self.seed = seed
        random.seed(self.seed)
        GeometricOp.seed = seed
        ColorOp.seed = seed


class GeometricOp:
    """
    Standard and basics geometric transformations. Each function applied a small variation on the channel corresponding
    to the function's name.

    The implemented method are:
        * :func:`~GeometricOp.rotation`
        * :func:`~GeometricOp.translation`
        * :func:`~GeometricOp.shear`
        * :func:`~GeometricOp.elastic_distorsion`

    .. note::

        As most of the operations are done using openCV for python (cv2), \
        we recommend the `OpenCV documentation <http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html>`_

    """
    seed = 0

    def horizontal_flip(self, input):
        return np.fliplr(input)

    def vertical_flip(self, input):
        return np.flipud(input)

    def rotation(self, input, rotation_range=(-25, 25),
                 interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_CONSTANT,
                 border_value=0):
        """Rotates the image with a random angle

        The random angle is chosen in a range defined as a parameter

        :param input: Input array of shape (h, w, c) or (h, w).
        :param rotation_range: Range in which the random rotation angle should be picked.
        :param interpolation: INTER_NEAREST,  INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
        :param border_mode: Pixel extrapolation method. When borderMode=BORDER_TRANSPARENT , \
        it means that the pixels in the destination image that corresponds to the “outliers” in the source image are not modified by the function.
        :param border_value: Value used in case of a constant border. By default, it is 0.
        :type rotation_range: tuple of int
        :type input: numpy.ndarray
        :return: Transformed image
        :rtype: numpy.ndarray

        """
        np.random.seed(GeometricOp.seed)

        angle = np.random.uniform(low=rotation_range[0], high=rotation_range[1])
        if input.ndim == 3:
            rows, cols, chan = input.shape
        elif input.ndim == 2:
            rows, cols = input.shape

        M = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=angle, scale=1)

        dst = cv2.warpAffine(input, M, (cols, rows), flags=interpolation, borderMode=border_mode,
                             borderValue=border_value)

        return dst.reshape(input.shape)

    def translation(self, input, translation_range=(-15, 15), interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    border_value=0.0):
        """Translates the image with a random factor

        The random factor is chosen in a range defined as a parameter

        :param input: Input array of shape (h, w, c) or (h, w).
        :param translation_range: Range in which the random translation factor should be picked.
        :param interpolation: INTER_NEAREST,  INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
        :param border_mode: Pixel extrapolation method. When borderMode=BORDER_TRANSPARENT , \
        it means that the pixels in the destination image that corresponds to the “outliers” in the source image are not modified by the function.
        :param border_value: Value used in case of a constant border. By default, it is 0.
        :type translation_range: tuple of int
        :type input: numpy.ndarray
        :return: Transformed image
        :rtype: numpy.ndarray

        """
        np.random.seed(GeometricOp.seed)

        tx = np.random.uniform(low=translation_range[0], high=translation_range[1])
        ty = np.random.uniform(low=translation_range[0], high=translation_range[1])
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        if input.ndim == 3:
            rows, cols, chan = input.shape
        elif input.ndim == 2:
            rows, cols = input.shape

        dst = cv2.warpAffine(input, M, (cols, rows), flags=interpolation, borderMode=border_mode,
                             borderValue=border_value)

        return dst.reshape(input.shape)

    def shear(self, input, shear_range=(-5, 5), interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
              border_value=0.0):
        """Shears the image with a random factor

        The random factor is chosen in a range defined as a parameter

        :param input: Input array of shape (h, w, c) or (h, w).
        :param shear_range: Range in which the random translation factor should be picked.
        :param interpolation: INTER_NEAREST,  INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
        :param border_mode: Pixel extrapolation method. When borderMode=BORDER_TRANSPARENT , \
        it means that the pixels in the destination image that corresponds to the “outliers” in the source \
        image are not modified by the function.
        :param border_value: Value used in case of a constant border. By default, it is 0.
        :type shear_range: tuple of int
        :type input: numpy.ndarray
        :return: Transformed image
        :rtype: numpy.ndarray

        """
        np.random.seed(GeometricOp.seed)

        shear = math.radians((np.random.uniform(low=shear_range[0], high=shear_range[1])))
        if input.ndim == 3:
            rows, cols, chan = input.shape
        elif input.ndim == 2:
            rows, cols = input.shape

        shear_matrix = np.array([[1.0, -np.sin(shear), 0.0],
                                 [0.0, np.cos(shear), 0.0]])
        dst = cv2.warpAffine(input, shear_matrix, (cols, rows), flags=interpolation, borderMode=border_mode,
                             borderValue=border_value)
        return dst.reshape(input.shape)

    def elastic_distortion(self, input, sigma=2, alpha=10, scale=1,
                           interpolation=cv2.INTER_LINEAR,
                           border_mode=cv2.BORDER_CONSTANT,
                           border_value=0.0, mask=None):
        np.random.seed(GeometricOp.seed)

        if input.dim == 3:
            rows, cols, chan = input.shape
        elif input.dim == 2:
            rows, cols = input.shape
        else:
            raise NotImplementedError

        shape = (rows, cols)
        dist_shape = (rows // scale, cols // scale)
        # Random
        dx = (np.random.rand(*dist_shape) * 2 - 1)
        dy = (np.random.rand(*dist_shape) * 2 - 1)

        # Low-pass filter
        if mask is not None:
            dx = cv2.GaussianBlur(dx, ksize=None, sigmaX=sigma) * alpha * mask
            dy = cv2.GaussianBlur(dy, ksize=None, sigmaX=sigma) * alpha * mask
        else:
            dx = cv2.GaussianBlur(dx, ksize=None, sigmaX=sigma) * alpha
            dy = cv2.GaussianBlur(dy, ksize=None, sigmaX=sigma) * alpha

        # Scale displacement map
        if scale != 1:
            x = np.linspace(0, cols, dist_shape[1])
            y = np.linspace(0, rows, dist_shape[0])
            dx = RectBivariateSpline(y, x, dx)
            dy = RectBivariateSpline(y, x, dy)

        range_x = range(input.shape[1])
        range_y = range(input.shape[0])
        x, y = np.meshgrid(range_x, range_y, indexing='xy')
        if scale != 1:
            dx = np.reshape(x + dx(range_y, range_x), shape).astype('float32')
            dy = np.reshape(y + dy(range_y, range_x), shape).astype('float32')
        else:
            dx = np.reshape(x + dx, shape).astype('float32')
            dy = np.reshape(y + dy, shape).astype('float32')

        # Apply displacement map
        dst = cv2.remap(input, map1=dx, map2=dy, interpolation=interpolation, borderMode=border_mode,
                        borderValue=border_value)
        return dst.reshape(input.shape)


class ColorOp:
    """Standard and basics color transformations.

    The implemented method are:
        * :func:`~ColorOp.brightness`
        * :func:`~ColorOp.contrast`
        * :func:`~ColorOp.gamma`
        * :func:`~ColorOp.saturation`
        * :func:`~ColorOp.value`

    """
    seed = 0

    def brightness(self, input, brightness_range=(-0.25, 0.25), mask=None):
        """Adjust the brightness of the image

        The operation done is:
        
        .. math:: output = input + \\beta

        Where :math:`\\beta` is a random integer, defined in the uint8 colorspace.

        :param input: Input array of shape (h, w, c) or (h, w). Image can be float (range between [0,1]) or uint8 (range between [0,255])
        :param brightness_range: Range in which :math:`\\beta` will be randomly picked.
        :param mask: A mask of shape (h, w) defining where to apply the transformation, (range between [0,1]). \
        The mask can be a grayscale image. In this case, the interpolation used is a multiplication.
        :type mask: numpy.ndarray
        :type brightness_range: tuple of int
        :type input: numpy.ndarray
        :return: transformed array
        :rtype: numpy.ndarray

        """
        np.random.seed(ColorOp.seed)

        beta = np.random.uniform(low=brightness_range[0], high=brightness_range[1])
        if mask is not None:
            if len(input.shape) == 3:
                alpha = np.stack([mask] * 3, axis=2)
            else:
                alpha = mask

        if mask is not None:
            input += (beta * alpha)
        else:
            input += beta
        np.clip(input, 0, 1, input)
        return input

    def contrast(self, input, constrast_range=(0.5, 1.5), mask=None):
        """Adjust the contrast of the image

        The operation done is:
        
        .. math:: output = input * \\alpha

        Where :math:`\\alpha` is a random float, generally around one.

        :param input: Input array of shape (h, w, c) or (h, w). Image can be float (range between [0,1]) or uint8 (range between [0,255])
        :param constrast_range: Range in which :math:`\\alpha` will be randomly picked.
        :param mask: A mask of shape (h, w) defining where to apply the transformation, (range between [0,1]). \
        The mask can be a grayscale image. In this case, the interpolation used is a multiplication.
        :type mask: numpy.ndarray
        :type constrast_range: tuple of float
        :type input: numpy.ndarray
        :return: transformed array
        :rtype: numpy.ndarray

        """
        np.random.seed(ColorOp.seed)
        alpha = np.random.uniform(low=constrast_range[0], high=constrast_range[1])
        if mask is not None:
            if len(input.shape) == 3:
                alpha_mask = np.stack([mask] * 3, axis=2)
            else:
                alpha_mask = mask
            org = (1. - alpha_mask) * input

        input *= alpha

        np.clip(input, 0, 1., input)

        if mask is not None:
            return input * alpha_mask + org
        else:
            return input

    def gamma(self, input, gamma_range=(0.5, 1.5), mask=None):
        """Adjust the contrast of the image

        The operation done is:
        
        .. math:: output = input^{\\gamma}

        Where :math:`\\gamma` is a random float, generally around one.

        :param input: Input array of shape (h, w, c) or (h, w). Image can be float (range between [0,1]) or uint8 (range between [0,255])
        :param gamma_range: Range in which :math:`\\gamma` will be randomly picked.
        :param mask: A mask of shape (h, w) defining where to apply the transformation, (range between [0,1]). \
        The mask can be a grayscale image. In this case, the interpolation used is a multiplication.
        :type mask: numpy.ndarray
        :type gamma_range: tuple of float
        :type input: numpy.ndarray
        :return: transformed array
        :rtype: numpy.ndarray

        """
        np.random.seed(ColorOp.seed)
        gamma = np.random.uniform(low=gamma_range[0], high=gamma_range[1])

        if mask is not None:
            if len(input.shape) == 3:
                alpha = np.stack([mask] * 3, axis=2)
            else:
                alpha = mask
            org = (1. - alpha) * input

        if mask is not None:
            return input ** gamma * alpha + org
        else:
            return input ** gamma

    def saturation(self, input, saturation_range=(-0.20, 0.20), mask=None):
        """Adjust the saturation of the image

        openCV for python is used for that (cv2).
        For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].

        :param input: Input array of shape (h, w, c). Image can be float (range between [0,1]) or uint8 (range between [0,255])
        :param saturation_range: Range in which the saturation random factor will be picked.
        :param mask: A mask of shape (h, w) defining where to apply the transformation, (range between [0,1]). \
        The mask can be a grayscale image. In this case, the interpolation used is a multiplication.
        :type mask: numpy.ndarray
        :type saturation_range: tuple of int
        :type input: numpy.ndarray
        :return: transformed array
        :rtype: numpy.ndarray

        """
        np.random.seed(ColorOp.seed)
        saturation_correct = np.random.uniform(low=saturation_range[0], high=saturation_range[1])
        hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        saturation_map = hsv[:, :, 1] + saturation_correct
        np.clip(saturation_map, 0, 1, saturation_map)

        if mask is not None:
            hsv[:, :, 1] = (saturation_map * mask + (1. - mask) * hsv[:, :, 1])
        else:
            hsv[:, :, 1] = saturation_map

        input = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return input

    def value(self, input, value_range=(-0.25, 0.25), mask=None):
        """Adjust the value of the image

        For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].

        :param input: Input array of shape (h, w, c). Image can be float (range between [0,1]) or uint8 (range between [0,255])
        :param value_range: Range in which the value random factor will be picked.
        :param mask: A mask of shape (h, w) defining where to apply the transformation, (range between [0,1]). \
        The mask can be a grayscale image. In this case, the interpolation used is a multiplication.
        :type mask: numpy.ndarray
        :type value_range: tuple of int
        :type input: numpy.ndarray
        :return: transformed array
        :rtype: numpy.ndarray

        """
        np.random.seed(ColorOp.seed)
        value_correct = np.random.uniform(low=value_range[0], high=value_range[1])

        hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        value_map = hsv[:, :, 2] + value_correct
        np.clip(value_map, 0, 1., value_map)
        if mask is not None:
            hsv[:, :, 2] = (value_map * mask + (1. - mask) * hsv[:, :, 2])
        else:
            hsv[:, :, 2] = value_map
        input = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return input
