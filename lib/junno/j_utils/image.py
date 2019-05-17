

def auto_crop(img, threshold, blur=0):
    import numpy as np
    import cv2

    threshold_img = img
    if blur:
        threshold_img = cv2.blur(img, (blur, blur), borderType=cv2.BORDER_REPLICATE)
    if threshold_img.ndim == 3:
        threshold_img = np.mean(threshold_img, axis=2)
    not_null_pixels = np.nonzero(threshold_img > threshold)
    x_range = (np.min(not_null_pixels[1]), np.max(not_null_pixels[1]))
    y_range = (np.min(not_null_pixels[0]), np.max(not_null_pixels[0]))
    return img[y_range[0]:y_range[1], x_range[0]:x_range[1]]


def crop_pad(img, shape):
    import numpy as np

    h, w = img.shape[-2:]

    if isinstance(shape, int):
        shape = (shape, shape)

    h_target, w_target = shape
    l_pad = int(np.floor((w_target - w)/2))
    r_pad = int(np.ceil((w_target - w)/2))
    t_pad = int(np.floor((h_target - h)/2))
    b_pad = int(np.ceil((h_target - h)/2))

    r = np.zeros(img.shape[:-2]+shape, dtype=img.dtype)
    r[..., max(0, t_pad):h_target-max(0, b_pad), max(0, l_pad):w_target-max(0, r_pad)] =\
                img[..., max(0, -t_pad):h-max(0, -b_pad), max(0, -l_pad):w-max(0, -r_pad)]
    return r


def compute_regular_patch_centers(patch_shapes, img_shape, stride=None, ignore_borders=False, mask=None):
    max_x = max([_[1] for _ in patch_shapes])
    max_y = max([_[0] for _ in patch_shapes])
    half_x = max_x // 2
    odd_x = max_x % 2
    half_y = max_y // 2
    odd_y = max_y % 2

    if stride is None:
        stride = patch_shapes

    if not ignore_borders:
        y_range = (half_y, img_shape[0])
        x_range = (half_x, img_shape[1])
    else:
        pad_h = img_shape[0] % stride[0] // 2
        pad_w = img_shape[1] % stride[1] // 2
        y_range = (pad_h, img_shape[0] - pad_h + 1)
        x_range = (pad_w, img_shape[1] - pad_w + 1)
    centers = []

    for y in range(y_range[0], y_range[1], stride[0]):
        for x in range(x_range[0], x_range[1], stride[1]):
            if mask is None or mask[y, x] > 0.5:
                centers.append((y, x))

    return centers


def compute_proba_patch_centers(proba_map, n, rng=None):
    import numpy as np

    if rng is None:
        rng = np.random.RandomState(1234)

    if proba_map.ndim == 3:
        proba_map = proba_map.reshape(proba_map.shape[1:])

    s = proba_map.shape
    dist = proba_map.flatten()/np.sum(proba_map)

    r = rng.uniform(size=n)

    cdf = dist.cumsum()
    centers_id = np.searchsorted(cdf, r)
    return [np.unravel_index(i, s) for i in centers_id]


_last_map = None
_last_lut = None


def prepare_lut(map, source_dtype=None, axis=None, sampling=None, default=None):
    assert isinstance(map, dict) and len(map)

    import numpy as np
    from .collections import if_none

    # Prepare map
    source_list = []
    dest_list = []
    source_shape = None
    dest_shape = None

    add_empty_axis = False
    for source, dest in map.items():
        if isinstance(source, str):
            source = str2color(source, uint8=str(source_dtype) == 'uint8')
        source = np.array(source)
        if source.ndim == 0:
            source = source.reshape((1,))
            add_empty_axis = True
        if source_shape is None:
            source_shape = source.shape
        elif source_shape != source.shape:
            raise ValueError('Invalid source values: %s (shape should be %s)' % (repr(source), source_shape))
        source_list.append(source)

        if isinstance(dest, str):
            dest = str2color(dest, uint8=str(source_dtype) == 'uint8')
        dest = np.array(dest)
        if dest.ndim == 0:
            dest = dest.reshape((1,))
        if dest_shape is None:
            dest_shape = dest.shape
        elif dest_shape != dest.shape:
            raise ValueError('Invalid destination values: %s (shape should be %s)' % (repr(source), dest_shape))
        dest_list.append(dest)

    if axis:
        if isinstance(axis, int):
            axis = np.array([axis])
        elif isinstance(axis, (list, tuple, np.ndarray)):
            axis = np.array(axis)
        else:
            raise ValueError('Invalid axis parameter: %s (should be one or a list of axis)' % repr(axis))
    elif axis is None:
        axis = np.arange(len(source_shape))

    # Read shape
    n_axis = len(axis)
    source_size = int(np.prod(source_shape))
    dest_size = int(np.prod(dest_shape))
    dest_axis = sorted(axis)[0]

    # Prepare lut table
    sources = []
    lut_dests = [if_none(default, np.zeros_like(dest))]
    for s, d in zip(source_list, dest_list):
        source = np.array(s).flatten()
        dest = np.array(d)
        if dest.shape:
            dest = dest.flatten()
        sources.append(source)
        lut_dests.append(dest)

    sources = np.array(sources).astype(dtype=source_dtype)
    lut_dests = np.array(lut_dests)

    mins = sources.min(axis=0)
    maxs = sources.max(axis=0)

    if sampling is None:
        if 'float' in str(sources.dtype) and mins.min() >= 0 and maxs.max() <= 1:
            sampling = 1 / 255
    elif sampling == 'nearest':
        sampling = np.zeros(sources.shape[1:], dtype=np.float)
        for i in range(sources.shape[0]):
            sampling[i] = 1 / np.gcd.reduce(sources[i]) / 2
    if not sampling:
        sampling = 1

    sources = (sources / sampling).astype(np.int32)
    mins = sources.min(axis=0)
    maxs = sources.max(axis=0)
    stride = np.cumprod([1] + list((maxs - mins + 1)[1:][::-1]), dtype=np.uint32)[::-1]

    flatten_sources = np.sum((sources-mins) * stride, dtype=np.uint32, axis=1)
    id_sorted = flatten_sources.argsort()
    flatten_sources = flatten_sources[id_sorted]
    lut_dests[1:] = lut_dests[1:][id_sorted]

    if np.all(flatten_sources == np.arange(len(flatten_sources))):
        lut_sources = None
    else:
        lut_sources = np.zeros((int(np.prod(maxs - mins + 1)),), dtype=np.uint32)
        for s_id, s in enumerate(flatten_sources):
            lut_sources[s] = s_id + 1

    def f_lut(array):
        if len(axis) > 1 and axis != np.arange(len(axis)):
            array = np.moveaxis(array, source=axis, destination=np.arange(len(axis)))
        elif add_empty_axis:
            array = array.reshape((1,) + array.shape)

        # if 'int' not in str(array.dtype):
        #     log.warn('Array passed to apply_lut was converted to int32. Numeric precision may have been lost.')

        # Read array shape
        a_source_shape = array.shape[:n_axis]
        map_shape = array.shape[n_axis:]
        map_size = int(np.prod(map_shape))

        # Check source shape
        if a_source_shape != source_shape:
            raise ValueError('Invalid dimensions on axis: %s. (expected: %s, received: %s)'
                             % (str(axis), str(source_shape), str(a_source_shape)))

        # Prepare table
        if sampling == 1:
            array = array.astype(np.int32)
        else:
            array = (array / sampling).astype(np.int32)

        a = np.moveaxis(array.reshape(source_shape + (map_size,)), -1, 0).reshape((map_size, source_size))
        id_mapped = np.logical_not(np.any(np.logical_or(a > maxs, a < mins), axis=1))
        array = np.sum((a - mins) * stride, axis=1).astype(np.uint32)

        # Map values
        a = np.zeros(shape=(map_size,), dtype=np.uint32)
        if lut_sources is not None:
            a[id_mapped] = lut_sources[array[id_mapped]]
        else:
            a[id_mapped] = array[id_mapped]+1
        array = lut_dests[a]

        del a
        del id_mapped

        # Reshape
        array = array.reshape(map_shape + dest_shape)

        return np.moveaxis(array, np.arange(len(map_shape), array.ndim),
                           np.arange(dest_axis, dest_axis + len(dest_shape)) if len(dest_shape) != len(axis) else axis)

    f_lut.sources = sources
    f_lut.lut_sources = lut_sources
    f_lut.mins = mins
    f_lut.maxs = maxs
    f_lut.stride = stride
    f_lut.lut_dests = lut_dests
    f_lut.sampling = sampling
    f_lut.source_dtype = source_dtype
    return f_lut


def apply_lut(array, map, axis=None, sampling=None, default=None):
    # import numpy as np
    #
    # a = array
    # if axis:
    #     if isinstance(axis, int):
    #         axis = np.array([axis])
    #     elif isinstance(axis, (list, tuple, np.ndarray)):
    #         axis = np.array(axis)
    #         a = np.moveaxis(a, source=axis, destination=np.arange(len(axis)))
    #     else:
    #         raise ValueError('Invalid axis parameter: %s (should be one or a list of axis)' % repr(axis))
    # elif axis is None:
    #     axis = np.arange(np.array(next(iter(map.keys()))).ndim)
    #     if len(axis) == 0:
    #         axis = None
    #         a = array.reshape((1,) + a.shape)
    #
    # n_axis = len(axis) if axis else 1
    # source_shape = a.shape[:n_axis]
    # source_size = int(np.prod(source_shape))
    # map_shape = a.shape[n_axis:]
    # map_size = int(np.prod(map_shape))
    #
    # a = a.reshape((source_size, map_size))
    # mins = a.min(axis=-1)
    # maxs = a.max(axis=-1)
    # a_minmax = (mins, maxs)

    f_lut = prepare_lut(map, source_dtype=array.dtype, axis=axis, sampling=sampling, default=default)
    return f_lut(array)


def str2color(str_color, bgr=True, uint8=True):
    import numpy as np
    if not str_color or not isinstance(str_color, str):
        return np.zeros((3,), dtype=np.uint8 if uint8 else np.float16)

    c = str_color.split('.')
    if len(c) == 1:
        c_str = c[0]
        m = 1
    else:
        c_str = c[0]
        m = float('0.'+c[1])
        if c_str.lower() == 'black':
            m = 1-m

    try:
        c = dict(
            blue=(0,0,255),
            magenta=(255,0,255),
            red=(255,0,0),
            yellow=(255,255,0),
            green=(0,255,0),
            cyan=(0,255,255),
            turquoise=(0,255,127),
            sky_blue=(0,127,255),
            orange=(255,127,0),
            apple_green=(127,255,0),
            pruple=(127,0,255),
            pink=(255,0,127),
            white=(255,255,255),
            grey=(127,127,127),
            black=(0,0,0)
        )[c_str.lower()]
    except KeyError:
        raise ValueError('Invalid color code: %s' % c) from None
    c = np.array(c, dtype=np.float16)*m
    if uint8:
        c = c.astype(dtype=np.uint8)
    else:
        c /= 255
    if bgr:
        c = c[::-1]
    return c


def cast_shape(array, target_shape, pad=False, crop=False,
                      first_last_channel=True, ignore_null_dimension=True):
    import numpy as np

    s = array.shape
    ini_shape = s
    true_target_shape = target_shape
    target_shape = np.array(target_shape, dtype=np.uint16)
    if np.all(s == target_shape):
        return array

    # -- NULL DIMENSIONS --
    if ignore_null_dimension:
        # Clear empty dimensions
        while len(s) > 0 and s[0] == 1:
            array = array[0]
            s = array.shape
        while len(s)>1 and s[-1] == 1:
            array = array[[slice(None, None)]*(len(s)-1) + [0]]
            s = array.shape
        while len(target_shape)>0 and target_shape[0] == 1:
            target_shape = target_shape[1:]
        while len(target_shape)>1 and target_shape[-1] == 1:
            target_shape = target_shape[:-1]

    def check_shape(a):
        s = a.shape
        if a.shape == true_target_shape:
            return a
        if ignore_null_dimension:
            # Add empty dimensions as target_shape
            if np.all(target_shape == 1):
                # If target_shape is pure empty dimensions
                if len(s) == 0:
                    a = a.reshape(target_shape)
            else:
                # Otherwise add empty dimensions before and after
                empty_before = 0
                while true_target_shape[empty_before] <= 1:
                    empty_before += 1
                empty_after = 0
                while true_target_shape[-1-empty_after] <= 1:
                    empty_before += 1
                s = (1,)*empty_before + s + (1,)*empty_after
                a = a.reshape(s)
        if a.shape == true_target_shape:
            return a

        raise ValueError('No shape match was found between input %s and target %s.'
                         % (repr(ini_shape), repr(true_target_shape)))

    try:
        return check_shape(array)
    except ValueError:
        pass

    # -- INVERSE FIRST AND LAST CHANNEL DIMENSIONS  --
    if first_last_channel and len(s) > 1 and len(target_shape) > 1:
        if s[0] != target_shape[0] and s[0] == target_shape[-1]:
            array = np.move_axis(array, 0,-1)
        elif s[-1] != target_shape[-1] and s[-1] == target_shape[0]:
            array = np.move_axis(array, -1, 0)

        try:
            return check_shape(array)
        except ValueError:
            pass
    try:
        return check_shape(array)
    except ValueError:
        raise ValueError('No shape match was found between input %s and target %s.'
                         % (repr(ini_shape), repr(true_target_shape))) from None
