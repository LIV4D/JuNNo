

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


def compute_regular_patch_centers(patch_shapes, img_shape, stride=(1,1), ignore_borders=False, mask=None):
    max_x = max([_[1] for _ in patch_shapes])
    max_y = max([_[0] for _ in patch_shapes])
    half_x = max_x // 2
    odd_x = max_x % 2
    half_y = max_y // 2
    odd_y = max_y % 2

    y_range = (half_y+(0 if not ignore_borders else - half_y - odd_y + 1), img_shape[0] + (0 if ignore_borders else - half_y - odd_y + 1))
    x_range = (half_x+(0 if not ignore_borders else - half_x - odd_x + 1), img_shape[1] + (0 if ignore_borders else - half_x - odd_x + 1))
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

    if proba_map.dim == 3:
        proba_map = proba_map.reshape(proba_map.shape[1:])

    s = proba_map.shape
    dist = proba_map.flatten()/np.sum(proba_map)

    r = rng.uniform(size=n)

    cdf = dist.cumsum()
    centers_id = np.searchsorted(cdf, r)
    return [np.unravel_index(i, s) for i in centers_id]


_last_map = None
_last_lut = None


def map_values(array, map, axis=None, sampling=None, default=None):
    assert isinstance(map, dict) and len(map)

    import numpy as np
    from .collections import if_none

    if axis:
        if isinstance(axis, int):
            axis=[axis]
        elif isinstance(axis, (list, tuple, np.ndarray)):
            array = np.moveaxis(array, source=axis, destination=np.arange(len(axis)))
        else:
            raise ValueError('Invalid axis parameter: %s (should be one or a list of axis)' % repr(axis))
    elif axis is None:
        axis = np.arange(len(np.array(next(iter(map.keys()))).shape))

    if sampling is None:
        if 'float' in str(array.dtype) and array.min() >= 0 and array.max() <= 1:
            sampling = 1/255
    if not sampling:
        sampling = 1
    else:
        array = array / sampling

    if 'int' not in str(array.dtype):
        from .j_log import log
        log.warn('Array passed to map_values was converted to int32. Numeric precision may have been lost.')
        array = array.astype(dtype=np.int32)

    # Read shape
    n_axis = len(axis)
    source_size = np.prod(tuple(array.shape[_] for _ in range(n_axis)))
    map_shapes = tuple(array.shape[_] for _ in range(n_axis, array.ndim))
    array = array.reshape((source_size,)+map_shapes)
    dest_sample = np.array(next(iter(map.values())))
    dest_shape = dest_sample.shape
    dest_size = int(np.prod(dest_shape))

    # Prepare lut table
    a = array.reshape((source_size, np.prod(map_shapes)))
    mins = a.min(axis=-1)
    maxs = a.max(axis=-1)
    stride = np.cumprod([1]+list((maxs-mins+1)[1::-1]))[::-1]
    array = array.reshape((source_size,) + map_shapes)

    sources = []
    lut_dests = [if_none(default, np.zeros_like(dest_sample))]
    lut_color = [np.zeros((3,), dtype=np.uint8) if not isinstance(default, str) else str2color(default)]
    for k, v in map.items():
        source = (np.array(k)/sampling).astype(dtype=array.dtype).flatten()
        dest = np.array(v)
        if dest.shape:
            dest = dest.flatten()
        if source_size != source.size:
            raise ValueError('Invalid source values: %s (length should be %i)' % (repr(source), source_size))
        if dest_size != dest.size:
            raise ValueError('Invalid destination values: %s (length should be %i)' % (repr(dest), dest_size))
        sources.append(source)
        lut_dests.append(dest)
        if not dest.shape and isinstance(dest[()], str):
            try:
                lut_color.append(str2color(dest[()]))
            except ValueError:
                pass

    array = np.sum((array.T - mins) * stride, axis=1)

    if np.all(sources > mins) and np.all(sources < maxs):
        sources = np.array(sources, dtype=array.dtype)-mins
    lut_sources = np.zeros((np.prod(maxs-mins+1),), dtype=np.uint32)
    for s_id, s in enumerate(sources):
        lut_sources[np.sum((s-mins)*stride)] = s_id+1

    if len(lut_color) == len(lut_dests):
        lut_dests = lut_color
    lut_dests = np.array(lut_dests)

    # Map values
    array = lut_sources[array]
    array = lut_dests[array]

    # Reshape
    array = array.reshape(map_shapes+dest_shape)
    f_axis = sorted(axis)[0]

    return np.moveaxis(array, np.arange(len(map_shapes), array.ndim),
                       np.arange(f_axis, f_axis+len(dest_shape)) if len(dest_shape)!=len(axis) else axis)


def str2color(str_color, bgr=True, uint8=True):
    import numpy as np
    if not str_color or not isinstance(str_color, 'str'):
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
            red=(255,0,0),
            magenta=(255,0,255),
            blue=(0,0,255),
            cyan=(0,255,255),
            green=(0,255,0),
            yellow=(255,255,0),
            orange=(255,127,0),
            apple_green=(127,255,0),
            turquoise=(0,255,127),
            sky_blue=(0,127,255),
            purple=(127,0,255),
            pink=(255,0,127),
            white=(255,255,255),
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
