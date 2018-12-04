

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
