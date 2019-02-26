import numpy as np
import cv2

from .dataset import AbstractDataSet, DSColumn
from ..j_utils.function import identity_function, not_optional_args, match_params
from ..j_utils.parallelism import parallel_exec
from ..j_utils.j_log import log, Process


########################################################################################################################
class DataSetReshape(AbstractDataSet):
    def __init__(self, dataset, columns, shape, label_columns=None, keep_original=False, name='reshape'):
        """
        :type dataset: AbstractDataSets

        """
        super(DataSetReshape, self).__init__(name, dataset, pk_type=dataset.pk.dtype)

        # Initialize shape
        if isinstance(shape, int) or isinstance(shape, float):
            shape = (shape, shape)
        if isinstance(shape, tuple):
            if len(shape) != 2:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.shape = shape

        # Initialize columns
        self._columns = dataset.copy_columns(self)
        if not isinstance(columns, list):
            columns = [columns]

        self._reshaped_columns = {}
        for c_id, c in enumerate(columns):
            if isinstance(c, DSColumn):
                columns[c_id] = c.name
                c = c.name
            if isinstance(c, str):
                if c not in self.columns_name():
                    raise ValueError('Unkown column: %s' % c)
            else:
                raise NotImplementedError('label_column should be a column name, a DatasetColumn or a list of those '
                                          '(type: %s)' % repr(type(c)))

            column = self.column_by_name(c)
            if len(column.shape) not in (2, 3):
                raise ValueError('%s does not support reshaping' % c)

            column_shape = tuple(k if k > 1 and isinstance(k, int) else int(k * s)
                                 for k, s in zip(shape, column.shape[-2:]))
            if not keep_original:

                column._shape = column._shape[:-2] + column_shape
                self._reshaped_columns[c] = c
            else:
                self.add_column(c+'_reshaped', column._shape[:-2] + column_shape, column.dtype, format=column.format)
                self._reshaped_columns[c + '_reshaped'] = c

        if label_columns is None:
            label_columns = []
        if not isinstance(label_columns, list):
            label_columns = [label_columns]

        self._label_columns = []
        for c in label_columns:
            if isinstance(c, DSColumn):
                c = c.name
            if isinstance(c, str):
                if c not in columns:
                    raise ValueError('Unkown label column: %s' % c)
            else:
                raise NotImplementedError('label_column should be a column name, a DatasetColumn or a list of those '
                                          '(type: %s)' % repr(type(c)))
            self._label_columns.append(c)

    def _generator(self, gen_context):
        from ..j_utils.math import cartesian

        i_global = gen_context.start_id
        columns = gen_context.columns

        copied_columns = ['pk']
        reshaped_columns = []
        parent_columns = set()

        for c in columns:
            if c in self._reshaped_columns:
                reshaped_columns.append(c)
                parent_columns.add(self._reshaped_columns[c])
            else:
                copied_columns.append(c)
                parent_columns.add(c)

        gen = gen_context.generator(self._parent, columns=list(parent_columns))

        while not gen_context.ended():
            global_i, n, weakref = gen_context.create_result()
            r = weakref()

            result = gen.next(copy={_: r[:, _] for _ in copied_columns}, limit=n, r=r)

            # Reshape data
            for c in reshaped_columns:
                c_parent = self._reshaped_columns[c]
                c_shape = result[c_parent].shape[:-2]
                target_shape = self.column_by_name(c).shape[-2:]
                indexes_list = cartesian((range(_) for _ in c_shape))

                i=0
                while i < len(indexes_list):
                    if len(indexes_list)-i >= 3:
                        list_id = [list(indexes_list[_]) for _ in (i, i+1, i+2)]
                        indexes = list(zip(*list_id))
                        i += 3
                    else:
                        indexes = tuple(indexes_list[i])
                        i += 1
                    tmp = result[c_parent][indexes]

                    if len(tmp.shape) == 3:
                        tmp = tmp.transpose((1, 2, 0))
                    else:
                        tmp = tmp.reshape(tmp.shape + (1,))
                    if c_parent in self._label_columns:
                        h, w, _ = tmp.shape
                        h_target, w_target = target_shape

                        if h > h_target or w > w_target:    # Downscale
                            h_bin = int(np.ceil(h / h_target))
                            w_bin = int(np.ceil(w / w_target))

                            tmp = cv2.resize(tmp, dsize=(w_target * w_bin, h_target * h_bin),
                                             interpolation=cv2.INTER_NEAREST)
                            # Most common value pooling
                            W = 1 / 4

                            tmp.shape = (h_target, h_bin, w_target, w_bin)
                            tmp = tmp.transpose((0, 2, 1, 3)).reshape(h_target, w_target, h_bin*w_bin)

                            u, indices = np.unique(tmp, return_inverse=True)
                            count = np.apply_along_axis(np.bincount, 2, indices.reshape(tmp.shape), None, indices.max()+1)\
                                      .astype(np.float)
                            count[:, :, 0] *= W
                            tmp = u[np.argmax(count, axis=2)]
                        else:               # Upscale
                            tmp = cv2.resize(tmp, dsize=target_shape[::-1], interpolation=cv2.INTER_NEAREST)

                    else:
                        interp = cv2.INTER_AREA
                        if tmp.shape[0] > target_shape[0] or tmp.shape[1] > target_shape[1]:
                            interp = cv2.INTER_LINEAR
                        tmp = cv2.resize(tmp, dsize=target_shape[::-1], interpolation=interp)
                    if len(tmp.shape) == 3:
                        tmp = tmp.transpose((2, 0, 1))
                    else:
                        tmp = tmp.reshape((1,) + tmp.shape)
                    r[c][indexes] = tmp

            result = None
            r = None
            yield weakref

    @property
    def size(self):
        return self.parent_dataset.size

    @property
    def label_columns(self):
        return self._label_columns.copy()


########################################################################################################################
class DataSetPatches(AbstractDataSet):
    def __init__(self, dataset, patch_shape, patches_function, post_process=None, n=None, center_pos=False,
                 cache_center=False, keep_patched_columns=False, name='patch'):
        """
        :type dataset: AbstractDataSets
        :param patch_shape: Shape of the patch
        :param columns: columns of 'dataset' that should be patchified
        :param patches_function: The function used to generate patch centers. The following parameters can be provided:
                                    - *column_name*: the column data for the current element
                                    - *column_shape*: a column shape
                                    - patch_shapes: the patch_shapes
                                    - img_shape: the shape of the image from which patches are extracted
                                Such function should either return a list of tuple (center_y, center_x) or
                                a post_process function should be passed to transform this function result to such list
        :param post_process: This function is called after patches function to transform its result into a list of
                             centers (tuple (center_y, center_x)). The default value is lambda x: x.
        """
        super(DataSetPatches, self).__init__(name, dataset, pk_type=str)
        self.patch_size = patch_shape

        # Initialize columns
        self._columns = dataset.copy_columns(self)
        self._patched_columns = {}
        img_shape = None
        for column, patch_def in patch_shape.items():
            # Definition formatting
            parent_column = column
            if isinstance(patch_def[0], int):
                patch_shape = patch_def
            elif isinstance(patch_def[0], str) or isinstance(patch_def[0], DSColumn):
                parent_column = patch_def[0]
                patch_shape = patch_def[1]
            else:
                raise NotImplementedError('Patch definition should be either (column, (path_shape)) or (patch_shape).')

            if isinstance(parent_column, DSColumn):
                if parent_column.dataset is not self.parent_dataset:
                    raise ValueError('%s is not a column of the parent dataset %s' % (parent_column.name, self.parent_dataset))
                parent_column = parent_column.name
            elif isinstance(parent_column, str):
                if parent_column not in self.parent_dataset.columns_name():
                    raise ValueError('%s is not a column of the parent dataset %s' % (parent_column, self.parent_dataset))
            else:
                raise NotImplementedError('patch_shape keys should be either a column name or a DataSetColumn')

            parent_c = self.parent_dataset.column_by_name(parent_column)
            if len(parent_c.shape) not in (2, 3):
                raise ValueError('%s does not support patch (shape: %s)' % (parent_c.name, str(parent_c.shape)))
            if img_shape is None:
                img_shape = parent_c.shape[-2:]
            elif img_shape != parent_c.shape[-2:]:
                raise ValueError('All patched columns should have the same shape (previous shape %s, %s shape: %s)'
                                 % (repr(img_shape), parent_c.name, parent_c.shape[-2:]))
            # Store patch information
            self._patched_columns[column] = (parent_column, patch_shape)

            # Store column information
            if not keep_patched_columns:
                c = self.column_by_name(parent_column)
                if c in self._columns:
                    self._columns.remove(c)

            if column in self.columns_name():
                c = self.column_by_name(column)
                if c in self._columns:
                    self._columns.remove(c)

            column_shape = parent_c.shape[:-2] + patch_shape
            self.add_column(name=column, shape=column_shape, dtype=parent_c.dtype, format=parent_c.format)

        # Initialize patch function
        if post_process is None:
            post_process = identity_function
        self._post_process_function = post_process

        self._patches_function = patches_function
        self._static_kwargs = {c.name + '_shape': c.shape for c in self._columns}
        self._static_kwargs['patch_shapes'] = list({_[1] for _ in self._patched_columns.values()})
        self._static_kwargs['img_shape'] = img_shape
        self._saved_patch_center = None
        self._n = n
        if n is None:
            cache_center = True

        if isinstance(self._patches_function, str):
            if self._patches_function not in dataset.columns_name():
                raise ValueError("%s is not  a column of %s" % (self._patches_function, dataset.dataset_name))
            p_params = [self._patches_function]
        else:
            p_params = not_optional_args(self._patches_function)
        self._kwargs_columns = [_ for _ in self.columns_name() if _ in p_params]

        static_kwargs = {}
        for p in p_params:
            if p not in self._kwargs_columns:
                if p not in self._static_kwargs:
                    raise ValueError("Unknown non optionnal parameter %s can't be supplied to the patches function" % p)
                static_kwargs[p] = self._static_kwargs[p]
        self._static_kwargs = static_kwargs

        self._is_patch_invariant = len(self._kwargs_columns) == 0
        # If needed generate patch
        if self.is_patch_invariant():
            centers = self.patches_function(**self._static_kwargs)
            self._n = len(centers)
            cache_center = False
            if not self._n:
                raise ValueError('No patch was specified!')
            self._saved_patch_center = np.zeros((self._n, 3), dtype=np.uint16)-1
            self._saved_patch_center[:, 1] = list(zip(*centers))[0]
            self._saved_patch_center[:, 2] = list(zip(*centers))[1]
        elif cache_center:
            self._saved_patch_center = self.generate_all_patches_center()
        else:
            self._saved_patch_center = None
        self._cache_center = cache_center

        if center_pos:
            self._columns.append(DSColumn('center_pos', (2,), dtype=int, dataset=self))

    def _setup_determinist(self):
        if not self._cache_center:
            if self._saved_patch_center is not None:
                return
            self._saved_patch_center = [None] * self.parent_dataset.size
            with Process("Caching center positions", self.parent_dataset.size) as p:
                def store_result(c):
                    self._saved_patch_center[c[0, 0]] = c
                    p.update(1)
                parallel_exec(self.generate_one_patches_center, seq=range(self.parent_dataset.size), cb=store_result)

    def generate_all_patches_center(self):
        with Process("Caching center positions", self.parent_dataset.size) as p:
            centers = [0]

            def store_result(c):
                if c.shape[0]:
                    centers.append(c)
                    centers[0] += c.shape[0]
                p.update(1)

            parallel_exec(self.generate_one_patches_center, seq=range(self.parent_dataset.size), cb=store_result)

        n = centers[0]
        centers = centers[1:]

        patch_centers = np.zeros((n, 3), dtype=np.uint16)

        n = 0
        for c in centers:
            n_c = c.shape[0]
            patch_centers[n:n+n_c] = c
            n += n_c

        return patch_centers

    def generate_one_patches_center(self, result):
        if isinstance(result, int):
            result = self.parent_dataset.read_one(result, columns=self._kwargs_columns, extract=False)

        kwargs = self._static_kwargs.copy()
        for c_name in self._kwargs_columns:
            kwargs[c_name] = result[0, c_name]
        centers = self.patches_function(**kwargs)
        if isinstance(centers, tuple):
            centers = [centers]
        np_array = np.zeros((len(centers), 3), dtype=np.uint16)+result.start_id
        if isinstance(centers, list):
            for c_id, c in enumerate(centers):
                np_array[c_id, 1] = c[0]
                np_array[c_id, 2] = c[1]
        else:
            np_array[:, 1:] = centers
        return np_array

    def _generator(self, gen_context):
        columns = gen_context.columns

        center_pos = 'center_pos' in columns

        patched_columns = {c: c_def for c, c_def in self._patched_columns.items() if c in columns}
        copied_columns = [_ for _ in columns if _ not in list(patched_columns.keys()) + ['center_pos']]
        gen_columns = set(copied_columns)
        for c, (c_parent, p_shape) in patched_columns.items():
            gen_columns.add(c_parent)
        if not self._cache_center:
            gen_columns.update(self._kwargs_columns)
        gen_columns = list(gen_columns)

        if not self._cache_center:
            gen = gen_context.generator(self._parent, n=1, columns=gen_columns,
                                        start=gen_context.start_id//self._n, stop=gen_context.stop_id//self._n)
        else:
            gen = gen_context.generator(self._parent, n=1, columns=gen_columns,
                                        start=self._saved_patch_center[gen_context.start_id, 0],
                                        stop=self._saved_patch_center[gen_context.stop_id, 0])

        if not self._cache_center:
            if gen_context.determinist:
                saved_centers = self._saved_patch_center
            else:
                saved_centers = [None] * self.parent_dataset.size
        else:
            saved_centers = self._saved_patch_center

        result = None
        if self._n is None:
            result = next(gen)
            center_i = -1

        while not gen_context.ended():
            i_global, n, weakref = gen_context.create_result()
            r = weakref()

            for i in range(n):
                # Read appropriate result and center
                if not self._cache_center:
                    center_i = (i + i_global) % self._n
                    if center_i == 0 or result is None:
                        try:
                            gen_current_index = gen.current_id
                            result = gen.next(copy={_: r[i:i+1, _] for _ in copied_columns}, r=r)
                        except StopIteration:
                            log.error('Reading from %s failed (i_global=%i, i_global/n=%i, gen_id=%i)'
                                       %(self._parent, i_global, i_global/n, gen_current_index))
                            raise StopIteration('Reading from %s failed (i_global=%i, i_global/n=%i, gen_id=%i)'
                                               %(self._parent, i_global, i_global/n, gen_current_index) )
                    gen_current_index = result.start_id
                    if self.is_patch_invariant():
                        center = saved_centers[center_i, 1:]
                    else:
                        if saved_centers[gen_current_index] is None:
                            saved_centers[gen_current_index] = self.generate_one_patches_center(result)
                        center = saved_centers[gen_current_index][center_i, 1:]
                else:
                    img_id, center_x, center_y = saved_centers[i+i_global]
                    while result is None or img_id != result.start_id:
                        if img_id != gen.current_id:
                            gen = gen_context.generator(self._parent, n=1, start=img_id, columns=gen_columns,
                                                        stop=saved_centers[gen_context.stop_id, 0])
                        try:
                            gen_current_index = gen.current_id
                            result = gen.next(copy={_: r[i:i+1, _] for _ in copied_columns}, r=r)
                        except StopIteration:
                            log.error('Reading from %s failed (i_global=%i, img_id=%i, gen_id=%i, result.start_id=%i)'
                                       % (self._parent, i_global, img_id, gen_current_index, result.start_id))
                            raise StopIteration('Reading from %s failed (i_global=%i, img_id=%i, gen_id=%i, result.start_id=%i)'
                                                % (self._parent, i_global, img_id, gen_current_index, result.start_id))
                        center_i = -1
                    center = (center_x, center_y)
                    center_i += 1

                # Compute patch
                y, x = center
                for c, (c_parent, patch_shape) in patched_columns.items():
                    column_parent = self.parent_dataset.column_by_name(c_parent)
                    h, w = column_parent.shape[-2:]

                    half_x = patch_shape[1] // 2
                    odd_x = patch_shape[1] % 2
                    half_y = patch_shape[0] // 2
                    odd_y = patch_shape[0] % 2

                    y0 = int(max(0, half_y - y))
                    y1 = int(max(0, y - half_y))
                    h0 = int(min(h, y + half_y + odd_y) - y1)

                    x0 = int(max(0, half_x - x))
                    x1 = int(max(0, x - half_x))
                    w0 = int(min(w, x + half_x + odd_x) - x1)
                    if h0 <= 0 or w0 <= 0:
                        continue

                    if len(column_parent.shape) == 2:
                        r[c][i, y0: y0 + h0, x0: x0 + w0] = \
                            result[c_parent][0, y1:y1 + h0, x1:x1 + w0]
                    elif len(column_parent.shape) == 3:
                        r[c][i, :, y0: y0 + h0, x0: x0 + w0] = \
                            result[c_parent][0, :, y1:y1 + h0, x1:x1 + w0]
                    else:
                        raise NotImplementedError

                # Other columns
                if center_pos:
                    r[i, 'center_pos'] = center
                for copied_column in copied_columns:
                    r[i, copied_column] = result[0, copied_column]
                r[i, 'pk'] = str(result[0, 'pk'])+'-'+repr((y, x))
            r = None
            yield weakref
            i_global += n

    def is_patch_invariant(self):
        return self._is_patch_invariant

    def patches_function(self, **kwargs):
        if isinstance(self._patches_function, str):
            return self._post_process_function(kwargs[self._patches_function])
        return self._post_process_function(match_params(self._patches_function, **kwargs))

    @property
    def size(self):
        if self._n is not None:
            return self._n * self.parent_dataset.size
        return self._saved_patch_center.shape[0]

    @property
    def patches_def(self):
        return self._patched_columns


########################################################################################################################
class DataSetUnPatch(AbstractDataSet):
    def __init__(self, dataset, columns=None, patch_mix='replace', restore_columns=None, columns_shape=None,
                 n_patches=None):
        super(DataSetUnPatch, self).__init__(name='unpatch', parent_datasets=dataset, pk_type=str)

        if columns_F is None:
            columns_shape = {}
        if n_patches is None:
            n_patches = 4

        # Find DataSetPatches ancestor
        self._patch_dataset = dataset
        while self._patch_dataset is not None and not isinstance(self._patch_dataset, DataSetPatches):
            from .datasets_core import DataSetShuffle
            if isinstance(self._patch_dataset, DataSetShuffle):
                raise ValueError('Impossible to unpatch %s once it has been shuffled!' % dataset)
            self._patch_dataset = self._patch_dataset.parent_dataset

        if self._patch_dataset is None:
            raise ValueError('None of the ancestors of %s is a patch dataset.' % dataset)

        if columns is not None:
            if isinstance(columns, str):
                if columns == '*':
                    columns = dataset.columns_name()
                else:
                    columns = [columns]

        # Initialize columns
        self._columns = dataset.copy_columns(self)
        if 'center_pos' in self.columns_name():
            del self._columns[self.column_index('center_pos')]

        if restore_columns is None:
            if columns is None:
                restore_columns = True
                for c in self._patch_dataset.patches_def.keys():
                    if c not in dataset.columns_name():
                        restore_columns = False
            else:
                restore_columns = True
                patched_columns = {_[0] for _ in self._patch_dataset.patches_def.values()}
                for c in columns:
                    if c not in patched_columns:
                        restore_columns = False

        self._unpatched_columns = {}  # {own_column_name: [column_source_1, ...], ...}

        if restore_columns:
            # Dataset will try to recreate columns as they where before patching
            if columns is None:
                columns = list({_[0] for _ in self._patch_dataset.patches_def.values()})
            for own_column in columns:
                child_columns = {c: p_shape for c, (c_parent, p_shape) in self._patch_dataset.patches_def.items()
                                            if c_parent == own_column}
                children_accessible = True
                for c in child_columns:
                    if c not in dataset.columns_name():
                        children_accessible = False
                        log.warn("Patched column %s is not accessible, %s won't be restored." % (c, own_column))
                        break

                if children_accessible:
                    patch_layer_shape = None
                    for c in child_columns:
                        s = self.column_by_name(c).shape[:-2]
                        if patch_layer_shape is None:
                            patch_layer_shape = s
                        elif patch_layer_shape != s:
                            raise ValueError('All unpatched columns should have the same layer shape! '
                                             '(%s shape: %s, previous shape: %s)' % (c, s, patch_layer_shape))

                        del self._columns[self.column_index(c)]
                    parent_column = self.patch_dataset.parent_dataset.column_by_name(own_column)
                    self._unpatched_columns[own_column] = list(child_columns.keys())
                    self.add_column(name=own_column, dtype=parent_column.dtype, format=parent_column.format,
                                    shape=patch_layer_shape+parent_column.shape[-2:])
        else:
            # Dataset will assemble patches into separated columns
            if columns is None:
                columns = list({_ for _ in self.columns_name() if _ in self._patch_dataset.patches_def.keys()
                                                               or _ not in self._patch_dataset.columns_name()})
            for own_column in columns:
                self._unpatched_columns[own_column] = [own_column]
                shape = self.column_by_name(own_column).shape[:-2]

                if own_column in columns_shape:
                    shape += columns_shape[own_column]
                else:
                    c_parent_name = list(self._patch_dataset.patches_def.values())[0][0]
                    c_parent = self._patch_dataset.parent_dataset.column_by_name(c_parent_name)
                    shape += c_parent.shape[-2:]
                self.column_by_name(own_column)._shape = shape

        if not self._unpatched_columns:
            raise ValueError('No columns was accessible to unpatch. Aborting...')

        # Options
        acceptable_mix = ('replace',)
        if patch_mix not in acceptable_mix:
            raise NotImplementedError('%s is an unknown patch_mix method.\n Should be one of %s.'
                                      % (patch_mix, repr(acceptable_mix)))
        self.patch_mix_method = patch_mix
        self._restore_columns = restore_columns
        self.n_patches = n_patches

    @property
    def size(self):
        return self._patch_dataset.parent_dataset.size

    @property
    def patch_dataset(self):
        return self._patch_dataset

    @property
    def unite_columns(self):
        return self._restore_columns

    def _generator(self, gen_context):
        columns = gen_context.columns

        unpatched_columns = {c: c_def for c, c_def in self._unpatched_columns.items() if c in columns}
        copied_columns = [_ for _ in columns if _ not in unpatched_columns]
        gen_columns = set(copied_columns)
        for c, c_parent in unpatched_columns.items():
            gen_columns.update(set(c_parent))
        gen_columns = list(gen_columns)

        gen = None

        while not gen_context.ended():
            i_global, n, weakref = gen_context.create_result()
            r = weakref()

            for i in range(n):
                # Read patches index
                start_patch, n_patch = self.patch_range(i+i_global)
                i_patch = 0
                while i_patch < n_patch:
                    # Read patch
                    if gen is None or gen.current_id != start_patch+i_patch:
                        gen = gen_context.generator(self.parent_dataset, n=self.n_patches, columns=gen_columns,
                                                    start=start_patch + i_patch, stop=self.parent_dataset.size)
                    n_patches = min(self.n_patches, n_patch-i_patch)
                    result = gen.next(limit=n_patches, r=r)

                    # Apply patch
                    for i_patches in range(n_patches):
                        key = result[i_patches, 'pk']
                        pk = key[:key.rfind('-')]
                        y, x = (int(_) for _ in key[key.rfind('(') + 1:key.rfind(')')].split(','))
                        if self.patch_mix_method == 'replace':
                            for c, parent_c in unpatched_columns.items():
                                for parent in parent_c:
                                    patch = result[i_patches, parent]
                                    patch_shape = result[i_patches, parent].shape[-2:]

                                    if hasattr(self.patch_dataset, "stride"):
                                        stride = self.patch_dataset.stride
                                        if stride[0] < patch_shape[0] or stride[1] < patch_shape[1]:
                                            y0 = max(0, (patch_shape[0] - stride[0]) // 2)
                                            x0 = max(0, (patch_shape[1] - stride[1]) // 2)
                                            h0 = min(patch_shape[0], stride[0])
                                            w0 = min(patch_shape[1], stride[1])
                                            patch = patch[..., y0:y0+h0, x0:x0+w0]
                                            patch_shape = (h0, w0)

                                    s_out = r[i, c].shape
                                    h, w = s_out[-2:]

                                    half_x = patch_shape[1] // 2
                                    odd_x = patch_shape[1] % 2
                                    half_y = patch_shape[0] // 2
                                    odd_y = patch_shape[0] % 2

                                    y0 = int(max(0, half_y - y))
                                    y1 = int(max(0, y - half_y))
                                    h0 = int(min(h, y + half_y + odd_y) - y1)

                                    x0 = int(max(0, half_x - x))
                                    x1 = int(max(0, x - half_x))
                                    w0 = int(min(w, x + half_x + odd_x) - x1)
                                    if h0 <= 0 or w0 <= 0:
                                        continue

                                    win_out = [slice(None, None)]*(len(s_out)-2) + \
                                              [slice(y1, y1+h0), slice(x1, x1+w0)]
                                    win_patch = [slice(None, None)]*(len(s_out)-2) + \
                                                [slice(y0, y0+h0), slice(x0, x0+w0)]
                                    r[i, c][win_out] = patch[win_patch]
                        else:
                            raise NotImplementedError
                    i_patch += n_patches

                for c in copied_columns+["pk"]:
                    r[i, c] = result[0, c]

            result = None
            r = None
            yield weakref

    def patch_range(self, image_id):
        if image_id > self.size:
            return None, 0

        n = self.patch_dataset._n
        if n is not None:
            return image_id*n, n

        ids = np.where(self.patch_dataset._stored_center[:, 0] == image_id)
        return ids[0], len(ids)
