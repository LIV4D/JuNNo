import numpy as np
import cv2
import pandas
from os import walk
from os.path import basename, join, abspath, dirname, normpath, splitext
import inspect
from re import search

from .dataset import AbstractDataSet
from ..j_utils import log
from ..j_utils.function import match_params


########################################################################################################################
def image_extensions():
    return '\.(gif|jpg|jpeg|JPG|JPEG|tiff|tif|png|ppm)$'


########################################################################################################################
class FilesCollection(AbstractDataSet):
    """
    Construct a dataset by listing and reading files from a folder. The exploration of the folder
    can be recursive and a filter can be applied to file's name by providing a regexp.
    One should also provide a function which will read the file and
    """

    def __init__(self, path, read_function, regexp='', remove_extension=True, filename_regexp=False, recursive=True,
                 name='FileCollection'):
        """
        :param path: Path of the folder to explore
        :param read_function: The function called to read data from a files.
                              The possible parameters of such function may be:
                                - bin: a binary reading stream to the file
                                - text: a text reading stream to the file
                                - path: an absolute path to the file
                                - sub_path: a relative path to the file
                                - sub_dir: a relative path to the parent directory of the file
                                - filename: the file name
                                - name: the file name without the extension
                                - file_ext: the file extension

        :param regexp: Regular Expression used to filter file's name. (If empty, all files are accepted)
        :param filename_regexp: If True, the regexp is only applied to filename, otherwise it's applied to file
                                path from the 'path' folder (the file path includes the recursive sub-folders name)
        :param recursive: If true the folder will be explored recursively (including all its sub-folders)
                          Otherwise only the files in the path folder will be listed
        """
        super(FilesCollection, self).__init__(name, pk_type=str)
        self.path = path
        self.regexp = regexp
        self.filename_regexp = filename_regexp
        self.recursive = recursive
        self.remove_extension = remove_extension
        self._files = np.zeros(shape=(), dtype=str)
        self.update_files()

        if not len(self._files):
            raise ValueError('No files in %s match the filter of this FilesCollection, aborting...' % path)

        params = inspect.signature(read_function).parameters.keys()
        self._open_as = 'path'
        if 'bin' in params:
            if 'text' in params:
                raise ValueError('The parameters of the read_function should contain either'
                                 ' a parameters named bin or text (not both...)')
            self._open_as = 'bin'
        elif 'text' in params:
            self._open_as = 'text'
        elif 'path' not in params:
            raise ValueError('The parameters of the read_function should contain either'
                             ' a parameters named bin or text or path (none was provided)')
        self._read = read_function
        self.len_path = len(abspath(self.path))
        self.len_path += 1 if self.len_path else 0

        self.add_column('name', (), str)
        sample = self._read_files(self._files[0])
        if type(sample) == np.ndarray:
            self.add_column('data', sample.shape, sample.dtype)
        else:
            self.add_column('data', (), type(sample))

    def update_files(self):
        """
        Update the files list. This may change the size property!
        """
        files_list = []
        for root, dirs, files in walk(self.path, topdown=True):
            abs_root = abspath(root)
            for file in files:
                filename = file if self.filename_regexp else normpath(join(root, file))
                if isinstance(self.regexp, str):
                    if search(self.regexp, filename) is None:
                        continue
                elif callable(self.regexp):
                    if not match_params(self.regexp, args=[filename], filename=file, path=root,
                                        filepath=normpath(join(abs_root, file)),  file=normpath(join(root, file))):
                        continue
                files_list.append(normpath(join(abs_root, file)))
            if not self.recursive:
                break

        self._files = np.array(files_list, dtype='str')

    @property
    def files(self):
        return self._files

    def _generator(self, gen_context):
        while not gen_context.ended():
            i_global, n, weakref = gen_context.create_result()
            r = weakref()

            for i in range(n):
                path = self._files[i+i_global]
                pk = path[self.len_path:]
                name = basename(path)
                if self.remove_extension:
                    name, ext = splitext(name)
                    if ext:
                        pk = pk[:-len(ext)-1]
                r[i, 'pk'] = pk
                r[i, 'name'] = name
                if 'data' in r:
                    sample = self._read_files(path)
                    if sample is not None:
                        if len(self.columns.data.shape):
                            shape = [min(s1, s2) for s1, s2 in zip(sample.shape, self.columns.data.shape)]
                            r[i, 'data'][[slice(_) for _ in shape]] = sample[[slice(_) for _ in shape]]
                        else:
                            r[i, 'data'] = sample
                        del sample
            r = None

            yield weakref

    def _read_files(self, path):
        sub_path = path[self.len_path:]
        sub_dir = dirname(sub_path)
        filename = basename(sub_path)
        splited = filename.split('.')
        name = '.'.join(splited[:-1])
        file_ext = splited[-1]
        kwarg = {'path': path, 'sub_path': sub_path, 'sub_dir': sub_dir,
                 'filename': filename, 'name': name, 'file_ext': file_ext}
        if self._open_as == 'path':
            sample = match_params(self._read, **kwarg)
        elif self._open_as == 'bin':
            with open(path, 'rb') as f:
                sample = match_params(self._read, bin=f, **kwarg)
        elif self._open_as == 'text':
            with open(path, 'r') as f:
                sample = match_params(self._read, bin=f, **kwarg)
        else:
            raise NotImplementedError
        return sample

    @property
    def size(self):
        return self._files.shape[0] if len(self._files.shape) else 0


########################################################################################################################
class ImagesCollection(FilesCollection):
    def __init__(self, path, name='ImagesCollection', regexp=image_extensions(), filename_regexp=False, recursive=True,
                 imread_flags=cv2.IMREAD_UNCHANGED, crop=None, reshape=None, normalize=True, keep_proportion=False):
        """

        :param path: Path of the folder to explore
        :param regexp: Regular Expression used to filter file's name. (If empty, all files are accepted)
        :param filename_regexp: If True, the regexp is only applied to filename, otherwise it's applied to file
                                path from the 'path' folder (the file path includes the recursive sub-folders name)
        :param recursive: If true the folder will be explored recursively (including all its sub-folders)
                          Otherwise only the files in the path folder will be listed
        :param imread_flags: flag passed to imread: cv2.IMREAD_COLOR or cv2.IMREAD_GRAYSCALE or cv2.IMREAD_UNCHANGED (default)
        :param crop: Define how the image should be cropped. If None, the image is not cropped.
                     crop can be:
                        - 'auto': Automatically crop the image with the default blur and threshold
                                  (applied on the mean of all the image's layers)
                        - {'threshold': , 'blur': }: Automatically crop the image with a specific threshold and blur
                        - (y, x, h, w): Crop the image using a specific rectangle
                        - (h, w): Same as above with (y, x) = (0, 0)
        :param reshape: Define how the image should be reshaped. If None, the image is not reshaped.
                        reshape can be:
                        - (h, w): a tupple containing the desired width and height
                        - f: a tuple containing a reduction factor (f<1)
                        - (fy, fx): a tuple containing reduction factors for horizontal and vertical dimensions
                        - {'shape': (h,w), 'f': (fy, fx), 'interp': }: a dictionnary containing either 'shape' or
                                                                            'f', and specifying an interpolation
                        Note: the default interpolation is cv2.INTER_AREA
        :param normalize: The output image's values are normalized (divided by 255) and casted into np.float32
        :param keep_proportion: Prevent the stretching of the image. It can be:
                        -'crop' Will crop the image to match the required size after resizing operation without stretching
                        -'pad' Will pad the image to match the required size after resizing operation without stretching
                        -None or False: The image will be stretched
        """

        # --- READ FUNCTION OPTIONS ---
        self.imread_flags = imread_flags

        # Crop
        self.crop = crop
        self.c_threshold = 10
        self.c_blur = 10
        if isinstance(crop, dict):  # Auto crop
            self.c_threshold = crop.get('threshold', self.c_threshold)
            self.c_blur = crop.get('blur', self.c_blur)

        self.keep_proportion = keep_proportion

        # Reshape
        self.file_reshape = reshape
        self.r_shape = None
        self.r_f = None
        self.r_interp = cv2.INTER_AREA
        if isinstance(reshape, float):
            self.r_f = (reshape, reshape)
        elif isinstance(reshape, tuple):
            h, w = reshape
            if w <= 1. and h <= 1. and (w != 1 or h != 1):
                self.r_f = reshape
            else:
                self.r_shape = reshape
        elif isinstance(reshape, dict):
            self.r_shape = reshape.get('shape', self.r_shape)
            self.r_f = reshape.get('f', self.r_f)
            self.r_interp = reshape.get('interp', self.r_interp)

        # Finalize
        self.normalize = normalize

        super(ImagesCollection, self).__init__(path=path, read_function=self.read_func, recursive=recursive,
                                               regexp=regexp, filename_regexp=filename_regexp, name=name)

    def __str__(self):
        return self.dataset_name + ' ' + self.path

    def read_func(self, path):
        # --- READ ---
        img = cv2.imread(path, flags=self.imread_flags)
        if img is None:
            log.error('%s is not an image!' % path)
            return None
        # --- CROP ---
        crop = self.crop
        if crop is not None:
            if isinstance(crop, tuple):
                if len(crop) == 2:
                    h, w = crop
                    x = 0
                    y = 0
                elif len(crop) == 4:
                    y, x, h, w = crop
                else:
                    raise NotImplementedError
                img = img[y:y + h, x:x + w]
            else:
                from ..j_utils.image import auto_crop
                img = auto_crop(img, threshold=self.c_threshold, blur=self.c_blur)

        # --- RESHAPE ---
        if self.file_reshape is not None:
            h, w, c = img.shape
            if self.r_shape is None:
                self.r_shape = (int(h * self.r_f[1]), int(w * self.r_f[0]))
            original_ratio = h / w
            new_ratio = self.r_shape[1] / self.r_shape[0]

            if self.keep_proportion and new_ratio != original_ratio:
                if self.keep_proportion == 'pad':
                    if original_ratio < new_ratio:
                        pad = int((new_ratio*w - h)//2)
                        pad_width = [(pad, pad), (0,0), (0,0)]
                    else:
                        pad = int((h/new_ratio - w)//2)
                        pad_width = [(0,0), (pad, pad), (0,0)]
                    img = np.pad(img, pad_width, 'constant')
                elif self.keep_proportion == 'crop':
                    if original_ratio > new_ratio:
                        crop = int(h-new_ratio*w)//2
                        img = img[crop:h-crop]
                    else:
                        crop = int((w-h/new_ratio)//2)
                        img = img[:, crop:w - crop]

            img = cv2.resize(img, dsize=self.r_shape, interpolation=self.r_interp)

        # --- FINALISE ---
        if self.normalize:
            img = img.astype(dtype=np.float32) / 255.
        if img.ndim == 3:
            return img.transpose((2, 0, 1))
        elif img.ndim == 2:
            return img.reshape((1, img.shape[0], img.shape[1]))
        else:
            raise NotImplementedError


########################################################################################################################
########################################################################################################################
class DataSetPandaDF(AbstractDataSet):
    def __init__(self, panda_dataframe, mapping=None, name='DataSetDataFrame'):
        pk_dtype = panda_dataframe.axes[0].dtype
        super(DataSetPandaDF, self).__init__(name, pk_type=pk_dtype)
        self._df = panda_dataframe
        if mapping is not None:
            self.column_index = {}
            for c_name, c in mapping.items():
                if isinstance(c, str):
                    self.column_index[c_name] = np.where(self._df.axes[1] == c)[0][0]
                elif isinstance(c, int):
                    self.column_index[c_name] = c
        else:
            self.column_index = {str(c_name): c_id for c_id, c_name in enumerate(self._df.axes[1])}

        for c_name, c_id in self.column_index.items():
            if self._df.dtypes[c_id] == object:
                single_element = self._df[self._df.columns[c_id]].values[0]
                self.add_column(c_name, (), type(single_element))
            else:
                self.add_column(c_name, (), self._df.dtypes[c_id])

    @property
    def size(self):
        return len(self._df.axes[0])

    def _generator(self, gen_context):
        while not gen_context.ended():
            i_global, n, weakref = gen_context.create_result()
            r = weakref()
            r[:, 'pk'] = list(self._df.axes[0][i_global:i_global+n])

            # data = self._df.values
            for c in gen_context.columns:
                if c != 'pk':
                    data = self._df[self._df.columns[self.column_index[c]]].values
                    r[:, c] = data[i_global:i_global+n]
            yield r
            i_global += n


########################################################################################################################
def excel_dataset(path, mapping=None, **kwargs):
    df = pandas.read_excel(path)
    if mapping is not None:
        kwargs.update(mapping)
    return DataSetPandaDF(df, mapping=kwargs, name=basename(path))


########################################################################################################################
def csv_dataset(path, mapping=None, **kwargs):
    df = pandas.read_csv(path)
    if mapping is not None:
        kwargs.update(mapping)
    return DataSetPandaDF(df, mapping=kwargs, name=basename(path))