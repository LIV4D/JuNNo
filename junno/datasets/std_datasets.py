import os.path
import pickle
import numpy as np

from ..j_utils import log as _log
from .dataset import NumPyDataSet


def cache_download(url, name):
    filename = url.split('/')[-1]
    download_dir = os.path.join(os.path.dirname(__file__), '__pycache__', name)
    file_path = os.path.join(download_dir, filename)

    if not os.path.exists(file_path):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        with _log.Process('Download %s' % name, total=1, verbose=False) as p:
            from six.moves import urllib

            def _print_progress(count, block_size, total_size):
                completion = float(count * block_size) / total_size
                p.step = completion

            file_path, _ = urllib.request.urlretrieve(url=url,
                                                      filename=file_path,
                                                      reporthook=_print_progress)
    return file_path


def cifar10_datasets(true_label=False):
    archive_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    archive_file = cache_download(archive_url, 'CIFAR10')

    with _log.Process('Extracting CIFAR10', total=6, verbose=False) as p:
        import tarfile
        tar = tarfile.open(archive_file, mode="r:gz")
        d = {}

        # Initialize reading function
        prefix = 'cifar-10-batches-py/'
        n_file = 10000

        def retreive_data_from_file(f):
            data = pickle.load(f, encoding='bytes')
            img = np.array(data[b'data'], dtype=np.float32)/255.
            img = img.reshape((-1, 3, 32, 32))
            # Convert RGB -> BGR
            img = img[:, ::-1]

            label = np.array(data[b'labels'])

            return img, label

        if true_label:
            with tar.extractfile(prefix+"batches.meta") as f_data:
                raw_labels = pickle.load(f_data)
            true_label = {i: n for (i,n) in enumerate(raw_labels['label_names'])}

        # Read train data
        x = np.zeros(shape=[5*n_file, 3, 32, 32], dtype=float)
        y = np.zeros(shape=[5*n_file], dtype=int)

        for i in range(5):
            # Load 5 train batches
            with tar.extractfile(prefix+'data_batch_%i' % (i+1)) as f_data:
                img, label = retreive_data_from_file(f_data)
                x[i*n_file:(i+1)*n_file] = img
                y[i*n_file:(i+1)*n_file] = label
            p.step = i+1
        if true_label:
            d['train'] = NumPyDataSet(x=x, y=y, labels=np.vectorize(true_label.get)(y))
        else:
            d['train'] = NumPyDataSet(x=x, y=y)

        # Read test data
        with tar.extractfile(prefix+'test_batch') as f_data:
            x, y = retreive_data_from_file(f_data)
        if true_label:
            d['test'] = NumPyDataSet(x=x, y=y, labels=np.vectorize(true_label.get)(y))
        else:
            d['test'] = NumPyDataSet(x=x, y=y)

    return d


def mnist_datasets(shuffle_train=False):
    archive_url = "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"
    archive_file = cache_download(archive_url, 'CIFAR10')

    with _log.Process('Extracting MNIST', verbose=False):
        import gzip
        with gzip.open(archive_file, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)

        def to_dataset(data_xy):
            x = np.reshape(data_xy[0], (data_xy[0].shape[0], 1, 28, 28))
            return NumPyDataSet(x=x, y=data_xy[1])

        d = {}
        if shuffle_train:
            train_idx = np.arange(train_set[0].shape[0])
            np.random.shuffle(train_idx)
            train_set = tuple(train_set[_][train_idx] for _ in (0, 1))
        d['train'] = to_dataset(train_set)
        d['validation'] = to_dataset(valid_set)
        d['test'] = to_dataset(test_set)

        return d
