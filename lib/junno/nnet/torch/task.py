import torch
from ... import log
import weakref


class Task:
    """

    =====
    Process
    ====
    Definition
    --------

    """
    __metrics = []

    def __init__(self):
        self._train_set = None
        self._valid_set = None
        self._test_set = None
        pass

    def fit(self, data=None, epoch=1, loss=None):
        pass

    def infer(self, data=None):
        pass

    def test(self, data=None, metrics=None):
        pass

    def __call__(self, *args, **kwargs):
        pass

    # ---  Processing ---
    class ProcessStep:
        def __init__(self, step, source=None, out=None, format='function'):
            pass

        def __set_name__(self, owner, name):
            print(owner, name)

        def __call__(self, *args, source=None):
            pass

    @staticmethod
    def process_step(source=None, out=None, format='function'):
        def decorator(f):
            step = Task.ProcessStep(step=f, source=source, out=out, format=format)
            return step
        return decorator

    @staticmethod
    def preprocess():
        pass

    @staticmethod
    def postprocess():
        pass

    @staticmethod
    def loss():
        pass

    @staticmethod
    def metric():
        pass

    # ---  Dataset  ---
    @property
    def train_set(self):
        return self._train_set

    @train_set.setter
    def train_set(self, set):
        pass

    # ---   Metrics   ---
    @property
    def metrics(self):
        return [_() for _ in self.__metrics]

    # ---  Hyper Parameters  ---
    def optimize_hyperparameters(self, hp, method):
        pass


class HyperParameter:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def expected_distribution(self):
        pass


class Metric:
    def __init__(self, f):
        self._f = f
        self._task = None
        self._name = None

    def __set_name__(self, owner: Task, name):
        self._task = owner
        self._name = name
        owner.__metrics.append(weakref.ref(self))

    def __call__(self, *args, **kwargs):
        pass

    def compute(self, data):
        pass


class Trials:
    def __init__(self,  path=None):
        pass

    def best(self):
        pass

    def summery(self):
        pass


class Trial:
    def __init__(self, model, hyperparameter, history):
        pass