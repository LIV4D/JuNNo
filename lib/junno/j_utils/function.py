import inspect
from abc import ABCMeta
from functools import partial, wraps
from .image import cast_shape


def match_params(method, args=None, **kwargs):
    """
    Call the specified method, matching the arguments it needs with those,
    provided in kwargs. The useless arguments are ignored.
    If some not optional arguments is missing, a ValueError exception is raised.
    :param method: The method to call
    :param kwargs: Parameters provided to method
    :return: Whatever is returned by method (might be None)
    """
    if issubclass(type(method), PickleableStaticMethod):
        method = method.fn
    method_params = inspect.signature(method).parameters.keys()
    method_params = {_: kwargs[_] for _ in method_params & kwargs.keys()}

    if args is None:
        args = []
    i_args = 0
    for not_opt in not_optional_args(method):
        if not_opt not in method_params:
            if i_args < len(args):
                method_params[not_opt] = args[i_args]
                i_args += 1
            else:
                raise ValueError('%s is not optional to call method: %s.' % (not_opt, method))

    return method(**method_params)


def not_optional_args(f):
    """
    List all the parameters not optional of a method
    :param f: The method to analise
    :return: The list of parameters
    :rtype: list
    """
    if issubclass(type(f), PickleableStaticMethod):
        f = f.fn
    sig = inspect.signature(f)
    return [p_name for p_name, p in sig.parameters.items()
            if isinstance(inspect._empty, type(p.default)) and inspect._empty == p.default]


def optional_args(f):
    """
    List all the parameters optional of a method
    :param f: The method to analise
    :return: Dictionary of optional params and their default value
    :rtype: dict
    """
    if issubclass(type(f), PickleableStaticMethod):
        f = f.fn
    sig = inspect.signature(f)
    return {p_name: p.default for p_name, p in sig.parameters.items() if p.default != inspect._empty}


def bind_args(f, args=(), kwargs=None):
    bind = bind_args_partial(f, args, kwargs)
    missing_args = set(not_optional_args(f)).intersection(bind.keys())
    missing_args.difference_update({'self'})
    if missing_args:
        raise ValueError("%s() missing %i required arguments: '%s'"
                         % (f.__name__, len(missing_args), "', '".join(missing_args)))
    return bind


def bind_args_partial(f, args=(), kwargs=None):
    from collections import OrderedDict
    if kwargs is None:
        kwargs = {}
    params = list(inspect.signature(f).parameters.keys())
    bind = OrderedDict()
    for i, a in enumerate(args):
        if params[i] in kwargs:
            raise ValueError("%s() got multiple value for argument '%s'" % (f.__name__, params[i]))
        bind[params[i]] = a
    for k, a in kwargs.items():
        bind[k] = a
    return bind


def memoized(f):
    memo = {}

    @wraps(f)
    def wrapper(*args, **kwargs):
        params = bind_args(f, args, kwargs)
        memo_key = repr(dict(params))[1:-1]
        try:
            return memo[memo_key]
        except KeyError:
            r = f(**params)
            memo[memo_key] = r
            return r

    return wrapper


def function2str(f):
    src_lines = inspect.getsourcelines(f)
    useless_space_count = len(src_lines[0][0]) - len(src_lines[0][0].lstrip())

    src = ''
    for s in src_lines[0][0:]:
        src += s[useless_space_count:]

    if not src.startswith('def'):
        raise NotImplementedError('Lambda function export to source is not yet implemented, use a standard def function')
    if 'import' in src:
        raise PermissionError('For security reasons, import are not allowed in exportable function')

    sig = inspect.signature(f)
    params = sig.parameters.keys()
    default_params = {n: p.default for n, p in sig.parameters.items() if p.default != inspect._empty}
    return src, params, default_params


def str2function(f, global_var=None):
    if global_var is None:
        global_var = dict()

    f = f.lstrip()
    if f.startswith('def'):
        f_name = f[4:f.index('(')]
    elif ' = lambda ' in f:
        f_name = f[:f.index(' ')]
    else:
        raise ValueError('Invalid function definition: %s' % f)
    exec(f, global_var)
    return global_var[f_name]


def to_callable(o):
    return partial(identity_function, o)


def identity_function(o):
    return o


def empty_function():
    pass


def theano_nice_function(f, inputs, format_f=None):
    return partial(_theano_nice_function, theano_nice_function_option=(f, inputs, format_f))


def _theano_nice_function(*args, theano_nice_function_option=None, **kwargs):
    import numpy as np
    f, inputs, format_f = theano_nice_function_option

    input_data = []
    use_kwargs = False

    if len(args) == 1:
        if 'dataset.DataSetResult' in str(type(args[0])):
            kwargs = dict(args[0])
            args = ()
        if isinstance(args, dict):
            kwargs = args[0]
            args = ()

    for input, shape in inputs.items():
        # Read input value
        if input in kwargs:
            use_kwargs = True
            val = kwargs[input]
        elif use_kwargs:
            raise ValueError('%s is not an optional parameter' % input)
        elif len(args) > 0:
            val = args[0]
            args = args[1:]

        # Check input shape
        if type(val) != np.ndarray:
            val = np.array(val)
        try:
            val = cast_shape(val, shape)
        except ValueError:
            raise ValueError('%s should be of shape %s and not %s'
                             % (input, repr(shape), repr(val.shape)))

        input_data.append(val)

    if callable(format_f):
        return format_f(f(*input_data))
    return f(*input_data)


########################################################################################################################
########################################################################################################################
class FFDictionary:
    """
    Dictionary of function (static method) used to generate FunctionPointer.
    This is the part of FunctionFactory that should be inherited
    """
    def __init__(self, name, params=None, imported_modules=None):
        """
        :param name: name of the dictionary, meanly used to generate explicit exceptions
        :param params: parameters passed to the functions when they will actually be called
        """
        if params is None:
            params = []
        self.imported_modules = imported_modules if imported_modules is not None else {}
        self.params = params
        self.name = name

    def __repr__(self):
        return 'functions dictionary: %s' % self.name

    def not_optional_args(self, f_name):
        f = getattr(self, f_name, None)
        if f is None:
            raise ValueError('%s is not a function in %s' % (f_name, self.name))
        return not_optional_args(f)

    def function_optional_args(self, f_name):
        f = getattr(self, f_name, None)
        if f is None:
            raise ValueError('%s is not a function in %s' % (f_name, self.name))
        return optional_args(f)

    def factory(self):
        return FunctionFactory(self)


########################################################################################################################
class FFunctionPointer:
    def __init__(self, f, definition, params):
        self.f = f
        self.definition = definition
        self.params = params

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.name
        elif not isinstance(other, FFunctionPointer):
            return False
        if other.definition != self.definition or other.params != self.params:
            return False
        return True

    def __repr__(self):
        return 'FFunctionPointer: '+repr(self.definition)

    @property
    def name(self):
        f_name = self.f.__name__
        return f_name if f_name != '<lambda>' else 'f'

    def __call__(self, *args, **params):
        complete_params = self.params.copy()
        complete_params.update(params)
        return match_params(self.f, args, **complete_params)


########################################################################################################################
class FFCallablePrototype:
    def __init__(self, name):
        self.name = name
        self.args = {}

    def __call__(self, **kwargs):
        self.args.update(kwargs)
        return self

    def __repr__(self):
        s = self.name + '('
        for name, arg in self.args.items():
            s += '%s=%s' % (name, str(arg))
        return s + ')'


########################################################################################################################
class FunctionFactory:
    Dictionary = FFDictionary

    def __init__(self, function_dict):
        """
        :type function_dict: FFDictionary
        """
        self.functions_dict = function_dict

    def __getattr__(self, item):
        if item.startswith('_'):
            return getattr(super(FunctionFactory, self), item)
        try:
            if not hasattr(self.functions_dict, item):
                raise ValueError('%s is not a function from %s' % (item, self.functions_dict.name))
            return FFCallablePrototype(item)

        except IndexError as err:
            raise AttributeError(err.args[0])

    @property
    def name(self):
        return self.functions_dict.name

    @property
    def params(self):
        return self.functions_dict.params

    def to_function_pointer(self, f_data):
        imported_modules = self.functions_dict.imported_modules

        if isinstance(f_data, FFunctionPointer):
            return f_data
        elif isinstance(f_data, str):
            if f_data.startswith('def') or ' = lambda' in f_data:
                f = str2function(f_data, imported_modules)
                return FFunctionPointer(f, f_data, {})
            else:
                f_data = (f_data, {})
        elif isinstance(f_data, FFCallablePrototype):
            f_data = (f_data.name, f_data.args)

        if isinstance(f_data, tuple):
            f_name, f_args = f_data

            # Check function name
            if not hasattr(self.functions_dict, f_name):
                raise ValueError('%s is not a function from %s' % (f_name, self.name))

            # Check function arguments
            f_optional_args = self.functions_dict.function_optional_args(f_name)
            for arg in f_args:
                if arg not in f_optional_args:
                    print('Argument %s is useless when using %s.%s, it will be ignored.'
                          % (arg, self.name, f_name))

            for arg in self.functions_dict.not_optional_args(f_name):
                if arg not in self.params and arg not in f_args.keys():
                    raise ValueError('Error when using %s.%s: Missing not optional parameter %s'
                                     % (self.name, f_name, arg))

            # Create function pointer
            f = getattr(self.functions_dict, f_name)
            if not f_data[1]:
                f_data = f_data[0]
            return FFunctionPointer(f, f_data, f_args)

        elif callable(f_data):
            f_code, f_args, f_default_args = function2str(f_data)
            f_not_optional_args = [_ for _ in f_args if _ not in f_default_args]

            # Check function arguments
            for arg in f_not_optional_args:
                if arg not in self.params:
                    raise ValueError('Error when using custom function as %s: unknown parameter %s'
                                     % (self.name, arg))

            return FFunctionPointer(str2function(f_code, imported_modules), f_code, {})


########################################################################################################################
########################################################################################################################
class PickleableStaticMethod:
    def __init__(self, fn, cls=None):
        self.cls = cls
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __get__(self, obj, cls):
        return PickleableStaticMethod(self.fn, cls)

    def __getstate__(self):
        return self.cls, self.fn.__name__

    def __setstate__(self, state):
        self.cls, name = state
        self.fn = getattr(self.cls, name).fn


########################################################################################################################
########################################################################################################################
class PostInitCaller(ABCMeta):
    def __call__(cls, *args, **kwargs):
        """Called when you call MyNewClass() """
        obj = type.__call__(cls, *args, **kwargs)
        obj.post_init()
        return obj


########################################################################################################################
########################################################################################################################
class MathDictionary(FFDictionary):
    def __init__(self):
        super(MathDictionary, self).__init__('MathFunctions', ['x'])

    @staticmethod
    def product(x):
        m = 1
        for _ in x:
            m = m * _
        return m

    multiplication = product

    @staticmethod
    def sum(x):
        s = 0
        for _ in x:
            s = s + _
        return s

    @staticmethod
    def substract(x):
        r = x[0]
        for _ in x[1:]:
            r = r - _
        return r

    @staticmethod
    def divide(x):
        r = x[0]
        for _ in x[1:]:
            r = r / _
        return r


MathFunctions = MathDictionary().factory()