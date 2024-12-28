import inspect
from abc import ABCMeta
from collections import OrderedDict
import weakref
from inspect import signature
from ..j_utils.function import match_params


_cls_attr_global_id = 0


def sort_by_global_id(cls_attr_list):
    result = []
    cls_attr_list = list(cls_attr_list)
    while cls_attr_list:
        # Find min global_id
        min = float('inf')
        min_proto = None
        for p in cls_attr_list:
            if p._global_id < min:
                min = p._global_id
                min_proto = p

        cls_attr_list.remove(min_proto)
        result.append(min_proto)
    return result


class ClsAttribute:
    def __init__(self, read_only=False):
        self._name = None
        self._handler_cls = None
        self.read_only = read_only

        global _cls_attr_global_id
        self._global_id = _cls_attr_global_id
        _cls_attr_global_id += 1

    def _name_changed(self, n):
        pass

    def __set_name__(self, owner, name):
        if self.name is None:
            self.name = name
        if self._handler_cls is None:
            self._handler_cls = weakref.ref(owner)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, n):
        self._name = n
        self._name_changed(n)

    @property
    def handler_cls(self):
        return self._handler_cls() if self._handler_cls else None

    @property
    def full_name(self):
        if self.handler_cls is None:
            if self.name is None:
                return None
            return self.name

        return type(self.handler_cls).__name__ + '.' + self.name

    def __str__(self):
        return self.full_name

    # --- INIT ---
    def init_attr(self, handler, new_attr):
        handler.__dict__[self.name] = new_attr

    def new_attr(self, handler):
        return None

    # --- GETTER / SETTER ---
    def set(self, handler, value):
        if handler is None:
            raise AttributeError('Read only')
        else:
            value = self.check_attr(handler=handler, value=value)
            v = self.get_attr(handler=handler)
            if v != value:
                self.set_attr(handler=handler, value=value)
                handler._attr_changed(attr_name=self.name, previous=v)
                self.attr_changed(handler=handler, value=value)

    def __set__(self, handler, value):
        return self.set(handler, value)

    def check_attr(self, handler, value):
        return value

    def set_attr(self, handler, value):
        handler.__dict__[self.name] = value

    def attr_changed(self, handler, value):
        pass

    def get(self, handler):
        if handler is None:
            return self
        else:
            return self.get_attr(handler=handler)

    def __get__(self, instance, owner_type):
        return self.get(handler=instance)

    def get_attr(self, handler, default=None):
        return handler.__dict__.get(self.name, default)

    # --- DECORATORS ---
    def setter(self, f):
        def set_attr(self, handler, value):
            f(handler, value)
        self.set_attr = set_attr

    def check(self, f):
        legacy_check = self.check_attr

        def check_attr(self, handler, value):
            return match_params(f, args=(handler, value), legacy_check=legacy_check)
        self.check_attr = check_attr

    def changed(self, f):
        def attr_changed(self, handler, value):
            f(handler, value)
        self.attr_changed = attr_changed

    def getter(self, f):
        param_count = len(signature(f).parameters)
        if param_count==2:
            def get_attr(self, handler, default=None):
                value = handler.__dict__.get(self.name, default)
                return f(handler, value)
        else:
            def get_attr(self, handler, default=None):
                return f(handler, self.__get_attr(handler, default=default))
            self.__get_attr = self.get_attr
        self.get_attr = get_attr


class MetaClassAttrHandler(ABCMeta):
    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)

        l = []
        for attr in inspect.getmembers(cls, lambda e: not inspect.isfunction(e)):
            n, attr = attr
            if isinstance(attr, ClsAttribute):
                attr.name = n
                attr._handler_cls = cls
                l.append(attr)
        cls._attributes_list = sort_by_global_id(l)

        if hasattr(cls, "__register_subclass__"):
            cls.__register_subclass__(name, bases)

    def __repr__(cls):
        s = cls.__name__
        for hp in cls._attributes_list:
            s += "\n\t"+str(hp)
        return s

    def __str__(cls):
        return cls.__name__


class ClassAttrHandler(metaclass=MetaClassAttrHandler):
    def __new__(cls, *args, **kwargs):
        obj = super(ClassAttrHandler, cls).__new__(cls)
        new_args = args
        new_kwargs = kwargs

        obj._change_listeners = {}
        obj.prepare_populate_attr()
        obj.populate_attr(new_args=new_args, new_kwargs=new_kwargs)
        return obj

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def cls_attributes(cls, types=None):
        if types is not None:
            r = OrderedDict()
            for attr in cls._attributes_list:
                if (isinstance(types, str) and type(attr).__name__ == types) or isinstance(attr, types):
                    r[attr.name] = attr
            return r
        else:
            r = OrderedDict()
            for attr in cls._attributes_list:
                r[attr.name] = attr
            return r

    def attributes(self, types=None):
        r = OrderedDict()
        for attr in self.cls_attributes(types=types).values():
            r[attr.name] = attr.get_attr(self)
        return r

    def prepare_populate_attr(self):
        pass

    def populate_attr(self, new_args, new_kwargs):
        alias = [_ for _ in new_kwargs if len(_) > 1 and _[-1] == '_' and _[-2] != '_']
        for a in alias:
            new_kwargs[a[:-1]] = new_kwargs[a]
        for name, attr in self.__class__.cls_attributes().items():
            try:
                new_attr = new_kwargs[name]
            except KeyError:
                new_attr = attr.new_attr(self)
            attr.init_attr(self, new_attr)

    def __getattribute__(self, item):
        getattribute = super(ClassAttrHandler, self).__getattribute__

        attr = getattribute('__class__').cls_attributes().get(item, None)
        if attr is not None:
            return attr.get(self)

        return getattribute(item)

    def __setattr__(self, key, value):
        attr = self.__class__.cls_attributes().get(key, None)
        if attr is not None:
            attr.set(handler=self, value=value)
        else:
            super(ClassAttrHandler, self).__setattr__(key, value)

    def on_changed(self, key, cb, call_now=False):
        if isinstance(key, tuple):
            for k in key:
                self.on_changed(k, cb)
        else:
            if key not in self.cls_attributes():
                raise AttributeError('No attributes is named %s.' % key)

            listeners = self._change_listeners.get(key, None)
            if listeners is None:
                listeners = []
                self._change_listeners[key] = listeners
            listeners.append(cb)

        if call_now:
            cb()

    def __attr_changed(self, attr_name, previous):
        callbacks = self._change_listeners.get(attr_name, [])
        for cb in callbacks:
            cb()

    def _attr_changed(self, attr_name, previous):
        # self.__attr_changed(attr_name)
        called = []     # [self.__class__.__attr_changed]
        for parent_class in inspect.getmro(self.__class__):
            if parent_class.__attr_changed not in called:
                parent_class.__attr_changed(self, attr_name, previous)
                called.append(parent_class.__attr_changed)
            if parent_class is ClassAttrHandler:
                break
