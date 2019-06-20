import inspect
from abc import ABCMeta
from collections import OrderedDict

from junno.j_utils.function import match_params


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
        self.handler_cls = None
        self.read_only = read_only

        global _cls_attr_global_id
        self._global_id = _cls_attr_global_id
        _cls_attr_global_id += 1

    def _name_changed(self, n):
        pass

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, n):
        self._name = n
        self._name_changed(n)

    @property
    def full_name(self):
        if self.handler_cls is None:
            if self.name is None:
                return None
            return self.name

        return self.handler_cls.__name__ + '.' + self.name

    def set_attr(self, handler, value):
        v = self.get_attr(handler)
        if v != value:
            handler.__dict__[self.name] = value
            self.notify_change(handler=handler)

    def get_attr(self, handler, default=None):
        return handler.__dict__.get(self.name, default)

    def init_attr(self, handler, new_attr):
        handler.__dict__[self.name] = new_attr

    def new_attr(self, handler):
        return None

    def __str__(self):
        return self.name

    def notify_change(self, handler):
        handler._attr_changed(attr_name=self.name)


class MetaClassAttrHandler(ABCMeta):
    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)

        l = []
        for attr in inspect.getmembers(cls, lambda e: not inspect.isfunction(e)):
            n, attr = attr
            if isinstance(attr, ClsAttribute):
                attr.name = n
                attr.handler_cls = cls
                l.append(attr)
        cls._attributes_list = sort_by_global_id(l)

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
        match_params(obj.populate_attr, new_args=new_args, new_kwargs=new_kwargs)
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
        for name, attr in self.__class__.cls_attributes().items():
            try:
                new_attr = new_kwargs[name]
            except KeyError:
                new_attr = attr.new_attr(self)
            attr.init_attr(self, new_attr)

    def __getattribute__(self, item):
        getattribute = super(ClassAttrHandler, self).__getattribute__
        attributes = getattribute('__class__').cls_attributes()

        if item in attributes:
            return attributes[item].get_attr(self)

        return getattribute(item)

    def __setattr__(self, key, value):
        attributes = self.__class__.cls_attributes()
        attr = attributes.get(key, None)
        if attr is not None:
            if attr.read_only:
                raise AttributeError('%s is a read-only attribute' % attr.name)
            attr.set_attr(handler=self, value=value)
        else:
            super(ClassAttrHandler, self).__setattr__(key, value)

    def on_changed(self, key, cb, call_now=False):
        if isinstance(key, tuple):
            for k in key:
                self.on_changed(k, cb)
        else:
            if not key in self.cls_attributes():
                raise AttributeError('No attributes is named %s.' % key)

            listeners = self._change_listeners.get(key, None)
            if listeners is None:
                listeners = []
                self._change_listeners[key] = listeners
            listeners.append(cb)

        if call_now:
            cb()

    def _attr_changed(self, attr_name):
        callbacks = self._change_listeners.get(attr_name, [])
        for cb in callbacks:
            cb()
