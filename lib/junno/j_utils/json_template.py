from .class_attribute import ClassAttrHandler, ClsAttribute
from .collections import is_dict, OrderedDict
from .j_log import log


class JSONAttribute(ClsAttribute):
    def __init__(self, types, default=None, islist=False):
        super(JSONAttribute, self).__init__()
        self.default = default
        self.islist = islist
        self.types = tuple(types)

    def new_attr(self, handler):
        return self.default

    def set_attr(self, handler, value):
        if isinstance(value, self.types):
            raise ValueError('%s type must be one of %s.\n'
                             'but type(%s) == %s' % (self.name, list(_.__name__ for _ in self.types),
                                                     repr(value), type(value)))
        return super(JSONAttribute, self).set_attr(handler=handler, value=value)


class JSONClass(ClassAttrHandler):
    __template__ = None
    __dict_template__ = None
    __classtype__ = None

    @classmethod
    def classtype(cls):
        if cls.__classtype__ is None:
            return cls.__name__
        return cls.__classtype__

    @classmethod
    def template(cls):
        if cls.__template__ is None:
            template = "{\n"
            for attr in cls.cls_attributes(JSONAttribute):
                template += '\t"' + attr.name + '": ' + attr.name + '\n'
            cls.__template__ = template + '}'
        return cls.__template__

    def __init__(self, **kwargs):
        super(JSONClass, self).__init__(**kwargs)

    def get(self, item, default=None):
        getattribute = super(ClassAttrHandler, self).__getattribute__
        attributes = getattribute('__class__').cls_attributes(JSONAttribute)

        attr = attributes.get(item, None)
        if attr is not None:
            return attributes[item].get_attr(self)
        return default

    def set(self, key, value):
        getattribute = super(ClassAttrHandler, self).__getattribute__
        attributes = getattribute('__class__').cls_attributes(JSONAttribute)
        attr = attributes.get(key, None)
        if attr is not None:
            if attr.read_only:
                raise AttributeError('%s is a read-only attribute' % attr.name)
            attr.set_attr(handler=self, value=value)
        else:
            raise AttributeError('%s unknown.' % key)

    def update(self, model):
        if isinstance(model, str):
            import json
            model = json.loads(model)
        if is_dict(model):
            getattribute = super(ClassAttrHandler, self).__getattribute__
            attributes = getattribute('__class__').cls_attributes(JSONAttribute)
            for k, v in model.items():
                attr = attributes.get(k, None)
                if attr is not None:
                    attr.set_attr(handler=self, value=v)
        else:
            raise ValueError('Invalid model: %s.' % repr(model))

    @classmethod
    def from_dict(cls, d):
        keys = set(d.keys())
        attrs = {_.name for _ in cls.cls_attributes(JSONAttribute)}
        unused = keys.difference(attrs)
        if unused:
            log.warn('Warning when initializing %s from dict:\n'
                     '%s keys will be ignored.' % (cls.__name__, unused))

        return cls.__new__(**{k: d[k] for k in keys.intersection(attrs)})

    def to_dict(self):
        return self.attributes(JSONAttribute)

    @staticmethod
    def from_json(json):
        if isinstance(json, str):
            from json import loads
            json = loads(json)
        if not is_dict(json):
            raise ValueError('Invalid json type: %s.' % type(json))



    def to_json(self):
        pass


