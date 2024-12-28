from ..j_utils.json_template import JSONClass, JSONAttr, JSONAttribute


class SpecObject(JSONClass):
    __auto_apply__ = 'all'
    __ABSTRACT__ = "SpecObject"

    def __init__(self,  **kwargs):
        super(SpecObject, self).__init__(**kwargs)

    def _apply_change(self):
        if self.spec:
            self.spec.apply_change()

    def __attr_changed(self, attr_name, previous):
        if self.__auto_apply__ == 'all' or attr_name in self.__auto_apply__:
            self._apply_change()

    @property
    def spec(self):
        from .vega_graph import VegaSpec
        spec = self.root_json_class
        return spec if isinstance(spec, VegaSpec) else None


########################################################################################################################
class Ref(JSONAttribute):
    def __init__(self, type, key='name', other_types=(), optional=False):
        types = str
        self._other_types = other_types
        if other_types:
            types = (str,) + tuple(t for t in other_types if t is not str)
        super(Ref, self).__init__(types, optional=optional)
        self._ref_type = type
        self._ref_key = key
    
    def get_attr(self, handler, default=None):
        if not isinstance(self._ref_key, str):
            return self._ref_key
        if isinstance(handler, SpecObject):
            from .vega_graph import Signal, Data, Scale, Axes, Legend, Mark
            if type == Signal:
                return handler.spec.signals.get(self._ref_key, default)
            elif type == Data:
                return handler.spec.data.get(self._ref_key, default)
            elif type == Scale:
                return handler.spec.scales.get(self._ref_key, default)
            elif type == Axes:
                return handler.spec.axes.get(self._ref_key, default)
            elif type == Legend:
                return handler.spec.legends.get(self._ref_key, default)
            elif type == Mark:
                return handler.spec.marks.get(self._ref_key, default)
        return super(Ref, self).get_attr(handler=handler, default=default)

    def set_attr(self, handler, value):
        if isinstance(value, self._ref_type):
            value = getattr(value, self._ref_key)
        if isinstance(value, str):
            return super(Ref, self).set_attr(handler, value)
        #elif self._field_value:
        #    from .vega_interactive import Signal
        #    if isinstance(value, Signal):
        #        value = {"signal": value.name}
        #    return super(Ref, self).set_attr(handler, value)
        raise AttributeError('Invalid value or key value type. Expected %s or str, received %s.' %
                             (self._ref_type.__name__, type(value).__name__))
