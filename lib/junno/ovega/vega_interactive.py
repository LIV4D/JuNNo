from ..j_utils.json_template import JSONClass, JSONAttr
from .vega_common import SpecObject


########################################################################################################################
Expression = str


########################################################################################################################
class Trigger(SpecObject):
    trigger = JSONAttr.String()
    insert = JSONAttr.String(optional=True)
    remove = JSONAttr((bool, str), optional=True)
    toggle = JSONAttr.String(optional=True)
    modify = JSONAttr.String(optional=True)
    values = JSONAttr.String(optional=True)


########################################################################################################################
class AbstractEventStream(SpecObject):
    between = JSONAttr.List(('same_class', str))
    filter = JSONAttr.List(str)

    consume = JSONAttr.Bool(optional=True)
    debounce = JSONAttr.Float(optional=True)
    markname = JSONAttr.String(optional=True)   # TODO: MarkRef
    marktype = JSONAttr.String(optional=True)
    throttle = JSONAttr.Float(optional=True)


class EventStreamObject(AbstractEventStream):
    source = JSONAttr.String(default='view')
    type = JSONAttr.String()


class DerivedEventStream(AbstractEventStream):
    stream = JSONAttr((AbstractEventStream, str))


class MergedEventStream(AbstractEventStream):
    merge = JSONAttr.List((AbstractEventStream, str))


EventStream = (EventStreamObject, DerivedEventStream, MergedEventStream, str)


########################################################################################################################
class EventHandler(SpecObject):
    events = JSONAttr(EventStream)
    update = JSONAttr(Expression, optional=True)
    encode = JSONAttr.String(optional=True)
    force = JSONAttr.Bool(optional=True)


########################################################################################################################
class Signal(SpecObject):
    name = JSONAttr.String()
    description = JSONAttr.String()
    on = JSONAttr.List(EventHandler)
    init = JSONAttr(Expression, optional=True)
    update = JSONAttr(Expression, optional=True)
    react = JSONAttr.Bool(default=True)
    value = JSONAttr(optional=True)


########################################################################################################################
class NestedSignal(SpecObject):
    name = JSONAttr.String()
    push = JSONAttr.String()
    description = JSONAttr.String()
    on = JSONAttr.List(EventHandler)
    
    
########################################################################################################################
class SignalRef(JSONAttr):
    def __init__(self):
        super(SignalRef, self).__init__((Signal, NestedSignal, str))

    def attr_to_json(self, handler, value, context=None):
        if isinstance(value, Signal):
            return value.name
        elif isinstance(value, NestedSignal):
            return value.to_json(False, context=context)
        return value
