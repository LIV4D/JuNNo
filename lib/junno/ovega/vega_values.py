from .vega_common import Ref, SpecObject, JSONAttr


########################################################################################################################
#   ---  SCALES  ---
########################################################################################################################
Domain = (list, dict)
Range = (list, dict, str)


class Scale(SpecObject):
    name = JSONAttr.String()
    type = JSONAttr.OneOf('linear', 'log', 'pow', 'sqrt', 'symlog', 'time', 'utc', 'sequential', 'ordinal',
                          'band', 'point', 'quantile', 'quantize', 'threshold', 'bin-ordinal')
    domain = JSONAttr(Domain)
    range = JSONAttr(Range)
    interpolate = JSONAttr.String()
    reverse = JSONAttr.Bool(default=False, optional=True)
    round = JSONAttr.Bool(default=False, optional=True)
    opt = JSONAttr.GenericProperties()


########################################################################################################################
class Value(SpecObject):
    __ABSTRACT__ = "Value"

    scale = Ref(Scale, optional=True)
    band = JSONAttr.Float(optional=True)

    exponent = JSONAttr(types=(float, JSONAttr.SAME_CLASS), optional=True)
    mult = JSONAttr(types=(float, JSONAttr.SAME_CLASS), optional=True)
    offset = JSONAttr(types=(float, JSONAttr.SAME_CLASS), optional=True)
    round = JSONAttr.Bool(default=False, optional=True)


class ConstantValue(Value):
    value = JSONAttr.Any()

    def __init__(self, value, **kwargs):
        super(ConstantValue, self).__init__(value=value, **kwargs)


class SignalValue(Value):
    signal = JSONAttr.String()


class FieldValueValue(Value):
    field = JSONAttr.Any()


class FieldValue(Ref):
    def __init__(self, type, other_types=(), optional=False):
        super(FieldValue, self).__init__()
