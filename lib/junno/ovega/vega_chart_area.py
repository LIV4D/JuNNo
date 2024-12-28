from ..j_utils.json_template import JSONAttr, JSON_TYPES
from ..j_utils.string import auto_increment
from .vega_common import SpecObject, Ref
from .vega_values import Scale


########################################################################################################################
class Axes(SpecObject):
    scale = Ref(Scale)
    orient = JSONAttr.OneOf('left', 'right', 'top', 'bottom')
    encode = JSONAttr.Dict()
    opt = JSONAttr.GenericProperties()


########################################################################################################################
class Legend(SpecObject):
    type = JSONAttr.OneOf('symbol', 'gradient')
    direction = JSONAttr.OneOf('vertical', 'horizontal')
    orientation = JSONAttr.OneOf('left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'none')
    fill = Ref(Scale, optional=True)
    opacity = Ref(Scale, optional=True)
    shape = Ref(Scale, optional=True)
    size = Ref(Scale, optional=True)
    stroke = Ref(Scale, optional=True)
    strokeDash = Ref(Scale, optional=True)
    strokeWidth = Ref(Scale, optional=True)
    encode = JSONAttr.Dict()
    opt = JSONAttr.GenericProperties()


########################################################################################################################
class Title(SpecObject):
    text = JSONAttr.String()
    orient = JSONAttr.OneOf('top', 'bottom', 'left', 'right')
    encode = JSONAttr.Dict()
    opt = JSONAttr.GenericProperties()

