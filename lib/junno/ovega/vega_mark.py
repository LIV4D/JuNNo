from ..j_utils.json_template import JSONAttr, TypesUnion
from ..j_utils.string import auto_increment
from .vega_common import SpecObject, Ref
from .vega_interactive import Trigger
from .vega_data import Transform, AgregateTransfomParameters, Data, Field
from .vega_chart_area import Scale


class Encode(SpecObject):
    pass


########################################################################################################################
class Facet(SpecObject):
    name = JSONAttr.String()
    data = Ref(Data)
    field = JSONAttr(Field, optional=True)
    groupby = JSONAttr.List(Field, optional=True)
    aggregate = JSONAttr(AgregateTransfomParameters, optional=True)


class From(SpecObject):
    data = Ref(Data)
    facet = JSONAttr(Facet, optional=True)


########################################################################################################################
class Mark(SpecObject):
    __template__ = """
        "name": "{{name}}",
        "type": "{{type}}",
        "description": "{{description}}",
        "encode": "{{encode}}",        
        "from": "{{from}}",
        "interactive": "{{interactive}}",
        "key": "{{key}}",
        "on": "{{on}}",
        "sort": "{{sort}}",
        "transform": "{{transform}}",
        "role"`"{{role}}",
        "style": "{{style}}",
        "zindex": ""{{zindex}}"
        
    }"""
    name = JSONAttr.String()
    description = JSONAttr.String(optional=True)
    encode = JSONAttr.List(Encode)
    from_ = JSONAttr.List(From)
    interactive = JSONAttr.Bool(default=False)
    key = JSONAttr.String(optional=True)
    on = JSONAttr.List(Trigger)

    transform = JSONAttr.List(Transform)
    role = JSONAttr.String(optional=True)
    style = JSONAttr.String(list=True, optional=True)
    zindex = JSONAttr.Float(optional=True)
