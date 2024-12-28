from ..j_utils.json_template import JSONAttr, TypesUnion
from ..j_utils.string import auto_increment
from .vega_common import SpecObject, Ref
from .vega_interactive import Trigger


class Transform(SpecObject):
    type = JSONAttr.String(read_only=True)
    options = JSONAttr.GenericProperties()


class DataFormat(SpecObject):
    type = JSONAttr.String('json')
    parse = JSONAttr(types=(str, dict), default='auto')
    options = JSONAttr.GenericProperties(names=('property', 'copy',
                                                'header', 'delimiter',
                                                'features', 'mesh', 'filter', 'copy'))


class Data(SpecObject):
    __auto_apply__ = ('transform',)
    name = JSONAttr.String()
    format = JSONAttr(DataFormat, default=DataFormat())
    transform = JSONAttr.List(Transform)
    on = JSONAttr.List(Trigger)

    @property
    def Values(self):
        return ValuesData

    @property
    def Source(self):
        return SourceData

    @property
    def URL(self):
        return URLData

    def transformed(self, transforms, name=None):
        if name is None:
            name = self.name+'Transformed'
        while name in self.spec.data:
            name = auto_increment(name)
        transfomed_data = SourceData(source=self, name=name)
        transfomed_data.transform = transforms
        self.spec.data.append(transfomed_data)
        return transfomed_data


class ValuesData(Data):
    values = JSONAttr()

    @values.changed
    def values_changed(self, handler, value):
        pass


class SourceData(Data):
    __auto_apply__ = Data.__auto_apply__ + ('source',)
    source = JSONAttr.List(Data, key='name')


    @source.to_json
    def source_to_json(self):
        source = self.source
        if len(source) == 1:
            return source[0].name
        else:
            return [_.name for _ in source]


class URLData(Data):
    __auto_apply__ = Data.__auto_apply__ + ('source',)
    url = JSONAttr.String()


class DataField(SpecObject):
    dataset = Ref(Data)
    field = JSONAttr.String(optional=True)

    def to_json(self, to_str=True, context=None):
        dataset = self.cls_attributes(Ref)['dataset'].attr_to_json
        if self.field:
            return "%s['%s']" % (dataset, self.field)
        else:
            return dataset

    @classmethod
    def from_json(cls, json):
        if isinstance(json, str):
            if '.' in json:
                dataset, field = json.split('.', 1)
                return DataField(dataset=dataset, field=field)
            elif '[' in json and json.endswith(']'):
                dataset, field = json[:-1].split('[')
                field = field.strip()
                if (field.startswith("'") and field.endswith("'")) or (field.startswith('"') and field.endswith('"')):
                    field = field[1:-1]
                return DataField(dataset=dataset, field=field)
            return DataField(dataset=json)
        return super(DataField).from_json(cls, json)


class FieldAs(SpecObject):
    field = JSONAttr(DataField)
    as_ = JSONAttr.String()


def _parse_field(dataset, field=None, field_as=None):
    f = DataField(dataset=dataset, field=field)
    return FieldAs(field=f, as_=field_as) if field_as else f


Field = TypesUnion(DataField, FieldAs, default_init=_parse_field)


class AgregateTransfomParameters(SpecObject):
    groupby = JSONAttr.List(Field, optional=True)
    fields = JSONAttr.List(Field, optional=True)
    ops = JSONAttr.String(list=True, optional=True)
    as_ = JSONAttr.String(list=True, optional=True)
    cross = JSONAttr.Bool(optional=True)
    drop = JSONAttr.Bool(optional=True)
    key = JSONAttr(Field, optional=True)