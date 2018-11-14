import numpy as np
from collections import OrderedDict
import os.path

from .function import match_params


def cast_to_list(l):
    if isinstance(l, str):
        return [l]
    try:
        return list(l)
    except TypeError:
        return [l]


def is_dict(d):
    return isinstance(d, (dict, OrderedDict))


def recursive_dict_update(destination, origin):
    for n, v in origin.items():
        dest_v = destination.get(n, None)
        if is_dict(v) and is_dict(dest_v):
            recursive_dict_update(destination[n], v)
        else:
            destination[n] = v

def recursive_dict(dictionnary, function):
    r = {}
    for n, v in dictionnary.items():
        if is_dict(v):
            v = recursive_dict(v, function=function)
        else:
            v = function(n, v)
        r[n] = v
    return r

def if_none(v, default=None):
    if default is None:
        return v is None
    return default if v is None else v


########################################################################################################################
class PropertyAccessor:
    def __init__(self, list):
        self._list = list

    def __getattr__(self, item):
        c = None
        for node in self._list:
            if hasattr(node, item):
                p = getattr(node, item, [])

                if c is None:
                    if isinstance(p, DictList) or isinstance(p, list):
                        c = p.copy()
                        continue
                    elif hasattr(type(p), 'List'):
                        c = type(p).List(p)
                    else:
                        c = []

                if isinstance(p, list) or isinstance(p, DictList):
                    c += p
                else:
                    c.append(p)
        return c

    def __setattr__(self, key, value):
        if key == '_list':
            super(PropertyAccessor, self).__setattr__('_list', value)
        for node in self._list:
            if hasattr(node, key):
                setattr(node, key, value)


########################################################################################################################
class DictList:
    """
    This list has both behaviour of a list and a dictionnary: elements can be retrieved by index (the order in which
    the points was added in the list is preserved) or by name. item2key(e) retreive the primary key of any element for
    the dictionary, thus a point can only appear once in such List. Also, the list handle boolean set operations such as
    union, intersection, difference...

    When subclassing, item2key may be overloaded to provide specific key to an item.
    """

    def __init__(self, *args, dtype=None, on_changed=None, exact_key=False):
        """

        :param args: Elements and/or iterables of elements which will initialize the list
        :param dtype: If not None, only objects of type **dtype** can be added to this list
        :param on_changed: Change callbacks
        :param exact_key: If false, the keys radical is stored, when a `key` is prompted: first `key` will be searched
        in the keys list, then `radical+key` will be searched.
        :Example:
        """
        self._dtype = dtype
        self._list = []
        self._dict = {}
        self.exact_key = exact_key
        self._radical = ''

        for arg in args:
            if arg is not None:
                self.append(arg, emit_change=False)

        self.on_changed = []
        if on_changed is not None:
            if not isinstance(on_changed, list):
                on_changed = [on_changed]
            self.on_changed = on_changed

    @property
    def dtype(self):
        return self._dtype

    def check_item(self, item, raise_exception=True):
        if self._dtype is None:
            return True
        if isinstance(item, self._dtype):
            return True
        if raise_exception:
            raise TypeError('type(%s) is %s but should be %s' % (repr(item), type(item), self._dtype))
        return False

    def check_key(self, key, item, raise_exception=True):
        item_key = self.item2key(item)

        if isinstance(key, int):
            key = self.item2key(self._list[key])
        elif not isinstance(key, str):
            key = self.item2key(key)

        if key == item_key or (not self.exact_key and self._radical+key == item_key):
            return item_key
        if not self.exact_key and len(self)==0 and item_key.endswith(key):
            self._radical = item_key[:-len(key)]
            return item_key
        if raise_exception:
            raise ValueError('%s is not a valid key for %s' % (key, item_key))
        return None


    @classmethod
    def item2key(self, item):
        return str(item)

    def _update_dict(self):
        self._dict.clear()
        try:
            for id, p in enumerate(self._list):
                self._dict[self.item2key(p)] = id
        except:
            raise ValueError(repr(['%s: %s' % (_, self.check_item(_, raise_exception=False)) for _ in self._list]))
        if not self.exact_key and self._list:
            radical = os.path.commonprefix(list(self._dict.keys()))
            if '_' in radical:
                radical = radical[:radical.rindex('_')+1]
            self._radical = radical
        else:
            self._radical = ''

    def copy(self):
        l = self.DictList(self._list)
        l._radical = self._radical
        return l

    def __getitem__(self, item):
        """
        :rtype: self.dtype
        """
        if isinstance(item, str):
            item = self.index_by_key(item)
        if isinstance(item, int) or isinstance(item, np.int64):
            return self._list[item]
        elif isinstance(item, slice):
            return self.DictList(self._list[item])
        else:
            raise TypeError('Invalid index: %s (type: %s)' % (repr(item), type(item)))

    def get(self, item, default=None):
        try:
            return self[item]
        except LookupError:
            return default

    def __setitem__(self, key, value):
        self.check_item(value)
        if isinstance(key, str):
            self.check_key(key, value)
            self.update(value)
            return

        if isinstance(key, int):
            if 0 <= key < len(self):
                self.check_key(key, value)
                self.update(value)
            else:
                self.append(value)
        else:
            raise TypeError("%s is not a valid index" % key)
        self._update_dict()

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return self._list.__iter__()

    def __contains__(self, item):
        if isinstance(item, str):
            try:
                self.index_by_key(item)
            except LookupError:
                return False
            return True
        elif issubclass(type(item), self._dtype):
            return self.item2key(item) in self._dict
        else:
            try:
                for p in item:
                    if p not in self:
                        return False
                return True
            except TypeError:
                pass
        return False

    def __repr__(self):
        return str([self.item2key(_) for _ in self._list])

    def __bool__(self):
        return len(self)>0

    def __eq__(self, other):
        if isinstance(other, DictList) and self._dtype == other._dtype:
            return self.keys_set() == other.keys_set()
        return False

    @property
    def keys_radical(self):
        return self._radical

    def index_by_key(self, key):
        id = self._dict.get(key, -1)
        if id > -1:
            return id

        if not self.exact_key and self._radical != '':
                id = self._dict.get(self._radical + key, -1)
                if id > -1:
                    return id

        # for n in self._dict:
        #     if n.endswith(key) and n[-len(key) - 1] == '_':
        #         return self._dict[n]
        raise KeyError("The list doesn't contain any elements named: %s" % key)

    def index(self, value):
        return self.index_by_key(self.item2key(value))

    def _changed(self, last):
        for cb in self.on_changed:
            match_params(cb, last=last)

    def insert(self, index, other, emit_change=True):
        assert isinstance(index, int)
        if index < 0:
            index += len(self)
        if not 0 <= index <= len(self):
            raise IndexError('%i is an invalid index for List of length %i'
                             % (index, len(self)))

        other = cast_to_list(other)

        if other:
            if emit_change:
                last = self.copy()
            for p in other:
                if p is None:
                    continue
                self.check_item(p)
                try:
                    previous_index = self.index(p)
                    self._list.pop(previous_index)
                    if previous_index < index:
                        index -= 1
                except LookupError:
                    pass

                self._list.insert(index, p)
                self._update_dict()
                index += 1

            if emit_change:
                self._changed(last)

    def append(self, other, emit_change=True):
        """
        If the List is viewed as a dictionary, this is equivalent to update(other).
        With a list point of view, the only difference with the classic append, is that a point can only appeared
        once in a List. Thus if other contains an item already added in this list, it will be ignored.
        :param other: a item or an iterable of items to add in the list
        """
        self.insert(index=len(self), other=other, emit_change=emit_change)

    def __add__(self, other):
        l = self.copy()
        l.append(other)
        return l

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self.append(other)
        return self

    def __or__(self, other):
        return self.__add__(other)

    def __ror__(self, other):
        return self.__add__(other)

    def __ior__(self, other):
        return self.__iadd__(other)

    def update(self, point):
        try:
            last = self.copy()
            i = self.index_by_key(self.item2key(point))
            self[i] = point
            self._changed(last)
        except LookupError:
            self.append(point)

    def remove(self, other):
        other = cast_to_list(other)

        if not other:
            return True

        last = self.copy()
        id_to_remove = set()

        for p in other:
            if self.check_item(p, raise_exception=False):
                p = self.item2key(p)
            elif not isinstance(p, str):
                continue
            id = self._dict.pop(p, None)
            if id is not None:
                id_to_remove.add(id)

        if not id_to_remove:
            return False

        removed = False
        try:
            id_to_remove.remove(-1)
        except KeyError:
            removed = True

        id_to_remove = list(id_to_remove)
        id_to_remove.sort(reverse=True)

        for id in id_to_remove:
            del self._list[id]

        self._update_dict()
        self._changed(last)
        return removed

    def __sub__(self, other):
        l = self.copy()
        l.remove(other)
        return l

    def __isub__(self, other):
        self.remove(other)
        return self

    def __and__(self, other):
        r = self.copy()
        r.intersect(other)
        return r

    def __rand__(self, other):
        return self & other

    def __iand__(self, other):
        self.intersect(other)

    def intersect(self, other):
        other = self.DictList(other)
        inter_keys = self._dict.keys() & other._dict.keys()
        diff = self._dict.keys() - inter_keys
        self.remove(diff)

    def clear(self):
        last = self.copy()
        self._list.clear()
        self._dict.clear()
        self._changed(last)

    def keys_set(self):
        """
        Return a set containing all the full names of this list's points.
        """
        return {self.item2key(_) for _ in self}

    def keys(self):
        return [self.item2key(_) for _ in self._list]

    def list(self):
        """
        Convert this List to a standard python list
        :rtype: list
        """
        return self._list.copy()

    def items(self):
        for _ in self._list:
            yield self.item2key(_), _

    def dict(self):
        """
        Convert this List to a standard python dictionary
        :rtype: dict
        """
        d = OrderedDict()
        for n, e in self.items():
            d[n] =e
        return d

    @property
    def all(self):
        return PropertyAccessor(self)

    def DictList(self, *args):
        if len(args) == 1 and isinstance(args[0], DictList) and args[0]._dtype == self._dtype:
            return args[0]
        l = self.__class__(*args)
        l._dtype = self.dtype
        l.exact_key = self.exact_key
        return l


########################################################################################################################
class Tree:
    def __init__(self):
        self._root = Tree.Node('', None, None)
        self._root._tree = self

    @staticmethod
    def create_from_dict(dictionary):
        tree = Tree()

        def recursive_create_from_dict(node, data):
            for k, v in data.items():
                child_node = node.add_node(name=k)
                if is_dict(v):
                    recursive_create_from_dict(node=child_node, data=v)
                else:
                    child_node.data = v
        recursive_create_from_dict(tree.root, dictionary)
        return tree

    @property
    def root(self):
        return self._root

    @property
    def all(self):
        return Tree.NodeChildView(self._root)

    def __getitem__(self, item):
        return self._root[item]

    def __setitem__(self, key, value):
        self._root[key] = value

    def walk(self):
        for n in self.root.walk():
            yield n

    class Node:
        def __init__(self, name, data, parent):
            self.name = name
            self.data = data
            self._parent = parent
            self._children = []
            self._tree = None
            self._index = -1

        @property
        def children(self):
            return self._children

        @property
        def parent(self):
            return self._parent

        @property
        def tree(self):
            if self._tree:
                return self._tree
            return self.parent.tree

        def add_node(self,  name, data=None):
            n = Tree.Node(name, data, self)
            self._children.append(n)
            return n

        def clear_children(self):
            self._children = []

        def node_by_name(self, name, recursive=True, first=False):
            r = []
            for n in self._children:
                if n.name == name:
                    if first:
                        return n
                    r += n

            if recursive:
                for n in self._children:
                    if isinstance(recursive, int) and not isinstance(recursive, bool):
                        recursive -= 1
                    r += n.node_by_name(name, recursive, first)
                    if r and first:
                        return r[0]

            if first:
                return None
            return r

        def search(self, data, recursive=True):
            for n in self._children:
                if n.data == data:
                    return n
                if recursive:
                    if isinstance(recursive, int) and not isinstance(recursive, bool):
                        recursive -= 1
                    r = n.search(data, recursive)
                    if r is not None:
                        return r
            return None

        def __len__(self):
            return len(self._children)

        def __getitem__(self, item):
            if isinstance(item, str):
                n = self.node_by_name(item, recursive=False, first=True)
                if n is None:
                    raise IndexError("Wrong index '%s'" % item)
                return n
            elif isinstance(item, int):
                if not 0 < item < len(self._children):
                    raise IndexError("Wrong index %i (length: %i)" % (item, len(self._children)))
                return self._children[item]
            elif isinstance(item, tuple):
                child_id = item[0]
                child_tuple = item[1:]
                return self[child_id][child_tuple]
            else:
                raise NotImplementedError

        def __setitem__(self, key, value):
            if isinstance(key, str):
                n = self.node_by_name(key, recursive=False, first=True)
                if n is None:
                    self.add_node(key, value)
                else:
                    n.data = value
            else:
                raise NotImplementedError

        def __contains__(self, item):
            if isinstance(item, str):
                return self.node_by_name(item, recursive=False, first=True) is not None
            else:
                raise NotImplementedError

        def indexof(self, item, force_data=False):
            if isinstance(item, Tree.Node):
                for i, n in enumerate(self._children):
                    if n is item:
                        return i
            elif isinstance(item, str) and not force_data:
                for i, n in enumerate(self._children):
                    if n.name == item:
                        return i
            else:
                for i, n in enumerate(self._children):
                    if n.data == item:
                        return i

        def index(self):
            if not self.parent:
                return None
            if self._index < 0:
                self._index = self.parent.indexof(self)
            return self._index

        @property
        def address(self):
            if self.parent:
                return self.parent.address + (self.index(),)
            return ()

        @property
        def all(self):
            return Tree.NodeChildView(self)

        def walk(self):
            if self._children:
                for id, c in enumerate(self._children):
                    c._index = id
                    yield c
                    for n in c.walk():
                        yield n

    class NodeChildView:
        def __init__(self, node):
            self._node = node

        def __getitem__(self, item):
            if isinstance(item, str):
                return self._node.node_by_name(item)
            else:
                return self._node[item]

        def __setitem__(self, key, value):
            if isinstance(key, str):
                nodes = self._node.node_by_name(key)
                for n in nodes:
                    n.data = value
            else:
                self._node[key] = value

        def __contains__(self, item):
            if isinstance(item, str):
                return self._node.node_by_name(item) is not None
            else:
                return item in self._node


########################################################################################################################
class AttributeDict(OrderedDict):
    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError('%s is unknown' % item)

    def __iter__(self):
        for v in self.values():
            yield v
