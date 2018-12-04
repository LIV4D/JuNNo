from IPython.display import Javascript, display
from ..path import abs_path
from ipywidgets.widgets import DOMWidget

js_module_cache = {}


def import_js(module_name, skip=None):
    """
    :param module_name:
    :return:
    """

    if skip is None:
        skip = []

    if module_name in skip:
        return []

    imported_js = []

    if module_name in js_module_cache:
        js = js_module_cache[module_name][1]
        for dep in js_module_cache[module_name][0]:
            imported_js += import_js(dep, skip+imported_js)
    else:
        try:
            if module_name.endswith('.css'):
                with open(abs_path('javascript/'+module_name, __file__), 'r') as file:
                    js_src = file.read()
            else:
                with open(abs_path('javascript/'+module_name+'.js', __file__), 'r') as file:
                    js_src = file.read()
        except UnicodeDecodeError as err:
            raise ValueError('%s is not readable:\n%s' % (module_name, err))
        except IOError:
            raise ValueError('%s is not a valid module name.' % module_name)


        js = ""
        dependencies = []
        for line in js_src.splitlines():
            if line.startswith('#INCLUDE '):
                js_include = line[9:]
                if js_include.startswith('CSS '):
                    try:
                        js_include = js_include[4:]
                        js_include += '.css'
                        dependencies.append(js_include)
                        imported_js += import_js(js_include, skip+imported_js)
                    except ValueError:
                        raise ValueError('%s is not a valid module name (while loading module %s)'
                                         % (import_js, module_name))
                else:
                    try:
                        dependencies.append(js_include)
                        imported_js += import_js(js_include, skip+imported_js)
                    except ValueError:
                        raise ValueError('%s is not a valid module name (while loading module %s)'
                                         % (import_js, module_name))
            else:
                js += line + '\n'
        js_module_cache[module_name] = (dependencies, js)

    if module_name.endswith('.css'):
        js = '''
        if(!IPython.hasOwnProperty('imported_js')){
            IPython.imported_js = [];
        }
        var module_name = '%s';
        if(IPython.imported_js.indexOf(module_name)<0){
            IPython.imported_js.push(module_name);
            var style = document.createElement('style');
            document.head.appendChild(style);
            style.innerHTML = `%s`;
        }''' % (module_name, js)
    else:
        js = '''
        if(!IPython.hasOwnProperty('imported_js')){
            IPython.imported_js = [];
        }
        var module_name = '%s';
        if(IPython.imported_js.indexOf(module_name)<0){
            IPython.imported_js.push(module_name);
            %s
        }''' % (module_name, js)

    display(Javascript(js))
    imported_js.append(module_name)
    return imported_js


class AutoImportDOMWidget(DOMWidget):
    def __init__(self, dependencies='', **kwargs):
        if isinstance(dependencies, str):
            dependencies = (dependencies,)
        self._dependencies = dependencies
        super(AutoImportDOMWidget, self).__init__(**kwargs)

    def _ipython_display_(self):
        for dep in self._dependencies:
            import_js(dep)
        display(super(AutoImportDOMWidget, self))


def import_display(dom_widget):
    def recursive_import(w, imported_js):
        if isinstance(w, AutoImportDOMWidget):
            for dep in w._dependencies:
                imported_js += import_js(dep, imported_js)
        elif hasattr(w, 'children'):
            for c in w.children:
                recursive_import(c, imported_js)

    recursive_import(dom_widget, [])
    display(dom_widget)
