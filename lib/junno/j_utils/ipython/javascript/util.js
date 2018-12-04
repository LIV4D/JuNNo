IPython.util = {
    
    recursiveDOMDelete: function (DOMobject) {
        if (DOMobject) {
            while (DOMobject.hasChildNodes() === true) {
                IPython.util.recursiveDOMDelete(DOMobject.firstChild);
                DOMobject.removeChild(DOMobject.firstChild);
            }
        }
    },
}; 
