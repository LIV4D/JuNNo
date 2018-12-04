from .import_js import import_js
from IPython.display import Javascript, display


class FullScreenView:
    @staticmethod
    def setHTML(html):
        FullScreenView._exec('setHTML(%s)' % html)

    @staticmethod
    def show():
        FullScreenView._exec('show()')

    @staticmethod
    def hide():
        FullScreenView._exec('hide()')

    @staticmethod
    def showLoading():
        FullScreenView._exec('showLoading()')

    @staticmethod
    def _exec(js):
        import_js('FullscreenView')
        display(Javascript('console.log("EXEC!");'))
        display(Javascript('IPython.FullscreenView.%s;' % js))

