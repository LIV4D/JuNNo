#INCLUDE vega
require.undef('vega_graph');

define('vega_graph', ["@jupyter-widgets/base"], function(widgets, vega) {

    const VegaGraph = widgets.DOMWidgetView.extend({
        render: function () {

        }
    });

    return {
        VegaGraph : VegaGraph,
        };
});
