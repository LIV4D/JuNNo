#INCLUDE CSS GD3Graphs/CustomChart.css
#INCLUDE GD3Graphs/D3CustomChart
#INCLUDE GD3Graphs/ChartArea2D
#INCLUDE GD3Graphs/LineChart
#INCLUDE GD3Graphs/AreaLineChart
#INCLUDE GD3Graphs/ChronologicalOverlay
#INCLUDE GD3Graphs/ContextLineChart
#INCLUDE GD3Graphs/ScatterPlot

require.undef('gd3graphs');

define('gd3graphs', ["@jupyter-widgets/base"], function(widgets) {
    
    // ___________________________
    // ---    SeriesPlot       ---
    var D3LineChart = widgets.DOMWidgetView.extend({
        render: function(){
            var container = document.createElement('div');
            container.style.width = '100%';
            container.style.overflowX = 'auto';
            this.el.appendChild(container);
            
            var style = document.createElement('style');
            document.head.appendChild(style);
            
            
            this.model.on('change:length', this.updateBody, this);
            
            var self = this;
        },
        
            
    });
    
    return {
        D3LineChart : D3LineChart,
    };

}); 
