"use strict";

class AreaLineChart extends LineChart{
    constructor(canva, name, chartType=[]){
        chartType.unshift('AreaLineChart');
        super(canva, name, chartType);
        var self = this;
        
        this.d3Area = D3.svg.area()
                        .x(d => this.x(this._dataX(d)))
                        .y1(d => this.y(this._dataY(d)))
                        .y0(d => this.height)
                        .interpolate("cardinal");
                        
        this.chainableFunctionProperty('areaColor', d => d.color!==undefined ? d.color : '#444', 'draw');
        this.on('hoveredSerieChanged', function(e){
            var areas = self.gRoot.selectAll('path.serieArea');
            if(e.hoveredSerie!=null)
                areas.classed('hovered', d=>self._seriesName(d)==e.hoveredSerie);
            else
                areas.classed('hovered', false);
        });
    }
    
    drawData(g, data){
        super.drawData(g,data);
        
        var areas = g.selectAll('path.serieArea')
                     .data(data);
                     
        areas.exit().remove(); // Remove old lines
        var newAreas = areas.enter().append('path').classed('serieArea', true)
                                    .attr('fill', 'none');
         
        var self = this;
        areas.transition().duration(this._animDuration)
             .attr('d', d => this.d3Area(this._seriesData(d).filter(d=>notNaN(this._dataX(d)) && notNaN(this._dataY(d)))))
             .attr('fill', this._areaColor);
             
        for(var c in this._classed){
            var f = this._classed[c];
            areas.classed(c, f);
        };
    }
    
    
}

