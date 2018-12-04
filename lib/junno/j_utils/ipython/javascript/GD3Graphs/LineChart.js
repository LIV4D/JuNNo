"use strict";

/**
* Data with shape:
*   [
*       {
*           name: 'Serie 1',
*           data: [{x: 0, y: .}, {x: 1, y: .}, ...]
*       },
*       {
*           name: 'Serie 2',
*           data: [{x: 0, y: .}, {x: 1, y: .}, ...]
*       }
*   ]
*/

class LineChart extends ChartArea2D{
    constructor(canva, name, chartType=[]){
        chartType.unshift('SimpleLineChart');
        super(canva, name, chartType);
        var self = this;
        
        this.marginBottom = 50;
        
        // D3 functions components
        this.d3Line = d3.svg.line()
                        .x(d => this.x(this._dataX(d)))
                        .y(d => this.y(this._dataY(d)))
                        .interpolate("cardinal");
                        
        this.chainableFunctionProperty('lineColor', d => d.color!==undefined ? d.color : '#444', 'draw'); 
        this.chainableFunctionProperty('lineWidth', d => d.width!==undefined ? d.width : undefined, 'draw'); 
        this.chainableFunctionProperty('enable', d=>true, 'draw');
        this.chainableFunctionProperty('htmlTip', d => this.defaultHtmlTip(d), 
                                       function(){self.tip.html(self._htmlTip);});
        this.tip.html(this._htmlTip);
        
        this.circleCursorGroup = this.gData.append('g');
        this.circleCursor = this.circleCursorGroup.append('circle').classed('circleCursor', true)
                                                  .attr('visibility', 'hidden')
                                                  .attr('r', 2);
                                                       
        this.tipRootElement = this.circleCursorGroup;
        this.tip.direction('e').offset([-2, +10]);
                                    
    }
    
    drawData(g, data){
        var lines = g.selectAll('.serieLine').data(data);

        lines.exit().remove(); // Remove old lines
        var newLines = lines.enter().append('path').classed('serieLine', true)
                                    .style('fill', 'none')
                                    .attr('clip-path', 'url(#clipPath'+this.name+')')
                                    .style('stroke', this._lineColor)
                                    .style('stroke-width', this._lineWidth)
                                    .attr('d', d => this.d3Line(this._seriesData(d).filter(d2=>notNaN(this._dataX(d2)) && notNaN(this._dataY(d2)))));
        var self = this;
        lines.transition().duration(this._animDuration)
             .attr('d', d => this.d3Line(this._seriesData(d).filter(d2=>notNaN(this._dataX(d2)) && notNaN(this._dataY(d2)))))
             .style('stroke', this._lineColor)
             .style('stroke-width', this._lineWidth);
        lines.classed('disabled', d=>!this._enable(d));
        lines.classed('hovered', d=>this._seriesName(d)==this._hoveredSerie);
        
        for(var c in this._classed){
            var f = this._classed[c];
            lines.classed(c, f);
        };
    }
    
    hoverNearestData(mousePos){
        var paths = this.gData.selectAll('path');
        var self = this;
        
        if(mousePos !== null){
            var mouseX = mousePos[0], mouseY = mousePos[1];
            
            
            var self = this;
            var closestPoints = paths.data().filter(this._enable).map(function(d){
                var serieData = self._seriesData(d);
                var dataX = serieData.map(self._dataX);
                var nextXi = D3.bisectLeft(dataX.map(self.x), mouseX);
                var nextX = self.x(dataX[nextXi]), previousX = previousX;
                if(nextXi > 0)
                    previousX = self.x(dataX[nextXi-1]);
                
                var nearestXi = Math.abs(nextX-mouseX) <= Math.abs(previousX-mouseX) ? nextXi : nextXi-1;
                
                var nearestX = dataX[nearestXi];
                var nearestY = self._dataY(serieData[nearestXi]);
                
                return {i: nearestXi, dataX: nearestX, dataY: nearestY, screenX: self.x(nearestX), screenY: self.y(nearestY), serieName: self._seriesName(d)};
            });
            
            var distance = d => (d.screenX-mouseX)**2 + (d.screenY-mouseY)**2;
            
            var p = closestPoints[argmin(closestPoints.map(distance))];
            
            if(distance(p) < 75**2){
                this.circleCursorGroup.attr('transform', 'translate('+p.screenX+', '+p.screenY+')');
                this.vCursor.attr('x1', p.screenX).attr('x2', p.screenX);
                
                if(this.hoveredSerie != p.serieName){
                    var previousPath = paths.filter(d=>self._seriesName(d)==this.hoveredSerie);
                }
                
                var newPath = paths.filter(d=>self._seriesName(d)==p.serieName);
                this.circleCursor.attr('visibility', 'visible')
                                 .style('stroke', function(){return self._lineColor(newPath.datum())});
                this.emit('dataHovered', [{d:newPath.datum(), nearest: p, mousePos: mousePos}]);
                this.hoveredSerie = p.serieName;
                this.tip.show(p, this.circleCursor.node());
                D3.selectAll('.d3-tip-'+this.name).style('pointer-events', 'none');
                return;
            }
        }
        
        this.hoveredSerie = null;
        this.circleCursor.attr('visibility', 'hidden');
        this.emit('dataHovered', [{d:null, nearest: null, mousePos: mousePos}]);
        this.tip.hide();
        this.requestDraw();
    }
    
    defaultHtmlTip(data){
        return `
            <h3>  %x </h3>
            <p><b>%n</b>: %y %u</p>
        `.replace('%n', data.serieName)
        .replace('%x', data.dataX)
        .replace('%u', this._yUnit)
        .replace('%y', D3.format('f.1')(data.dataY));
    }
    
}
