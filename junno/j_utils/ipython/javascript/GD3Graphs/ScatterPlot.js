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

class ScatterPlot extends ChartArea2D{
    constructor(canva, name, chartType=[]){
        chartType.unshift('ScatterPlot');
        super(canva, name, chartType);
        var self = this;
        
        // D3 functions components
        this.chainableFunctionProperty('dotColor', d => d.color!==undefined ? d.color : '#444', 'draw'); 
        this.chainableFunctionProperty('dataR', d => d.width!==undefined ? d.width : 0.5, function(){self.updateRadius();}); 
        this.chainableFunctionProperty('htmlTip', d => this.defaultHtmlTip(d), 
                                       function(){self.tip.html(self._htmlTip);});
        this._seriesData = d=>[d];
        this.tip.html(this._htmlTip).offset([-9, 0]);
        
        this.r = D3.scale.sqrt().range([0, 40]);
    }
    
    drawData(g, data){
        var dots = g.selectAll('circle')
                     .data(data);
                     
        dots.exit().remove(); // Remove old lines
        var newLines = dots.enter().append('circle').classed('chartDot', true)
                                   .attr('serie', d=>this._seriesName(d))
                                   .style('fill', this._dotColor);
         
        var self = this;
        dots.classed('hovered', d=>this._seriesName(d)==this._hoveredSerie)
            .transition()
            .ease('easeLinear')
            .duration(this._animDuration)
            .attr('r', function(d){ var r = self._dataR(self._seriesData(d)[0]); return notNaN(r)? self.r(r) : undefined; })
            .attr('cx', function(d){ var x = self._dataX(self._seriesData(d)[0]); return notNaN(x)? self.x(x) : undefined; })
            .attr('cy', function(d){ var y = self._dataY(self._seriesData(d)[0]); return notNaN(y)? self.y(y) : undefined; })
            .style('fill', this._dotColor)
            .each('end', function(){dots.attr('visibility', d=>(notNaN(self._dataR(self._seriesData(d)[0]))&&
                                                    notNaN(self._dataX(self._seriesData(d)[0]))&&
                                                    notNaN(self._dataY(self._seriesData(d)[0])) )?'visible':'hidden')});
        
        for(var c in this._classed){
            var f = this._classed[c];
            dots.classed(c, f);
        };
        
        if(this._hoveredSerie!=null){
            this.tip.show(data.filter(d=>this._seriesName(d)==this._hoveredSerie)[0], 
                          dots.filter(d=>this._seriesName(d)==this._hoveredSerie).node());
        }else{
            this.tip.hide();
        }
    }
    
    hoverNearestData(mousePos){
        var dots = this.gData.selectAll('circle');
        
        if(mousePos !== null && dots.data().length){
            var mouseX = mousePos[0], mouseY = mousePos[1];
            
            var self = this;
            var distance = d => (self.x(self._dataX(self._seriesData(d)[0]))-mouseX)**2 + (self.y(self._dataY(self._seriesData(d)[0]))-mouseY)**2;
            
            var dotsData = dots.data();
            var p = dotsData[argmin(dotsData.map(distance))];
            
            if(distance(p) < 25**2){
                dots.classed('hovered', d=>this._seriesName(d)==this._seriesName(p));
                var hoveredDots = dots.filter(d=>self._seriesName(d)==self._seriesName(p));
                
                this.emit('dataHovered', [{d:hoveredDots.datum(), nearest: p, mousePos: mousePos}]);
                this.hoveredSerie = this._seriesName(p);
                return;
            }
        }
        this.emit('dataHovered', [{d:null, nearest: null, mousePos: mousePos}]);
        this.hoveredSerie = null;
    }
    
    defaultHtmlTip(data){
        return `
            <h3>  %n </h3>
            <p><b>%X</b>: %x <br/>
               <b>%Y</b>: %y</p>
        `.replace('%n', this._seriesName(data))
        .replace('%X', this._xTitleShort?this._xTitleShort:this._xTitle)
        .replace('%x', D3.format('f.2')(this._dataX(data))+' '+this._xUnit)
        .replace('%Y', this._yTitleShort?this._yTitleShort:this._yTitle)
        .replace('%y', D3.format('f.2')(this._dataY(data))+' '+this._yUnit);
    }
    
    dataChanged(){
        this.lockDraw();
        
        super.dataChanged();
        
        this.updateRadius();
        
        this.unlockDraw();
    }
    
    updateRadius(){
        if(this._data==null)
            return;
        var data = this._data.filter(d => this._seriesFilter(d));
        this.domainR = [D3.min(data, serie => D3.min(this._seriesData(serie), d => this._dataR(d))),
                        D3.max(data, serie => D3.max(this._seriesData(serie), d => this._dataR(d)))];
        this.r.domain(this.domainR);
        this.requestDraw();
    }
}
