"use strict";

class ChronologicalOverlay extends D3CustomChart{
    constructor(area2d, name, chartType=[]){
        chartType.unshift('ChronologicalOverlay');
        super(area2d.canva, name, chartType);
        var self = this;
        
        
        this.x = D3.scale.linear().domain(area2d._dataDomainX);
        this.area2d = area2d;
        area2d.on('domainChange', function(e){
            self.x.domain(e.domainX);
            self.requestDraw();
        });
        
        this.tip = d3.tip().attr('class', 'd3-tip')
                           .direction('e').offset([-2, 10]);
        
        this.chainableFunctionProperty('dataX', d=>d.x, 'draw');
        this.chainableFunctionProperty('color', d=>d.color!==undefined?d.color:'#111', 'draw');
        this.chainableFunctionProperty('dataTitle', d=>d.title);
        this.chainableFunctionProperty('dataInfo', d=>d.info);
        this.chainableFunctionProperty('htmlTip', d => this.defaultHtmlTip(d), 
                                       function(){self.tip.html(self._htmlTip);});
        this.tip.html(this._htmlTip);
        
        this.marginLeft(area2d.marginLeft())
            .marginRight(area2d.marginRight())
            .marginTop(area2d.marginTop())
            .marginBottom(area2d.marginBottom());
        
        this._classed = {};
    }
    
    drawChart(gRoot, data){
        this.x.range([0, this.width]);
        
        var lines = gRoot.selectAll('line.vLine')
                     .data(data);
                     
        lines.exit().remove(); // Remove old lines
        var newLines = lines.enter().append('line').classed('vLine', true)
                                    .attr('fill', 'none')
                                    .attr('y1', 0);
         
        var self = this;
        lines.attr('x2', d => this.x(this._dataX(d)))
             .attr('x1', d => this.x(this._dataX(d)))
             .attr('y2', this.height)
             .attr('stroke', this._color)
             
        var circleHats = gRoot.selectAll('circle.hat')
                     .data(data);
                     
        circleHats.exit().remove(); // Remove old lines
        var newHat = circleHats.enter().append('circle').classed('hat', true)
                                      .attr('fill', 'none')
                                      .attr('r', 7)
                                      .attr('cy', 0)
                                      .on('mouseover', self.tip.show)
                                      .on('mouseout', self.tip.hide);
         
        var self = this;
        circleHats.attr('cx', d => this.x(this._dataX(d)))
                  .attr('fill', d=>this._color(d));
        
        for(var c in this._classed){
            var f = this._classed[c];
            lines.classed(c, f);
            circleHats.classed(c, f);
        };
        
        gRoot.call(this.tip);
    }

    defaultHtmlTip(data){
        return `<h3>%T</h3>
        <p>%i</p>
        `.replace('%T', this._dataTitle(data))
         .replace('%i', this._dataInfo(data));
    }
    
    classed(name, f){
        if(!isFunction(f)){
            var a = f;
            f = d=>a;
        }
        this._classed[name] = f;
        this.requestDraw();
    }
}
