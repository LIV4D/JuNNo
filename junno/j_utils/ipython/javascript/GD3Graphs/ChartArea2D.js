"use strict";

class ChartArea2D extends D3CustomChart{
    constructor(canva, name, chartType=[]){
        chartType.unshift('ChartArea2D');
        super(canva, name, chartType);
        var self=this;
        
        // Graph svg components
        this.background = this.gRoot.append('g').classed('areaBackground', true);
        this.gXAxis = this.gRoot.append('g').attr('id', name+'_XAxis')
                                            .classed('axis', true)
                                            .classed('xaxis', true);
        this.gYAxis = this.gRoot.append('g').attr('id', name+'_YAxis')
                                            .classed('axis', true)
                                            .classed('yaxis', true);
        this.gData = this.gRoot.append('g').attr('id', name+'_ChartData')
                                           .classed('chartData', true);
        this.tip = d3.tip().attr('class', 'd3-tip d3-tip-'+name);
        this.tipRootElement = this.gData;
        
        this.initChart();
        
        // Axis
        this.y = D3.scale.linear();
        this.x = D3.scale.linear();
        
        this.xAxis = D3.svg.axis().scale(this.x).orient('bottom');
        this.yAxis = D3.svg.axis().scale(this.y).orient('left');
        
        // ChartArea properties
        this.chainableFunctionProperty('dataX', d=>d.x, 'dataChanged');
        this.chainableFunctionProperty('dataY', d=>d.y, 'dataChanged');
        this.chainableFunctionProperty('seriesData', d=>d.data, 'dataChanged');
        this.chainableFunctionProperty('seriesName', d=>d.name, 'draw');
        this.chainableFunctionProperty('seriesFilter', d=>true, 'dataChanged');
        
        this.chainableProperty('xTitle', '', 'draw');
        this.chainableProperty('xUnit', '', 'draw');
        this.chainableProperty('xTitleShort', null);
        this.chainableFunctionProperty('domainX', d=>d, 'dataChanged');
        
        this.chainableProperty('yTitle', '', 'draw');
        this.chainableProperty('yUnit', '', 'draw')
        this.chainableProperty('yTitleShort', null);
        this.chainableFunctionProperty('domainY', d=>d, 'dataChanged');
        
        this.chainableProperty('backgroundLabel', '', 'draw');
        this.chainableProperty('dataHoverable', true);
        this.chainableProperty('hoveredSerie', null, 'draw');
        
        // ChartArea cache
        this._dataDomainX = [0,0];
        this._dataDomainY = [0,0];
        this._mouseDown = false;
        this._classed = {};
    }
    
    initChart(){
        var self = this;
        this.gXAxis.append('g').classed('axisTitle', true).append('text');
        this.gYAxis.append('g').classed('axisTitle', true).append('text');
        
        // Background
        this.background.append('rect').classed('backgroundSurface', true);
        this.background.append('text').classed('backgroundLabel', true);
        
        this.clipRect = this.gData.append('clipPath').attr('id', 'clipPath'+this.name)
                                  .append('rect');
        
        // Cursors
        this.gData.append('line').classed('vCursor', true).classed('lineCursor', true)
                                 .attr('visibility', 'hidden');
        this.gData.append('line').classed('hCursor', true).classed('lineCursor', true)
                                 .attr('visibility', 'hidden');
        
        this.on('mousein', function(e){
            self.vCursor.attr('visibility', 'visible');
            self.hCursor.attr('visibility', 'visible');
        });
        this.on('mouseout', function(e){
            self.vCursor.attr('visibility', 'hidden');
            self.hCursor.attr('visibility', 'hidden');
        });
        
        this.on('mousemove', function(e){self.mousePosChanged(e.pos)});
                                 
        this.hoverArea = this.gData.append('rect').classed('dataHoverArea', true)
                                   .attr('fill', '#fff')
                                   .attr('opacity', 0.00001);
                                   
        var self = this;
        this.gRoot.on('mouseover', function(){
            var e = [{pos:D3.mouse(this), mouseDown:self._mouseDown}];
            self.emit('mousein', e);
            self.emit('mousemove', e);
        }).on('mousemove', function(){
            var e = [{pos:D3.mouse(this), mouseDown:self._mouseDown}];
            self.emit('mousemove', e);
        }).on('mouseout', function(){
            self._mouseDown = false;
            var e = [{pos:null, mouseDown:self._mouseDown}];
            self.emit('mouseout', e);
            self.emit('mousemove', e);
        }).on('click', function(){
            var e = [{pos:D3.mouse(this), mouseDown:self._mouseDown}];
            self.emit('click', e);
        }).on('mousedown', function(){
            self._mouseDown = true;
            var e = [{pos:D3.mouse(this), mouseDown:self._mouseDown}];
            self.emit('mousedown', e);
        }).on('mouseup', function(){
            self._mouseDown = false;
            var e = [{pos:D3.mouse(this), mouseDown:self._mouseDown}];
            self.emit('mouseup', e);
        });
    }
    
    drawChart(gRoot, data){
        
        // Draw 2DChartArea
        this.x.range([0, this.width]);
        this.gXAxis.attr('transform', 'translate(0, '+this.height+')');
        this.gXAxis.transition().duration(this._animDuration).call(this.xAxis);
        
        this.y.range([this.height, 0]);
        this.gYAxis.transition().duration(this._animDuration).call(this.yAxis);
        
        this.gXAxis.select('.axisTitle').attr('transform', 'translate('+this.width/2+',35)');
                   
        if(this._xUnit!='')
            this.gXAxis.select('text').html('<tspan class="axisTitleText">'+this._xTitle+'</tspan> <tspan class="axisUnitText"> ('+this._xUnit+')</tspan>');
        else
            this.gXAxis.select('text').html('<tspan class="axisTitleText">'+this._xTitle+'</tspan>');
            
        this.gYAxis.select('.axisTitle').attr('transform', 'translate('+(15-this._marginLeft).toString()+', '+(this.height/2+50)+')rotate(-90)');
        if(this._yUnit!='')
            this.gYAxis.select('text').html('<tspan class="axisTitleText">'+this._yTitle+'</tspan> <tspan class="axisUnitText"> ('+this._yUnit+')</tspan>');
        else
            this.gYAxis.select('text').html('<tspan class="axisTitleText">'+this._yTitle+'</tspan>');
        
        // Draw data
        this.background.select('.backgroundSurface').attr('width', toPx(this.width)).attr('height', toPx(this.height));
        this.background.select('.backgroundLabel').html(this._backgroundLabel);
        this.clipRect.attr('width', toPx(this.width)).attr('height', toPx(this.height));
        try{
            var bckLabelBBox = this.background.select('.backgroundLabel').node().getBBox();
            this.background.select('.backgroundLabel').attr('x', toPx(this.width - bckLabelBBox.width - 20))
                                                      .attr('y', toPx(this.height - 20));
        }catch(error){}
        
                                                  
        this.hoverArea.attr('width', toPx(this.width)).attr('height', toPx(this.height));
        
        data = this._data.filter(d => this._seriesFilter(d));
        this.drawData(this.gData, data);
        this.emit('dataDrawn', [{data:data, gData:this.gData}]);
        
        // Draw cursors
        var vCursor = this.gData.selectAll('.vCursor');
        var hCursor = this.gData.selectAll('.hCursor');
        
        vCursor.attr('y1', 0)
        vCursor.attr('y2', this.height)
        
        hCursor.attr('x1', 0)
        hCursor.attr('x2', this.width)
        
        this.tipRootElement.call(this.tip);
    }
    
    mousePosChanged(mousePos){
        if(mousePos != null){
            var vCursor = this.gData.selectAll('.vCursor');
            var hCursor = this.gData.selectAll('.hCursor');
            
            vCursor.attr('x1', mousePos[0]).attr('x2', mousePos[0]);
            hCursor.attr('y1', mousePos[1]).attr('y2', mousePos[1]);
        }
        if(this._dataHoverable)
            this.hoverNearestData(mousePos);
    }
    
    hoverNearestData(mousePos){}
    
    dataChanged(){
        this.lockDraw();
        
        var data = this._data.filter(d => this._seriesFilter(d));
        
        this._dataDomainX = [D3.min(data, serie => D3.min(this._seriesData(serie), d => this._dataX(d))),
                             D3.max(data, serie => D3.max(this._seriesData(serie), d => this._dataX(d)))];
                        
        this._dataDomainY  = [D3.min(data, serie => D3.min(this._seriesData(serie), d => this._dataY(d))),
                        D3.max(data, serie => D3.max(this._seriesData(serie), d => this._dataY(d)))];
        this.x.domain(this._domainX(this._dataDomainX));
        var realDomainY = this._domainY(this._dataDomainY);
        this.y.domain(realDomainY);
        
        this.emit('domainChange', [{domainX: this._dataDomainX, domainY: this._dataDomainY}]);
        
        this.unlockDraw(true);
    }
    
    get vCursor(){return this.gData.selectAll('.vCursor');}
    get hCursor(){return this.gData.selectAll('.hCursor');}
    
    classed(name, f){
        if(!isFunction(f)){
            var a = f;
            f = d=>a;
        }
        this._classed[name] = f;
        this.requestDraw();
    }
}
