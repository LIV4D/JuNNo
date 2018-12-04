"use strict";

class ContextLineChart extends LineChart{
    constructor(canva, name, brush=false, chartType=[]){
        chartType.unshift('ContextLineChart');
        super(canva, name, chartType);
        var self = this;
        
        this.marginTop = 2;
        this.marginBottom = 17;
        
        if(brush){
            this.brush = D3.svg.brush().x(this.x)
                               .on('brush', function(){
                                   self.focusDomain = self.brush.empty() ? self._dataDomainX : self.brush.extent();
                                   self.emit('brush',[{empty:self.brush.empty(), extent:self.brush.extent()}]);
                             }).on('brushend', function(){self.emit('brushend', [{empty:self.brush.empty(), extent:self.brush.extent()}]);});
            this.chainableProperty('focusDomain', this._dataDomainX, function(){
                var fullDomain = false;
                if(self._focusDomain==null){
                    self._focusDomain = self._dataDomainX;
                    fullDomain = true;
                }else if(self._focusDomain[0]==self._dataDomainX[0] && self._focusDomain[1]==self._dataDomainX[1])
                    fullDomain = true;
                
                if((self.brush.empty()!= fullDomain) || (!fullDomain && (self._focusDomain[0]!=self.brush.extent()[0] || self._focusDomain[1]!=self.brush.extent()[1]))){
                    // Update brush
                    if(fullDomain){
                        self.brush.clear();
                        self.brush.event();
                    }else{
                        self.brush.extent(self._focusDomain);
                        self.brush();
                        self.brush.event();
                    }
                }
            });
            this.gBrush = this.gData.append('g').attr('class', 'brush');
            this.on('dataDrawn', function(){self.gBrush.attr('width', self.width).attr('height', self.height).call(self.brush)
                                                .selectAll('rect').attr('height', self.height);});
            this.dataHoverable(false);
        }else{
            this.brush = null;
            this.cursor = this.gData.append('line').classed('cursor', true)
                                    .attr('y1', 0).attr('y2', this.height+5);
            this.chainableProperty('cursorX', this.x.domain()[0], 'draw');
            this.x.clamp(true);
            var updateCursorPos = function(){self.cursorX = self.x.invert(D3.mouse(this)[0]);};
            this.gRoot.call(D3.behavior.drag().on('drag', updateCursorPos).on('dragstart', updateCursorPos))
            this.on('dataDrawn', function(){
                var x = self.x(self.cursorX());
                self.cursor.attr('y2', self.height+5).attr('x1',x).attr('x2',x); 
              }).on('domainChange', function(e){
                if(self.cursorX()<self.x.domain()[0])
                    self.cursorX = self.x.domain()[0];
                else if(self.cursorX()>self.x.domain()[1])
                    self.cursorX = self.x.domain()[1];
            });
        }
        this.chainableProperty('speed', 1);
                
        this._playing = false;
        setInterval(function(){self.tick()}, 50);
    }
    
    play(speed=null){
        if(speed != null)
            this.speed = speed;
        if(!this._playing && this.cursorX() == this.x.domain()[1])
            this.cursorX = this.x.domain()[0];
        
        this._playing = true;
    }
    
    pause(){
        this._playing = false;
    }
    
    tick(){
        if(!this._playing)
            return;
        var nextX = this.cursorX() + this._speed/20;
        if(nextX > this.x.domain()[1]){
            nextX = this.x.domain()[1];
            this.pause();
            this.emit('reachEnd');
        }
        this.cursorX = nextX;
    }
}

