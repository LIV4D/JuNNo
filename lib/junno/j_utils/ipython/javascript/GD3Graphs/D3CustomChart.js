"use strict"; 

/**WHEN INHERITING
 * Must define:
 *  - drawChart(gRoot, data)
 * 
 * Shoud re-define:
 *  - dataChanged()
 *  
 */

class D3CustomChart{
    constructor(canva, name, chartType=[]){
        this.canva = canva;
        this.name = name;
        chartType.unshift('D3CustomChart');
        this.chartType = chartType;
        
        var self = this;
        
        // Graph svg components
        this.gRoot = this.canva.append('g').attr('id', name+'_'+chartType[0]);
        this.chartType.forEach(function(d){self.gRoot.classed(d, true)});
        this.gRoot.node().d3customchart = this;
        
        
        // ChartArea attributes
        this._canvaShape = [0,0];   // [width, height]  (auto updated)
        this.chainableProperty('marginLeft',   70, 'draw');
        this.chainableProperty('marginTop',    20, 'draw');
        this.chainableProperty('marginRight',  20, 'draw');
        this.chainableProperty('marginBottom', 50, 'draw');
        this.chainableProperty('animDuration', 1000);
        
        this.chainableProperty('data', null, 'dataChanged');
        
        // Events handler
        this._events = {};
        this._drawLockCount = 0;
        this._drawRequested = 0;
        
        $(this.canva.node().ownerDocument.defaultView).resize(function(){self.sizeChanged();});
        setTimeout(100, this.checkDataChanged);
    }
    
    draw(){
        if(this._data == null)
            return;
        // Update shape
        var r = this.canva.node().getBoundingClientRect();
        this._canvaShape =  [r.right-r.left, r.bottom-r.top];
        
        var margin = this.margins;
        this.gRoot.attr('transform', 'translate('+margin.left+', '+margin.top+')');
        
        // Draw Chart
        this.drawChart(this.gRoot, this._data);
    }
    
    checkDataChanged(){
        if(this._data != null)
            this.dataChanged();
        
    }
    
    sizeChanged(){
        var prevAnimDuration = this._animDuration;
        this._animDuration = 0;
        this.checkDataChanged();
        this._animDuration = prevAnimDuration;
    }
    
    dataChanged(){
        this.requestDraw();
    }
    
    get margins(){
        return {left: this.marginLeft(),
                right: this.marginRight(),
                top: this.marginTop(),
                bottom: this.marginBottom()
        };
    }
    
    set margins(m){
        this.lockDraw();
        if(m instanceof Array){
            if(m.length == 4){
                this.marginLeft = m[0];
                this.marginTop = m[1];
                this.marginRight = m[2];
                this.marginBottom = m[3];
            }
        }else if(m instanceof Object){
            m.left!==undefined && (this.marginLeft = m.left);
            m.right!==undefined && (this.marginRight = m.right);
            m.top!==undefined && (this.marginTop = m.top);
            m.bottom!==undefined && (this.marginBottom = m.bottom);
        }
        this.unlockDraw();
    }
    
    get width(){ return this._canvaShape[0] - this._marginLeft - this._marginRight; }
    get height(){ return this._canvaShape[1] - this._marginTop - this._marginBottom; }
    
    get canvaWidth(){ return this._canvaShape[0]; }
    get canvaHeight(){ return this._canvaShape[1]; }
    
    on(event, task){ this.addEventListener(event, task); return this;}
    
    addEventListener(name, handler){
        if (this._events.hasOwnProperty(name))
            this._events[name].push(handler);
        else
            this._events[name] = [handler];
    }
    
    removeEventListener(name, handler) {
        if (!this._events.hasOwnProperty(name))
            return;

        var index = this.events[name].indexOf(handler);
        if (index != -1)
            this.events[name].splice(index, 1);
    }
    
    emit(name, args=[]) {
        if (!this._events.hasOwnProperty(name))
            return;
        var evs = this._events[name], l = evs.length;
        for (var i = 0; i < l; i++) {
            evs[i].apply(null, args);
        }
    };
    
    
    lockDraw(){
        this._drawLockCount += 1;
        return this;
    }
    requestDraw(){
        if(!this._drawLockCount){
            if(this.data != null)
                this.draw();
        }else
            this._drawRequested = true;
    }
    unlockDraw(draw=false){
        if(this._drawLockCount > 0)
            this._drawLockCount -= 1;
        if(draw)
            this._drawRequested = true;
        if (this._drawLockCount== 0 && this._drawRequested){            
            if(this.data != null)
                this.draw();
            this._drawRequested = false;
        }
        return this;
    }
    
    chainableProperty(name, defaultValue, customSetter=null, castValue=null){
        var self = this;
        if(castValue == null)
            castValue = d=>d;
        
        this['_'+name] = castValue(defaultValue);
        
        if(customSetter=='draw')
            customSetter = function(){self.requestDraw();};
        else if(customSetter == 'dataChanged')
            customSetter = function(){self.checkDataChanged();};
        
        var updateV = function(v){
            v = castValue(v);
            if((v!=null && v == undefined) || v==self['_' + name])
                return false;
            self['_'+name] = v;
            return true;
        }
        
        if(customSetter!=null)
            this['_'+name+'Setter'] = function(v){
                var e = {};
                e[name+'Old'] = self['_'+name];
                
                if(!updateV(v))
                    return false;
                customSetter();
                e[name] = self['_'+name];
                self.emit(name+'Changed', [e]);
                return true;
            };
        else
            this['_'+name+'Setter'] = function(v){
                var e = {};
                e[name+'Old'] = self['_'+name];
                if(updateV(v)){
                    e[name] = self['_'+name];
                    self.emit(name+'Changed', [e]);
                }
            }
        
        // Accessor object
        var accessor = function(arg=null){
                                    if(arg==null)
                                        return self['_'+name];
                                    self['_'+name+'Setter'](arg);
                                    return self;
                        };
        Object.defineProperty(accessor, 'setter', {get: function(){return self['_'+name+'Setter'];},
                                                   set: function(s){self['_'+name+'Setter'] = s;} });
        accessor.extendSetter = function(s){
            accessor.setter = function(v){
                accessor.setter(v);
                s(v);
            };
        };
        
        Object.defineProperty(this, name, {set: accessor.setter, 
                                           get: function(){return accessor;}});
    }
    
    chainableFunctionProperty(name, defaultValue, customSetter=null){
        var castValue = function(v){
            if(isFunction(v))
                return v;
            else{
                var a = v;
                return d=>a;
            }
        }
        return this.chainableProperty(name, defaultValue, customSetter, castValue);
    }
}
