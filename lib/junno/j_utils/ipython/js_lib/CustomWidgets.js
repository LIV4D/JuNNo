require.undef('gcustom');


define('gcustom', ["@jupyter-widgets/base"], function(widgets) {

    // ___________________________
    // ---    TinyLoading      ---
    var TinyLoading = widgets.DOMWidgetView.extend({
        render: function() {
            var loadingBar = document.createElement("div");
            loadingBar.style.width = '100%';
            loadingBar.style.height = "6px";
            loadingBar.style.borderRadius = "5px";
            loadingBar.style.backgroundColor = "#CCCCCC";
            this.loadingBar = loadingBar;
            
            var bar = document.createElement("div");
            bar.style.width = "50%";
            bar.style.height = "100%";
            bar.style.borderRadius = "5px";
            bar.style.backgroundColor = "#2facff";
            bar.style.position = 'relative';
            bar.style.left = '0%';

            this.barStyle = bar.style;
            
            loadingBar.appendChild(bar);
            this.el.appendChild(loadingBar);
            
            this.value_changed();
            this.model.on('change:value', this.value_changed, this);
                    
            this.color_changed();
            this.model.on('change:color', this.color_changed, this);
            
            this.background_color_changed();
            this.model.on('change:background_color', this.background_color_changed, this);
            
            this.bar_height_changed();
            this.model.on('change:bar_height', this.bar_height_changed, this);
            
            this.delta = 2;
            this.pos = 0;
            var self = this;
            setInterval(function(){
                if(self.anim){
                    if(self.pos + self.delta <= 0)
                        self.delta = +2;
                    else if(self.pos + self.delta >= 92)
                        self.delta = -2;
                    self.pos += self.delta;
                    self.barStyle.left = self.pos.toString() + '%';
                }
            }, 25);
        },
        
        value_changed: function() {
            var progress = this.model.get('value');
            if(progress >= 0){
                if(this.anim){
                    this.anim = false;
                    this.barStyle.left = '0%';
                }
                this.barStyle.width = (progress*100).toString() + '%';
            }else if(!this.anim){
                this.barStyle.width = '8%';
                this.anim = true;
            }
        },
        
        color_changed: function() {
            this.barStyle.backgroundColor = this.model.get('color');
        },
        
        background_color_changed: function(){
            this.loadingBar.style.backgroundColor = this.model.get('background_color');
        },
        
        bar_height_changed: function(){
            this.loadingBar.style.height = this.model.get('bar_height');
        },
    });
    
    // ___________________________
    // ---    Label      ---
    var RichLabel = widgets.DOMWidgetView.extend({
        render: function() {
            this.value_changed();
            this.model.on('change:value', this.value_changed, this);
                    
            this.color_changed();
            this.model.on('change:color', this.color_changed, this);
            
            this.font_changed();
            this.model.on('change:font', this.font_changed, this);
            
            this.align_changed();
            this.model.on('change:align', this.align_changed, this);
        },
        
        value_changed: function() {
            this.el.textContent = this.model.get('value');
        },
        
        color_changed: function() {
            this.el.style.color = this.model.get('color');
        },
        
        font_changed: function() {
            this.el.style.font = this.model.get('font');
        },
        
        align_changed: function() {
            this.el.style.textAlign = this.model.get('align');
        },
    });
    
    // ___________________________
    // ---     TimerLabel      ---
    var TimerLabel = RichLabel.extend({
        render: function() {
            
            this.timer = 0;
            this.initial_timer = 0;
            this.start_time = 0;
            this.update_changed();
            
            this.timer_changed()
            this.model.on('change:_time', this.timer_changed, this);
            this.model.on('change:running', this.timer_changed, this);
            
            this.value_changed();
            this.model.on('change:value', this.value_changed, this);
                    
            this.color_changed();
            this.model.on('change:color', this.color_changed, this);
            
            this.font_changed();
            this.model.on('change:font', this.font_changed, this);
             
            this.align_changed();
            this.model.on('change:align', this.align_changed, this);
            
            var self = this;
            setInterval(function(){
                if(self.model.get('running')){
                    self.timer = self.local_timer()
                    self.update_changed();
                }
            }, 1000);
        },
        
        timer_changed: function() {
            this.timer = this.model.get('_time');
            if(this.model.get('running')){
                this.initial_timer = this.timer;
                var d = new Date();
                this.start_time = d.getTime()/1000;
            }
            this.update_changed();
        },
        
        local_timer: function(){
            if(this.model.get('running')){
                var d = new Date();
                return this.initial_timer  + this.model.get('step')*Math.round(d.getTime()/1000 - this.start_time);
            }else
                return this.timer;
        },
        
        update_changed: function(){
            var t = this.timer;
            var d = Math.floor(t / 86400);
            var h = Math.floor(t % 86400 / 3600);
            var m = Math.floor(t % 3600 / 60);
            var s = t % 60;
            h = h>9 ? h.toString() : '0' + h.toString();
            m = m>9 ? m.toString() : '0' + m.toString();
            s = s>9 ? s.toString() : '0' + s.toString();
            if(d > 0){
                if(d==1)
                    this.model.set('value', '1 jour ' + h + ' : ' + m + ' : ' + s);
                else
                    this.model.set('value', d.toString() + ' jours ' + h + ' : ' + m + ' : ' + s);
            }else
                this.model.set('value', h + ' : ' + m + ' : ' + s);
        },
    });
    
    // ___________________________
    // ---       LogView       ---
    var LogView = widgets.DOMWidgetView.extend({
        render: function() {
            this.row_tpl = `
            <tr style="background-color: %backColor;"
                onmouseover="this.style.boxShadow =  '0 0 7px -2px rgba(0, 0, 0, 0.2) inset';"
                onmouseout="this.style.boxShadow = '0 0px 0px 0 rgba(0, 0, 0, 0.2)';">
                <td style="width 5%; margin: 7px; padding: 8px"> %icon </td>
                <td style="width: 80%; padding: 4px"> %txt </td>
                <td style="text-align: right; color: %timingColor; font-size: 12px;
                           width: 15%; padding-right: 10px; "> %timing </td>
            </tr>
            `;
            this.icon_tpl = '<i class="fa %icon " style="font-size:20px; color: %iconColor;"></i>';
            
            this.update_log();
            this.model.on('change:value', this.update_log, this);
            this.model.on('change:filter_debug', this.update_log, this);
            this.model.on('change:filter_error', this.update_log, this);
            this.model.on('change:filter_warning', this.update_log, this);
            this.model.on('change:filter_info', this.update_log, this);
            this.model.on('change:search_filter', this.update_log, this);
            this.model.on('change:case_sensitive', this.update_log, this);
        },
    
        update_log: function() {
            const self = this;
            let getScrollMax = function(){
            const ref = self.el.scrollTopMax;
              return  ref !== undefined
                  ? ref
                  : (self.el.scrollHeight - self.el.clientHeight);
            };

            var log = this.model.get('value');
            var logs_list = log.split('\\\\');
            
            var autoscroll = this.el.scrollTop === getScrollMax();
            
            var html = '<table style="width: 100%;">';
            var trStyle = 'margin: 7px; padding: 7px';
            var previousType = '';
            for(var i=0; i<logs_list.length; i++){
                var l = logs_list[i];
                if(l.length > 0){
                    var type = l.substr(0, l.indexOf('|'));
                    var txt = l.substr(l.indexOf('|')+1, l.indexOf('@') - l.indexOf('|') - 1);
                    var timing = l.substr(l.indexOf('@')+1);
                    
                    var search = this.model.get('search_filter');
                    if(search != ''){
                        var txt_searched = txt;
                        if(!this.model.get('case_sensitive')){
                            txt_searched = txt.toLowerCase();
                            search = search.toLowerCase();
                        }
                        if(txt_searched.indexOf(search) == -1)
                            continue;
                        txt = txt.replace(search, '<span style="background-color: #fbffc9">'+search +'</span>');
                    }
                    
                    var icon = type == previousType ? '' : this.icon_tpl;
                    var row = this.row_tpl;
                    row = row.replace('%icon', icon)
                             .replace('%txt ', txt)
                             .replace('%timing ', timing);
                    
                    if(type == 'info' && this.model.get('filter_info')){
                        row = row.replace('%iconColor', '#2176ff')
                                 .replace('%icon', 'fa-info-circle')
                                 .replace('%backColor', '#eff7ff')
                                 .replace('%timingColor', '#86c3dd');
                        html += row;
                    }else if(type == 'debug' && this.model.get('filter_debug')){
                        row = row.replace('%iconColor', '#777')
                                 .replace('%icon', 'fa-code')
                                 .replace('%backColor', '#eee')
                                 .replace('%timingColor', '#838383');
                        html += row;
                    }else if(type == 'error' && this.model.get('filter_error')){
                        row = row.replace('%iconColor', '#c71e1e')
                                 .replace('%icon', 'fa-exclamation-circle')
                                 .replace('%backColor', '#ffd2ca')
                                 .replace('%timingColor', '#de7575');
                        html += row;
                    }else if(type == 'warning' && this.model.get('filter_warning')){
                        row = row.replace('%iconColor', '#eba427')
                                 .replace('%icon', 'fa-exclamation-triangle')
                                 .replace('%backColor', '#fff3cf')
                                 .replace('%timingColor', '#eba427');
                        html += row;
                    }
                    previousType = type;
                }
            }
            html += '</table>';
            
            this.el.innerHTML = html;
            
            if(autoscroll)
                this.el.scrollTop = getScrollMax(this.el);
        },
    });
    
    // ___________________________
    // ---     HTMLButton      ---
    var HTMLButton = widgets.DOMWidgetView.extend({
        render: function() {
            var button = document.createElement("div");
            button.style.width = '80%';
            button.style.height = '80%';
            button.style.boxShadow =  '1px 1px 3px rgba(0, 0, 0, 0.2)';
            button.style.display = 'flex';
            button.style.alignItems = 'center';
            button.style.justifyContent = 'center';
            button.style.outline = 'none';
            button.style.cursor = 'pointer';
            button.setAttribute('tabindex', '0')
            
            var self = this;
            button.onmouseover = function(){
                this.style.boxShadow =  '1px 1px 5px rgba(0, 0, 0, 0.3)';
            };
            button.onmouseout =  function(){
                this.style.boxShadow =  '1px 1px 3px rgba(0, 0, 0, 0.2)';
            };
            button.onmousedown = function(){
                if(self.model.get('toggleable')){
                    if(!self.model.get('value'))
                        self.model.set('value', true);
                    else if(self.model.get('resettable'))
                        self.model.set('value', false);
                    self.touch();
                }else{
                    self.model.set('value', true);
                    self.touch();
                }
            };
            button.onmouseup = function(){
                if(!self.model.get('toggleable'))
                    self.model.set('value', false);
                self.model.set('_clicked', true);
                self.touch();
                self.onclick();
            };   
            button.addEventListener('focus', function(){
                self.model.set('has_focus', true);
                self.touch();
                button.style.border = '1px solid #65b3e7';
            });
            button.addEventListener('blur', function(){
                self.model.set('has_focus', false);
                self.touch();
                button.style.border = '0px solid white';
            });
                
            this.el.appendChild(button);
            this.button = button;
            
            this.type_changed();
            this.model.on('change:type', this.type_changed, this);
            
            this.html_changed();
            this.model.on('change:html', this.html_changed, this);
            
            this.value_changed();
            this.model.on('change:value', this.value_changed, this);
            this.model.on('change:button_color', this.value_changed, this);
            this.model.on('change:pressed_color', this.value_changed, this);
            
            
        },
        
        value_changed: function() {
            if(this.model.get('value')){
                this.button.style.backgroundColor = this.model.get('pressed_color');   
            }else{
                this.button.style.backgroundColor = this.model.get('button_color');
            }
        },
        
        html_changed: function() {
            this.button.innerHTML = this.model.get('html');
        },
        
        type_changed: function(){
            var type = this.model.get('type');
            this.button.style.width = '80%';
            if(type=='left'){
                this.button.style.borderTopLeftRadius = '5px';
                this.button.style.borderBottomLeftRadius = '5px';
                this.button.style.borderTopRightRadius = '0px';
                this.button.style.borderBottomRightRadius = '0px';
                this.button.style.margin = '7% 0% 13% 20%';
            }else if(type=='right'){
                this.button.style.borderTopLeftRadius = '0px';
                this.button.style.borderBottomLeftRadius = '0px';
                this.button.style.borderTopRightRadius = '5px';
                this.button.style.borderBottomRightRadius = '5px';
                this.button.style.margin = '7% 20% 13% 0%';
            }else if(type=='middle'){
                this.button.style.borderTopLeftRadius = '0px';
                this.button.style.borderBottomLeftRadius = '0px';
                this.button.style.borderTopRightRadius = '0px';
                this.button.style.borderBottomRightRadius = '0px';
                this.button.style.margin = '7% 0% 13% 0%';
                this.button.style.width = '100%';
            }else{
                this.button.style.borderTopLeftRadius = '5px';
                this.button.style.borderBottomLeftRadius = '5px';
                this.button.style.borderTopRightRadius = '5px';
                this.button.style.borderBottomRightRadius = '5px';
                this.button.style.margin = '7% 13% 13% 7%';
            }
        },
        
        focusin: function(){
        
        },
        focusout:function(){
        },
        
        onclick: function(){},
        
    });
    
    
    // ___________________________
    // ---     HierarchyBar    ---
    var HierarchyBar = widgets.DOMWidgetView.extend({
        render: function() {
            var container = document.createElement("div");
            container.style.width = '100%';
            container.style.minHeight = '30px';
            container.style.paddingTop = '10px';
            container.style.alignItems = 'center';
            this.el.appendChild(container);
            this.container = container;
            
            this.value_changed();
            this.model.on('change:value', this.value_changed, this);
            this.model.on('change:current_id', this.current_id_changed, this);
        },
        
        value_changed: function() {
            while (this.container.hasChildNodes() === true)
                this.container.removeChild(this.container.firstChild);
            
            var names = this.model.get('value');
            for(var i=0; i<names.length; i++){
                this.container.appendChild(this.create_button(names[i], i));
                if(i!=names.length-1)
                    this.container.appendChild(this.create_arrow(i));
            }
            
            this.current_id_changed();
        },
        
        current_id_changed: function(){
            var id = this.model.get('current_id');
            if(id==-1)
                    id = this.model.get('value').length-1;
            id *= 2;
            for(var i=0; i<this.container.childNodes.length; i++){
                if(i==id){
                    this.container.childNodes[i].style.fontWeight = 'bold';
                    this.container.childNodes[i].style.color = '#444';
                }else{
                    this.container.childNodes[i].style.fontWeight = 'normal';
                    
                    if(i<id){
                        if(i%2)
                            this.container.childNodes[i].style.color = '#888';
                        else
                            this.container.childNodes[i].style.color = '#444';
                    }else{
                        if(i%2)
                            this.container.childNodes[i].style.color = '#CCC';
                        else
                            this.container.childNodes[i].style.color = '#888';
                    }
                }
            }
        },
        
        create_button: function(text, id){
            var t = document.createElement('span');
            t.textContent = text;
            t.style.fontSize = '25px';
            t.style.fontFamily = 'Arial';
            t.style.color = '#444';
            t.borderRadius = '5px';
            t.height = '50px';
            t.onmouseover = function(){
                t.style.backgroundColor =  '#EEE';
            };
            t.onmouseout =  function(){
                t.style.backgroundColor =  '#FFF';
            };
            
            var self = this;
            t.onmousedown = function(){
                self.model.set('current_id', id);
                self.touch();
            };
            return t;
        },
        
        create_arrow: function(id){
            var a = document.createElement('span');
            a.className = "fa fa-angle-right";
            a.style.color = '#888';
            a.style.fontSize = '20px';
            a.style.marginLeft = '8px';
            a.style.marginRight = '8px';
            
            return a;
        },
        
    });

    return {
        TinyLoading : TinyLoading,
        RichLabel: RichLabel,
        TimerLabel: TimerLabel,
        LogView:  LogView,
        HTMLButton: HTMLButton,
        HierarchyBar: HierarchyBar,
    };
}); 
