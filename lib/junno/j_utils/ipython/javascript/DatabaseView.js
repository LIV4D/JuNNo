#INCLUDE util
#INCLUDE FullscreenView

require.undef('gdatabaseview');

define('gdatabaseview', ["@jupyter-widgets/base"], function(widgets) {
    
    // ___________________________
    // ---    SeriesPlot       ---
    var DatabaseView = widgets.DOMWidgetView.extend({
        render: function(){
            var container = document.createElement('div');
            container.style.width = '100%';
            container.style.overflowX = 'auto';
            
            var style = document.createElement('style');
            document.head.appendChild(style);
            
            // -- Defining header --
            var tableHeader = document.createElement('tr');
            this.tableHeader = tableHeader;
            
            this.tableHeaderCells = new Array(0);
            
            var tableHead = document.createElement('thead'); 
            tableHead.appendChild(tableHeader);
            
            style.sheet.insertRule(` .CustomDatabaseTable thead th{ 
                    margin-left: 1px;
                    display: table-cell;
                    border: 1px solid #cfcfcf;
                    background-color: #f7f7f7;
                }`);
            
            style.sheet.insertRule(`.CustomDatabaseTable thead tr:after{
                    content: '';
                    overflow-y: scroll;
                    visibility: hidden;
                    height: 0;
                }`);
            style.sheet.insertRule(`.CustomDatabaseTable thead h3{
                    margin: 2px 10px 2px 10px;
                    text-align: center;
                    color: #444;
                    font-size: 20px;
                }`);
            style.sheet.insertRule(`.CustomDatabaseTable thead h4{
                    margin: 2px 5px 2px 5px;
                    text-align: center;
                    color: #666;
                    font-size: 13px;
                }`);
            
            
            
            // -- Defining body --
            var tableBody = document.createElement('tbody');
            tableBody.style.overflowY = 'auto';
            tableBody.style.overflowX = 'hidden';
            tableBody.style.display = 'block';
            tableBody.style.width = '100%';
            tableBody.style.maxHeight = '400px';
            this.tableBody = tableBody;
            
            style.sheet.insertRule(` .CustomDatabaseTable tr{
                    display: block;
                    border-bottom: 1px solid #dfdfdf;
                }`);
            
            style.sheet.insertRule(` .CustomDatabaseTable td{
                    padding: 3px;
                    justify-content: middle;
                    align-content: middle;
                    min-width: 120px;
                }`);
            
            style.sheet.insertRule(` .CustomDatabaseTable p{
                    text-align: center;
                    overflow-wrap: break-word;
                    width: 194px;
                }`);
            
            style.sheet.insertRule(` .CustomDatabaseTable img{
                }`);

            style.sheet.insertRule(` .CustomDatabaseTableIndex{
                    
                }`);
            
            style.sheet.insertRule(` .CustomDatabaseTableIndex h6{
                    text-align: center;
                    color: #bbb;
                    font-size: 10px;
                    width: 24px;
                }`);
            
            // -- Defining data storage --
            this.data = {};
            this.next = {'row':0, 'col': 0};
            this.current = {'row':-1};
            
            this.model.on('change:clear_cache_data', this.clear_cache_data_changed, this);
            this.clear_cache_data_changed();
            
            // -- Defining table --
            var table = document.createElement('table');
            table.className = 'CustomDatabaseTable';
            table.appendChild(tableHead);
            table.appendChild(tableBody);
            
            table.style.height = '100%';
            table.style.border = '0';
            table.style.borderCollapse = 'collapse';
            
            
            this.table = table;
            this.container = container;
            container.appendChild(table);
            this.el.appendChild(container);
            
            
            this.visibleChanged();
            this.model.on('change:visible', this.visibleChanged, this);
            
            this.columnsChanged();
            this.model.on('change:columns_name', this.columnsChanged, this);
            this.model.on('change:limit', this.updateBody, this);
            this.model.on('change:offset', this.updateBody, this);
            this.model.on('change:length', this.updateBody, this);
            
            var self = this;
            this.db_comm = null;
            setTimeout(function(){
                self.db_comm = Jupyter.notebook.kernel.comm_manager.new_comm('database_comm'+self.model.get('_database_id'));
                self.db_comm.on_msg(function(msg){self.msg_handle(msg['content']['data'])});
                self.request_next();
            }, 1000);
        },
        
        visibleChanged: function(){
            this.container.style.display = this.model.get('visible') ? 'block' : 'none';
            if(this.model.get('visible')){
                this.request_next();
            }
        },
        
        columnsChanged: function(){
            
            IPython.util.recursiveDOMDelete(this.tableHeader);
            this.clear_data();
            
            var columns_name = this.model.get('columns_name').split('|');
            this.columnsCount = columns_name.length;
            this.tableHeaderCells = new Array(this.columnsCount);
            
            var indexCorner = document.createElement('th');
            indexCorner.style.width = '30px';
            indexCorner.style.backgroundColor = "white";
            indexCorner.style.border = "none";
            this.tableHeader.appendChild(indexCorner);
            for(var i=0; i<this.columnsCount; i++){
                var column = columns_name[i].split(';');
                var cell = this.createHeader(column[0], column[1], i)
                this.tableHeader.appendChild(cell);
                this.tableHeaderCells[i] = cell;
            }                
            
            this.updateBody();
        },
        
        createHeader: function(title, subtitle, colId){
            var headerTitle = document.createElement('h3');
            headerTitle.textContent = title;
            
            var headerSubtitle = document.createElement('h4');
            headerSubtitle.textContent = subtitle;
            
            var header = document.createElement('th');
            header.style.width = '200px';
            header.appendChild(headerTitle);
            header.appendChild(headerSubtitle);
            return header;
        },
        
        updateBody: function(){
            var limit = this.model.get('limit');
            this.offset = this.model.get('offset');
            this.length = Math.min(limit, this.model.get('length')-this.offset);
            
            this.cells = new Array(this.length);
            IPython.util.recursiveDOMDelete(this.tableBody);
            
            for(var i=0; i<this.length; i++){
                var row = document.createElement('tr');
                
                var index = document.createElement('td');
                var indexP = document.createElement('h6');
                index.className = 'CustomDatabaseTableIndex';
                index.style.width = '30px';
                index.style.minWidth = '0px';
                indexP.style.padding = '1px';
                indexP.textContent = (i+this.offset).toString();
                index.appendChild(indexP);
                row.appendChild(index);
                
                this.cells[i] = new Array(this.columnsCount);
                
                for(var j=0; j<this.columnsCount; j++){
                    var cell = document.createElement('td');
                    row.appendChild(cell);
                    this.cells[i][j] = cell;
                    if(this.data_contains(i+this.offset, j))
                        this.updateCellData(i+this.offset, j);
                }
                
                this.tableBody.appendChild(row);
                
            }
            
            
            this.next.row = this.offset;
            this.next.col = 0;
            if(this.current.row != -1)
                this.current.row = -2;
            else
                this.request_next();
        },
        
        request_next: function(){
            if(this.current.row != -1 || this.db_comm==null)
                return;
            
            while(this.data_contains(this.next.row, this.next.col)){
                if(!this.increment_next())
                    return;
            }
            
            this.current.row = this.next.row;
            this.current.col = this.next.col;
            this.db_comm.send('m' + this.current.row.toString() + ',' + this.current.col.toString())
            this.increment_next();
        },
        
            
        msg_handle: function(msg){
            var msg_data = msg;
            if(msg_data.startsWith('$')){
                IPython.FullscreenView.setContent(msg.substring(1));
                return;
            }
            
            if(this.current.row==-1)
                return;
            else if(this.current.row==-2){
                this.current.row = -1;
                this.request_next();
                return;
            }
            
            var row = this.current.row;
            var col = this.current.col;
            
            this.updateCellData(row, col, msg_data);
            
            this.model.set('cache_progress', (this.current.row-this.offset)/(this.length-1));

            this.current.row = -1;
            
            if(this.model.get('visible'))
                this.request_next();
        },
        
        updateCellData: function(row, col, data=null){
            if(data !== null){
                if(! (row in this.data))
                    this.data[row] = {};
                this.data[row][col] = data;
            }else if(this.data_contains(row, col)){
                data = this.data[row][col];
            }else
                return;
            
            var cell = this.cells[row-this.offset][col];
            
            var rowCount = 1;
            var colCount = 1;
            var container = cell;
            
            if(data.startsWith('#')){
                data = data.split('|');
                var shape = data[0].substring(1).split(',');
                data = data[1];
                rowCount = parseInt(shape[0]);
                colCount = parseInt(shape[1]);    
                container = document.createElement('div');
                container.style.display = 'grid';
                container.style.setProperty('grid-template-columns','repeat('+colCount.toString()+', 1fr)');
                container.style.gridGap = '2px';
                container.style.height = '100%';
                container.style.width = '100%';
                cell.appendChild(container);
            }
            container.innerHTML = data;
            
            var create_callback = function(channel, self){
                var c = channel.toString();
                return function(){
                    IPython.FullscreenView.showLoading();
                    self.db_comm.send('f'+row.toString()+','+col.toString()+','+channel);
                    console.log('f'+row.toString()+','+col.toString()+','+channel);
                };
            };
            
            for(var i=0; i<container.children.length; i++){
                var node = container.children[i];
                node.style.height = Math.max(200/rowCount-10, 32).toString() + 'px';
                node.style.width = 'auto';
                if(node.nodeName.toLowerCase() === 'img')
                    node.onmousedown = create_callback(i, this);

            }
            this.updateColWidth(col);
        },
        
        updateColWidth: function(col){
            var width = this.cells[0][col].clientWidth;
            this.tableHeaderCells[col].style.width = width.toString()+'px';
            this.tableHeaderCells[col].style.maxWidth = width.toString()+'px';
        },
        
        data_contains: function(row, col){
            return row in this.data && col in this.data[row];
        },
        
        clear_cache_data_changed: function(){
            var self = this;
            if(this.model.get('clear_cache_data')){
                setTimeout(function(){
                self.clear_data();
                self.updateBody();
            }, 10);
            }
        },
        
        
        clear_data: function(){
            for(var i in this.data){
                for(var j in this.data[i])
                    delete this.data[i][j];
                delete this.data[i];
            }
            
            this.next.row = 0;
            this.next.col = 0;
            this.model.set('clear_cache_data', false);
            this.touch();
            if(this.current.row!=-1)
                this.current.row = -2;
            else
                this.request_next();
        },
        
        increment_next: function(){
            if(this.next.col+1==this.columnsCount){
                if(this.next.row+1 == this.offset+this.length)
                    return false;
                this.next.row++;
                this.next.col = 0;
            }else
                this.next.col++;
            return true;
        },
            
    });
    
    return {
        DatabaseView : DatabaseView,
    };

}); 

