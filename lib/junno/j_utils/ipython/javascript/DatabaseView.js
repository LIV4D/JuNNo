#INCLUDE util
#INCLUDE FullscreenView

require.undef('gdatabaseview');

define('gdatabaseview', ["@jupyter-widgets/base"], function(widgets) {
    
    // ___________________________
    // ---    SeriesPlot       ---
    const DatabaseView = widgets.DOMWidgetView.extend({
        render: function(){
            const container = document.createElement('div');
            container.style.width = '100%';
            container.style.overflowX = 'auto';
            
            const style = document.createElement('style');
            document.head.appendChild(style);

            // -- Defining header --
            const tableHeader = document.createElement('tr');
            this.tableHeader = tableHeader;
            
            this.tableHeaderCells = new Array(0);
            
            const tableHead = document.createElement('thead');
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
            const tableBody = document.createElement('tbody');
            tableBody.style.overflowY = 'auto';
            tableBody.style.overflowX = 'hidden';
            tableBody.style.display = 'block';
            tableBody.style.width = '100%';
            tableBody.style.maxHeight = '600px';
            this.tableBody = tableBody;
            
            style.sheet.insertRule(` .CustomDatabaseTable tr{
                    display: block;
                    border-bottom: 1px solid #dfdfdf;
                }`);
            
            style.sheet.insertRule(` .CustomDatabaseTable td{
                    padding: 3px;
                    justify-content: middle;
                    align-content: middle;
                }`);
            
            style.sheet.insertRule(` .CustomDatabaseTable p{
                    text-align: center;
                    overflow-wrap: break-word;
                    width: 100%;
                }`);
            
            style.sheet.insertRule(` .CustomDatabaseTable img{
                    max-width: none;
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
            const table = document.createElement('table');
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
            
            this.updateColWidthTimer = {};
            this.fullscreenControlCb = null;

            this.visibleChanged();
            this.model.on('change:visible', this.visibleChanged, this);
            
            this.columnsChanged();
            this.model.on('change:columns_name', this.columnsChanged, this);
            this.model.on('change:limit', this.updateBody, this);
            this.model.on('change:offset', this.updateBody, this);
            this.model.on('change:length', this.updateBody, this);
            
            const self = this;
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
            
            const columns_name = this.model.get('columns_name').split('|');
            this.columnsCount = columns_name.length;
            this.tableHeaderCells = new Array(this.columnsCount);
            
            const indexCorner = document.createElement('th');
            indexCorner.style.width = '30px';
            indexCorner.style.minWidth =  '30px';
            indexCorner.style.backgroundColor = "white";
            indexCorner.style.border = "none";
            this.tableHeader.appendChild(indexCorner);
            for(let i=0; i<this.columnsCount; i++){
                const column = columns_name[i].split(';');
                const cell = this.createHeader(column[0], column[1], i);
                this.tableHeader.appendChild(cell);
                this.tableHeaderCells[i] = cell;
            }                
            
            this.updateBody();
        },
        
        createHeader: function(title, subtitle, colId){
            const headerTitle = document.createElement('h3');
            headerTitle.textContent = title;
            
            const headerSubtitle = document.createElement('h4');
            headerSubtitle.textContent = subtitle;
            
            const header = document.createElement('th');
            header.style.width = '200px';
            header.appendChild(headerTitle);
            header.appendChild(headerSubtitle);
            return header;
        },
        
        updateBody: function(){
            const limit = this.model.get('limit');
            this.offset = this.model.get('offset');
            this.length = Math.min(limit, this.model.get('length')-this.offset);

            // Discard useless data
            for(let i in this.data){
                if(i < this.offset || i >= this.offset + this.length){
                    for(let j in this.data[i])
                        delete this.data[i][j];
                    delete this.data[i];
                }
            }

            this.cells = new Array(this.length);
            IPython.util.recursiveDOMDelete(this.tableBody);
            
            for(let i=0; i<this.length; i++){
                const row = document.createElement('tr');
                
                const index = document.createElement('td');
                const indexP = document.createElement('h6');
                index.className = 'CustomDatabaseTableIndex';
                index.style.width = '30px';
                index.style.minWidth = '0px';
                indexP.style.padding = '1px';
                indexP.textContent = (i+this.offset).toString();
                index.appendChild(indexP);
                row.appendChild(index);
                
                this.cells[i] = new Array(this.columnsCount);
                
                for(let j=0; j<this.columnsCount; j++){
                    const cell = document.createElement('td');
                    row.appendChild(cell);
                    this.cells[i][j] = cell;
                    if(this.data_contains(i+this.offset, j))
                        this.updateCellData(i+this.offset, j);
                }
                
                this.tableBody.appendChild(row);
                
            }
            
            
            this.next.row = this.offset;
            this.next.col = 0;
            if(this.current.row !== -1)
                this.current.row = -2;
            else
                this.request_next();
        },
        
        request_next: function(){
            if(this.current.row !== -1 || this.db_comm==null)
                return;
            
            while(this.data_contains(this.next.row, this.next.col)){
                if(!this.increment_next())
                    return;
            }
            
            this.current.row = this.next.row;
            this.current.col = this.next.col;
            this.db_comm.send('m' + this.current.row.toString() + ',' + this.current.col.toString());
            this.increment_next();
        },
        
            
        msg_handle: function(msg){
            const msg_data = msg;
            if(msg_data.startsWith('$')){
                IPython.FullscreenView.setContent(msg.substring(1), this.fullscreenControlCb);
                return;
            }
            
            if(this.current.row===-1)
                return;
            else if(this.current.row===-2){
                this.current.row = -1;
                this.request_next();
                return;
            }
            
            const row = this.current.row;
            const col = this.current.col;
            
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
            
            const cell = this.cells[row-this.offset][col];
            
            let rowCount = 1;
            let colCount = 1;
            let container = cell;

            const fullscreenable = data[0] === 'f';
            data = data.substr(1);

            if(data.startsWith('#')){
                data = data.split('|');
                const shape = data[0].substring(1).split(',');
                data = data[1];
                rowCount = parseInt(shape[0]);
                colCount = parseInt(shape[1]);    
                container = document.createElement('div');
                container.style.display = 'grid';
                container.style.setProperty('grid-template-columns','repeat('+colCount.toString()+', 1fr)');
                container.style.gridGap = '2px';
                container.style.height = '100%';
                container.style.width = '100%';
                container.style.justifyItems = 'center';
                container.style.alignItems = 'center';
                cell.appendChild(container);
            }
            container.innerHTML = data;
            
            const create_callback = function(channel, self){
                const c = channel.toString();
                const ask = function(row, col, c){
                        self.db_comm.send('f'+row.toString()+','+col.toString()+','+c);
                };
                return function(){
                    ask(row, col, c);
                    IPython.FullscreenView.showLoading();
                    this.fullscreenControlCb = function(control) {
                        switch(control){
                            case 'next':
                                if(row+1 < this.offset+this.length)
                                    ask(row+1, col, c);
                                break;
                            case 'previous':
                                if(row-1 >= this.offset)
                                    ask(row-1, col, c);
                                break;

                            case 'right':
                                if(col+1 < this.columnsCount)
                                    ask(row, col+1, 0);
                                break;
                            case 'left':
                                if(col-1 > 0)
                                    ask(row, col-1, 0);
                                break;

                            case 'bottom':
                                if(c+1 < rowCount*colCount)
                                    ask(row, col, c+1);
                                break;
                            case 'top':
                                if(c-1 > 0)
                                    ask(row, col, 0);
                                break;
                        }
                    }
                };
            };
            
            for(let i=0; i<container.children.length; i++){
                const node = container.children[i];
                //node.style.height = Math.max(200/rowCount-10, 32).toString() + 'px';
                node.style.max_width = 'none';
                if(fullscreenable)
                    node.onmousedown = create_callback(i, this);

            }
            this.updateColWidth(col);
        },
        
        updateColWidth: function(col){
            if(!this.updateColWidthTimer.hasOwnProperty(col.toString())){
                this._updateColWidth(col);
                this.updateColWidthTimer[col.toString()] = false;
                const self = this;
                setTimeout(()=>{
                    let updateNeeded = this.updateColWidthTimer[col.toString()];
                    delete self.updateColWidthTimer[col.toString()];
                    if(updateNeeded)
                        this._updateColWidth(col);

                }, 500);
            }else{
                this.updateColWidthTimer[col.toString()] = true;
            }
        },

        _updateColWidth: function(col){
            const MIN_WIDTH = 120;
            // Clear width
            this.cells.forEach((row)=>{
                row[col].style.minWidth = MIN_WIDTH.toString()+'px';
                row[col].style.width = 'auto';
                row[col].style.maxWidth = null;
            });

            // Let some delay so the width is recomputed
            setTimeout(()=>{
                // Read width
                let width = MIN_WIDTH;
                this.cells.forEach((row)=>{
                    width = Math.max(row[col].clientWidth, width);
                });

                // Apply width
                this.cells.forEach((row) => {
                    row[col].style.width = width.toString()+'px';
                    row[col].style.minWidth = width.toString()+'px';
                    row[col].style.maxWidth = width.toString()+'px';
                });

                if(col === this.tableHeaderCells.length-1 && this.tableBody.scrollHeight > this.tableBody.clientHeight){
                    width += 17;    // Scroll bar width on most web browser
                }
                this.tableHeaderCells[col].style.width = width.toString()+'px';
                this.tableHeaderCells[col].style.minWidth = width.toString()+'px';
                this.tableHeaderCells[col].style.maxWidth = width.toString()+'px';
            }, 1);

        },

        data_contains: function(row, col){
            return row in this.data && col in this.data[row];
        },
        
        clear_cache_data_changed: function(){
            const self = this;
            if(this.model.get('clear_cache_data')){
                setTimeout(function(){
                self.clear_data();
                self.updateBody();
            }, 10);
            }
        },
        
        
        clear_data: function(){
            for(let i in this.data){
                for(let j in this.data[i])
                    delete this.data[i][j];
                delete this.data[i];
            }
            
            this.next.row = 0;
            this.next.col = 0;
            this.model.set('clear_cache_data', false);
            this.touch();
            if(this.current.row!==-1)
                this.current.row = -2;
            else
                this.request_next();
        },
        
        increment_next: function(){
            if(this.next.col+1 === this.columnsCount){
                if(this.next.row+1 === this.offset+this.length)
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

