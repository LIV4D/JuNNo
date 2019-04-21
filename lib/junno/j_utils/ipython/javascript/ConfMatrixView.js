#INCLUDE util


require.undef('gconfmatrix');

define('gconfmatrix', ["@jupyter-widgets/base"], function(widgets) {

    const ConfMatrix = widgets.DOMWidgetView.extend({
        render: function(){

            this.table = $('<table>');
            this.table.css('text-align', 'center')
                      .css('border-collapse', 'separate')
                      .css('border-spacing', '2px 2px');

            this.n = 0;
            this.normed = this.model.get('normed');
            this.model.on('change:normed', this.normedChanged, this);
            this.labels = this.model.get('labelsStr').split('|');
            this.model.on('change:labelsStr', this.labelsChanged, this);
            this.data = null;

            this.truth_H = null;
            this.prediction_H = null;
            this.true_headers = [];
            this.pred_headers = [];
            this.cells = [[]];
            this.table.appendTo(this.el);
            this.dataChanged();
            this.model.on('change:dataStr', this.dataChanged, this);

        },

        setupTable: function(n){
            const self = this;

            this.n = n;
            const N = n.toString();
            this.true_headers = [];
            this.pred_headers = [];
            this.cells = [];

            function recursive_remove(ele){
                const c = $(ele).contents();
                c.each(()=>{recursive_remove(this); $(this).remove();});
            }
            recursive_remove(this.table);
            this.prediction_H = $('<th>').text('prediction')
                                        .css('font-variant', 'small-caps')
                                        .css('font-family', 'Sans-serif')
                                        .css('font-weight', this.normed==='pred'?'bold':'normal')
                                        .css('cursor', 'pointer')
                                        .css('text-align', 'center')
                                        .css('font-size', '12px')
                                        .css('color', '#444')
                                        .css('color', '#444')
                                        .attr('colspan', N)
                                        .click(()=>{self.setNormed(self.normed==='pred'?'none':'pred')});
            this.table.append($('<tr>').append($('<td>')).append($('<td>')).append(this.prediction_H));
            const pred_tr = $('<tr>').append($('<td>')).append($('<td>'));
            for(let i=0; i<n; i++){
                const l = this.labels.length > i ? this.labels[i] : '';
                let cell = $('<th>').text(l)
                                    .css('text-align', 'center')
                                    .css('background-color', '#eee')
                                    .css('min-width', '30px')
                                    .css('padding', '5px 10px 1px 10px')
                                    .css('color', '#333');
                this.pred_headers.push(cell);
                pred_tr.append(cell);
                this.true_headers.push(cell.clone());
            }
            this.table.append(pred_tr);


            for(let i=0; i<n; i++){
                const cellsList = [];
                const tr = $('<tr>');
                if(i===0){
                    this.truth_H = $('<th>').text('truth')
                                           .css('font-variant', 'small-caps')
                                           .css('font-family', 'Sans-serif')
                                           .css('font-weight',  this.normed==='true'?'bold':'normal')
                                           .css('cursor', 'pointer')
                                           .css('font-size', '12px')
                                           .css('color', '#444')
                                           .css('transform', 'rotate(-90deg)')
                                           .css('width', '20px')
                                           .attr('rowspan', N)
                                           .click((ev)=>{self.setNormed(self.normed==='true'?'none':'true')});
                    tr.append(this.truth_H);
                }

                tr.append(this.true_headers[i]);
                for(let j=0; j<n; j++){
                    const td = $('<td>').css('padding', '5px 10px 1px 10px');
                    tr.append(td);
                    cellsList.push(td);
                }
                this.cells.push(cellsList);
                this.table.append(tr);
            }
        },

        dataChanged: function(){
            const dataStr = this.model.get('dataStr');
            this.data = dataStr.split(';').map((r) => r.split(',').map((c) => parseInt(c)));

            if(this.data.length !== this.n)
                this.setupTable(this.data.length);

            this.formatData();
        },

        labelsChanged: function(){
            this.labels = this.model.get('labelsStr').split('|');

            if(this.labels.length !== this.n){
                this.setupTable(this.labels.length);
                this.formatData();
            } else {
                for(let i=0; i<this.n; i++){
                    this.true_headers[i].html(this.labels[i]);
                    this.pred_headers[i].html(this.labels[i]);
                }
            }
        },

        normedChanged: function(){
            this.normed = this.model.get('normed');
            this.truth_H.css('font-weight',  this.normed==='true'?'bold':'normal');
            this.prediction_H.css('font-weight',  this.normed==='pred'?'bold':'normal');
            this.formatData();
        },

        setNormed: function(n){
            this.model.set('normed', n);
        },

        formatData: function(){
            let sums = new Array(this.n).fill(0);
            if(this.normed==='true'){
                for(let i=0; i<this.data.length; i++){
                    for(let j=0; j<this.data[i].length; j++){
                        sums[i] += this.data[i][j];
                    }
                }
            }else if(this.normed==='pred'){
                for(let i=0; i<this.data.length; i++){
                    for(let j=0; j<this.data[i].length; j++){
                        sums[j] += this.data[i][j];
                    }
                }
            }else{
                for(let i=0; i<this.data.length; i++){
                    for(let j=0; j<this.data[i].length; j++){
                        sums[i] += this.data[i][j];
                        sums[j] += this.data[i][j];
                    }
                }
            }

            for(let i=0; i<this.n; i++){
                const rowData = this.data.length > i ? this.data[i] : [];
                for(let j=0; j<this.n; j++){
                    const d = rowData.length > j ? rowData[j] : 0;
                    const cell = this.cells[i][j];

                    let t = d.toString();
                    const sum = this.normed==='true' ? sums[i] : sums[j];
                    const f = d/sum;
                    if(this.normed!=='none'){
                        t = (f*100).toFixed(2)+'<small>%</small>';
                    }
                    cell.html(t);

                    let r = 255-f*(255-179);
                    let g = 255-f*(255-39);
                    let b = 255-f*(255-39);
                    if (i === j) {
                        r = 255-f*(255-163);
                        g = 255-f*(255-209);
                        b = 255-f*(255-76);
                    }
                    cell.css('background-color', 'rgb('+r.toString()+','+g.toString()+','+b.toString()+')');
                }
            }
        }
            
    });
    
    return {
        ConfMatrix: ConfMatrix,
    };

}); 

