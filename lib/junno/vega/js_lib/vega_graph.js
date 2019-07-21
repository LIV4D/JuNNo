require.undef('vega_graph');

define('vega_graph', ["@jupyter-widgets/base", "nbextensions/jupyter-vega/index"], function(widgets, vega) {

    console.log('VeGA', vega);
     
    const VegaGraph = widgets.DOMWidgetView.extend({
        render: function () {
            this.uuid = this.model.get('_uuid');

            this.vis = document.createElement("div");
            this.vis.id = 'vegagraph_vis'+this.uuid;
            this.vis.className = "vega-embed";
            this.el.appendChild(this.vis);

            this.width_changed();
            this.model.on('change:_width', this.width_changed, this);
            this.height_changed();
            this.model.on('change:_height', this.height_changed, this);

            this.view = null;
            this.spec_changed();
            this.model.on('change:_spec', this.spec_changed, this);


            // const self = this;
            // this.graph_comm = null;
            // setTimeout(function(){
            //     self.graph_comm = Jupyter.notebook.kernel.comm_manager.new_comm('vegagraph_comm'+self.uuid);
            //     self.graph_comm.on_msg(function(msg){self.recv_msg(msg['content']['data'])});
            // }, 1000);
        },

        spec_changed: function() {
            let spec = this.model.get('_spec');
            if(spec[0] === '{')
                spec = JSON.parse(spec);
            const self = this;
            vega.vegaEmbed('#'+this.vis.id, spec,
                {
                    'defaultStyle': true,
                    'actions': true
                }
                ).then(embeding => {self.view = embeding.view;})
                                         .catch(console.error);
        },

        recv_msg: function(msg){

        },

        width_changed: function() {
            const width = this.model.get('_width');
            this.vis.style.width = width.toString()+'px';
        },
        height_changed: function() {
            const height = this.model.get('_height');
            this.vis.style.height = height.toString()+'px';
        }
    });

    return {
        VegaGraph : VegaGraph,
        };
});
