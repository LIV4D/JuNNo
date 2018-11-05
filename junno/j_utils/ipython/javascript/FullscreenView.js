#INCLUDE ImageViewer
//  Background panel
var fullscreenPanel = document.getElementById('fullscreenPanel');

if(fullscreenPanel === null){
    fullscreenPanel = document.createElement('div');
    fullscreenPanel.setAttribute("id", "fullscreenPanel");
    fullscreenPanel.style.background = '#000000AA';
    fullscreenPanel.style.top = '0';
    fullscreenPanel.style.left = '0';
    fullscreenPanel.style.width = '100%';
    fullscreenPanel.style.height = '100%';
    fullscreenPanel.style.display = 'none';
    fullscreenPanel.style.position = 'absolute';
    fullscreenPanel.style.zIndex = 10;

    fullscreenPanel.onmousedown = function(e){
        var x = e.x / fullscreenPanel.clientWidth, 
        y = (e.y-100) / (fullscreenPanel.clientHeight-100);
        //console.log(e.offsetX, fullscreenPanel.clientWidth, x, e.offsetY, fullscreenPanel.clientHeight, y);
        if( x<0.05 || x>0.95 || y<0.05 || y>0.95) 
            IPython.FullscreenView.hide();
        
    };

    //  Close Button
    var closeButton = document.createElement('div');
    closeButton.innerHTML = '<i class="fa fa-window-close" aria-hidden="true"></i>';
    closeButton.style.color = "#CCC";
    closeButton.style.fontSize = '25px';
    closeButton.style.right = '25px';
    closeButton.style.top = '125px';
    closeButton.style.position = 'absolute',
    closeButton.onmousedown = function(){IPython.FullscreenView.hide()};

    //  Content
    var content = document.createElement('div');
    content.setAttribute("id", "fullscreenPanel_content");
    content.style.width = '90%';
    content.style.height = '90%';
    content.style.margin = 'auto';
    content.style.display = 'flex';
    content.style.paddingTop = '100px';
    content.style.alignItems = 'center';
    content.style.justifyContent = 'center';
    
    var htmlContent = document.createElement('html');
    htmlContent.style.width = '100%';
    htmlContent.style.height = '100%';
    htmlContent.style.x = '0';
    htmlContent.style.y = '0';
    htmlContent.style.display = 'block';
    content.appendChild(htmlContent);
    
    var imgViewerContent = document.createElement('div');
    imgViewerContent.style.width = '90%';
    imgViewerContent.style.height = '90%';
    imgViewerContent.style.x = '0';
    imgViewerContent.style.y = '0';
    imgViewerContent.style.display = 'none';
    content.appendChild(imgViewerContent);
    
    var imgViewer = ImageViewer(imgViewerContent);
    
    var style = document.createElement('style');
    document.head.appendChild(style);
    style = style.sheet;
    
    style.insertRule(` #fullscreenPanel_content img {
            display: block
            max-width: 80%;
            max-height: 80%;
            width: auto;
            height: auto;
        }`);

    fullscreenPanel.appendChild(closeButton);
    fullscreenPanel.appendChild(content);
    document.getElementById('site').appendChild(fullscreenPanel);
}else{
    var content = document.getElementById('fullscreenPanel_content');
}

IPython.FullscreenView = {
    show: function(){
        fullscreenPanel.style.display = 'block';
    },
    showLoading: function(){
        IPython.FullscreenView.setHTML(`
        <i class="fa fa-spinner fa-spin fa-3x fa-fw"
           style='color: white; font-size: 150px;'></i>
        `);
        IPython.FullscreenView.show();
    },
    hide: function(){
        fullscreenPanel.style.display = 'none';
    },
    setHTML: function(html){
        imgViewerContent.style.display = 'none';
        htmlContent.style.display = 'block';
        htmlContent.innerHTML = html;
        imgViewer.hide();
    },
    setContent: function(content){
        htmlContent.style.display = 'none';
        imgViewerContent.style.display = 'none';
        if(content.startsWith('I ')){
            var imgs_data = content.substr(1).split('||');
            var small_img = imgs_data[0];
            var big_img = imgs_data[1];
            var legend = imgs_data[2];

            imgViewerContent.style.display = 'block';
            imgViewer.load(small_img, big_img, legend);
        }
    }
    
}
