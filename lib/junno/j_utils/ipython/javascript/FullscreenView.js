#INCLUDE ImageViewer
//  Background panel

let fullscreenPanel = document.getElementById('fullscreenPanel');
let self = fullscreenPanel;

const onKeyPress = function(event) {
    console.log('keypress: ', event.code);
    if (fullscreenPanel.style.display === 'none')
        return;

    switch (event.code) {
        case "Escape":
            IPython.FullscreenView.hide();
            break;

        case "PageUp":
            if (IPython.FullscreenView.controlCb !== null)
                IPython.FullscreenView.controlCb('previous');
            break;
        case "PageDown":
            if (IPython.FullscreenView.controlCb !== null)
                IPython.FullscreenView.controlCb('next');
            break;

        case "ArrowLeft":
            if (IPython.FullscreenView.controlCb !== null)
                IPython.FullscreenView.controlCb('left');
            break;
        case "ArrowRight":
            if (IPython.FullscreenView.controlCb !== null)
                IPython.FullscreenView.controlCb('right');
            break;
        case "ArrowTop":
            if (IPython.FullscreenView.controlCb !== null)
                IPython.FullscreenView.controlCb('top');
            break;
        case "ArrowBottom":
            if (IPython.FullscreenView.controlCb !== null)
                IPython.FullscreenView.controlCb('bottom');
            break;
    }
};

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
    fullscreenPanel.style.zIndex = 100;
    self = fullscreenPanel;

    fullscreenPanel.onmousedown = function(e){
        const x = e.x / fullscreenPanel.clientWidth,
        y = (e.y-100) / (fullscreenPanel.clientHeight-100);
        //console.log(e.offsetX, fullscreenPanel.clientWidth, x, e.offsetY, fullscreenPanel.clientHeight, y);
        if( x<0.05 || x>0.95 || y<0.05 || y>0.95) 
            IPython.FullscreenView.hide();
        
    };

    //  Close Button
    const closeButton = document.createElement('div');
    closeButton.innerHTML = '<i class="fa fa-window-close" aria-hidden="true"></i>';
    closeButton.style.color = "#CCC";
    closeButton.style.fontSize = '25px';
    closeButton.style.right = '15px';
    closeButton.style.top = '10px';
    closeButton.style.position = 'absolute';
    closeButton.onmousedown = function(){IPython.FullscreenView.hide()};

    //  Content
    const content = document.createElement('div');
    content.setAttribute("id", "fullscreenPanel_content");
    content.style.width = 'calc(100% - 90px)';
    content.style.height = 'calc(100% - 90px)';
    content.style.left = '45px';
    content.style.top = '45px';
    content.style.position = 'relative';
    content.style.display = 'flex';
    content.style.alignItems = 'center';
    content.style.justifyContent = 'center';
    self.content = content;
    
    const htmlContent = document.createElement('div');
    htmlContent.setAttribute('id', 'fullscreenPanel_htmlContent');
    // htmlContent.style.width = '100%';
    // htmlContent.style.height = '100%';
    // htmlContent.style.position = 'relative';
    // htmlContent.style.x = '0';
    // htmlContent.style.y = '0';
    // htmlContent.style.display = 'block';
    self.htmlContent = htmlContent;
    content.appendChild(htmlContent);
    
    const imgViewerContent = document.createElement('div');
    imgViewerContent.setAttribute('id', 'fullscreenPanel_imgViewerContent');
    imgViewerContent.style.width = '100%';
    imgViewerContent.style.height = '100%';
    imgViewerContent.style.x = '0';
    imgViewerContent.style.y = '0';
    imgViewerContent.style.position = 'relative';
    imgViewerContent.style.display = 'none';
    self.imgViewerContent = imgViewerContent;
    content.appendChild(imgViewerContent);
    
    self.imgViewer = ImageViewer(imgViewerContent);

    let style = document.createElement('style');
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
    self.content = document.getElementById('fullscreenPanel_content');
    self.imgViewerContent = document.getElementById('fullscreenPanel_imgViewerContent');
    self.htmlContent = document.getElementById('fullscreenPanel_htmlContent');
}

document.addEventListener('keypress', onKeyPress, true);


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
        self.style.display = 'none';
    },
    setHTML: function(html, controlCb=null){
        self.imgViewerContent.style.display = 'none';
        self.htmlContent.style.display = 'block';
        self.htmlContent.innerHTML = html;
        self.imgViewer.hide();
        self.controlCb = controlCb;
    },
    setContent: function(content, controlCb=null){
        self.htmlContent.style.display = 'none';
        self.imgViewerContent.style.display = 'none';
        if(content.startsWith('I ')){
            const imgs_data = content.substr(1).split('||');
            const small_img = imgs_data[0];
            const big_img = imgs_data[1];
            const legend = imgs_data[2];

            self.imgViewerContent.style.display = 'block';
            self.imgViewer.load(small_img, big_img, legend);
        }
        self.controlCb = controlCb;
    }
};
