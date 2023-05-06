console.log("Smart-viewer-script included successfully");
let threshold_input = document.getElementById('threshold');
let threshold_preview = document.getElementById('threshold_preview');
let group_list = document.getElementById('group_list');
let groups_li = document.getElementsByClassName('group_li');
let origin_photo_block = document.getElementById('origin_photo_block');
let viewer = document.getElementById('viewer');

const options = { click: "toggleCover" };

threshold_input.oninput = function(){
    threshold_preview.innerHTML = threshold_input.value;
}

function reloadSmartViewer(){
    let old_href = location.href;
    if(old_href.includes('?')){
        old_href = old_href.slice(0, old_href.indexOf('?'));
    }
    location.href = `${old_href}?threshold=${threshold_input.value}`;
    
}
threshold_input.onchange = reloadSmartViewer;
document.getElementById('db').onchange = function(){
    console.log('Smart Script');
    SendRequest('get',
                '/set_db/'+ this.value,
                '',
                (Request) => console.log(JSON.parse(Request.response)));
    reloadSmartViewer();
}

for(let i = 0; i < groups_li.length; i++){
    groups_li[i].onclick = function(){loadGroup(this)};
}

/* Активирует Panzoom для всех текущих похожих фото */
let activatePanzoom = function(){
    const containers = document.querySelectorAll(".similar_figure .f-panzoom");
    for(let i = 0;  i < containers.length; i++){
        new Panzoom(containers[i], options);
    }
}

function loadGroup(elem){
    console.log(elem);
    SendRequest('get',
                '/get_group',
                `threshold=${threshold_input.value}&origin_photo_id=${elem.dataset.origin_photoId}&origin_face_id=${elem.dataset.origin_faceId}&origin_photo_title=${elem.dataset.origin_photoTitle}&origin_photo_docs=${elem.dataset.origin_photoDocs}&origin_photo_x1=${elem.dataset.origin_photoX1}&origin_photo_y1=${elem.dataset.origin_photoY1}&origin_photo_x2=${elem.dataset.origin_photoX2}&origin_photo_y2=${elem.dataset.origin_photoY2}`,
                (Request) => {
                    data = JSON.parse(Request.response);
                    console.log(data);
                    origin_photo_block.innerHTML = data['origin_photo_block'];
                    let originPhotoContainer = document.querySelector("#origin_photo_block .f-panzoom");
                    new Panzoom(originPhotoContainer, options);
                    viewer.innerHTML = data['view_panel'];
                    activatePanzoom();
                });
    for(let i = 0; i < groups_li.length; i++){
        if(groups_li[i].className.includes('active')){
            groups_li[i].className = groups_li[i].className.replace(' active', '');
        }
    }
    elem.className += ' active';

}

// on page load
loadGroup(groups_li[0]);

group_list.addEventListener('keydown', function(evt){
    console.log('"Arrow"')
    let active = group_list.querySelector('.active');
    if(evt.key == "ArrowUp" && active.previousElementSibling){
        loadGroup(active.previousElementSibling)
        console.log('"ArrowUp"');
    }
    else if(evt.key == "ArrowDown" && active.nextElementSibling){
        loadGroup(active.nextElementSibling)
        console.log('"ArrowDown"');
    }
}, false);



