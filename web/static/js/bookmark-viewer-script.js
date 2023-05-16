console.log("Bookmark-viewer-script included successfully");
let group_list = document.getElementById('group_list');
let groups_li = document.getElementsByClassName('group_li');
let origin_photo_block = document.getElementById('origin_photo_block');
let viewer = document.getElementById('viewer');
let all_group_count_span = document.getElementById('all_group_count');

const options = { click: "toggleCover" };

document.getElementById('db').onchange = function(){
    console.log('Smart Script');
    SendRequest('get',
                '/set_db/'+ this.value,
                '',
                (Request) => console.log(JSON.parse(Request.response)));
}

for(let i = 0; i < groups_li.length; i++){
    groups_li[i].onclick = function(){loadBookmarkGroup(this)};
}

/* Активирует Panzoom для всех текущих похожих фото */
let activatePanzoom = function(){
    const containers = document.querySelectorAll(".similar_figure .f-panzoom");
    for(let i = 0;  i < containers.length; i++){
        new Panzoom(containers[i], options);
    }
}

function loadBookmarkGroup(elem){
    console.log(elem);
    SendRequest('get',
                '/get_bookmark_group',
                `origin_photo_id=${elem.dataset.origin_photoId}&origin_photo_title=${elem.dataset.origin_photoTitle}&origin_photo_docs=${elem.dataset.origin_photoDocs}&origin_face_x1=${elem.dataset.origin_faceX1}&origin_face_y1=${elem.dataset.origin_faceY1}&origin_face_x2=${elem.dataset.origin_faceX2}&origin_face_y2=${elem.dataset.origin_faceY2}`,
                (Request) => {
                    let data = JSON.parse(Request.response);
                    console.log(data);
                    if(data['status'] == 'ok'){
                        origin_photo_block.innerHTML = data['origin_photo_block'];
                        let originPhotoContainer = document.querySelector("#origin_photo_block .f-panzoom");
                        new Panzoom(originPhotoContainer, options);
                        viewer.innerHTML = data['view_panel'];
                        activatePanzoom();
                    }
                    else{
                        let active_group = group_list.querySelector('.active');
                        let nextGroup = active_group.nextElementSibling || active_group.previousElementSibling;
                        group_list.removeChild(active_group);
                        // обновим groups_li
                        groups_li = document.getElementsByClassName('group_li');
                        // обновим количество групп
                        all_group_count_span.innerText = groups_li.length;
                        // обновим индексы в названиях групп
                        updateGroupNameIndexes(groups_li);

                        if(nextGroup){
                            loadBookmarkGroup(nextGroup);
                        }
                        else{
                            viewer.innerHTML = '<p>У вас еще нет ни одной закладки</p>';
                            origin_photo_block.innerHTML = `<div class="f-panzoom" style="width: 100%;">
                            <img src="/static/img/origin_photo.png" alt="Искомое фото">                                
                            </div>`;
                        }
                    }
                });

    for(let i = 0; i < groups_li.length; i++){
        if(groups_li[i].className.includes('active')){
            groups_li[i].className = groups_li[i].className.replace(' active', '');
        }
    }
    elem.className += ' active';

}

// on page load
if(groups_li[0]){
    loadBookmarkGroup(groups_li[0]);
}
else{
    viewer.innerHTML = '<p>У вас еще нет ни одной закладки</p>';
    origin_photo_block.innerHTML = `<div class="f-panzoom" style="width: 100%;">
                            <img src="/static/img/origin_photo.png" alt="Искомое фото">                                
                            </div>`;
}

group_list.addEventListener('keydown', function(evt){
    console.log('"Arrow"')
    let active = group_list.querySelector('.active');
    if(evt.key == "ArrowUp" && active.previousElementSibling){
        loadBookmarkGroup(active.previousElementSibling)
        console.log('"ArrowUp"');
    }
    else if(evt.key == "ArrowDown" && active.nextElementSibling){
        loadBookmarkGroup(active.nextElementSibling)
        console.log('"ArrowDown"');
    }
}, false);


function updateGroupNameIndexes(list_of_group_li){
    for(let i = 0; i < list_of_group_li.length; i++){
        list_of_group_li[i].innerText = list_of_group_li[i].innerText.replace(/^\d+/, `${i+1} `);
    }
}


/*---------------------- Bookmark -------------------- */
let menu = document.getElementById('right-menu');
let menu_ul = document.getElementById('right-menu-items');
let target_photo = NaN;

// Функция для определения координат указателя мыши
function defPosition(event) {
	let x = y = 0;
	let d = document;
	let w = window;

	if (d.attachEvent != null) { // Internet Explorer & Opera
		x = w.event.clientX + (d.documentElement.scrollLeft ? d.documentElement.scrollLeft : d.body.scrollLeft);
		y = w.event.clientY + (d.documentElement.scrollTop ? d.documentElement.scrollTop : d.body.scrollTop);
	} else if (!d.attachEvent && d.addEventListener) { // Gecko
		x = event.clientX + w.scrollX;
		y = event.clientY + w.scrollY;
	}

	return {x:x, y:y};
}

function showMenu(element, event) {
  // Блокируем всплывание события contextmenu
	event = event || window.event;
	event.cancelBubble = true;

	// Задаём позицию контекстному меню
    let cursorPosition = defPosition(event);
	if(window.innerWidth - cursorPosition.x < menu.offsetWidth){
        cursorPosition.x = cursorPosition.x - menu.offsetWidth;
    }
    if(window.innerHeight - cursorPosition.y < menu.offsetHeight){
        cursorPosition.y = cursorPosition.y - menu.offsetHeight;
    }
    menu.style.top =`${cursorPosition.y}px`;
    menu.style.left =`${cursorPosition.x}px`;

    // В target_photo поместим обект для которого было вызвано меню
    target_photo = element;

	// Показываем собственное контекстное меню
	menu.style.display = 'block';

	// Блокируем всплывание стандартного браузерного меню
	return false;
}

// Закрываем контекстное при клике правой кнопкой по документу
document.oncontextmenu = function(){
	menu.style.display = 'none';
};

// Закрываем контекстное при клике левой кнопкой по документу
document.onclick = function(){
    menu.style.display = 'none';
};

function deleteBookmark(){
    console.log('DELETE BOOKMARK')
    let active_group = document.querySelector('.group_li.active');
    let origin_photo = {"photo_id": active_group.dataset.origin_photoId,
    "x1": active_group.dataset.origin_faceX1,
    "y1": active_group.dataset.origin_faceY1,
    "x2": active_group.dataset.origin_faceX2,
    "y2": active_group.dataset.origin_faceY2,
    "photo_title": active_group.dataset.origin_photoTitle,
    "docs": active_group.dataset.origin_photoDocs};

    let similar_photos = []
    similar_photos.push({"photo_id": target_photo.dataset.photoId,
    "x1": target_photo.dataset.photoX1,
    "y1": target_photo.dataset.photoY1,
    "x2": target_photo.dataset.photoX2,
    "y2": target_photo.dataset.photoY2,
    "photo_title": target_photo.dataset.photoTitle,
    "docs": target_photo.dataset.photoDocs});
    
    SendRequest('get',
                '/delete_bookmark',
                `origin_photo=${JSON.stringify(origin_photo)}&similar_photos=${JSON.stringify(similar_photos)}`,
                (Request) => {
                    let data = JSON.parse(Request.response);
                    console.log(data);
                    if(data['status']=='ok'){
                        target_photo.parentElement.removeChild(target_photo);
                        let other_photos = document.getElementsByClassName('similar_figure');
                        if(other_photos.length == 0){
                            let nextGroup = active_group.nextElementSibling || active_group.previousElementSibling;
                            group_list.removeChild(active_group);
                            // обновим groups_li
                            groups_li = document.getElementsByClassName('group_li');
                            // обновим количество групп
                            all_group_count_span.innerText = groups_li.length;
                            // обновим индексы в названиях групп
                            updateGroupNameIndexes(groups_li);

                            if(nextGroup){
                                loadBookmarkGroup(nextGroup);
                            }
                            else{
                                viewer.innerHTML = '<p>У вас еще нет ни одной закладки</p>';
                                origin_photo_block.innerHTML = `<div class="f-panzoom" style="width: 100%;">
                                <img src="/static/img/origin_photo.png" alt="Искомое фото">                                
                                </div>`;
                            }
                        }
                    }
                });
}

/*---------------------- -------- -------------------- */

