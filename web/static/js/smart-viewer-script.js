console.log("Smart-viewer-script included successfully");
let threshold_input = document.getElementById('threshold');
let threshold_preview = document.getElementById('threshold_preview');
let group_list = document.getElementById('group_list');
let groups_li = document.getElementsByClassName('group_li');
let origin_photo_block = document.getElementById('origin_photo_block');
let viewer = document.getElementById('viewer');
let checkedPhotoList = []; // список выбранных фото (для закладок)
let bookmark_menu = document.getElementById('bookmark_menu');

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
    // console.log('Smart Script');
    SendRequest('get',
                '/set_db/'+ this.value,
                '',
                function(Request){
                    console.log(JSON.parse(Request.response));
                    reloadSmartViewer();
                });
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
                    let data = JSON.parse(Request.response);
                    console.log(data);
                    origin_photo_block.innerHTML = data['origin_photo_block'];
                    let originPhotoContainer = document.querySelector("#origin_photo_block .f-panzoom");
                    new Panzoom(originPhotoContainer, options);
                    viewer.innerHTML = data['view_panel'];
                    activatePanzoom();
                });

    closeBookmarkMenu();

    for(let i = 0; i < groups_li.length; i++){
        if(groups_li[i].className.includes('active')){
            groups_li[i].className = groups_li[i].className.replace(' active', '');
        }
    }
    elem.className += ' active';

}

// on page load
if(groups_li[0]){
    loadGroup(groups_li[0]);
}

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

/*---------------------- Bookmark -------------------- */
let menu = document.getElementById('right-menu');
let menu_ul = document.getElementById('right-menu-items');
let target_photo = NaN;
let checked_photo_count_span = document.getElementById('checked-photo-count');
let close_bookmark_menu_btn = document.getElementById('close-bookmark_menu-btn');
let add_bookmark_btn = document.getElementById('add-bookmark-btn');
let delete_bookmark_btn = document.getElementById('delete-bookmark-btn');


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

    // Наполним меню содержмым(в соответствии с контекстом)
    menu_ul.innerHTML='';
    let newli = document.createElement('li');
    if(checkedPhotoList.includes(target_photo)){
        newli.innerHTML = `<li>
        <button type="button" onclick="deselectPhoto()">
            <span>Отменить</span>
        </button>
    </li>`;
    }
    else{
        newli.innerHTML = `<li>
        <button type="button" onclick="selectPhoto()">
            <span>Выделить</span>
        </button>
    </li>`;
    }
    menu_ul.appendChild(newli);

	// Показываем собственное контекстное меню
	menu.style.display = 'block';

	// Блокируем всплывание стандартного браузерного меню
	return false;
}

// Закрываем контекстное при клике правой кнопкой по документу
document.oncontextmenu = function(){
	menu.style.display = 'none';
    target_photo = NaN;
};

// Закрываем контекстное при клике левой кнопкой по документу
document.onclick = function(){
    menu.style.display = 'none';
    target_photo = NaN;
};

// Закрываем bookmark_menu и отменяем выделения
close_bookmark_menu_btn.onclick = closeBookmarkMenu;
function closeBookmarkMenu(){
    checkedPhotoList = []; // очищаем список выбранных фото (для закладок)
    bookmark_menu.className = bookmark_menu.className.replace(' active', '');
    let all_selected = document.querySelectorAll('.similar_figure.selected');
    for(let i = 0; i < all_selected.length; i++)
    {
        all_selected[i].className = all_selected[i].className.replace(' selected', ''); 
    }
}


function selectPhoto(){
    if(!bookmark_menu.className.includes(' active'))
    { 
        bookmark_menu.className += ' active';
    }
    checkedPhotoList.push(target_photo);
    target_photo.className += ' selected';
    target_photo = NaN;
    checked_photo_count_span.innerText = checkedPhotoList.length;

    menu.style.display = 'none';
};

function deselectPhoto(){
    for(let i = 0; i < checkedPhotoList.length; i++){
        if(checkedPhotoList[i] == target_photo){
            checkedPhotoList.splice(i, 1);
            break;
        }
    }
    target_photo.className = target_photo.className.replace(' selected', '');
    target_photo = NaN;
    checked_photo_count_span.innerText = checkedPhotoList.length;
    if(checkedPhotoList.length == 0){
        bookmark_menu.className = bookmark_menu.className.replace(' active', '');
    }

    menu.style.display = 'none';
};

add_bookmark_btn.onclick = addBookmark;

function addBookmark(){
    let active_group = document.querySelector('.group_li.active');
    let origin_photo = {"photo_id": active_group.dataset.origin_photoId,
    "x1": active_group.dataset.origin_photoX1,
    "y1": active_group.dataset.origin_photoY1,
    "x2": active_group.dataset.origin_photoX2,
    "y2": active_group.dataset.origin_photoY2,
    "photo_title": active_group.dataset.origin_photoTitle,
    "docs": active_group.dataset.origin_photoDocs};

    let similar_photos = []
    for(let i = 0; i < checkedPhotoList.length; i++){
        similar_photos.push({"photo_id": checkedPhotoList[i].dataset.photoId,
        "x1": checkedPhotoList[i].dataset.photoX1,
        "y1": checkedPhotoList[i].dataset.photoY1,
        "x2": checkedPhotoList[i].dataset.photoX2,
        "y2": checkedPhotoList[i].dataset.photoY2,
        "photo_title": checkedPhotoList[i].dataset.photoTitle,
        "docs": checkedPhotoList[i].dataset.photoDocs});
    }
    
    SendRequest('get',
                '/add_bookmark',
                `origin_photo=${JSON.stringify(origin_photo)}&similar_photos=${JSON.stringify(similar_photos)}`,
                (Request) => {
                    let data = JSON.parse(Request.response);
                    console.log(data);
                    if(data['status']=='ok'){
                        for(let i = 0; i < checkedPhotoList.length; i++){
                            checkedPhotoList[i].className += ' bookmarked';
                        }
                    }
                    closeBookmarkMenu()
                });
}

delete_bookmark_btn.onclick = deleteBookmark;

function deleteBookmark(){
    console.log('DELETE BOOKMARK')
    let active_group = document.querySelector('.group_li.active');
    let origin_photo = {"photo_id": active_group.dataset.origin_photoId,
    "x1": active_group.dataset.origin_photoX1,
    "y1": active_group.dataset.origin_photoY1,
    "x2": active_group.dataset.origin_photoX2,
    "y2": active_group.dataset.origin_photoY2,
    "photo_title": active_group.dataset.origin_photoTitle,
    "docs": active_group.dataset.origin_photoDocs};

    let similar_photos = []
    for(let i = 0; i < checkedPhotoList.length; i++){
        similar_photos.push({"photo_id": checkedPhotoList[i].dataset.photoId,
        "x1": checkedPhotoList[i].dataset.photoX1,
        "y1": checkedPhotoList[i].dataset.photoY1,
        "x2": checkedPhotoList[i].dataset.photoX2,
        "y2": checkedPhotoList[i].dataset.photoY2,
        "photo_title": checkedPhotoList[i].dataset.photoTitle,
        "docs": checkedPhotoList[i].dataset.photoDocs});
    }
    
    SendRequest('get',
                '/delete_bookmark',
                `origin_photo=${JSON.stringify(origin_photo)}&similar_photos=${JSON.stringify(similar_photos)}`,
                (Request) => {
                    let data = JSON.parse(Request.response);
                    console.log(data);
                    if(data['status']=='ok'){
                        for(let i = 0; i < checkedPhotoList.length; i++){
                            checkedPhotoList[i].className = checkedPhotoList[i].className.replace(' bookmarked', '');
                        }
                    }
                    closeBookmarkMenu()
                });
}

/*---------------------- -------- -------------------- */

