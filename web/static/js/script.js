console.log('START')
let odbCheckbox = document.getElementById('only_different_biographies');
let viewer = document.getElementById('viewer');
let groupMenu = document.getElementById('group_list');
let originPhotoBlock = document.getElementById('origin_photo_block');
let file_loader_form = document.getElementById('file_loader');
let file_name_input = document.getElementById('file_input');
let load_btm = document.getElementById('load_file_btn');
let close_detailed_viewer_btn = document.getElementById('close_detailed_viewer');
let groupsList;
let metadata;
let dataURLs;
let currentSimilar;
const options = { click: "toggleCover" };

/* ------------------- DitailedViewer -------------------------*/
// VARIANT I
// new Panzoom(document.querySelector('#origin-detailed_viewer'), options);
// new Panzoom(document.querySelector('#similar-detailed_viewer'), options);

close_detailed_viewer_btn.onclick = function(){
    detailed_viewer.className += ' hidden';
}

/* Обрабатывает ответ сервера, на просьбу дать origin-фото и текущее similar-фото
   с отрисованными квадратиками вокруг лиц. Если все Ок, то отображает их. */
function showDitailedViewer(Request){
    // console.log('RESPONSE: ', Request.responseText);
    let response = JSON.parse(Request.response);
    detailed_viewer.className = detailed_viewer.className.replace(' hidden', '');
    // VARIANT I
    // document.querySelector('#origin-detailed_viewer img').setAttribute('src', response['origin']);
    // document.querySelector('#similar-detailed_viewer img').setAttribute('src', response['similar']);

    // VARIANT II
    document.querySelector('.detailed_viewer .main_panel').innerHTML = `
            <div onclick="nextSimilar()" id="next_similar">
                <img src="static/img/right-arrow.png" alt="Предыдущий">
            </div>
            <div class="f-panzoom" id="origin-detailed_viewer">
                <img class="f-panzoom__content" src="${response['origin']}" alt="Origin">
            </div>
            <div class="f-panzoom" id="similar-detailed_viewer">
                <img class="f-panzoom__content" src="${response['similar']}" alt="Similar">
            </div>
            <div onclick="previousSimilar()" id="previous_similar">
                <img src="static/img/left-arrow.png" alt="Следующий">
            </div> `;

    new Panzoom(document.querySelector('#origin-detailed_viewer'), options);
    new Panzoom(document.querySelector('#similar-detailed_viewer'), options);
}

/* Загружает с сервера origin-фото и текущее similar-фото
 с отрисованными квадратиками вокруг лиц. Отображает загруженные фото */
function loadFaceAreas(element){
    // console.log('Load Face Areas');
    currentSimilar = element.parentElement;
    SendRequest('get',
                '/get_img_with_face_area',
                `origin_path=${originPhotoBlock.dataset.origin_path}&similar_path=${element.dataset.path}&file_path=${file_name_input.value}`,
                showDitailedViewer);
}

/* Вызывает детальный просмотр фото для соседнего фото справа */
function nextSimilar(){
    let next = currentSimilar.nextElementSibling;
    if(!next){
       next = currentSimilar.parentElement.firstElementChild; 
    }
    loadFaceAreas(next.firstElementChild);
}

/* Вызывает детальный просмотр фото для соседнего фото слева  */
function previousSimilar(){
    let previous = currentSimilar.previousElementSibling;
    if(!previous){
        previous = currentSimilar.parentElement.lastElementChild;
    }
    loadFaceAreas(previous.firstElementChild);
}

/* -------------------------- Group Viewer --------------------------*/

/* Из пути к файлу извлекает его имя */
let getPhotoName = function(path){
    return path.split(/\\\\|\//).reverse()[0];
}

/* Отображает origin фото выбранной группы */
let originPhotoShow = function(index=0){
    if (groupsList.length > 0){
        let origin_path = groupsList[index]['origin'];
        let filename = getPhotoName(origin_path);
        originPhotoBlock.dataset.origin_path = origin_path;
        originPhotoBlock.innerHTML = `
        <div class="caption">
            <p class="person_name">${filename.slice(0,10)} . . . ${filename.slice(30)}<br>${metadata['by_photo'][filename]['title']}</p>
            <p class="document_id">id: ${metadata['by_photo'][filename]['docs']}</p>
        </div>
        <div class="f-panzoom" style="width: 100%;">
            <img class="f-panzoom__content" src="/img_show/${origin_path}" alt="Искомое фото">
        </div>`;
        let originPhotoContainer = document.querySelector("#origin_photo_block .f-panzoom");
        new Panzoom(originPhotoContainer, options);
    }
}

/* Отображает список групп */
let groupMenuShow = function(active=0){
    groupMenu.innerHTML = "";
    document.getElementById('all_group_count').innerHTML = `всего: ${groupsList.length}`;
    for(let i = 0; i < groupsList.length; i++){
        let newGroupLi = document.createElement('li');
        newGroupLi.className = `group_li ${i==active ? "active" : ""}`;
        newGroupLi.innerHTML=`
        <p>Группа ${i+1}</p>
        <p>${groupsList[i]['similar'].length} фото</p>`;
        newGroupLi.dataset.group_id = i;
        newGroupLi.onclick =chooseGroup;
        groupMenu.appendChild(newGroupLi);
    }
}

/* Отвечает за переключение между группами */
let chooseGroup = function(){
    for(let i = 0; i < groupMenu.children.length; i++){
        if(groupMenu.children[i].className.includes('active')){
            groupMenu.children[i].className = 'group_li';
        }
    }
    this.className = 'group_li active';
    groupShow(this.dataset.group_id);
    originPhotoShow(this.dataset.group_id);
}

/* Отображает список похожих фото выбранной группы */
let groupShow = function(index=0){
    viewer.innerHTML = "";
    for(let i = 0; i < groupsList[index]['similar'].length; i++){
        let path = groupsList[index]['similar'][i]['path'];
        let filename = getPhotoName(path);
        let newSimilarPhoto = document.createElement('div');
        newSimilarPhoto.className = 'similar_figure';
        newSimilarPhoto.innerHTML = `
        <div class="caption" onclick="loadFaceAreas(this)" data-path="${path}">
            <p class="person_name">${filename.slice(0,10)} . . . ${filename.slice(30)}<br>${metadata['by_photo'][filename]['title']}</p>
            <p class="document_id">id: ${metadata['by_photo'][getPhotoName(path)]['docs']}</p>    
        </div>
        <div class="f-panzoom">
            <img class="f-panzoom__content" src="/img_show/${path}" alt="Схожее фото">
        </div>`;
        viewer.appendChild(newSimilarPhoto);
    }
        activatePanzoom();
}

/* Активирует Panzoom для всех текущих похожих фото */
let activatePanzoom = function(){
    const containers = document.querySelectorAll(".similar_figure .f-panzoom");
    for(let i = 0;  i < containers.length; i++){
        new Panzoom(containers[i], options);
    }
}

/* Обрабатывает ответ сервера, на просьбу дать файл с результатами.
   Если все Ок, то из json вытягивает groupsList и metadata.
   Затем пропускает groupsList через фильтры и отображает на странице. */
let fileShow = function(Request){
    // console.log('RESPONSE: ', Request.responseText);
    let response = JSON.parse(Request.response);
    groupsList = response['group_list'];
    metadata = response['metadata'];
    if(odbCheckbox.checked){
        filterOnlyFotoFromDifferentBiographies();
    }
    console.log('groupsList: ', groupsList);
    console.log('metadata: ', metadata);
    originPhotoShow();
    groupMenuShow();
    groupShow();
}

/* При клике по load_btm происходит загрузка данных */
load_btm.onclick = function(){
    console.log('BUTTON_CLICK');
    if (file_name_input.value ){
        SendRequest(file_loader_form.getAttribute('method'),
                    file_loader_form.getAttribute('action'),
                    `${file_name_input.getAttribute('name')}=${file_name_input.value}`,
                    fileShow);
    }
    else{
        alert('Input file name');
    }
}

/* При изменении odbCheckbox происходит перезагрузка данных */
odbCheckbox.onchange = function(){
    console.log('ODBCheckbox');
    if (file_name_input.value ){
        SendRequest(file_loader_form.getAttribute('method'),
                    file_loader_form.getAttribute('action'),
                    `${file_name_input.getAttribute('name')}=${file_name_input.value}`,
                    fileShow);
    }
}

/* Фильтрует groupsList. В каждой группе из списка похожих фото удаляет все фото
 у которых id биографии совпадает с origin фото. Если в группе не остается похожих фото,
 то группа удаляется из groupsList */ 
function filterOnlyFotoFromDifferentBiographies(){
    console.log('FILTRED');
    for(let groupIndex = 0; groupIndex < groupsList.length; groupIndex++){
        let originBioIdList = metadata['by_photo'][getPhotoName(groupsList[groupIndex]['origin'])]['docs'];
        // console.log(originBioIdList);
        groupsList[groupIndex]['similar'] = groupsList[groupIndex]['similar'].filter(function(element){
            let similarBioIdList = metadata['by_photo'][getPhotoName(element['path'])]['docs'];
            // console.log(similarBioIdList);
            isSame = false;
            for(id of similarBioIdList){
                isSame = originBioIdList.includes(id);
                if (isSame){
                    // console.log("DELETE ", originBioIdList, similarBioIdList);
                    break;
                }
            return !isSame;
            }

        });
    }
    groupsList = groupsList.filter(function(element){
        return element['similar'].length;
    });
}