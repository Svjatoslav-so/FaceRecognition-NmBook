console.log('START')

let viewer = document.getElementById('viewer');
let groupMenu = document.getElementById('group_list');
let originPhotoBlock = document.getElementById('origin_photo_block');
let file_loader_form = document.getElementById('file_loader');
let file_name_input = document.getElementById('file_input');
let load_btm = document.getElementById('load_file_btn');
let groupsList;
let metadata;

let getPhotoName = function(path){
    return path.split(/\\\\|\//).reverse()[0];
}

let originPhotoShow = function(index=0){
    if (groupsList.length > 0){
        let origin_path = groupsList[index]['origin'];
        let filename = getPhotoName(origin_path);
        originPhotoBlock.innerHTML = `
        <figcaption>
            <p class="person_name">${filename.slice(0,10)} . . . ${filename.slice(30)}<br>${metadata['by_photo'][filename]['title']}</p>
            <p class="document_id">id: ${metadata['by_photo'][getPhotoName(origin_path)]['docs']}</p>    
        </figcaption>
        <img src="static/${origin_path}" alt="Искомое фото">`;
    }
    // ${getPhotoName(origin_path)}<br>
}

let groupMenuShow = function(active=0){
    document.getElementById('all_group_count').innerHTML = `всего: ${groupsList.length}`;
    for(let i = 0; i < groupsList.length; i++){
        let newGroupLi = document.createElement('li');
        newGroupLi.className = `group_li ${i==active ? "active" : ""}`;
        newGroupLi.innerHTML=`
        <p>Группа ${i+1}</p>
        <p>${groupsList[i]['group'].length} фото</p>`;
        newGroupLi.dataset.group_id = i;
        newGroupLi.onclick =chooseGroup;
        groupMenu.appendChild(newGroupLi);
    }
}

let groupShow = function(index=0){
    viewer.innerHTML = "";
    for(let i = 0; i < groupsList[index]['group'].length; i++){
        let path = groupsList[index]['group'][i];
        let filename = getPhotoName(path);
        let newSimilarPhoto = document.createElement('figure');
        newSimilarPhoto.className = 'similar_figure';
        newSimilarPhoto.innerHTML = `
        <figcaption>
            <p class="person_name">${filename.slice(0,10)} . . . ${filename.slice(30)}<br>${metadata['by_photo'][filename]['title']}</p>
            <p class="document_id">id: ${metadata['by_photo'][getPhotoName(path)]['docs']}</p>    
        </figcaption>
        <img src="static/${path}" alt="Схожее фото">`;
        viewer.appendChild(newSimilarPhoto);
    }
}

let fileShow = function(Request){
    // console.log('RESPONSE: ', Request.responseText);
    let response = JSON.parse(Request.response);
    groupsList = response['group_list'];
    metadata = response['metadata'];
    console.log('groupsList: ', groupsList);
    console.log('metadata: ', metadata);
    originPhotoShow();
    groupMenuShow();
    groupShow();

}

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


function CreateRequest()
{
    var Request = false;

    if (window.XMLHttpRequest)
    {
        //Gecko-совместимые браузеры, Safari, Konqueror
        Request = new XMLHttpRequest();
    }
    else if (window.ActiveXObject)
    {
        //Internet explorer
        try
        {
             Request = new ActiveXObject("Microsoft.XMLHTTP");
        }    
        catch (CatchException)
        {
             Request = new ActiveXObject("Msxml2.XMLHTTP");
        }
    }
 
    if (!Request)
    {
        alert("Невозможно создать XMLHttpRequest");
    }
    
    return Request;
} 

/*
Функция посылки запроса к файлу на сервере
r_method  - тип запроса: GET или POST
r_path    - путь к файлу
r_args    - аргументы вида a=1&b=2&c=3...
r_handler - функция-обработчик ответа от сервера
*/
function SendRequest(r_method, r_path, r_args, r_handler)
{
    //Создаём запрос
    var Request = CreateRequest();
    
    //Проверяем существование запроса еще раз
    if (!Request)
    {
        return;
    }
    
    //Назначаем пользовательский обработчик
    Request.onreadystatechange = function()
    {
        //Если обмен данными завершен
    if (Request.readyState == 4)
    {
        if (Request.status == 200)
        {
            //Передаем управление обработчику пользователя
            // alert("Передаем управление обработчику пользователя")
            r_handler(Request);
        }
        else
        {
            //Оповещаем пользователя о произошедшей ошибке
            alert("Оповещаем пользователя о произошедшей ошибке")
        }
    }
    else
    {
        //Оповещаем пользователя о загрузке
        // alert("Оповещаем пользователя о загрузкее")
    }
    }
    
    //Проверяем, если требуется сделать GET-запрос
    if (r_method.toLowerCase() == "get" && r_args.length > 0)
    r_path += "?" + r_args;
    
    //Инициализируем соединение
    Request.open(r_method, r_path, true);
    
    if (r_method.toLowerCase() == "post")
    {
        //Если это POST-запрос
        
        //Устанавливаем заголовок
        Request.setRequestHeader("Content-Type","application/x-www-form-urlencoded; charset=utf-8");
        //Посылаем запрос
        Request.send(r_args);
    }
    else
    {
        //Если это GET-запрос
        
        //Посылаем нуль-запрос
        Request.send(null);
    }
} 
