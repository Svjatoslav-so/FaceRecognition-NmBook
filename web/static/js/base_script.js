/*--------------------- Preloader -----------------------*/
window.onload = function () {
    console.log("LOAD");
    document.body.classList.add('loaded_hiding');
    window.setTimeout(function () {
        document.body.classList.add('loaded');
        document.body.classList.remove('loaded_hiding');
    }, 500);
}
window.onbeforeunload = function () {
    console.log("BEFOREUNLOAD");
    document.body.classList.remove('loaded');
}
/*-------------------------------------------------------*/

document.getElementById('db').onchange = function () {
    // console.log('Base Script');
    SendRequest('get',
        '/set_db/' + this.value,
        '',
        (Request) => console.log(JSON.parse(Request.response)));
}



function CreateRequest() {
    var Request = false;

    if (window.XMLHttpRequest) {
        //Gecko-совместимые браузеры, Safari, Konqueror
        Request = new XMLHttpRequest();
    }
    else if (window.ActiveXObject) {
        //Internet explorer
        try {
            Request = new ActiveXObject("Microsoft.XMLHTTP");
        }
        catch (CatchException) {
            Request = new ActiveXObject("Msxml2.XMLHTTP");
        }
    }

    if (!Request) {
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
function SendRequest(r_method, r_path, r_args, r_handler) {
    //Создаём запрос
    var Request = CreateRequest();

    //Проверяем существование запроса еще раз
    if (!Request) {
        return;
    }

    //Назначаем пользовательский обработчик
    Request.onreadystatechange = function () {
        //Если обмен данными завершен
        if (Request.readyState == 4) {
            if (Request.status == 200) {
                //Передаем управление обработчику пользователя
                // alert("Передаем управление обработчику пользователя")
                r_handler(Request);
            }
            else {
                //Оповещаем пользователя о произошедшей ошибке
                alert("Оповещаем пользователя о произошедшей ошибке")
            }
        }
        else {
            //Оповещаем пользователя о загрузке
            // alert("Оповещаем пользователя о загрузкее")
        }
    }

    //Проверяем, если требуется сделать GET-запрос
    if (r_method.toLowerCase() == "get" && r_args.length > 0)
        r_path += "?" + r_args;

    //Инициализируем соединение
    Request.open(r_method, r_path, true);

    if (r_method.toLowerCase() == "post") {
        //Если это POST-запрос

        //Устанавливаем заголовок
        Request.setRequestHeader("Content-Type", "application/x-www-form-urlencoded; charset=utf-8");
        //Посылаем запрос
        Request.send(r_args);
    }
    else {
        //Если это GET-запрос

        //Посылаем нуль-запрос
        Request.send(null);
    }
}

/*-------------------------Comment--------------------------*/

function showComment(elem, event){
    event.stopPropagation(); // предотвращает всплытие события
    elem.style.visibility = "hidden";
    let comment = '';
    let origin_dataset = document.querySelector('.group_li.active').dataset;
    let similar_dataset = elem.parentElement.parentElement.dataset;
    let div_FPanzoom = elem.parentElement;

    SendRequest('get',
                '/get_comment',
                `origin_photo_id=${origin_dataset.origin_photoId}&origin_face_x1=${origin_dataset.origin_faceX1}&origin_face_y1=${origin_dataset.origin_faceY1}&origin_face_x2=${origin_dataset.origin_faceX2}&origin_face_y2=${origin_dataset.origin_faceY2}&similar_photo_id=${similar_dataset.photoId}&similar_face_x1=${similar_dataset.faceX1}&similar_face_y1=${similar_dataset.faceY1}&similar_face_x2=${similar_dataset.faceX2}&similar_face_y2=${similar_dataset.faceY2}`,
                (Request) => {
                    let data = JSON.parse(Request.response);

                    if(data['status']=='ok'){
                        comment = data['comment'];

                        let comment_viewer = document.createElement('div');
                        comment_viewer.className = "comment_viewer";

                        let close_comment_viewer_button = document.createElement('div');
                        close_comment_viewer_button.className = "close_comment";
                        close_comment_viewer_button.onclick = function(evt){
                              evt.stopPropagation(); // предотвращает всплытие события
                            elem.style.visibility = "visible";
                            div_FPanzoom.removeChild(comment_viewer);
                        }

                        let comment_viewer_header = document.createElement('div');
                        comment_viewer_header.className = "comment_viewer__header";
                        comment_viewer_header.innerHTML = "<p>Комментарий к закладке</p>";
                        comment_viewer_header.appendChild(close_comment_viewer_button);

                        let comment_p = document.createElement('p');
                        comment_p.innerText = comment

                        comment_viewer.appendChild(comment_viewer_header);
                        comment_viewer.appendChild(comment_p);

                        div_FPanzoom.appendChild(comment_viewer);
                    }
                });
}

//добавить html для отображения иконки комментария в similar_figure (используется при добавления новой закладки)
function addCommentBlock(elem){
    let div_FPanzoom = null;
    for(let i = 0; i < elem.children.length; i++){
        if (elem.children[i].classList.contains("f-panzoom")){
            div_FPanzoom = elem.children[i]
        }
    }
    if(div_FPanzoom){
        let newCommentButton = document.createElement('button');
        newCommentButton.className = "comment_button";
        newCommentButton.setAttribute("onclick", "showComment(this, event)");
        newCommentButton.setAttribute("type", "button");
        newCommentButton.setAttribute("title", "Показать комментарий к закладке");
        div_FPanzoom.appendChild(newCommentButton);
    }
}

//удалить html иконки комментария из similar_figure (используется при удалении закладки)
function deleteCommentBlock(elem){
    let div_FPanzoom = null;
    for(let i = 0; i < elem.children.length; i++){
        if (elem.children[i].classList.contains("f-panzoom")){
            div_FPanzoom = elem.children[i]
        }
    }
    if(div_FPanzoom){
        for(let i = 0; i < div_FPanzoom.children.length; i++){
            if (div_FPanzoom.children[i].classList.contains("comment_button")){
                div_FPanzoom.removeChild(div_FPanzoom.children[i]);
            }
        }
    }
}