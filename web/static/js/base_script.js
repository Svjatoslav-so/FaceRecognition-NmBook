/*--------------------- Preloader -----------------------*/
window.onload = function () {
    document.body.classList.add('loaded_hiding');
    window.setTimeout(function () {
        document.body.classList.add('loaded');
        document.body.classList.remove('loaded_hiding');
    }, 500);
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