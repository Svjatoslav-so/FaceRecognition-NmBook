"""
    Данный модуль содержит функции по загрузке и отображению файлов(фото)
"""
import json
import math
import multiprocessing
import os
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm


def demo():
    p = Path('out')
    with p.joinpath('metadata.json').open(encoding="utf8") as f:
        metadata = json.load(f)
    for photo_id in metadata['by_photo']:
        info = metadata['by_photo'][photo_id]
        photo_path = p.joinpath('photo', photo_id[0], photo_id)
        print(f"Photo {info['title']} in docs {info['docs']} at path {photo_path}")


def get_optimal_process_count(task_count, process_count=None):
    """
        Высчитывает оптимальное для task_count (количества задач), количество параллельных процессов.
        process_count - желаемое количество процессов.
        Возвращает кортеж вида: (количество процессов, количество задач на один процесс)
    """
    if not process_count:
        process_count = multiprocessing.cpu_count()
    if process_count > task_count:
        process_count = task_count
    data_block_size = math.ceil(task_count / process_count)
    process_count = math.ceil(task_count / data_block_size)
    print(f"process_count: {process_count}, data_block_size: {data_block_size}")
    return process_count, data_block_size


def get_all_photo_in_directory(start_directory, pattern='*.jpg'):
    """ Возвращает список файлов удовлетворяющих pattern в директории start_directory"""
    path_list = Path(start_directory).glob(pattern)
    return [str(p) for p in path_list]


def get_all_file_in_directory(start_directory: str, file_extension: list[str] | None = None) -> list[str]:
    """
        Возвращает список файлов(с расширениями из списка file_extension)
        в директории start_directory и ее подкаталогах
    """
    if file_extension is None:
        file_extension = ['.jpg', '.JPG', '.jpeg']
    all_files = []
    for r, d, f in os.walk(start_directory):  # r=root, d=directories, f = files
        for file in f:
            if os.path.splitext(file)[1] in file_extension:
                exact_path = r + "/" + file
                all_files.append(exact_path)
    return all_files


def get_photo_paths(foto_directory='out'):
    """
        Узко специализированный метод, позволяет получить список фото новомучеников из файла 'metadata.json',
        находящегося в каталоге out - foto_directory
    """
    p = Path(foto_directory)
    with p.joinpath('metadata.json').open(encoding="utf8") as f:
        metadata = json.load(f)
        return [p.joinpath('photo', photo_id[0], photo_id) for photo_id in metadata['by_photo']]


def show_photo(photo_paths):
    """ Открывает фото из списка photo_paths в стандартной утилите отображения """
    images = []
    for path in tqdm(photo_paths, desc='show_photo'):
        img_pil = Image.open(path)
        images.append(img_pil)
        img_pil.show()


def get_all_unique_photo_group(file_name='result.json', disable=False):
    """ Из файла file_name с результатами группировки схожих фото возвращает только уникальные группы. """
    result = []
    result_group_sets = []
    if os.path.exists(file_name):
        with open(file_name, encoding="utf8") as f:
            group_list = json.load(f)
            try:
                for g in tqdm(group_list, disable=disable, desc='get_all_unique_photo_group'):
                    if len(g['similar']):
                        new_group = set([g['origin'], ] + list(map(lambda s: s['path'], g['similar'])))
                        # print(new_group, end=" ----- ")
                        if not (new_group in result_group_sets):
                            # print("True")
                            result_group_sets.append(new_group)
                            result.append(g)
            except Exception as e:
                print(f'Error: in tool_module => get_all_unique_photo_group\n\t{e}')
    return result


def show_all_group(file_name='result.json', column=3):
    """
        Отображает уникальные группы фото из файла file_name с результатами группировки схожих фото
        column - количество фото в одной строке
    """
    for g in tqdm(get_all_unique_photo_group(file_name), desc='show_all_group'):
        # Создадим фигуру размером 16 на 8 дюйма
        pic_box = plt.figure(figsize=(18, 9))
        # выводим оригинальное фото
        picture = np.asarray(Image.open(g['origin']))
        pic_box.add_subplot(math.ceil(len(g['similar']) + 1 / column), column, 1)
        plt.title(label="Origin", loc="center")
        plt.imshow(picture)
        plt.axis('off')

        # В переменную i записываем номер итерации
        for i, path in enumerate([s['path'] for s in g['similar']], start=1):
            try:
                # считываем изображение в picture
                picture = np.asarray(Image.open(path))
                # добавляем ячейку в pix_box для вывода текущего изображения
                pic_box.add_subplot(math.ceil(len(g['similar']) + 1 / column), column, i + 1)
                plt.imshow(picture)
                # отключаем отображение осей
                plt.axis('off')
            except Exception as e:
                print(e)
        # выводим все созданные фигуры на экран
        plt.show()


def result_analyzer(file_name='result.json', disable=False):
    """
        Анализирует уникальные группы фото из файла file_name и возвращает словарь вида:
            {
                'count_of_group': количество уникальных групп,
                'photo_count_list': список количества фото в каждой группе,
                'error_count_list': список количества ошибок в каждой группе,
            }
        ВАЖНО: Для корректной работы метода необходимо чтобы название фото было составлено из имен лиц,
        изображенных на нем, разделенных &. Если необходим уникальный идентификатор, то он добавляется в конец через -:
        Например: FaceName1&FaceName2&FaceName3-UniqueId.JPG
    """
    unique_groups = get_all_unique_photo_group(file_name, disable=True)
    num_of_errors_in_groups = []
    num_of_photos_in_groups = []
    num_of_correct_in_groups = []
    for group in tqdm(unique_groups, disable=disable, desc='result_analyzer'):
        origin_name_start = group['origin'].rfind("/") + 1
        origin_face_set = set(group['origin'][origin_name_start:].split('-')[0].split('&'))
        # print(origin_face_set, end='\t\t\t')
        errors = 0
        for photo in map(lambda s: s['path'], group['similar']):
            photo_name_start = photo.rfind("/") + 1
            photo_face_set = set(photo[photo_name_start:].split('-')[0].split('&'))
            # print(f' U {photo_face_set}', end='')
            if not origin_face_set.intersection(photo_face_set):
                errors += 1
        num_of_photos_in_groups.append(len(group['similar']))
        num_of_errors_in_groups.append(errors)
        # print()
        num_of_correct_in_groups = list(map(lambda x, y: x - y, num_of_photos_in_groups, num_of_errors_in_groups))
    report = {
        'count_of_group': len(unique_groups),
        'photo_count_list': num_of_photos_in_groups,
        'error_count_list': num_of_errors_in_groups,
        'correct_count_list': num_of_correct_in_groups
    }
    return report


def statistic_analyzer(report):
    """Расширенная аналитика результатов полученных в методе result_analyzer"""
    if report['count_of_group'] > 0:
        return {
            'max_error_in_group': max(report['error_count_list']),
            'total_errors': sum(report['error_count_list']),
            'median_error': sorted(report['error_count_list'])[report['count_of_group'] // 2],
            'average_error': sum(report['error_count_list']) / report['count_of_group'],
            'max_correct_in_group': max(report['correct_count_list']),
            'median_correct': sorted(report['correct_count_list'])[report['count_of_group'] // 2],
            'average_correct': sum(report['correct_count_list']) / report['count_of_group']
        }
    else:
        return {'max_error_in_group': 0, 'total_errors': 0, 'median_error': 0, 'average_error': 0,
                'max_correct_in_group': 0, 'median_correct': 0, 'average_correct': 0}


if __name__ == '__main__':
    # result_path = 'D:/Hobby/NmProject/nmbook_photo/out/photo/dfv2_facenet512_result.json'
    # result_path = 'D:/Hobby/NmProject/nmbook_photo/out/photo/dfv2_openface_result.json'
    # result_path = 'D:/Hobby/NmProject/nmbook_photo/out/photo/dfv2_sface_result.json'
    # result_path = 'D:/Hobby/NmProject/nmbook_photo/out/photo/dfv2_sface_t(0.32499999999999996)_result.json'
    # result_path = 'D:/Hobby/NmProject/nmbook_photo/out/photo/dfv2_dlib_t(0.02499999999999969)_result.json'
    # result_path = 'D:/Hobby/NmProject/nmbook_photo/out/photo/dfv2_facenet512_t(0.2499999999999999)_result.json'
    # result_path = 'D:/FOTO/Original photo/Olympus/dfv2_facenet512_t(0.2499999999999999)_result.json'
    # result_path = 'D:/FOTO/Original photo/Olympus/dfv2_sface_t(0.32499999999999996)_result.json'
    # result_path = 'D:/FOTO/Original photo/Olympus/dfv2_vgg-face_t(0.12499999999999978)_result.json'
    # result_path = 'D:/FOTO/Original photo/Olympus/dfv2_dlib_t(0.02499999999999969)_result.json'
    # result_path = 'D:/FOTO/Original photo/Olympus/dfv2_facenet512_result.json'
    # result_path = 'D:/FOTO/Original photo/Olympus/dfv2_facenet512_result.json'
    # result_path = '../Test_photo/dfv2_facenet512_result.json'
    # result_path = '../Test_photo/Test_1-Home_photos/dfv2_facenet512(0.2499999999999999)_result.json'
    result_path = '../Test_photo/Test_1-Home_photos/fr_result.json'
    # result_path = '../Test_photo/Telegram_photo_set/dfv2_sface_t(0.32499999999999996)_result.json'
    # result_path = '../Test_photo/Telegram_photo_set/dfv2_facenet512_t(0.2499999999999999)_result.json'

    # show_all_group(result_path)

    model_report = result_analyzer(result_path)
    print(model_report)
    print(statistic_analyzer(model_report))
