"""
    Данный модуль содержит функции по загрузке и отображению файлов(фото)
"""
import json
import math
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
    for path in tqdm(photo_paths):
        img_pil = Image.open(path)
        images.append(img_pil)
        img_pil.show()


def get_all_photo_group(file_name='result.json'):
    """
        Из файла file_name с результатами группировки схожих фото извлекает уникальные группы(в виде множеств путей)
        возвращает список кортежей вида: (фото по которому искалась группа, группа(в виде множеств путей))
    """
    with open(file_name, encoding="utf8") as f:
        group_list = json.load(f)
        result = []
        for g in tqdm(group_list):
            if len(g['similar']):
                new_group = set([g['origin'], ] + g['similar'])
                # print(new_group, end=" ----- ")
                if not (new_group in [group for origin, group in result]):
                    # print("True")
                    result.append((g['origin'], new_group))
        return result


def show_all_group(file_name='result.json', column=3):
    """
        Отображает уникальные группы фото из файла file_name с результатами группировки схожих фото
        column - количество фото в одной строке
    """
    for g in tqdm(get_all_photo_group(file_name)):
        # Создадим фигуру размером 16 на 8 дюйма
        pic_box = plt.figure(figsize=(18, 9))
        # В переменную i записываем номер итерации
        for i, path in enumerate(g[1]):
            try:
                # считываем изображение в picture
                picture = np.asarray(Image.open(path))
                # добавляем ячейку в pix_box для вывода текущего изображения
                pic_box.add_subplot(math.ceil(len(g[1]) / column), column, i + 1)
                if path == g[0]:
                    plt.title(label="Origin", loc="center")
                plt.imshow(picture)
                # отключаем отображение осей
                plt.axis('off')
            except Exception as e:
                print(e)
        # выводим все созданные фигуры на экран
        plt.show()


if __name__ == '__main__':
    # result_path = 'D:/Hobby/NmProject/nmbook_photo/out/photo/dfv2_facenet512_result.json'
    # result_path = 'D:/Hobby/NmProject/nmbook_photo/out/photo/dfv2_openface_result.json'
    # result_path = 'D:/Hobby/NmProject/nmbook_photo/out/photo/dfv2_sface_result.json'
    # result_path = 'D:/FOTO/Original photo/Olympus/fr_result.json'
    result_path = 'D:/FOTO/Original photo/Olympus/dfv2_facenet512_result.json'
    show_all_group(result_path)
