"""
    Данный модуль содержит функции по распознаванию лиц на фото в основе которых лежит deepface
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from deepface import DeepFace

from tool_module import get_all_file_in_directory

# список самых популярных детекторов лиц
DETECTOR_BACKEND = ['retinaface', 'mtcnn', 'opencv', 'ssd', 'dlib']
# список самых популярных моделей распознавания лиц
MODELS = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
# доступные показатели расстояния
DISTANCE_METRIC = ['cosine', 'euclidean', 'euclidean_l2']


def face_verify(img1, img2):
    """Сравнение двух фото"""
    try:
        result_dict = DeepFace.verify(img1_path=str(img1), img2_path=str(img2))
        return result_dict
    except Exception as e:
        print(e)


def group_similar_faces(path_to_photos, result_file='result.json', tolerance=0.4, model_name='VGG-Face',
                        distance_metric='cosine', detector_backend='mtcnn'):
    """
            Группирует схожие фото из списка path_to_photos. Результат записывает в json-файл result_file
            в формате:
            {
                "origin": str,          # сравниваемый файл
                "similar": list[str]    # список схожих файлов
            }
            tolerance - точность, чем меньше тем точнее
            model_name - модель распознавания лиц
            detector_backend - детектор лиц
            distance_metric - показатель расстояния
        """
    all_photos_paths = get_all_file_in_directory(path_to_photos, ['.jpg', '.JPG', '.jpeg'])

    with Path(result_file).open('x', encoding="utf8") as rf:
        result = []
        i = 0
        for path in all_photos_paths:
            print(f"{i} {path}")
            current_result = []
            try:
                for face_results in DeepFace.find(img_path=path, db_path=path_to_photos, model_name=model_name,
                                                  distance_metric=distance_metric, detector_backend=detector_backend):
                    for r in face_results.values.tolist():
                        if not(r[0] in current_result) and r[5] < tolerance:
                            current_result.append(r[0])
            except Exception as e:
                print(e)

            result.append({"origin": path,
                           "similar": current_result})
            i += 1
        json.dump(result, rf, indent=4)


def find_face(k_img, db_path, need_show=True, tolerance=0.4, detector_backend=DETECTOR_BACKEND[1],
              model_name=MODELS[0]):
    """
        Возвращает список фото из директории db_path схожих с k_img
        с точностью tolerance, детектором лиц detector_backend и моделью model_name
        Если need_show=True результат также отображается на экран
    """
    try:
        num = 0
        result = DeepFace.find(img_path=k_img, db_path=db_path, detector_backend=detector_backend,
                               model_name=model_name)
        if need_show:
            for f_rs in result:
                for r in f_rs.values.tolist():
                    print(f'{num} --- {r}')
                    if r[5] < tolerance:
                        img_pil = Image.open(r[0])
                        img_pil.show()
                    num += 1
        return result
    except Exception as e:
        print(e)


def show_recognized_faces(img, detector_backend):
    """Выводит на экран лица распознанные на фото img детектором лиц detector_backend"""
    for ef in DeepFace.extract_faces(img, detector_backend=detector_backend):
        Image.fromarray((ef['face'] * 255).astype(np.uint8)).show()


if __name__ == '__main__':
    print("START")
    start_time = time.time()

    known_img = 'D:/FOTO/Original photo/Olympus/P9120310.JPG'
    directory = 'D:/FOTO/Original photo/Olympus'

    # img = Image.open(known_img)
    # fig = plt.figure(figsize=(6, 4))
    # ax = fig.add_subplot()
    # ax.imshow(img)
    # plt.show()

    group_similar_faces(directory, directory + '/df-result.json', detector_backend=DETECTOR_BACKEND[0])

    # show_recognized_faces(known_img, DETECTOR_BACKEND[0])
    # find_face(known_img, directory, detector_backend=DETECTOR_BACKEND[0], model_name=MODELS[1])

    end_time = time.time()
    print("time in second: ", end_time - start_time, "\ntime in min: ", (end_time - start_time) / 60)
