"""
    Данный модуль содержит вторую версию функций по распознаванию лиц на фото в основе которых лежит deepface
"""
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from deepface import DeepFace
from deepface.commons import functions, distance as dst
from tqdm import tqdm

import tool_module
from tool_module import get_all_file_in_directory

# список самых популярных детекторов лиц
DETECTOR_BACKEND = ['retinaface', 'mtcnn', 'opencv', 'ssd', 'dlib', 'mediapipe']
# список самых популярных моделей распознавания лиц
MODELS = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
# доступные показатели расстояния
DISTANCE_METRIC = ['cosine', 'euclidean', 'euclidean_l2']


def all_df_encodings_to_file(paths_to_photos, file_name='encodings.json', model_name=MODELS[1],
                             detector_backend=DETECTOR_BACKEND[0], enforce_detection=True):
    """
        Записывает кодировки лиц(deepface) для всех фото из paths_to_photos в файл file_name в формате json
    """
    name_start = file_name.rfind("/") + 1
    file_name = file_name[:name_start] + f'dfv2_{model_name.lower()}_{file_name[name_start:]}'
    if not os.path.exists(file_name):
        with Path(file_name).open('x', encoding="utf8") as f:
            data = []
            target_size = functions.find_target_size(model_name=model_name)
            align = True

            for path in tqdm(paths_to_photos):
                if os.path.exists(path):
                    try:
                        img_objs = functions.extract_faces(img=path,
                                                           target_size=target_size,
                                                           detector_backend=detector_backend,
                                                           grayscale=False,
                                                           enforce_detection=enforce_detection,
                                                           align=align
                                                           )
                        img_encodings = []
                        for img_content, img_region, img_confidence in img_objs:
                            embedding_obj = DeepFace.represent(img_path=img_content
                                                               , model_name=model_name
                                                               , enforce_detection=enforce_detection
                                                               , detector_backend="skip"
                                                               , align=align
                                                               , normalization='base'
                                                               )

                            img_representation = embedding_obj[0]["embedding"]
                            img_encodings.append(img_representation)
                    except ValueError as ve:
                        print(ve)
                        img_encodings = []
                    data.append({"path": str(path),
                                 "encodings": img_encodings})

            json.dump(data, f, indent=4)
            return data

    else:
        raise Exception(f'File "{file_name}" already exists')


def group_similar_faces(encodings_file, result_file='dfv2_result.json', model_name=None, threshold=None,
                        distance_metric=DISTANCE_METRIC[0], disable=False):
    """
       Группирует схожие фото из файла с их кодировками(encodings_file). Результат записывает в json-файл result_file
       в формате:
       {
           "origin": str,          # сравниваемый файл
           "similar": list[str]    # список схожих файлов
       }
       Сравнение идет по всем лицам которые были распознаны на сравниваемом фото.
       model_name - должна совпадать с той моделью которая использовалась при создании кодировок, если model_name задан,
                    то threshold рассчитывается автоматически на основании model_name и distance_metric.
       threshold - точность, пороговое значение для расстояния. Лица расстояния между которыми меньше threshold
                   считаются похожими.
   """
    if not model_name and not threshold:
        raise ValueError('model_name or threshold must be specified')
    with Path(encodings_file).open(encoding="utf8") as ef, Path(result_file).open('x', encoding="utf8") as rf:
        data = json.load(ef)
        result = []
        if not threshold:
            threshold = dst.findThreshold(model_name, distance_metric)

        for current_find_photo in tqdm(data, disable=disable):
            cfp_result = []
            for cfp_encode in current_find_photo.get('encodings'):
                for other_photo in data:
                    if not (other_photo.get('path') in cfp_result) and len(other_photo.get('encodings')) and not (
                            other_photo.get('path') == current_find_photo.get('path')):
                        for other_encode in other_photo.get('encodings'):
                            if distance_metric == 'cosine':
                                distance = dst.findCosineDistance(cfp_encode, other_encode)
                            elif distance_metric == 'euclidean':
                                distance = dst.findEuclideanDistance(cfp_encode, other_encode)
                            elif distance_metric == 'euclidean_l2':
                                distance = dst.findEuclideanDistance(dst.l2_normalize(cfp_encode),
                                                                     dst.l2_normalize(other_encode))
                            else:
                                raise ValueError("Invalid distance_metric passed - ", distance_metric)

                            comparison_result = distance <= threshold
                            if comparison_result:
                                cfp_result.append(other_photo.get('path'))
                                break

            result.append({"origin": current_find_photo.get('path'),
                           "similar": cfp_result})
        json.dump(result, rf, indent=4)


def find_face(k_img, encodings_file, need_show=True, model_name=MODELS[1], detector_backend=DETECTOR_BACKEND[0],
              distance_metric=DISTANCE_METRIC[0]):
    """
        Возвращает список фото из файла с их кодировками encodings_file схожих с k_img
        model_name - должна совпадать с той моделью которая использовалась при создании кодировок
        Если need_show=True результат также отображается на экран
    """
    if os.path.exists(k_img):
        try:
            target_size = functions.find_target_size(model_name=model_name)
            img_objs = functions.extract_faces(img=k_img,
                                               target_size=target_size,
                                               detector_backend=detector_backend,
                                               grayscale=False,
                                               enforce_detection=True,
                                               align=True
                                               )
            # print("Known faces count: ", len(img_objs))
            img_encodings = []
            for img_content, img_region, img_confidence in img_objs:
                embedding_obj = DeepFace.represent(img_path=img_content
                                                   , model_name=model_name
                                                   , enforce_detection=True
                                                   , detector_backend="skip"
                                                   , align=True
                                                   , normalization='base'
                                                   )
                img_representation = embedding_obj[0]["embedding"]
                img_encodings.append(img_representation)
        except ValueError as ve:
            print(ve)
            img_encodings = []
    else:
        raise ValueError(f"File {k_img} doesn't exist")
    with Path(encodings_file).open(encoding="utf8") as ef:
        data = json.load(ef)
        result = []
        threshold = dst.findThreshold(model_name, distance_metric)

        for cfp_encode in tqdm(img_encodings):
            for other_photo in data:
                if not (other_photo.get('path') in result) and len(other_photo.get('encodings')) and not (
                        other_photo.get('path') == k_img):
                    for other_encode in other_photo.get('encodings'):
                        if distance_metric == 'cosine':
                            distance = dst.findCosineDistance(cfp_encode, other_encode)
                        elif distance_metric == 'euclidean':
                            distance = dst.findEuclideanDistance(cfp_encode, other_encode)
                        elif distance_metric == 'euclidean_l2':
                            distance = dst.findEuclideanDistance(dst.l2_normalize(cfp_encode),
                                                                 dst.l2_normalize(other_encode))
                        else:
                            raise ValueError("Invalid distance_metric passed - ", distance_metric)

                        comparison_result = distance <= threshold
                        if comparison_result:
                            result.append(other_photo.get('path'))
                            break

        if need_show:
            for p in tqdm(result):
                try:
                    img_pil = Image.open(p)
                    img_pil.show()
                except Exception as e:
                    print(e)
        return result


def show_recognized_faces(img, detector_backend):
    """Выводит на экран лица распознанные на фото img детектором лиц detector_backend"""
    for ef in DeepFace.extract_faces(img, detector_backend=detector_backend):
        Image.fromarray((ef['face'] * 255).astype(np.uint8)).show()


if __name__ == '__main__':
    print("START")
    start_time = time.time()

    # known_img = 'D:/FOTO/Original photo/Olympus/P720/0154.JPG'
    # known_img = 'D:/FOTO/Original photo/Olympus/P9170480.JPG'
    known_img = 'D:/FOTO/Original photo/Olympus/P1011618.JPG'

    # directory = 'D:/FOTO/Original photo/Olympus'
    # directory = 'D:/FOTO/Finished photo'
    # directory = 'out/photo'
    directory = '../Test_photo/Test_1-Home_photos'

    # all_df_encodings_to_file(tool_module.get_all_file_in_directory(directory), directory + '/encodings.json',
    #                          model_name=MODELS[8])
    group_similar_faces(directory + '/dfv2_facenet512_encodings.json',
                        directory + '/dfv2_facenet512_t(0.2499999999999999)_result.json',
                        model_name=MODELS[8], threshold=0.2499999999999999)

    # find_face(known_img, directory + '/dfv2_facenet512_encodings.json', distance_metric=DISTANCE_METRIC[0],
    #           model_name=MODELS[2])
    # find_face(known_img, directory + '/dfv2_vgg-face_encodings.json', distance_metric=DISTANCE_METRIC[0],
    #           model_name=MODELS[0])
    # find_face(known_img, directory + '/dfv2_facenet_encodings.json', distance_metric=DISTANCE_METRIC[0],
    #           model_name=MODELS[1])
    # find_face(known_img, directory + '/dfv2_openface_encodings.json', distance_metric=DISTANCE_METRIC[0],
    #           model_name=MODELS[3])
    # find_face(known_img, directory + '/dfv2_deepface_encodings.json', distance_metric=DISTANCE_METRIC[0],
    #           model_name=MODELS[4])
    # find_face(known_img, directory + '/dfv2_deepid_encodings.json', distance_metric=DISTANCE_METRIC[0],
    #           model_name=MODELS[5])  # have a problem, need to solve
    # find_face(known_img, directory + '/dfv2_arcface_encodings.json', distance_metric=DISTANCE_METRIC[0],
    #           model_name=MODELS[6])
    # find_face(known_img, directory + '/dfv2_dlib_encodings.json', distance_metric=DISTANCE_METRIC[0],
    #           model_name=MODELS[7])
    # find_face(known_img, directory + '/dfv2_sface_encodings.json', distance_metric=DISTANCE_METRIC[0],
    #           model_name=MODELS[8])

    # show_recognized_faces(known_img, DETECTOR_BACKEND[0])
    # print(
    #     DeepFace.verify('../Test_photo/Test_1-Home_photos/Arsen-1.JPG',
    #                     '../Test_photo/Test_1-Home_photos/Varl-4.jpg',
    #                     detector_backend=DETECTOR_BACKEND[0],
    #                     model_name=MODELS[0]))
    end_time = time.time()
    print("time in second: ", end_time - start_time, "\ntime in min: ", (end_time - start_time) / 60)
