"""
    Данный модуль содержит функции по распознаванию лиц на фото в основе которых лежит face_recognition
"""
import os
import time
from pathlib import Path
import json
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from deepface.commons import functions
from face_recognition import face_locations
from tqdm import tqdm

import tool_module
from tool_module import get_all_photo_in_directory


def get_face_locations_by_deepface(known_img_path, need_show=False):
    """
        На фото known_img_path с помощью deepface находит лица и возвращает их координаты в формате face_recognition,
        а именно в виде массива кортежей.
        Если need_show=True, то выведет на экран фото с обведенными распознанными лицами
    """
    try:
        img_objs = functions.extract_faces(img=known_img_path,
                                           target_size=functions.find_target_size(model_name='Facenet512'),
                                           detector_backend='retinaface',
                                           grayscale=False,
                                           enforce_detection=True,
                                           align=True
                                           )
    except ValueError as ve:
        print(ve)
        img_objs = []

    known_face_locations = []
    for img_content, img_region, img_confidence in img_objs:
        known_face_locations.append(
            (img_region['y'], img_region['x'] + img_region['w'], img_region['y'] + img_region['h'], img_region['x']))
    # print(known_face_locations)
    if need_show:
        img_p = Image.open(known_img_path)
        draw1 = ImageDraw.Draw(img_p)
        for (top, right, bottom, left) in known_face_locations:
            draw1.rectangle((left, top, right, bottom), outline=225, width=2)
        img_p.show()
    return known_face_locations


def all_fr_encodings_to_file(paths_to_photos, to_directory='', number_of_times_to_upsample=1, face_location_model="hog",
                             num_jitters=1, face_encoding_model='large'):
    """
        Записывает кодировки лиц(face_recognition) для всех фото из paths_to_photos в файл в папку to_directory
        в формате json:
        {
            "path": str(путь к файлу),
            "faces": {
                "face_area": (список координат лица),
                "encoding": (вектор кодировки лица)
            }
        }
        number_of_times_to_upsample - How many times to upsample the image looking for faces.
                                      Higher numbers find smaller faces.
        face_location_model - Which face detection model to use. "hog" is less accurate but faster on CPUs.
                              "cnn" is a more accurate deep-learning model which is GPU/CUDA accelerated (if available).
                               The default is "hog".
        num_jitters - How many times to re-sample the face when calculating encoding. Higher is more accurate,
                      but slower (i.e. 100 is 100x slower)
        face_encoding_model - Optional - which model to use. "large" (default) or "small" which only returns 5 points
                              but is faster.
    """
    file_name = f'{to_directory}/face_recognition({number_of_times_to_upsample}&{face_location_model}-{num_jitters}&{face_encoding_model})encodings.json'
    if not os.path.exists(file_name):
        with Path(file_name).open('x', encoding="utf8") as f:
            i = 0
            data = []
            for path in tqdm(paths_to_photos):
                img = face_recognition.load_image_file(path)
                if face_location_model == 'cnn':  # на моем компе cnn встроенная в библиотеку вылетает, поэтому так
                    known_face_locations = get_face_locations_by_deepface(path)
                else:
                    known_face_locations = face_locations(img, number_of_times_to_upsample=number_of_times_to_upsample,
                                                          model=face_location_model)
                img_encodings = face_recognition.face_encodings(img, known_face_locations=known_face_locations,
                                                                num_jitters=num_jitters, model=face_encoding_model)
                data.append({"path": str(path),
                             "faces": list(map(lambda fa, e: {"face_area": fa, "encoding": e.tolist()},
                                               known_face_locations,
                                               img_encodings))})
                print(f"{i}/{path}")
                i += 1
            json.dump(data, f, indent=4)
    else:
        raise Exception(f'File "{file_name}" already exists')


def group_similar_faces(encodings_file, result_directory='', tolerance=0.6):
    """
        Группирует схожие фото из файла с их кодировками(encodings_file). Результат записывает в json-файл
        в папке result_directory в формате:
         {
           "origin": str(путь к искомому фото), # сравниваемое фото
           "similar": list[{"path": str(путь к найденному фото),
                           "face_areas": {
                               "origin": list(список координат лица из искомого фото),
                               "similar": list(список координат лица из найденного фото)}
                           }]    # список схожих фото
         }
        Сравнение идет по всем лицам которые были распознаны на сравниваемом фото
        tolerance - точность, чем меньше тем точнее
    """
    result_file = f'{result_directory}/{encodings_file.split("/")[-1].replace("encodings.json","")}result.json'
    with Path(encodings_file).open(encoding="utf8") as ef, Path(result_file).open('x', encoding="utf8") as rf:
        data = json.load(ef)
        result = []
        i = 0
        for c_p_data in tqdm(data):
            print(f"{i} {c_p_data.get('path')}")
            c_p_result = []
            for c_p_face in c_p_data.get('faces'):
                for same_data in data:
                    if not (same_data.get('path') in [cp.get('path') for cp in c_p_result]) \
                            and len(same_data.get('faces')) \
                            and not (same_data.get('path') == c_p_data.get('path')):
                        comparison_result = face_recognition.compare_faces(c_p_face.get('encoding'),
                                                                           np.array([f.get('encoding') for f in
                                                                                     same_data.get('faces')]),
                                                                           tolerance=tolerance)
                        if True in comparison_result:
                            c_p_result.append({"path": same_data.get('path'),
                                               "face_areas": {"origin": c_p_face.get('face_area'), "similar": None}})
            result.append({"origin": c_p_data.get('path'),
                           "similar": c_p_result})
            i += 1
        json.dump(result, rf, indent=4)


def find_similar_faces(known_img_path, encodings_file='encodings.json', tolerance=0.6):
    """Возвращает список фото из файла с их кодировками(encodings_file) схожих с known_img_path с точностью tolerance"""
    known_img = face_recognition.load_image_file(known_img_path)
    # known_face_locations = face_recognition.face_locations(known_img, model='cnn')
    # print(known_face_locations)
    known_face_locations = get_face_locations_by_deepface(known_img_path)
    known_img_encodes = face_recognition.face_encodings(known_img, known_face_locations=known_face_locations,
                                                        model='large')
    if len(known_img_encodes):
        # i = 0
        with Path(encodings_file).open(encoding="utf8") as fr:
            all_data = json.load(fr)
            result_photo_paths = []
            for known_encode in tqdm(known_img_encodes):
                for d in all_data:
                    # print(f"{i} {d.get('path')}", end=" - ")
                    if len(d.get('faces')) > 0:
                        comparison_result = face_recognition.compare_faces(known_encode,
                                                                           np.array([f.get('encoding') for f in
                                                                                     d.get('faces')]),
                                                                           tolerance=tolerance)
                        # print(comparison_result)
                        if True in comparison_result and not (d.get('path') in result_photo_paths) \
                                and not (Path(d.get('path')) == Path(known_img_path)):
                            result_photo_paths.append(d.get('path'))
                        else:
                            pass
                            # print("no faces in the photo")
        return result_photo_paths
    else:
        raise Exception(f'Failed to recognize a face in a known photo "{known_img_path}"')


if __name__ == '__main__':
    print("START")
    start_time = time.time()

    # k_img_path = 'D:/FOTO/Original photo/Saved Pictures/Фото cкачаные с GooglePhoto/IMG_20200719_192402604.jpg'
    # k_img_path = 'D:/Hobby/NmProject/nmbook_photo/out/photo/9/918a5b2b80b57bd096a6b957251db58403145334.jpeg'
    # k_img_path = 'D:/FOTO/Original photo/Olympus/P7200154.JPG'
    # k_img_path = 'D:/FOTO/Original photo/Olympus/P5111402.JPG'
    # k_img_path = 'D:/FOTO/Original photo/Olympus/P1011618.JPG'
    # k_img_path = 'D:/FOTO/Original photo/Olympus/P9170480.JPG'
    # k_img_path = 'D:/FOTO/Original photo/Moto/photo_2021-08-13_21-37-01.jpg'

    # img = Image.open(k_img_path)
    # img.show()

    # directory = 'D:/FOTO/Original photo/Olympus'
    directory = '../Test_photo/Test_1-Home_photos'
    # directory = 'D:/FOTO/Original photo/Moto'
    # directory = 'D:/Hobby/NmProject/nmbook_photo/web/static/out/photo'
    # directory = 'D:/FOTO/Original photo/Saved Pictures/Фото cкачаные с GooglePhoto'
    # p_paths = get_all_photo_in_directory(directory, '*.jpg')
    p_paths = tool_module.get_all_file_in_directory(directory)
    # p_paths = get_photo_paths()
    c_directory_path = Path(directory)
    # if False:
    #     # similar_faces = find_similar_faces(k_img_path, 'D:/FOTO/Original photo/Olympus/encodings.json')
    #     similar_faces = find_similar_faces(k_img_path, directory + '/fr_encodings.json')
    #     print('Similar faces: ', similar_faces)
    #     tool_module.show_photo(similar_faces)
    # if False:
    #     all_fr_encodings_to_file(p_paths, directory, number_of_times_to_upsample=1, face_location_model="hog",
    #                              num_jitters=1, face_encoding_model='small')
    # if True:
    #     for e in ['/face_recognition(1&cnn-1&large)encodings.json', '/face_recognition(1&hog-1&large)encodings.json',
    #               '/face_recognition(1&cnn-1&small)encodings.json', '/face_recognition(1&hog-1&small)encodings.json']:
    #         group_similar_faces(directory + e, directory, tolerance=0.4)

    for p in p_paths:
        img = face_recognition.load_image_file(p)
        locations = face_locations(img, number_of_times_to_upsample=1, model='hog')
        if locations:
            img_p = Image.open(p)
            draw1 = ImageDraw.Draw(img_p)
            for (top, right, bottom, left) in locations:
                draw1.rectangle((left, top, right, bottom), outline=(250, 0, 0), width=5)
            img_p.show()
        else:
            print(p)

    end_time = time.time()
    print("time in second: ", end_time - start_time, "\ntime in min: ", (end_time - start_time) / 60)
