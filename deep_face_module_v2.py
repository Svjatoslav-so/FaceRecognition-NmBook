"""
    Данный модуль содержит вторую версию функций по распознаванию лиц на фото в основе которых лежит deepface
"""
import json
import multiprocessing
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
DETECTOR_BACKEND = ['retinaface', 'mtcnn', 'opencv', 'ssd', 'dlib']
# список самых популярных моделей распознавания лиц
MODELS = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
# доступные показатели расстояния
DISTANCE_METRIC = ['cosine', 'euclidean', 'euclidean_l2']


def _encoding_process(paths, queue, model_name, target_size, detector_backends, enforce_detection, align, disable):
    """ Функция выполняемая отдельными процессами в all_df_encodings_to_file """
    data = []
    for path in tqdm(paths, disable=disable):
        if os.path.exists(path):
            img_objs = []
            for d_backend in detector_backends:
                try:
                    img_objs = functions.extract_faces(img=path,
                                                       target_size=target_size,
                                                       detector_backend=d_backend,
                                                       grayscale=False,
                                                       enforce_detection=enforce_detection,
                                                       align=align)
                    break
                except ValueError as ve:
                    img_objs = []
                    if d_backend == detector_backends[-1]:
                        print(f'{ve} Photo: {path}')

            img_faces = []
            for img_content, img_region, img_confidence in img_objs:
                embedding_obj = DeepFace.represent(img_path=img_content
                                                   , model_name=model_name
                                                   , enforce_detection=enforce_detection
                                                   , detector_backend="skip"
                                                   , align=align
                                                   , normalization='base'
                                                   )
                face_representation = {"face_area": (img_region['y'],
                                                     img_region['x'] + img_region['w'],
                                                     img_region['y'] + img_region['h'],
                                                     img_region['x']),
                                       "encoding": embedding_obj[0]["embedding"]}
                img_faces.append(face_representation)
            data.append({"path": path,
                         "faces": img_faces})
    queue.put(data)  # проблемное место, если вызывать join() перед извлечением данных из queue (get()).
    # queue.cancel_join_thread() # не безопасно использовать, могут быть потеряны данные добавленные в очередь,
    # но при этом действительно процесс не виснет, а завершается.
    # print("@Encoding_process end!!!")


def all_df_encodings_to_file(paths_to_photos, file_name='encodings.json', model_name=MODELS[1],
                             detector_backends=None, enforce_detection=True, process_count=None,
                             disable=False):
    """
        Записывает кодировки лиц(deepface) для всех фото из paths_to_photos в файл file_name в формате json:
        {
            "path": str(путь к файлу),
            "faces": {
                "face_area": (список координат лица),
                "encoding": (вектор кодировки лица)
                }
        }
        model_name - модель используемая для получения кодировки лица.
        detector_backends - список детекторов распознавания лиц. Если не сработает первый, пробуем следующий.
                            Если None, то по умолчанию используются ['retinaface', 'mtcnn'].
        enforce_detection - если False, то для фото без лица не будет выбрасываться ошибка.
        process_count - количество процессов для расчета. (Введенное значение может быть автоматически изменено).
                       По умолчанию None - определяется автоматически(равно количеству логических ядер).
                       (Оптимальное количество 4).
        disable - если True, то прогресс-бар будет отключен.
    """
    if detector_backends is None:
        detector_backends = ['retinaface', 'mtcnn']
    name_start = file_name.rfind("/") + 1
    file_name = file_name[:name_start] + f'dfv2_{model_name.lower()}_{file_name[name_start:]}'
    if not os.path.exists(file_name):
        with Path(file_name).open('x', encoding="utf8") as f:
            all_data = []

            process_list = []
            queue = multiprocessing.Queue()
            process_count, pt_count = tool_module.get_optimal_process_count(len(paths_to_photos), process_count)

            target_size = functions.find_target_size(model_name=model_name)
            align = True

            for i in range(process_count):
                process_list.append(multiprocessing.Process(target=_encoding_process,
                                                            kwargs={'paths': paths_to_photos[
                                                                             i * pt_count: i * pt_count + pt_count],
                                                                    'queue': queue,
                                                                    'model_name': model_name,
                                                                    'target_size': target_size,
                                                                    'detector_backends': detector_backends,
                                                                    'enforce_detection': enforce_detection,
                                                                    'align': align,
                                                                    'disable': disable}))
                process_list[i].start()
            for _ in range(process_count):
                all_data.extend(queue.get())  # данные из очереди нужно извлечь до join()
            for p in process_list:
                p.join()
            print(f"All process finished!")

            json.dump(all_data, f, indent=4)
            return all_data
    else:
        raise Exception(f'File "{file_name}" already exists')


def _group_process(origin_data, all_data, queue, threshold, distance_metric, disable):
    """ Функция выполняемая отдельными процессами в group_similar_faces """
    result = []
    for current_find_photo in tqdm(origin_data, disable=disable):
        cfp_result = []
        for cfp_face in current_find_photo.get('faces'):
            for other_photo in all_data:
                if not (other_photo.get('path') in [cfp.get('path') for cfp in cfp_result]) \
                        and len(other_photo.get('faces')) \
                        and not (other_photo.get('path') == current_find_photo.get('path')):
                    for other_face in other_photo.get('faces'):
                        if distance_metric == 'cosine':
                            distance = dst.findCosineDistance(cfp_face.get('encoding'),
                                                              other_face.get('encoding'))
                        elif distance_metric == 'euclidean':
                            distance = dst.findEuclideanDistance(cfp_face.get('encoding'),
                                                                 other_face.get('encoding'))
                        elif distance_metric == 'euclidean_l2':
                            distance = dst.findEuclideanDistance(dst.l2_normalize(cfp_face.get('encoding')),
                                                                 dst.l2_normalize(other_face.get('encoding')))
                        else:
                            raise ValueError("Invalid distance_metric passed - ", distance_metric)

                        comparison_result = distance <= threshold
                        if comparison_result:
                            cfp_result.append({"path": other_photo.get('path'),
                                               "face_areas": {
                                                   "origin": cfp_face.get('face_area'),
                                                   "similar": other_face.get('face_area')}
                                               })
                            break

        result.append({"origin": current_find_photo.get('path'),
                       "similar": cfp_result})
    queue.put(result)


def group_similar_faces(encodings_file, result_file='dfv2_result.json', model_name=None, threshold=None,
                        distance_metric=DISTANCE_METRIC[0], disable=False, process_count=None):
    """
       Группирует схожие фото из файла с их кодировками(encodings_file). Результат записывает в json-файл result_file
       в формате:
       {
           "origin": str(путь к искомому фото), # сравниваемое фото
           "similar": list[{"path": str(путь к найденному фото),
                           "face_areas": {
                               "origin": list(список координат лица из искомого фото),
                               "similar": list(список координат лица из найденного фото)}
                           }]    # список схожих фото
       }
       Сравнение идет по всем лицам которые были распознаны на сравниваемом фото.
       model_name - должна совпадать с той моделью которая использовалась при создании кодировок, если model_name задан,
                    то threshold рассчитывается автоматически на основании model_name и distance_metric.
       threshold - точность, пороговое значение для расстояния. Лица расстояния между которыми меньше threshold
                   считаются похожими.
       distance_metric - метрика используемая для расчета расстояния между кодировками лиц.
       disable - если True, то прогресс-бар будет отключен.
       process_count - количество процессов для расчета. (Введенное значение может быть автоматически изменено).
                       По умолчанию None - определяется автоматически(равно количеству логических ядер).
   """
    if not model_name and not threshold:
        raise ValueError('model_name or threshold must be specified')
    with Path(encodings_file).open(encoding="utf8") as ef, Path(result_file).open('x', encoding="utf8") as rf:
        data = json.load(ef)
        result = []
        if not threshold:
            threshold = dst.findThreshold(model_name, distance_metric)

        process_list = []
        queue = multiprocessing.Queue()
        process_count, pt_count = tool_module.get_optimal_process_count(len(data), process_count)

        if process_count == 1:
            _group_process(data, data, queue, threshold, distance_metric, disable)
            result = queue.get()
        else:
            for i in range(process_count):
                process_list.append(multiprocessing.Process(target=_group_process,
                                                            kwargs={
                                                                'origin_data': data[
                                                                               i * pt_count: i * pt_count + pt_count],
                                                                'all_data': data,
                                                                'queue': queue,
                                                                'threshold': threshold,
                                                                'distance_metric': distance_metric,
                                                                'disable': disable}))
                process_list[i].start()
            for _ in range(process_count):
                result.extend(queue.get())  # данные из очереди нужно извлечь до join()
            print("All data received!")
            for p in process_list:
                p.join()
            print("All process finished!")
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
                if not (other_photo.get('path') in result) and len(other_photo.get('faces')) and not (
                        other_photo.get('path') == k_img):
                    for other_face in other_photo.get('faces'):
                        if distance_metric == 'cosine':
                            distance = dst.findCosineDistance(cfp_encode, other_face.get('encoding'))
                        elif distance_metric == 'euclidean':
                            distance = dst.findEuclideanDistance(cfp_encode, other_face.get('encoding'))
                        elif distance_metric == 'euclidean_l2':
                            distance = dst.findEuclideanDistance(dst.l2_normalize(cfp_encode),
                                                                 dst.l2_normalize(other_face.get('encoding')))
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


def show_recognized_faces(img, detector_backend, enforce_detection=True):
    """Выводит на экран лица распознанные на фото img детектором лиц detector_backend"""
    for ef in DeepFace.extract_faces(img, detector_backend=detector_backend, enforce_detection=enforce_detection):
        # print(ef)
        Image.fromarray((ef['face'] * 255).astype(np.uint8)).show()


if __name__ == '__main__':
    print("START")
    start_time = time.time()

    # known_img = 'D:/FOTO/Original photo/Olympus/P720/0154.JPG'
    # known_img = 'D:/FOTO/Original photo/Olympus/P9170480.JPG'
    # known_img = 'D:/FOTO/Original photo/Olympus/P1011618.JPG'
    # known_img = 'D:/FOTO/Original photo/Moto/photo_2021-08-13_21-37-01.jpg'

    # directory = 'D:/FOTO/Original photo/Olympus'
    # directory = 'D:/FOTO/Finished photo'
    # directory = 'D:/Hobby/NmProject/nmbook_photo/web/static/out/photo'
    # directory = 'D:/FOTO/Original photo/Moto'
    directory = 'D:/Hobby/NmProject/Test_photo/Test_1-Home_photos'
    # directory = '../Test_photo/Telegram_photo_set'

    all_df_encodings_to_file(tool_module.get_all_file_in_directory(directory),
                             directory + '/encodings_parallel.json',
                             model_name=MODELS[8], process_count=4)
    # group_similar_faces_(directory + '/dfv2_facenet512_encodings_parallel.json',
    #                      directory + '/dfv2_facenet512(0.2499999999999999)_result_parallel.json',
    #                      threshold=0.2499999999999999, process_count=None)

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

    # with open('not_detected_face_list.json') as ndflf:
    #     not_detected_photo_list = json.load(ndflf)
    #     for photo in not_detected_photo_list:
    #         print(photo)
    #         try:
    #             show_recognized_faces(photo, DETECTOR_BACKEND[0])
    #         except Exception as ve:
    #             print(ve)
    # # 0 1 4?

    # print(
    #     DeepFace.verify('../Test_photo/Test_1-Home_photos/Arsen-1.JPG',
    #                     '../Test_photo/Test_1-Home_photos/Varl-4.jpg',
    #                     detector_backend=DETECTOR_BACKEND[0],
    #                     model_name=MODELS[0]))
    end_time = time.time()
    print("time in second: ", end_time - start_time, "\ntime in min: ", (end_time - start_time) / 60)
