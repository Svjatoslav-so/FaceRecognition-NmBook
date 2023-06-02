"""
    Данный модуль содержит вторую версию функций по распознаванию лиц на фото в основе которых лежит deepface
"""
import functools
import json
import multiprocessing
import os
import time
from math import copysign
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from deepface import DeepFace
from deepface.commons import functions, distance as dst
from deepface.commons.functions import load_image
from deepface.detectors import FaceDetector, OpenCvWrapper, SsdWrapper, DlibWrapper, MtcnnWrapper, MediapipeWrapper
from retinaface import RetinaFace
from retinaface.commons import postprocess
from tqdm import tqdm

import tool_module

tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
if tf_major_version == 1:
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image


# список самых популярных детекторов лиц
DETECTOR_BACKEND = ['retinaface', 'mtcnn', 'opencv', 'ssd', 'dlib']
# список самых популярных моделей распознавания лиц
MODELS = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
# доступные показатели расстояния
DISTANCE_METRIC = ['cosine', 'euclidean', 'euclidean_l2']


def special_retinaface_detect_face(face_detector, img, align=True):
    """
        Кастомная функция, используется в special_detect_faces.
        Точная копия deepface.detectors.RetinaFaceWrapper.detect_face, но дополнительно возвращает координат частей лица
        ('right_eye', 'left_eye', 'nose', 'mouth_left', 'mouth_right'), которые используются для определения того
        является ли лицо профилем.
        Координаты частей лица можно получить из landmarks.
        Возвращает кортеж (detected_face, img_region, confidence, landmarks)
    """

    resp = []

    # The BGR2RGB conversion will be done in the preprocessing step of retinaface.
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #retinaface expects RGB but OpenCV read BGR

    """
    face = None
    img_region = [0, 0, img.shape[1], img.shape[0]] #Really?

    faces = RetinaFace.extract_faces(img_rgb, model = face_detector, align = align)

    if len(faces) > 0:
        face = faces[0][:, :, ::-1]

    return face, img_region
    """
    # --------------------------

    obj = RetinaFace.detect_faces(img, model=face_detector, threshold=0.9)

    if type(obj) == dict:
        for key in obj:
            identity = obj[key]
            facial_area = identity["facial_area"]

            y = facial_area[1]
            h = facial_area[3] - y
            x = facial_area[0]
            w = facial_area[2] - x
            img_region = [x, y, w, h]
            confidence = identity["score"]

            # detected_face = img[int(y):int(y+h), int(x):int(x+w)] #opencv
            detected_face = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

            if align:
                landmarks = identity["landmarks"]
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
                # mouth_right = landmarks["mouth_right"]
                # mouth_left = landmarks["mouth_left"]

                detected_face = postprocess.alignment_procedure(detected_face, right_eye, left_eye, nose)

            resp.append((detected_face, img_region, confidence, identity["landmarks"]))

    return resp


def special_detect_faces(face_detector, detector_backend, img, align=True):
    """
        Кастомная функция, используется в special_extract_faces.
        Точная копия deepface.detectors.FaceDetector.detect_faces, но использующая
        кастомную функцию special_retinaface_detect_face вместо deepface.detectors.RetinaFaceWrapper.detect_face.
        Необходима для получения координат частей лица ('right_eye', 'left_eye', 'nose', 'mouth_left', 'mouth_right'),
        при использовании detector_backend=retinaface. Которые используются для определения является ли лицо профилем.
    """
    backends = {
        'opencv': OpenCvWrapper.detect_face,
        'ssd': SsdWrapper.detect_face,
        'dlib': DlibWrapper.detect_face,
        'mtcnn': MtcnnWrapper.detect_face,
        'retinaface': special_retinaface_detect_face,
        'mediapipe': MediapipeWrapper.detect_face
    }

    detect_face = backends.get(detector_backend)

    if detect_face:
        obj = detect_face(face_detector, img, align)
        # obj stores list of (detected_face, region, confidence)

        return obj
    else:
        raise ValueError("invalid detector_backend passed - " + detector_backend)


def special_extract_faces(img, target_size=(224, 224), detector_backend='opencv', grayscale=False,
                          enforce_detection=True,
                          align=True):
    """
        Кастомная функция, используется в _encoding_process.
        Точная копия deepface.commons.functions.extract_faces, но использующая кастомную функцию special_detect_faces
        для получения face_objs.
        Необходима для получения координат частей лица ('right_eye', 'left_eye', 'nose', 'mouth_left', 'mouth_right'),
        при использовании detector_backend=retinaface. Которые используются для определения является ли лицо профилем.
        Если detector_backend=retinaface, то словарь содержащий координаты частей лица будет передан 4-м параметром каждого extracted_face,
        если detector_backend!=retinaface, то 4-й параметр каждого extracted_face - None
    """

    # this is going to store a list of img itself (numpy), it region and confidence
    extracted_faces = []

    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img = load_image(img)
    img_region = [0, 0, img.shape[1], img.shape[0]]

    if detector_backend == 'skip':
        face_objs = [(img, img_region, 0)]
    else:
        face_detector = FaceDetector.build_model(detector_backend)
        face_objs = special_detect_faces(face_detector, detector_backend, img, align)
        if not detector_backend == 'retinaface':
            face_objs = list(map(lambda x: (x[0], x[1], x[2], None), face_objs))

    # in case of no face found
    if len(face_objs) == 0 and enforce_detection == True:
        raise ValueError(
            "Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")
    elif len(face_objs) == 0 and enforce_detection == False:
        face_objs = [(img, img_region, 0)]

    for current_img, current_region, confidence, landmarks in face_objs:
        if current_img.shape[0] > 0 and current_img.shape[1] > 0:

            if grayscale == True:
                current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

            # resize and padding
            if current_img.shape[0] > 0 and current_img.shape[1] > 0:
                factor_0 = target_size[0] / current_img.shape[0]
                factor_1 = target_size[1] / current_img.shape[1]
                factor = min(factor_0, factor_1)

                dsize = (int(current_img.shape[1] * factor), int(current_img.shape[0] * factor))
                current_img = cv2.resize(current_img, dsize)

                diff_0 = target_size[0] - current_img.shape[0]
                diff_1 = target_size[1] - current_img.shape[1]
                if grayscale == False:
                    # Put the base image in the middle of the padded image
                    current_img = np.pad(current_img, (
                        (diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
                else:
                    current_img = np.pad(current_img,
                                         ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)),
                                         'constant')

            # double check: if target image is not still the same size with target.
            if current_img.shape[0:2] != target_size:
                current_img = cv2.resize(current_img, target_size)

            # normalizing the image pixels
            img_pixels = image.img_to_array(current_img)  # what this line doing? must?
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # normalize input in [0, 1]

            # int cast is for the exception - object of type 'float32' is not JSON serializable
            region_obj = {"x": int(current_region[0]), "y": int(current_region[1]), "w": int(current_region[2]),
                          "h": int(current_region[3])}

            extracted_face = [img_pixels, region_obj, confidence, landmarks]
            extracted_faces.append(extracted_face)

    if len(extracted_faces) == 0 and enforce_detection == True:
        raise ValueError("Detected face shape is ", img.shape, ". Consider to set enforce_detection argument to False.")

    return extracted_faces


def _encoding_process(paths, queue, model_name, target_size, detector_backends, enforce_detection, align, disable):
    """ Функция выполняемая отдельными процессами в all_df_encodings_to_file """
    data = []
    for path in tqdm(paths, disable=disable):
        if os.path.exists(path):
            img_objs = []
            for d_backend in detector_backends:
                try:
                    img_objs = special_extract_faces(img=path,
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
            for img_content, img_region, img_confidence, landmarks in img_objs:
                embedding_obj = DeepFace.represent(img_path=img_content
                                                   , model_name=model_name
                                                   , enforce_detection=enforce_detection
                                                   , detector_backend="skip"
                                                   , align=align
                                                   , normalization='base'
                                                   )
                face_representation = {"face_area": (img_region['x'],
                                                     img_region['y'],
                                                     img_region['x'] + img_region['w'],
                                                     img_region['y'] + img_region['h']),
                                       "encoding": embedding_obj[0]["embedding"],
                                       "is_side_view": False if landmarks is None else
                                       not _is_point_in_tetragon(landmarks['nose'],
                                                                 landmarks['mouth_right'], landmarks['mouth_left'],
                                                                 landmarks['left_eye'], landmarks['right_eye'])}
                img_faces.append(face_representation)
            data.append({"path": path,
                         "faces": img_faces})
    queue.put(data)  # проблемное место, если вызывать join() перед извлечением данных из queue (get()).
    # queue.cancel_join_thread() # не безопасно использовать, могут быть потеряны данные добавленные в очередь,
    # но при этом действительно процесс не виснет, а завершается.
    # print("@Encoding_process end!!!")


def all_df_encodings_to_file(paths_to_photo, file_name='encodings.json', model_name=MODELS[1],
                             detector_backends=None, enforce_detection=True, process_count=None,
                             disable=False):
    """
        Записывает кодировки лиц(deepface) для всех фото из paths_to_photo в файл file_name в формате json:
        {
            "path": str(путь к файлу),
            "faces": [
                {
                    "face_area": (список координат лица),
                    "encoding": (вектор кодировки лица)
                    "is_side_view": (профиль, если True, если False - анфас)
                }
            ]
        }
        paths_to_photo - список строк.
        model_name - модель используемая для получения кодировки лица.
        file_name - имя файла в который нужно записать результат,
                    к указанному имени автоматически добавляется название используемой модели.
        detector_backends - список детекторов распознавания лиц. Если не сработает первый, пробуем следующий.
                            Если None, то по умолчанию используются ['retinaface', 'mtcnn'].
        enforce_detection - если False, то для фото без лица не будет выбрасываться ошибка.
        process_count - количество процессов для расчета. (Введенное значение может быть автоматически изменено).
                       По умолчанию None - определяется автоматически(равно количеству логических ядер).
                       (Оптимальное количество 4).
        disable - если True, то прогресс-бар будет отключен.

        Функция возвращает словарь:
            {
                file_name: имя файла в который записаны кодировки,
                result: список кодировок (тот что записан в файл)
            }
    """
    if detector_backends is None:
        detector_backends = ['retinaface', 'mtcnn']

    d_backs_str = functools.reduce(lambda x, y: x+str(y)+'-', detector_backends, '')[:-1]
    name_start = file_name.rfind("/") + 1
    file_name = file_name[:name_start] + f'dfv2_{model_name.lower()}_{d_backs_str}_{file_name[name_start:]}'
    if not os.path.exists(file_name):
        with Path(file_name).open('x', encoding="utf8") as f:
            all_data = []

            process_list = []
            queue = multiprocessing.Queue()
            process_count, pt_count = tool_module.get_optimal_process_count(len(paths_to_photo), process_count)

            target_size = functions.find_target_size(model_name=model_name)
            align = True
            try:
                for i in range(process_count):
                    process_list.append(multiprocessing.Process(target=_encoding_process,
                                                                kwargs={'paths': paths_to_photo[
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
            except KeyboardInterrupt as ki:
                for p in process_list:
                    p.kill()
                raise ki
            print(f"All process finished!")

            json.dump(all_data, f, indent=4)
            return {'file_name': file_name, 'result': all_data}

    else:
        raise ValueError(f'File "{file_name}" already exists')


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


def _is_point_in_tetragon(p: (float, float), a: (float, float), b: (float, float), c: (float, float),
                          d: (float, float)):
    """
        Проверяет находится ли точка p в четырехугольнике abcd
        a___b
        |   |
        d___c
        p - кортеж с координатами точки p
        a - кортеж с координатами точки a
        b - кортеж с координатами точки b
        c - кортеж с координатами точки c
        d - кортеж с координатами точки d
        Возвращает True, если p находится внутри abcd и False, если не находится
    """

    def make_point(x, y):
        return {'x': x, 'y': y}

    A = make_point(*a)
    B = make_point(*b)
    C = make_point(*c)
    D = make_point(*d)
    E = make_point(*p)

    def side(start, end, point):
        return copysign(1,
                        (end['x'] - start['x']) * (point['y'] - start['y'])
                        - (end['y'] - start['y']) * (point['x'] - start['x']))

    return side(A, B, E) == -1 and side(B, C, E) == -1 and side(C, D, E) == -1 and side(D, A, E) == -1


if __name__ == '__main__':
    print("START")
    start_time = time.time()

    # directory = 'D:/FOTO/Original photo/Olympus'
    # directory = 'D:/FOTO/Finished photo'
    directory = 'D:/Hobby/NmProject/nmbook_photo/web/static/out/photo'
    # directory = 'D:/FOTO/Original photo/Moto'
    # directory = 'D:/Hobby/NmProject/Test_photo/Test_1-Home_photos'
    # directory = '../Test_photo/Telegram_photo_set'

    # # img_path = '/3/3746c174a92b48ed4929dfe4ffaafda7d931629e.jpeg'
    # # img_path = '/6/6ac86c8c1fae77b67adada7372464be4e941ada3.jpeg'
    # # img_path = '/0/09e16386146c68fdba6b4abcb317e0c9ae5819e1.jpeg'
    # # img_path = '/2/22ac1b452181f3274f3005ac49f27cb3b8100609.jpeg'
    # # img_path = '/2/2f74a31ac0f7d4c27e14673c7910b0c6c3078fdf.jpeg'
    # img_path = '/9/99405a718e5e9be1c3903be0180a1a4cc0c34c80.jpeg'
    # # show_recognized_faces(directory+img_path, DETECTOR_BACKEND[0], enforce_detection=True)
    # resp = RetinaFace.detect_faces(directory+img_path)
    # print(resp)
    # img = Image.open(directory+img_path).convert('RGB')
    # draw = ImageDraw.Draw(img)
    # for i in range(1, len(resp)+1):
    #     face = resp[f'face_{i}']
    #     draw.rectangle(face['facial_area'], outline=(255, 0, 0), width=5)
    #     draw.point(face['landmarks']['right_eye'], fill=(0, 255, 0))
    #     draw.point(face['landmarks']['left_eye'], fill=(0, 255, 0))
    #     draw.point(face['landmarks']['nose'], fill=(0, 255, 0))
    #     draw.point(face['landmarks']['mouth_left'], fill=(0, 0, 255))
    #     draw.point(face['landmarks']['mouth_right'], fill=(0, 0, 255))
    # img.show()

    all_df_encodings_to_file(tool_module.get_all_file_in_directory(directory),
                             directory + '/encodings_with_profile.json',
                             model_name=MODELS[6],
                             detector_backends=[DETECTOR_BACKEND[0]],  # DETECTOR_BACKEND[1]],
                             process_count=2)
    # for t in np.arange(0.3, 0.05, -0.05):
    # t = 0.05
    # group_similar_faces(directory + '/dfv2_sface_encodings.json',
    #                     directory + f'/dfv2_sface({t})_result.json',
    #                     threshold=t, process_count=8)

    # print(_is_point_in_tetragon((203, 290), (182, 323), (230, 319), (231, 253), (169, 258)))
    end_time = time.time()
    print("time in second: ", end_time - start_time, "\ntime in min: ", (end_time - start_time) / 60)
