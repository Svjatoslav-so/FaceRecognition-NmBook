import base64
import json
import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

import deep_face_module_v2
from tool_module import result_analyzer, statistic_analyzer


def test_data_url(origin_photo, similar_photo, result_file):
    with open(result_file, encoding="utf8") as rf:
        all_data = json.load(rf)
        try:
            origin_block = list(filter(lambda d: d['origin'] == origin_photo, all_data))[0]
            face_areas = list(filter(lambda s: s['path'] == similar_photo, origin_block['similar']))[0]['face_areas']
            # origin_img = Image.open(origin_photo)
            origin_img = Image.fromarray(cv2.cvtColor(cv2.imread(origin_photo), cv2.COLOR_BGR2RGB))
            origin_draw = ImageDraw.Draw(origin_img)
            origin_draw.rectangle(face_areas['origin'], outline=(255, 255, 255), width=5)

            # similar_img = Image.open(similar_photo)
            similar_img = Image.fromarray(cv2.cvtColor(cv2.imread(similar_photo), cv2.COLOR_BGR2RGB))
            similar_draw = ImageDraw.Draw(similar_img)
            similar_draw.rectangle(face_areas['similar'], outline=(255, 255, 255), width=5)
            origin_img.show()
            similar_img.show()

            # for img in [origin_img, similar_img]:
            #     if img.size[0] > 1500 or img.size[1] > 1500:
            #         img = ImageOps.contain(img, (1500, 1500))
            #     buffered = BytesIO()
            #     img.save(buffered, format="JPEG")
            #     img_str = str(base64.b64encode(buffered.getvalue()))[2:-1]
            #     with open(f'temp.txt', 'w') as tf:
            #         tf.write(f'data:image/jpeg;base64,{img_str}')
        except Exception as e:
            print(e)
            pass


def find_optimal_threshold(encodings_file, start_threshold=0.4, step=0.05, max_total_errors=0):
    temp_result_file = 'temp_result.json'
    optimal_threshold = False
    if os.path.exists(temp_result_file):
        os.remove(temp_result_file)
    for threshold in tqdm(np.arange(start_threshold, 0, -step), desc='find_optimal_threshold '):
        deep_face_module_v2.group_similar_faces(encodings_file, temp_result_file, threshold=threshold, disable=True)
        report = result_analyzer(temp_result_file, disable=True)
        statistic = statistic_analyzer(report)
        if statistic['total_errors'] <= max_total_errors:
            optimal_threshold = threshold
            print('OPTIMAL_THRESHOLD: ', optimal_threshold)
            print(report, statistic)
            break
        os.remove(temp_result_file)
    if not optimal_threshold:
        print('FAIL')


if __name__ == '__main__':
    # statistics = []
    # for model in deep_face_module_v2.MODELS:
    #     if not model == 'DeepID':
    #         result_file = f'../Test_photo/Test_1-Home_photos/dfv2_{model.lower()}_result.json'
    #         model_stat = {'model_name': model}
    #         analyzer_report = result_analyzer(result_file)
    #         # model_stat.update(analyzer_report)
    #         model_stat.update(statistic_analyzer(analyzer_report))
    #         statistics.append(model_stat)
    # for s in statistics:
    #     print(s, '\t\t', s['average_error'], s['average_correct'])

    # for model in deep_face_module_v2.MODELS:
    #     if not model == 'DeepID':
    #         print(model)
    #         find_optimal_threshold(f'../Test_photo/Test_1-Home_photos/dfv2_{model.lower()}_encodings.json',
    #                                step=0.025)

    # find_optimal_threshold(f'../Test_photo/Test_1-Home_photos/dfv2_facenet512_encodings.json',
    #                        step=0.025)

    test_data_url('D:/Hobby/NmProject/Test_photo/Test_1-Home_photos/Arsen-1.JPG',
                  'D:/Hobby/NmProject/Test_photo/Test_1-Home_photos/Arsen-2.JPG',
                  '../Test_photo/Test_1-Home_photos/dfv2_facenet512_result.json')

# Facenet512 > OpenFace > SFace

# Facenet512(35, 6, 3) > VGG-Face(8, 3, 2) > SFace(5, 3, 2) > Dlib(1, 2, 2)
