import os

import numpy as np
from tqdm import tqdm

import deep_face_module_v2
from tool_module import result_analyzer, statistic_analyzer


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

    find_optimal_threshold(f'../Test_photo/Test_1-Home_photos/dfv2_facenet512_encodings.json',
                           step=0.025)

# Facenet512 > OpenFace > SFace

# Facenet512(35, 6, 3) > VGG-Face(8, 3, 2) > SFace(5, 3, 2) > Dlib(1, 2, 2)
