import base64
import json
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageOps
from flask import Flask, render_template, request, abort, jsonify, send_from_directory

import tool_module

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/group_viewer")
def group_viewer():
    return render_template('group-viewer.html')


@app.post("/open_group_file")
def open_group_file():
    if request.method == 'POST':
        file_name = request.form.get('group_file', None)
        metadata_file = request.form.get('metadata_file', 'static/out/metadata.json')
        if file_name and file_name.endswith('.json'):
            print('File Name: ', file_name)
            unique_group_list = tool_module.get_all_unique_photo_group(file_name, True)
            with open(metadata_file, encoding="utf8") as mf:
                metadata = json.load(mf)
            return jsonify({'group_list': unique_group_list, 'metadata': metadata})
        else:
            print('Same thing went wrong: ', file_name)
            return "File can not be open!"
    else:
        abort(401)


@app.route("/get_img_with_face_area")
def get_img_with_face_area():
    result = {}
    if request.method == 'GET':
        origin_path = request.args.get('origin_path', None)
        similar_path = request.args.get('similar_path', None)
        file_path = request.args.get('file_path', None)
        if origin_path and similar_path and file_path:
            with open(file_path, encoding="utf8") as rf:
                all_data = json.load(rf)
                # try:
                origin_block = list(filter(lambda d: d['origin'] == origin_path, all_data))[0]
                face_areas = list(filter(lambda s: s['path'] == similar_path, origin_block['similar']))[0][
                    'face_areas']
                print(face_areas)
                origin_img = Image.fromarray(cv2.cvtColor(cv2.imread(origin_path), cv2.COLOR_BGR2RGB))
                origin_draw = ImageDraw.Draw(origin_img)
                origin_draw.rectangle(face_areas['origin'], outline=(255, 0, 0), width=5)

                similar_img = Image.fromarray(cv2.cvtColor(cv2.imread(similar_path), cv2.COLOR_BGR2RGB))
                similar_draw = ImageDraw.Draw(similar_img)
                similar_draw.rectangle(face_areas['similar'] if face_areas['similar'] else (0, 0, 0, 0),
                                       outline=(255, 0, 0), width=5)

                for key, img in [('origin', origin_img), ('similar', similar_img)]:
                    if img.size[0] > 1500 or img.size[1] > 1500:
                        img = ImageOps.contain(img, (1500, 1500))
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = str(base64.b64encode(buffered.getvalue()))[2:-1]
                    result[key] = f'data:image/jpeg;base64,{img_str}'
                # except Exception as e:
                #     print(e)
    return jsonify(result)


@app.route("/img_show/<path:path_to_img>")
def img_show(path_to_img):
    path_to_img = str(Path(path_to_img))
    last_slash_index = path_to_img.rfind('\\')
    return send_from_directory(path_to_img[:last_slash_index + 1], path_to_img[last_slash_index + 1:],
                               as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
