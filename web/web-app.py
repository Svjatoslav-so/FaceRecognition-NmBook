import base64
import json
import os
import pathlib
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageOps
from flask import Flask, render_template, request, abort, jsonify, send_from_directory, session, redirect, url_for

import tool_module
from dbManager import DBManager

MIN_FACE_AREA = 650

app = Flask(__name__)

app.secret_key = os.environ.get('secret_key')


@app.context_processor
def common_context():
    # определение пути
    current_dir = pathlib.Path('./db/')
    # определение шаблона
    db_pattern = "*.db"
    dbs = []
    for file in current_dir.glob(db_pattern):
        dbs.append(file.name.split('.')[0])
    return dict(dbs=dbs)


@app.route("/")
def home():
    return render_template('home.html')


# old
@app.route("/group_viewer")
def group_viewer():
    return render_template('group-viewer.html')


# old
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


# old
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


@app.route("/smart_group_viewer")
def smart_group_viewer():
    if request.method == 'GET':
        # print('START', request)
        threshold = request.args.get('threshold', 0.2)
        # print('threshold', threshold)
        manager = DBManager(f"./db/{session.get('db', common_context()['dbs'][0])}.db")
        groups = manager.get_photo_group_list_of_similar(threshold, MIN_FACE_AREA)
        # print('END')
        return render_template('smart-viewer.html', **{'groups': groups,
                                                       'threshold': threshold,
                                                       })
    return redirect(url_for('home'))


@app.route("/get_group")
def get_group():
    if request.method == 'GET':
        threshold = request.args.get('threshold', 0.2)
        origin_photo_id = request.args.get('origin_photo_id', None)
        origin_face_id = request.args.get('origin_face_id', None)
        origin_photo_title = request.args.get('origin_photo_title', None)
        origin_photo_docs = request.args.get('origin_photo_docs', None)
        origin_photo_x1 = float(request.args.get('origin_photo_x1', 0))
        origin_photo_y1 = float(request.args.get('origin_photo_y1', 0))
        origin_photo_x2 = float(request.args.get('origin_photo_x2', 0))
        origin_photo_y2 = float(request.args.get('origin_photo_y2', 0))

        print(origin_face_id, origin_photo_title, origin_photo_docs)
        manager = DBManager(f"./db/{session.get('db', common_context()['dbs'][0])}.db")
        group = manager.get_group_photo_with_face(origin_face_id, threshold, MIN_FACE_AREA)
        result = {
            'origin_photo_block': render_template('origin-photo-block.html',
                                                  **{'origin_photo_id': origin_photo_id,
                                                     'origin_photo_title': origin_photo_title,
                                                     'origin_photo_docs': origin_photo_docs,
                                                     'origin_photo_x1': origin_photo_x1,
                                                     'origin_photo_y1': origin_photo_y1,
                                                     'origin_photo_x2': origin_photo_x2,
                                                     'origin_photo_y2': origin_photo_y2,
                                                     'foto_with_face': get_foto_with_square_around_face}),
            'view_panel': render_template('view-panel.html',
                                          **{'group': group,
                                             'foto_with_face': get_foto_with_square_around_face})
        }
        return jsonify(result)
    return redirect(url_for('home'))


@app.route("/img_show/<path:path_to_img>")
def img_show(path_to_img):
    path_to_img = str(Path(path_to_img))
    last_slash_index = path_to_img.rfind('\\')
    return send_from_directory(path_to_img[:last_slash_index + 1], path_to_img[last_slash_index + 1:],
                               as_attachment=True)


@app.route("/set_db/<string:db>")
def set_db(db):
    session['db'] = db
    return {"current_db": session['db']}


def get_foto_with_square_around_face(photo_path, face_area):
    img = Image.fromarray(cv2.cvtColor(cv2.imread(photo_path), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.rectangle(face_area, outline=(255, 0, 0), width=5)

    if img.size[0] > 1500 or img.size[1] > 1500:
        img = ImageOps.contain(img, (1500, 1500))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = str(base64.b64encode(buffered.getvalue()))[2:-1]
    return f'data:image/jpeg;base64,{img_str}'


if __name__ == "__main__":
    app.run(debug=True)
