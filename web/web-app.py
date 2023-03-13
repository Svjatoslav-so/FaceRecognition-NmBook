import json
from pathlib import Path

from PIL import Image
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
            print('Same thing went wrong: ',  file_name)
            return "File can not be open!"
    else:
        abort(401)


@app.route("/img_show/<path:path_to_img>")
def img_show(path_to_img):
    path_to_img = str(Path(path_to_img))
    last_slash_index = path_to_img.rfind('\\')
    return send_from_directory(path_to_img[:last_slash_index+1], path_to_img[last_slash_index+1:], as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
