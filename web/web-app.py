import json

from PIL import Image
from flask import Flask, render_template, request, abort, jsonify

import tool_module

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/group_viewer")
def group_viewer():
    # img_pil = Image.open('C:/Users/DellM/Pictures/photo_2022-01-26_13-40-12.jpg')
    # img_pil.show()
    return render_template('group-viewer.html')


@app.post("/open_group_file")
def open_group_file():
    if request.method == 'POST':
        file_name = request.form.get('group_file', None)
        if file_name and file_name.endswith('.json'):
            print('File Name: ', file_name)
            unique_group_list = tool_module.get_all_unique_photo_group(file_name, True)
            with open('static/out/metadata.json', encoding="utf8") as mf:
                metadata = json.load(mf)
            return jsonify({'group_list': unique_group_list, 'metadata': metadata})
        else:
            print('Same thing went wrong: ',  file_name)
            return "File can not be open!"
    else:
        abort(401)


if __name__ == "__main__":
    app.run(debug=True)
