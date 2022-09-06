import os
from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
import numpy as np
from PIL import Image
from icecream import ic


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

app = Flask(__name__, static_folder="./templates/processed")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def add_grid(filename, num):
    img = np.array(Image.open(filename), dtype=np.float64)

    y, x, _ = img.shape
    x_step = int(x / num)
    y_step = int(y / num)
    if x_step < y_step:
        step = x_step
    else:
        step = y_step
    grid_width = 4
    grid = np.zeros([y, x, 4], dtype=np.float64)

    _, x_lines, _ = img[:, step:x - 2 * grid_width - 1:step, 0:3].shape
    y_lines, _, _ = img[step:y - 2 * grid_width - 1:step, :, 0:3].shape

    for i in range(-grid_width + 1, grid_width):
        img[:, step + i:x - 2 * grid_width - 1:step, 0:3] = \
            0.5 * img[:, step + i:x - 2 * grid_width - 1:step, 0:3] + 0.5 * 20 * np.ones([y, x_lines, 3])
        img[step + i:y - 2 * grid_width - 1:step, :, 0:3] =  \
            0.5 * img[step + i:y - 2 * grid_width - 1:step, :, 0:3] + 0.5 * 20 * np.ones([y_lines, x, 3])
        grid[:, step + i:x - 2 * grid_width - 1:step, :] = \
            np.concatenate([20 * np.ones([y, x_lines, 3]), 255 * np.ones([y, x_lines, 1])], 2)
        grid[step + i:y - 2 * grid_width - 1:step, :, :] = \
            np.concatenate([20 * np.ones([y_lines, x, 3]), 255 * np.ones([y_lines, x, 1])], 2)

    img = 255.0 * (img / 255.0)**(1 / 3.0)
    grid = 255.0 * (grid / 255.0)**(1 / 3.0)

    img = img.astype(np.uint8)
    grid = grid.astype(np.uint8)

    grided_filename = "grided.png"
    grid_filename = "grid.png"
    Image.fromarray(img).save(os.path.join("templates", "processed", grided_filename), quality=95)
    Image.fromarray(grid).save(os.path.join("templates", "processed", grid_filename), quality=95)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def uploads_file():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def display_file():
    if 'file' not in request.files:
        flash('ファイルがありません')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('ファイルがありません')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        grid_num = int(request.form.get("num"))
        add_grid(os.path.join(app.config["UPLOAD_FOLDER"], filename), grid_num)
        return redirect(url_for('processed_file'))


@app.route('/processed', methods=['GET'])
def processed_file():
    return render_template("processed.html")
