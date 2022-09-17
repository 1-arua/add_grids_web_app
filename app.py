import os
from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
import numpy as np
from PIL import Image
import math


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

app = Flask(__name__, static_folder="./templates/processed")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def add_grid(filename, num):
    img = np.array(Image.open(filename), dtype=np.float64)

    y, x, _ = img.shape
    if x < y:
        step = int(x / num)
        x_grid_num = num - 1
        y_grid_num = math.ceil(y / step) - 1
        x_remainder = x % num
        y_remainder = y - step * (y_grid_num + 1)
    else:
        step = int(y / num)
        x_grid_num = math.ceil(x / step) - 1
        y_grid_num = num - 1
        x_remainder = x - step * (x_grid_num + 1)
        y_remainder = y % num

    grid_width = 4
    grid = np.zeros([y, x, 4], dtype=np.float64)

    x_grid_base_pos = step * np.array(list(range(1, x_grid_num + 1))) + int(x_remainder / 2)
    y_grid_base_pos = step * np.array(list(range(1, y_grid_num + 1))) + int(y_remainder / 2)

    x_grid_over_img = 20 * np.ones([y, x_grid_num, 3])
    y_grid_over_img = 20 * np.ones([y_grid_num, x, 3])
    x_grid_line_for_grid_img = np.concatenate([20 * np.ones([y, x_grid_num, 3]), 255 * np.ones([y, x_grid_num, 1])], 2)
    y_grid_line_for_grid_img = np.concatenate([20 * np.ones([y_grid_num, x, 3]), 255 * np.ones([y_grid_num, x, 1])], 2)
    x_border_over_img = 5 * np.ones([y, grid_width, 3])
    y_border_over_img = 5 * np.ones([grid_width, x, 3])
    x_border_for_grid_img = np.concatenate(
        [20 * np.ones([y, grid_width, 3]), 255 * np.ones([y, grid_width, 1])], 2)
    y_border_for_grid_img = np.concatenate(
        [20 * np.ones([grid_width, x, 3]), 255 * np.ones([grid_width, x, 1])], 2)

    for i in range(-grid_width + 1, grid_width):
        img[:, x_grid_base_pos + i, 0:3] = 0.5 * img[:, x_grid_base_pos + i, 0:3] + 0.5 * x_grid_over_img
        img[y_grid_base_pos + i, :, 0:3] = 0.5 * img[y_grid_base_pos + i, :, 0:3] + 0.5 * y_grid_over_img
        grid[:, x_grid_base_pos + i, :] = x_grid_line_for_grid_img
        grid[y_grid_base_pos + i, :, :] = y_grid_line_for_grid_img
    img[:, :grid_width, 0:3] = 0.1 * img[:, :grid_width, 0:3] + 0.9 * x_border_over_img
    img[:, x - grid_width:, 0:3] = 0.1 * img[:, x - grid_width:, 0:3] + 0.9 * x_border_over_img
    img[:grid_width, :, 0:3] = 0.1 * img[:grid_width, :, 0:3] + 0.9 * y_border_over_img
    img[y - grid_width:, :, 0:3] = 0.1 * img[y - grid_width:, :, 0:3] + 0.9 * y_border_over_img
    grid[:, :grid_width, :] = x_border_for_grid_img
    grid[:, x - grid_width:, :] = x_border_for_grid_img
    grid[:grid_width, :, :] = y_border_for_grid_img
    grid[y - grid_width:, :, :] = y_border_for_grid_img

    img = 255.0 * (img / 255.0)**(1 / 3.0)
    grid = 255.0 * (grid / 255.0)**(1 / 3.0)

    img = img.astype(np.uint8)
    grid = grid.astype(np.uint8)

    grided_filename = "grided.png"
    grid_filename = "grid.png"
    Image.fromarray(img).save(os.path.join("templates", "processed", grided_filename), quality=95)
    Image.fromarray(grid).save(os.path.join("templates", "processed", grid_filename), quality=95)
    os.remove(filename)


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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
