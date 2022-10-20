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


def process(filename, num):
    img = np.array(Image.open(filename), dtype=np.float64)
    y, x, _ = img.shape
    a = 4000 // x if x < y else 4000 // y
    b = a + 1
    ratio = a if (4000 - a * min(x, y)) < (b * min(x, y) - 4000) else b
    size = (x * ratio, y * ratio)
    print(size)

    enlarge_image(img, size)
    create_grid(size, num)

    os.remove(filename)


def enlarge_image(img, size):
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.resize(size, Image.BICUBIC).save(os.path.join("templates", "processed", "enlarged.png"), quality=95)


def create_grid(size, num):
    x, y = size[0], size[1]
    standard_length = x if x < y else y
    step = int(standard_length / num)
    x_grid_num = num - 1 if x == standard_length else math.ceil(x / step) - 1
    y_grid_num = num - 1 if y == standard_length else math.ceil(y / step) - 1
    x_remainder = x % num if x == standard_length else x - step * (x_grid_num + 1)
    y_remainder = y % num if y == standard_length else y - step * (y_grid_num + 1)

    grid_width = max(step // 50, 1)
    grid = np.zeros([y, x, 4], dtype=np.float64)

    x_grid_base_pos = step * np.array(list(range(1, x_grid_num + 1))) + int(x_remainder / 2)
    y_grid_base_pos = step * np.array(list(range(1, y_grid_num + 1))) + int(y_remainder / 2)

    x_grid_line_for_grid_img = np.concatenate([20 * np.ones([y, x_grid_num, 3]), 255 * np.ones([y, x_grid_num, 1])], 2)
    y_grid_line_for_grid_img = np.concatenate([20 * np.ones([y_grid_num, x, 3]), 255 * np.ones([y_grid_num, x, 1])], 2)
    x_border_for_grid_img = np.concatenate(
        [20 * np.ones([y, grid_width, 3]), 255 * np.ones([y, grid_width, 1])], 2)
    y_border_for_grid_img = np.concatenate(
        [20 * np.ones([grid_width, x, 3]), 255 * np.ones([grid_width, x, 1])], 2)

    for i in range(-grid_width + 1, grid_width):
        grid[:, x_grid_base_pos + i, :] = x_grid_line_for_grid_img
        grid[y_grid_base_pos + i, :, :] = y_grid_line_for_grid_img
    grid[:, :grid_width, :] = x_border_for_grid_img
    grid[:, x - grid_width:, :] = x_border_for_grid_img
    grid[:grid_width, :, :] = y_border_for_grid_img
    grid[y - grid_width:, :, :] = y_border_for_grid_img

    grid = 255.0 * (grid / 255.0)**(1 / 3.0)

    grid = grid.astype(np.uint8)

    grid_filename = "grid.png"
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
        process(os.path.join(app.config["UPLOAD_FOLDER"], filename), grid_num)
        return redirect(url_for('processed_file'))


@app.route('/processed', methods=['GET'])
def processed_file():
    return render_template("processed.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
