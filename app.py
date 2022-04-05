import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

model=tf.keras.models.load_model("model/model.h5",compile=False)
IMAGE_SIZE=(256,256)
UPLOAD_FOLDER=os.path.join('static')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def segment(path):
    input_image=cv2.imread(path)
    plt.imsave("static/input_.jpg",input_image)
    os.remove(path)
    img = cv2.resize(input_image ,IMAGE_SIZE)
    img = img / 255
    img = img[np.newaxis, :, :, :]

    pred=model.predict(img)
    output=np.squeeze(pred> .5)
    plt.imsave("static/output.jpg",output)
@app.route('/display')
def display():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpeg')
    return render_template('display.html',img_file=full_filename)
@app.route('/', methods=['GET', 'POST'])
def home():      
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file = request.files['file1']
        # path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpeg')
        path =UPLOAD_FOLDER+'/test.jpg'
        print(path)
        file.save(path)
        segment(path)
        # return render_template("display.html",input_file=UPLOAD_FOLDER+'/input.jpeg',output_file=UPLOAD_FOLDER+'/output.jpeg')
        return render_template("display.html",input_file=UPLOAD_FOLDER+'/input_.jpg',output_file=UPLOAD_FOLDER+'/output.jpg')

    return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
