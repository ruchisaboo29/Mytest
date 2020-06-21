from __future__ import division, print_function

# Keras
from tensorflow.keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

import numpy as np
np.random.seed(2)

import cv2
import os

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'digit_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

model._make_predict_function()
print('Model loaded..')


def model_predict(img_path, model):
    img = cv2.imread(img_path, 0)
    print('img',img)

    # Preprocessing the image
    x = np.array(img)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = x.reshape(1, 28, 28, 1)

    # converting data to higher precision
    x = x.astype('float32')

    # normalizing the data to help with the training
    x /= 255
    print(x)
    #x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)

    # Checking random predicted value if it's predicted correctly or not
    print('Actual label for this observation is ', preds)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        try:
            preds = model_predict(file_path, model)
            predicted_classes = np.argmax(np.round(preds), axis=1)
            return str(predicted_classes)  #Convert to string
        except Exception:
            return 'Image should be of 28 pixel. Please try it again! '
    return None


if __name__ == '__main__':
    app.run(debug=True)
