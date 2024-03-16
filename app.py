from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

app = Flask(__name__)

model = load_model('animekah.h5') 
width, height = 224, 224
labels = ['Anime', 'Non-Anime']

@app.route('/')
def upload_page():
    return render_template('index.html', prediction=None) 

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded.'

    file = request.files['image']
    img = image.load_img(io.BytesIO(file.read()), target_size=(width, height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    pred = model.predict(img).argmax()
    predicted_class = labels[pred]

    if predicted_class == 'Anime':
        predicted_class = 'your uploaded pic is definitely ANIME!'
    elif predicted_class == 'Non-Anime':
        predicted_class = 'nope.'

    return render_template('index.html', prediction=predicted_class)  

if __name__ == '__main__':
    app.run()