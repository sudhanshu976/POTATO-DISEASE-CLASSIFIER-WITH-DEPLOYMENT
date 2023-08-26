from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import base64
from PIL import Image
from numpy import asarray
from io import BytesIO,StringIO
import cv2

app=Flask(__name__)
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    image = request.files['file']
    #image.save('static/file.jpg')
 
    
    image = Image.open(image)

    
    resized = image.resize((256, 256))
    data = asarray(resized)
    
    
    img_array = tf.expand_dims(data,0)
        
    predictions = model.predict(img_array)


    
    result=np.argmax(predictions[0])

    if result == 0:
        result='This potato has a Early Blight Disease '
    elif result == 1:
        result='This potato has a Late Blight Disease '
    else:
        result = "This potato is Healthy"
    
    return render_template('home.html',result=result)


if __name__=="__main__":
    app.run(debug=True)
