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

app=Flask(__name__)
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    image = request.files['file']
    image.save('static/file.jpg')
 
    # asarray() class is used to convert
    # PIL images into NumPy arrays
    image = Image.open('static/file.jpg')
    data = asarray(image)
    
    #img_array = img_to_array('static/file.jpg')
    img_array = tf.expand_dims(data,0)
        
    predictions = model.predict(img_array)


    
    result=np.argmax(predictions[0])

    if result == 0:
        result='Early Blight'
    elif result == 1:
        result='Late Blight '
    else:
        result = "Healthy"
    
    return render_template('home.html',result=result)


if __name__=="__main__":
    app.run(debug=True)