import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image
import streamlit as st

def warn(*args, **kwargs):
	pass
	import warnings
	warnings.warn = warn


import streamlit as st

def warn(*args, **kwargs):
	pass
	import warnings
	warnings.warn = warn

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import pickle, os

model_path = os.path.join("plantDNet.h5")
model = tf.keras.models.load_model(model_path, compile=False)
disease_class = ['Potato__bell___Bacterial_spot', 'Potato__bell___healthy', 'Pepper___Early_blight',
                         'Pepper___Late_blight', 'Pepper___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                         'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                         'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                         'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    show_img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds

st.title("Leaves Disease Prediction")

options = st.radio("Select Image", ("Upload", "Open Camera"))

if options == "Upload":
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        name = str(uploaded_file.name)
        with open(os.path.join("uploads",uploaded_file.name),"wb") as f: 
            f.write(uploaded_file.getbuffer())

        image_path = os.path.join("uploads")
        image_path += "/"+name
        preds = model_predict(image_path, model)
        st.write(image_path)
        print(preds[0])
        a = preds[0]
        ind=np.argmax(a)
        print('Prediction:', disease_class[ind])
        result=disease_class[ind]
        st.header(result)

if options == "Open Camera":
    picture = st.camera_input("Take a picture")
    if picture:
        name = str(picture.name)
        with open(os.path.join("uploads",picture.name),"wb") as f: 
            f.write(picture.getbuffer())


        image_path = os.path.join("uploads")
        image_path += "/"+name
        preds = model_predict(image_path, model)
        st.write(image_path)
        print(preds[0])
        a = preds[0]
        ind=np.argmax(a)
        print('Prediction:', disease_class[ind])
        result=disease_class[ind]
        st.header(result)
