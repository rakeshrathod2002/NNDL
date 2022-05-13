import streamlit as st
import cv2
# from cv2 import imread, resize
from matplotlib import pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

###########################################


import os, shutil 
dir = os.path.join('uploads') + "/"
for files in os.listdir(dir):
    path = os.path.join(dir, files)
    try:
        shutil.rmtree(path)
    except OSError:
        os.remove(path)

st.title("Leaves Disease Prediction")
options = st.selectbox("Select Crop", ["Tomato", "Maize"])
# options = st.radio("Select Image", ("Upload", "Open Camera"))
# model = st.radio("Select Crop", ("Tomato", "Maize"))

tomato_classes = ["Tomato - Bacteria Spot Disease",
                    "Tomato - Early Blight Disease",
                    "Tomato - Healthy and Fresh", 
                    "Tomato - Late Blight Disease", 
                    "Tomato - Leaf Mold Disease",
                    "Tomato - Septoria Leaf Spot Disease",
                    "Tomato - Target Spot Disease",
                    "Tomato - Tomoato Yellow Leaf Curl Virus Disease",
                    "Tomato - Tomato Mosaic Virus Disease",
                    "Tomato - Two Spotted Spider Mite Disease"]



maize_classes =["Maize-Blight", "Maize-common_rust", 
                "Maize-gray_leaf_spot", "Maize-healthy"]



def predictDisease(modelpath_, image_):
    model = load_model(modelpath_)
    image = cv2.imread(image_)
    test_image = cv2.resize(image, (128, 128))
    test_image = img_to_array(test_image)/255 # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
    result = model.predict(test_image) # predict diseased palnt or not
    pred = np.argmax(result, axis=1)
    return pred

if options == "Tomato":
    modelpath_ = "Tomato_model.h5"
    file_options = st.radio("Select Tomato Leaf", ("Upload", "Open Camera"))
    if file_options == "Upload":
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            name = str(uploaded_file.name)
            with open(os.path.join("uploads",uploaded_file.name),"wb") as f: 
                f.write(uploaded_file.getbuffer())
            image_path = os.path.join("uploads")
            image_path += "/"+name
            res = predictDisease(modelpath_= modelpath_, image_ = image_path)
            # st.write(res[0])
            st.subheader(f"Possible Disease: {tomato_classes[res[0]]}")

    
    if file_options == "Open Camera":
        picture = st.camera_input("Take a picture")
        if picture:
            name = str(picture.name)
            with open(os.path.join("uploads",picture.name),"wb") as f: 
                f.write(picture.getbuffer())
            
            image_path = os.path.join("uploads")
            image_path += "/"+name
            res = predictDisease(modelpath_= modelpath_, image_ = image_path)
            st.subheader(f"Possible Disease: {tomato_classes[res[0]]}")
    
if options == "Maize":
    modelpath_ = "maize_model.h5"
    file_options = st.radio("Select Maize Leaf", ("Upload", "Open Camera"))
    if file_options == "Upload":
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            name = str(uploaded_file.name)
            with open(os.path.join("uploads",uploaded_file.name),"wb") as f: 
                f.write(uploaded_file.getbuffer())
            image_path = os.path.join("uploads")
            image_path += "/"+name
            res = predictDisease(modelpath_= modelpath_, image_ = image_path)
            # st.subheader(f"Possible Disease: {maize_classes[res]}")
            st.subheader(f"Possible Disease: {maize_classes[res[0]]}")

    
    if file_options == "Open Camera":
        picture = st.camera_input("Take a picture")
        if picture:
            name = str(picture.name)
            with open(os.path.join("uploads",picture.name),"wb") as f: 
                f.write(picture.getbuffer())

            image_path = os.path.join("uploads")
            image_path += "/"+name
            res = predictDisease(modelpath_= modelpath_, image_ = image_path)
            st.subheader(f"Possible Disease: {maize_classes[res[0]]}")
