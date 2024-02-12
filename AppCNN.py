import streamlit as st 
import pandas as pd 
import numpy as np
import sklearn
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow_hub as hub
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py


st.markdown("<h1 style='text-align: center; color: black;'>Dog Breed Classification</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: grey;'>Author: Alseny </h2>", unsafe_allow_html=True)

#st.header("Dog Breed Classification ")

def main():
    file = st.file_uploader("Choose the file", type = ["jpg", "png"])
    if file is not None:
        image = Image.open(file)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis("off")
        resultr = predict_class(image)
        st.writ(result)
        st.pyplot(figure)
       

def predict_class(image):
    classifier_model = tf.keras.models.load_model("inceptionV3.h5")
    shape = (299, 299, 3)
    model = tf.keras.Sequential(hub[hub.kerasLayers(classifier_model, input_shape = shape)])
    test_image = image.resize((299, 299))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    class_names = ['dhole', 'cairn', 'kelpie', 'Shih-Tzu', 
                  'Airedale', 'bull_mastiff', 'Doberman', 
                  'Lakeland_terrier', 'Chihuahua', 
                  'Rhodesian_ridgeback', 'Kerry_blue_terrier',
                   'Welsh_springer_spaniel', 'Brittany_spaniel', 
                   'giant_schnauzer', 'West_Highland_white_terrier', 
                   'Sealyham_terrier', 'Maltese_dog', 
                   'Norwich_terrier', 'beagle', 'Japanese_spaniel']

    predictions = model.prediction(test_image)
    scores = ft.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    result = "The image uploaded is: {}".format(image_class)
    return result

if __name__ == "__main__":
    main()

