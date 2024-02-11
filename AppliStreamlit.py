import streamlit as st 
import pandas as pd 
import numpy as np
import sklearn
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.hub as hub
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py


st.header("Image class predictor")

def main():
    file.uploaded( = st.file.uploader("Choose the file", type = ["jpg", "png", "jpeg"])
    if file.uploaded is not Non:
       image = Image.open(file_uploaded)
       figure = plt.figure()
       plt.imshow(image)
       plt.axis("off")
       resultr = predict_class(image)
       st.writ(result)
       st.pyplot(figure)
       
       
def predict_class(image):
    classifier_model = tf.keras.models.load_model(r"lien vers models./my_model.hdf5")
    shape = (244, 244, 3)
    model = tf.keras.Sequential(hub[hub.kerasLayers(classifier_model, input_shape = shape)])
    test_image = image.resize((244, 244))
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



