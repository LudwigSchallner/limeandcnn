import cv2
import tensorflow as tf
from keras.models import load_model
import os

CATEGORIES = ["Cat", "Dog"]  # will use this to convert prediction num to string value


def prepare(filepath):
    IMG_SIZE = 100  # 50 in txt-based
    img_array = cv2.imread(filepath)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) 


file_name = os.path.join('trainedModels',
                         'catdog',
                         '3-conv-64-nodes-0-dense-1541770577')
model = load_model(file_name+".h5")
prediction = model.predict([prepare(os.path.join('data','catdog','test','Dog','6.jpg'))]) 

print((prediction[0]))