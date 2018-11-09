import cv2
import tensorflow as tf
from keras.models import load_model

CATEGORIES = ["Cat", "Dog"]  # will use this to convert prediction num to string value


def prepare(filepath):
    IMG_SIZE = 100  # 50 in txt-based
    img_array = cv2.imread(filepath)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) 

model = load_model("myModel.h5")
prediction = model.predict([prepare('cat.jpg')]) 

print(CATEGORIES[int(prediction[0][0])])