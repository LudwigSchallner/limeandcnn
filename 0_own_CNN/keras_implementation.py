import numpy as np
import cv2
import matplotlib.pyplot as plt

num_class = 5
train_samples = 3000 #sumoftrain
test_samples = 670  #sumofvalidation
epochs = 12
batch_size = 16

from keras.applications import inception_v3 as inc_net
base_model = inc_net.InceptionV3(weights='imagenet', include_top=False)
print("InceptionV3 loaded")

from keras.preprocessing import image
# Function for valdiate img
def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

    from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
# idk needs a loot of improvement, but just for testing


x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#freezing all layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range = 0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set= train_datagen.flow_from_directory('./data/flower_photos/train/', target_size = (299,299), batch_size=32)

model.fit_generator(training_set, steps_per_epoch=400, epochs=5, validation_steps=100)
