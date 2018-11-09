import pickle
import tensorflow as tf
import sys 
import keras
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LeakyReLU, ELU 
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import pickle
import time


print("Tensorflow Version: "+tf.__version__)
print("Keras Version: "+keras.__version__)


if sys.argv[1] == "catdog":
    CATEGORIES = ["Cat","Dog"]
    IMG_SIZE = 100
    CHANNEL = 3
    BATCH_SIZE = 32
    TRAIN_SIZE = 22451
    TEST_SIZE = 2495
    STEPS_EPOCH_TRAIN = int(TRAIN_SIZE / BATCH_SIZE)
    STEPS_EPOCH_TEST = int(TEST_SIZE / BATCH_SIZE)
elif sys.argv[1] == "flowers":
    CATEGORIES = ["daisy","dandelion","roses","sunflowers","tulips"]
    IMG_SIZE = 100
    CHANNEL = 3
    BATCH_SIZE = 32
    TRAIN_SIZE = 3000
    TEST_SIZE = 670
    STEPS_EPOCH_TRAIN = int(TRAIN_SIZE / BATCH_SIZE)
    STEPS_EPOCH_TEST = int(TEST_SIZE / BATCH_SIZE)
else:
    raise NameError("Dataset "+sys.argv[1]+" not supported!")

NAME = sys.argv[1]+" {}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# pickle_in = open(os.path.join('.',"data",sys.argv[1],"X_"+sys.argv[1]+".pickle"),"rb")
# X = pickle.load(pickle_in)

# pickle_in = open(os.path.join('.',"data",sys.argv[1],"y_"+sys.argv[1]+".pickle"),"rb")
# y = pickle.load(pickle_in)

# X = X/255.0
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range = 0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set= train_datagen.flow_from_directory(os.path.join('.',"data",sys.argv[1],'train'),
                                                target_size = (IMG_SIZE, IMG_SIZE),
                                                batch_size=32,
                                                class_mode='categorical')
val_set = test_datagen.flow_from_directory(os.path.join('.',"data",sys.argv[1],'test'),
                                           target_size= (IMG_SIZE, IMG_SIZE),
                                           batch_size=32,
                                           class_mode='categorical')

#training_set = train_datagen.flow(X,y,batch_size=32)

layer_sizes = [64]
conv_layers = [3]
dense_layers = [0]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=(IMG_SIZE,IMG_SIZE,CHANNEL)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

          
            model.add(Dense(len(CATEGORIES),activation="softmax"))
            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            #model.fit(X, y, batch_size=32,
                    #   epochs=10,
                    #   validation_split=0.3,
                    #   callbacks=[tensorboard])
            model.fit_generator(training_set,
                    epochs=10,
                    steps_per_epoch=STEPS_EPOCH_TRAIN,
                    validation_data=val_set,
                    validation_steps=STEPS_EPOCH_TEST,
                    callbacks=[tensorboard]
                   )
model.save(os.path.join('trainedModels','catdog',NAME+".h5"))