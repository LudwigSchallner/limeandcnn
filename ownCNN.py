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
else:
    raise NameError("Dataset "+sys.argv[1]+" not supported!")

NAME = sys.argv[1]+" {}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

pickle_in = open(os.path.join('.',"data","X_"+sys.argv[1]+".pickle"),"rb")
X = pickle.load(pickle_in)

pickle_in = open(os.path.join('.',"data","y_"+sys.argv[1]+".pickle"),"rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(ELU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(ELU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(ELU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(ELU(alpha=0.1))
#model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128))
model.add(ELU(alpha=0.1))
model.add(Dropout(0.3))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, callbacks=[tensorboard], validation_split=0.05)
model.save(NAME+".h5")