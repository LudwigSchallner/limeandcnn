import tensorflow as tf
import sys 
import keras
import os
from imutils import paths
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.callbacks import TensorBoard, EarlyStopping
from datetime import datetime
from keras.regularizers import l2
#from pyimagesearch.resnet import ResNet


print("Tensorflow Version: "+tf.__version__)
print("Keras Version: "+keras.__version__)


CATEGORIES = ["Parasitized","Uninfected"]
BASE_PATH = os.path.sep.join(["cell_images"])
BASE_PATH = Path("/mnt/Multimedia/GitHub/Code/data/malaria/")
TRAIN_PATH = BASE_PATH / "train"
TEST_PATH = BASE_PATH / "test"
VAL_PATH = BASE_PATH / "val"
TEST_100_PATH = Path("/run/media/eisbergsalat/Transcend/Code/data/100TestImages/evaluate/")
#TRAIN_PATH = os.path.sep.join([BASE_PATH,"train"])
#TEST_PATH = os.path.sep.join([BASE_PATH,"test"])
#VAL_PATH = os.path.sep.join([BASE_PATH,"val"])
IMG_SIZE = 64
NUM_EPOCH = 50
CHANNEL = 3
BATCH_SIZE = 32
INIT_LR = 0.0001
TRAIN_SIZE = len(list(paths.list_images(TRAIN_PATH)))
VAL_SIZE = len(list(paths.list_images(VAL_PATH)))
TEST_SIZE = len(list(paths.list_images(TEST_PATH)))
STEPS_EPOCH_TRAIN = int(TRAIN_SIZE / BATCH_SIZE)
STEPS_EPOCH_VAL = int(VAL_SIZE / BATCH_SIZE)

train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
	horizontal_flip=True,
    vertical_flip=True,
	fill_mode="nearest")

val_datagen = ImageDataGenerator(rescale=1/255.0)

training_set= train_datagen.flow_from_directory(TRAIN_PATH,
                                                target_size = (IMG_SIZE, IMG_SIZE),
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical')
val_set = val_datagen.flow_from_directory(VAL_PATH,
                                           target_size= (IMG_SIZE, IMG_SIZE),
                                           batch_size=BATCH_SIZE,
                                           class_mode='categorical')
test_set = val_datagen.flow_from_directory(TEST_PATH,
                                           target_size= (IMG_SIZE, IMG_SIZE),
                                           batch_size=BATCH_SIZE,
                                           class_mode='categorical')
test_100_set = val_datagen.flow_from_directory(TEST_100_PATH,
                                            target_size= (IMG_SIZE, IMG_SIZE),
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical'
                                            )                                        
NAME = "malaria-{}".format(str(datetime.now().strftime("%d-%m-%y %H%M")))


tensorboard = TensorBoard(log_dir='./logs/{}'.format(NAME))
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
opt = SGD(lr=INIT_LR, momentum=0.9, nesterov=True)
base_model = ResNet50(include_top=False, input_shape= (IMG_SIZE,IMG_SIZE,3), weights='imagenet')

x = base_model.output
x = AveragePooling2D()(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax', kernel_regularizer=l2(0.0001))(x)
model = Model(base_model.input, x)



model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
hist = model.fit_generator(training_set,
        epochs=NUM_EPOCH,
        verbose=2,
        steps_per_epoch=STEPS_EPOCH_TRAIN,
        validation_data=val_set,
        validation_steps=STEPS_EPOCH_VAL,
        callbacks=[tensorboard,earlystopping]
        )

model.save("./output/cell_images/ResNet_valloss{0:.4f}.h5".format(min(hist.history['val_loss'])))
x = model.evaluate_generator(test_set, steps=(TEST_SIZE/BATCH_SIZE))
print(model.metrics_names)
print(x)
x = model.evaluate_generator(test_100_set, steps=(TEST_SIZE/BATCH_SIZE))
print(model.metrics_names)
print(x)