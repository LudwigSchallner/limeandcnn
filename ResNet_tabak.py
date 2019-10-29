import tensorflow as tf
import sys 
import keras
import os
import math
from imutils import paths
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

def create_class_weight(labels_dict,mu=0.15):
    total = sum(labels_dict.values())
    #keys = labels_dict.keys()
    class_weight = dict()

    for key in labels_dict:
        score = math.log(total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

CATEGORIES = ["Gestresst","Gesund"]
BASE_PATH = os.path.sep.join(["data","tabakpflanzen"])
TRAIN_PATH = os.path.sep.join([BASE_PATH,"train"])
TEST_PATH = os.path.sep.join([BASE_PATH,"test"])
#VAL_PATH = os.path.sep.join([BASE_PATH,"val"])
IMG_SIZE = 224
NUM_EPOCH = 50
CHANNEL = 3
BATCH_SIZE = 32
INIT_LR = 0.0001
TRAIN_SIZE = len(list(paths.list_images(TRAIN_PATH)))
#VAL_SIZE = len(list(paths.list_images(VAL_PATH)))
TEST_SIZE = len(list(paths.list_images(TEST_PATH)))
STEPS_EPOCH_TRAIN = int(TRAIN_SIZE / BATCH_SIZE)
#STEPS_EPOCH_VAL = int(VAL_SIZE / BATCH_SIZE)

label_count = {0: len(list(paths.list_images(os.path.sep.join([TRAIN_PATH,CATEGORIES[0]])))),
1: len(list(paths.list_images(os.path.sep.join([TRAIN_PATH,CATEGORIES[1]]))))}
class_weights = create_class_weight(label_count)

train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
	rotation_range=45,
	zoom_range=0.15,
	width_shift_range=0.20,
	height_shift_range=0.20,
	shear_range=0.20,
	horizontal_flip=True,
	fill_mode="nearest")

test_datagen = ImageDataGenerator(rescale=1/255.0)

training_set= train_datagen.flow_from_directory(TRAIN_PATH,
                                                target_size = (IMG_SIZE, IMG_SIZE),
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical')                                        
test_set = test_datagen.flow_from_directory(TEST_PATH,
                                                target_size = (IMG_SIZE, IMG_SIZE),
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical')    
NAME = "tabakpflanzen-{}".format(str(datetime.now().strftime("%d-%m-%y %H%M")))


tensorboard = TensorBoard(log_dir='./logs/{}'.format(NAME))
earlystopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
opt = SGD(lr=INIT_LR, momentum=0.9, nesterov=True)
base_model = ResNet50(include_top=False, input_shape= (IMG_SIZE,IMG_SIZE,3), weights="imagenet")

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = AveragePooling2D()(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax', kernel_regularizer=l2(0.0001))(x)
model = Model(base_model.input, x)



model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
hist = model.fit_generator(training_set,
        class_weight=class_weights,
        epochs=NUM_EPOCH,
        verbose=2,
        steps_per_epoch=STEPS_EPOCH_TRAIN,
        #validation_data=val_set,
        #validation_steps=STEPS_EPOCH_VAL,
        callbacks=[tensorboard,earlystopping]
        )

model.save("./output/tabakpflanzen/ResNet_tabakpflanzen_loss{0:4f}.h5".format(min(hist.history['loss'])))
x = model.evaluate_generator(test_set, steps=(TEST_SIZE/BATCH_SIZE))
print(model.metrics_names)
print(x)