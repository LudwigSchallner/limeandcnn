#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
#from keras.models import model_from_json
from datetime import datetime
from keras.applications import inception_v3 as inc_net
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D 
from keras.models import load_model
from keras.callbacksTest import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam, Adamax
from keras.preprocessing.image import ImageDataGenerator
import os


# In[2]:


img_size = 299
my_epochs_basic = 3
my_epochs_retrain = 11
is_shuffle = True
write_batch_per = True

data_type = sys.argv[1]
prefix = sys.argv[2]
if(data_type == 'mushrooms128'):
    num_classes = 128
    my_batch_size = 128
    my_steps_per_epoch = 850
    my_val_steps = 18
elif(data_type == 'mushrooms24'):
    num_classes = 24
    my_batch_size = 64
    my_steps_per_epoch = 376
    my_val_steps = 18
elif(data_type == 'flowers'):
    num_classes = 5
    my_batch_size = 16
    my_steps_per_epoch = 188
    my_val_steps = 42
elif(data_type == 'tabak'):
    num_classes = 2
    my_batch_size = 16
    my_steps_per_epoch = 40
    my_val_steps = 4
    my_epochs_basic = 3
    my_epochs_retrain = 30
    
data_path_train = os.path.join('data',data_type,'train')
data_path_test = os.path.join('data',data_type,'test')


# In[3]:


#LOAD BASE MODEL 
base_model = inc_net.InceptionV3(weights='imagenet', include_top=False)
'Model Loaded'


# In[4]:


#train / test data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range = 0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set= train_datagen.flow_from_directory(data_path_train,
                                                target_size = (img_size, img_size),
                                                batch_size=my_batch_size,
                                                class_mode='categorical')
val_set = test_datagen.flow_from_directory(data_path_test,
                                           target_size= (img_size, img_size),
                                           batch_size=my_batch_size,
                                           class_mode='categorical')


# In[5]:


"""
top_model = Sequential()
#top_model.add(Flatten(input_shape=base_model.output_shape[1:])) obsolet because include_top=False = includes a flatten
#output_shape[1:] because else we get 5 dimension but we need 4
top_model.add(Dense(1024, activation='relu', input_shape=base_model.output_shape[1:]))
top_model.add(MaxPooling2D(pool_size=(2, 2)))
top_model.add(Dropout(0.5))
top_model.add(Dense(1024, activation='relu'))
#top_model.load_weights('InceptionV3_weights.h5')
top_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
top_model.fit_generator(training_set)
"""


# In[6]:


x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have num_classes classes
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

#freezing all layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[7]:


time_of_test = str(datetime.now().strftime("%d-%m-%y %H%M"))
file_name= prefix+time_of_test+'_Epochs'+str(my_epochs_basic)+'_ImageSize'+str(img_size)+'_BatchSize'+str(my_batch_size)

log_path = os.path.join('logs',data_type,'pretrain','')
model_path = os.path.join('trainedModels',data_type,'pretrain','')
if not os.path.exists(log_path):
    os.makedirs(log_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

model.fit_generator(training_set,
                    steps_per_epoch=my_steps_per_epoch,
                    epochs=my_epochs_basic,
                    validation_data=val_set,
                    validation_steps=my_val_steps,
                    shuffle=is_shuffle,
                    callbacks=[
                        TensorBoard(
                            log_dir=log_path,
                            write_batch_performance=write_batch_per)]
                   )


# In[8]:


# SAVE THE CURRENT MODEL
print(model_path+file_name+'.h5')
model.save(model_path+file_name+'.h5')  # creates a HDF5 file 'my_model.h5'
print("Saved model to disk as> "+file_name)


# In[9]:


# LOAD THE CURRENT MODEL
#file_name= '16-07-18 15:22_Epochs3_Steps40_ImageSize299_BatchSize16'
model = load_model(model_path+file_name+'.h5')
print('Model loaded> '+file_name)


# In[10]:


#tbCallBack_finetune = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True
# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

#model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'] )


# In[11]:


time_of_real = str(datetime.now().strftime("%d-%m-%y %H%M"))
file_name = prefix+time_of_real+'__Epochs'+str(my_epochs_retrain)+'_ImageSize'+str(img_size)+'_BatchSize'+str(my_batch_size)
log_path = os.path.join('logs',data_type,'completetrain','')
#model.load_weights('./ModelCheckpoints/26-06-18 16:15__weights.11-2.52.hdf5')

if not os.path.exists(log_path):
    os.makedirs(log_path)

    
model.fit_generator(training_set,
                    steps_per_epoch=my_steps_per_epoch,
                    epochs=my_epochs_retrain, 
                    validation_data=val_set,
                    validation_steps=my_val_steps,
                    shuffle=is_shuffle,
                    #initial_epoch=10,
                    callbacks=[TensorBoard(log_dir=log_path,
                                           write_batch_performance=write_batch_per),
                              ModelCheckpoint('./ModelCheckpoints/'
                                              +time_of_real
                                              +'__weights.{epoch:02d}-val_acc{val_acc:.2f}--train_acc{acc:.2f}.hdf5',
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=False,
                                              save_weights_only=False,
                                              mode='auto',
                                              period=4)
                              ]
                   )


# In[12]:


## SAVE THE MODEL
model_path = os.path.join('trainedModels',data_type,'completetrain','')
if not os.path.exists(model_path):
    os.makedirs(model_path)
print('try to save model'+
      model_path+file_name+'.h5')
model.save( model_path+file_name+'.h5')  # creates a HDF5 file 'my_model.h5'
print("Saved model to disk as> "+file_name)


# In[ ]:




