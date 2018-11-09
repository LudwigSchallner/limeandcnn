import os
import numpy as np
import sys
from tqdm import tqdm
import random
import pickle
import cv2 


DATADIR_TRAIN = os.path.join('.','data', sys.argv[1],'train')
DATADIR_TEST = os.path.join('.','data', sys.argv[1],'test')
if sys.argv[1] == "catdog":
    CATEGORIES = ["Cat","Dog"]
    IMG_SIZE = 100
else:
    raise NameError("Dataset "+sys.argv[1]+" not supported!")

training_data = []
test_data = []

def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR_TRAIN,category) 
        print(path) 
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)): 
            try:
                img_array = cv2.imread(os.path.join(path,img))# ,cv2.IMREAD_GRAYSCALE)  
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                training_data.append([new_array, class_num]) 
            except OSError as e:
                print("OSError Bad img most likely", e, os.path.join(path,img))
            except Exception as e:
                print("general exception", e, os.path.join(path,img))

def create_test_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR_TEST,category) 
        print(path) 
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)): 
            try:
                img_array = cv2.imread(os.path.join(path,img))# ,cv2.IMREAD_GRAYSCALE)  
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

create_training_data()
create_test_data()

print(len(training_data))
print(len(test_data))

random.shuffle(training_data)

X = [] #features
y = [] #label

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)#1)

#X
pickle_out = open(os.path.join('.',"data",sys.argv[1],"X_train_"+sys.argv[1]+".pickle"),"wb")
pickle.dump(X, pickle_out)
pickle_out.close()
#y
pickle_out = open(os.path.join('.',"data","y_train_"+sys.argv[1]+".pickle"),"wb")
pickle.dump(y, pickle_out)
pickle_out.close()

