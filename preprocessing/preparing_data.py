import os
import numpy as np
import sys
from tqdm import tqdm
import random
import pickle
import cv2 


DATADIR = os.path.join('.','data', sys.argv[1],'train')
if sys.argv[1] == "catdog":
    CATEGORIES = ["Cat","Dog"]
    IMG_SIZE = 100
else:
    raise NameError("Dataset "+sys.argv[1]+" not supported!")

training_data = []

def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category) 
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

create_training_data()
print(len(training_data))

random.shuffle(training_data)

X = [] #features
y = [] #label

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)#1)

#X
pickle_out = open(os.path.join('.',"data","X_"+sys.argv[1]+".pickle"),"wb")
pickle.dump(X, pickle_out)
pickle_out.close()
#y
pickle_out = open(os.path.join('.',"data","y_"+sys.argv[1]+".pickle"),"wb")
pickle.dump(y, pickle_out)
pickle_out.close()

