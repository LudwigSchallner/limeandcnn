import os
import numpy as np
import sys
from tqdm import tqdm
import pickle
import cv2

DATADIR= os.path.join('.','data', sys.argv[1])
if sys.argv[1] == "catdog":
    CATEGORIES = ["Cat","Dog"]
    IMG_SIZE = 100
else:
    raise NameError("Dataset "+sys.argv[1]+" not supported!")

def delete_files():
    count_bad = 0
    count_general = 0
    for category in CATEGORIES:  
        path = os.path.join(DATADIR,category) 
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)): 
            try:
                img_array = cv2.imread(os.path.join(path,img))# ,cv2.IMREAD_GRAYSCALE)  
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                #print(path+img)
            except OSError as e:
                count_bad = count_bad+1
                print("OSError Bad img most likely", e, os.path.join(path,img))
            except Exception as e:
                count_general = count_general+1
                print("general exception", e, os.path.join(path,img))
                os.remove(os.path.join(path,img))
    print(count_bad)
    print(count_general)


delete_files()
