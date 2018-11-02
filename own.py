import tensorflow as tf 
import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from tqdm import tqdm

print("Tensorflow Version: "+tf.__version__)
print("Keras Version: "+keras.__version__)


DATADIR = os.path.join('data', sys.argv[1])
if sys.argv[1] == "catdog":
    CATEGORIES = ["CAT","DOG"]
else:
    raise NameError("Dataset "+sys.argv[1]+" not supported!")

