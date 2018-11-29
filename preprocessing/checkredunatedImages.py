import sys
import os
import glob
import numpy
import time
import datetime


start_time = time.time()
print(datetime.datetime.now().time())

img_dir = sys.argv[1] 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
something_went_wrong = False

for f1 in files:
    for f2 in files:
        if f1 != f2:
            if open(f1,"rb").read() == open(f2,"rb").read():
                something_went_wrong = True
                print(f1)
                print(f2)
                print("---")

print(something_went_wrong)
print("--- %s seconds ---" % (time.time() - start_time))