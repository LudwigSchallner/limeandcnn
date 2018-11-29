import sys
import os
import glob
import numpy
import time
import datetime
import hashlib


def hash_file(filename):
   # use sha1 or sha256 or other hashing algorithm
   h = hashlib.sha1()

   # open file and read it in chunked
   with open(filename,'rb') as file:
       chunk = 0
       while chunk != b'':
           chunk = file.read(1024)
           h.update(chunk)

   # return string
   return h.hexdigest()

start_time = time.time()
img_dir = sys.argv[1] 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)

list_files = list()
for f1 in files:
    list_files.append(hash_file(f1))
print(len(list_files))
list_files = set(list_files)
print(len(list_files))
""""
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
"""
print("--- %s seconds ---" % (time.time() - start_time))