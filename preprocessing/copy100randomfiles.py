import os
import random
import shutil
"""
Script for copy 100 files for users classification
"""
CATEGORIES = ["Parasitized"]
PATH = "./data/cell_images/"
x = set()
for c in CATEGORIES:
    files = os.listdir(PATH+c+"/")
    #for _ in range(0,13):
    random.shuffle(files)

randomlist = set()
while len(randomlist)<100:
    randomlist.add(files[random.randint(0,len(files))])    
for c in CATEGORIES:
    for rnimg in randomlist:
        shutil.copy(PATH+c+'/'+rnimg, PATH+'/choosen/'+rnimg)
     