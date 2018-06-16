import shutil
import os
import csv

#synonym_list = list(csv.reader(open('mushroom_synonyms.csv', 'r')))
path_list = list([x[0] for x in os.walk('../data/mushrooms_filled/train/')][1:])
for i in range(len(path_list)):
    dest = path_list[i].replace("train","test")+'/'
    source = path_list[i]+'/'
    files = os.listdir(source)
    os.mkdir(dest)
    for k, f in enumerate(files):
        if not k%10:
            shutil.move(source+f,dest)
