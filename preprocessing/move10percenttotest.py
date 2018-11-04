import shutil
import os

#synonym_list = list(csv.reader(open('mushroom_synonyms.csv', 'r')))
path_list = ["./data/catdog/train/Cat","./data/catdog/train/Dog"]#list([x[0] for x in os.walk('../data/catdog/train/')][1:])
print(path_list)
for i in range(len(path_list)):
    dest = path_list[i].replace("train","test")+'/'
    source = path_list[i]+'/'
    files = os.listdir(source)
    #os.mkdir(dest)
    for k, f in enumerate(files):
        if not k%10:
            shutil.move(source+f,dest)
