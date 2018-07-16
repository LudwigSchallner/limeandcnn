from os import listdir, path
from os.path import isfile, join
import shutil
import csv

dirpath = path.abspath(path.join('..', 'labels','.'))
csvpath = path.abspath(path.join(dirpath, "Tabak_all_fortraining.csv"))
with open(csvpath, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
# count_g = 0
# count_s = 0
# count_else = 0
# for x in data:
#     if(x[1] == 'g'):
#         count_g += 1
#     elif(x[1]=='s'):
#         count_s += 1
#     else:
#         count_else +=1
#
# print(count_g)
# print(count_s)
# print(count_else)
filespath = path.abspath(path.join('..', 'data','tabak'))
path_gut = path.join(filespath,'Gut')
path_schlecht = path.join(filespath,'Schlecht')
print(path_gut)
onlyfiles = [f for f in listdir(filespath) if isfile(join(filespath, f))]
for x in onlyfiles:
    file_id = int(x.split('_')[0])-1
    status = data[file_id][1]
    if status == 'g':
        #path.join(path_schlecht,x)
        shutil.move(path.join(filespath,x),path.join(path_gut,x))
    else:
        shutil.move(path.join(filespath,x),path.join(path_schlecht,x))
