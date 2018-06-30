import shutil
import os
import csv

synonym_list = list(csv.reader(open('mushroom_synonyms.csv', 'r')))



for i in range(len(synonym_list)):
    dest1 = '../data/mushrooms_with_seperates_synonms/'+synonym_list[i][0]+'/'
    print(dest1)
    for j in range(len(synonym_list[i])):
        if (j != 0):
            source = '../data/mushrooms_with_seperates_synonms/'+synonym_list[i][j]+'/'
            files = os.listdir(source)
            for f in files:
                    shutil.move(source+f, dest1)
            shutil.rmtree(source)
