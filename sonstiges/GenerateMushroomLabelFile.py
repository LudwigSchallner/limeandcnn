import os

dir_list = [x[0] for x in os.walk('../data/mushrooms_filled/train/')][1:]

name_list = list()
for x in dir_list:
    #print()
    name_list.append(x.rsplit('/', 1)[1])

print(name_list)

with open("mushroom_labels.txt","w+") as output:
    for item in name_list:
        output.write("%s\n" % item)
