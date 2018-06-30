import csv

file = open("synonyms.txt", "r")
#print(file.read())
real_and_syonyms = list()
real_and_syonyms = file.read().replace(":","").split("\n\n")

real_and_syonyms_listed = list()
for i in range(len(real_and_syonyms)):
    real_and_syonyms_listed.append(real_and_syonyms[i].split("\n"))
print(real_and_syonyms_listed[0])
for i in range(len(real_and_syonyms_listed)):
    tmp = set(real_and_syonyms_listed[i])
    real_and_syonyms_listed[i] = list(tmp)
print(real_and_syonyms_listed[0])




with open("mushroom_synonyms.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(real_and_syonyms_listed)
