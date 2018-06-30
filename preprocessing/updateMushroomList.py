import csv

old_file = open('mushroomworld.csv', 'r')
old_csv = list()
old_csv = old_file.read().split("\n")
mushroomworld_csv_list = list()
for i in range(len(old_csv)):
    mushroomworld_csv_list.append(old_csv[i].split(","))

update_file = open('needtoupdatelist.txt', 'r')
mushrooms_wntu = update_file.read().split("\n")
wntu_list = list()
for i in range(len(mushrooms_wntu)):
    wntu_list.append(mushrooms_wntu[i].split(" = "))

#print(mushroomworld_csv_list[10])
id_list = list()
for i in range(len(mushroomworld_csv_list)):
    for j in range(len(wntu_list)):
        if(mushroomworld_csv_list[i][0]==wntu_list[j][0]):
            id_list.append(i)
            mushroomworld_csv_list[i][0]=wntu_list[j][1]


with open("updated_mushroomlist.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(mushroomworld_csv_list)
