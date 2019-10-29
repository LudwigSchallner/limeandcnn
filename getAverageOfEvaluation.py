from PIL import Image
import xml.etree.ElementTree as ET
import glob
import os

"""
Generate average expert images
"""


def PasteImage(source, target, pos):
    smap = source.load()
    tmap = target.load()
    for i in range(pos[0], pos[0] + source.size[0]): 
        for j in range(pos[1], pos[1] + source.size[1]): 
            sx = i - pos[0]
            sy = j - pos[1]
            tmap[i, j] = smap[sx, sy]
    return target

def generateAverageImages(average_list, BASE_PATH, mycolor=(127,127,127)):
    for i in range(len(average_list)):
        file_name = average_list[i][0].replace(".xml",".png")
        org = Image.open(BASE_PATH+"/org/Parasitized/"+file_name)
        
        cords = average_list[i][1]
        cropped = org.crop(cords)
        x = Image.new(mode='RGB',size=org.size, color=mycolor)
        x = PasteImage(cropped,x,(average_list[i][1][0],average_list[i][1][1]))
        x = x.resize((64,64))
        x.save(BASE_PATH+"/average/"+file_name)
    print("Finished generating average of user selected files")

def getCordsOfXml(xml_path):
    xml_file = (ET.parse(xml_path).getroot())
    return (int(xml_file[6][4][0].text),int(xml_file[6][4][1].text),int(xml_file[6][4][2].text),int(xml_file[6][4][3].text))

def getAllXML(dir_path):
    list_users =[x[0] for x in os.walk(dir_path)][1:]
    xml_list = []
    for i in range(len(list_users)):
        xml_list.append([])
        for xml_file in glob.glob(list_users[i]+"/*.xml"):
            xml_list[i].append((xml_file,getCordsOfXml(xml_file)))
    return list_users,xml_list

def getAverageList(list_users, xml_list):
    average_list = []
    for file_id in range(len(xml_list[0])):
        sum_xmin = 0
        sum_xmax = 0
        sum_ymin = 0
        sum_ymax = 0
        for dir_id in range(len(list_users)):
            sum_xmin += xml_list[dir_id][file_id][1][0]
            sum_xmax += xml_list[dir_id][file_id][1][1]
            sum_ymin += xml_list[dir_id][file_id][1][2]
            sum_ymax += xml_list[dir_id][file_id][1][3]
        
        if (dir_id==len(list_users)-1):
            avg_xmin = int(sum_xmin/len(list_users))
            avg_xmax = int(sum_xmax/len(list_users))
            avg_ymin = int(sum_ymin/len(list_users))
            avg_ymax = int(sum_ymax/len(list_users))
            file_name = xml_list[dir_id][file_id][0][xml_list[dir_id][file_id][0].rfind("/")+1:]
            average_list.append((file_name,(avg_xmin,avg_xmax,avg_ymin,avg_ymax)))
    return average_list


list_users, xml_list =getAllXML("data/UserSelectedImages")
average_list = getAverageList(list_users, xml_list)
BASE_PATH = "data/100TestImages"
generateAverageImages(average_list,BASE_PATH,mycolor=(255,255,255))