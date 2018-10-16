import numpy as np
import os
from PIL import Image

dir_list = [x[0] for x in os.walk('../data/tabak/train/')][1:]
for j in range(len(dir_list)):
    for filename in os.listdir(dir_list[j]):
        img = Image.open(dir_list[j]+'/'+filename)
        img = img.convert("RGBA")
        datas = img.getdata()

        newData = []
        for item in datas:
            if item[0] <= 8 and item[1] <= 8 and item[2] <= 8:
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)

        img.putdata(newData)
        img.save(filename, "PNG")
