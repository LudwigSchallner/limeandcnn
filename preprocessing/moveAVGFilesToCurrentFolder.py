import os
import shutil


avg_dir = "./data/100TestImages/average"
dst_dir = "./data/lime_erg/cell_images/27-03-2019 1254"
for root, directories, filenames in os.walk(avg_dir+"/"):
    for filename in filenames:
        shutil.copy(avg_dir+"/"+filename,dst_dir+"/"+filename[:-4]+"/average.png")

