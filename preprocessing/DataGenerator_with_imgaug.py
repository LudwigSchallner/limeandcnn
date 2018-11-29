import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
import os
import glob
import sys
import time
import datetime


start_time = time.time()
# random example images
print(datetime.datetime.now().time())

img_dir = "./data/tabak_new/train/"+sys.argv[1]+"/" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
images = []
name_of_files = []
for f1 in files:
    img = Image.open(f1)
    file_name = f1.rsplit('/',1)[1].replace(".png","")
    img.save("./data/datagen/"+sys.argv[1]+"/"+file_name+".png")
    images.append(np.asarray(img,dtype=np.uint8))
    name_of_files.append(file_name)
    

"""    
images = [np.asarray(Image.open('/run/media/eisbergsalat/Transcend/Code/data/tabak_new/train/Gut/3_53Tage.png'),dtype=np.uint8),
            np.asarray(Image.open('/run/media/eisbergsalat/Transcend/Code/data/tabak_new/train/Schlecht/33_55Tage.png'),dtype=np.uint8)]
"""
# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf(3,
            [
                #sometimes(iaa.Superpixels(p_replace=(0.10), n_segments=(300))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0.25, 1.5)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 5)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(1, 3)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.25)), # sharpen images

                #iaa.AdditiveGaussianNoise(loc=0, scale=(0.03*255, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                    #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                
                #iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-30, 30)), # change brightness of images (by -10 to 10 of original value)
                iaa.Add((-20, 20), per_channel=0.5),
                iaa.AddElementwise((-40, 40)),
                iaa.AddToHueAndSaturation((-10, 10)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.DirectedEdgeDetect(alpha=(0.0, 0.625), direction=(0.0)),
                iaa.Multiply((0.8, 1.8), per_channel=0.5),
                
                iaa.ContrastNormalization((0.5, 1.5)), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.2, 1.0))#,
                #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
               
            ],
            random_order=True
        ),
        iaa.OneOf([iaa.PiecewiseAffine(scale=(0.01, 0.05)), # sometimes move parts of the image around                
                iaa.Fliplr(0.5)]),
        iaa.OneOf([iaa.PerspectiveTransform(scale=(0.02, 0.08)),
                iaa.Affine(rotate=(-25,25)),])
    ],
    random_order=True
)

for i in range(len(images)):
    #images_aug = seq.augment_images(images)
    for count in range(17):
        images_aug = seq.augment_image(images[i])
        #ia.imshow(images_aug[x])
        im = Image.fromarray(images_aug)#[i])
        im.save("./data/datagen/"+sys.argv[1]+"/"+name_of_files[i]+"_"+str(count)+".png")

print("--- %s seconds ---" % (time.time() - start_time))