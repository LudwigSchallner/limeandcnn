from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os


dir_list = [x[0] for x in os.walk('./data/mushrooms_filled/train/')][1:]
datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

#for each img
for j in range(len(dir_list)):
    count_of_files = len([name for name in os.listdir(dir_list[j]) if os.path.isfile(os.path.join(dir_list[j], name))])
    file_mul = 1000 - count_of_files
    print(file_mul)
    file_mul = file_mul / count_of_files
    print(file_mul)
    for filename in os.listdir(dir_list[j]):
        count_of_filesandnews = len([name for name in os.listdir(dir_list[j]) if os.path.isfile(os.path.join(dir_list[j], name))])
        saved_as = filename.replace(".jpg","")
        print(str(j)+' Folder> '+dir_list[j].rsplit('/', 1)[-1]+' Count of files> '+str(count_of_filesandnews) + " " + str(file_mul))
        img = load_img(dir_list[j]+'/'+filename)  # this is a PIL image
        x = img_to_array(img)  # this is a Numppy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        if (count_of_filesandnews>= 1000):
            break
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=dir_list[j], save_prefix=saved_as, save_format='jpeg'):
            i += 1
            if i > min(file_mul, 20): #i>(1000-anzahl an files im dir)
                break  # otherwise the generator would loop indefinitely
