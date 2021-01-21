
import os
import PIL
from PIL import Image
from shutil import copyfile

image_dir = '../Dataset/train2019/images/'
data_image = "../Dataset/dataset_color_image_gpu/images/"
delimiter = 1
image_need = 1000
i = 0
for r, d, f in os.walk(image_dir):
    for fi in f:
        if delimiter > image_need:
            break
        if fi.endswith('.jpg'):
            print(fi)
            file = os.path.join(r, fi)
            dest = str(data_image + fi)
            copyfile(file, dest)

            # conver_img = Image.open(file).convert('L')
            # conver_img.save(dest)

        delimiter += 1
        i += 1
        print(i)





