import os
from PIL import Image
from shutil import copyfile

image_dir = '../Dataset/traindata/test/images/'
data_image = '../Dataset/l_images/'
delimiter = 1
image_need = 2287
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
            convert_img = Image.open(file).convert('L')
            convert_img.save(dest)
            delimiter += 1
            i += 1
            print(i)