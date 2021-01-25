import requests
import os
import json
from PIL import Image
from skimage.io import imsave
import numpy as np


image_need = 100
delimiter = 1
j = 1
dirs = '../Dataset/l_images/'
image_list = []
file_name = []
for r, d, f in os.walk(dirs):
    for fi in f:
        if delimiter > image_need:
            break
        if fi.endswith('.jpg'):
            image_list.append(os.path.join(r, fi))
            file_name.append(fi)
        delimiter += 1
for j, image in enumerate(image_list):
    resp = requests.post("http://127.0.0.1:5000/predict",
                         files={"file": open(image, 'rb')})
    img_mode = Image.open(image).mode

    if resp.status_code == 200:
        img = np.asarray(json.loads(resp.text)['predictions'], dtype=np.float32)
        imsave('./images_predict/%s' % file_name[j], img)
        colorImage = Image.open('./images_predict/%s' % file_name[j])
        image_rotate = colorImage
        image_rotate.save('./images_predictnn1/%s' % file_name[j])
    else:
        print('file:{} status:{}'.format(image, resp.status_code))
        print('Status: %s, Image_mode: %s' % (resp.status_code, img_mode))