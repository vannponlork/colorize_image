from flask import Flask, redirect, url_for, send_from_directory, render_template, flash
import json
import torchvision.transforms as transforms
from flask import Flask, request
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from networknn19 import Generator
from skimage.color import lab2rgb, rgb2lab, rgb2gray, grey2rgb
from skimage.transform import resize
from skimage.io import imsave
import os
from shutil import copyfile
import warnings


app = Flask(__name__)
@app.route('/')
def home():
   return render_template('index.html')

@app.route('/flask')
def hello_flask():
   return 'Hello Flask'

@app.route('/python/')
def hello_python():
   return 'Hello Python'


n_gpu = 1
use_cuda = True if (torch.cuda.is_available() and n_gpu > 0) else False

device = torch.device("cuda:0" if use_cuda else "cpu")
model = Generator(n_gpu, use_cuda).to(device)
# GEN_MODEL_PATH = '/opt/colorize_flask/model/gen.pt'
GEN_MODEL_PATH = 'model/gen.pt'
with torch.no_grad():
    model.load_state_dict(torch.load(GEN_MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

def get_limg(x_batch):
    imgGray = []
    imgAb = []
    for im in x_batch:
        im = np.asarray(im.cpu())
        imgLab = np.transpose(im, [1, 2, 0])
        img_lab = rgb2lab(imgLab)
        img_lab = (img_lab + 128) / 255
        img_ab = img_lab[:, :, 1:3]
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
        img_original = rgb2gray(im.transpose((1, 2, 0)))
        img_original = torch.from_numpy(img_original).unsqueeze(0).float()
        imgAb.append(img_ab)
        imgGray.append(img_original)
    return torch.stack(imgGray).to(device), torch.stack(imgAb).to(device)

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.CenterCrop(256),
        transforms.ToTensor(),

    ])
    image = image_bytes
    return my_transforms(image).unsqueeze(0)
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
@app.route('/predict', methods=['POST', 'GET'])

def predict():
    j = 1
    f = []
    print(request.method)
    if request.method == 'POST':
        print(request.form)
        print(request.url)
        print(request.path)
        print(request.files['file'])

        if 'file' not in request.files:
            print(request.url)
            return redirect(request.url)

        image = request.files['file']
        img = Image.open(image)
        if img.mode == 'L':
            img = img.convert('RGB')
        ori_img = img
        imgName = image.filename

        origin_img_l = Image.open(image).convert('L')
        origin_img = transforms.ToTensor()(origin_img_l).unsqueeze(0)
        w, h = img.size
        tf = transform_image(img).to(device)
        limg, cab = get_limg(tf)

        img_ab = model(limg).cpu().detach().numpy()

        img_ab = img_ab[0].transpose([1, 2, 0])

        img_ab = resize(img_ab, (h, w))
        img_ab = np.float32(img_ab)
        # img_ab = np.transpose(img_ab, [2, 0, 1])
        img_ab = transforms.ToTensor()(img_ab).unsqueeze(0)
        color_image = torch.cat((origin_img, img_ab), 1).cpu().detach().numpy()
        color_image = color_image[0].transpose((1, 2, 0))  # rescale for matplotlib
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
        color_image = lab2rgb(color_image.astype(np.float64))

        img_array = json.dumps({'predictions': color_image.tolist()})
        img = np.asarray(json.loads(img_array)['predictions'], dtype=np.float32)

        imsave('./static/upload/{}'.format(imgName), img)
        # imsave('./static/upload/{}_{}'.format('ori_', imgName), ori_img)
        ori_img.save('./static/upload/{}_{}'.format('ori', imgName))
        j += 1
        image_names = './static/upload/{}'.format(imgName)
        ori_image_names = './static/upload/{}_{}'.format('ori', imgName)
        return render_template('index.html', string_variable=image_names, data=ori_image_names)

@app.route('/predictga', methods=['POST', 'GET'])
def predictga():
    j = 1
    f = []
    print(request.get_json('data'))
    print(request.json['data'])
    if request.method == 'POST':
        request.get_json('data')
        image = request.json['data']

        img = Image.open(image)
        ori_img = img

        imgName = img.filename[-5:]

        origin_img = transforms.ToTensor()(img).unsqueeze(0)

        w, h = img.size
        img = transform_image(img).to(device)
        img_ab = model(img).cpu().detach().numpy()

        img_ab = img_ab[0].transpose([1, 2, 0])

        img_ab = resize(img_ab, (h, w))
        img_ab = np.float32(img_ab)
        # img_ab = np.transpose(img_ab, [2, 0, 1])
        img_ab = transforms.ToTensor()(img_ab).unsqueeze(0)
        color_image = torch.cat((origin_img, img_ab), 1).cpu().detach().numpy()
        color_image = color_image[0].transpose((1, 2, 0))  # rescale for matplotlib
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
        color_image = lab2rgb(color_image.astype(np.float64))

        img_array = json.dumps({'predictions': color_image.tolist()})
        img = np.asarray(json.loads(img_array)['predictions'], dtype=np.float32)

        imsave('./static/upload/{}'.format(imgName), img)
        # imsave('./static/upload/{}_{}'.format('ori_', imgName), ori_img)
        ori_img.save('./static/upload/{}_{}'.format('ori', imgName))
        j += 1
        image_names = './static/upload/{}'.format(imgName)
        ori_image_names = './static/upload/{}_{}'.format('ori', imgName)
        return json.dumps({'prediction': image_names, 'origin': ori_image_names})


if __name__ == '__main__':
   app.run(host='0.0.0.0')
   # app.run()
