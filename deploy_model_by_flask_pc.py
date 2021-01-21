import json
import torchvision.transforms as transforms
from flask import Flask, request
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from networknn4 import Generator
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage.transform import resize

app = Flask(__name__)
n_gpu = 1
use_cuda = True if (torch.cuda.is_available() and n_gpu > 0) else False


device = torch.device("cuda:0" if use_cuda else "cpu")
model = Generator(n_gpu, use_cuda).to(device)
GEN_MODEL_PATH = 'model/gen.pt'
# model.load_state_dict(torch.load(GEN_MODEL_PATH, map_location={'cuda:0': 'cpu'}))
model.load_state_dict(torch.load(GEN_MODEL_PATH, map_location=torch.device('cpu')))

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.CenterCrop(256),
        transforms.ToTensor(),

    ])
    image = image_bytes
    return my_transforms(image).unsqueeze(0)

@app.route('/predict', methods=['POST'])
# def convert_grey_image(x):
#     pil_images = [transforms.ToPILImage()(img) for img in x.cpu()]
#     grey_image = [transforms.ToTensor()(img.convert('L')) for img in pil_images]
#     img = grey_image
#
#     return torch.stack(img).to(device)

def predict():
    j = 1
    f = []
    if request.method == 'POST':
        image = request.files['file']
        img = Image.open(image)
        origin_img = transforms.ToTensor()(img).unsqueeze(0)

        w, h = img.size
        img = transform_image(img).to(device)
        img_ab = model(img).cpu().detach().numpy()

        img_ab = img_ab[0].transpose([1, 2, 0])

        img_ab = resize(img_ab, (h, w))
        # img_ab = np.transpose(img_ab, [2, 0, 1])
        img_ab = transforms.ToTensor()(img_ab).unsqueeze(0)

        color_image = torch.cat((origin_img, img_ab), 1).cpu().detach().numpy()
        color_image = color_image[0].transpose((1, 2, 0))  # rescale for matplotlib
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
        color_image = lab2rgb(color_image.astype(np.float64))
        j += 1
        return json.dumps({'predictions': color_image.tolist()})

if __name__ == '__main__':
    app.run()