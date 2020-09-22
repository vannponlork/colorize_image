from __future__ import print_function
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter
from networknn4 import Generator, Discriminator
from model import UNet, DNet
import numpy as np
import cv2
from torchsummary import summary
from shutil import rmtree
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}





workers = 0
num_epochs = 10000
lr = 0.001
beta1 = 0.5
ngpu = 1
last_layer = ''
tar = 0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
use_cuda = True if (torch.cuda.is_available() and ngpu > 0) else False

GEN_MODEL_PATH = './model/gen.pt'
DISC_MODEL_PATH = './model/dis.pt'
epoch_note = './modelnn1/epoch_note.pt'
print("Train process start...")

def count_data(path):
    i = 0
    for r, d, f in os.walk(path):
        for fi in f:
            if fi.endswith('.jpg'):
                i += 1
    return i
if torch.cuda.is_available():
    dataroot = "../Dataset/dataset_color_image_gpu/"
    batch_size = 2
    image_size = 256
    print('image to train %s' % count_data(dataroot))
else:
    dataroot = "../Dataset/dataset_color_image_pc/"
    batch_size = 2
    image_size = 256
    print('image to train %s' % count_data(dataroot))

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                           ]))

len_dataset = len(dataset)
x_train_rate = int(len_dataset * 0.8)

data_train = torch.utils.data.Subset(dataset, range(x_train_rate))
data_test = torch.utils.data.Subset(dataset, range(x_train_rate, len_dataset))
dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=workers)
dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=workers)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




print('==' * 100)
netg = Generator(ngpu, use_cuda).to(device)
summary(netg, (1, image_size, image_size))


if (device.type == 'cuda') and (ngpu > 1):
    netg = nn.DataParallel(netg, list(range(ngpu)))
netg.apply(weights_init)

netd = Discriminator(ngpu, use_cuda).to(device)
summary(netd, (3, image_size, image_size))

if (device.type == 'cuda') and (ngpu > 1):
    netd = nn.DataParallel(netd, list(range(ngpu)))
netd.apply(weights_init)

criterion = nn.MSELoss()
d_criterion = nn.BCELoss()
g_criterion_1 = nn.BCELoss()
g_criterion_2 = nn.MSELoss()


real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(netd.parameters(), lr=0.0002, betas=(beta1, 0.999))
optimizerG = optim.Adam(netg.parameters(), lr=0.0002, betas=(beta1, 0.999))

img_list = []
g_losses = []
d_losses = []

iters = 0
g_lambda = 1000
d_running_loss = 0.0
g_running_loss = 0.0


test_data = [t_data for i, t_data in enumerate(dataloader_test)]
j = 0
k = 0
iteration = 1

if os.path.isfile(GEN_MODEL_PATH):
    netg.load_state_dict(torch.load(GEN_MODEL_PATH, map_location=torch.device('cpu')))
    print('Generator model is loaded ...')

if os.path.isfile(DISC_MODEL_PATH):
    netd.load_state_dict(torch.load(DISC_MODEL_PATH, map_location=torch.device('cpu')))
    print('Discriminator model is loaded ...')

def add_to_summary(lab_two_rgb, lab_two_rgb_test,
                    gray_image,
                    err_d_loss,
                    err_d_fake,
                    err_d_real,
                    err_g,
                    generated_image,
                    test_generated_image,
                    writer,
                    x_batch,
                    y_batch):

    gray_image = torchvision.utils.make_grid(gray_image)
    lab_two_rgb = torchvision.utils.make_grid(lab_two_rgb)
    grid_gen = torchvision.utils.make_grid(generated_image)

    test_grid_gen = torchvision.utils.make_grid(test_generated_image)

    grid_ori = torchvision.utils.make_grid(x_batch)
    test_grid_ori = torchvision.utils.make_grid(y_batch)
    lab_two_rgb_test = torchvision.utils.make_grid(lab_two_rgb_test)


    writer.add_image('grey_image', gray_image, iteration)
    writer.add_image('RGB from Lab', lab_two_rgb, iteration)
    writer.add_image('generated Lab image', grid_gen, iteration)


    writer.add_image('test_generated', test_grid_gen, iteration)
    writer.add_image('RGB from Lab for test image', lab_two_rgb_test, iteration)

    writer.add_image('original', grid_ori, iteration)
    writer.add_image('test_data', test_grid_ori, iteration)

    writer.add_scalar('Loss_G_MSELoss', err_g.item(), iteration)
    writer.add_scalar('Loss_D_BCE_TOTAL', err_d_loss.item(), iteration)
    writer.add_scalar('Loss_D_FAKE', err_d_fake.item(), iteration)
    writer.add_scalar('Loss_D_REAL', err_d_real.item(), iteration)

def count_data(path):
    i = 0
    for r, d, f in os.walk(image_dir):
        for fi in f:
            if fi.endswith('.jpg'):
                i += 1
    return i

def convert_grey_image(x):
    pil_images = [transforms.ToPILImage()(img) for img in x.cpu()]
    grey_image = [transforms.ToTensor()(img.convert('L')) for img in pil_images]
    img = grey_image

    return torch.stack(img).to(device)

def convert_torch_tensor(x):
    img = [transforms.ToTensor()(img) for img in x]
    return torch.stack(img).to(device)

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.CenterCrop(256),
        transforms.ToTensor(),

    ])
    image = image_bytes
    return my_transforms(image).unsqueeze(0)
def merge_convert_lab2rgb(image_list):
    g_img_list = []
    for k, color_image in enumerate(image_list):
        # color_image = torch.cat((f, img_l[k]), 0).cpu().detach().numpy()
        color_image = color_image.cpu().detach().numpy()
        color_image = color_image.transpose((1, 2, 0))

        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
        g_img = lab2rgb(color_image.astype(np.float64))
        g_img = g_img.transpose(2, 0, 1)
        g_img = torch.from_numpy(g_img)
        g_img_list.append(g_img)
    return torch.stack(g_img_list).to(device)
def get_l_and_ab_color(x_batch):
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

writer = SummaryWriter()

for epoch in range(1, num_epochs):
    for i, data in enumerate(dataloader_train):
        netd.zero_grad()
        if j > len(test_data) - 1:
            j = 0
        x_batch = data[0].to(device)
        y_batch = test_data[j][0].to(device)
        # l_images = convert_grey_image(x_batch)
        # l_images_test = convert_grey_image(y_batch)

        real_image = x_batch
        grey_image, c_image = get_l_and_ab_color(x_batch)
        grey_image_test, c_image_test = get_l_and_ab_color(y_batch)

        g_output_ab = netg(grey_image)
        g_output_ab_test = netg(grey_image_test)

        fake_image = torch.cat([grey_image, g_output_ab], 1)
        fake_image_test = torch.cat([grey_image_test, g_output_ab_test], 1)

        lab_two_rgb = merge_convert_lab2rgb(fake_image)
        lab_two_rgb_test = merge_convert_lab2rgb(fake_image_test)


        label_real = torch.full((real_image.size(0),), real_label, device=device)
        label_fake = torch.full((fake_image.size(0),), fake_label, device=device)

        d_real_logits = netd(torch.cat([grey_image, c_image], 1))
        D_x = d_real_logits.mean().item()

        d_real_loss = d_criterion(d_real_logits, label_real)
        d_fake_logits = netd(torch.cat([grey_image, g_output_ab], 1))
        d_fake_loss = d_criterion(d_fake_logits, label_fake)
        D_G_z1 = d_fake_logits.mean().item()
        d_loss = d_real_loss + d_fake_loss

        d_loss.backward(retain_graph=True)
        optimizerD.step()
        netg.zero_grad()

        g_real_logits = netd(torch.cat([grey_image, g_output_ab], 1))

        g_real_loss = g_criterion_1(g_real_logits, (torch.ones(batch_size).to(device)))
        g_mse = g_criterion_2(g_output_ab, c_image)
        g_image_distance_loss = g_lambda * g_criterion_2(g_output_ab, c_image)
        g_loss = g_real_loss + g_image_distance_loss
        g_loss.backward()
        optimizerG.step()

        D_G_z2 = g_real_logits.mean().item()

        d_real = d_real_loss
        d_fake = d_fake_loss
        if i % 1 == 0:
           print('[Epoch:{} Step:{}] <--> d_loss:{} g_lambda: {} g_loss: {} g_real_loss:{}'
                 ' g_image_distance_loss: {} g_mse: {}'
                 .format(epoch, i+1, d_loss, g_lambda, g_loss, g_real_loss, g_image_distance_loss, g_mse))
        # add_to_summary(lab_two_rgb, lab_two_rgb_test, grey_image, d_loss, d_fake, d_real,
        #                g_loss, fake_image, fake_image_test, writer, x_batch, y_batch)

        add_to_summary(lab_two_rgb, lab_two_rgb_test, grey_image, d_loss, d_fake, d_real,
                       g_loss, fake_image, fake_image_test, writer, x_batch, y_batch)

        j += 1
        iteration += 1

    if epoch % 5 == 0:
        if os.path.isfile(GEN_MODEL_PATH):
            os.remove(GEN_MODEL_PATH)
        if os.path.isfile(DISC_MODEL_PATH):
            os.remove(DISC_MODEL_PATH)

        torch.save(netg.state_dict(), GEN_MODEL_PATH)
        torch.save(netd.state_dict(), DISC_MODEL_PATH)

writer.close()