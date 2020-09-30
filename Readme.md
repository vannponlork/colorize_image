# Image Colorization with DCGAN 


## Prerequisites
- Linux
- pytorch
- NVIDIA GPU (8G8) + CUDA cuDNN

## Getting Started
### Installation
- Reference code:
```bash
https://github.com/vannponlork/colorize_image
```
### Dataset
- you can use [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [Places365](http://places2.csail.mit.edu) datasets. To train a model on the full dataset, download datasets from official websites.
After downloading, put then under the `datasets` folder.
- For this task I use 10000 image for dataset

### Training
- To train the model, run `nn4.py`
- Network is networknn4.py
```bash
python3 nn4.py
```


```bash

```


## Networks Architecture
The architecture of generator is inspired by  [U-Net](https://arxiv.org/abs/1505.04597):  The architecture of the model is symmetric, with `n` encoding units and `n` decoding units. The contracting path consists of 4x4 convolution layers with stride 2 for downsampling, each followed by batch normalization and Leaky-ReLU activation function with the slope of 0.2. The number of channels are doubled after each step. Each unit in the expansive path consists of a 4x4 transposed convolutional layer with stride 2 for upsampling, concatenation with the activation map of the mirroring layer in the contracting path, followed by batch normalization and ReLU activation function. The last layer of the network is a 1x1 convolution which is equivalent to cross-channel parametric pooling layer. We use `tanh` function for the last layer.
