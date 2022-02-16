import os
import time
import zipfile
import random
import imageio
import glob
import copy
import torch
import shutil
import torchvision
import cv2

import torch.utils.data as data
from collections import defaultdict
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn.functional as F
from PIL import Image
from torch import nn

from DataPipeline import *



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TRAIN_PATH = 'data'

# commented the Normalize transformation of TrainTtransform.py
train_dataset = DataLoaderSegmentation(TRAIN_PATH, transform=get_train_transform())

image, mask = train_dataset.__getitem__(0)
# print(image.shape)
# print(mask.shape)

# print(train_dataset.__len__())

visualize_dataset(train_dataset, 3)