import torch 
import numpy as np 
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torchvision
import cv2

from torch.utils.data import DataLoader
from unet_model import UNet
from cityscapes_dataset import CityscapesDataset
from torch.autograd import Variable  


# TODO(AJ): Verify that there are no duplicate labels in the cityscape labels e.g bicycle and bicycle_group
DATA_DIR = "./../cityscapes_data"
ANNOTATION_DATA_DIR = DATA_DIR + "/gtFine"
IMG_DATA_DIR = DATA_DIR + "/leftImg8bit"
TRAIN_DIR_IMG = IMG_DATA_DIR + "/train"
TRAIN_DIR_ANN = ANNOTATION_DATA_DIR + "/train"

_NUM_EPOCHS_ = 50
_NUM_CHANNELS_= 3
_IMAGE_SIZE_ = 250 #Ideal image size should be 3000 for final training using all channels
_COMPUTE_DEVICE_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")