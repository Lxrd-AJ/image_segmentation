import torch 
import numpy as np 
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torchvision
import cv2
import json
from torch.utils.data import DataLoader
from model.unet_model import UNet
from ETL.cityscapes_dataset import CityscapesDataset, ToTensor, ToNumpy, Rescale, Resize
from ETL.cityscapes_labels import name2label, labels
from torch.autograd import Variable  
from torchvision import transforms, utils


# TODO(AJ): Verify that there are no duplicate labels in the cityscape labels e.g bicycle and bicycle_group
# TODO(AJ): Add Tensorboard support for visualisations, Try visdom
DATA_DIR = "./cityscapes_data"
ANNOTATION_DATA_DIR = DATA_DIR + "/gtFine"
IMG_DATA_DIR = DATA_DIR + "/leftImg8bit"
TRAIN_DIR_IMG = IMG_DATA_DIR + "/train"
TRAIN_DIR_ANN = ANNOTATION_DATA_DIR + "/train"

_NUM_EPOCHS_ = 1
_NUM_CHANNELS_= 3
_IMAGE_SIZE_ = (500,500) 
_COMPUTE_DEVICE_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_tensor_type(torch.FloatTensor)

if __name__ == "__main__":
    categories = [label.name for label in labels]#list(name2label.keys())
    print("Training categories:")
    print(categories)
    cityscapes_dataset = CityscapesDataset(
        TRAIN_DIR_IMG, TRAIN_DIR_ANN, "gtFine", 
        categories, transform=transforms.Compose([Resize(_IMAGE_SIZE_), Rescale(), ToTensor()]))
    trainloader = DataLoader(cityscapes_dataset, batch_size=5, shuffle=True, num_workers=4)

    model = UNet( n_classes=len(categories), in_channels=_NUM_CHANNELS_ )
    if torch.cuda.device_count() >= 1:
        print("Training model on ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # TODO: use soft dice loss as it performs better but BCELoss still works fine https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c
    criterion = nn.BCELoss()
    optimizer = optim.SGD( model.parameters(), lr=0.001, momentum=0.9 )
    
    # Network training
    epoch_data = {}
    toNumpy = ToNumpy()
    for epoch in range(_NUM_EPOCHS_):
        epoch_loss = 0.0
        epoch_data[epoch] = {}
        for i, data in enumerate(trainloader,0):
            inputs = data["image"].to(_COMPUTE_DEVICE_).type(torch.FloatTensor) #The images are returned as ByteTensor
            labels = data["segments"].to(_COMPUTE_DEVICE_).type(torch.FloatTensor)
            #TODO(AJ)!!: Fix the bug in `toNumpy` operation
            # true_segments = toNumpy(data["segmented_image"]) #for your eyes only
            #TODO(AJ): Visualise the outputs of the network (log to visdom)
            #TODO(AJ): Visualise the weights of the network (log to visdom/tensorboard etc)
            optimizer.zero_grad()#zero the parameter gradients

            # forward pass + backward pass + optimisation
            outputs = model(inputs)
            loss = criterion( outputs, labels )
            loss.backward()
            optimizer.step()

            epoch_loss = loss.item()
            # print("Training iteration {:} => Loss ({:})".format(i,epoch_loss))
        epoch_data[epoch]["loss"] = epoch_loss
        #TODO: Save the model on every epoch
        print("[Epoch %d] loss: %.2f" % (epoch+1, epoch_loss))
    print("Training complete .....")

    with open("epoch_data.json",'w') as file:
        json.dump(epoch_data, file)

    # TODO(AJ) Test the network

    # TODO(AJ): Utilise the trained network to perform object detection