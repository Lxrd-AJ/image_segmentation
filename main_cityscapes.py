import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F  
import numpy as np 
import os
import matplotlib.pyplot as plt
import torchvision
import cv2
import json
import psutil
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.unet_model import UNet
from ETL.cityscapes_dataset import CityscapesDataset, ToTensor, ToNumpy, Normalize, Resize
from ETL.cityscapes_labels import name2label, labels
from torch.autograd import Variable  
from torchvision import transforms, utils
from PIL import Image 
from statistics import mean 

def soft_dice_loss(predictions, target):    
    eps = 1e-6
    b = predictions.size(0)
    x = predictions.view(b,-1)
    y = target.view(b,-1)    
    intersection = 2 * (x * y).sum()
    union = torch.pow(x,2).sum() + torch.pow(y,2).sum()    
    score = (intersection / (union + eps)) / b #average over the batch    
    loss = 1 - score    
    return loss

DATA_DIR = "./cityscapes_data"
ANNOTATION_DATA_DIR = DATA_DIR + "/gtFine"
IMG_DATA_DIR = DATA_DIR + "/leftImg8bit"
TRAIN_DIR_IMG = IMG_DATA_DIR + "/train"
TRAIN_DIR_ANN = ANNOTATION_DATA_DIR + "/train"

_NUM_EPOCHS_ = 20
_NUM_CHANNELS_= 3
_IMAGE_SIZE_ = (800,800) 
_COMPUTE_DEVICE_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_tensor_type(torch.FloatTensor)

writer = SummaryWriter()

if __name__ == "__main__":
    categories = [label.name for label in labels if label.name == 'car']#list(name2label.keys())    
    print(f"Training categories {len(categories)}:")
    print(categories)
    cityscapes_dataset = CityscapesDataset(
        TRAIN_DIR_IMG, TRAIN_DIR_ANN, "gtFine", 
        categories, transform=transforms.Compose([
            Resize(_IMAGE_SIZE_), 
            Normalize(), 
            ToTensor(),            
            #TODO: Apply random color changes
            #TODO: Apply random spatial changes (rotation, flip etc)
            ]))
    trainloader = DataLoader(cityscapes_dataset, batch_size=8, shuffle=True, num_workers=4)

    model = UNet( n_classes=len(categories), in_channels=_NUM_CHANNELS_, writer=writer )
    if torch.cuda.device_count() >= 1:
        print("Training model on ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # softmax = nn.Softmax2d()
    # criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam( model.parameters(), lr=0.001, weight_decay=0.0001 )
    
    # Network training
    epoch_data = {}
    float_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    for epoch in range(_NUM_EPOCHS_):
        epoch_loss = 0.0
        epoch_data[epoch] = {}
        losses = []
        for i, data in enumerate(trainloader,0):
            inputs = data["image"].to(_COMPUTE_DEVICE_).type(float_type) #The images are returned as ByteTensor            
            _inputs = data["_image"].to(_COMPUTE_DEVICE_).type(float_type)
            labels = data["segments"].to(_COMPUTE_DEVICE_).type(float_type)
            color_segmented_img = data["segmented_image"].to(_COMPUTE_DEVICE_)#.type(torch.FloatTensor)                     

            # forward pass + backward pass + optimisation            
            optimizer.zero_grad()#zero the parameter gradients
            outputs = model(inputs) 
                                                                   
            # labels_joined = labels[:,0,:,:] + labels[:,1,:,:]
            # loss = criterion( outputs, labels )
            loss = soft_dice_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss = loss.item()
            losses.append(epoch_loss)
            print("Training iteration {:} => Loss ({:})".format(i,epoch_loss)) 
             
            #Tensorboard visualisations
            writer.add_image("INPUT_0", _inputs[0], i, dataformats="HWC")
            # Visualise the activations of the network
            masks = outputs.detach() * 255                       
            writer.add_image("MASKS_CAR_0", masks[0,0,:], i, dataformats="HW")  
            # writer.add_image("MASKS_VOID_0", masks[1], i, dataformats="HW")
            #Since there is only one class, use crossentropy and have only 1 mask
            mask = labels[0]       
            mask = mask.detach().numpy()
            mask = (mask * 255).astype(np.uint8)
            writer.add_image("TARGET_0", mask[0,:], i, dataformats="HW")
            # writer.add_image("TARGET_1", mask[1,:], i, dataformats="HW")
            #show the loss
            writer.add_scalar("TRAIN_LOSS", epoch_loss, i)

        epoch_data[epoch]["loss"] = epoch_loss
        writer.add_scalar("AVG_EPOCH_LOSS", mean(losses), epoch)
        model = model.cpu() if torch.cuda.is_available() else model
        torch.save({ 
            'epoch': epoch, 'model_dict': model.state_dict(), 
            'optimiser_dict': optimizer.state_dict(), 'loss': epoch_loss 
        }, f"./checkpoint_epoch_{epoch}.pth")
        #See https://pytorch.org/tutorials/beginner/saving_loading_models.html for more info
        print("[Epoch %d] Avg loss: %.2f" % (epoch+1, mean(losses)))

    print("Training complete .....")

    with open("epoch_data.json",'w') as file:
        json.dump(epoch_data, file)

    # TODO(AJ) Test the network

    # TODO(AJ): Utilise the trained network to perform object detection


writer.close()