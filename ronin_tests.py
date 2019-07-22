import torch
import numpy as np
import torch.nn as nn
import random
from model.unet_model import UNet
from ETL.cityscapes_dataset import CityscapesDataset, ToTensor, ToNumpy, Rescale, Resize
from PIL import Image
from ETL.cityscapes_labels import name2label, labels
from torchvision import transforms, utils
from torch.utils.data import DataLoader

# Test the dataset class
if __name__ == "__main__":
    DATA_DIR = "./cityscapes_data"
    ANNOTATION_DATA_DIR = DATA_DIR + "/gtFine"
    IMG_DATA_DIR = DATA_DIR + "/leftImg8bit"
    TRAIN_DIR_IMG = IMG_DATA_DIR + "/train"
    TRAIN_DIR_ANN = ANNOTATION_DATA_DIR + "/train"
    size = (500,500)
    _IMAGE_SIZE_ = size

    categories = [label.name for label in labels]
    cityscapes_dataset = CityscapesDataset(
        TRAIN_DIR_IMG, TRAIN_DIR_ANN, "gtFine", 
        categories, transform=transforms.Compose([Resize(_IMAGE_SIZE_), Rescale()])) # transforms.Compose([ToTensor()])

    rand_idx = random.randint(0, len(cityscapes_dataset))
    print(len(cityscapes_dataset))
    sample = cityscapes_dataset[rand_idx]

    

    # Using numpy as the underlying datatype
    x = Image.fromarray(sample["image"])
    # x.thumbnail(size, Image.ANTIALIAS)
    x.show()
    y = Image.fromarray(sample["segmented_image"])
    # y.thumbnail(size, Image.ANTIALIAS)    
    y.show()
    print("{:} masks available".format(len(sample["segments"])))
    # Random masks
    # rand_mask = random.randint(0, len(sample["segments"]))
    # print(f"Showing random mask at {rand_mask}")
    # z = Image.fromarray(sample["segments"][rand_mask])
    # z.thumbnail(size, Image.ANTIALIAS)
    # z.show()
    # print("{:} mask being shown".format(labels[rand_mask].name))

    #Show all masks
    for mask in sample["segments"]:
        mask = mask * 255                
        z = Image.fromarray(mask)
        # z.thumbnail(size, Image.ANTIALIAS)
        z.show()        