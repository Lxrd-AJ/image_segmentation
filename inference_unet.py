import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.unet_model import UNet
from ETL.cityscapes_dataset import CityscapesDataset, ToTensor, ToNumpy, Normalize, Resize
from PIL import Image
from ETL.cityscapes_labels import name2label, labels
from torchvision import transforms, utils
from torch.utils.data import DataLoader

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    print(target.size())
    nz, nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss

_NUM_CHANNELS_= 3
_IMAGE_SIZE_ = (400,400)
_COMPUTE_DEVICE_ = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./cityscapes_data"
ANNOTATION_DATA_DIR = DATA_DIR + "/gtFine"
IMG_DATA_DIR = DATA_DIR + "/leftImg8bit"
TRAIN_DIR_IMG = IMG_DATA_DIR + "/test"
TRAIN_DIR_ANN = ANNOTATION_DATA_DIR + "/test"

categories = [label.name for label in labels]
criterion = nn.BCELoss()

if __name__ == "__main__":    
    cityscapes_dataset = CityscapesDataset(
        TRAIN_DIR_IMG, TRAIN_DIR_ANN, "gtFine", 
        categories, transform=transforms.Compose([Resize(_IMAGE_SIZE_), Normalize(), ToTensor()]))
    testloader = DataLoader(cityscapes_dataset, batch_size=1, shuffle=True, num_workers=4)

    model = UNet( n_classes=len(categories), in_channels=_NUM_CHANNELS_ )
    checkpoint = torch.load("./checkpoint_epoch_2.pth")
    model.load_state_dict(checkpoint['model_dict'])
    print(f"Training loss was {checkpoint['loss']}")
    model.eval()

    softmax = nn.Softmax2d()

    # sample_input = Image.open('val.png')
    # sample_input.thumbnail(_IMAGE_SIZE_, Image.ANTIALIAS)
    # sample_input.show()
    # sample_input = np.array(sample_input, dtype=np.uint8)
    # sample_input = torch.from_numpy( sample_input.transpose((2,0,1)) )
    float_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    with torch.no_grad(): 
        for i, data in enumerate(testloader,0):
            inputs = data["image"].to(_COMPUTE_DEVICE_).type(torch.FloatTensor)
            labels = data["segments"].to(_COMPUTE_DEVICE_).type(torch.FloatTensor)
            color_seg = data["segmented_image"][0]#.type(torch.FloatTensor)        
            color_seg = ToNumpy()(color_seg)       
            color_seg = Image.fromarray(color_seg)
            # color_seg.show()

            #Show the input image
            # input_img = ToNumpy()(inputs[0])
            # Image.fromarray(input_img).show()

            # masks = data["segments"][0]
            # for i in range(masks.size()[0]):
            #     mask = masks[i].detach().numpy() # round to nearest integer to remove vals < 0.5
            #     # mask = np.array(mask, dtype=np.uint8)
            #     mask = mask * 255
            #     Image.fromarray(mask).show()   
            # img = inputs[0]            
            # img[0] = (img[0] - torch.min(img[0])) / (torch.max(img[0]) - torch.min(img[0]))
            # img[1] = (img[1] - torch.min(img[1])) / (torch.max(img[1]) - torch.min(img[1]))
            # img[2] = (img[2] - torch.min(img[2])) / (torch.max(img[2]) - torch.min(img[2]))
            # inputs[0] = img

            outputs = model(inputs)
            
            # print(inputs[0].size())
            
            # print(img.contiguous().view(img.size()[0],-1))
            # print(img.contiguous().view(img.size()[0],-1).size())
            # mean = img.contiguous().view(img.size()[0],-1).mean(dim=-1)
            # std = img.contiguous().view(img.size()[0],-1).std(dim=-1)

            # img[0] = img[0] - mean[0]
            # img[1] -= mean[1]
            # img[2] -= mean[2]
            # print(img)

            # img[0] /= std[0]
            # img[1] /= std[1]
            # img[2] /= std[2]            
            # print(img)

            
            



            # print(torch.mean(inputs[0],dim=2,keepdim=True))
            # masks = outputs[0]
            # for i in range(masks.size()[0]):
            #     mask = masks[i].detach().numpy() # round to nearest integer to remove vals < 0.5
            #     # print(mask)
            #     mask = np.array(mask , dtype=np.uint8)
            #     mask *= 255
            #     print(mask)
            #     Image.fromarray(mask).show()
            
            masks = softmax(outputs)
            masks = masks[0]
            for i in range(masks.size()[0]):
                mask = masks[i].detach().numpy()                 
                mask = (mask * 255).astype(np.uint8)
                print(mask)                
                Image.fromarray(mask).show()
            
            # loss = criterion( outputs, labels )
            # print(loss.item())
            input("Press Enter to continue...")


        #NB: majority of labels are empty, figure out why