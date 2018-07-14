#!/bin/bash python
import torch 
import numpy as np 
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torchvision
import json
import cv2
from dataset_DSTL import datasetDSTL
from torch.utils.data import DataLoader
from unet_model import UNet
from torch.autograd import Variable  
from sklearn.metrics import confusion_matrix
from data_import import convTifToPng

# TODO: Verify and Test this function https://tuatini.me/practical-image-segmentation-with-unet/
# """
# This is also known as Intersection-over-union
# - https://github.com/NVIDIA/DIGITS/tree/digits-5.0/examples/medical-imaging#dice-metric
# """
# def jacquard_index(pred, target, n_classes = 10):
#     ious = []
#     pred = pred.view(-1)
#     target = target.view(-1)

#     # Ignore IoU for background class ("0")
#     for cls in xrange(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
#         pred_inds = pred == cls
#         target_inds = target == cls
#         intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
#         union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
#         if union == 0:
#             ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
#         else:
#             ious.append(float(intersection) / float(max(union, 1)))
#     return np.array(ious)

"""
- [x] TODO: Confusion matrix
- [ ] TODO: Average confusion matrix across epochs
- [ ] TODO: Plot confusion matrix http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
- [x] TODO: Log the IoU for each class after every epoch
- [ ] TODO: Plot the average IoU after every epoch
- [ ] TODO: Make Sample prediction for report
- [ ] TODO: Sklearn classification report
"""
def compute_confusion_matrix(predictions, ground_truth):
    """
    predictions => torch.Size([4, 10, 600, 600]) 
    """
    if predictions.size() != ground_truth.size():
        print("***** Error")

    matrix = []
    predictions = predictions[-1]
    ground_truth = ground_truth[-1]
    num_pred = predictions.size()[0]
    imsize = predictions.size()[1] # Assume the image sizes are even ie. 300x300
    pred = predictions.view(-1, imsize * imsize).detach()
    target = ground_truth.view(-1, imsize * imsize).detach()

    pred = pred.cpu().numpy() if torch.cuda.is_available() else pred.numpy()
    target = target.cpu().numpy() if torch.cuda.is_available() else target.numpy()

    for i in range(0, num_pred):
        matrix.append( confusion_matrix( target[i], np.round(pred[i]) ) )        

    return np.array(matrix)

"""
Computes the average jacquard index across the confusion matrix of shape (10, 2, 2)
"""
def jacquard_index( confusion_matrix ):
    count = confusion_matrix.shape[0]
    ttn = tfp = tfn = ttp = 0
    for i in range(count):
        cm = confusion_matrix[i]
        if cm.shape[0] == 1:
            continue
        tn, fp, fn, tp = cm.ravel()
        ttn += tn
        tfp += fp
        tfn += fn 
        ttp += tp
    ttn = ttn / count 
    tfp = tfp / count 
    tfn = tfn / count 
    ttp = ttp / count 
    return (ttp / (ttp + tfp + tfn))


dir_path = os.path.dirname(os.path.realpath(__file__)) + ""
inputPath = "dstl_satellite_data/" #"dstl_satellite_data\\"
_NUM_EPOCHS_ = 50
_NUM_CHANNELS_= 3
_IMAGE_SIZE_ = 250 #Ideal image size should be 3000 for final training using all channels
_COMPUTE_DEVICE_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# torch.set_default_tensor_type(torch.FloatTensor)

if __name__ == "__main__":
    trainset = datasetDSTL(dir_path, inputPath, channel='rgb', res=(_IMAGE_SIZE_,_IMAGE_SIZE_))
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

    classes = ('Buildings','MiscMan-made','Road','Track','Trees','Crops','Waterway','Standing_Water','Vehicle_Large','Vehicle_Small')

    # Model definition
    model = UNet( n_classes=len(classes), in_channels=_NUM_CHANNELS_ )
    if torch.cuda.device_count() >= 1:
        print("Training model on ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Loss function and Optimizer definitions
    criterion = nn.BCELoss() 
    optimizer = optim.SGD( model.parameters(), lr=0.001, momentum=0.9 )

    # Network training
    epoch_data = {}
    for epoch in range(_NUM_EPOCHS_):
        epoch_loss = 0.0
        epoch_data[epoch] = {}
        for i, data in enumerate(trainloader, 0):
            # Get the inputs for the network
            inputs = data['image'].to(_COMPUTE_DEVICE_) 
            labels = data['masks'].to(_COMPUTE_DEVICE_) 

            optimizer.zero_grad() # zero the parameter gradients

            # Forward pass + Backward pass + Optimisation
            outputs = model(inputs)

            loss = criterion( outputs, labels )
            loss.backward()
            optimizer.step()

            epoch_loss = loss.item()
            # print("[%d, %5d] loss: %.3f" % (epoch+1, i+1, loss.item())) 
            # TODO: [Visualisation] Add confusion matrix and Running metrics
            # https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py
            matrix = compute_confusion_matrix( outputs,labels )
            mean_iou = jacquard_index( matrix )
        epoch_data[epoch]['loss'] = epoch_loss
        epoch_data[epoch]['MeanIOU'] = mean_iou

        print("[%d, %5d] loss: %.3f" % (epoch+1, i+1, loss.item()))
        print("[Epoch {:}] Avg Jacquard Index = {:}".format(epoch+1, round(mean_iou,3)))
    print("Training complete .....")

    with open("epoch_data.json",'w') as file:
        json.dump(epoch_data, file)

    

    #Test the network
    sample = trainset[1]
    input = sample['image']
    dim = input.size()
    input = input.view(1,dim[0],dim[1],dim[2])
    
    prediction = model(input)[0]
    prediction = prediction.cpu().detach().numpy()
    prediction = np.round(prediction)
    
    input = convTifToPng( input.cpu().numpy() )

    print(input)
    print(type(input))
    result = np.array(input).astype(np.uint8)

    for cl in range(prediction.shape[0]): #([10, 600, 600])
        mask = prediction[cl] * 255
        mask = mask.numpy().astype(np.uint8)
        
        
        cv2.addWeighted(result, 0.8, mask, 0.2, 0.0, result)
    
    # masks = torch.from_numpy(masks)
    cv2.imshow("Result", result)



# export PYTHONPATH='/usr/local/lib/python3.5/dist-packages'
