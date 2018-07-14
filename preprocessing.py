import os, torch, warnings, cv2
import numpy as np
from torch.utils.data import DataLoader
from dataset_DSTL import datasetDSTL

warnings.filterwarnings("ignore")
dir_path = os.path.dirname(os.path.realpath(__file__))
inputPath = '\\input\\'
# gray, rgb and rgb-g work. (0,0) will use original resolution, last (0.1,0.4) gives range for random crop; check datasetDSTL.py for more Details
dataset = datasetDSTL(dir_path, inputPath, 'rgb-g', (2500,2500), (0.1, 0.4))

# Testing etc., just leave it collapsed
if True:
    idx = 3
    sample = dataset[idx]
    image = sample['image']
    print(idx, image.shape, sample['masks'][5].shape)
    True

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)