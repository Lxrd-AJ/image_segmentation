import os, random, torch
import numpy as np 
import cv2
from . import cityscapes_etl
from . import cityscapes_labels
import random 
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image 
from pprint import pprint

class ToTensor(object):
    def __call__(self, sample):
        image, segments, segmented_image = sample["image"], sample["segments"], sample["segmented_image"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = torch.from_numpy( image.transpose((2,0,1)) )
        segmented_image = torch.from_numpy( segmented_image.transpose((2,0,1)) )
        segments_ = torch.from_numpy( np.array(segments) )
        return {"image": image, "segments": segments_, "segmented_image": segmented_image }

class Rescale(object):
    def __call__(self, sample):
        image, segments, segmented_image = sample["image"], sample["segments"], sample["segmented_image"]

        # segments should be a python list of numpy arrays 
        segments_ = []
        for segment in segments:
            segments_.append(segment/255)

        return {"image": image, "segments": segments_, "segmented_image": segmented_image }

class Resize(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def __resize(self, img):
        #TODO(AJ): If img_size is 0, just return image as it is
        img = Image.fromarray(img)
        img.thumbnail(self.img_size, Image.ANTIALIAS)
        return np.array(img)

    """
    Assumes the given sample contains numpy arrays and `img_size` is a tuple 
    """
    def __call__(self, sample):
        image, segments, segmented_image = sample["image"], sample["segments"], sample["segmented_image"]

        image = self.__resize(image)

        segments_ = []
        for segment in segments:
            segment = self.__resize(segment)
            segments_.append(segment)

        segmented_image = self.__resize(segmented_image)

        return {"image": image, "segments": segments_, "segmented_image": segmented_image }


class ToNumpy(object):
    """
    Transforms a tensor of rank 3 to an image format for display
    """
    def __call__(self, torch_image ):
        return torch_image.numpy().transpose((1,2,0))

class CityscapesDataset(Dataset):
    def __init__(self, train_dir, ann_dir, type, labels, transform):
        self.df = cityscapes_etl.build_index(train_dir, ann_dir, type )
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        #TODO: Need to merge some classes e.g bicycle and bicycle_group, doesnt affect training 
        item = {}
        sample = self.df.iloc[idx]
        item["image"] = np.array(Image.open(sample["img_url"]))
        item["segmented_image"] = np.array(Image.open(sample["color_seg_url"]))
        masks = []
        cat_segments = cityscapes_etl.parse_annotation_file( sample["ann_polygon_url"])
        for label in self.labels:
            if label in cat_segments["segment"].keys():
                masks.append(cat_segments["segment"][label])
            else:
                sample_size = cat_segments["img_size"]
                masks.append(np.zeros(sample_size, dtype=np.uint8))

        item["segments"] = masks     

        if self.transform:
            item = self.transform(item)

        return item   


# Test the dataset class
if __name__ == "__main__":
    DATA_DIR = "./../cityscapes_data"
    ANNOTATION_DATA_DIR = DATA_DIR + "/gtFine"
    IMG_DATA_DIR = DATA_DIR + "/leftImg8bit"
    TRAIN_DIR_IMG = IMG_DATA_DIR + "/train"
    TRAIN_DIR_ANN = ANNOTATION_DATA_DIR + "/train"

    labels = list(cityscapes_labels.name2label.keys())
    cityscapes_dataset = CityscapesDataset(
        TRAIN_DIR_IMG, TRAIN_DIR_ANN, "gtFine", 
        labels, transform=None) # transforms.Compose([ToTensor()])

    rand_idx = random.randint(0, len(cityscapes_dataset))
    sample = cityscapes_dataset[rand_idx]

    size = (500,500)

    # Using numpy as the underlying datatype
    x = Image.fromarray(sample["image"])
    x.thumbnail(size, Image.ANTIALIAS)
    x.show()
    y = Image.fromarray(sample["segmented_image"])
    y.thumbnail(size, Image.ANTIALIAS)
    y.show()
    print("{:} masks available".format(len(sample["segments"])))
    # Random masks
    rand_mask = random.randint(0, len(sample["segments"]))
    z = Image.fromarray(sample["segments"][rand_mask])
    z.thumbnail(size, Image.ANTIALIAS)
    z.show()
    print("{:} mask being shown".format(labels[rand_mask]))
    
    # Using tensors as the underlying datatype
    # toNumpy = ToNumpy()
    # Image.fromarray( toNumpy(sample["image"]) ).show()
    # Image.fromarray( toNumpy(sample["segmented_image"]) ).show()
    # len_segments = sample["segments"].shape[0]
    # print("{:} masks available".format(len_segments))
    # # Random masks
    # rand_mask = random.randint(0, len_segments)
    # Image.fromarray(sample["segments"][rand_mask].numpy()).show()
    # print("{:} mask being shown".format(labels[rand_mask]))

    # print("Rescaled segments")
    # sample = Rescale()(sample)
    # torch.set_printoptions(threshold=3000000, linewidth=100)
    # pprint( sample["segments"][rand_mask] )