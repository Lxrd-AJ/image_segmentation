import pandas as pd 
import numpy as np 
import os 
import json 
import cv2
import matplotlib.pyplot as plt 
from PIL import Image
from pprint import pprint
from collections import defaultdict

DATA_DIR = "./../cityscapes_data"
ANNOTATION_DATA_DIR = DATA_DIR + "/gtFine"
IMG_DATA_DIR = DATA_DIR + "/leftImg8bit"

def parse_annotation_file( filename ):
    mask_dict = defaultdict(dict)
    with open(filename) as file:
        data = json.load(file)
        img_size = (data["imgHeight"], data["imgWidth"])
        mask_dict["img_size"] = img_size
        #create a dictomary of labels to objects
        label2obj = defaultdict(list)
        for obj in data["objects"]:
            key = obj["label"]
            label2obj[key].append(obj)
        # create a mask for each label
        for cat, ann in label2obj.items():
            mask = np.zeros(img_size, dtype=np.uint8)
            for segment in ann:
                polygons = np.array([point for point in segment["polygon"]], dtype=np.int32)
                mask = cv2.fillConvexPoly(mask, polygons, (255,255,255))
            # im = Image.fromarray(mask)
            # im.show()
            mask_dict["segment"][cat] = mask
    return mask_dict
            

def parse_id_filename( filename ):
    f = filename.split("_")
    return "{:}_{:}_{:}".format(f[0],f[1],f[2])

def build_index( train_dir, ann_dir, type ):
    data = {}
    pd_data = []
    # Populate the index with urls to the train image
    for p, subdirs, f in os.walk(train_dir):
        for dir in subdirs:
            images = os.listdir(os.path.join(train_dir,dir))
            for idx, filename in enumerate(images):
                img_url = os.path.join(train_dir,dir,filename)
                img_id = parse_id_filename(filename)
                data[img_id] = img_url
    # Populate the index with the url to the annotation json (annotation masks are created as needed)
    for p, subdirs, f in os.walk(ann_dir):
        for dir in subdirs:
            images = os.listdir(os.path.join(ann_dir,dir))
            for img_id, img_url in data.items():
                ann_polygon_url = "{:}/{:}/{:}_{:}_polygons.json".format(p,dir,img_id,type)
                color_seg_url = "{:}/{:}/{:}_{:}_color.png".format(p,dir,img_id,type)
                if os.path.isfile(ann_polygon_url) and os.path.isfile(color_seg_url):
                    pd_data.append([img_id,img_url,ann_polygon_url,color_seg_url])
                    
    return pd.DataFrame(data=pd_data,columns=["img_id","img_url","ann_polygon_url","color_seg_url"])

def show_mask_portion(image, mask):
    # Assumes mask is single-channel only
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return cv2.bitwise_and(image, mask)    

def blend_img_mask( image, mask ):
    # mask[mask==255] = 200
    mask = cv2.applyColorMap(mask, np.array([200,15,100]))#cv2.COLORMAP_JET
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(image, 0.8, mask, 0.3, 0.0)


if __name__ == "__main__":
    TRAIN_DIR_IMG = IMG_DATA_DIR + "/train"
    TRAIN_DIR_ANN = ANNOTATION_DATA_DIR + "/train"
    
    df = build_index( TRAIN_DIR_IMG,TRAIN_DIR_ANN,"gtFine" )
    
    sample = df.iloc[200]
    og_im = Image.open( sample["img_url"] )
    # og_im.show()
    res = np.array(og_im)
    masks_dict = parse_annotation_file( sample["ann_polygon_url"] )
    for cat, segment in masks_dict["segment"].items():
        # Show the current segment in the original image
        # x = show_mask_portion(res, segment)
        # Image.fromarray(x).show()

        x = blend_img_mask(res,segment)
        Image.fromarray(x).show()
        exit(0)

    
        