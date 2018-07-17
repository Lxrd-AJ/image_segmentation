import os, random, torch
import cv2
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import tifffile as tiff
from data_import import generate_mask_for_image_and_class, _get_polygon_list, query_yes_no


class datasetDSTL(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dir_path, inputPath, channel='rgb',res=(0,0),includeEmpties = False,ForceRedo=False, crange=(0.1, 0.4)):
        """
        Args:
            dir_path (string):  Working Directory.
            inputPath (string): Input directory with all the images.
            channel (string):   Channels used for the input, eiter 'gray', 'rgb', 'rgb-g' (for rgb and gray) or 'all' (then resolution needs to be chosen).
            res (tuple/list):   Resolution for the input images and masks. If 'all' is selected must be <= (132, 133) pixel.
                                If resolution == (0, 0) then it will remains unchanges for the data
            crange (tuple):     Defines the range of how strong the random crop for images can be. The direction is also chosen randomly
        """        
        
        inpDir = os.path.join(os.sep, dir_path, inputPath) #str(dir_path)+str(inputPath) + "/"
        inputs, imgIDs = self.getIDsAndFiles(inpDir)
        
        self.res = res
        self.inputPath = inputPath
        self.dir_path = dir_path
        self.imgIDs = imgIDs
        self.channel = channel
        self.crange = crange
        df = pd.read_csv(inpDir + 'train_wkt_v4.csv')
        gs = pd.read_csv(inpDir + 'grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
        newInputs = []
        masks = []
        empties = {}
        listFilename = "inputs-" + str(channel) + '-' + str(self.res[0]) + 'x' + str(self.res[1])+".txt"
        maskFilename = "masks-" + str(self.res[0]) + 'x' + str(self.res[1])+".txt"
        isEmptyFilename = "empty-masks.txt"
        listFile = Path(os.path.join( os.sep,dir_path,listFilename))
        maskFile = Path(os.path.join( os.sep,dir_path,maskFilename))
        isEmptyFile = Path(os.path.join( os.sep,dir_path,isEmptyFilename))
        doPreprocessing = ForceRedo
        if not doPreprocessing:
            if listFile.is_file() and maskFile.is_file() and isEmptyFile.is_file():
                empties = eval(open(isEmptyFile, 'r').read())
                with open(listFile, 'r') as f:
                    newInputs = [line.rstrip('\n') for line in f]
                with open(maskFile, 'r') as f:
                    i = 0
                    mask = []
                    for line in f:
                         if i < 10:
                             mask.append(line.rstrip('\n'))
                             i = i + 1
                         else:
                             i = 1
                             masks.append(mask)
                             mask = []
                             mask.append(line.rstrip('\n'))
                    masks.append(mask)

                print ("Input files have been found! ("+str(len(newInputs))+" inputs)")
                if len(newInputs) != len(masks):                
                    print("Error, masks and inputs list are of different size! Inputs: " +str(len(newInputs))+ "Masks: "+str(len(masks)))
                    doPreprocessing = query_yes_no("Do you want to redo the preprocessing?")
                if not doPreprocessing and not len(imgIDs) == len(newInputs):
                    print ("It seems the number of images in the Inpute directory has changed!")
                    doPreprocessing = query_yes_no("Do you want to redo the preprocessing?")
            else:
                doPreprocessing = True
        if doPreprocessing:
            if not ((channel=='rgb' or channel=='gray') and res == (0,0)):
                if channel=='rgb':
                    for idx, imageId in enumerate(imgIDs): 
                        rgbImage = self.getImageType('rgb',imageId, inputs)
                        rgbImage = cv2.resize(rgbImage, res)
                        newInputs.append(self.saveNewImage('rgb-rscl',rgbImage, imageId))
                        print('Processing Images: '+str(round((idx/len(imgIDs))*100, 2))+'%')  
                else:
                    if channel=='gray':
                        for idx, imageId in enumerate(imgIDs): 
                            grayImg = self.getImageType('gray',imageId, inputs)
                            grayImg = cv2.resize(grayImg, res)
                            newInputs.append(self.saveNewImage('gray',grayImg, imageId))
                            print('Processing Images: '+str(round((idx/len(imgIDs))*100, 2))+'%')  
                    else:
                        if channel=='rgb-g':
                            for idx, imageId in enumerate(imgIDs): 
                                rgbImage = self.getImageType('rgb',imageId, inputs)
                                grayImg = self.getImageType('gray',imageId, inputs)
                                if res!=(0,0):
                                    rgbImage = cv2.resize(rgbImage, res)
                                    grayImg = cv2.resize(grayImg, res)
                                rgbgImage = cv2.merge([rgbImage,grayImg])
                                newInputs.append(self.saveNewImage('rgb-g',rgbgImage, imageId))
                                print('Processing Images: '+str(round((idx/len(imgIDs))*100, 2))+'%')  
                        else:
                            if channel=='all':
                                for idx, imageId in enumerate(imgIDs): 
                                    rgbImage = self.getImageType('rgb',imageId, inputs)
                                    grayImg = self.getImageType('gray',imageId, inputs)
                                    A8c, M8c = self.getImageType('16c',imageId, inputs)
                                    rgbImage = cv2.resize(rgbImage, res)
                                    grayImg = cv2.resize(grayImg, res)
                                    A8c = cv2.resize(A8c, res)
                                    M8c = cv2.resize(M8c, res)
                                    c20Image = cv2.merge([rgbImage,grayImg, A8c, M8c])
                                    newInputs.append(self.saveNewImage('20c', c20Image, imageId))
                                    print('Processing Images: '+str(round((idx/len(imgIDs))*100, 2))+'%')                    
            else:
                if(channel=='rgb'):
                    newInputs = [x for x in inputs if (not x.endswith('_P.tif') and not x.endswith('_M.tif') and not x.endswith('_A.tif'))]
                else: 
                    if(channel=='gray'):
                        newInputs = [x for x in inputs if x.endswith('_P.tif')]

            if res == (0,0):
                (width, height, depth) = cv2.imread(newInputs[0]).shape
                res = (width,height)
            for idx, imageId in enumerate(imgIDs):
                masksNames = []
                for classType in list(range(1,11)):
                    #d = _get_polygon_list(df,imageId,classType)
                    #polygons.append(d)
                    mask, isEmpty = generate_mask_for_image_and_class(res,imageId,classType,gs,df)
                    empties[imageId] = isEmpty
                    # filename = str(dir_path)+'\\masks\\'+str(imageId)+'-'+str(classType)+'-'+str(self.res[0])+'x'+str(self.res[1])+'.png'
                    filename = os.path.join(os.sep, dir_path, 'masks', str(imageId) + '-' + str(classType) + '-' + str(self.res[0]) + 'x' + str(self.res[1])) + ".png"
                    my_file = Path(filename)
                    if not my_file.is_file():
                        cv2.imwrite(filename,mask) #*255
                    masksNames.append(filename)
                masks.append( masksNames )
                print('Processing Masks: '+str((idx/len(imgIDs))*100)+'%')

            with open("masks-" + str(self.res[0]) + 'x' + str(self.res[1])+".txt", 'w') as f:
                for mask in masks:
                    for s in mask:
                        f.write(s + '\n')

            with open("inputs-" + str(channel) + '-' + str(self.res[0]) + 'x' + str(self.res[1])+".txt", 'w') as f:
                for s in newInputs:
                    f.write(s + '\n')

            target = open("empty-masks.txt", 'w')
            target.write(str(empties))
            print (empties)

        if not includeEmpties:
            c = list(zip(newInputs, masks, imgIDs))
            d = list(c)
            for item in c:
                if (empties[item[2]]==True):
                    d.remove(item)
            newInputs, masks, imgIDs = zip(*d)

        #random.shuffle(c)

        self.imgIDs = imgIDs
        self.empties = empties
        self.masks = masks
        self.res = res
        self.inputs = newInputs

    def saveNewImage(self, path, img, imageId):
            # filename = str(self.dir_path)+str(path)+str(imageId)+'-'+str(self.res[0])+'x'+str(self.res[1])+'.png'
            filename = os.path.join( os.sep, self.dir_path, path, str(imageId)+'-'+str(self.res[0])+'x'+str(self.res[1]))
            if len(img.shape) == 3:
                img = img.transpose((2,0,1))
            my_file = Path(filename)
            if not my_file.is_file():
                tiff.imsave(filename , img)
            return filename

    def getIDsAndFiles(self, inpDir):
        inputs = []
        imgIDs = []

        for p, subdirs, f in os.walk(inpDir):     
            # print(subdirs)       
            for dir in subdirs:
                # print(dir)
                images = os.listdir(os.path.join(os.sep, inpDir, dir))                
                for idx, filename in enumerate(images):
                    n = len(filename)
                    if filename.endswith(".tif") == False:
                       images.pop(idx)

                if (str(dir) == 'three_band'):
                    imgIDs = imgIDs + images
                    imgIDs = [os.path.splitext(x)[0] for x in images]

                # images = [(str(inpDir)+str(dir)+'\\'+str(x)) for x in images]                
                images = [os.path.join(os.sep, inpDir, dir, x) for x in images]                
                inputs = inputs + images
        return inputs, imgIDs

    def stretchMinMax(self, img):
       if(len(img.shape)==2):
            gray = img[:,:]
            rmax = np.max(gray)
            rmin = np.min(gray)
            gray =  (255 * ((gray - rmin) / (rmax - rmin))).astype(np.uint8)
            return gray
       if (len(img[:,0,0])==3):
            r = img[0,:,:]
            rmax = np.max(r)
            rmin = np.min(r)
            ri =  (255 * ((r - rmin) / (rmax - rmin))).astype(np.uint8)
            g = img[1,:,:]
            gmax = np.max(g)
            gmin = np.min(g)
            b = img[2,:,:]
            bmax = np.max(b)
            bmin = np.min(b)
            gi =  (255 * ((g - gmin) / (gmax - gmin))).astype(np.uint8)
            bi =  (255 * ((b - bmin) / (bmax - bmin))).astype(np.uint8)
            return cv2.merge([bi,gi,ri])

    def getImageType(self, type,ImgId, inputs):
        if (type == 'gray'):
            imgFile = [x for x in inputs if x.endswith(str(ImgId)+'_P.tif')]
        else:
            if (type == 'rgb'):
                imgFile = [x for x in inputs if x.endswith(str(ImgId)+'.tif')]
            else:
                if (type == '16c'):
                    imgFile = [x for x in inputs if x.endswith(str(ImgId)+'_A.tif')]
                    imgFile2 = [x for x in inputs if x.endswith(str(ImgId)+'_M.tif')]
                    img = tiff.imread(imgFile[0])
                    img2 = tiff.imread(imgFile2[0])
                    img = img.transpose((1,2,0))
                    img2 = img2.transpose((1,2,0))
                    return img, img2

        img = tiff.imread(imgFile[0])
        if len(img.shape) == 3:
            img = img.transpose((1,2,0))
        #imgPng = self.stretchMinMax(img)
        return img

    def __len__(self):
        return len(self.inputs)

    def randomCrop(self, image, dir, strength):

        h, w = image.shape[:2]
        x = np.cos((dir*np.pi)/180)
        y = np.sin((dir*np.pi)/180)
        x_new = (x/(np.abs(x)+np.abs(y)))*strength
        y_new = (y/(np.abs(x)+np.abs(y)))*strength

        ox = int(x_new*w)
        oy = int(y_new*h)
        non = lambda s: s if s < 0 else None
        mom = lambda s: max(0,s)
        shift_img = np.zeros_like(image)
        shift_img[mom(oy):non(oy), mom(ox):non(ox)] = image[mom(-oy):non(-oy), mom(-ox):non(-ox)]
        #cv2.imshow(" ",shift_img)
        #cv2.waitKey(0)
        return shift_img

    def toTensor(self, image, dtype='float32'):
        # is this necessary? The image can contain also other channels ...

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        # TODO: Refactor this code here
        if len(image.shape) == 3 and image.shape[2] < 21:
            image = image.transpose((2,0,1))

        # Added Cuda support http://pytorch.org/tutorials/beginner/former_torchies/tensor_tutorial.html#cuda-tensors
        # if torch.cuda.is_available():
        #     return torch.from_numpy(image, device=torch.device('cuda'))  
        image = image.astype(dtype)
        return torch.from_numpy(image)

    def __getitem__(self, idx):

        r =  random.random()
        probCrop = 0.1
        imageId = self.imgIDs[idx]
        #image = cv2.imread(self.inputs[idx],cv2.IMREAD_UNCHANGED)
        image = tiff.imread(self.inputs[idx])
        if (r<=probCrop):
            strength = random.uniform(self.crange[0],self.crange[1])
            dir = np.random.randint(0,360) 
            image = self.randomCrop(image,dir,strength)
        image = self.toTensor(image)
        masks = self.masks[idx]
        masksImgs = []

        for maskFile in masks:
            # masksImgs.append(self.toTensor(self.randomCrop(cv2.imread(maskFile),dir,strength)))
            mask = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)           
            if (r<=probCrop):
                mask = self.randomCrop(mask,dir,strength)
            #print("MASK")
            #print(mask)
            #cv2.imshow("mask", mask)
            #print(mask.shape)
            #print()
            masksImgs.append(self.toTensor(mask) )
        masksImgs_ = torch.cat(masksImgs).view(len(masksImgs), self.res[0], self.res[1])
        item = {'image': image, 'masks': masksImgs_}

        return item