###############################################################################
# Pre-processing script for training images for the                           #
# Severstal Steel Defect Detection challenge on Kaggle                        #
# (https://www.kaggle.com/c/severstal-steel-defect-detection)                 #
#                                                                             #
#                                                                             #
# (c) Simon Wenkel (https://www.simonwenkel.com)                              #
# Released under the 3-clause BSD License                                     #
# see license file for further information                                    #
#                                                                             #
# last change: October 16, 2019                                               #
###############################################################################



###############################################################################
#                             External Libraries                              #
#                                                                             #
#                                                                             #
import time
runTimeStart = time.time()
import os
import sys
import numpy as np
import pandas as pd
import tqdm
import pickle
import random
import cv2
from joblib import Parallel, delayed
#                                                                             #
#                                                                             #
###############################################################################



###############################################################################
#                             Constants                                       #
#                                                                             #
#                                                                             #
CONSTANTS = {}
CONSTANTS["SEED"] = 1
CONSTANTS["PATH"] = "../input/severstal-steel-defect-detection/"
CONSTANTS["IMAGESIZE"] = (256,256)
#                                                                             #
#                                                                             #
###############################################################################



###############################################################################
#                       	 Functions and classes                            #
#                                                                             #
#                                                                             #
def seed_everything(seed):
    """
    Getting rid of all the randomness in the world :(
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def generateMask(image, df):
    """
    Generate a 5 class mask for each training image
    Mask classes (unit8):
        - 0 : background/no error
        - 1 : error type 1
        - 2 : error type 2
        - 3 : error type 3
        - 4 : error type 4
    """
    def rle2Mask(encodedMask,imageShape):
        """
        Each error mask is encoded as follows:
        1. a flatten image is assumed (vector)
        2. the first value denotes the start point of a mask
        3. the second value denotes the length of a mask
        """
        encodedMask = encodedMask.split()
        mask = np.zeros(imageShape[0]*imageShape[1]).astype(np.uint8)
        startLocs, lengths = [np.asarray(errorMask, dtype=np.int32) for errorMask in (encodedMask[0:][::2],encodedMask[1:][::2])]
        startLocs -= 1 # correct for index 0
        endLocs = startLocs + lengths
        for start, end in zip(startLocs, endLocs):
            # start with binary mask first, add other classes after collision check
            mask[start:end] = 1 # change to 60 for better visualization (exploration phase only)
        return mask.reshape(imageShape[1],imageShape[0]).T

    def checkOverlappingClasses(masks):
        """
        Check if we face any problems with multiple labels per pixel
        print out an error (but not raise one) for resolving this problem manually
        """
        checkMask = np.zeros_like(masks[0])
        for i in range(len(masks)):
            checkMask += masks[i]
        maxValue = np.max(mask.ravel())
        if maxValue <= 1:
            overlapping = False
        else:
            overlapping = True
        return overlapping

    # load image
    img = cv2.cvtColor(cv2.imread(CONSTANTS["PATH"]+"train_images/"+image),
                       cv2.COLOR_BGR2GRAY)
    imageShape = img.shape
    # load error masks for image
    errors = df[df["ImageName"] == image]["EncodedPixels"].values
    # create a mask per error
    masks = []
    mask = np.zeros_like(img, dtype=np.uint8) # background/no error class
    for errorMaskRLE in errors:
        if type(errorMaskRLE) != float:
            masks.append(rle2Mask(errorMaskRLE,imageShape))
        else:
            masks.append(mask)

    overlapping = checkOverlappingClasses(masks)
    if overlapping:
        print(image,"shows overlapping labels!")

    # concatenate mask layers to single mask for semantic segmentation
    for i in range(len(masks)):
        mask += (masks[i] * i)

    # save mask
    cv2.imwrite(CONSTANTS["PATH"]+"train_images/masked/full/"+image.split(".")[0]+".png", mask)

    # split images and masks into smaller chunks to increase model performance
    # smaller image size (e.g. 256x256) makes it easier to
    # train models for semantic segmentation
    subImages = []
    subMasks = []
    for j in range(0,img.shape[0],CONSTANTS["IMAGESIZE"][0]):
        for i in range(0,img.shape[1],CONSTANTS["IMAGESIZE"][1]):
            tmpImg = np.zeros((CONSTANTS["IMAGESIZE"]),dtype=np.uint8)
            tmpMask = np.zeros((CONSTANTS["IMAGESIZE"]),dtype=np.uint8)

            if i+CONSTANTS["IMAGESIZE"][1] > img.shape[1]:
                end_i = img.shape[1]
                end_i_status = True
            else:
                end_i = i+CONSTANTS["IMAGESIZE"][1]
                end_i_status = False

            if j+CONSTANTS["IMAGESIZE"][0] > img.shape[0]:
                end_j = img.shape[0]
                end_j_status = True
            else:
                end_j = j+CONSTANTS["IMAGESIZE"][0]
                end_j_status = False

            if end_i_status and end_j_status:
                tmpImg[0,end_j-j,0:end_i-i] = img[j:end_j,i:end_i]
                tmpMask[0,end_j-j,0:end_i-i] = mask[j:end_j,i:end_i]
            elif end_i_status:
                tmpImg[:,0:end_i-i] = img[j:end_j,i:end_i]
                tmpMask[:,0:end_i-i] = mask[j:end_j,i:end_i]
            elif end_j_status:
                tmpImg[0,end_j-j,:] = img[j:end_j,i:end_i]
                tmpMask[0,end_j-j,:] = mask[j:end_j,i:end_i]
            else:
                tmpImg = img[j:end_j,i:end_i]
                tmpMask = mask[j:end_j,i:end_i]
            subImages.append(tmpImg)
            subMasks.append(tmpMask)
    for i in range(len(subImages)):
        cv2.imwrite(CONSTANTS["PATH"]+"train_images/masked/split/img_"+image.split(".")[0]+"_"+str(i)+".png", subImages[i])
        cv2.imwrite(CONSTANTS["PATH"]+"train_images/masked/split/mask_"+image.split(".")[0]+"_"+str(i)+".png", subMasks[i])


def preprocessTrainingImages(df):
    """
    using joblib for parallel processing
    """
    if not os.path.exists(CONSTANTS["PATH"]+"train_images/"+"masked/"):
        os.makedirs(CONSTANTS["PATH"]+"train_images/"+"masked/")
    if not os.path.exists(CONSTANTS["PATH"]+"train_images/"+"masked/full/"):
        os.makedirs(CONSTANTS["PATH"]+"train_images/"+"masked/full/")
    if not os.path.exists(CONSTANTS["PATH"]+"train_images/"+"masked/split/"):
        os.makedirs(CONSTANTS["PATH"]+"train_images/"+"masked/split/")
    images = df["ImageName"].unique()
    Parallel(n_jobs=8, prefer="threads")(
        delayed(generateMask)(image,df)
        for image in images
    )
#                                                                             #
#                                                                             #
###############################################################################



###############################################################################
#                            Main function                                    #
#                                                                             #
#                                                                             #
def main():
    print("=" * 80)
    print()
    print("-" * 80)
    print(">>>Software versions<<<")
    print()
    print("Python version:",sys.version)
    print("NumPy version:",np.__version__)
    print("Pandas version:",pd.__version__)
    print("OpenCV version:",cv2.__version__)
    print("-" * 80)
    print()
    seed_everything(CONSTANTS["SEED"])
    print()
    print("-" * 80)
    print(">>>Pre-processing training images<<<")
    print()
    startTimePreprocessing = time.time()
    inputData = pd.read_csv(CONSTANTS["PATH"]+"train.csv")
    inputData["Defects"] = inputData["EncodedPixels"].notnull()
    inputData["ClassID"] = inputData['ImageId_ClassId'].str[-1:]
    inputData['ImageName'] = inputData['ImageId_ClassId'].str[:-2]
    preprocessTrainingImages(inputData)
    print()
    print("Pre-processing time:",time.time()-startTimePreprocessing,"s")
    print("-" * 80)
    print()
    runTimeEnd = time.time()
    print("Total runtime:",runTimeEnd-runTimeStart,"s")
    print("=" * 80)
#                                                                             #
#                                                                             #
###############################################################################

if __name__ == "__main__":
    main()
