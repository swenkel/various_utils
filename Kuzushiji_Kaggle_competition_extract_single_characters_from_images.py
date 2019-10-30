###############################################################################
#  Extract single characters from book scans                                  #
#  for the Kuzushiji Kaggle competition                                       #
#                                                                             #
# (c) Simon Wenkel                                                            #
# released under a 3-clause BSD license                                       #
#                                                                             #
###############################################################################


###############################################################################
#                                                                             #
#                                                                             #
# import libraries                                                            #
import time
scriptStartTime = time.time()
import sys
import os
import numpy as np
import pandas as pd
import cv2
from joblib import Parallel, delayed
#                                                                             #
#                                                                             #
###############################################################################


###############################################################################
#                                                                             #
#                                                                             #
# function and classes                                                        #
def generateBoundingBoxes(dfFile):
    """

    """
    inputDF = pd.read_csv(dfFile)
    boundingBoxes = []
    for i in range(len(inputDF.values)):
        if type(inputDF.values[i][1]) != float:
            string = inputDF.values[i][1].split(" ")
            for j in range(0,len(string),5):
                boundingBoxes.append([inputDF.values[i][0],
                                      string[j],
                                      (int(string[j+1]),
                                       int(string[j+2]),
                                       int(string[j+3]),
                                       int(string[j+4]))])
    boundingBoxes = pd.DataFrame(boundingBoxes,
                                 columns=["image","label","bbox"])
    print(boundingBoxes)
    return boundingBoxes

def extractAndDumpCharacters(boundingBoxes):
    """
    Create an numpy array containing all examples for a certain character
    as gray scale images (rescaled) and dump it for later usage.
    >> Joblib parallelization <<
    """
    def extractAndDumpCharacter(boundingBoxes,character):
        """
        Create an numpy array containing all examples for a certain character
        as gray scale images (rescaled) and dump it for later usage.
        """
        def fitToImgResolution(img):
            """
            Reduce resolution to make hierarchical training easier and faster
            Deformation of images is acceptible, so we don't need any
            image augmentation ;).
            """
            img = cv2.resize(img,(
                                  CONSTANTS["IMGResolution"],
                                  CONSTANTS["IMGResolution"]))

            return img
        imgs = []
        for example in boundingBoxes[boundingBoxes["label"] == character].values:
            img = cv2.imread(CONSTANTS["PATH"]+"train_images/"+example[0]+".jpg")
            img = cv2.cvtColor(img[example[2][1]:example[2][1]+example[2][3],
                                    example[2][0]:example[2][0]+example[2][2],:],
                                    cv2.COLOR_RGB2GRAY)
            img = fitToImgResolution(img)
            imgs.append(img)
        imgs = np.array(imgs)
        np.save(CONSTANTS["PATH"]+"single_characters_train/"+character, imgs)

        tick = time.time()
        print(character,"dumped.")

    characters = boundingBoxes["label"].unique()
    Parallel(n_jobs=4, prefer="threads")(
        delayed(extractAndDumpCharacter)(boundingBoxes,character)
        for character in characters
    )
#                                                                             #
#                                                                             #
###############################################################################



###############################################################################
#                                                                             #
#                                                                             #
# CONSTANTS                                                                   #
CONSTANTS = {}
CONSTANTS["PATH"] = "../input/kuzushiji-recognition/"
CONSTANTS["IMGResolution"] = 28
#                                                                             #
#                                                                             #
###############################################################################


###############################################################################
#                                                                             #
#                                                                             #
# main function                                                               #
def main():
    print("=" * 80)
    print("-" * 80)
    print("Generate bounding boxes for each character on each image")
    startTime = time.time()
    boundingBoxes = generateBoundingBoxes(CONSTANTS["PATH"]+"train.csv")
    print("Bounding boxes generated in {} s".format(time.time()-startTime))
    print("-" * 80)
    print("-" * 80)
    print("Extract characters")
    if not os.path.exists(CONSTANTS["PATH"]+"single_characters_train/"):
        os.makedirs(CONSTANTS["PATH"]+"single_characters_train/")
    extractAndDumpCharacters(boundingBoxes)
    print("-" * 80)
    print("Total runtime: {} min".format((time.time()-scriptStartTime)/60))
    print("=" * 80)
#                                                                             #
#                                                                             #
###############################################################################

if __name__ == "__main__":
    main()
