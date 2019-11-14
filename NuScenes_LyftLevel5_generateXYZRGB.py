################################################################################
#                                                                              #
# Dataset preprocessing for the LyftLevel5 Kaggle Challenge                    #
# (https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles)       #
#                                                                              #
#                                                                              #
# (c) Simon Wenkel                                                             #
# released under a 3-Clause BSD license (see license file)                     #
#                                                                              #
# some parts originate from the reference implementation by Guido Zuidhof      #
# (github.com/lyft/nuscenes-devkit/blob/master/notebooks/Reference%20Model.ipynb
# and from the source code of the SDK/toolkit                                  #
# These parts are marked and remain under the original license (CC BY-NC-SA 4.0)
#                                                                              #
#                                                                              #
# Expected (final) folder and file structure:                                  #
#   .                                                                          #
#   ├── input                                                                  #
#   │   └── 3d-object-detection-for-autonomous-vehicles                        #
#   │       ├── sample_submission.csv                                          #
#   │       ├── train.csv                                                      #
#   │       ├── test                                                           #
#   │       │   ├── data                                                       #
#   │       │   ├── images                                                     #
#   │       │   ├── lidar                                                      #
#   │       │   └── maps                                                       #
#   │       └── train                                                          #
#   │           ├── data                                                       #
#   │           ├── images                                                     #
#   │           ├── lidar                                                      #
#   │           └── maps                                                       #
#   └── kernel                                                                 #
#       ├── datasetPreprocessing.py                                            #
#       ├── getAnalytics.py                                                    #
#       ├── run_model.py                                                       #
#       ├── train_model.py                                                     #
#       ├── dumps                                                              #
#       |   └── submission.csv                                                 #
#       ├── dataset                                                            #
#       |   ├── train.csv                                                      #
#       |   ├── test.csv                                                       #
#       |   ├── train                                                          #
#       |   └── test                                                           #
#       └── models                                                             #
#                                                                              #
#                                                                              #
################################################################################



################################################################################
#                                                                              #
#                                                                              #
# import libraries                                                             #
import time
import random
import argparse
import datetime
import sys
import os
import copy
import pickle
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R
import cv2
import PIL
from PIL import Image
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
#                                                                              #
#                                                                              #
################################################################################



################################################################################
#                                                                              #
#                                                                              #
# functions and classes                                                        #

def parseARGS():
    """
    Parsing args and generate config file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--dumpFolder", type=str,
                        help="Folder to store dumps. (default=./dumps/)",
                        default="./dumps/")
    parser.add_argument("-dsf", "--datasetFolder", type=str,
                        help="Folder to store dumps. (default=./dataset/)",
                        default="./dataset/")
    parser.add_argument("-trf", "--trainFolder", type=str,
                        help="Folder to store dumps. (default=train/)",
                        default="train/")
    parser.add_argument("-tsf", "--testFolder", type=str,
                        help="Folder to store dumps. (default=test/)",
                        default="test/")
    parser.add_argument("-p", "--PATH", type=str, help="Path to input data \
                        (default=../input/3d-object-detection-for-autonomous-vehicles/)",
                        default="../input/3d-object-detection-for-autonomous-vehicles/")
    parser.add_argument("-j", "--jobs", type=int, default=8,
                        help="No. of threads. (default=8)")
    parser.add_argument("-m", "--mode", type=str, default="train",
                        help="Training or testing mode {train, test} (default=train)")

    config = {}
    args = parser.parse_args()
    config["dumpFolder"] = args.dumpFolder
    config["datasetFolder"] = args.datasetFolder
    config["trainFolder"] = args.trainFolder
    config["testFolder"] = args.testFolder
    config["PATH"] = args.PATH
    config["jobs"] = args.jobs
    if args.mode == "train":
        config["isTrain"] = True
    elif args.mode == "test":
        config["isTrain"] = False
    else:
        raise Exception("Invalid mode! train or test supported")
    return config


def seed_everything(seed:int):
    """
    Getting rid of all the randomness in the world :(
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def checkCreateFolders(config:dict):
    """
    Create folder to dump pre-processed stuff
    """
    if not os.path.exists(config["dumpFolder"]):
        try:
            os.makedirs(config["dumpFolder"])
        except:
            raise Exception("Could not create folder to dump various things!")

    if not os.path.exists(config["datasetFolder"]):
        try:
            os.makedirs(config["datasetFolder"])
        except:
            raise Exception("Could not create folder to store the preprocessed dataset!")

    if not os.path.exists(config["datasetFolder"]+config["trainFolder"]):
        try:
            os.makedirs(config["datasetFolder"]+config["trainFolder"])
        except:
            raise Exception("Could not create folder to store training data!")

    if not os.path.exists(config["datasetFolder"]+config["testFolder"]):
        try:
            os.makedirs(config["datasetFolder"]+config["testFolder"])
        except:
            raise Exception("Could not create folder to store test data!")



def getXYZRGB(ds:LyftDataset,
              pc,
              sample:dict,
              pointsensor):
    """ Implements the extraction of RGB values for each LiDAR point




    inputs:
        - ds: dataset class (LyftDataset)
        - pc: pointcloud (LiDAR/RADAR)
        - sample: sample (dict)

    output:
        - xyzrgb (np.ndarray): point cloud containing xyz coordinates
                             and rgb values
    This function is based on the 'map_pointcloud_to_image' function
    of the LyftDataset class (lyftdataset.py in the SDK).

    """

    #############################################################
    # This part exactly the same as in 'map_pointcloud_to_image'#
    # some parts were slightly adapted to work within this      #
    # function                                                  #
    #                                                           #


    # load point cloud first
    # no merging of all 3 LiDARs since they are not available
    

    cameras = []
    for sensor in sample["data"]:
        if "CAM" in sensor:
            cameras.append(sensor)


    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = ds.get("calibrated_sensor",
                       pointsensor["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_record["translation"]))

    # Second step: transform to the global frame.
    poserecord = ds.get("ego_pose",
                        pointsensor["ego_pose_token"])
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
    pc.translate(np.array(poserecord["translation"]))

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    cam = ds.get("sample_data", sample["data"][cameras[0]])
    poserecord = ds.get("ego_pose", cam["ego_pose_token"])
    pc.translate(-np.array(poserecord["translation"]))
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)
    #                                                           #
    #                                                           #
    #############################################################

    #############################################################
    # This is the new part                                      #
    #                                                           #
    # for each but the zoomed front camera:                     #
    #                                                           #
    # 1. points are projected into a 2D camera plane            #
    # 2. points are selected (masked) that are within the image #
    # 3. RGB values are extracted                               #
    #                                                           #
    #    !The Point Cloud shape is preserved!                   #
    #                                                           #
    RGB = np.zeros_like(pc.points[0:3,:], dtype=np.uint8)
    XYZ = copy.deepcopy(pc.points)
    for camera in cameras:
        pcc = copy.deepcopy(pc)
        cam = ds.get("sample_data", sample["data"][camera])
        img = Image.open(str(ds.data_path / cam["filename"]))
        cs_record = ds.get("calibrated_sensor",
                           cam["calibrated_sensor_token"])
        pcc.translate(-np.array(cs_record["translation"]))
        pcc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

        points = view_points(pcc.points[0:3, :],
                             np.array(cs_record["camera_intrinsic"]),
                             normalize=True)
        # generate mask
        mask = np.ones(pcc.points.shape[1], dtype=np.uint8)
        # filter points that are behind the camera
        depths = pcc.points[2,:]
        mask = np.logical_and(mask, depths >0)
        # filter points outside the image
        mask = np.logical_and(mask, points[0,:]<img.size[0]) # x coordinate
        mask = np.logical_and(mask, points[0,:]>0)
        mask = np.logical_and(mask, points[1,:]<img.size[1]) # y coordinate
        mask = np.logical_and(mask, points[1,:]>0)

        # get RGB values for each point
        img = np.array(img)
        coords = np.floor(points[0:2,mask]).astype(dtype=np.uint8)
        rgblist = []
        for i in range(coords.shape[1]):
            rgblist.append(img[coords[1][i],coords[0][i],:])

        rgblist = np.array(rgblist)
        RGB[:,mask] = rgblist.T
    #                                                           #
    #                                                           #
    #############################################################

    return np.vstack((XYZ[0:3,:],RGB))



def generateXYZRGB(ds:LyftDataset,
                   sample_token:str,
                   config:dict,
                   folder:str):

    sample = ds.get("sample", sample_token)
    
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = ds.get("sample_data", sample_lidar_token)
    lidar_filepath = ds.get_sample_data_path(sample_lidar_token)

    ego_pose = ds.get("ego_pose", lidar_data["ego_pose_token"])
    calibrated_sensor = ds.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])


    global_from_car = transform_matrix(ego_pose['translation'],
                                       Quaternion(ego_pose['rotation']), inverse=False)

    car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                        inverse=False)

    try:
        lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
        pc = copy.deepcopy(lidar_pointcloud)
        lidar_pointcloud.transform(car_from_sensor)

    except Exception as e:
        print ("Failed to load Lidar Pointcloud for {}: {}:".format(sample_token, e))
        pass


    XYZRGB = getXYZRGB(ds,pc,sample,lidar_data)
    np.savez(folder+sample_lidar_token+".xyzrgb")
    




def generateDataset(ds:LyftDataset,
                    config:dict):
    if config["isTrain"]:
        folder = config["datasetFolder"]+config["trainFolder"]
        dsDF = pd.read_csv(config["PATH"]+"train.csv")
    else:
        folder = config["datasetFolder"]+config["testFolder"]
        dsDF = pd.read_csv(config["PATH"]+"sample_submission.csv")

    sample_tokens = dsDF["Id"].values
    Parallel(n_jobs=config["jobs"], prefer="threads")(
        delayed(generateXYZRGB)(ds, sample_token, config, folder)
            for sample_token in tqdm(sample_tokens)
    )



def main(config:dict):
    scriptStart = time.time()
    print("=" * 80)
    seed_everything(1)
    if config["isTrain"]:
        print("Preprocessing train set")
        ll5ds = LyftDataset(data_path=config["PATH"]+"train/",
                            json_path=config["PATH"]+"train/data/",
                            verbose=True)
    else:
        print("Preprocessing test set")
        ll5ds = LyftDataset(data_path=config["PATH"]+"test/",
                            json_path=config["PATH"]+"test/data/",
                            verbose=True)
    checkCreateFolders(config)
    generateDataset(ll5ds, config)
    print("Dataset generated in {:.2f} min.".format((time.time()-scriptStart)/60))
    print("=" * 80)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    config =  parseARGS()
    main(config)
