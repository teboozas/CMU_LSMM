#!/usr/bin/env python3

import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb
import time

import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from pathlib import Path


def get_cnn_features_from_video(downsampled_video_filename, cnn_feat_video_filename, keyframe_interval, model):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."
    tmp = []
    for img in get_keyframes(downsampled_video_filename, keyframe_interval):
        feature = model(Variable(to_tensor(img).unsqueeze(0))).detach().numpy()
        tmp.append(feature)
    with open(cnn_feat_video_filename, 'wb') as f:
        pickle.dump(tmp, f)


def get_keyframes(downsampled_video_filename, keyframe_interval):
    "Generator function which returns the next keyframe."

    # Create video capture object
    video_cap = cv2.VideoCapture(downsampled_video_filename)
    frame = 0
    while True:
        frame += 1
        ret, img = video_cap.read()
        if ret is False:
            break
        if frame % keyframe_interval == 0:
            yield img
    video_cap.release()


if __name__ == '__main__':
#    if len(sys.argv) != 2:
#        print("Usage: {0} video_list config_file".format(sys.argv[0]))
#        print("video_list -- file containing video names")
#        print("config_file -- yaml filepath containing all parameters")
#        exit(1)

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))

    # Get parameters from config file
    keyframe_interval = my_params.get('keyframe_interval')
    cnn_features_folderpath = my_params.get('cnn_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # TODO: Create SURF object

    # Check if folder for SURF features exists
    if not os.path.exists(cnn_features_folderpath):
        os.mkdir(cnn_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    fread = open(all_video_names, "r")
    
    total = 2934
    
    model = models.mobilenet_v2(pretrained = True)
#    layer = model._modules.get('avgpool')
    to_tensor = transforms.ToTensor()
    model.eval()
    
    iteration = 0
    start = time.time()
    for line in fread.readlines():
        iteration += 1
        video_name = line.replace('\n', '')
        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
        cnn_feat_video_filename = os.path.join(cnn_features_folderpath, video_name + '.cnn')

        if not os.path.isfile(downsampled_video_filename):
            continue
        
        if os.path.isfile(cnn_feat_video_filename):
            end = time.time()
            print("Progress: {}/{} ({} min total (file already exists))".format(iteration, total, round((end-start) / 60, 2)))
            continue

        # Get SURF features for one video
        get_cnn_features_from_video(downsampled_video_filename,
                                    cnn_feat_video_filename, keyframe_interval, model)
        end = time.time()
        print("Progress: {}/{} ({} min total)".format(iteration, total, round((end-start) / 60, 2)))