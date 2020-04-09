#!/bin/bash

# This script performs a complete Media Event Detection pipeline (MED) using video features:
# a) preprocessing of videos, b) feature representation,
# c) computation of MAP scores

# You can pass arguments to this bash script defining which one of the steps you want to perform.
# This helps you to avoid rewriting the bash script whenever there are
# intermediate steps that you don't want to repeat.

# execute: bash run.pipeline.sh -p true -f true -m true -y filepath

# Reading of all arguments:
while getopts p:f:m:y: option		# p:f:m:y: is the optstring here
	do
	case "${option}"
	in
	p) PREPROCESSING=${OPTARG};;       # boolean true or false
	f) FEATURE_REPRESENTATION=${OPTARG};;  # boolean
	m) MAP=${OPTARG};;                 # boolean
    y) YAML=$OPTARG;;                  # path to yaml file containing parameters for feature extraction
	esac
	done

export PATH=~/anaconda3/bin:$PATH

if [ "$PREPROCESSING" = true ] ; then

    echo "#####################################"
    echo "#         PREPROCESSING             #"
    echo "#####################################"

    # steps only needed once
    video_path=/home/ubuntu/11775-hws/hw2_code/videos  # path to the directory containing all the videos.
#    mkdir -p list downsampled_videos surf cnn kmeans  # create folders to save features
#    awk '{print $1}' ../hw1_code/list/train > list/train.video  # save only video names in one file (keeping first column)
#    awk '{print $1}' ../hw1_code/list/val > list/val.video
#    cat list/train.video list/val.video list/test.video > list/all.video    #save all video names in one file
    downsampling_frame_len=60
    downsampling_frame_rate=15

    # 1. Downsample videos into shorter clips with lower frame rates.
    # TODO: Make this more efficient through multi-threading f.ex.
    start=`date +%s`
    for line in $(cat "list/all.video"); do
        ffmpeg -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t $downsampling_frame_len -r $downsampling_frame_rate downsampled_videos/$line.ds.mp4
    done
    end=`date +%s`
    runtime=$((end-start))
    echo "Downsampling took: $runtime" #28417 sec around 8h without parallelization

    # 2. TODO: Extract SURF features over keyframes of downsampled videos (0th, 5th, 10th frame, ...)
    python surf_feat_extraction.py list/all.video config.yaml

    # 3. TODO: Extract CNN features from keyframes of downsampled videos
    python cnn_feat_extraction.py list/all.video config.yaml
	

fi

if [ "$FEATURE_REPRESENTATION" = true ] ; then

    echo "####################################################"
    echo "#  FEATURE REPRESENTATION (for both SURF and CNN)  #"
    echo "####################################################"

    python feature_representation.py
    
fi

if [ "$MAP" = true ] ; then

    echo "#######################################"
    echo "# MED with SURF Features: MAP results #"
    echo "#######################################"

    # Paths to different tools;
    map_path=/home/ubuntu/tools/mAP
    export PATH=$map_path:$PATH

    python map_surf.py
    
    echo "#######################################"
    echo "# MED with CNN Features: MAP results  #"
    echo "#######################################"

    python map.cnn.py
fi
