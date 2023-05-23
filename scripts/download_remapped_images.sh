#!/bin/bash

DSEC_ROOT=$1
mkdir -p $DSEC_ROOT
cd $DSEC_ROOT || exit
for split in test train; do
    mkdir -p $DSEC_ROOT/$split
    filename="$split"_left_images_distorted.zip
    wget https://download.ifi.uzh.ch/rpg/DSEC/"$split"_object_detection_coarse/$filename
    unzip $filename
    rm $filename
done