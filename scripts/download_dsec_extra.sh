#!/bin/bash
DSEC_ROOT=$1
cd $DSEC_ROOT || exit
for split in test train; do
    for mod in events images calibration object_detections; do
        filename="$split"_"$mod".zip
        wget https://download.ifi.uzh.ch/rpg/DSEC/"$split"_object_detection_coarse/$filename
        unzip $filename
        rm $filename
    done
done
