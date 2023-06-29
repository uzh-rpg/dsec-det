#!/bin/bash
DSEC_ROOT=$1
for split in test train; do
    mkdir -p $DSEC_ROOT/$split
    cd $DSEC_ROOT/$split || exit
    for mod in events images calibration object_detections; do
        filename="$split"_"$mod".zip
        wget https://download.ifi.uzh.ch/rpg/DSEC/"$split"_object_detection_coarse/$filename
        unzip $filename
        rm $filename
    done
done
