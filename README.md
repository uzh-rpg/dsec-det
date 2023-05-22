# DSEC-DET

To set up the DSEC-DET dataset, you need to
1. download the original dataset, let us denote the path to this dataset with $DSEC_ROOT
2. download the extra datasets, let us denote the path to the extra data with $DSEC_EXTRA_ROOT
3. install the package
4. merge the datasets
4. remap the images into the event view
5. test alignment

## Download DSEC
Run the following commands to download the original DSEC dataset

```bash
DSEC_ROOT=$DATA/DSEC_original
for split in test train; do 
    mkdir -p $DSEC_ROOT/$split
    cd $DSEC_ROOT/$split 
    for mod in events images disparity optical_flow calibration; do 
        filename="$split"_"$mod".zip
        wget https://download.ifi.uzh.ch/rpg/DSEC/"$split"_coarse/$filename
        unzip $filename
        rm $filename
    done
done
```

## Download DSEC-extra
Run the following command to download the extra data
```bash
DSEC_ROOT=$DATA/DSEC_extra
for split in test train; do 
    mkdir -p $DSEC_ROOT/$split
    cd $DSEC_ROOT/$split 
    for mod in events images disparity optical_flow calibration object_detections; do 
        filename="$split"_"$mod".zip
        wget https://download.ifi.uzh.ch/rpg/DSEC/"$split"_coarse_extra/$filename
        unzip $filename
        rm $filename
    done
done
```

## Install the Package
To install run
```bash
git clone git@github.com:uzh-rpg/dsec-det.git
cd dsec-det/

mamba create -n dsec-det python=3.7
pip install -e .

mamba install -y -c conda-forge h5py blosc-hdf5-plugin opencv tqdm imageio pyyaml
mamba install 
```

## Merge the Datasets
Then to merge the datasets run the following
```bash
python scripts/merge_datasets.py --dsec $DATA/DSEC_original --dsec_det $DATA/DSEC_extra --output_path $DATA/DSEC_merged
```

## Remap Images
Since images are given in the left rectified image view, and labels are in the distorted event view, we need
to remap the images. Since this is a time consuming process, you can simply download them with the following commands: this
```bash
DSEC_ROOT=$DATA/DSEC_extra
for split in test train; do 
    mkdir -p $DSEC_ROOT/$split
    cd $DSEC_ROOT/$split 
    for mod in left_images_distorted; do 
        filename="$split"_"$mod".zip
        wget https://download.ifi.uzh.ch/rpg/DSEC/"$split"_coarse/$filename
        unzip $filename
        rm $filename
    done
done
```
Or regenerate them with
```bash
python scripts/remap_images_to_events.py --dsec_merged $DATA/DSEC_merged
```
This will generate a new subfolder in `$DATA/DSEC_merged/$split/$sequence/images/left` called 'distorted', where the distorted
images are stored.

## Test Alignment
You can now test alignment by running the following visualization script:
```bash
python scripts/visualize_example.py --dsec_merged $DATA/DSEC_merged --split test
```
and this will load random samples from the dataset by generating a `DSECDet` dataset class. Feel free to use it for
your deep learning applications.
