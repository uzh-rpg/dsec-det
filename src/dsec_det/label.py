import numpy as np
from pathlib import Path
import seaborn as sns

"""
The file `tracks.npy` has a compound numpy datatype and is of size (n,), where n is the total number of detections

Each element has the following fields: 

t:                (uint64)  timestamp of the detection in microseconds.
x:                (float64) x-coordinate of the top-left corner of the bounding box
y:                (float64) y-coordinate of the top-left corner of the bounding box
h:                (float64) height of the bounding box
w:                (float64) width of the bounding box
class_id:         (uint8)   Class of the object in the bounding box. 
                            The classes are ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train')
class_confidence: (float64) Confidence of the detection. Can usually be ignored.
track_id:         (uint64)  ID of the track. Bounding boxes with the same ID belong to one track. 
"""

CLASSES = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train')
COLORS = np.array(sns.color_palette("hls", len(CLASSES)))
DTYPE = np.dtype([('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4'), ('img_name', '<U10')])


