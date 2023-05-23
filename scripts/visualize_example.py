from pathlib import Path
import numpy as np
import cv2
import argparse

from dsec_det.dataset import DSECDet


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Visualize an example.""")
    parser.add_argument("--dsec_merged", type=Path, default="/data/storage/daniel/DSEC_with_detections_merged")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    dataset = DSECDet(args.dsec_merged, split=args.split, sync="back", debug=True)

    while True:
        index = np.random.randint(0, len(dataset))
        output = dataset[index]
        cv2.imshow("Visualization", output['debug'])
        cv2.waitKey(0)








