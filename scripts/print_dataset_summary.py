
from pathlib import Path
import numpy as np
import cv2
import argparse

from dsec_det.dataset import DSECDet


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Visualize an example.""")
    parser.add_argument("--dsec_merged", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    assert args.split in ['train', 'test']
    assert args.dsec_merged.exists() and args.dsec_merged.is_dir()

    dataset = DSECDet(args.dsec_merged, split=args.split, sync="back", debug=True)
    dataset.print_summary()










