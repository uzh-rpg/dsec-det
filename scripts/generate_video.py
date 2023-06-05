from pathlib import Path
import numpy as np
import cv2
import argparse

from dsec_det.dataset import DSECDet

def generate_image_panel(data, panel_shape=[5, 3], padding=30):
    seqs = sorted(list(data.keys()))
    images = [data[s] for s in seqs]

    panel_size = [s * p + (p-1) * padding for s, p in zip(images[0].shape, panel_shape)] # height x width

    panel = np.full((panel_size[0], panel_size[1], 3), fill_value=255,  dtype="uint8")
    for i, image in enumerate(images):
        height, width = image.shape[:2]
        row, col = divmod(i, 3)
        panel[(height+padding) * row:(height+padding) * row + height, (width+padding) * col: (width+padding) * col + width] = image

    #panel = cv2.resize(panel, None, fx=0.5, fy=0.5)

    return panel


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Visualize an example.""")
    parser.add_argument("--dsec_merged", type=Path, required=True)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    assert args.split in ['train', 'test']
    assert args.dsec_merged.exists() and args.dsec_merged.is_dir()

    dataset = DSECDet(args.dsec_merged, split=args.split, sync="back", debug=True, max_num_events=100000)

    for i, data in enumerate(dataset.zipped_dataset(seqs=[5,4,7])):
        panel = generate_image_panel(data, panel_shape=[1,3])

        cv2.imshow("Visualization", panel)
        cv2.waitKey(3)

        if args.output_path is not None:
            args.output_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite( str(args.output_path / ("%06d.png" % i)), panel)




