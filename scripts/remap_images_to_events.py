import argparse
import cv2
import tqdm
from filecmp import cmp
from pathlib import Path

from dsec_det.taskmanager import TaskManager
from dsec_det.io import yaml_file_to_dict, h5_file_to_dict
from dsec_det.remapping import compute_remapping
from PIL import Image


def is_corrupted(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return False
    except Exception:
        return True

def read_remap_and_write(input_file, output_directory, remapping):
    output_file = output_directory / input_file.name

    if output_file.exists() and not is_corrupted(output_file):
        return

    image = cv2.imread(str(input_file))
    image = cv2.remap(image, remapping, None, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(output_directory / input_file.name), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Remaps images from image view to distorted event view""")
    parser.add_argument("--dsec_merged", type=Path, required=True)

    args = parser.parse_args()

    assert args.dsec_merged.exists() and args.dsec_merged.is_dir()

    splits = ['test', 'train']
    for split in splits:
        sequences = sorted(list((args.dsec_merged / split).glob("*/")))
        for sequence in sequences:

            images_directory = sequence / "images/left"
            rectified_images_directory = images_directory / "rectified"
            distorted_images_directory = images_directory / "distorted"

            rectification_map_file = sequence / "events/left/rectify_map.h5"
            cam_to_cam_file = sequence / "calibration/cam_to_cam.yaml"

            calibration = yaml_file_to_dict(cam_to_cam_file)
            rectification_map = h5_file_to_dict(rectification_map_file)

            remapping_map = compute_remapping(calibration, rectification_map)
            image_files = sorted(list(rectified_images_directory.glob("*.png")))

            distorted_images_directory.mkdir(parents=True, exist_ok=True)
            with TaskManager(total=len(image_files), processes=4, queue_size=4, use_pbar=True, pbar_desc=f"Processing {sequence.name}") as tm:
                for f in tqdm.tqdm(image_files):
                    tm.new_task(read_remap_and_write, f, distorted_images_directory, remapping_map)
