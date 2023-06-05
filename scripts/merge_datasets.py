import argparse
import shutil
from typing import List, Tuple

from dsec_det.taskmanager import TaskManager
from dsec_det.io import compare_dirs

from pathlib import Path

def compile_paths(input_path: Path, output_path: Path)->Tuple[List[Path],List[Path]]:
    output_folders = []
    input_folders = sorted(list(input_path.glob("*/*")))
    for folder in input_folders:
        output_folders.append(output_path / folder.relative_to(input_path))
    return input_folders, output_folders

def process(input_folder: Path, output_folder: Path)->None:
    if input_folder.is_dir():
        if output_folder.exists():
            if compare_dirs(input_folder, output_folder):
                return
            else:
                shutil.rmtree(output_folder)
        else:
            output_folder.parent.mkdir(parents=True, exist_ok=True)

        shutil.copytree(input_folder, output_folder)
    else:
        shutil.copyfile(input_folder, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Merge existing DSEC dataset into the DSEC-DET dataset""")
    parser.add_argument("--dsec", type=Path, required=True)
    parser.add_argument("--dsec_det", type=Path, required=True)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--inplace", action="store_true")

    args = parser.parse_args()

    if args.inplace:
        args.output_path = args.dsec

    assert args.dsec.exists() and args.dsec.is_dir()
    assert args.dsec_det.exists() and args.dsec_det.is_dir()
    assert args.output_path is not None and args.output_path.parent.exists()

    input_folders = []
    output_folders = []

    splits = ['train', "test"]

    for split in splits:
        input_folders_dsec, output_folders_dsec = compile_paths(args.dsec / split, args.output_path / split)
        input_folders_dsec_det, output_folders_dsec_det = compile_paths(args.dsec_det / split, args.output_path / split)

        if not args.inplace:
            input_folders.extend(input_folders_dsec)
            output_folders.extend(output_folders_dsec)

        input_folders.extend(input_folders_dsec_det)
        output_folders.extend(output_folders_dsec_det)

    with TaskManager(total=len(input_folders), processes=4, queue_size=4, use_pbar=True) as tm:
        for input_folder, output_folder in zip(input_folders, output_folders):
            tm.new_task(process, input_folder, output_folder)
