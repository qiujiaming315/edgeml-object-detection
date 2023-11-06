import argparse
import os
import sys
from pathlib import Path

from lib.data import extract_output_feature

# adding Folder_2 to the system path
sys.path.insert(0, "../lib/")

"""Extract features from detection output according to the standard described in
'Adaptive Feeding: Achieving Fast and Accurate Detections by Adaptively Combining Object Detectors'."""


def main(opts):
    num_class = 20 if opts.dataset == "voc" else 80
    # Create the directories to save the extracted features (if they are not created yet).
    img_names = ['.'.join(f.split('.')[:-1]) for f in sorted(os.listdir(opts.label_dir))]
    for img_name in img_names:
        Path(os.path.join(opts.save_dir, img_name)).mkdir(parents=True, exist_ok=True)
    new_img_names = sorted([f for f in os.listdir(opts.save_dir) if not os.path.isfile(os.path.join(opts.save_dir, f))])
    assert len(img_names) == len(new_img_names) and all([i == n for i, n in zip(img_names, new_img_names)])
    extract_output_feature(opts.output_dir, opts.save_dir, num_class, opts.k)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('output_dir', help="Directory to the (weak detector's) detection output files.")
    args.add_argument('save_dir', help="Directory to save the extracted features.")
    args.add_argument('label_dir', help="Directory to the ground truth annotations.")
    args.add_argument('--k', type=int, default=25, help="Top-K bounding boxes to collect.")
    args.add_argument('--dataset', type=str, default="coco", help="The dataset to process ('coco' or 'voc').")
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
