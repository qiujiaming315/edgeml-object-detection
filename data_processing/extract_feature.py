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
    # Create the directories to save the feature maps (if they are not created yet).
    img_names = ['.'.join(f.split('.')[:-1]) for f in sorted(os.listdir(opts.label))]
    for img_name in img_names:
        Path(os.path.join(opts.feature, img_name)).mkdir(parents=True, exist_ok=True)
    new_img_names = sorted([f for f in os.listdir(opts.feature) if not os.path.isfile(os.path.join(opts.feature, f))])
    assert len(img_names) == len(new_img_names) and all([i == n for i, n in zip(img_names, new_img_names)])
    extract_output_feature(opts.output, opts.feature, num_class, opts.k)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('output', help="Path to the (weak detector) detection output files.")
    args.add_argument('feature', help="Directory where the (weak detector) feature maps are stored.")
    args.add_argument('label', help="Path to the ground truth labels.")
    args.add_argument('--k', type=int, default=25, help="Top-K bounding boxes to collect.")
    args.add_argument('--dataset', type=str, default="coco", help="The dataset to process ('coco' or 'voc').")
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
