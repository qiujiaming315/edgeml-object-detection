import argparse

from lib.data import extract_output_feature


"""Extract features from detection output according to the standard described in
'Adaptive Feeding: Achieving Fast and Accurate Detections by Adaptively Combining Object Detectors'."""


def main(opts):
    num_class = 20 if opts.dataset == "voc" else 80
    extract_output_feature(opts.output, opts.feature, num_class, opts.k)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('output', help="Path to the (weak detector) detection output files.")
    args.add_argument('feature', help="Directory where the (weak detector) feature maps are stored.")
    args.add_argument('--k', type=int, default=25, help="Top-K bounding boxes to collect.")
    args.add_argument('--dataset', type=str, default="coco", help="The dataset to process ('coco' or 'voc').")
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
