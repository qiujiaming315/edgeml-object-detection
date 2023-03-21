import numpy as np
import argparse
import os
import time
# from concurrent.futures import ProcessPoolExecutor as PPE
from concurrent.futures import ThreadPoolExecutor as TPE
from itertools import repeat
from pathlib import Path

from lib.data import set_data
from lib.metrics import ap_per_class

"""Calculate mAPI+ value for each image in a dataset as its offloading reward."""


def cal_mapi(img_idx, weak_data, strong_data, labels, num_sample=1000, offload=None):
    """
    Calculate mAPI+ for an image given its index.
    :param img_idx: the image index.
    :param weak_data: weak detector's processed output.
    :param strong_data: the strong detector's processed output.
    :param labels: the ground truth annotations.
    :param num_sample: number of images in the sample dataset on which mAPI+ is evaluated
                       (mAPI+ reduces to mAPI when num_sample=0).
    :param offload: edge offloading related configuration details.
    :return: the mAPI+ value.
    """
    # Randomly select a sample dataset.
    num_img = len(labels)
    if num_sample > num_img - 1:
        num_sample = num_img - 1
        print("Sample number is too large. Set to the dataset size.")
    if num_sample < 0:
        num_sample = 0
        print("Sample number is negative. Set to 0.")
    sample_idx = np.arange(num_img - 1)
    if img_idx < num_img - 1:
        sample_idx[img_idx:] += 1
    # TODO: may add a seed for random number generator.
    sample_idx = np.random.permutation(sample_idx)[:num_sample]
    sample_mask = np.zeros((num_sample,), dtype=bool)
    if offload is not None and num_sample != 0:
        method, thresh, ref = offload
        if method == "binary":
            # Randomly sample images from the easy and hard classes.
            all_img_idx = np.arange(num_img)
            hard_idx, easy_idx = all_img_idx[ref], all_img_idx[np.logical_not(ref)]
            hard_idx, easy_idx = hard_idx[hard_idx != img_idx], easy_idx[easy_idx != img_idx]
            hard_idx, easy_idx = np.random.permutation(hard_idx), np.random.permutation(easy_idx)
            sample_idx, sample_choice = [], np.random.choice(2, num_sample, p=[1 - thresh, thresh]).astype(bool)
            num_hard, num_easy = 0, 0
            for choice in sample_choice:
                if choice:
                    sample_idx.append(hard_idx[num_hard % len(hard_idx)])
                    num_hard += 1
                else:
                    sample_idx.append(easy_idx[num_easy % len(easy_idx)])
                    num_easy += 1
            sample_idx = np.array(sample_idx)
        else:
            # Determine a subset of images to offload based on the reference mapi.
            sample_mapi = ref[sample_idx]
            mapi_thresh = sample_mapi[np.argsort(-sample_mapi)[int(num_sample * thresh)]]
            mapi_thresh = max(0, mapi_thresh)
            sample_mask = sample_mapi > mapi_thresh
    # Retrieve the labels and detection outputs for the sample dataset.
    sample_labels = [labels[s] for s in sample_idx]
    sample_labels.append(labels[img_idx])
    sample_labels = np.concatenate(sample_labels).astype(int)
    sample_detection = [strong_data[s] if m else weak_data[s] for s, m in zip(sample_idx, sample_mask)]
    # Compute the difference in mAP when the target image is offloaded to the strong detector instead.
    sample_detection.append(weak_data[img_idx])
    weak_map = ap_per_class(*[np.concatenate(x, axis=0) for x in zip(*sample_detection)], sample_labels)
    sample_detection.pop()
    sample_detection.append(strong_data[img_idx])
    strong_map = ap_per_class(*[np.concatenate(x, axis=0) for x in zip(*sample_detection)], sample_labels)
    mapi = (np.mean(strong_map) - np.mean(weak_map)) * (num_sample + 1)
    print(f"mAPI+ for image {img_idx}: {mapi:.2f}.")
    return mapi


def main(opts):
    weak_data, strong_data, labels = set_data(opts.weak, opts.strong, opts.label)
    num_sample = opts.num_sample
    num_img = len(labels)
    # TODO: Delete code for computing offloading ratio aware mAPI+.
    offload = None
    if opts.thresh > 0 and opts.mapi != '':
        if opts.mapi == 'binary':
            # Binary classify the dataset into a hard and an easy subset.
            class_ap = ap_per_class(*[np.concatenate(x, axis=0) for x in zip(*strong_data)],
                                    np.concatenate(labels).astype(int))
            class_ap = np.mean(class_ap, axis=1)
            class_rank = np.argsort(class_ap)
            hard_class, easy_class = class_rank[:len(class_rank)//2], class_rank[len(class_rank)//2:]
            image_class = list()
            for label in labels:
                img_hard_num = np.sum([lbl in hard_class for lbl in label])
                image_class.append(img_hard_num >= len(label) / 2)
            offload = ("binary", opts.thresh, image_class)
        else:
            # Load the reference mapi from pre-computed data.
            ref_mapi = np.load(opts.mapi)['mapi']
            offload = ("mapi", opts.thresh, ref_mapi)
    start = time.perf_counter()
    # Compute mAPI for every image in the dataset in parallel.
    with TPE() as pool:
        mapi = np.array(list(pool.map(cal_mapi, range(num_img), repeat(weak_data), repeat(strong_data), repeat(labels),
                                      repeat(num_sample), repeat(offload))))
    # Handle the special case when no sampled image has labels.
    mapi = np.where(np.isnan(mapi), 0., mapi)
    finish = time.perf_counter()
    execution_time = finish - start
    print(f"Program takes {execution_time:.1f} seconds ({execution_time / 60:.1f}m/{execution_time / 3600:.2f}h).")
    Path(opts.save).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(opts.save, f'sample{num_sample}.npz'), mapi=mapi, time=execution_time)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('weak', help="Path to the weak detector output files.")
    args.add_argument('strong', help="Path to the strong detector output files.")
    args.add_argument('label', help="Path to the ground truth labels.")
    args.add_argument('save', help="Path to save the computed mAPI results.")
    args.add_argument('--num-sample', type=int, default=1000,
                      help="Number of images in the sample dataset when computing mAPI+.")
    args.add_argument('--thresh', type=float, default=0)
    args.add_argument('--mapi', type=str, default='')
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
