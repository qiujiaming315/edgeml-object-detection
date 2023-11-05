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

"""Calculate the offloading reward value for each image in a dataset."""


def compute_orie(img_idx, weak_data, strong_data, labels, num_ensemble=1000):
    """
    Compute ORIE for an image given its index.
    :param img_idx: the image index.
    :param weak_data: weak detector's processed output.
    :param strong_data: the strong detector's processed output.
    :param labels: the ground truth annotations.
    :param num_ensemble: number of images in the ensemble set on which ORIE is evaluated
                         (ORIE defaults to ORI when num_ensemble=0).
    :return: the ORIE value.
    """
    # Randomly select an ensemble image set.
    num_img = len(labels)
    if num_ensemble > num_img - 1:
        num_ensemble = num_img - 1
        print("Ensemble size is too large. Set to the dataset size.")
    if num_ensemble < 0:
        num_ensemble = 0
        print("Ensemble size is negative. Set to 0.")
    ensemble_idx = np.arange(num_img - 1)
    if img_idx < num_img - 1:
        ensemble_idx[img_idx:] += 1
    ensemble_idx = np.random.permutation(ensemble_idx)[:num_ensemble]
    # Retrieve the labels and detection outputs for the ensemble dataset.
    ensemble_labels = [labels[s] for s in ensemble_idx]
    ensemble_labels.append(labels[img_idx])
    ensemble_labels = np.concatenate(ensemble_labels).astype(int)
    ensemble_detection = [weak_data[s] for s in ensemble_idx]
    # Compute the difference in mAP when the target image is offloaded to the strong detector instead.
    ensemble_detection.append(weak_data[img_idx])
    weak_map = ap_per_class(*[np.concatenate(x, axis=0) for x in zip(*ensemble_detection)], ensemble_labels)
    ensemble_detection.pop()
    ensemble_detection.append(strong_data[img_idx])
    strong_map = ap_per_class(*[np.concatenate(x, axis=0) for x in zip(*ensemble_detection)], ensemble_labels)
    orie = (np.mean(strong_map) - np.mean(weak_map)) * (num_ensemble + 1)
    print(f"ORIE for image {img_idx}: {orie:.2f}.")
    return orie


def compute_dcsb(img_idx, weak_data, strong_data):
    """
    Compute DCSB reward for an image given its index.
    Paper link: https://ieeexplore.ieee.org/abstract/document/10272511.
    :param img_idx: the image index.
    :param weak_data: weak detector's processed output.
    :param strong_data: the strong detector's processed output.
    :return: the DCSB reward value.
    """
    weak_detection, strong_detection = weak_data[img_idx], strong_data[img_idx]
    weak_num = np.sum(weak_detection[1] > 0.5)
    strong_num = np.sum(strong_detection[1] > 0.5)
    dcsb = strong_num - weak_num
    print(f"DCSB reward for image {img_idx}: {dcsb}.")
    return dcsb


def main(opts):
    weak_data, strong_data, labels = set_data(opts.weak_dir, opts.strong_dir, opts.label_dir)
    num_ensemble = opts.num_ensemble
    num_img = len(labels)
    start = time.perf_counter()
    # Compute offloading reward for every image in the dataset in parallel.
    with TPE() as pool:
        if opts.method == "orie":
            reward = np.array(list(pool.map(compute_orie, range(num_img), repeat(weak_data), repeat(strong_data),
                                            repeat(labels), repeat(num_ensemble))))
        else:
            reward = np.array(list(pool.map(compute_dcsb, range(num_img), repeat(weak_data), repeat(strong_data))),
                              dtype=int)
    # Handle the special case when no evaluated image has labels.
    reward = np.where(np.isnan(reward), 0, reward)
    finish = time.perf_counter()
    execution_time = finish - start
    print(f"Program takes {execution_time:.1f} seconds ({execution_time / 60:.1f}m/{execution_time / 3600:.2f}h).")
    Path(opts.save_dir).mkdir(parents=True, exist_ok=True)
    file_name = f"orie{num_ensemble}.npz" if opts.method == "orie" else "dcsb.npz"
    np.savez(os.path.join(opts.save_dir, file_name), reward=reward, time=execution_time)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('weak_dir', help="Directory to the weak detector output files.")
    args.add_argument('strong_dir', help="Directory to the strong detector output files.")
    args.add_argument('label_dir', help="Directory to the ground truth annotations.")
    args.add_argument('save_dir', help="Directory to save the computed computed offloading rewards.")
    args.add_argument('--method', type=str, default="orie", choices=['orie', 'dcsb'],
                      help="Method used to compute the offloading reward.")
    args.add_argument('--num-ensemble', type=int, default=1000,
                      help="Number of ensemble images when computing the offloading reward, only active when method"
                           "is 'orie', in which case setting num-ensemble to 0 yields ORI as the reward metric.")
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
