import numpy as np
import argparse
import os
from pathlib import Path

from lib.data import set_data
from lib.metrics import ap_per_class

"""Test the performance of reward estimation(s) by computing the realized mAP in relation to the offloading ratios."""
# The offloading ratios to evaluate.
offloading_ratios = np.arange(0, 1.01, 0.1)


def test_map(weak_data, strong_data, labels, reward_estimates, dataset_split):
    """
    Compute realized mAP given reward estimates.
    :param weak_data: weak detector's processed output.
    :param strong_data: the strong detector's processed output.
    :param labels: the ground truth annotations.
    :param reward_estimates: a list containing path(s) to the file(s) with estimated reward.
    :param dataset_split: dataset split for cross validation.
    :return: an array of the computed mAP values, one for each reward estimate.
    """
    map_results = []
    # Load reward estimates of each trained model.
    for estimate_path in reward_estimates:
        map_result = np.zeros((len(offloading_ratios),))
        offload_mask = np.zeros((len(offloading_ratios), len(weak_data)), dtype=bool)
        # Check the mAP of each fold in cross validation.
        for cv_idx, val_mask in enumerate(dataset_split):
            # Load the reward estimate.
            reward_data = np.load(os.path.join(estimate_path, f"estimate{cv_idx + 1}.npz"))
            train_reward, val_reward = reward_data['train_est'], reward_data['val_est']
            for ratio_idx, offload_ratio in enumerate(offloading_ratios):
                reward_thresh = train_reward[np.argsort(-train_reward)[int((len(train_reward) - 1) * offload_ratio)]]
                # reward_thresh = max(0, reward_thresh)
                reward_mask = val_reward > reward_thresh
                offload_mask[ratio_idx, val_mask] = reward_mask
        for ratio_idx, mask in enumerate(offload_mask):
            detection = [strong_data[s] if m else weak_data[s] for s, m in enumerate(mask)]
            # Compute the mAP after offloading using the estimated reward with a fixed threshold policy.
            map_result[ratio_idx] = np.mean(ap_per_class(*[np.concatenate(x, axis=0) for x in zip(*detection)], labels))
        map_results.append(map_result)
    return np.array(map_results)


def main(opts):
    weak_data, strong_data, labels = set_data(opts.weak_dir, opts.strong_dir, opts.label_dir)
    labels = np.concatenate(labels).astype(int)
    dataset_split = np.load(opts.split_path)
    # Compute realized mAP of each reward estimate.
    estimates = []
    if isinstance(opts.estimates, list):
        estimates = opts.estimates
    elif opts.estimates is not None:
        estimates = [opts.estimates]
    map_result = test_map(weak_data, strong_data, labels, estimates, dataset_split)
    Path(opts.save_dir).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(opts.save_dir, f'test_map.npy'), map_result)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('weak_dir', help="Directory to the weak detector output files.")
    args.add_argument('strong_dir', help="Directory to the strong detector output files.")
    args.add_argument('label_dir', help="Directory to the ground truth annotations.")
    args.add_argument('split_path', help="Path to the dataset split (for cross validation).")
    args.add_argument('save_dir', help="Directory to save the achieved mAP.")
    args.add_argument('--estimates', nargs='+', type=str, help='Directories to the reward estimation file(s).')
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
