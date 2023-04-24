import numpy as np
import argparse
import os
from pathlib import Path

from lib.data import set_data
from lib.metrics import ap_per_class

"""Test the performance of offloading reward estimation(s) by computing the realized mAP after offloading."""


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
    mAP = []
    # Load reward estimates of each trained model.
    for estimate_path in reward_estimates:
        mAP_ratios = np.zeros((10,))
        # Check the mAP of each fold in cross validation.
        for cv_idx, val_mask in enumerate(dataset_split):
            # Prepare the detection outputs and ground truth labels for the validation set.
            strong_val = [strong_data[i] for i, m in enumerate(val_mask) if m]
            weak_val = [weak_data[i] for i, m in enumerate(val_mask) if m]
            labels_val = [labels[i] for i, m in enumerate(val_mask) if m]
            labels_val = np.concatenate(labels_val).astype(int)
            # Load the reward estimate.
            reward_data = np.load(os.path.join(estimate_path, f"estimate{cv_idx + 1}.npz"))
            train_reward = reward_data['train_est']
            val_reward = reward_data['val_est']
            for ratio_idx, offload_ratio in enumerate(np.arange(0.1, 1.1, 0.1)):
                mapi_thresh = train_reward[np.argsort(-train_reward)[int((len(train_reward) - 1) * offload_ratio)]]
                mapi_thresh = max(0, mapi_thresh)
                mapi_mask = val_reward > mapi_thresh
                detection = [strong_val[s] if m else weak_val[s] for s, m in enumerate(mapi_mask)]
                # Compute the mAP after offloading using the estimated reward with a fixed threshold policy.
                mAP_ratios[ratio_idx] += np.mean(
                    ap_per_class(*[np.concatenate(x, axis=0) for x in zip(*detection)], labels_val))
        mAP.append(mAP_ratios / len(dataset_split))
    return np.array(mAP)


def main(opts):
    weak_data, strong_data, labels = set_data(opts.weak, opts.strong, opts.label)
    dataset_split = np.load(opts.split_path)
    # Compute realized mAP of each reward estimate.
    estimates = []
    if isinstance(opts.estimates, list):
        estimates = opts.estimates
    elif opts.estimates is not None:
        estimates = [opts.estimates]
    mAP = test_map(weak_data, strong_data, labels, estimates, dataset_split)
    Path(opts.save).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(opts.save, f'test_map.npy'), mAP)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('weak', help="Path to the weak detector output files (for the validation set).")
    args.add_argument('strong', help="Path to the strong detector output files (for the validation set).")
    args.add_argument('label', help="Path to the ground truth labels.")
    args.add_argument('split_path', help="Path to the dataset split.")
    args.add_argument('save', help="Path to save the test results.")
    # args.add_argument('--offload_ratio', type=float, default=0.1,
    #                   help="Offloading ratio for evaluating the estimated offloading reward.")
    args.add_argument('--estimates', nargs='+', type=str, help='Directories to the reward estimation file(s).')
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
