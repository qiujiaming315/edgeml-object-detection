import numpy as np
import argparse
import os
from pathlib import Path

from lib.data import set_data
from lib.metrics import ap_per_class


"""Test the performance of offloading reward estimation(s) by computing the realized mAP after offloading."""


def test_map(weak_data, strong_data, labels, reward_estimates, offload_ratio):
    """
    Compute realized mAP given reward estimates.
    :param weak_data: weak detector's processed output.
    :param strong_data: the strong detector's processed output.
    :param labels: the ground truth annotations.
    :param reward_estimates: a list containing path(s) to the file(s) with estimated reward.
    :param offload_ratio: the offloading ratio.
    :return: an array of the computed mAP values, one for each reward estimate.
    """
    num_img = len(weak_data)
    labels = np.concatenate(labels).astype(int)
    mAP = []
    # Load estimated reward from each file.
    for f in reward_estimates:
        reward = np.load(f)['val_est']
        assert len(reward) == num_img, "Inconsistent number of images from the detection outputs and reward estimates."
        mapi_thresh = reward[np.argsort(-reward)[int(num_img * offload_ratio)]]
        mapi_thresh = max(0, mapi_thresh)
        mapi_mask = reward > mapi_thresh
        detection = [strong_data[s] if m else weak_data[s] for s, m in zip(range(num_img), mapi_mask)]
        # Compute the mAP after offloading using the estimated reward with a fixed threshold policy.
        mAP.append(np.mean(ap_per_class(*[np.concatenate(x, axis=0) for x in zip(*detection)], labels)))
    return np.array(mAP)


def main(opts):
    weak_data, strong_data, labels = set_data(opts.weak, opts.strong, opts.label)
    # Compute realized mAP of each reward estimate.
    estimates = []
    if isinstance(opts.estimates, list):
        estimates = opts.estimates
    elif opts.estimates is not None:
        estimates = [opts.estimates]
    mAP = test_map(weak_data, strong_data, labels, estimates, opts.offload_ratio)
    Path(opts.save).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(opts.save, f'test_map.npy'), mAP)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('weak', help="Path to the weak detector output files (for the validation set).")
    args.add_argument('strong', help="Path to the strong detector output files (for the validation set).")
    args.add_argument('label', help="Path to the ground truth labels.")
    args.add_argument('save', help="Path to save the computed mAPI results.")
    args.add_argument('--offload_ratio', type=float, default=0.1,
                      help="Offloading ratio for evaluating the estimated offloading reward.")
    args.add_argument('--estimates', nargs='+', type=str, help='Reward estimation file(s) to be evaluated.')
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
