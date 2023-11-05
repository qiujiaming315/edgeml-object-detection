import numpy as np
import argparse
import os
import time
import pickle
from dataclasses import dataclass
from pathlib import Path

from sklearn.svm import LinearSVC

import lib.utils as ut
from lib.data import load_data, load_feature

"""Implementation of some baseline methods for comparison."""


@dataclass
class SaveOpt:
    """Options for loading/saving model weights."""
    model_dir: str = ''  # Directory to save the model weights.
    load: bool = False  # If model is loaded from pre-trained weights.
    save: bool = True  # If model weights need to be saved after training.
    model_idx: int = 1  # The index of model in cross validation.


_SaveOPT = SaveOpt()


def fit_af(data, weight=3.0, save_opts=_SaveOPT):
    """
    Implementation of the Adaptive Feeding paper.
    Link to paper: https://ieeexplore.ieee.org/abstract/document/8237641
    Code references: https://github.com/funnyzhou/Adaptive_Feeding
    :param data: features (inputs) and rewards (labels) for training and validation.
    :param weight: weight for the positive reward class.
    :param save_opts: model saving options.
    :return: the estimated offloading decisions for the training and validation images,
             and the average inference time for each image.
    """
    # Retrieve data and flatten the feature maps.
    train_feature, val_feature, train_reward, val_reward = data
    train_feature = [x.flatten() for x in train_feature]
    val_feature = [x.flatten() for x in val_feature]
    # Load model if specified.
    if save_opts.load and save_opts.model_dir != '':
        cls = pickle.load(open(os.path.join(save_opts.model_dir, f'wts{save_opts.model_idx}.pickle'), 'rb'))
    else:
        sample_weight = {0: 1.0, 1: weight}
        cls = LinearSVC(dual=False, class_weight=sample_weight).fit(train_feature, train_reward)
    # Make predictions for both the training and test set.
    time1 = time.perf_counter()
    train_est = cls.predict(train_feature)
    time2 = time.perf_counter()
    val_est = cls.predict(val_feature)
    time3 = time.perf_counter()
    train_time, val_time = (time2 - time1) / len(train_reward), (time3 - time2) / len(val_reward)
    train_acc = np.sum(train_reward == train_est) / len(train_reward)
    val_acc = np.sum(val_reward == val_est) / len(val_reward)
    print(f"Trained Adaptive Feeding SVM with training accuracy: {train_acc:.3f}, validation accuracy: {val_acc:.3f}")
    # Save model if specified.
    if save_opts.save and save_opts.model_dir != '':
        Path(save_opts.model_dir).mkdir(parents=True, exist_ok=True)
        pickle.dump(cls, open(os.path.join(save_opts.model_dir, f'wts{save_opts.model_idx}.pickle'), 'wb'))
    return {"train_est": train_est, "val_est": val_est, "train_time": train_time, "val_time": val_time}


def fit_dcsb(data, train_label, save_opts=_SaveOPT):
    """
    Implementation of the DCSB paper.
    Link to paper: https://ieeexplore.ieee.org/abstract/document/10272511
    :param data: features (inputs) and rewards (labels) for training and validation.
    :param train_label: number of ground truth annotations in each training image.
    :param save_opts: model saving options.
    :return: the estimated offloading decisions for the training and validation images,
             and the average inference time for each image.
    """
    # Retrieve data and flatten the feature maps.
    train_feature, val_feature, train_reward, val_reward = data

    def filter_box(feature, conf):
        # Filter bounding boxes given a confidence threshold.
        num_estimate, min_area = [], []
        for box in feature:
            conf_mask = box[0] > conf
            num_estimate.append(np.sum(conf_mask))
            mask_area = box[1][conf_mask]
            min_area.append(0 if len(mask_area) == 0 else np.amin(mask_area))
        return np.array(num_estimate, dtype=int), np.array(min_area)

    # Load model if specified.
    if save_opts.load and save_opts.model_dir != '':
        conf_thresh, num_thresh, area_thresh = pickle.load(
            open(os.path.join(save_opts.model_dir, f'wts{save_opts.model_idx}.pickle'), 'rb'))
    else:
        # Binary search to find the confidence threshold.
        low_conf, high_conf, mid_conf = 0, 1, 0
        min_loss, tor = False, 1e-4
        while not min_loss:
            mid_conf = (low_conf + high_conf) / 2
            num, area = filter_box(train_feature, mid_conf)
            num_diff = np.sum(num) - np.sum(train_label)
            if num_diff >= 0:
                low_conf = mid_conf
            else:
                high_conf = mid_conf
            min_loss = abs(num_diff) / np.sum(train_label) < tor
        conf_thresh = mid_conf
        estimate_num, estimate_area = filter_box(train_feature, conf_thresh)
        detect_num, detect_area = filter_box(train_feature, 0.5)
        # Search for the best object number threshold and minimum area threshold.
        train_class = estimate_num != detect_num
        best_accuracy = 0
        for n_thresh in range(1, 11):
            a_thresh_range = np.arange(0.2, 0.9, 0.01)
            accuracy = np.zeros_like(a_thresh_range)
            for a_idx, a_thresh in enumerate(a_thresh_range):
                train_class_copy = np.copy(train_class)
                sub_num, sub_area = estimate_num[train_class], estimate_area[train_class]
                train_class_copy[train_class] = np.logical_or(sub_num > n_thresh, sub_area < a_thresh)
                train_est = train_class_copy.astype(int)
                accuracy[a_idx] = np.sum(train_est == train_reward) / len(train_reward)
            a_best = np.argmax(accuracy)
            if accuracy[a_best] > best_accuracy:
                best_accuracy = accuracy[a_best]
                num_thresh = n_thresh
                area_thresh = a_thresh_range[a_best]
    # Make predictions for both the training and test set.
    time1 = time.perf_counter()
    train_est_num, train_est_area = filter_box(train_feature, conf_thresh)
    train_det_num, train_det_area = filter_box(train_feature, 0.5)
    train_class = train_est_num != train_det_num
    train_sub_num, train_sub_area = train_est_num[train_class], train_est_area[train_class]
    train_class[train_class] = np.logical_or(train_sub_num > num_thresh, train_sub_area < area_thresh)
    train_est = train_class.astype(int)
    time2 = time.perf_counter()
    val_est_num, val_est_area = filter_box(val_feature, conf_thresh)
    val_det_num, val_det_area = filter_box(val_feature, 0.5)
    val_class = val_est_num != val_det_num
    val_sub_num, val_sub_area = val_est_num[val_class], val_est_area[val_class]
    val_class[val_class] = np.logical_or(val_sub_num > num_thresh, val_sub_area < area_thresh)
    val_est = val_class.astype(int)
    time3 = time.perf_counter()
    train_time, val_time = (time2 - time1) / len(train_reward), (time3 - time2) / len(val_reward)
    train_acc = np.sum(train_reward == train_est) / len(train_reward)
    val_acc = np.sum(val_reward == val_est) / len(val_reward)
    print(f"Computed DCSB thresholds with training accuracy: {train_acc:.3f}, validation accuracy: {val_acc:.3f}")
    # Save model if specified.
    if save_opts.save and save_opts.model_dir != '':
        Path(save_opts.model_dir).mkdir(parents=True, exist_ok=True)
        pickle.dump((conf_thresh, num_thresh, area_thresh),
                    open(os.path.join(save_opts.model_dir, f'wts{save_opts.model_idx}.pickle'), 'wb'))
    return {"train_est": train_est, "val_est": val_est, "train_time": train_time, "val_time": val_time}


def get_area(bbox_coord):
    # Compute the area of the bounding boxes given coordinates (in xyxy format).
    area = (bbox_coord[:, 2] - bbox_coord[:, 0]) * (bbox_coord[:, 3] - bbox_coord[:, 1])
    return area


def main(opts):
    # Load the offloading rewards.
    reward_data = np.load(opts.reward_path)['reward']
    # Transform the reward into binary values since both baseline approaches perform binary classification.
    thresh = 0
    # thresh = np.sort(reward_data)[int(0.6 * len(reward_data))]
    reward_data = np.where(reward_data > thresh, 1, 0)
    # Load the dataset split.
    data_split = np.load(opts.split_path)
    assert len(reward_data) == data_split.shape[1], "Inconsistent number of data points from the dataset and the split."
    if opts.baseline == "af":
        # Load the weak detector feature maps.
        feature_data = load_feature(opts.data_dir, 24, pool=False)
        _SaveOPT.model_dir = os.path.join(opts.model_dir, f"{opts.positive_weight}")
    else:
        img_names = sorted(os.listdir(opts.label_dir))
        # Remove the extension from the file names.
        img_names = ['.'.join(name.split('.')[:-1]) for name in img_names]
        weak_data = load_data(opts.data_dir, img_names, True)
        feature_data = [(np.array([]), np.array([])) if len(wd) == 0 else (wd[2], get_area(wd[1])) for wd in weak_data]
        labels = load_data(opts.label_dir, img_names)
        label_num = np.array([0 if len(l) == 0 else len(l[0]) for l in labels], dtype=int)
        _SaveOPT.model_dir = opts.model_dir
    assert len(feature_data) == len(reward_data), "Inconsistent number of feature maps and offloading rewards."
    # Cross validation.
    for cv_idx, val_mask in enumerate(data_split):
        # Split the dataset.
        train_feature = [f for f, v in zip(feature_data, val_mask) if not v]
        val_feature = [f for f, v in zip(feature_data, val_mask) if v]
        train_reward = reward_data[np.logical_not(val_mask)]
        val_reward = reward_data[val_mask]
        # Train the model.
        print(f"==============================Cross Validation Fold {cv_idx + 1}==============================")
        _SaveOPT.model_idx = cv_idx + 1
        if opts.baseline == "af":
            result = fit_af((train_feature, val_feature, train_reward, val_reward), opts.positive_weight)
            # Save the estimated offloading reward.
            ut.save_result(os.path.join(opts.save_dir, f"{opts.positive_weight}"), result, cv_idx)
        else:
            train_label = label_num[np.logical_not(val_mask)]
            result = fit_dcsb((train_feature, val_feature, train_reward, val_reward), train_label)
            # Save the estimated offloading reward.
            ut.save_result(opts.save_dir, result, cv_idx)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('data_dir',
                      help="Directory that saves the data needed for predicting the offloading reward. "
                           "For Adaptive Feeding, this should be features extracted from the weak detector outputs. "
                           "For DCSB, this should be the weak detector's outputs.")
    args.add_argument('reward_path', help="Path to the (pre-computed) offloading reward.")
    args.add_argument('split_path', help="Path to the dataset split (for cross validation).")
    args.add_argument('save_dir', help="Directory to save the estimated offloading reward.")
    args.add_argument('--baseline', type=str, default="af", choices=['af', 'dcsb'],
                      help="The baseline method. Available choices include 'af' (Adaptive Feeding) "
                           "and 'dcsb' (difficult-case based small-big model).")
    args.add_argument('--positive_weight', type=float, default=3.0,
                      help="The weight for the positive reward class. Only active when baseline is 'af'.")
    args.add_argument('--label_dir', type=str, default='',
                      help="Directory that saves the ground truth annotations of the dataset. Only active when "
                           "baseline is 'dcsb'.")
    args.add_argument('--model_dir', type=str, default='', help="Directory to save the model weights.")
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
