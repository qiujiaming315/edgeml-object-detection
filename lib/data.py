import numpy as np
import os
import torch
from torchvision.ops import roi_align, roi_pool

from .metrics import xywh2xyxy, box_correct, roi_padding

"""Utility functions for loading and processing data."""


def load_data(path, files, with_conf=False):
    """
    Load the bounding box data.
    :param path: path to the folder where the data is stored.
    :param files: the files to find in the folder.
    :param with_conf: whether confidence scores are stored.
    :return: the loaded data as a list of tuples.
    """
    data = list()
    # Read the data from each file.
    for file in files:
        file_path, file_data = os.path.join(path, file), tuple()
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                file_data = [line.strip().split(' ') for line in lines]
                # Parse the labels.
                if len(file_data) > 0:
                    file_data = [np.array(x).astype(float) for x in zip(*file_data)]
                    if with_conf:
                        file_data = (file_data[0].astype(int),
                                     xywh2xyxy(np.concatenate([x[:, np.newaxis] for x in file_data[1:-1]], axis=1)),
                                     file_data[-1])
                    else:
                        file_data = (file_data[0].astype(int),
                                     xywh2xyxy(np.concatenate([x[:, np.newaxis] for x in file_data[1:]], axis=1)))
                else:
                    # Place an empty tuple when there is no label.
                    file_data = tuple()
        data.append(file_data)
    return data


def set_data(weak, strong, label):
    """
    Set the data for computing mAPI from the weak and strong detectors' outputs and the ground truth labels.
    :param weak: path to the weak weak detector output files.
    :param strong: path to the weak strong detector output files.
    :param label: path to the ground truth labels.
    :return: the weak detector's processed output, the strong detector's processed output, and the labels.
    """
    img_names = os.listdir(label)
    weak_data = load_data(weak, img_names, True)
    strong_data = load_data(strong, img_names, True)
    labels = load_data(label, img_names)
    # Replace with the second line to compute mAP@0.5:0.95 instead of mAP@0.5.
    iouv = np.array([0.5])
    # iouv = np.linspace(0.5, 0.95, 10)
    for idx, (w, s, l) in enumerate(zip(weak_data, strong_data, labels)):
        # Initialize arrays to handle the special case when the detector reports no detection results.
        weak_correct, strong_correct = np.zeros((0, len(iouv)), dtype=bool), np.zeros((0, len(iouv)), dtype=bool)
        weak_conf, weak_cls, strong_conf, strong_cls = np.array([]), np.array([]), np.array([]), np.array([])
        if len(w) > 0:
            weak_correct, weak_conf, weak_cls = np.zeros((len(w[0]), len(iouv)), dtype=bool), w[2], w[0]
        if len(s) > 0:
            strong_correct, strong_conf, strong_cls = np.zeros((len(s[0]), len(iouv)), dtype=bool), s[2], s[0]
        if len(l) > 0:
            labels[idx] = l[0]
            if len(w) > 0:
                weak_correct = box_correct(np.concatenate((w[1], w[2][:, np.newaxis], w[0][:, np.newaxis]), axis=1),
                                           np.concatenate((l[0][:, np.newaxis], l[1]), axis=1), iouv)
            if len(s) > 0:
                strong_correct = box_correct(np.concatenate((s[1], s[2][:, np.newaxis], s[0][:, np.newaxis]), axis=1),
                                             np.concatenate((l[0][:, np.newaxis], l[1]), axis=1), iouv)
        else:
            # Handle the special case when the image has no label.
            labels[idx] = np.array([])
        weak_data[idx] = (weak_correct, weak_conf, weak_cls)
        strong_data[idx] = (strong_correct, strong_conf, strong_cls)
    return weak_data, strong_data, labels


def load_feature(path, stage, pool=True, batch_size=128, func="avg", size=8):
    """
    Load the feature maps.
    :param path: path to the folder where the feature maps are stored.
    :param stage: the stage number of the feature map.
    :param pool: whether the feature maps should be resized with roi pooling.
    :param batch_size: batch size for pooling image, only active when "pool" is enabled.
    :param func: function for pooling image, "avg" or "max", only active when "pool" is enabled.
    :param size: size (H, W) of the feature map after pooling, only active when "pool" is enabled.
    :return: the loaded (pooled) feature maps as a list or ndarray.
    """
    # The stage names of yolov5 detectors.
    v5_names = ['Conv', 'Conv', 'C3', 'Conv', 'C3', 'Conv', 'C3', 'Conv', 'C3', 'SPPF', 'Conv', 'Upsample', 'Concat',
                'C3', 'Conv', 'Upsample', 'Concat', 'C3', 'Conv', 'Concat', 'C3', 'Conv', 'Concat', 'C3']
    data = list()
    # Read the data from each npy file.
    images = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
    if pool:
        pool_func = roi_align if func == "avg" else roi_pool
        # Split images into batches and resize the feature maps through roi pooling.
        for ndx in range(0, len(images), batch_size):
            features, coords = list(), list()
            for img_name in images[ndx:min(ndx + batch_size, len(images))]:
                file_path = os.path.join(path, img_name, f"stage{stage}_{v5_names[stage]}_features.npy")
                file_data = np.load(file_path)
                # Pad the feature maps to set the original feature map as an roi for the pooling operation.
                feature_map, coord = roi_padding(file_data)
                features.append(feature_map)
                coords.append(torch.tensor(coord, dtype=torch.float))
            data.append(pool_func(torch.from_numpy(np.array(features)), coords, size).numpy())
        data = np.concatenate(data)
    else:
        # Directly load the feature maps without modification.
        for img_name in images:
            file_path = os.path.join(path, img_name, f"stage{stage}_{v5_names[stage]}_features.npy")
            file_data = np.load(file_path)
            data.append(file_data)
    return data
