import numpy as np

"""Utility functions for computing object detection related metrics (e.g. mAP@0.5)."""


def xywh2xyxy(x):
    """
    Convert bounding boxes from (x, y, w, h) to (x1, y1, x2, y2) format.
    Adapted from xywh2xyxy() in https://github.com/ultralytics/yolov5/blob/master/utils/general.py
    :param x: bounding boxes in (x, y, w, h) format (n*4 ndarray).
    :return: an n*4 array for the converted bounding boxes in (x1, y1, x2, y2) format.
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def roi_padding(x):
    """
    Pad the feature map and set it as an roi for pooling.
    :param x: the feature map for padding.
    :return: the feature map and its coordinates.
    """
    c, h, w = x.shape
    if h < w:
        padding = ((0, 0), (0, w - h), (0, 0))
    else:
        padding = ((0, 0), (0, 0), (0, h - w))
    # Padding value does not matter, use the default zero padding mode.
    feature_map = np.pad(x, padding)
    coord = [[0, 0, w, h]]
    return feature_map, coord


def box_correct(detections, labels, iouv):
    """
    Compute the correct predictions matrix.
    Adapted from process_batch() in https://github.com/ultralytics/yolov5/blob/master/val.py
    :param detections: the detection output in (x1, y1, x2, y2, conf, class) format (n*6 ndarray).
    :param labels: the ground truth labels in (class, x1, y1, x2, y2) format (m*5 ndarray).
    :param iouv: vector of IoU thresholds (t ndarray).
    :return: an n*t array reporting the correctness of every detection at every IoU threshold.
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        # IoU > threshold and classes match.
        x = np.where(np.logical_and((iou >= iouv[i]), correct_class))
        if x[0].shape[0]:
            # Summarize the pairwise matching result between labels and detections
            # in (label index, detection index, IoU value) format.
            matches = np.concatenate((np.stack(x, axis=1), iou[x[0], x[1]][:, np.newaxis]), axis=1)
            if x[0].shape[0] > 1:
                # Sort by the IoU value.
                matches = matches[matches[:, 2].argsort()[::-1]]
                # Keep the matching pair with the highest IoU value.
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def box_iou(bbox1, bbox2):
    """
    Compute the IoU between two sets of bounding boxes.
    Adapted from box_iou() in https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py
    :param bbox1: the first set of bounding boxes in (x1, y1, x2, y2) format (m*4 ndarray).
    :param bbox2: the second set of bounding boxes in (x1, y1, x2, y2) format (n*4 ndarray).
    :return: an m*n array with the IoU between each pair of bounding boxes from set bbox1 and bbox2.
    """
    x1 = np.maximum(bbox1[:, 0:1], bbox2[:, 0])
    y1 = np.maximum(bbox1[:, 1:2], bbox2[:, 1])
    x2 = np.minimum(bbox1[:, 2:3], bbox2[:, 2])
    y2 = np.minimum(bbox1[:, 3:4], bbox2[:, 3])
    # Calculate the area of the intersection.
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    # Calculate the area of the union.
    box1_area = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    box2_area = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    # Calculate IoU.
    iou = inter_area / (box1_area[:, np.newaxis] + box2_area - inter_area)
    return iou


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute the AP (average precision) for each class and each IoU threshold given the recall and precision curves.
    Adapted from ap_per_class() in https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py
    :param tp: true positives (n*t nparray, n is the number of predictions and t is the number of IoU thresholds).
    :param conf: confidence score from 0-1 for the predicted class (n nparray).
    :param pred_cls: predicted object classes (n nparray).
    :param target_cls: true object classes (m nparray, m is the number of true objects).
    :param eps: a small value to avoid dividing by zero.
    :return: a k*t average precision array, k is the number of classes.
    """
    # Sort by confidence score.
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    # Find unique classes and number of true objects for each class.
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    # Create Precision-Recall curve and compute AP for each class.
    ap = np.zeros((nc, tp.shape[1]))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of true objects
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue
        # Accumulate FPs and TPs.
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        # Recall.
        recall = tpc / (n_l + eps)  # recall curve
        # Precision.
        precision = tpc / (tpc + fpc)  # precision curve
        # AP from recall-precision curve.
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    return ap


def compute_ap(recall, precision, method='interp'):
    """
    Compute the AP (average precision) given the precision and recall curves.
    Adapted from compute_ap() in https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py
    :param recall: the recall curve (list).
    :param precision: The precision curve (list).
    :param method: The method to compute average precision (the 101-point COCO standard is applied by default).
    :return: the average precision.
    """
    # Append sentinel values to beginning and end.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # Compute the precision envelope.
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    # Integrate area under curve.
    if method == 'interp':  # 101-point interp (COCO)
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
    return ap
