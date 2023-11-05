# Title: To be Added

todo: insert system diagram here.

## Overview

todo: add link to the paper.

## Requirements

We recommend a recent Python 3.7+ distribution of [Anaconda](https://www.anaconda.com/products/individual) with `numpy`, `torch`, `scikit-learn`, and `pycocotools` installed.

## ORIE Calculation and Estimation

#### Dataset Preparation

We use public datasets like [Microsoft COCO](https://cocodataset.org/#home) and [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) to train object detectors and evaluate our ORIE estimation approaches.

Upon downloading the datasets, you should use our code in `data_processing/label.py` to transform the ground truth bounding box annotations into standard format (according to the [YOLOv5](https://github.com/ultralytics/yolov5) standard) stored in `.txt` files.

We perform cross validation on the validation sets of COCO and VOC to evaluate the offloading reward estimation approaches. To evenly split these datasets into k-folds, you can use our code in `data_processing/dataset_split.py`.
