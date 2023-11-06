# Title: To be Added

todo: insert system diagram here.

## Overview

todo: add link to the paper.

## Requirements

We recommend a recent Python 3.7+ distribution of [Anaconda](https://www.anaconda.com/products/individual) with `numpy`, `torch`, `torchvision`, `scikit-learn`, and `pycocotools` installed.

## ORIE Calculation and Estimation

#### Dataset Preparation

We use public datasets like [Microsoft COCO](https://cocodataset.org/#home) and [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) to train object detectors and evaluate our ORIE estimation approaches.

Upon downloading the datasets, you should use our code in `data_processing/label.py` to transform the ground truth bounding box annotations into standard format (according to the [YOLOv5](https://github.com/ultralytics/yolov5) standard) stored in `.txt` files.

We perform cross validation on the validation sets of COCO and VOC to evaluate the offloading reward estimation approaches. To evenly split these datasets into k-folds, you can use our code in `data_processing/dataset_split.py`.

#### Object Detector Training

The training datasets of COCO and VOC are reserved for training the object detectors. Pre-trained weights on COCO can usually be found from the public release of most object detectors. For example, the four object detectors we investigated in the paper: [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt), [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt), [SSD+MobileNetv3](https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth), [Faster R-CNN+ResNet-50-FPN](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth).

You will need to train an object detector pair by yourself if you want to evaluate our ORIE-based offloading reward on other datasets. Take VOC as an example, you may refer to our `yolov5_scripts.md` for the training scripts of YOLOv5 models and use our `torch_models/train.py` to train the `torchvision` models (SSD and Faster R-CNN).

#### Detection Output Collection

With a trained object detector pair, the next step is to collect their detection outputs so that the offloading reward values can be computed.

You may again refer to our `yolov5_scripts.md` for the scripts that collect the detection outputs of the YOLOv5 models. For the `torchvision` models we provide `torch_models/detect.py` to facilitate the output collection.

#### ORIE Calculation

Once you have collected the detection outputs of an object detector pair, you can use our `reward.py` to compute the ORIE (or ORI and DCSB) value of each image as its offloading reward. An example run looks like:
```script
# Compute ORIE using 1000 ensemble images.
# WEAK_PATH, STRONG_PATH: path to your weak/strong detector outputs.
# LABEL_PATH: path to the dataset annotations.
# SAVE_PATH: path to save the computed rewards.
python reward.py WEAK_PATH STRONG_PATH LABEL_PATH SAVE_PATH --method orie --num-ensemble 1000
```

#### Feature Extraction

Next, you need to extract features from the weak detector's results to estimate the offloading reward. We considered two types of features: the outputs, and the intermediate feature maps of the weak detector.

`data_processing/extract_feature.py` can be used to extract features from the top bounding box proposals of the weak detector's outputs.

`yolov5_scripts.md` again provides scripts to extract feature maps from the hidden layers of the weak detector (YOLOv5n).

#### ORIE Estimation

You can now use our `regression.py` to train a regression model as an offloading reward estimator using the extracted features. Here is an example run:
```script
# Train a CNN-based model as the reward estimator.
# FEATURE_PATH: path to the extracted features.
# REWARD_PATH: path to the computed offloading rewards.
# SPLIT_PATH: path to save the dataset split (for cross validation).
# SAVE_PATH: path to save the estimated offloading rewards.
python regression.py FEATURE_PATH REWARD_PATH SPLIT_PATH SAVE_PATH --normalize --weight --stage 24 --resize 0 --model CNN
```

Note that, if you are using a CNN (or MLP) based regression model, you first need to manually configure the `CNNOpt` class to set the architecture and other hyper-parameters (e.g., learning rate, weight decay) of the CNN model before running `regression.py`. `CNNOpt` supports setting multiple convolutional layers followed multiple fully-connected layers. Setting no convolutional layer defaults to an MLP model. Please make sure the input features, as well as each hidden layer in the CNN model, have consistent shapes with the next layer before you execute `regression.py`.

#### Offloading Performance Evaluation

Finally, to evaluate the performance of the reward estimation models in offloading images, you can run our `test.py` like:
```script
# Train a CNN-based model as the reward estimator.
# WEAK_PATH, STRONG_PATH: path to your weak/strong detector outputs.
# LABEL_PATH: path to the dataset annotations.
# SPLIT_PATH: path to save the dataset split (for cross validation).
# SAVE_PATH: path to save the achieved mAP for various offloading ratios.
# ESTIMATE_PATH: path to the estimated offloading rewards.
python test.py WEAK_PATH STRONG_PATH LABEL_PATH SPLIT_PATH SAVE_PATH --estimates ESTIMATE_PATH1 ESTIMATE_PATH2 ESTIMATE_PATH3 ...
```
