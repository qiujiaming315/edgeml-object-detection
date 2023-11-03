import argparse
import os
import numpy as np
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from coco_labelmap import coco_to_yolov5

"""Collect the detection outputs of torchvision models."""


def load_weak_models(model_name: str, model_path: str, num_class: int):
    """
    Collection of strong object detectors.
    * ssd: https://pytorch.org/vision/stable/models/ssdlite.html
    * yolov5: https://github.com/ultralytics/yolov5
    """
    if model_name == "ssd":
        if model_path == "":
            # Load the model pretrained on COCO.
            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
        else:
            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(num_classes=num_class)
    elif model_name == "faster_rcnn":
        if model_path == "":
            # Load the model pretrained on COCO.
            model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        else:
            model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn_v2(num_classes=num_class)
    else:
        if model_path == "":
            # Load the model pretrained on COCO.
            model = torchvision.models.detection.retinanet.retinanet_resnet50_fpn_v2(weights="DEFAULT")
        else:
            model = torchvision.models.detection.retinanet.retinanet_resnet50_fpn_v2(num_classes=num_class)
    if model_path != "":
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt)
    return model


class ObjectDetectionDataset(Dataset):
    """Data loader for the COCO and VOC datasets."""

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_names = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.RGB)
        image = image / 255
        return image


def main(opts):
    # Load the images from the directory and create a dataset.
    img_names = sorted(os.listdir(opts.img_dir))
    dataset = ObjectDetectionDataset(opts.img_dir)
    dataloader = DataLoader(dataset, batch_size=1)
    num_class = 91 if opts.dataset == "coco" else 21
    # Get cpu or gpu device for data collection.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # Load the object detector and collect the outputs.
    model = load_weak_models(opts.model, opts.model_path, num_class).to(device)
    model.eval()
    # Create the directory to save detection outputs.
    Path(opts.save_dir).mkdir(parents=True, exist_ok=True)
    for img_idx, img in enumerate(dataloader):
        # Retrieve the detection output data.
        predictions = model(img.to(device))[0]
        labels = predictions["labels"].cpu().numpy()
        boxes = predictions["boxes"].detach().cpu().numpy()
        scores = predictions["scores"].detach().cpu().numpy()
        # Parse the output data.
        x_center = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
        y_center = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        img_height = img.size(2)
        img_width = img.size(3)
        if opts.dataset == "coco":
            labels = np.array([coco_to_yolov5[l] for l in labels])
        else:
            labels -= 1
        # Ignore mis-detected labels.
        label_mask = labels != -1
        labels = labels[label_mask]
        x_center = x_center[label_mask] / img_width
        y_center = y_center[label_mask] / img_height
        width = width[label_mask] / img_width
        height = height[label_mask] / img_height
        scores = scores[label_mask]
        # Save the detection output to .npy file.
        detection_output = np.concatenate((labels[:, np.newaxis], x_center[:, np.newaxis], y_center[:, np.newaxis],
                                           width[:, np.newaxis], height[:, np.newaxis], scores[:, np.newaxis]), axis=1)
        img_name = img_names[img_idx][:-4]
        np.save(os.path.join(opts.save_dir, img_name + ".npy"), detection_output)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('img_dir', help="Directory that saves the image dataset for detection.")
    args.add_argument('save_dir', help="Directory to save the detection outputs.")
    args.add_argument('--dataset', type=str, default="coco", help="The dataset to process ('coco' or 'voc').")
    args.add_argument('--model', type=str, default="ssd",
                      help="The object detector from torchvision. Available choices include 'ssd', 'faster_rcnn', " +
                           "and 'retinanet'.")
    args.add_argument("--model_path", type=str, default="",
                      help="Location of the saved object detection model weights. Use empty string to load default "
                           "weights pre-trained on COCO.")
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
