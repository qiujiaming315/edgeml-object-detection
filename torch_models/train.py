import argparse
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
from pathlib import Path
from collections import Counter

# Import downloaded reference codes from torchvision.references.detection
from references.engine import train_one_epoch, evaluate
from references.utils import collate_fn, save_on_master

"""Train the torchvision models from scratch using the VOC dataset."""

# Collection of object detectors.
models = {"ssd": torchvision.models.detection.ssdlite320_mobilenet_v3_large,
          "faster_rcnn": torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn_v2,
          "retinanet": torchvision.models.detection.retinanet.retinanet_resnet50_fpn_v2}

# Class names of the VOC dataset.
voc_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
             'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def image_transform(image):
    """
    Transform a PIL image into tensor.
    :param image: the PIL image.
    :return: the transformed image.
    """
    image = torchvision.transforms.functional.pil_to_tensor(image)
    image = image / 255
    return image


def target_transform(target):
    """
    Transform the VOC image annotation to a dictionary of tensors.
    :param target: the VOC image annotation.
    :return: the transformed annotation.
    """
    # Get bounding box coordinates and class label for each object.
    objects = target['annotation']['object']
    num_objs = len(objects)
    boxes, labels = [], []
    for obj in objects:
        coord = obj['bndbox']
        boxes.append([int(coord['xmin']), int(coord['ymin']), int(coord['xmax']), int(coord['ymax'])])
        labels.append(voc_names.index(obj['name']))
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    # # Create unique identifier for the image.
    # image_name = re.sub("[^0-9]", "", target['annotation']['filename'])
    # image_id = torch.tensor([int(image_name)])
    # # Compute the bounding boxes area.
    # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    # # Suppose all instances are not crowd.
    # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
    target = {"boxes": boxes, "labels": labels}
    return target


def create_dataloader(img_dir, batch_size):
    """
    Create the training and test dataloader for the VOC dataset.
    :param img_dir: root directory of VOC dataset.
    :param batch_size: training batch size.
    :return: the training and test dataloader.
    """
    train_data1 = torchvision.datasets.VOCDetection(img_dir, year="2007", image_set="trainval",
                                                    transform=image_transform, target_transform=target_transform)
    train_data2 = torchvision.datasets.VOCDetection(img_dir, year="2012", image_set="trainval",
                                                    transform=image_transform, target_transform=target_transform)
    train_dataset = torch.utils.data.ConcatDataset([train_data1, train_data2])
    test_dataset = torchvision.datasets.VOCDetection(img_dir, year="2007", image_set="test",
                                                     transform=image_transform, target_transform=target_transform)
    train_sampler = RandomSampler(train_dataset)
    test_sampler = SequentialSampler(test_dataset)
    train_batch_sampler = BatchSampler(train_sampler, batch_size, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, collate_fn=collate_fn)
    return train_dataloader, test_dataloader


def main(opts):
    # Load the VOC dataset.
    train_dataloader, test_dataloader = create_dataloader(opts.img_dir, opts.batch_size)
    Path(opts.save_dir).mkdir(parents=True, exist_ok=True)
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # Load the object detector (without pre-trained weights from COCO).
    model = models[opts.model](num_classes=21).to(device)
    # Construct an optimizer and a learning rate scheduler.
    params = [p for p in model.parameters() if p.requires_grad]
    if opts.opt == "sgd":
        optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
    elif opts.opt == "adamw":
        optimizer = torch.optim.AdamW(params, lr=opts.lr, weight_decay=opts.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {opts.opt}. Only SGD and AdamW are supported.")
    if opts.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_steps, gamma=opts.lr_gamma)
    elif opts.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{opts.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )
    # Load the checkpoint.
    if opts.resume:
        checkpoint = torch.load(opts.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        lr_scheduler.milestones = Counter(opts.lr_steps)
        lr_scheduler.gamma = opts.lr_gamma
        opts.start_epoch = checkpoint["epoch"] + 1
    print("Start training")
    for epoch in range(opts.start_epoch, opts.epochs):
        # Train for one epoch, printing every 100 iterations.
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=100)
        # Update the learning rate.
        lr_scheduler.step()
        # Save the model.
        if opts.save_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": opts,
                "epoch": epoch,
            }
            if epoch % 10 == 0:
                save_on_master(checkpoint, os.path.join(opts.save_dir, f"model_{epoch}.pth"))
            save_on_master(checkpoint, os.path.join(opts.save_dir, "checkpoint.pth"))
        # Evaluate on the test dataset.
        # evaluate(model, test_dataloader, device=device)
        print(f"Epoch {epoch} finished")
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('img_dir', help="Root directory of the VOC dataset.")
    args.add_argument('save_dir', help="Directory to save the trained model weights.")
    args.add_argument('--model', type=str, default="ssd",
                      help="The object detector from torchvision. Available choices include 'ssd', 'faster_rcnn', " +
                           "and 'retinanet'.")
    args.add_argument('-b', '--batch-size', default=32, type=int, help="Batch size for model training.")
    args.add_argument('--epochs', type=int, default=30, help="Number of total epochs to run.")
    args.add_argument('--opt', default="sgd", type=str, help="optimizer")
    args.add_argument('--lr', default=0.02, type=float, help="initial learning rate")
    args.add_argument('--momentum', default=0.9, type=float, help="momentum")
    args.add_argument('-wd', '--weight-decay', default=1e-4, type=float, help="weight decay")
    args.add_argument('--lr-scheduler', default="multisteplr", type=str, help="name of lr scheduler")
    args.add_argument('--lr-steps', default=[16, 22], nargs="+", type=int,
                      help="decrease lr every step-size epochs (multisteplr scheduler only)")
    args.add_argument('--lr-gamma', default=0.1, type=float,
                      help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)")
    args.add_argument("--resume", default="", type=str, help="path of checkpoint")
    args.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
