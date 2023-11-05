import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

"""Parse the annotations in object detection datasets into standard .txt files."""


def coco_label(data, save):
    """
    Parse the annotations in COCO dataset to retrieve the labels.
    :param data: path to the dataset.
    :param save: path to save the extracted labels.
    """
    # Import the standard API for parsing COCO dataset.
    from pycocotools.coco import COCO
    # Identify the annotation file for each train/val split of the dataset.
    path = os.path.join(data, "annotations")
    for year, image_set in ('2017', 'train'), ('2017', 'val'):
        lbs_path = os.path.join(save, f'{image_set}{year}')
        Path(lbs_path).mkdir(parents=True, exist_ok=True)
        anno_path = os.path.join(path, f'instances_{image_set}{year}.json')
        anno = COCO(annotation_file=anno_path)
        # Parse the annotation file.
        cls_ids = anno.getCatIds()
        img_ids = anno.getImgIds()
        # Retrieve the labels of each image.
        for img_id in img_ids:
            img_info = anno.loadImgs([img_id])[0]
            img_name, w, h = img_info['file_name'], img_info['width'], img_info['height']
            img_name = img_name.split('.')[0]
            lb_path = os.path.join(lbs_path, f'{img_name}.txt')
            out_file = open(lb_path, 'w')
            anno_ids = anno.getAnnIds(imgIds=[img_id])
            objs = anno.loadAnns(anno_ids)
            for obj in objs:
                obj_box, obj_cls = obj['bbox'], obj['category_id']
                bbox = [(obj_box[0] + obj_box[2] / 2) / w, (obj_box[1] + obj_box[3] / 2) / h, obj_box[2] / w,
                        obj_box[3] / h]
                cls_id = cls_ids.index(obj_cls)
                out_file.write(" ".join([str(a) for a in (cls_id, *bbox)]) + '\n')
            out_file.close()
    return


def voc_label(data, save):
    """
    Parse the annotations in Pascal VOC dataset to retrieve the labels.
    :param data: path to the dataset.
    :param save: path to save the extracted labels.
    """
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def convert_box(size, box):
        # Normalize the bounding box by the image size and convert it to (x, y, w, h) format.
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    # Find all the images in the dataset according to the train/val split.
    path = os.path.join(data, "VOCdevkit")
    for year, image_set in ('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test'):
        lbs_path = os.path.join(save, f'{image_set}{year}')
        Path(lbs_path).mkdir(parents=True, exist_ok=True)
        # Retrieve the image ids in each sub-dataset.
        with open(os.path.join(path, f'VOC{year}/ImageSets/Main/{image_set}.txt')) as f:
            img_ids = f.read().strip().split()
        # Retrieve the labels of each image from its annotation file.
        for img_id in img_ids:
            lb_path = os.path.join(lbs_path, f'{img_id}.txt')
            out_file = open(lb_path, 'w')
            anno_path = os.path.join(path, f'VOC{year}/Annotations/{img_id}.xml')
            tree = ET.parse(anno_path)
            root = tree.getroot()
            img_size = root.find('size')
            w = int(img_size.find('width').text)
            h = int(img_size.find('height').text)
            # Collect the label of each object inside the image.
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls in class_names and not int(difficult) == 1:
                    xmlbox = obj.find('bndbox')
                    bbox = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
                    cls_id = class_names.index(cls)
                    out_file.write(" ".join([str(a) for a in (cls_id, *bbox)]) + '\n')
            out_file.close()
    return


def main(opts):
    # Choose label processing function according to the specified dataset.
    if opts.dataset == 'coco':
        coco_label(opts.data_dir, opts.save_dir)
    else:
        voc_label(opts.data_dir, opts.save_dir)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('data_dir', help="Directory where the dataset is downloaded and extracted.")
    args.add_argument('save_dir', help="Directory to save the processed object annotations.")
    args.add_argument('--dataset', type=str, default="coco", help="The dataset to process ('coco' or 'voc').")
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
