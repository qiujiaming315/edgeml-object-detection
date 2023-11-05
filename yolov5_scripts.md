This documents present the scripts we used to train and collect data from the YOLOv5 models. To run these scripts, you need to first clone the [YOLOv5](https://github.com/ultralytics/yolov5) GitHub repository and navigate into the cloned directory.

The following scripts use YOLOv5n as an example, you can replace `yolov5n` with `yolov5m` to easily switch to YOLOv5m.

#### Training

To train a model on VOC from scratch you should run:
``` shell
python train.py --data VOC.yaml --weights '' --cfg yolov5n.yaml --epochs 300
```
Note: you should set the dataset directory to your local copy of the VOC dataset in VOC.yaml to avoid automatic downloading of VOC.

#### Detection Output

To collect the detection outputs of the object detector on COCO you should run:
```shell
python val.py --weights (PATH_TO_YOUR_MODEL_WEIGHTS) --data coco.yaml --save-txt --save-conf
```
Note: again, set coco.yaml appropriately to avoid repeated downloading of the dataset.

#### Intermediate Feature Map
To collect feature maps from hidden layers you should run:
```shell
python detect.py --weights (PATH_TO_YOUR_MODEL_WEIGHTS) --source (PATH_TO_YOUR_DATASET) --visualize --nosave
```
Note: by default the above script will save feature maps from all the 23 hidden layers of YOLOv5 models to `npy` files and create a `png` image to visualize the feature maps. To keep feature maps from only the interested layers (and discard the png visualization), you need to modify the `feature_visualization` function in `utils/plots.py`. For example, in our paper we focused on four hidden layers with a stage number of 9 (backbone), 17, 20, and 23 (detection heads), respectively. To collect the feature maps from these four layers we modified `feature_visualization` to:

```shell
def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    if 'Detect' not in module_type:
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1 and stage in [9, 17, 20, 23]:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename
            np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())  # npy save
```
