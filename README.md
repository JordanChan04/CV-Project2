# CV-Project2
Bird Species Recognition and Object Detection

# Task 2: Object Detection (Faster-RCNN & YOLOV3)

## Prerequisites

1. Python 3.7 or later
2. Pytorch 1.5.0 or later
3. CUDA 10.1 or later (If you are using a GPU)

## Step 1: Prepare the MMdetection Environment

Before starting, ensure you have the MMDetection environment prepared. For complete instructions on setup, please refer to the [official MMDetection installation guide](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation).

## Step 2: Prepare the PASCAL VOC Dataset

Download the PASCAL VOC Dataset from its [official website](http://host.robots.ox.ac.uk/pascal/VOC/). Please organize the data in the following format:

```
VOCdevkit
├── VOC2007
│   ├── Annotations
│   ├── ImageSets
│   ├── JPEGImages
│   ├── SegmentationClass
│   └── SegmentationObject
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    └── SegmentationObject
```

## Step 3: Training the Model

Navigate to your MMdetection directory. Use the script `tools/train.py` followed by the configuration file intended for Faster R-CNN / YOLOv3 on PASCAL VOC Dataset. Please replace `path/to/your/work_dir` with the actual path.

```bash
!python tools/train.py configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py --work-dir path/to/your/work_dir
```

```bash
!python tools/train.py configs/pascal_voc/yolov3_d53_8xb8-ms-608-273e_coco.py --work-dir path/to/your/work_dir
```

## Step 4: Review the Results

Import the necessary modules and the model checkpoint for the test image:

```python
import mmcv
from mmdet.apis import init_detector, inference_detector
img = mmcv.imread('/path/to/your/test_image.jpg', channel_order='rgb')
checkpoint_file = '/path/to/your/checkpoint_file.pth'
model = init_detector(cfg, checkpoint_file, device='cpu')
new_result = inference_detector(model, img)
```
To visualize the detection results:

```python
from mmengine.visualization import Visualizer
visualizer_now = Visualizer.get_current_instance()
visualizer_now.dataset_meta = model.dataset_meta
visualizer_now.add_datasample(
    'new_result',
    img,
    data_sample=new_result,
    draw_gt=False,
    wait_time=0,
    out_file=None,
    pred_score_thr=0.5
)
visualizer_now.show()
```

Make sure to replace `/path/to/your/test_image.jpg` and `/path/to/your/checkpoint_file.pth` with the path to your test image and checkpoint file, respectively. 
