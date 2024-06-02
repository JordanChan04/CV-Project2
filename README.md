# CV-Project2
Bird Species Recognition and Object Detection

# Task 1: Bird Species Recognition (Resnet-18)

## 1. Introduction

This guide will walk you through creating an image classifier for the CUB_200_2011 bird species dataset. We provide a step by step example on how to train a deep learning model using PyTorch.

## 2. Setup

Before you start, ensure that you have the following requirements:

* Python 3.6 or later.
* PyTorch 1.0 or later.
* A CUDA-compatible GPU is strongly recommended for training. 

## 3. Getting the data

In our example, we will use the CUB_200_2011 dataset. First, download the dataset from `https://www.kaggle.com/cub200/home` and extract it in the project directory with the following directory structure:

```bash
CUB_200_2011/
    images/
        [class directories]/
            [image files]
```

## 4. Training the Model

The script for training the model is provided in the same directory as this README file. The training process includes:

1. Loading and dividing the data into training, validation, and testing sets.
2. Creating and initializing a pre-trained ResNet-18 model from PyTorch.
3. Defining the loss function and optimizer.
4. Training the model on the training data.
5. Validating the model on the validation data.
6. Saving the best model based on validation performance.

To train the model, run the script using the Python interpreter:

    python resnet18_train.py

The training process can take several hours depending on the computing power of your system. By default, this script runs the training process for 20 epochs, but this can be adjusted by modifying the `EPOCHS` variable in the script.

## 5. Evaluating the Model

After training, the model's performance can be evaluated on the test data by loading the saved model and using it to make predictions. Run the evaluation script using the Python interpreter:

    python evaluate.py

The script will print the loss and accuracy of the model on the test data.

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

The default configuration for YOLOv3 in MMDetection is to train on the COCO dataset. When you are dealing with the PASCAL VOC dataset, certain parameters in the config file needs to be modified accordingly.

To make your task easier, I have prepared a customized version of config for YOLOv3 training on PASCAL VOC, which is included in this Github repository named 'yolov3_d53_8xb8-ms-608-273e_coco.py'. Please replace the default config with this file. 

Once the configuration file is updated, use the following command to train the model:

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
