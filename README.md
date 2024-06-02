# CV-Project2
Bird Species Recognition and Object Detection

Faster R-CNN Training on PASCAL VOC with MMDetection

This repository provides instructions to train a Faster R-CNN model on the PASCAL VOC dataset using the MMDetection framework.
Table of Contents

    Installation
        Prepare MMDetection Environment
        Prepare PASCAL VOC Dataset
    Training
    Visualizing Results

Installation
Prepare MMDetection Environment

    Clone the MMDetection repository:

    bash

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

Install the dependencies:

bash

pip install -r requirements/build.txt
pip install -v -e .

Install MMCV:

bash

    pip install mmcv-full

Prepare PASCAL VOC Dataset

    Download the PASCAL VOC 2007 and 2012 datasets from the official website and unzip them into a directory:

    bash

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar -C /path/to/voc/
tar -xvf VOCtrainval_11-May-2012.tar -C /path/to/voc/

Create symbolic links for VOC2007 and VOC2012:

bash

    ln -s /path/to/voc/VOCdevkit /mmdetection/data/VOCdevkit

Training

Train the Faster R-CNN model using MMDetection tools and configurations. Use the following command to start training:

bash

python tools/train.py configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py --work-dir path/to/your/work_dir

Replace path/to/your/work_dir with your desired working directory to save checkpoints and logs.
Visualizing Results

After training, you can visualize the results using the following script:

python

import mmcv
from mmdet.apis import init_detector, inference_detector
from mmengine.visualization import Visualizer

# Load an image
img = mmcv.imread('/kaggle/input/pascal-voc-2007-and-2012/VOCdevkit/VOC2007/JPEGImages/000001.jpg', channel_order='rgb')

# Initialize the model
checkpoint_file = '/kaggle/working/work_dir/epoch_4.pth'  # Replace with your checkpoint file
cfg = 'configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py'  # Replace with your config file
model = init_detector(cfg, checkpoint_file, device='cpu')

# Perform inference
new_result = inference_detector(model, img)

# Visualize the results
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

Make sure to replace the paths with your actual paths to the image, configuration file, and checkpoint file.
