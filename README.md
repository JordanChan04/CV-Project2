# CV-Project2
Bird Species Recognition and Object Detection

Faster R-CNN Training with Pascal VOC Dataset using MMDetection

This guide will walk you through the steps to train a Faster R-CNN model on the Pascal VOC dataset using MMDetection.
Table of Contents

    Prepare MMDetection Environment
    Prepare Pascal VOC Dataset
    Train the Model
    Show the Results

Prepare MMDetection Environment

    Clone the MMDetection repository:

    bash

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

Install dependencies:

bash

pip install -r requirements/build.txt
pip install -v -e .

Install MMCV:

bash

pip install mmcv-full

Verify the installation:

bash

    python tools/misc/print_config.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py

Prepare Pascal VOC Dataset

    Download the Pascal VOC dataset:

    You can download the dataset from here.

    Extract the dataset:

    bash

tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtrainval_11-May-2012.tar

Organize the dataset directory structure as follows:

kotlin

    mmdetection
    ├── data
    │   └── VOCdevkit
    │       ├── VOC2007
    │       └── VOC2012

Train the Model

    Use the following command to start training the Faster R-CNN model:

    bash

    python tools/train.py configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py --work-dir path/to/your/work_dir

        Replace path/to/your/work_dir with your desired working directory where the training results will be saved.

Show the Results

    Use the following script to visualize the results:

    python

    import mmcv
    from mmdet.apis import init_detector, inference_detector
    from mmengine.visualization import Visualizer

    # Load an image
    img = mmcv.imread('/kaggle/input/pascal-voc-2007-and-2012/VOCdevkit/VOC2007/JPEGImages/000001.jpg', channel_order='rgb')

    # Path to your trained model checkpoint
    checkpoint_file = '/kaggle/working/work_dir/epoch_4.pth'

    # Load the configuration file
    cfg = 'configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py'

    # Initialize the detector
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

        Replace checkpoint_file with the path to your model's checkpoint file.
        Ensure the img variable points to the correct image file path.

By following these steps, you can successfully train a Faster R-CNN model on the Pascal VOC dataset and visualize the results using MMDetection. For more detailed information, refer to the MMDetection documentation.
