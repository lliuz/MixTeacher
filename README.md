# MixTeacher: Mining Promising Labels with Mixed Scale Teacher for Semi-supervised Object Detection

This is the PyTorch implementation of our paper: 

[[Paper](https://arxiv.org/abs/2303.09061)] **MixTeacher: Mining Promising Labels with Mixed Scale Teacher for Semi-supervised Object Detection** 

Liang Liu, Boshen Zhang, Jiangning Zhang, Wuhao Zhang, Zhenye Gan, Guanzhong Tian, Wenbing Zhu, Yabiao Wang, Chengjie Wang

The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR), 2023

\* The code will be available after the completion of the Tencent open-source review process.

![pipeline](pipeline.png)

## Usage

1. Prepare data and environment as SoftTeacher
2. Training and Testing as follows:
```shell
cd ./thirdparty/mmdetection && pip install -r requirements/build.txt && pip install -v -e .
cd ...
bash tools/dist_train.sh configs/soft_teacher_msi/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py 8 --cfg-options fold=1 percent=2
```



## Acknowledgments 

This work is highly dependent on a series of excellent preliminary work. We would like to express our utmost thank to SoftTeacher, PseCo, mmdetection, and other projects not mentioned. 

We fully comply with the license of these projects, and there is no additional license for this project.



