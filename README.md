# 3D Vehicle Detection

## Table of Contents :deciduous_tree:
1. [Introduction](#Introduction)
2. [2D Vehicle Detection](./README.md#2d-vehicle-detection-blue_square)
3. [3D Vehicle Detection](./README.md3d-vehicle-detection-blue_square)
4. [Tables](./README.md#Tables)
5. [Figures](./README.md#Figures)
6. [Demo](./README.md#Demo)
7. [References](./README.md#References)

## Introduction  :red_car:
The main purpose of this repository is to provide an unofficial implementation of the 3D vehicle detection framework introduced in the "[3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/abs/1612.00496v1)" paper.

The paper mainly focuses on the 3D bounding box estimation from given 2D detections. However, in this repository, we also look into the 2D detection part, train our own model with Tensorflow Object Detection API, and test it together with the rest of the framework.

I tried my best to make this into a complete tutorial for each section so it can answer different questions. I hope you find what you are looking for without getting too bored. Do not hesitate on creating issues or contacting me for any question/suggestion on the code.

This repository  contains a fairly organized notebook called `AllinOne.ipynb` that you can run on Google Colab without worrying too much about anything. The notebook goes through these tasks:
1. Getting and preparing dataset.
2. Implementing the 2D to 3D model (from the paper) together with its pre-processing and post-processing.
3. Using a pre-trained model from Tensorflow 1's model zoo for 2D detection.
4. Visually evaluating 3D model's performance on boxes detected by part 3.
5. Training a custom 2D model from Tensorflow 1's object detection API.
6. Measuring translation vector accuracy*.

*\** Other estimations are evaluated while and after training models (both 2D and 3D). However, since the translation vector is regressed through geometry connstraints, it can only be evaluated separately. 

**Note**: You may also view the notebook in Github website. If it fails to load, hit reload a couple of times.

In the following, I provide a brief documentation on the implementation of different sections.

## 2D Vehicle Detection :blue_square:
### What we do here
Here we want to receive images and detect cars, vans and trucks in the scene in 2D (with a 2D bounding box). We also need to classify the detected patches from among the same classes (cars, vans, trucks). Later, when we estimate vehicle dimensions, these classes are needed.

### How we do it
Since 2D object detection is an already well-addressed problem, we try to keep things simple. That is, we try to use high level APIs and pretrained models to the best we can so that we can focus on our main purpose: the 3D part!

We can start from Tensorflow 1's model zoo that has a Faster RCNN checkpoint trained on the KITTI 2D object dataset. We can use this to detect cars. However, the available graph only classifies cars and pedestrain. Therefore, we cannot classify anything as van or truck with it. Hence, it is not exactly what we are looking for. Nevertheless, we can use it for cars and it can be sufficient for quick testing of the entire framework. Having this in mind, the notebook first uses this checkpoint for evaluating the whole framework.

Later, at the end of the notebook, we train our own model using Tensorflow 1 Object Detection API. Of course, we are free to use any object detection checkpoint. In this repository, we use Mobilenetv2+SSD and Faster RCNN (Resnet 101) models. The former is pre-trained on COCO and the latter is the same discussed in the last paragraph. It makes sense that the Faster RCNN model can be trained easier, since its feature vectors are more relevant. In [table 1](./README.md#Tables), you can view my results on training these models.

Training with this API requires fixing some paths and some training configurations. The pipeline.config files provided in this repo, serve as the training configurations and the paths are compatible with the rest of the notebook, give or take. In all likelihood, you won't have to make too many changes to them. I appreciate any suggestions and improvement that you can think of for these configurations.

The rest of the workflow, such as setting up the environment, creating TFRecords, extracting frozen graph, and evaluating on test set is also provided in the notebook. The detection results of the Faster RCNN model is presented in [figure 1](./README.md#Figures) on the left.

## 3D Vehicle Detection :package:

For implementing this section, I started with an [unofficial implementation](./README.md#References). My main focus was to make modifications for improving the accuracy of the model. I also tried my best to clean up the code and add documentation so that it is more useful. In the future, I try to add brief descriptions to this readme as well. Indeed, it is best to read the original paper and implement it yourself with as few hints as possible.

My final results are described in table 2 and 3. The final 3D boxes can be seen in [figure 1](./README.md#Figures) on the right.

## Tables
### Table 1
Evaluation results of training two object detectors.
+ Train size: 6981 images.
+ Test size: 500 images.
+ IoU: 0.5

| Model | Previous Dataset | Input Dimensions | Epochs | val AP Car | val AP Van | val AP Truck | mAP |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Faster RCNN (Resnet 101) | KITTI | 300\*993\*3 | 14 | 0.8328 | 0.6252 | 0.6859 | 0.7146 |
| Movilenetv2 + SSD | COCO | 300\*300\*3 | 63 | 0.6353 | 0.3580 | 0.5497 | 0.5143 |

**Disclaimer**: These results do not necessarily relate to model's overall performance and capabilities. One may achieve better results with either by modifying parameters or longer training. This is not a reliable general comparison of the two models.

### Table 2
Evaluation results of training the 3D network with two different feature extractors.
+ Train size: 19308 patches extracted directly from the dataset.
+ Test size: 3408 patches extracted directly from the dataset.
+ Confidence error metric: MSE
+ Dimension error metric: MSE
+ Angle error metric: 1 - cos(|pred| - true) (ideal would be 1 - cos(0) = 0)

| Feature extractor | Previous Dataset | Input Dimensions | Epochs | # Parameters | Confidence Error | Angle error | Dimension Error |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| VGG-16 | Imagenet | 224\*224\*3 | 15 | 40,411,989 | 0.0076 | 0.0078 | 0.1100 |
| Movilenetv2 | Imagenet | 224\*224\*3 | 25 | 66,490,453 | 0.0093 | 0.0073 | 0.1084 |

**Disclaimer**: These results do not necessarily relate to model's overall performance and capabilities. One may achieve better results with either by modifying parameters or longer training. This is not a reliable comparison of the two models.

### Table 3
Translation vector estimation accuracy.

+ Error metric: L2_norm(true - pred)
+ Normalized Error metric: L2_norm( (true - pred) / |true| )

| Max Truncation | Max Occlusion | Min 2D Bbox Width | Min 2D Bbox Height | Final Sample Count | Average Error | Average Normalized Error * |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 0.15 | 1 | 60 | 60 | 967 | 1.8176 | 0.1151 |
| 0.40 | 1 | 40 | 40 | 1788 | 2.4165 | 0.1168 |

**Disclaimer**: These results do not necessarily relate to the performance of the paper's method. These are just the results that I achieved. For accurate benchmarks, refer to the original paper.

## Figures :eyes:

### Figure 1
+ Left: 2D detections from the Faster RCNN (Resnet 101) model from tf1 model zoo that I fine-tuned to detect cars, vans, and trucks.
+ Right: 3D detections on the same boxes from left using Mobilenetv2 backbone as for the feature extractor.

<img src="/samples/2D/2_2D.jpeg" width="400"> <img src="/samples/3D/2_3D.jpeg" width="400">
<img src="/samples/2D/7_2D.jpeg" width="400"> <img src="/samples/3D/7_3D.jpeg" width="400">
<img src="/samples/2D/9_2D.jpeg" width="400"> <img src="/samples/3D/9_3D.jpeg" width="400">
<img src="/samples/2D/12_2D.jpeg" width="400"> <img src="/samples/3D/12_3D.jpeg" width="400">
<img src="/samples/2D/14_2D.jpeg" width="400"> <img src="/samples/3D/14_3D.jpeg" width="400">

## Demo :clapper:

A [short video demo](https://mparchami.com/3DVehicleDetection_Demo.mp4) on my webpage.

# References :clap:
### Papers and Datasets
- [3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/abs/1612.00496)
- [KITTI Dataset (raw)](http://www.cvlibs.net/datasets/kitti/raw_data.php) and their [2D object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
### Codes and Tools
- [https://github.com/smallcorgi/3D-Deepbox/issues](github.com/smallcorgi/3D-Deepbox/issues)
- [https://github.com/cersar/3D_detection](github.com/cersar/3D_detection)
- [Tensorflow Object Detection API](github.com/tensorflow/models) + [Pretrained models(model zoo)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
