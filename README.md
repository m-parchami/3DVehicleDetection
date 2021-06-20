# 3DVehicleDetection

### Introduction
The main purpose of this repository is to provide an unofficial implementation of the 3D vehicle detection framework discussed in "3D Bounding Box Estimation Using Deep Learning and Geometry". You can access the paper [here](https://arxiv.org/abs/1612.00496v1).

The paper focuses on 2D to 3D bounding box estimation. However, we also look into the 2D detection part, train our own model with Tensorflow Object Detection API, and test it together with the rest of the framework.

I tried my best to make this into a complete tutorial in each section so it can serve different questionns. I hope you find what you are looking for without getting too bored.

This repository  contains a fairly organized notebook called `AllinOne.ipynb`that you can run on Google Colab without worrying about anything (not even the dataset). This notebook goes through these tasks:
1. Getting and preparing dataset.
2. Defining the 2D to 3D model (the paper).
3. Using a pre-trained model from Tensorflow 1's model zoo for 2D detection
4. Visually evaluating 3D model's performance on boxes detected by part 3.
5. Training a custom 2D model from Tensorflow 1's object detection API.
6. Measuring translation vector accuracy*.

** * **: Other estimations are evaluated while and after training models (both 2D and 3D). However, since the translation vector is regressed through geometry connstraints, it should be evaluated separately. 

**Note**: You may also view the notebook in Github website. If it fails to load, hit reload.

In the following, I provide a brief documentation on the implementation of different sections.

# 2D Vehicle Detection
## What we do here
Here we want to receive images and detect cars, vans and trucks present in the scene in 2D (a 2D bounding box). We also need to classify the detected patches from among the same classes (cars, vans, trucks). Later, when we estimate vehicle dimensions, these classes will be used to apply mean.

## How we do it
Since 2D object detection is a well-addressed problem nowadays, we try to keep things as simple as possible. That is, we try to use high level APIs and pretrained models to the best we can so that we can focus on our main purpose: the 3D part!

We can start from Tensorflow 1's model zoo that has a Faster RCNN checkpoint trained on KITTI. We can use this to detect cars. However, this checkpoint only classifies betweenn car and pedestrain. Therefore, we cannot classify anything as van or truck with it. Hence, it is not exactly what we need. But we can still use it for cars. This can be sufficient for quick testing of the entire framework.

Later, at the end of the notebook, we train our own model using Tensorflow 1 Object Detection API. We can use any object detection checkpoint. Here, we use Mobilenetv2 SSD and Faster RCNN (Resnet 101) models. Training with this API, required fixing some paths and some training configurations. The pipeline.config files provided in this repo, serve as the training configurations and the paths are almost handled in the notebook.

The data for training must be connverted to TFRecords. In the notebook, this is done as well. Of course, we rely heavily on the API to do this for us.

# 3D Vehicle Detection

For implementing this section, I started with an unofficial implementaation. I tried to modify different sections to make improvements.

# References
### Papers and Datasets
- [3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/abs/1612.00496)
- [KITTI Dataset (raw)](http://www.cvlibs.net/datasets/kitti/raw_data.php) and their [2D object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
### Codes and Tools
- [https://github.com/smallcorgi/3D-Deepbox/issues](github.com/smallcorgi/3D-Deepbox/issues)
- [https://github.com/cersar/3D_detection](github.com/cersar/3D_detection)
- [Tensorflow Object Detection API](github.com/tensorflow/models) + [Pretrained models(model zoo)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
