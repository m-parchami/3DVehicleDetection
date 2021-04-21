# 3DVehicleDetection

### Introduction
The main purpose of this repository is to provide an unofficial implementation of the 3D vehicle detection framework discussed in "3D Bounding Box Estimation Using Deep Learning and Geometry". You can access the paper [here](url).

We also look into the 2D detection part, train our own model with Tensorflow Object Detection API, and test it together with the rest of the framework.

I tried my best to make this into a complete tutorial, rather than a sole implementation. I hope you find what you are looking for without getting too bored.

This repository also contains a fairly organized notebook that you can run on Google Colab without worrying about anything (not even the dataset).

### Table of Contents

Each section provides a brief documentation on file structure, codes, and main functions.

1. [2D Vehicle Detection](url)
2. [3D Vehicle Detection](url)
3. [Finalizing Parameters](url)
4. [Results](url)
5. [Models](url) 
# 2D Vehicle Detection
## What we do here
This is the first section of the framework proposed in the official paper. Here we want to receive images and detection vehicles in 2D. We also need to classify the patches we detect, because they will be needed in [Section 3](url).

## How we do it
Since 2D object detection is a well-addressed problem nowadays, we try to keep things as simple as possible. That is, we try to use high level APIs and pretrained models to the best we can, so that we can focus on our main purpose: the 3D detection!

In this repo we use Tensorflow Object Detection API. To be short, it provides an object detection training and evaluation framework. It also makes transfer learning much easier. Using this API, we start from a pre-trained model and fine-tune it for our task, almost without a single line of code.
### Step 1: Choose the model
Since we are using the KITTI dataset, we prefer models pre-trained on KITTI. Looking at the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md), we can only see Resnet101_FRCNN. Though it can give us the detections we need, it might not necessarily know the same classes we are looking for. In fact, according to my experience, it was trained to classify cars and pedestrains. However, here we are looking for cars, trucks, and vans, so we have to go through fine-tuning anyway. 

Now that we have have to get our hands dirty, we can also try other models that are not trained on KITTI. We choose Mobilenet_v2+SSD which is trained on COCO. As long as you are choosing a detection model, you are free to choose any other model as well.

### Step 2: Setup
If you are using Jupyter notebooks (or Google Colab), you can refer to the notebook uploaded on this repository and run the section which prepares the Tensorflow Object Detection API. Otherwise, you can refer to [this tutorial](url).

### Step 3: Gather and Prepare Data
Here we can use the KITTI's 2D object. It containes 7481 full images. Of course, you can also use the raw data but they provide you with too much unnecessary information. Using the 2D Object dataset, you get a simpler dataset that is much easier to work with.

After you have downloaded the data, you need TFRecords, which are used by the API. For converting the dataset, you can simply use the [script provided by the API itself](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_kitti_tf_record.py). Here is the command I use:

```bash

the command

```
### Step 4: Prepare for Training
Now that we have the data and our model, we just have to prepare the training pipeline. That is, we have to specify to the API to train the specific model we have in mind with our dataset (TFRecords). For this we configure the `pipeline.config` file provided with the model. There are several parameters that we can configure there but there are two things to consider here:
1- For quick experiments it may be wise to trust the parameters the way they are.
2- We are using a checkpoint, and because of this we cannot change every parameter! For instance if you change the feature extractor architecture, the API will not be able to work with this configuration.

But there are things that we can and should configre:



# References
### Papers and Datasets
- [3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/abs/1612.00496)
- [KITTI Dataset (raw)](http://www.cvlibs.net/datasets/kitti/raw_data.php) and their [2D object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
### Codes and Tools
- [https://github.com/smallcorgi/3D-Deepbox/issues](github.com/smallcorgi/3D-Deepbox/issues)
- [https://github.com/cersar/3D_detection](github.com/cersar/3D_detection)
- [Tensorflow Object Detection API](github.com/tensorflow/models) + [Pretrained models(model zoo)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
