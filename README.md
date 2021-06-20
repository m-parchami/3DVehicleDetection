# 3DVehicleDetection

### Introduction
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

*\**: Other estimations are evaluated while and after training models (both 2D and 3D). However, since the translation vector is regressed through geometry connstraints, it can only be evaluated separately. 

**Note**: You may also view the notebook in Github website. If it fails to load, hit reload.

In the following, I provide a brief documentation on the implementation of different sections.

# 2D Vehicle Detection
## What we do here
Here we want to receive images and detect cars, vans and trucks in the scene in 2D (with a 2D bounding box). We also need to classify the detected patches from among the same classes (cars, vans, trucks). Later, when we estimate vehicle dimensions, these classes are needed.

## How we do it
Since 2D object detection is an already well-addressed problem, we try to keep things simple. That is, we try to use high level APIs and pretrained models to the best we can so that we can focus on our main purpose: the 3D part!

We can start from Tensorflow 1's model zoo that has a Faster RCNN checkpoint trained on the KITTI 2D object dataset. We can use this to detect cars. However, the available graph only classifies cars and pedestrain. Therefore, we cannot classify anything as van or truck with it. Hence, it is not exactly what we are looking for. Nevertheless, we can use it for cars and it can be sufficient for quick testing of the entire framework. Having this in mind, the notebook first uses this checkpoint for evaluating the whole framework.

Later, at the end of the notebook, we train our own model using Tensorflow 1 Object Detection API. Of course, we are free to use any object detection checkpoint. In this repository, we use Mobilenetv2+SSD and Faster RCNN (Resnet 101) models. The former is pre-trained on COCO and the latter is the same discussed in the last paragraph. It makes sense that the Faster RCNN model cann be trained easier, since its feature vectors are more relevant. In table 1, you can view my results on training these models.

Training with this API requires fixing some paths and some training configurations. The pipeline.config files provided in this repo, serve as the training configurations and the paths are compatible with the rest of the notebook, give or take. In all likelihood, you won't have to make too many changes to them. I appreciate any suggestions and improvement that you can think of for these configurations.

The rest of the workflow, such as setting up the environment, creating TFRecords, extracting frozen graph, and evaluating on test set is also provided in the notebook. The detection results of the Faster RCNN model is presented in figure 1.

# 3D Vehicle Detection

For implementing this section, I started with an unofficial implementation (check the references). My main focus was to make modifications for improving the accuracy of the model. I also tried my best to clean up the code. I also made sure that most of the functions and segments are documented. In the future, I try to add brief descriptions to this readme as well. Indeed, it is best to read the original paper and implement it yourself with as few hints as possible.

My final results are described in table 2 and 3. The final 3D boxes can be seen in figure 2.



Figure 1

![1_2D](https://github.com/m-parchami/3DVehicleDetection/samples/2D/2_2D.jpeg)
![2_2D](https://github.com/m-parchami/3DVehicleDetection/samples/2D/7_2D.jpeg)
![3_2D](https://github.com/m-parchami/3DVehicleDetection/samples/2D/9_2D.jpeg)
![4_2D](https://github.com/m-parchami/3DVehicleDetection/samples/2D/10_2D.jpeg)
![5_2D](https://github.com/m-parchami/3DVehicleDetection/samples/2D/12_2D.jpeg)
![6_2D](https://github.com/m-parchami/3DVehicleDetection/samples/2D/14_2D.jpeg)


Figure 2

![1_3D](https://github.com/m-parchami/3DVehicleDetection/samples/3D/2_3D.jpeg)
![2_3D](https://github.com/m-parchami/3DVehicleDetection/samples/3D/7_3D.jpeg)
![3_3D](https://github.com/m-parchami/3DVehicleDetection/samples/3D/9_3D.jpeg)
![4_3D](https://github.com/m-parchami/3DVehicleDetection/samples/3D/10_3D.jpeg)
![5_3D](https://github.com/m-parchami/3DVehicleDetection/samples/3D/12_3D.jpeg)
![6_3D](https://github.com/m-parchami/3DVehicleDetection/samples/3D/14_3D.jpeg)

# References
### Papers and Datasets
- [3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/abs/1612.00496)
- [KITTI Dataset (raw)](http://www.cvlibs.net/datasets/kitti/raw_data.php) and their [2D object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
### Codes and Tools
- [https://github.com/smallcorgi/3D-Deepbox/issues](github.com/smallcorgi/3D-Deepbox/issues)
- [https://github.com/cersar/3D_detection](github.com/cersar/3D_detection)
- [Tensorflow Object Detection API](github.com/tensorflow/models) + [Pretrained models(model zoo)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
