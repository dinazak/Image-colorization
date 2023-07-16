# Image Colorization using CIFAR-10 Dataset

This repository contains code for converting black and white images to colored images using the CIFAR-10 dataset. The project focuses on training a model to learn the mapping from black and white images to their corresponding colored versions.

## Problem Description

The task is to convert black and white images to colored images using the CIFAR-10 dataset. The goal is to train a model that can accurately colorize black and white images by learning from the colored images in the CIFAR-10 dataset.

## Dataset

The CIFAR-10 dataset is used for this project. The dataset consists of 60,000 colored images across 10 classes, with 6,000 images per class. It is divided into 50,000 training images and 10,000 testing images. Each image in the dataset has a resolution of 32x32 pixels. We will use the colored images from the CIFAR-10 dataset to train the model and evaluate its performance.

## Image Colorization

The model is trained to learn the mapping from black and white images to their corresponding colored versions. The training data consists of the black and white images from the CIFAR-10 dataset and their corresponding colored images. The model is trained to minimize the difference between the predicted colored images and the ground truth colored images.

## Implementation

The code is implemented using Python and deep learning libraries such as TensorFlow and Keras. The model is built using convolutional neural network (CNN) layers to capture spatial features and learn the colorization process. The model is trained on the CIFAR-10 dataset using appropriate loss functions and optimization techniques.


## Evaluation and Results

Evaluate the performance of the model on a separate test set from the CIFAR-10 dataset. Measure metrics such as color accuracy, structural similarity index (SSIM), and visual quality of the colorized images. Provide visual examples of the colorized images along with their corresponding ground truth colored images. Discuss any challenges faced during training, potential limitations, and suggestions for improvement.

