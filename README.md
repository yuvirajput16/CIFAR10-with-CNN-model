# CIFAR10-with-CNN-model

This repository contains a Jupyter notebook that demonstrates training and evaluating Artificial Neural Network (ANN) and Convolutional Neural Network (CNN) models on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset) 
- [Models](#models)
- [Dependencies](#dependencies)
- [Results](#results)
  
## Introduction
The purpose of this notebook is to compare the performance of ANN and CNN models on image classification tasks using the CIFAR-10 dataset. Convolutional Neural Networks are typically more effective for image data due to their ability to capture spatial hierarchies, while ANN models can serve as a baseline.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 50,000 training images and 10,000 test images. The dataset is commonly used for training machine learning and computer vision algorithms.

## Models
The notebook includes the following models:
- **Artificial Neural Network (ANN)**: A fully connected neural network used as a baseline.
- **Convolutional Neural Network (CNN)**: A deep learning model specifically designed for processing and classifying images.

## Dependencies
The following libraries are required to run the notebook:
- Python 3.x
- Jupyter Notebook
- NumPy
- TensorFlow
- Keras
- Matplotlib

## Results
After just 10 epochs
ANN Model - Accuracy: 0.4619
CNN Model - Accuracy: 0.6984

