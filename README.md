# EC601-MiniProject2
Mini Project 2 from BU EC601.

## Requirements
TensorFlow, Numpy need to be installed first.

## Introduction
This mini project aims to better understand the basic workflow of deep learning.

* Step1: DownloadImage

> Using DownloadImage.py to download images from website by key words.

* Step2: Preprepocessing

> Prepocessing images downloaded from step1, including reshape and rename.

* Step3: Add Label to Image

> Making label list for every catogory.

* Step4: Build Neural Network

> Building a CNN to classify images. The CNN model only has two convolution layers.

* Step5: Validate & Test

## Analysis
When training the model, if we set the learning rate to 0.01, the loss function will keep at 0.69 after 100 epochs. However, if we set the learning rate to 0.001, the loss function will decrease to nearly 0 after 100 epochs.
