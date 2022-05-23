
# Skin Disease Detection 

## Aim And Objectives

## Aim

#### To create a skin disease detection which will detect the type of skin diseases and identify an effective barrier mediating protection against environmental danger and foreign substances as well as to identify the clinical features of the skin diseases.

## Objectives

• The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and
  start detecting using the camera module on the device.

• Using appropriate datasets for recognizing and interpreting data using machine learning.

• To identify skin lesion based on in the input skin images texture analysis based on thresholding and neural network to detect and 
  diagnose skin diseases.

## Abstract

• Skin diseases are a common problem among young adults. There is paucity of data about it among medical students. This study aimed 
  to find out the pattern of skin disorders and to describe their association with various socio-demographic factors among medical students.

• We have completed this project on jetson nano which is a very small computational device.

• A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to 
 identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

• One such technique is to use YOLOv5 with Roboflow model , which generates a small size trained model and makes ML integration
  easier.

• Data on skin morbidities suffered over past 1 year and its associated factors were collected using a self-administered 
  questionnaire.

• This cross-sectional study for skin disorders particularly the cosmetice problems are common in people. Gender and place of origin 
  were found to significantly influence the development of certain morbidities. 



## Introduction

• This project is based on a skin diseases detection model with modifications. We are going to implement this project with Machine 
  Learning and this project can be even run on jetson nano which we have done.

• This project can also be used to gather information about skin diseases, i.e., clear or damage.

• Skin disease can be classified into Acne , Carbuncle, Ichthyosis, Melanoma, Psoriasis and Ringworm based on the image annotation we
  give in roboflow.

• Skin disease detection in our model sometime become difficult because of various kind of skin legions like oily skin, dry skin,
  spots on the skin , birth mark on the skin and skin irritations by sun however training or model with the images of is this skin disease makes the model more accurate.

• Neural networks and machine learning have been used for these tasks and have obtained good results.

• Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for skin
  disease detection as well.

## Literature Review

• This project is based on a skin disease detection model with modifications. We are going to implement this project with Machine 
  Learning and this project can be even run on jetson nano which we have done.

• This project can also be used to gather information about skin diseases, i.e., clear or damage.

• Skin disease can be classified into Acne , Carbuncle, Ichthyosis, Melanoma, Psoriasis and Ringworm based on the image annotation we
  give in roboflow.

• Skin disease detection in our model sometime become difficult because of various kind of skin legions like oily skin, dry skin, 
  spots on the skin, birth mark on the skin and skin irritations by sun however training or model with the images of is this skin disease makes the model more accurate.

• Neural networks and machine learning have been used for these tasks and have obtained good results.

## Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers everywhere.
    
• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.
    
• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

## Jetson Nano 2GB

![Jetson Nano](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Proposed System

1] Study basics of machine learning and image recognition.

2]Start with implementation
    
    • Front-end development
    • Back-end development

3] Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether a person is wearing a Helmet or not.

4] Use datasets to interpret the skin disease and suggest whether the skin are clean or damage.

## Methodology

   The skin disease detection system is a program that focuses on implementing real time skin disease detection. It is a prototype of a new product that comprises of the main module: skin disease detection and then showing on viewfinder whether clean or damage. skin disease Detection Module.


#### Skin Disease Detection Module

### This Module is divided into two parts:


#### 1] Skin Disease Detection 

• Ability to detect the disease on skin in any input image or frame. The output is the bounding box coordinates on the detected   
  skin disease. 

• This Datasets identifies skin disease in a Bitmap graphic object and returns the bounding box image with annotation of disease  
  present in a given image.

#### 2] Clear Skin Detection

• Classification of the skin disease based on whether it is clean or damage. Hence YOLOv5 which is a model library from roboflow for 
  image classification and vision was used.

• There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in 
  PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.

•YOLOv5 was used to train and test our model for various classes like clean, damage. We trained it for 149 epochs and achieved an 
 accuracy of approximately 65%.



## Installation

#### Initial Configuration

```bash
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*

```
#### Create Swap 
```bash
udo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
# make entry in fstab file
/swapfile1  swap    swap    defaults    0 0
```
#### Cuda env in bashrc
```bash
vim ~/.bashrc

# add this lines
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

```
#### Update & Upgrade
```bash
sudo apt-get update
sudo apt-get upgrade
```
#### Install some required Packages
```bash
sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo apt-get install libopenblas-base libopenmpi-dev
```
#### Install Torch
```bash
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True" 
sudo python3 -c "import torch; print(torch.cuda.is_available())"
```
#### Install Torchvision
```bash
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
```
#### Clone Yolov5 
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
```
#### Download weights and Test Yolov5 Installation on USB webcam
```bash
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt  --source 0
```
##  Skin Disease Dataset Training

### We used Google Colab And Roboflow

#### train your model on colab and download the weights and past them into yolov5 folder


## Running Skin Disease Detection Model
source '0' for webcam

```bash
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0
```
## Demo


https://user-images.githubusercontent.com/89011801/169752721-0e5cd061-b016-4ff7-a145-081e4d8df0b0.mp4




## Advantages

• It will be useful to users who are very busy because of work or are because of prior schedules. 
• Just place the viewfinder showing the skin on screen and it will detect it.
• It might be easy to find the actual disease of skin.
• It can be helpful to take a proper treatment to clean the pigmentation.

## Application

• Detects marks and discoloration of skin in a given image frame or viewfinder using a camera module.

• Can be used to detect disease of skin when used with proper hardware like machines which can clean.

• Can be used as a reference for other ai models based on skin disease detection.

## Future Scope

• As we know technology is marching towards automation, so this project is one of the step towards automation.
    
• Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.
    
• Healthy skin, pimples less skin, remove unwanted hair on face, clear and attractive skin is considering a good used of our model.

## Conclusion
    
• In this project our model is trying to find the unwanted legions on face, marks, pigmentation, discoloration blister on over
  all body called as a unclear skin.

• This model helps to solve the problems which is commonly faced by people in daily routing.

• It may easy  work on the problem solving for the people in daily life style to control the morbidity level. 


## Reference

1] Roboflow:- https://roboflow.com/

2] Google images

## Articles :-

1] https://www.healthline.com/health/skin-disorders

2]  https://www.niams.nih.gov/health-topics/skin-diseases
