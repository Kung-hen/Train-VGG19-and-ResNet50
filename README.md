# Structure of the training model

*Discription:*

***1.Training Model Structure of VGG19 and ResNet50 for Classification task.***

![image](GUI.png)


**1.Requirements and dependencies**
  * Python 3.8 (https://www.python.org/downloads/)
  * Opencv-contrib-python (3.4.2.17)
  * Matplotlib 3.7.3
  * UI framework: pyqt5 (5.15.10)
  * Pytorch 2.1.0
  * Torchvision 0.16.0
  * Torchsummary 1.5.1
  * Tensorboard  2.14.0
  * Pillow 10.1.0

**2.Usage:**

1. Downloads whole repository.
2. Run `python start.py` .
3. Input the image.
4. Run the whole code.

## 1.VGG19-BN

1.Objective :
  
  * Train a VGG19 with BN (Batch Normalization) model to classify 10 different classes images of CIFAR10.

2.Structure discription :

  * VGG19: A convolutional neural network that is 19 layers deep.
  * BN (Batch Normalization): used to make training of artificial neural networks faster and more stable.<img src="Figures/Bn.png" width="500">

## 2.CIFAR10

1.Dataset discription : 

  * A collection of 60,000 32x32 color images in 10 different classes that is commonly used to train machine learning and computer vision algorithms.
  * 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
  * Datasets :
      * Training dataset: 50000 images in total.
      * Validation dataset: 10000 images in total.
      * Testing dataset: 10 images in total. (Generating from validation dataset.)



