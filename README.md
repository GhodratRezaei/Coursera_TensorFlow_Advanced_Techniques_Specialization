# TensorFlow: Advanced Techniques Specialization

Used the Functional API to built custom layers and non-sequential model types in TensorFlow, performed object detection, Image segmentation, and Interpretation of convolutions. Used generative Deep Learning(Unsupervised Learning) including Auto Encoding, VAEs, and GANs to create new content.

Organized by: [DeepLearning.AI](https://www.deeplearning.ai/), [Coursera](https://coursera.org/)

# Table of Contents

1. [Introduction and Instructions](#my-first-title)
2. [Overview of Learning from the Specialization](#my-second-title)
3. [Labs and Projects](#my-third-title)
4. [Results and Conclusion](#my-fourth-title)
## Introduction and Instructions
All the Dataset, refrences, links and materials for Projects in this specialization are taken from the [TensorFlow: Advanced Techniques Specialization](https://www.coursera.org/specializations/tensorflow-advanced-techniques)). 
Clone this repisitory to use it. 

## Overview of Learning from the Specialization
This Specialization is devided in four main parts:
* [Custom Models, Layers, and Loss Functions with TensorFlow](https://www.coursera.org/learn/custom-models-layers-loss-functions-with-tensorflow?specialization=tensorflow-advanced-techniques)
  * Build Network using Functional and Sequential API with main focus on Functional API including Siamese Network 
  * Build Custom Layers, Custom Loss Function (including Contrasive Loss Function in Siamese Network)
  * Building Network with Standard commom layers and using Lamda Layers for complex and arbitrary netwotrks 
  * Build custom classes instead of many Functional or Sequential APIs
  * Build models that can be inherited from the TensorFlow Model class, and build a residual network (ResNet) through defining a custom model class
  * Build Custom Callbacks using Class and also manually implement following callbacks : Tensorboard Callbacks, Early Stopping, Model Checkpoint, CSV Logger 
* [Custom and Distributed Training with TensorFlow](https://www.coursera.org/learn/custom-distributed-training-with-tensorflow?specialization=tensorflow-advanced-techniques)
  * Eager and Grapgh Mode in TensorFlow
  * Build Custom training Loops using Gradient Tape for more visibility and flexibility
  * Training using Distributed Training Methods and Training using multiple GPU or TPU cores 
* [Advanced Computer Vision with TensorFlow](https://www.coursera.org/learn/advanced-computer-vision-with-tensorflow?specialization=tensorflow-advanced-techniques)
  * Building Object Detection Networks (Classification and Localization(Regression)) including: R_CNN, Faster_RCNN, YOlO, SSD
  * Building Segmentation Networks (Semantic Segmentation and Instance Segmentation) incuding:
  Fully CNN, U-Net, DeepLab
  * Using Transfer Learning and Fine Tunning for Making Network incase of Data Scarcity and Time-consumption of training. 
  * Implemented image segmentation using variations of the fully convolutional network (FCN) including U-Net and Mask-RCNN to identify and detect numbers, pets, zombies
  * Building Saliency Maps and Class activation Maps(CAM and gradCAM) to identify which parts of the image are used by the classification model to take a decision.    
* [Generative Deep Learning with TensorFlow](https://www.coursera.org/learn/generative-deep-learning-with-tensorflow?specialization=tensorflow-advanced-techniques)
  * Build  a Generative Model(Neural Style Transfer and Fast Neural Style Transfer) to make a new image using Style and Content Image. 
  * Build simple and Stacked AutoEncoders on the familiar MNIST dataset, and more complex deep and convolutional architectures on the Fashion MNIST dataset to get the Latent representation of the image.
  * Denoising the Noisy image using CNN Auto Encoder 
  * Building VAriational AutoEncoder to Generate a new data not just reconstructed one.
  * Building GANs(Generative Adversial Networks) implementing the concept of two training phases(Discriminator and Generative)



## Labs and Projects
* [Course 1: Custom Models, Layers, and Loss Functions with TensorFlow](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%201%20-%20Custom%20Models%2C%20Layers%2C%20and%20Loss%20Functions%20with%20TensorFlow)
   * [Week 1: Multiple Output Models using the Keras Functional API](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%201%20-%20Custom%20Models%2C%20Layers%2C%20and%20Loss%20Functions%20with%20TensorFlow/Week%201)
   * [Week 2: Creating a Custom Loss Function](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%201%20-%20Custom%20Models%2C%20Layers%2C%20and%20Loss%20Functions%20with%20TensorFlow/Week%202)
   * [Week 3: Implement a Quadratic Layer](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%201%20-%20Custom%20Models%2C%20Layers%2C%20and%20Loss%20Functions%20with%20TensorFlow/Week%203)
   * [Week 4: Create a VGG network](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%201%20-%20Custom%20Models%2C%20Layers%2C%20and%20Loss%20Functions%20with%20TensorFlow/Week%204)
   * [Week 5: Introduction to Keras callbacks](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%201%20-%20Custom%20Models%2C%20Layers%2C%20and%20Loss%20Functions%20with%20TensorFlow/Week%205)

* [Course 2: Custom and Distributed Training with TensorFlow](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%202%20-%20Custom%20and%20Distributed%20Training%20with%20TensorFlow)
   * [Week 1: Basic Tensor operations and GradientTape](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%202%20-%20Custom%20and%20Distributed%20Training%20with%20TensorFlow/Week%201)
   * [Week 2: Breast Cancer Prediction](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%202%20-%20Custom%20and%20Distributed%20Training%20with%20TensorFlow/week%202)
   * [Week 3: Horse or Human? In-graph training loop](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%202%20-%20Custom%20and%20Distributed%20Training%20with%20TensorFlow/Week%203)
   * [Week 4: Custom training with tf.distribute.Strategy](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%202%20-%20Custom%20and%20Distributed%20Training%20with%20TensorFlow/week%204)
* [Course 3: Advanced Computer Vision with TensorFlow](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%203%20-%20Advanced%20Computer%20Vision%20with%20TensorFlow)
   * [Week 1: Predicting Bounding Boxes](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%203%20-%20Advanced%20Computer%20Vision%20with%20TensorFlow/Week%201)
   * [Week 2: Zombie Detection](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%203%20-%20Advanced%20Computer%20Vision%20with%20TensorFlow/Week2)
   * [Week 3: Image Segmentation of Handwritten Digits](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%203%20-%20Advanced%20Computer%20Vision%20with%20TensorFlow/Week3)
   * [Week 4: Saliency Maps](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%203%20-%20Advanced%20Computer%20Vision%20with%20TensorFlow/Week4)
* [Course 4: Generative Deep Learning with TensorFlow](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%204%20-%20Generative%20Deep%20Learning%20with%20TensorFlow)
   * [Week 1: Neural Style Transfer](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%204%20-%20Generative%20Deep%20Learning%20with%20TensorFlow/Week1)
   * [Week 2: CIFAR-10 Autoencoder](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%204%20-%20Generative%20Deep%20Learning%20with%20TensorFlow/Week2)
   * [Week 3: Variational Autoencoders on Anime Faces](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%204%20-%20Generative%20Deep%20Learning%20with%20TensorFlow/Week3)
   * [Week 4: GANs with Hands](https://github.com/GhodratRezaei/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/Course%204%20-%20Generative%20Deep%20Learning%20with%20TensorFlow/Week4)



## Results and Conclusion
* 1. Object Detection task using MNIST digits dataset 

![image1](https://user-images.githubusercontent.com/75788150/173193074-bbf05133-c7d3-418b-b714-329baffb4cf4.jpg)

* 2. Object detection using RetinaNet

![image2](https://user-images.githubusercontent.com/75788150/173193189-0d2ee0c3-aa10-4c03-8529-9fa419ec8e9e.jpg)


* 3. Object detection using Mask RCNN

![image3](https://user-images.githubusercontent.com/75788150/173193269-9795d199-8ada-4cd7-9356-a5c4813d5f27.jpg)


* 4. Image segmentation on MNIST digits dataset

![image4](https://user-images.githubusercontent.com/75788150/173193305-fc2a3a1e-ad0d-4b00-a0be-81a7d19c67fc.jpg)

* 5. Image segmentation on pets dataset

![image5](https://user-images.githubusercontent.com/75788150/173193322-3503f63a-9955-4dad-9b16-07ceb50f8049.jpg)


* 6. Scene segmentation

![image6](https://user-images.githubusercontent.com/75788150/173193355-097ee340-2446-4bd9-ac47-5fb07cbe0c51.jpg)


* 7. Saliency Map

![image7](https://user-images.githubusercontent.com/75788150/173193392-68f56b55-2547-4049-b1c0-99fc2fcacaae.jpg)

