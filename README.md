---
title: ironhack-project-1
app_file: app.py
sdk: gradio
sdk_version: 5.30.0
---
# Ironhack Project 1

# CNN architecture
For this project, we selected the CIFAR-10 dataset as it was already familiar to us from previous model implementation.

Note: Initially we tried to use the Animals 10 because it is a much lighter dataset, consisting of 50% fewer images than CIFAR and we believed that this would ultimately help us save time processing the model epochs later, but we encountered so many bugs just reading the data the that we decided to switch to CIFAR-10.

# Preprocessing Steps
We used multiple preprocessing steps.

After loading the data and plotting a matrix of 10 x 10 images, we preprocessed the data with:
- One-hot-encoding in order to transform categorical data into numerical format
- Normalization in order to scale the data between 0 - 1 so that any feature does not disproportionately influence the algorithm
- Data Augmentation in order to increase the diversity and size of a training dataset by creating modified copies of existing data. This is to help machine learning models generalize better and reduces overfitting.

# Training Process Details
- All models compile using Adam optimizer, categorical loss entropy, and accuracy metrics. Each model modifies the learning rate to see how it effects the accuracy of the model.
- We used ReduceLROnPlateau to reduce the learning rate when validation loss plataeus
- We used early stopping to prevent overfitting by halting training when the model's performance on a validation set starts to decline.


# Architecture:
- VGG16 
- Resnet50V2

# Optimizers
 - RMSprop
 - Adam
 - SGD

 # Weights
 - Imagenet

# Models Descriptions: 
- Main Model implements and original simple model and then builds upon it adding multiple convolutional blocks, and then gradually increases the epochs
- vgg16_model uses a very low learning rate of .001 and 100 epochs to train the model
- VGG16 Model with Batch Normalization implements batch normalization which scales the minin batch as it is transformed between the layers
- VGG16 Model with Several Layers implements extra layers to the VGG16 type of model so that the model can build upon these features to create more abstract and complex representations.
    - It freezes the first 15 layers to preserve the learned features from the original training.
- VGG16 Simple Model implements basic layering with two relu layers and max-pooling and a softmax output layer.
- RESNET50V2 models first resizes the images to 224, 224 as Resnet50V2 cannot be optimized on the small image size of 32, 32 (standard for CIFAR-10)
 - uses special perams in defining the model
 -Input(shape=input_shape),
    layers.GaussianNoise(0.1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.05),
    layers.Dense(num_classes, activation='softmax')
- uses transfer learning technique
