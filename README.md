# Neura - Brain tumor detection model ![Project Logo/Image](https://img.icons8.com/ios-filled/40/000000/brain.png)  
          
          
          
          
<img width="1000" height="370" alt="image" src="https://github.com/user-attachments/assets/94ea2a49-06b5-49b9-b7c3-32398a128df0" />




## ➲ Overview  

Brain tumors are among the most critical health issues worldwide, requiring early and accurate diagnosis to improve patient outcomes. This project focuses on **automatic brain tumor detection from MRI scans** using a deep learning approach.  

The model is based on a **Convolutional Neural Network (CNN)** architecture (Xception-based), trained to classify MRI images into **tumor** or **non-tumor** categories. The goal is to provide a reliable, fast, and scalable solution to assist radiologists and healthcare professionals.  

Our trained model achieves:  
- **~98% training accuracy**  
- **~96% validation accuracy**  
- **~95% test accuracy**  
- **ROC-AUC: 0.97–0.99 across classes**  

## ➲ Prerequisites  

Install the following packages before running the project:  

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

## ➲ The Dataset

We used publicly available MRI brain tumor datasets containing labeled images for classification.

Classes: Tumor, Non-Tumor

Data Split: 70% training, 15% validation, 15% testing

Images were preprocessed (resizing, normalization, augmentation) to improve model performance.

Sample MRI images:

## ➲ Dataset Description

### Tumor: MRI scans indicating the presence of a brain tumor.

### Non-Tumor: MRI scans of healthy brains without visible tumors.

The preprocessing pipeline includes:

Resizing images to uniform dimensions

Normalizing pixel values

Data augmentation (rotation, flip, zoom)

## ➲ Model Architecture

We implemented a CNN model with transfer learning (Xception).

Key layers include:

Convolutional layers with ReLU activation

MaxPooling for feature extraction

Dropout for regularization

Fully connected dense layers

Softmax activation for classification

Libraries used
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

## ➲ Model Performance
<img width="980" height="281" alt="image" src="https://github.com/user-attachments/assets/901545b7-b964-492a-a27b-8e6b05d07a3c" />


Confusion Matrix & Classification Report confirm strong performance across both classes.

## ➲ Results Visualization

Training & Validation Accuracy/Loss curves

ROC curves for classification

Example predictions with MRI scan input vs. predicted label
