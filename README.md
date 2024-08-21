# CT Image-Based Kidney Disease Classification using Convolutional Neural Networks and Transfer Learning

## Project Overview

This project implements a deep learning model that can classify CT scan images of kidneys into the following four categories: normal, cyst, stone, and tumor. To achieve this, the model uses a convolutional neural network (CNN) architecture which is trained through transfer learning with pre-existing models to enhance the accuracy of classification. A comparative study is conducted to evaluate the performance of different transfer learning models, including VGG16, InceptionV3, and ResNet50. The dataset used for training and testing consists of over 12,000 CT images of kidneys, which are annotated by expert radiologists.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8+
- Jupyter Notebook
- NumPy
- OpenCV
- TensorFlow/PyTorch
- scikit-learn
- Matplotlib

## Methodology

![image](https://github.com/user-attachments/assets/e08b5a06-2f9c-4711-a01f-9f8e69564862)

We split the dataset into training and testing folders in a 3:1 ratio (75% training, 25% testing) using the splitfolders library in Python. We then augmented the images using ImageDataGenerator from the Keras library in Python. Augmentation was done to add variation to the training data, which helps models to generalize better to new images. We implemented two different sets of augmentation techniques.
Image Augmentation 1: The training dataset images were augmented by:
• Rescaling the pixel values of the images between 0 and 1.
• Shearing the images by a factor between 0 and 0.2
• Zooming in or out of the images by a factor between 0 and 0.2
• Randomly flipping the images horizontally and vertically
• Rotating the images by an angle between -20 and 20 degrees
• Shifting the images horizontally by 20% of their width
• Shifting the images vertically by 20% of their height
• Adjusting the brightness of the images by a factor between 0.2 and 1
• Filling any gaps created by all the transformations with the nearest pixel value.
Image Augmentation 2 The training dataset images were augmented by:
• Rescaling the pixel values of the images between 0 and 1.
• Randomly flipping the images horizontally and vertically
• Rotating the images by an angle between -20 and 20 degrees
• Shifting the images horizontally by 20% of their width
• Shifting the images vertically by 20% of their height
• Filling any gaps created by all the transformations with the nearest pixel value.
The images of the testing dataset were augmented only by rescaling their pixel values between 0 and 1.

## Results and Conclusion

We put four different models to the test, including our own CNN model, as well as transfer learning models such as VGG16, ResNet50, and InceptionV3. We trained these models under two different scenarios using distinct image augmentation techniques.

Based on our findings, we discovered that the InceptionV3 transfer learning model performed the best when it comes to validation accuracy, while the proposed CNN model and VGG16 transfer learning model also showed decent results. However, the ResNet50 transfer learning model exhibited signs of both overfitting and underfitting.

Furthermore, we observed that the choice of image augmentation techniques had an impact on model performance. In particular, our second image augmentation technique yielded better results for the proposed CNN model, while the first technique was more effective for the other models.

To sum it up, our research suggests that incorporating transfer learning and exploring different image augmentation techniques can lead to improved performance of models for detecting kidney disease.
