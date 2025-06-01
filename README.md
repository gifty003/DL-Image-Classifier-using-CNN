# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Classify handwritten digits (0–9) using the MNIST dataset.


### STEP 2: 
Import required libraries such as TensorFlow/Keras, NumPy, and Matplotlib. Load the MNIST dataset using keras.datasets.mnist.load_data().


### STEP 3: 
Normalize the image pixel values (scale from 0-255 to 0-1). Reshape the images to match CNN input shape.




### STEP 4: 
Initialize a Sequential model. Add convolutional layers with activation (ReLU), followed by pooling layers. Flatten the output and add Dense layers. Use a softmax layer for classification.


### STEP 5: 
Compile the model with an optimizer (e.g., Adam), loss function (e.g., categorical crossentropy), and metrics (accuracy). Train the model using training data and validate using validation split or test data.


### STEP 6: 
Evaluate the model on test data and print accuracy. Plot training/validation loss and accuracy curves. Optionally, display a confusion matrix or sample predictions.




### Name:ONTEDDU SIRISHA REDDY

### Register Number:212222230103
### OUTPUT

## Training Loss per Epoch

Include the Training Loss per epoch

## Confusion Matrix

Include confusion matrix here

## Classification Report
Include classification report here

### New Sample Data Prediction
Include your sample input and output here

## RESULT
Include your result here
