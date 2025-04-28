# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
![image](https://github.com/user-attachments/assets/f776c302-d9b8-450a-98ea-2552f68fe084)

## DESIGN STEPS
### STEP 1: 
Import all the required libraries (PyTorch, TorchVision, NumPy, Matplotlib, etc.).

### STEP 2: 
Download and preprocess the MNIST dataset using transforms.

### STEP 3: 
Create a CNN model with convolution, pooling, and fully connected layers.

### STEP 4: 
Set the loss function and optimizer. Move the model to GPU if available.

### STEP 5: 
Train the model using the training dataset for multiple epochs.

### STEP 6: 
Evaluate the model using the test dataset and visualize the results (accuracy, confusion matrix, classification report, sample prediction).

## PROGRAM
### Name: GIFTSON RAJARATHINAM N
### Register Number: 212222233002

```python
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Preprocessing
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)) # Normalizing
])

# Load dataset
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms, download=True)

# Check dataset
image, label = train_dataset[0]
print("Image SHAPE:",image.shape,"\nNumber of training samples:", len(train_dataset))

image, label = test_dataset[0]
print("Image Shape:",image.shape)
print("Number of testing samples:",len(test_dataset))

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNNClassifier(nn.Module):
  def __init__(self):
    super(CNNClassifier, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(128 * 3 * 3, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = self.pool(torch.relu(self.conv3(x)))
    x = x.view(x.size(0), -1)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# Print model summary
from torchsummary import summary

model = CNNClassifier()
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

print('Name: Giftson Rajarathinam N')
print('Register Number: 212222233002')

# Print model summary
summary(model, input_size=(1, 28, 28))  # MNIST image size

# Define Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = opt.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs=10):
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
      if torch.cuda.is_available():
        images, labels=images.to(device),labels.to(device)


      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print('Name: Giftson Rajarathinam N')
print('Register Number: 212222233002')

train_model(model, train_loader, num_epochs=10)

def test_model(model, test_loader):
  model.eval()
  correct = 0
  total = 0
  all_preds = []
  all_labels = []

  with torch.no_grad():
    for images, labels in test_loader:
      if torch.cuda.is_available():
        images, labels = images.to(device), labels.to(device)

      outputs = model(images)
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      all_preds.extend(predicted.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

  accuracy = correct/total
  print('Name: Giftson Rajarathinam N')
  print('Register Number: 212222233002')
  print(f"Test Accuracy: {accuracy:.4f}")

  cm = confusion_matrix(all_labels, all_preds)
  plt.figure(figsize=(8, 6))
  print('Name: Giftson Rajarathinam N')
  print('Register Number: 212222233002')
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title("Confusion Matrix")
  plt.show()

  print('Name: Giftson Rajarathinam N')
  print('Register Number: 212222233002')
  print("Classification Report:")
  print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)]))
test_model(model, test_loader)

test_model(model, test_loader)

def predict_image(model, image_index, dataset):
  model.eval()
  image, label = dataset[image_index]
  if torch.cuda.is_available():
    image = image.to(device)

  with torch.no_grad():
    output = model(image.unsqueeze(0))
    _, predicted = torch.max(output, 1)

  class_names = [str(i) for i in range(10)]

  print('Name: Giftson Rajarathinam N')
  print('Register Number: 212222233002')
  plt.imshow(image.cpu().squeeze(), cmap="gray")
  plt.title(f"Actual: {class_names[label]}, \nPredicted: {class_names[predicted.item()]}")
  plt.axis("off")
  plt.show()
  print(f"Actual: {class_names[label]}, \nPredicted: {class_names[predicted.item()]}")

predict_image(model, image_index=80, dataset=test_dataset)




```
### OUTPUT

## Training Loss per Epoch
![Screenshot 2025-04-28 092434](https://github.com/user-attachments/assets/c96a7d04-8911-4e87-8f84-ce01123c0b64)


## Confusion Matrix
![Screenshot 2025-04-28 092507](https://github.com/user-attachments/assets/8b589ef5-a6ef-46bf-a19f-1cda78f73674)


## Classification Report
![Screenshot 2025-04-28 092534](https://github.com/user-attachments/assets/c9bbeafa-3a77-4a0e-9e2c-fe3cb1deb143)


### New Sample Data Prediction
![Screenshot 2025-04-28 092601](https://github.com/user-attachments/assets/d2cd7163-6c8f-47c7-8254-4f3f5ef7220e)


## RESULT
Thus the CNN model was trained and tested successfully on the MNIST dataset.
