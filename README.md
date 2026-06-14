# cnn-cat-dog-classifier
# CNN Image Classifier — Cat vs Dog

Explored Convolutional Neural Network (CNN) architecture through a hands-on 
workshop project, building a binary image classifier to distinguish cats from 
dogs using the CIFAR-10 dataset.

## What This Project Covers
- Manual convolution using edge detection, blur, and sharpen kernels
- Visualizing feature maps before and after training to understand how filters learn
- Building a CNN with Conv2D, ReLU, MaxPooling, and Dense layers using Keras
- Data augmentation (random flip, zoom, translation) to improve generalization
- Training with EarlyStopping and ReduceLROnPlateau callbacks
- Plotting training vs validation accuracy and loss curves

## Model Architecture
| Layer | Details |
|---|---|
| Input | 32×32 RGB images |
| Conv Block 1 | 32 filters, BatchNormalization, ReLU, MaxPool |
| Conv Block 2 | 64 filters, BatchNormalization, ReLU, MaxPool |
| Conv Block 3 | 128 filters, BatchNormalization, ReLU, MaxPool |
| Dropout | 0.3 |
| Dense | 64 units, ReLU |
| Output | 1 unit, Sigmoid (dog=1, cat=0) |

## Dataset
CIFAR-10 — built into TensorFlow/Keras
- Training: 5,000 cats + 5,000 dogs
- Validation: 1,000 cats + 1,000 dogs

## Tools & Libraries
Python | TensorFlow | Keras | NumPy | Matplotlib

## Key Learnings
- Random filters before training produce noisy, unstructured feature maps
- After training, filters learn to detect meaningful edges and textures
- BatchNormalization stabilizes training and speeds up convergence
- Data augmentation significantly reduces overfitting on small datasets
