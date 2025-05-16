# Custom CNN for MNIST Digit Classification

This repository contains a simple implementation of a Convolutional Neural Network (CNN) from scratch using **NumPy** to classify handwritten digits from the MNIST dataset.

---

## Overview

- Loads MNIST dataset images and labels from raw binary files.
- Implements CNN layers including:
  - Convolutional layers with ReLU activation
  - Max Pooling layer
  - Fully connected (dense) layer with softmax output
- Trains the CNN using cross-entropy loss and gradient descent.
- Evaluates test accuracy on MNIST test set.

---

## Files

- `CustomCNN-digitClassification.ipynb` — Jupyter notebook containing:
  - Data loading functions
  - CNN class and layer implementations
  - Training loop and evaluation

---

## Usage

1. Download the MNIST dataset, I have used kaggle.

2. Update the file paths in the notebook or script accordingly
3. Run the training:
    model = CNN()
    train(model, train_images[:1000], train_labels[:1000], epochs=10, lr=0.005)
4. Evaluate on test data:
    correct = 0
    for i in range(len(test_images[:200])):
        pred = np.argmax(model.forward(test_images[i]))
        correct += (pred == test_labels[i])
    
    print(f"Test Accuracy: {correct / 200:.4f}")

   
## Dependencies
-python
-numpy

   
## Notes
-This CNN implementation is educational and focuses on clarity rather than performance.
-The model is trained on a subset of 1000 training samples and tested on 200 samples for demonstration.
-Extend training to full dataset and tune hyperparameters to improve accuracy.


