# MNIST Digit Classifier

This project involves training a neural network model to classify handwritten digits using the MNIST dataset. The model is built using TensorFlow and Keras.

## Project Structure

- `train_model.py`: Script to train the model.
- `my_mnist_model.keras`: Trained model saved in Keras format.

## Prerequisites

Ensure you have the following installed:
- Python 3.8 or later
- TensorFlow 2.x
- Matplotlib
- NumPy

You can install the required packages using pip:

`Bash
 pip install tensorflow matplotlib numpy`

The script will:

Load and normalize the MNIST dataset.
Define and compile the model.
Train the model for 10 epochs.
Save the model to my_mnist_model.keras.
Evaluating the Model
After training, the script evaluates the model on the test dataset and prints the test accuracy.

Plotting Training Results
Accuracy over training and validation is plotted using Matplotlib to visualize the performance of the model during training.

Saving and Downloading the Model
After training, the model is saved as my_mnist_model.keras. If you're running this in an environment like Google Colab, you can download the model using:

from google.colab import files
files.download('my_mnist_model.keras')

Usage
To use the trained model for prediction, load it using TensorFlow Keras:

import tensorflow as tf
model = tf.keras.models.load_model('my_mnist_model.keras')
