

import sys
print('Python: {}'.format(sys.version))
import scipy
print('scipy: %s' % scipy.__version__)
import numpy as np
print('numpy: %s' % np.__version__)
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
import pandas as pd
print('pandas: %s' % pd.__version__)
# import statsmodels
# print('statsmodels: %s' % statsmodels.__version__)
import sklearn  #scikit-learn
print('sklearn: %s' % sklearn.__version__)
import tensorflow
print('tensorflow: %s' % tensorflow.__version__)
import keras
print('keras: %s' % keras.__version__)

import numpy as np
import pandas as pd
from pandas import read_csv
import sklearn
import xgboost
from pickle import dump
from pickle import load
import matplotlib.pyplot as plt


# ml mastery ebook: deep learning w python chp 1-5
# intro: tensorflow and theano r libraries for numerical computing from google, open source
# tensorflow: nodes ( data moving between nodes called tensors), edges, operation

# Example of TensorFlow library
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# declare two symbolic floating-point scalars
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# create a simple symbolic expression using the add function
add = tf.add(a, b)
# bind 1.5 to ’ a ’ , 2.5 to ’ b ’ , and evaluate ’ c ’
sess = tf.Session()
binding = {a: 1.5, b: 2.5}
c = sess.run(add, feed_dict=binding)
print(c)

# keras intro: keras python library runs on top of tensorflow and theano,
# provides convenience for creating deep learning models on top of tensorflow
# runs on gpu or cpu,  provides a consistent interface to eﬃcient numerical libraries called backends
# add or edit keras configuration file in home directory
# confirm which backend is being used with: python -c "from keras import backend; print(backend.backend())"
# main type of model is a sequence of layers called a Sequential which is a linear stack of layers

# main steps: define model, compile model, fit model, make predictions


# crash course multilayer perceptrons
# 1. Multilayer Perceptrons.
# 2. Neurons, Weights and Activations.
# 3. Networks of Neurons.
# 4. Training Networks.

# goal:  develop robust algorithms and data structures that we can use to model difficult problems
# Neural Networks = universal approximation algorithm
# The power of neural networks come from their ability to learn the representation in your
# training data and how to best relate it to the output variable that you want to predict.
# artifical neurons, building block of neural network, have input output weight and activation function
# neuron has a bias and inputs and has weights for bias and inputs
# weights often initialized, larger weights indicate increased complexity, want small weights
# activation function (transfer function) takes weighted inputs and produces an output, simple mapping
# governs the threshold at which neuron is activated and strength of output signal

# network of neurons, row of neurons called layer
# network can have multiple layers, architecture of neurons also called network topology...
# the bottom layer that takes input from your dataset is called the visible layer
# Layers after the input layer are called hidden layers because they are not directly exposed to the input
# The ﬁnal hidden layer is called the output layer, outputting a value or vector of values
# activation function in output layer depends on type of problem
# A multiclass classiﬁcation problem may have multiple neurons in the output layer, one for each class

# prepare data for training neural network, data must be numerical
# categorical data converted into values, this is called hot encoding
# one new column added for each class value ( example: for sex male of female two columns added with 0s and 1s)
# same can be done for output variables in classification problems w more than one class
# scaled required, normalizaiton or standardization

# stochastic gradient descent, preferred training algo for neutral networks
# one row of data exposed to network at a time as input, layer by layer called forward pass
# network output compared to expected and error calculated
# error propagated back through network, layer per layer and weights updated, called backpropagation algo
# weights are updated according to the amount that they contributed to the error
# One round of updating the network for the entire training dataset is called an epoch, network trained on many epochs

# weights updated immediately called online learning, if errors saved and updated at the end, called batch learning, more stable
# amount weights are updated, controlled by learning rate, also called step size, controls network weights change for given error
# small learning rates used 0.1 0.01
# momentum: properties from the previous weight update incorporated, to allow weights to continue to change
# in the same direction even when there is less error being calculated
# Learning Rate Decay: learning rate over epochs decreases, beginning more change and at end less
# end of neural network training, predictions can be made by feeding network unknown data, need to save topology and final set of weights






