
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
from pandas import read_csv
import matplotlib.pyplot as plt
from pandas import set_option
from pandas.plotting import scatter_matrix
from pandas.plotting import hist_frame


# ml mastery ebook: deep learning w python chp 6

# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
# dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# # split into input (X) and output (y) variables
# X = dataset[:, 0:8]
# y = dataset[:, 8]
filename = 'pima-indians-diabetes.csv'
names = ['nr preg', 'plas test', 'pressure', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'class']
data = read_csv(filename, names=names)
array = data.values
X = array[:, 0:8]
y = array[:, 8]

# Statistical Summary
# data = pd.DataFrame(dataset)
print('raw data\n', data.head(4))
np.set_printoptions(precision=4)
print('shape: ', data.shape)
data.info()
# class_counts = data.groupby('class').size()
# print(class_counts)
# print(data.dtypes)
set_option('display.width', 100)
set_option('precision', 3)
description = data.describe()
print('description\n', description)
correlations = data.corr(method='pearson')
print('correlation\n', correlations)
skew = data.skew()
print('skew\n', skew)   # positive (right) or negative (left) skew

# Scatter Plot Matrix
hist_frame(data, bins=10)
plt.xlabel("nr")
plt.ylabel("frec")
plt.show()
data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
plt.show()
data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
plt.show()

#  linear and logistic regression can have poor performance if there are
#  highly correlated input variables in your data
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, 9, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
scatter_matrix(data)
plt.show()


# define the KERAS model
# models as sequence of layers, use sequential model, number of layers and type w trial n error
# fully connected layers defined using Dense class
# specify nr of neurons in layer and activation function
# here relu (rectifier fkt) on first 2 layers and 3 output layer sigmoid activation fkt
# sigmoid act fkt to set output to 0 or 1
# first hidden layer 12 neurons w 8 input variables
# second layer 8 neurons and output layer 1 neuron to predict class, onset diabetes or not
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
# compile model, here tensorflow used as backend
# need add. properties for training network:
#     loss fkt to evaluate set of weights
#     optimizer used to search through different weights
#     optional metrics to report during training
# here: log loss, in keras: binary_crossentropy
# gradient descent algorithm adam, efficient default algo
# opt. metric for report: accuracy for classification
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
# epochs: nr of iterations
# batch_size nr of instances evaluated before a weight update in network
model.fit(X, y, epochs=150, batch_size=16, verbose=0)  # verbose surpresses output

# evaluate the keras model
# here for simplicity no split data into train and test done
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# neural networks are stochastic algorithm! same algo on same data can train dif model
# each time code is run. may need to fit model many times and take avg of acc score

# make class predictions with the model
# no train test split done here, pretend...
# used sigmoid fkt, probability will be between 0 and 1
# this fkt below converts probabilities into binary predictions of 0s and 1s
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))















