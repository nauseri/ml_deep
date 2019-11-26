
# ml mastery ebook: deep learning w python chp 16

# Lift Performance With Learning Rate Schedules

# Adapting the learning rate for your stochastic gradient descent optimization
# procedure can increase performance and reduce training time.
# (names:  (here) learning rate schedule,  learning rate annealing, adaptive learning rates)
# simple and most used lrs: large lr at beginning and small at end of model training
# effect of quickly learning good weights early and ﬁne tuning them later


# Tips for Using Learning Rate Schedules
# Increase the initial learning rate. Because the learning rate will decrease, start with
# a larger value to decrease from. A larger learning rate will result in much larger changes
# to the weights, at least in the beginning, allowing you to benefit from ﬁne tuning later.

# Use a large momentum. Using a larger momentum value will help the optimization
# algorithm continue to make updates in the right direction when your learning rate shrinks
# to small values.

# Experiment with different schedules. It will not be clear which learning rate schedule
# to use so try a few with different configuration options and see what works best on your
# problem. Also try schedules that change exponentially and even schedules that respond to
# the accuracy of your model on the training or test datasets.


# 2 popular and easy to use lrs:
# - Decrease the learning rate gradually based on the epoch, see below
# - Decrease the learning rate using punctuated large drops at speciﬁc epochs, see further below


# for lr decay time based, see ebook chp 16.3
# Time Based Learning Rate Decay
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

import math
from keras.callbacks import LearningRateScheduler

# load dataset
dataframe = read_csv("ionosphere.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 0:34].astype(float)
Y = dataset[:, 34]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
# create model
model = Sequential()
model.add(Dense(34, input_dim=34, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
epochs = 100
learning_rate = 0.1
# create a nice default schedule by setting the decay value as follows:
decay_rate = learning_rate / epochs
momentum = 0.9
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Fit the model
model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=28, verbose=2)


# for lr decay drop schedule, see ebook chp 16.4
# another method,  Learning Rate Using a Drop Schedule:
# drop the learning rate at speciﬁc times during training
# zb dropping the learning rate by half every ﬁxed number of epochs
# implement w LearningRateScheduler, for equation, see ebook
# InitialLearningRate: lr at start of run / EpochDrop: how often to drop / DropRate: how much to drop

## Drop-Based Learning Rate Decay
# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

# Compile model
sgd = SGD(lr=0.0, momentum=0.9)  # here lr set zero because this is the learning rate of
# stochastic gradient descent, ignore this and use lrs from jsut above
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
# Fit the model
model.fit(X, Y, validation_split=0.33, epochs=50, batch_size=28, callbacks=callbacks_list, verbose=2)









