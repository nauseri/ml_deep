

# ml mastery ebook: deep learning w python chp 14

# Understand Model Behavior During Training By Plotting History

# History call back records training metrics for each epoch, zb loss and accuracy metrics
# use data from History and plot them to extract trends in the data, like:
# 1 speed of convergence over epochs (slope).
# 2 Whether the model may have already converged (plateau of the line).
# 3 Whether the model may be over-learning the training data (inﬂection for validation line).
# 4 And more


# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8,  activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1,  activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
# model could be trained a little more, as trends are still rising a bit
# not yet over-learned the training dataset, since both curves show comparable skill
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
# comparable performance, if curves start to depart consistently, might be sign to
# stop training at earlier epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()












