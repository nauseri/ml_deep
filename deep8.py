

# ml mastery ebook: deep learning w python chp 13

# Keep The Best Models During Training With Checkpointing
# Application checkpointing is a fault tolerance technique for long running processes.
# snapshot of system taken in case of system failure, captures weights of model
# keras: checkpoint with callback class ModelCheckpoint
# define where to checkpoint model WEIGHTS not model structure!
# file name, when to do a checkpoint, monitor which metric, look for an improvement

# checkpoint when during training, model improvement is achieved, then output model weights

# Checkpoint the weights when validation accuracy improves for each epoch...
# or checkpoint model weights to same file if validation accuracy improves, so not at epochs, see below


# Checkpoint the weights for best model on validation accuracy
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import numpy
from keras.models import load_model
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# Fit the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)


# how to load and use checkpoint model, only knows model weights, model structure from JSON OR YAML file

# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")
# load model
model = load_model('model.h5')
# summarize model.
model.summary()
# load weights
model.load_weights("weights.best.hdf5")
# Compile model (required to make predictions)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Created model and loaded weights from file")
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X1 = dataset[:, 0:8]
Y1 = dataset[:, 8]
# estimate accuracy on whole dataset using loaded weights
scores = model.evaluate(X1, Y1, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))









