

# ml mastery ebook: deep learning w python chp 8

# keras library focus on deep learning in python, for general purpose use sklearn (scikit-learn) library
# sklearn use for zb model evaluation w kfold cv or search of hyperparameters
# keras as wrapper for classification neural network w KerasClassifier or KerasRegressor
# use stratifiedkfold for kfold cv and cross_val_score() for evaluation
# grid search to evaluate dif configs for nn model, find best hyperparameters
# find best optimizer, initializer, nr of epochs and nr in a batch
# this takes a lot of time! maybe use a smaller subset of data to reduce computation time



# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy


# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, Y, cv=kfold)
print('Accuracy: %.2f  %.2f' % results.mean()*100, results.std()*100)





















