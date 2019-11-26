

# ml mastery ebook: deep learning w python chp 15


# Reduce Overﬁtting With Dropout Regularization
# dropout: randomly selected neurons are ignored/ dropped out during training
# their contribution to the activation of downstream neurons is temporally removed on the forward pass
# and any weight updates are not applied to the neuron on the backward pass.

# Weights of neurons are tuned for speciﬁc features providing some specialization. Neighboring
# neurons come to rely on this specialization, which if taken too far can result in a fragile model
# too specialized to the training data. This reliance on context for a neuron during training is
# referred to as complex co-adaptations.

# You can imagine that if neurons are randomly dropped
# out of the network during training, that other neurons will have to step in and handle the
# representation required to make predictions for the missing neurons. This is believed to result
# in multiple independent internal representations being learned by the network.

# The effect is that the network becomes less sensitive to the speciﬁc weights of neurons. This
# in turn results in a network that is capable of better generalization and is less likely to overﬁt
# the training data.

# dropout used only during model training and not during evaluation
# can be applied to visible and invisible layers
# recommended in original dropout paper, imposed constraint on weights for each hidden layer, max norm of weights < 3
# done w kernel_constraint on Dense class

# Learning Rate Decay is used to decrease the learning rate over epochs to allow the
# network to make large changes to the weights at the beginning and smaller ﬁne tuning
# changes later in the training schedule.

# Momentum is a term that incorporates the properties from the previous weight update
# to allow the weights to continue to change in the same direction even when there is less
# error being calculated.


# Tips For Using Dropout
# Generally use a small dropout value of 20%-50% of neurons with 20% providing a good
# starting point. A probability too low has minimal effect and a value too high results in
# under-learning by the network.

#  Use a larger network. You are likely to get better performance when dropout is used
# on a larger network, giving the model more of an opportunity to learn independent
# representations.

# Use dropout on input (visible) as well as hidden layers. Application of dropout at each
# layer of the network has shown good results.
# Use a large learning rate with decay and a large momentum. Increase your learning rate
# by a factor of 10 to 100 and use a high momentum value of 0.9 or 0.99. for learning rate, see next lesson!

# Constrain the size of network weights. A large learning rate can result in very large
# network weights. Imposing a constraint on the size of network weights such as max-norm
# regularization with a size of 4 or 5 has been shown to improve results.


# Example of Dropout on the Sonar Dataset: Visible Layer
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset
dataframe = read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# dropout in the input layer with weight constraint
def create_model():
    # create model
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(60,)))
    model.add(Dense(60, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.1, momentum=0.9)  # lr=0.1 is a high learning rate? apparently...
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Visible: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# additional training epochs required or further tuning to the learning rate











