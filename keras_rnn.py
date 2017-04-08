#!/usr/bin/env python3

# LSTM for sequence classification in the IMDB dataset
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import KFold

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def load():
    data = np.load('tweets.npz')
    x_train = data['x']
    y_train = data['y']

    return (x_train, y_train)

# load the dataset but only keep the top n words, zero the rest
top_words = 2000
(x, y) = load()
# truncate and pad input sequences
max_review_length = 50
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
embedding_vector_length = 32

cvscores = []
# k-fold cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
for train, test in kfold.split(x, y):
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x[train], y[train], epochs=50, batch_size=64, verbose=True)
    # Final evaluation of the model
    scores = model.evaluate(x[test], y[test], verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    cvscores.append(scores[1]*100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
