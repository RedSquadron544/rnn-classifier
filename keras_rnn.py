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

# fix random seed for reproducibility
np.random.seed(7)

def load():
    data = np.load('tweets.npz')
    x_train = data['x']
    y_train = data['y']

    # get a random sample as the test set
    sample_size=int(len(x_train)*0.25)
    indices = np.random.randint(x_train.shape[0], size=sample_size)
    x_test = x_train[indices]
    y_test = y_train[indices]
    # remove the test rows from the training set
    x_train = np.delete(x_train, indices, axis=0)
    y_train = np.delete(y_train, indices, axis=0)

    return ((x_train, y_train), (x_test, y_test))

# load the dataset but only keep the top n words, zero the rest
top_words = 2000
(X_train, y_train), (X_test, y_test) = load()
# truncate and pad input sequences
max_review_length = 50
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print(X_train.shape)
print(y_train.shape)

# create the model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
