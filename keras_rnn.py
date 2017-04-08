#!/usr/bin/env python3

# LSTM for sequence classification in the IMDB dataset
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing import sequence
from sklearn.model_selection import KFold
import fasttext

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def load_training_data():
    data = np.load('tweets.npz')
    x_train = data['x']
    y_train = data['y']
    words = data['words']

    return (x_train, y_train, words)

(x, y, words) = load_training_data()

embed_dim = 100
def load_word_embeddings():
    model = fasttext.load_model('model.bin')
    embedding_matrix = np.zeros((len(words) + 1, embed_dim))
    for i in range(len(words)):
        word = words[i]
        embedding_vector = model[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# truncate and pad input sequences
max_review_length = 50

# create the model

embedding_matrix = load_word_embeddings()

def build_model():
    tweet_input = Input(shape=(max_review_length,), dtype='int32', name='tweet_input')
    # use a pretrained embedding, so set the weights and don't let it be trained
    embedding_1 = Embedding(
        input_dim=len(words)+1,
        output_dim=embed_dim,
        weights=[embedding_matrix],
        input_length=max_review_length,
        trainable=False)(tweet_input)

    dropout_embedding_1 = Dropout(0.2)(embedding_1)
    lstm_1 = LSTM(100)(dropout_embedding_1)
    dropout_lstm_1 = Dropout(0.2)(lstm_1)
    category_output = Dense(4, activation='softmax')(dropout_lstm_1)

    model = Model(inputs=tweet_input, outputs=category_output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


cvscores = []
# k-fold cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
for train, test in kfold.split(x, y):
    model = build_model()
    model.fit(x[train], y[train], epochs=50, batch_size=64, verbose=True)
    # Final evaluation of the model
    scores = model.evaluate(x[test], y[test], verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    cvscores.append(scores[1]*100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
