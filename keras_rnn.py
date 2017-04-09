#!/usr/bin/env python3

# LSTM for sequence classification in the IMDB dataset
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate
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
    topics_train = data['topics']
    words = data['words']

    return (x_train, topics_train, y_train, words)

(x, topics, y, words) = load_training_data()

print(x.shape)
print(topics.shape)
print(y.shape)

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

# make sure we have the right shape for inputs
max_tweet_length = x.shape[1]
max_topic_length = topics.shape[1]
# print(max_tweet_length)
# print(max_topic_length)

# create the model

embedding_matrix = load_word_embeddings()

def build_model():
    topic_input = Input(shape=(max_topic_length,), dtype='int32', name='topic_input')
    topic_embedding_1 = Embedding(
        input_dim=len(words)+1,
        output_dim=embed_dim,
        weights=[embedding_matrix],
        input_length=max_topic_length,
        trainable=False)(topic_input)
    topic_dropout_embedding_1 = Dropout(0.2)(topic_embedding_1)
    topic_lstm_1 = LSTM(100)(topic_dropout_embedding_1)
    topic_dropout_lstm_1 = Dropout(0.2)(topic_lstm_1)


    tweet_input = Input(shape=(max_tweet_length,), dtype='int32', name='tweet_input')
    # use a pretrained embedding, so set the weights and don't let it be trained
    embedding_1 = Embedding(
        input_dim=len(words)+1,
        output_dim=embed_dim,
        weights=[embedding_matrix],
        input_length=max_tweet_length,
        trainable=False)(tweet_input)

    dropout_embedding_1 = Dropout(0.2)(embedding_1)
    lstm_1 = LSTM(100)(dropout_embedding_1)
    dropout_lstm_1 = Dropout(0.2)(lstm_1)

    # combine the tweet network and the topic network
    combined = Concatenate()([topic_dropout_lstm_1, dropout_lstm_1])
    category_output = Dense(4, activation='softmax')(combined)

    model = Model(inputs=[topic_input, tweet_input], outputs=category_output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

validate = True
if validate:
    cvscores = []
    # k-fold cross validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    for train, test in kfold.split(x, y):
        model = build_model()
        model.fit([topics[train], x[train]], y[train], epochs=50, batch_size=64, verbose=True)
        # Final evaluation of the model
        scores = model.evaluate([topics[test], x[test]], y[test], verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        cvscores.append(scores[1]*100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

model = build_model()
model.fit([topics, x], y, epochs=50, batch_size=64, verbose=True)
model.save('model.h5')
