import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing import sequence
import fasttext


# evaluation functions
# copied from older version of Keras https://github.com/fchollet/keras/blob/2b51317be82d4420169d2cc79dc4443028417911/keras/metrics.py
import keras.backend as K
def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
        
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def f1(y_true, y_pred):
    return fbeta_score(y_true, y_pred, 1)
# end evaluation functions



def load_training_data():
    data = np.load('tweets.npz')
    x_train = data['x']
    y_train = data['y']
    topics_train = data['topics']
    words = data['words']

    return (x_train, topics_train, y_train, words)

(x, topics, y, words) = load_training_data()
# make sure we have the right shape for inputs
max_tweet_length = x.shape[1]
max_topic_length = topics.shape[1]

# TODO: is there a way to determine this automatically?
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

embedding_matrix = load_word_embeddings()

def build_embedding(input_length, name=None):
    return Embedding(
        name=name,
        input_dim=len(words)+1,
        output_dim=embed_dim,
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False)

def build_model():
    topic_input = Input(shape=(max_topic_length,), dtype='int32', name='topic_input')
    # use a pretrained embedding, so set the weights and don't let it be trained
    topic_embedding_1 = build_embedding(max_topic_length, name='topic_embedding')(topic_input)
    topic_dropout_embedding_1 = Dropout(0.2)(topic_embedding_1)
    topic_lstm_1 = LSTM(32, name='topic_lstm')(topic_dropout_embedding_1)
    topic_dropout_lstm_1 = Dropout(0.2)(topic_lstm_1)

    tweet_input = Input(shape=(max_tweet_length,), dtype='int32', name='tweet_input')
    tweet_embedding = build_embedding(max_tweet_length, name='tweet_embedding')(tweet_input)

    tweet_lstm = LSTM(32, name='tweet_lstm')(tweet_embedding)
    tweet_dropout_lstm = Dropout(0.2)(tweet_lstm)

    tweet_convolve = Conv1D(filters=1, kernel_size=3, activation='relu')(tweet_embedding)
    tweet_convolve_dropout = Dropout(0.2)(tweet_convolve)
    tweet_maxpool = MaxPooling1D(pool_size=32)(tweet_convolve_dropout)
    tweet_convolve_flatten = Flatten()(tweet_maxpool)

    merge = Concatenate()([topic_dropout_lstm_1, tweet_convolve_flatten, tweet_dropout_lstm])

    category_output = Dense(4, activation='softmax')(merge)

    model = Model(inputs=[topic_input, tweet_input], outputs=category_output)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1])

    return model
