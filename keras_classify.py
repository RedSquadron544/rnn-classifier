#!/usr/bin/env python3
import numpy as np
from nltk.tokenize import TweetTokenizer
from collections import defaultdict
import pickle

from model import build_model

model = build_model()
model.load_weights('model.h5')

data = np.load('tweets.npz')
max_tweet_length = data['x'].shape[1]
max_topic_length = data['topics'].shape[1]

vocab_file = open('vocab.pck', 'rb')
vocabulary = pickle.load(vocab_file)

# print(words)
print(len(vocabulary), 'unique tokens')

tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
def string_to_array(s, l):
    tokens = tokenizer.tokenize(s)
    if len(tokens) > l:
        tokens = tokens[0:l]

    def index_for_token(token):
        if token not in vocabulary:
            return 2
        return vocabulary[token]
    tokens = [index_for_token(t) for t in tokens]
    tokens = np.array(tokens)
    tokens = np.pad(tokens, (0, l - len(tokens)), 'constant')
    return tokens


topic = "Je soutiens Emmanuel Macron. J'encourage Emmanuel Macron. Emmanuel Macron pour président."

# text = "#Enmarche avec @EmmanuelMacron pour une #diplomatie renouvel\u00e9e, ambitieuse et adapt\u00e9e \u00e0 la donne et aux enjeux du monde d'aujourd'hui https:// twitter.com/enmarcheinter/ status/826580027693228034 \u2026"
text = "Macron est le pire président pour la France"

topic = string_to_array(topic, max_topic_length).reshape((1, max_topic_length))
text = string_to_array(text, max_tweet_length).reshape((1, max_tweet_length))

print(topic)
print(text)

labels = {
        0: 'agree',
        1: 'disagree',
        2: 'unrelated',
        3: 'neither',
}


prediction = model.predict([topic, text])
print(prediction)

most_likely = prediction[0].argmax()
most_likely = labels[most_likely]
print(most_likely)
