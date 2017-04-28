#!/usr/bin/env python3
import json
import sys
from nltk.tokenize import TweetTokenizer
import numpy as np

labeled = json.load(sys.stdin)

longest_tweet = 0
longest_topic = 0

tweets = []
topics = []

tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

for tweet in labeled['tweets']:
    if 'label' in tweet and tweet['label'] != 'other':
        text = tweet['text']
        if text[0] == '"' and text[-1] == '"':
            text = text[1:-1]
        text = tokenizer.tokenize(text)
        topic = tweet['topic']
        topic = tokenizer.tokenize(topic)
        hashtags = []
        if 'hashtags' in tweet:
            hashtags = tweet['hashtags'].split(' ')
        tweet_stripped = {
                'text': text,
                'hashtags': hashtags,
                'label': tweet['label'],
                'topic': topic,
        }
        tweets.append(tweet_stripped)
        if len(text) > longest_tweet:
            longest_tweet = len(text)
        if len(topic) > longest_topic:
            longest_topic = len(topic)

print(longest_tweet)
print(longest_topic)

# map each token to its number, use 0 for padding, 1 for start and 2 for out-of-vocabulary
vocabulary = {}

words = {}
for tweet in tweets:
    tweet['tokens'] = []
    for token in tweet['text']:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary) + 3 # real tokens start at index 3
            words[vocabulary[token]] = token
        tweet['tokens'].append(vocabulary[token])
    tweet['topic_tokens'] = []
    for token in tweet['topic']:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary) + 3 # real tokens start at index 3
            words[vocabulary[token]] = token
        tweet['topic_tokens'].append(vocabulary[token])

print(len(words), 'unique tokens')

labels = {
        'agree': 0,
        'disagree': 1,
        'unrelated': 2,
        'neither': 3,
        }

x = []
y = []
for tweet in tweets:
    tokens = tweet['tokens']
    tokens = np.pad(tokens, (0, longest_tweet - len(tokens)), 'constant')
    x.append(tokens)
    # label = float(np.array(labels[tweet['label']])) / 5.1
    label = np.zeros(4)
    label[labels[tweet['label']]] = 1
    y.append(label)

    tokens = tweet['topic_tokens']
    tokens = np.pad(tokens, (0, longest_topic - len(tokens)), 'constant')
    topics.append(tokens)

x = np.array(x)
y = np.array(y)
topics = np.array(topics)
words_np = np.zeros((len(words)+3), dtype=np.unicode_)
for index, word in words.items():
    words_np[index] = word

print(x.shape)
print(y.shape)
print(topics.shape)
print(words_np.shape)

np.savez('tweets.npz', x=x, y=y, topics=topics, words=words_np)

import pickle
# save the vocabulary
vocab_file = open('vocab.pck', 'wb')
pickle.dump(vocabulary, vocab_file)
