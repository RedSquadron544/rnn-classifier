#!/usr/bin/env python3
import json
import sys
from nltk.tokenize import TweetTokenizer
import numpy as np

labeled = json.load(sys.stdin)

longest_tweet = 0
longest_tweet_text = None
link_count = 0

tweets = []

tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

for tweet in labeled['tweets']:
    if 'label' in tweet and tweet['label'] != 'other':
        text = tweet['text']
        if text[0] == '"' and text[-1] == '"':
            text = text[1:-1]
        text = tokenizer.tokenize(text)
        tweet_stripped = {
                'text': text,
                'hashtags': tweet['hashtags'].split(' '),
                'label': tweet['label']
        }
        tweets.append(tweet_stripped)
        if len(text) > longest_tweet:
            longest_tweet = len(text)
            longest_tweet_text = tweet['text']

# for tweet in tweets:
#     print(json.dumps(tweet))

print (longest_tweet)
print (longest_tweet_text)
# print (link_count)
# print (len(tweets))

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

print(len(words), 'unique tokens')

labels = {
        'agree': 0,
        'disagree': 1,
        'unrelated': 2,
        'neither': 3,
        'other': 4
        }

tokens_length = max(longest_tweet, 50) #50 for LaPen
x = []
y = []
for tweet in tweets:
    tokens = tweet['tokens']
    tokens = np.pad(tokens, (0, tokens_length - len(tokens)), 'constant')
    x.append(tokens)
    # label = float(np.array(labels[tweet['label']])) / 5.1
    label = np.zeros(4)
    label[labels[tweet['label']]] = 1
    y.append(label)

x = np.array(x)
y = np.array(y)
words_np = np.zeros((len(words)+3), dtype=np.unicode_)
for index, word in words.items():
    words_np[index] = word

print(x.shape)
print(y.shape)
print(words_np.shape)

np.savez('tweets.npz', x=x, y=y, words=words_np)
