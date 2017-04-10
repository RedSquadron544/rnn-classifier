Recurrent Neural Network for Tweet Stance Detection
===================================================

Getting Started
---------------

It is recommended to use `virtualenv` to get the dependencies, you can create and use one in this directory by running
`virtualenv venv` to create the virtualenv in this directory and run `source venv/bin/activate` to start using it.
When you're done, run `deactivate` to stop using the `virtualenv`

To grab all the dependencies, run `pip3 install -r requirements.txt`

Architecture
------------

`process.py` takes a set of tweets and preprocesses them for training/test data. It tokenizes tweet text, and converts each tweet into a `numpy` array of integers representing a unique word in the vocabulary. This is then written to the `tweets.npz` file so that it can be used in the RNN implementation.

`keras_rnn.py` contains the implementation of the RNN model, will further document the RNN structure when finished. The RNN reads the `tweets.npz` file and runs 10-fold cross validation on the data from `tweets.npz`, it then saves the trained model to `model.h5`

`keras_classify.py` loads the saved model from `model.h5` and classifies a topic/tweet pair.
