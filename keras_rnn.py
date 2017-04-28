#!/usr/bin/env python3

# LSTM for sequence classification in the IMDB dataset
import numpy as np
from sklearn.model_selection import KFold
from keras.callbacks import TensorBoard, ModelCheckpoint
import os

from model import build_model, load_training_data

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

(x, topics, y, words) = load_training_data()

epochs=50
batch_size=16

validate = True
if validate:
    cvscores = []
    # k-fold cross validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    for split_index, (train, test) in enumerate(kfold.split(x, y)):
        os.makedirs('./weights/{}'.format(split_index), exist_ok=True)
        model = build_model()
        tensorboard = TensorBoard(log_dir='./logs/{}'.format(split_index), histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        # checkpoint
        filepath='weights/{}/weights.best.hdf5'.format(split_index)
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=False, save_best_only=True, mode='min')
        model.fit([topics[train], x[train]], y[train], epochs=epochs, batch_size=batch_size, verbose=True, callbacks=[tensorboard, checkpoint])
        # Final evaluation of the model
        # load the best model only
        model.load_weights('weights/{}/weights.best.hdf5'.format(split_index))
        scores = model.evaluate([topics[test], x[test]], y[test], verbose=False)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        print("Precision: %.2f%%" % (scores[2]*100))
        print("Recall: %.2f%%" % (scores[3]*100))
        print("F1: %.2f" % (scores[4],))
        print("=================")
        cvscores.append(np.array([scores[1]*100, scores[2]*100, scores[3]*100, scores[4]]))

    averages = np.mean(cvscores, axis=0)
    std_devs = np.std(cvscores, axis=0)
    print('Overall:')
    print('Accuracy: %.2f%% (+/- %.2f%%)' % (averages[0], std_devs[0]))
    print('Precision: %.2f%% (+/- %.2f%%)' % (averages[1], std_devs[1]))
    print('Recall: %.2f%% (+/- %.2f%%)' % (averages[2], std_devs[2]))
    print('F1: %.2f (+/- %.2f)' % (averages[3], std_devs[3]))

# train the model on the full training set
print('Training full model')
model = build_model()
model.fit([topics, x], y, epochs=epochs, batch_size=batch_size, verbose=False)
print('Finished training full model')
model.save('model.h5')
