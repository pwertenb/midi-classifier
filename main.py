import midi
import config

import numpy as np
import matplotlib.pyplot as plt

import os.path
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import pickle

model_filename = 'model.pkl'

if not os.path.exists(config.MODEL_FILENAME):
    print("Creating and training model...")
    train_X, train_y = midi.create_train_data()

    model = MLPClassifier(hidden_layer_sizes=config.HIDDEN_LAYERS, learning_rate_init=config.LR, max_iter=config.MAX_EPOCHS, random_state=config.RANDOM_SEED, tol=config.TOL, verbose=config.VERBOSE, early_stopping=config.EARLY_STOP, n_iter_no_change=config.ITER_NO_CHANGE)

    model = model.fit(train_X, train_y)
    pickle.dump(model, open(config.MODEL_FILENAME, 'wb'))
    
model = pickle.load(open(CONFIG.MODEL_FILENAME, 'rb'))

# create test data
test_X, test_y, chord_notes, answers = midi.create_test_data()

# scoring system
def score_func(model, X, y):
    preds = model.predict(X)
    tally = 0
    for i in range(len(preds)):
        if y[i] not in answers[chord_notes[preds[i]]]:
            #print('incorrect chord:', X[i])
            #print(y[i], '(actual) vs', preds[i], '(predicted)')
            #print(chord_notes[y[i]], chord_notes[preds[i]])
            y_notes = np.where(np.array(chord_notes[y[i]]) == 1)[0]
            pred_notes = np.where(np.array(chord_notes[preds[i]]) == 1)[0]
            diff = np.setxor1d(y_notes, pred_notes, assume_unique=True)

            score = (12.0 - len(diff)) / 12.0
            #print('Adding', 12 - len(diff), '/ 12 =', score, 'points\n')
            tally += score
        else:
            tally += 1

    #print('Total:', tally, '/', len(X), 'points')
    return tally / len(X)

train_X, train_y = midi.create_train_data()
chord_notes, answers = midi.get_base_data()
    
print("Model score with custom score function:")
cross_val_score(model, test_X, test_y, scoring=score_func)
