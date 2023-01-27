import midi
import config

import numpy as np
import matplotlib.pyplot as plt

import os.path
from sklearn.neural_network import MLPClassifier
import pickle

model_filename = 'model.pkl'
chord_notes, answers = midi.get_base_data()

#FIXME
print([i for i in chord_notes.keys() if "A#" in i or "Bb" in i or "Ddim" in i])
print([i for i in answers.items() if "Bb7" in i[1]])
#exit()


if not os.path.exists(model_filename):
    print("Creating and training model...")
    train_X, train_y = midi.create_train_data()

    model = MLPClassifier(hidden_layer_sizes=config.HIDDEN_LAYERS, learning_rate_init=config.LR, max_iter=config.MAX_EPOCHS, random_state=config.RANDOM_SEED, tol=config.TOL, verbose=config.VERBOSE, early_stopping=config.EARLY_STOP, n_iter_no_change=config.ITER_NO_CHANGE)

    model = model.fit(train_X, train_y)
    pickle.dump(model, open(model_filename, 'wb'))
    
model = pickle.load(open(model_filename, 'rb'))

# create test data
test_y = np.array(['B7', 'Em', 'Am', 'D7', 'G', 'D#dim', 'B7', 'Esus4', 'F#dim', 'B7+5', 'Em', 'D', 'A7', 'D', 'G', 'F#m', 'A7', 'Em', 'Bm'])
tmp = np.array([[11, 3, 6, 9],
                [4, 7, 11, 4],
                [9, 0, 4, 9],
                [2, 6, 9, 0],
                [7, 11, 2, 7],
                [6, 9, 3, 9],
                [11, 6, 3, 6, 11, 9],
                [4, 7, 11, 4, 9],
                [9, 0, 0, 9, 6],
                [11, 9, 3, 7],
                [4, 11, 4, 7, 4],
                [2, 6, 9],
                [4, 9, 1, 7],
                [6, 9, 2, 9, 2],
                [7, 2, 7, 11],
                [9, 1, 6, 9],
                [9, 1, 4, 7],
                [11, 11, 4, 7],
                [11, 11, 2, 6]], dtype=object)
test_X = np.empty((0,12))
for i in range(len(tmp)):
    test_X = np.append(test_X, np.array([midi.__format_chord__(tmp[i])]), axis=0)

# FIXME
test_X, test_y, chord_notes, answers = midi.create_test_data()

# scoring system
def score_func(model, X, y):
    preds = model.predict(X)
    tally = 0
    for i in range(len(preds)):
        if y[i] not in answers[chord_notes[preds[i]]]:
            print('incorrect chord:', X[i])
            print(y[i], '(actual) vs', preds[i], '(predicted)')
            print(chord_notes[y[i]], chord_notes[preds[i]])
            print()
            
            y_notes = np.where(np.array(chord_notes[y[i]]) == 1)[0]
            pred_notes = np.where(np.array(chord_notes[preds[i]]) == 1)[0]
            diff = np.setxor1d(y_notes, pred_notes, assume_unique=True)

            score = (12.0 - len(diff)) / 12.0
            print('Adding', 12 - len(diff), '/ 12 =', score, 'points')
            tally += score
        else:
            tally += 1

    print('Total:', tally, '/', len(X), 'points')
    return tally / len(X)
    
print("Model score with custom score function:", score_func(model, test_X, test_y))

