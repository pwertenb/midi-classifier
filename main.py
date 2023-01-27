import midi
import config

import numpy as np
import matplotlib.pyplot as plt

# get midi data
def format_chord(tmp):
    result = np.zeros((12,))
    for i in tmp:
        result[i] = 1
    return result.astype(int)

print("Importing files...")
chords, longest = midi.import_chords(midi.get_files(config.CHORDS_DIR))

# format midi data (extract just notes)
print("Extracting notes from data...")
draft = {}
for i in chords:
    tmp = np.array(chords[i][0])[:,0:1]
    tmp = np.reshape(tmp, (len(tmp))) % 12
    draft[i] = format_chord(tmp)

# discovering duplicate chords
def chord_to_tuple(chord):
    return tuple(chord)

print("discovering duplicate chords...")

chord_notes = dict()
tmp = []
for key, val in draft.items():
    tup = chord_to_tuple(val)
    if tup not in tmp:
        tmp.append(tup)
        chord_notes[key] = tup
answers = dict()
for key, val in draft.items():
    tup = chord_to_tuple(val)
    if tup in answers:
        answers[tup].append(key)
    else:
        answers[tup] = [key]
    
# set up model
import os.path
from sklearn.neural_network import MLPClassifier
import pickle

model_filename = 'model.pkl'
if not os.path.exists(model_filename):
    print("Creating and training model...")
    y = np.array(list(chord_notes.keys()))
    X = np.array([np.asarray(chord_notes[i]) for i in chord_notes])
    model = MLPClassifier(hidden_layer_sizes=config.HIDDEN_LAYERS, learning_rate_init=config.LR, max_iter=config.MAX_EPOCHS, random_state=config.RANDOM_SEED, tol=config.TOL, verbose=config.VERBOSE, early_stopping=config.EARLY_STOP, n_iter_no_change=config.ITER_NO_CHANGE)
    model = model.fit(X, y)
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
test_X = np.empty((0,config.NUM_NOTES))
for i in range(len(tmp)):
    test_X = np.append(test_X, np.array([format_chord(tmp[i])]), axis=0)

# scoring system
# if exact chord, give 1 point
# if different chord but contains same notes, give 0.5 points
def score_func(model, X, y):
    preds = model.predict(X)
    tally = 0
    for i in range(len(preds)):
        if preds[i] != y[i]:
            print('incorrect chord:')
            print(X[i])
            print(y[i], '(actual) vs', preds[i], '(predicted)')
            print(chord_notes[y[i]])
            print(chord_notes[preds[i]])
            
            y_notes = np.where(np.array(chord_notes[y[i]]) == 1)[0]
            pred_notes = np.where(np.array(chord_notes[preds[i]]) == 1)[0]
            intersect = np.intersect1d(y_notes, pred_notes, assume_unique=True)
            diff = np.setxor1d(y_notes, pred_notes, assume_unique=True)
            print('Adding', len(intersect), '/', len(intersect) + len(diff) - 1, '=', len(intersect) / (len(intersect) + len(diff) - 1), 'points')
            tally += len(intersect) / (len(intersect) + len(diff) - 1)   

    for i in range(len(preds)): 
        pred_bools = np.greater(np.array(chord_notes[preds[i]]), np.zeros(len(chord_notes[preds[i]])))
        for j in answers.values():
            j_bools = np.greater(np.array(chord_notes[j[0]]), np.zeros(len(chord_notes[j[0]])))
            if preds[i] in j and y[i] in j:
                print(preds[i], ': 1 point')
                tally += 1
                break
    print('Total:', tally, '/', len(X), 'points')
    return tally / len(X)
    
print("Model score with custom score function:", score_func(model, test_X, test_y))

