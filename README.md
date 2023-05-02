# Musical Chord Classifier

This project creates a classifier that inputs a few musical notes as MIDI data and determines what chord those notes would fall under. The model is trained on all of the chords listed in the `midi-chords` directory. Each chord is reduced down to a 12 tone space.

The model used is a simply PyTorch network, whose hyper-parameters can be found in `config.py`. Once the model with the given example configuration is trained, it can classify the chords within the `test` directory with 100% accuracy (this again can be changed in `config.py`). Note that the score for an enharmonic chord is the *same* as for an exact match.

## Installation

All Python packages to install are listed in `requirements.txt`. They can therefore be installed by running:

```
pip install -r requirements.txt
```

## Configuration

All hyper-parameters used to configure the model are found and can be tweaked in `config.py`.

## Building and Running Model

Run `src/main.py`. If no `model.pkl` is found in the directory, a new model will be trained. Once a `model.pkl` is found, the model will be tested on chords listed under the testing directory specified earlier.
