import config

from mido import MidiFile
import os
import numpy as np

def __filter_message__(message):
    allowed_types = ['note_on']
    # hopefully getting rid of correct channel (aiming for RHYTHM)
    channels_to_remove = [9]
    return ((message.type in allowed_types) and (message.channel not in channels_to_remove))

def __filtered_tracks__(track):
    return list(filter(__filter_message__, track))

def __include_track__(track):
    return (len(__filtered_tracks__(track)) > 0)

def __import_midi_file__(name):
    midi_file = MidiFile(name, clip=True)  
    # keep tracks with actual notes in them
    return [__filtered_tracks__(i) for i in midi_file.tracks if __include_track__(i)]

# transcribe midi data
# format: [note, time, channel]
def __message_to_arr__(message):
    return [message.note, message.time, message.channel]

# get list of all file names
def __get_files__(directory, ext = '.mid'):
    filenames = []
    for dirpath, dirs, files in os.walk(directory):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            if fname.endswith(ext):
                filenames.append(fname)
    return filenames

# import all midi chords and create tags
def __import_chords__(names):
    chord_dict = {}
    for name in names:
        chord = name[name.rfind(' ')+1:name.find('.')]
        midi = __import_midi_file__(name)
        chord_dict[chord] = [[__message_to_arr__(m) for m in t] for t in midi]

    draft = __extract_notes__(chord_dict)
    return draft

def get_base_data(allow_dups=True):
    draft = __import_chords__(__get_files__(config.TRAIN_DIR))
    return __create_answer_keys__(draft, allow_dups)

def create_train_data():
    chord_notes, answers = get_base_data(False)

    y = np.array(list(chord_notes.keys()))
    X = np.array([np.asarray(chord_notes[i]) for i in chord_notes])

    return X, y

def create_test_data():
    chord_notes, answers = get_base_data()

    y = np.array(list(chord_notes.keys()))
    X = np.array([np.asarray(chord_notes[i]) for i in chord_notes])

    return X, y, chord_notes, answers

def __create_answer_keys__(draft, allow_dups=True):
    chord_notes = dict()
    tmp = []
    for key, val in draft.items():
        tup = tuple(val)
        if allow_dups or (tup not in tmp):
            tmp.append(tup)
            chord_notes[key] = tup
    answers = dict()
    for key, val in draft.items():
        tup = tuple(val)
        if tup in answers:
            answers[tup].append(key)
        else:
            answers[tup] = [key]

    return chord_notes, answers

def __extract_notes__(chords):
    # format midi data (extract just notes)
    draft = {}
    for i in chords:
        tmp = np.array(chords[i][0])[:,0:1]
        tmp = np.reshape(tmp, (len(tmp))) % 12
        draft[i] = __format_chord__(tmp)

    return draft

# format midi chord to 12 note np array
def __format_chord__(tmp):
    result = np.zeros((12,))
    for i in tmp:
        result[i] = 1
    return result.astype(int)

