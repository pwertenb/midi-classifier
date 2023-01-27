from mido import MidiFile
import os

def __filter_message__(message):
    allowed_types = ['note_on']
    # hopefully getting rid of correct channel (aiming for RHYTHM)
    channels_to_remove = [9]
    return ((message.type in allowed_types) and (message.channel not in channels_to_remove))

def __filtered_tracks__(track):
    return list(filter(__filter_message__, track))

def __include_track__(track):
    return (len(__filtered_tracks__(track)) > 0)

def import_midi_file(name):
    midi_file = MidiFile(name, clip=True)  
    # keep tracks with actual notes in them
    return [__filtered_tracks__(i) for i in midi_file.tracks if __include_track__(i)]

# transcribe midi data
# format: [note, time, channel]
def __message_to_arr__(message):
    return [message.note, message.time, message.channel]

# get list of all file names
def get_files(directory, ext = '.mid'):
    filenames = []
    for dirpath, dirs, files in os.walk(directory):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            if fname.endswith(ext):
                filenames.append(fname)
    return filenames

# import all midi chords and create tags
def import_chords(names):
    chord_dict = {}
    longest = 0
    for name in names:
        chord = name[name.rfind(' ')+1:name.find('.')]
        midi = import_midi_file(name)
        chord_dict[chord] = [[__message_to_arr__(m) for m in t] for t in midi]
        if len(chord_dict[chord][0]) > longest:
            longest = len(chord_dict[chord][0])
    return chord_dict, longest
