import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import mido

import os

csv_metadata_path = "MAESTRO Data\maestro-v1.0.0\maestro-v1.0.0.csv"

def load_data(n_samples=None, shuffle=False, delta_t=0.02):
    metadata_df = pd.read_csv(csv_metadata_path)
    metadata_array = metadata_df.values  # to np array

    print(metadata_array)

    if shuffle:
        np.random.shuffle(metadata_array)
        print(metadata_array)

    if n_samples is not None:
        metadata_array = metadata_array[:n_samples]

    data, labels = [], []
    
    for sample in metadata_array:
        audio_path = os.path.normpath("MAESTRO Data/maestro-v1.0.0/" + sample[5])
        midi_path = os.path.normpath("MAESTRO Data/maestro-v1.0.0/" + sample[4])

        print(f"Audio path: {audio_path}")
        print(f"MIDI path: {midi_path}")

        y, sr = load_audio(audio_path)
        if y is None or sr is None:
            continue

        # CQT params
        hop_length = int(sr * delta_t) # 512 or power of 2 for comp efficiency
        n_bins = 88 # 88 keys on piano
        bins_per_octave = 12 # semitones
        fmin = librosa.note_to_hz('A0')

        # CQT
        C = librosa.cqt(y, sr=sr, hop_length=hop_length,
                        n_bins=n_bins, bins_per_octave=bins_per_octave,
                        fmin=fmin)

        # to dB
        C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)

        # plot
        plt.figure(figsize=(12, 5))
        librosa.display.specshow(C_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='cqt_note')
        plt.colorbar(label='dB')
        plt.title("Constant-Q Transform (CQT) Spectrogram")
        plt.show()

        data.append(C_db.T)

        mid = load_midi(midi_path)
        if mid is None:
            continue

        # Further processing can be done here
        # For example, compute CQT, extract features, etc.
        # For now, just print the lengths
        print(f"Audio length: {len(y)/sr:.2f} seconds")
        print(f"MIDI length: {mid.length:.2f} seconds")

    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = [], [], [], [], [], []

    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def load_midi(file_path):
    mid = mido.MidiFile(file_path)
    return mid

load_data(n_samples=1)

file_path = os.path.normpath("MAESTRO Data/maestro-v1.0.0/2017/MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--4.midi")

def analyze_midi(file_path):
    mid = mido.MidiFile(file_path)
    
    print(f"Format: {mid.type}, Ticks per beat: {mid.ticks_per_beat}")
    print(f"Length: {mid.length} seconds")
    print(f"Contains {len(mid.tracks)} tracks")
    
    for i, track in enumerate(mid.tracks):
        print(f"\nTrack {i}: {track.name if hasattr(track, 'name') else 'Unnamed'}")
        note_count = sum(1 for msg in track if msg.type in ['note_on', 'note_off'])
        print(f"  Messages: {len(track)}, Note events: {note_count}")

analyze_midi(file_path)
