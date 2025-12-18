import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from polyphonic_pitch_detection import bin_to_note_name

def plot_cqt_with_labels(cqt_db, labels, sr, hop_length, fmin=27.5, bins_per_octave=12):
    """
    Plot CQT spectrogram with MIDI labels overlaid
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot CQT spectrogram
    img = librosa.display.specshow(cqt_db, 
                                  sr=sr, 
                                  hop_length=hop_length,
                                  x_axis='time', 
                                  y_axis='cqt_note',
                                  fmin=fmin,
                                  bins_per_octave=bins_per_octave,
                                  ax=ax1)
    ax1.set_title('CQT Spectrogram')
    fig.colorbar(img, ax=ax1, format='%+2.0f dB')
    
    # Plot MIDI labels (piano roll)
    time_axis = np.arange(labels.shape[0]) * hop_length / sr
    bin_axis = np.arange(labels.shape[1])
    
    ax2.imshow(labels.T,  # Transpose for correct orientation
               aspect='auto', 
               origin='lower',
               extent=[time_axis[0], time_axis[-1], bin_axis[0], bin_axis[-1]],
               cmap='Greys',  # Black and white for clarity
               alpha=0.8)
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Note Bin')
    ax2.set_title('MIDI Labels (Piano Roll)')
    ax2.set_yticks(np.arange(0, 88, 12))
    ax2.set_yticklabels([bin_to_note_name(i) for i in range(0, 88, 12)])

    print("\nMIDI labels (active notes):")
    active_bins = np.where(labels[0] == 1)[0]
    for bin_idx in active_bins:
        note_name = bin_to_note_name(bin_idx)
        print(f"  {note_name}")
    
    plt.tight_layout()
    plt.show()


data = np.load('parsed data/cleaned_unseparated_dataset.npz', mmap_mode='r')
ind = 500
for i in range(5):
    CQT_data = data['features'][ind:ind+2]
    CQT_data = CQT_data.T
    labels = data['labels'][ind:ind+2]
    print(CQT_data.shape, labels.shape)
    plot_cqt_with_labels(CQT_data, labels, sr=44100, hop_length=512)
    ind += 200