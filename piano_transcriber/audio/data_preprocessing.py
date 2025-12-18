import os
import torch
import torchaudio
import pretty_midi
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
CSV_PATH = "MAESTRO Data\maestro-v1.0.0\maestro-v1.0.0.csv"
DATA_ROOT = "MAESTRO Data\maestro-v1.0.0" # Root folder where audio/midi folders are
OUTPUT_DIR = "processed_maestro"

# Audio Settings
SAMPLE_RATE = 16000
HOP_LENGTH = 512
N_MELS = 229
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 2048

def compute_log_mel(audio_path):
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        audio = resampler(audio)
    
    # Mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
        
    # Mel Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=WINDOW_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=MEL_FMIN,
        f_max=MEL_FMAX,
        power=1.0
    )
    
    mels = mel_transform(audio)
    log_mels = torch.log(mels + 1e-6)
    
    # Shape: (T, F)
    return log_mels.squeeze(0).transpose(0, 1)

def compute_labels(midi_path, num_frames):
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # Initialize labels
    onset_label = torch.zeros(num_frames, 88, dtype=torch.float32)
    frame_label = torch.zeros(num_frames, 88, dtype=torch.float32)
    velocity_label = torch.zeros(num_frames, 88, dtype=torch.float32)
    
    t_frame = HOP_LENGTH / SAMPLE_RATE
    
    for note in pm.instruments[0].notes:
        pitch_idx = note.pitch - 21
        if 0 <= pitch_idx < 88:
            # Convert time to frame indices
            start_frame = int(note.start / t_frame)
            end_frame = int(note.end / t_frame)
            
            # Clamp
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames - 1))
            
            # Onset
            onset_label[start_frame, pitch_idx] = 1.0
            velocity_label[start_frame, pitch_idx] = note.velocity / 128.0
            
            # Frame (sustain)
            frame_label[start_frame:end_frame, pitch_idx] = 1.0
            
    return onset_label, frame_label, velocity_label

def preprocess_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = pd.read_csv(CSV_PATH)
    
    print(f"Found {len(df)} files in MAESTRO CSV.")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        split = row['split'] # 'train', 'validation', or 'test'
        midi_rel_path = row['midi_filename']
        audio_rel_path = row['audio_filename']
        
        # Setup paths
        save_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(save_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(audio_rel_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}.pt")
        
        if os.path.exists(save_path):
            continue
            
        full_audio_path = os.path.join(DATA_ROOT, audio_rel_path)
        full_midi_path = os.path.join(DATA_ROOT, midi_rel_path)
        
        try:
            # 1. Compute Audio Features
            log_mels = compute_log_mel(full_audio_path)
            
            # 2. Compute Labels
            # Note: We pass the actual number of audio frames to ensure alignment
            onsets, frames, velocities = compute_labels(full_midi_path, log_mels.shape[0])
            
            # 3. Save as compressed dictionary
            torch.save({
                'audio': log_mels.half(), # Save as float16 to save space (optional)
                'onset': onsets.bool(),   # Save as bool to save space
                'frame': frames.bool(),
                'velocity': velocities.half()
            }, save_path)
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")

if __name__ == "__main__":
    preprocess_dataset()