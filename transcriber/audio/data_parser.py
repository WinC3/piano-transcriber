import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np

import warnings

SEQUENCE_LENGTH = 640 # ~20 seconds at 16khz/512 hop

class PianoRollDataset(Dataset):
    def __init__(self, data_dir, sequence_length=SEQUENCE_LENGTH, is_validation=False):
        """
        data_dir: Path to the 'train' or 'validation' folder containing .pt files
        """
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.sequence_length = sequence_length
        self.is_validation = is_validation

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            data = torch.load(path)
        
        # Unpack
        # Note: We saved as half/bool to save disk space, convert back to float for training
        full_audio = data['audio'].float()
        full_onset = data['onset'].float()
        full_frame = data['frame'].float()
        full_velocity = data['velocity'].float()
        
        total_frames = full_audio.shape[0]
        
        # Random Crop
        if total_frames <= self.sequence_length:
            # Padding if file is shorter than sequence (rare in MAESTRO)
            pad_len = self.sequence_length - total_frames
            audio = torch.nn.functional.pad(full_audio, (0, 0, 0, pad_len))
            onset = torch.nn.functional.pad(full_onset, (0, 0, 0, pad_len))
            frame = torch.nn.functional.pad(full_frame, (0, 0, 0, pad_len))
            velocity = torch.nn.functional.pad(full_velocity, (0, 0, 0, pad_len))
        else:
            if self.is_validation:
                # Deterministic crop for validation (e.g., start)
                start = 0
            else:
                # Random crop for training
                max_start = total_frames - self.sequence_length
                start = np.random.randint(0, max_start)
                
            end = start + self.sequence_length
            
            audio = full_audio[start:end]
            onset = full_onset[start:end]
            frame = full_frame[start:end]
            velocity = full_velocity[start:end]

        return {
            "audio": audio,     # (T, F)
            "onset": onset,     # (T, 88)
            "frame": frame,     # (T, 88)
            "velocity": velocity # (T, 88)
        }