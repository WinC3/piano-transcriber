"""Tests for audio preprocessing pipeline."""

import pytest
import torch
import torchaudio
import tempfile
import os
from piano_transcriber.audio.data_preprocessing import compute_log_mel


def _create_temp_wav(duration=1.0, sample_rate=16000):
    """Create a temporary WAV file with a sine wave for testing."""
    num_samples = int(sample_rate * duration)
    t = torch.linspace(0, duration, num_samples)
    audio = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # (1, N)
    
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, audio, sample_rate)
    tmp.close()
    return tmp.name


class TestAudioPreprocessing:
    """Test audio preprocessing functions."""
    
    def test_compute_log_mel_output_shape(self):
        """Test that compute_log_mel produces correct output shape."""
        wav_path = _create_temp_wav(duration=1.0)
        try:
            mel_spec = compute_log_mel(wav_path)
            
            # Expected shape: (time_steps, 229)
            assert isinstance(mel_spec, torch.Tensor)
            assert mel_spec.shape[1] == 229, f"Expected 229 mel bins, got {mel_spec.shape[1]}"
            assert mel_spec.ndim == 2, f"Expected 2D tensor, got {mel_spec.ndim}D"
        finally:
            os.unlink(wav_path)
    
    def test_compute_log_mel_dtype(self):
        """Test that output is float32 (suitable for model input)."""
        wav_path = _create_temp_wav(duration=1.0)
        try:
            mel_spec = compute_log_mel(wav_path)
            assert mel_spec.dtype == torch.float32
        finally:
            os.unlink(wav_path)
    
    def test_compute_log_mel_different_durations(self):
        """Test that compute_log_mel handles different audio lengths."""
        for duration in [0.5, 1.0, 2.0]:
            wav_path = _create_temp_wav(duration=duration)
            try:
                mel_spec = compute_log_mel(wav_path)
                
                assert mel_spec.shape[1] == 229
                assert mel_spec.shape[0] > 0  # Should have at least some frames
            finally:
                os.unlink(wav_path)
