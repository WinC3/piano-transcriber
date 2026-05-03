"""Tests for the PianoTranscriber inference engine."""

import pytest
import torch
from piano_transcriber.inference import PianoTranscriber


class TestPianoTranscriber:
    """Test the PianoTranscriber class."""
    
    def test_transcriber_instantiation(self):
        """Test that PianoTranscriber can be instantiated without a checkpoint."""
        transcriber = PianoTranscriber()
        
        assert transcriber is not None
        assert transcriber.device in ["cpu", "cuda"]
        assert transcriber.model_loaded is False
    
    def test_transcriber_audio_constants(self):
        """Test that transcriber has correct audio processing constants."""
        transcriber = PianoTranscriber()
        
        assert transcriber.sample_rate == 16000
        assert transcriber.hop_length == 512
        assert transcriber.sequence_length == 640
    
    def test_transcriber_device_selection(self):
        """Test that device is auto-selected correctly."""
        transcriber = PianoTranscriber()
        
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert transcriber.device == expected_device
    
    def test_transcriber_with_explicit_device(self):
        """Test that transcriber respects explicit device selection."""
        transcriber = PianoTranscriber(device="cpu")
        assert transcriber.device == "cpu"


class TestPianoTranscriberInference:
    """Test inference pipeline (requires checkpoint)."""
    
    @pytest.mark.skip(reason="Requires a model checkpoint file")
    def test_transcriber_with_checkpoint(self, checkpoint_path=None):
        """Test loading a checkpoint and running inference.
        
        To run this test, provide a valid checkpoint path.
        """
        if checkpoint_path is None:
            pytest.skip("No checkpoint provided")
        
        transcriber = PianoTranscriber(checkpoint_path=checkpoint_path)
        assert transcriber.model_loaded is True
        assert transcriber.model is not None
