"""Tests for neural network model instantiation and forward pass."""

import pytest
import torch
from piano_transcriber.model.nn_models import OnsetsAndFrames


class TestOnsetsAndFramesModel:
    """Test the OnsetsAndFrames model architecture."""
    
    def test_model_instantiation(self):
        """Test that the model can be instantiated without errors."""
        model = OnsetsAndFrames()
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_model_forward_pass(self):
        """Test that the model can process a batch of mel spectrograms."""
        model = OnsetsAndFrames()
        model.eval()
        
        # Create dummy mel spectrogram input (batch_size=2, time_steps=640, mel_bins=229)
        batch_size = 2
        time_steps = 640
        mel_bins = 229
        
        dummy_input = torch.randn(batch_size, time_steps, mel_bins)
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        # Check that all expected outputs are present
        assert 'onset' in outputs
        assert 'frame' in outputs
        assert 'velocity' in outputs
        assert 'onset_probs' in outputs
        
        # Check output shapes (should be [batch_size, time_steps, 88])
        assert outputs['onset'].shape == (batch_size, time_steps, 88)
        assert outputs['frame'].shape == (batch_size, time_steps, 88)
        assert outputs['velocity'].shape == (batch_size, time_steps, 88)
        assert outputs['onset_probs'].shape == (batch_size, time_steps, 88)
    
    def test_model_on_gpu_if_available(self):
        """Test that model can be moved to GPU if available."""
        model = OnsetsAndFrames()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Verify model is on correct device
        param_device = next(model.parameters()).device
        assert str(param_device).startswith(device.split(':')[0])
