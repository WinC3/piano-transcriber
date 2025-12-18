"""
Piano Transcriber - Automatic Music Transcription
A PyTorch implementation of the "Onsets and Frames" model for piano transcription.
"""

__version__ = "0.1.0"
__author__ = "Winston Chan"

from .inference import PianoTranscriber, transcribe_file, get_latest_checkpoint

__all__ = ['PianoTranscriber', 'transcribe_file', 'get_latest_checkpoint']