#!/usr/bin/env python3
"""
Prepare web app for deployment by copying required dependencies.
This script makes the web app self-contained for platforms like Vercel.
"""

import shutil
import os
from pathlib import Path

def prepare_web_deployment():
    """Copy all required dependencies into the web folder."""
    
    # Get paths
    project_root = Path(__file__).parent
    web_dir = project_root / "piano_transcriber" / "web"
    piano_transcriber_dir = project_root / "piano_transcriber"
    
    # Ensure web directory exists
    web_dir.mkdir(exist_ok=True)
    
    # Create subdirectories in web folder
    dirs_to_create = ["model", "audio", "midi"]
    for dir_name in dirs_to_create:
        (web_dir / dir_name).mkdir(exist_ok=True)
    
    # Copy model files
    model_source = piano_transcriber_dir / "model"
    model_dest = web_dir / "model"
    if model_source.exists():
        print(f"Copying model files: {model_source} -> {model_dest}")
        if model_dest.exists():
            shutil.rmtree(model_dest)
        shutil.copytree(model_source, model_dest)
    
    # Copy audio processing
    audio_source = piano_transcriber_dir / "audio"
    audio_dest = web_dir / "audio"
    if audio_source.exists():
        print(f"Copying audio files: {audio_source} -> {audio_dest}")
        if audio_dest.exists():
            shutil.rmtree(audio_dest)
        shutil.copytree(audio_source, audio_dest)
    
    # Copy MIDI processing
    midi_source = piano_transcriber_dir / "midi"
    midi_dest = web_dir / "midi"
    if midi_source.exists():
        print(f"Copying MIDI files: {midi_source} -> {midi_dest}")
        if midi_dest.exists():
            shutil.rmtree(midi_dest)
        shutil.copytree(midi_source, midi_dest)
    
    # Copy inference.py
    inference_source = piano_transcriber_dir / "inference.py"
    inference_dest = web_dir / "inference.py"
    if inference_source.exists():
        print(f"Copying inference.py: {inference_source} -> {inference_dest}")
        shutil.copy2(inference_source, inference_dest)
    
    # Create __init__.py files
    init_files = [
        web_dir / "__init__.py",
        web_dir / "model" / "__init__.py",
        web_dir / "audio" / "__init__.py",
        web_dir / "midi" / "__init__.py"
    ]
    
    for init_file in init_files:
        if not init_file.exists():
            init_file.write_text("")
            print(f"Created: {init_file}")
    
    print("✅ Web deployment preparation complete!")
    print(f"📁 Web app is now self-contained in: {web_dir}")
    print("🚀 Ready for Vercel deployment from piano_transcriber/web/ folder")

if __name__ == "__main__":
    prepare_web_deployment()