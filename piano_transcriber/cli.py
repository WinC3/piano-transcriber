#!/usr/bin/env python3
"""
Command-line interface for piano transcription.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List

from .inference import PianoTranscriber, transcribe_file, get_latest_checkpoint


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Transcribe piano audio to MIDI using neural networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe single file to MIDI
  piano-transcriber input.wav -o output.mid
  
  # Transcribe to JSON format
  piano-transcriber input.wav -f json -o output.json
  
  # Batch process multiple files
  piano-transcriber *.wav -o /path/to/output/
  
  # Use specific model
  piano-transcriber input.wav -m model_epoch_2000.pth
  
  # Adjust detection sensitivity
  piano-transcriber input.wav --onset-threshold 0.3 --frame-threshold 0.4
        """)
    
    # Input/Output
    parser.add_argument(
        "input", 
        nargs="+", 
        help="Input audio file(s) or glob pattern"
    )
    parser.add_argument(
        "-o", "--output", 
        type=Path,
        help="Output file or directory (auto-generated if not specified)"
    )
    parser.add_argument(
        "-f", "--format", 
        choices=["midi", "json"],
        default="midi",
        help="Output format (default: midi)"
    )
    
    # Model selection
    parser.add_argument(
        "-m", "--model", 
        type=Path,
        help="Path to model checkpoint (auto-detects latest if not specified)"
    )
    
    # Transcription parameters
    parser.add_argument(
        "--onset-threshold", 
        type=float, 
        default=0.5,
        help="Onset detection threshold (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--frame-threshold", 
        type=float, 
        default=0.5,
        help="Frame detection threshold (0.0-1.0, default: 0.5)"
    )
    
    # Device selection
    parser.add_argument(
        "--device", 
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for inference (default: auto)"
    )
    
    # Verbose output
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    
    return parser


def resolve_input_files(input_patterns: List[str]) -> List[Path]:
    """Resolve input file patterns to actual file paths."""
    import glob
    
    files = []
    for pattern in input_patterns:
        if "*" in pattern or "?" in pattern:
            # Handle glob patterns
            matches = glob.glob(pattern)
            if not matches:
                print(f"Warning: No files match pattern '{pattern}'")
                continue
            files.extend(Path(f) for f in matches)
        else:
            # Single file
            file_path = Path(pattern)
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                sys.exit(1)
            files.append(file_path)
    
    # Filter for audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    audio_files = [f for f in files if f.suffix.lower() in audio_extensions]
    
    if not audio_files:
        print("Error: No valid audio files found")
        sys.exit(1)
    
    return audio_files


def determine_output_path(input_file: Path, output_arg: Path, format_type: str, batch_mode: bool) -> Path:
    """Determine output path for a given input file."""
    extension = ".mid" if format_type == "midi" else ".json"
    
    if output_arg is None:
        # Auto-generate: input_file.mid or input_file.json
        return input_file.with_suffix(extension)
    
    if batch_mode:
        # Multiple inputs: output_arg should be directory
        if not output_arg.is_dir():
            output_arg.mkdir(parents=True, exist_ok=True)
        return output_arg / (input_file.stem + extension)
    else:
        # Single input: output_arg is the exact output file
        return output_arg


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate thresholds
    if not (0.0 <= args.onset_threshold <= 1.0):
        print("Error: onset-threshold must be between 0.0 and 1.0")
        sys.exit(1)
    if not (0.0 <= args.frame_threshold <= 1.0):
        print("Error: frame-threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Resolve input files
    input_files = resolve_input_files(args.input)
    batch_mode = len(input_files) > 1
    
    if args.verbose:
        print(f"Found {len(input_files)} audio file(s) to process")
    
    # Determine model path
    model_path = args.model
    if model_path is None:
        model_path = get_latest_checkpoint()
        if model_path is None:
            print("Error: No model checkpoint found. Please specify --model or ensure a trained model exists.")
            sys.exit(1)
        if args.verbose:
            print(f"Using auto-detected model: {model_path}")
    elif not Path(model_path).exists():
        print(f"Error: Model checkpoint not found: {model_path}")
        sys.exit(1)
    
    # Set device
    device = None if args.device == "auto" else args.device
    
    # Initialize transcriber
    try:
        transcriber = PianoTranscriber(str(model_path), device=device)
        if args.verbose:
            print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Process files
    success_count = 0
    for input_file in input_files:
        try:
            if args.verbose:
                print(f"Processing: {input_file}")
            
            # Determine output path
            output_path = determine_output_path(input_file, args.output, args.format, batch_mode)
            
            # Transcribe
            if args.format == "midi":
                transcriber.transcribe_audio(
                    input_file, 
                    onset_threshold=args.onset_threshold,
                    frame_threshold=args.frame_threshold
                )
                predictions = transcriber.transcribe_audio(input_file)
                transcriber.predictions_to_midi(predictions, output_path)
            else:  # json
                predictions = transcriber.transcribe_audio(
                    input_file,
                    onset_threshold=args.onset_threshold, 
                    frame_threshold=args.frame_threshold
                )
                notes = transcriber.predictions_to_json(predictions)
                
                # Save JSON
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(notes, f, indent=2)
                
                if args.verbose:
                    print(f"JSON saved to {output_path}")
            
            success_count += 1
            if not args.verbose:
                print(f"✓ {input_file} -> {output_path}")
            
        except Exception as e:
            print(f"✗ Error processing {input_file}: {e}")
            continue
    
    # Summary
    if batch_mode or args.verbose:
        print(f"\nProcessed {success_count}/{len(input_files)} files successfully")
    
    if success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()