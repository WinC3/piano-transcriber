# Piano Transcription Neural Network - AI Agent Instructions

## Project Overview

This is a PyTorch implementation of the "Onsets and Frames" model for automatic music transcription (AMT), specifically designed for polyphonic piano music. The core neural network predicts onset times, frame activations, and velocities for 88 piano keys using the MAESTRO dataset.

**Project Roadmap & Current Status**:

**Phase 1: Core Infrastructure** ✅ COMPLETED
- ✅ **Inference Pipeline**: Production-ready transcription engine ([piano_transcriber/inference.py](piano_transcriber/inference.py))
- ✅ **CLI Tool**: Command-line interface for batch processing ([piano_transcriber/cli.py](piano_transcriber/cli.py))
- ✅ **Package Distribution**: Installable via GitHub (`pip install git+https://github.com/winc3/project-one.git`)

**Phase 2: Desktop Application** ✅ COMPLETED
- ✅ **Desktop GUI**: PyQt6-based desktop application ([piano_transcriber/gui/](piano_transcriber/gui/))
- ✅ **Standalone Executables**: PyInstaller-built binaries for Windows/Mac/Linux
- ✅ **GitHub Releases**: Downloadable applications via GitHub Releases

**Phase 3: Web Platform** ✅ COMPLETED
- ✅ **FastAPI Backend**: Complete web API for transcription ([piano_transcriber/web/app.py](piano_transcriber/web/app.py))
- ✅ **Web UI**: Modern browser-based interface with drag-and-drop ([piano_transcriber/web/templates/](piano_transcriber/web/templates/))
- ✅ **Docker Deployment**: Container orchestration with nginx proxy ([Dockerfile.web](Dockerfile.web), [docker-compose.yml](docker-compose.yml))
- ✅ **Production Ready**: Health checks, monitoring, and deployment documentation

### Research Components: Neural network training and research tools remain available for advanced users

**Next Phase: Optimization & Advanced Features**
- 🔲 **Mobile App**: React Native or Flutter mobile application
- 🔲 **GPU Cloud Deployment**: AWS/Azure GPU instance deployment guides  
- 🔲 **Real-time Processing**: WebSocket-based live audio transcription
- 🔲 **Advanced Export**: MusicXML, LilyPond notation export options
- 🔲 **Batch Processing**: Web-based batch file processing interface

## Architecture Understanding

### Core Model Structure ([nn_models.py](nn_models.py))
- **OnsetsAndFrames**: Main model with three prediction branches (onset, frame, velocity)
- **AcousticModel**: Shared CNN feature extractor with 3 conv layers, (1,2) pooling pattern
- **Critical Pattern**: Frame branch uses `onset_probs.detach()` to prevent gradient backprop to onset weights
- **Input Shape**: Mel spectrograms (B, T, 229) → (B, 1, T, 229) for CNN processing
- **Output**: 88-dimensional predictions matching piano key range (A0-C8, MIDI 21-108)

### Data Pipeline ([data_preprocessing.py](data_preprocessing.py), [data_parser.py](data_parser.py))
- **MAESTRO Dataset**: Expects CSV at `MAESTRO Data/maestro-v1.0.0/maestro-v1.0.0.csv`
- **Audio Processing**: 16kHz, 512 hop length, 229 mel bins (30Hz-8kHz)
- **Storage Optimization**: Uses `.half()` and `.bool()` to compress `.pt` files
- **SEQUENCE_LENGTH**: 640 frames (~20 seconds) for training chunks
- **Train/Val Split**: Deterministic crops for validation, random for training

### Training Workflow ([nn_pitch.py](nn_pitch.py))
- **Interactive Training**: Manual epoch/LR control via console input
- **Checkpoint Pattern**: `checkpoints/model_epoch_N.pth` with full state dict
- **Multi-task Loss**: BCE for onset/frame + MSE for velocity (masked by onsets)
- **Gradient Clipping**: 3.0 norm to prevent exploding gradients
- **Metrics**: Frame-wise P/R/F1 for onset and frame branches

### CLI Tool Architecture
- **Model Loading**: Load trained checkpoint for inference-only mode
- **Audio Processing Pipeline**: File input → mel spectrogram → model prediction → output format
- **Argument Parsing**: Support for input files, output formats (MIDI, JSON, CSV), model selection
- **Batch Processing**: Handle multiple files with progress tracking
- **Output Formats**: MIDI file generation, timestamped note data, confidence scores

### Desktop App Architecture (Next Phase)
- **GUI Framework**: PyQt6 (preferred) or Tkinter for cross-platform compatibility
- **File Operations**: Drag-and-drop audio file loading with progress tracking
- **Real-time Display**: Piano roll visualization of transcription results
- **Audio Playback**: Optional synchronized playback with transcription overlay
- **Export Options**: Save transcriptions in MIDI/JSON formats with file dialogs
- **Model Management**: Auto-detect bundled model, with option to load custom checkpoints
- **Threading**: Non-blocking inference to maintain UI responsiveness
- **Error Handling**: User-friendly error dialogs and status messages

### Web Platform Architecture ✅ COMPLETED
- **FastAPI Backend**: RESTful API endpoints for file upload and transcription
- **Modern Frontend**: HTML/JavaScript interface with drag-and-drop, progress tracking, and note preview
- **Async Processing**: Non-blocking transcription with job tracking and status updates
- **Docker Deployment**: Complete container orchestration with nginx reverse proxy
- **Production Features**: Health checks, file cleanup, error handling, and mobile-responsive design
- **Multiple Deployment Options**: Local development, Docker Compose, cloud deployment documentation

## Essential Constants & Configuration

```python
# Audio processing (data_preprocessing.py)
SAMPLE_RATE = 16000
HOP_LENGTH = 512
N_MELS = 229
SEQUENCE_LENGTH = 640  # ~20 seconds

# Training (nn_pitch.py)
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
default_lr = 0.0006

# Model architecture (nn_models.py)
model_complexity = 48  # CNN filters
lstm_units = 128       # BiLSTM hidden size
output_features = 88   # Piano keys
```

## Critical Development Patterns

### Checkpoint Management
- Always use relative paths: `checkpoints/model_epoch_N.pth`
- Resume training by loading both model and optimizer state
- Extract epoch number from filename for continuation

### Data Format Expectations
- Preprocessed data: `{'audio': half, 'onset': bool, 'frame': bool, 'velocity': half}`
- Runtime conversion: `.float()` for training, compressed types for storage
- Directory structure: `processed_maestro/{train,validation,test}/`

### Model Input/Output Contract
```python
# Input: (batch_size, time_steps, mel_bins=229)
outputs = model(audio_features)
# Returns: {'onset': logits, 'frame': logits, 'velocity': logits, 'onset_probs': sigmoid}
```

### Loss Calculation Pattern
```python
loss_onset = bce_loss(outputs['onset'], onset_label)
loss_frame = bce_loss(outputs['frame'], frame_label) 
vel_loss_raw = mse_loss(outputs['velocity'], velocity_label)
onset_mask = (onset_label == 1).float()  # Only compute velocity where onsets occur
loss_velocity = (vel_loss_raw * onset_mask).sum() / (onset_mask.sum() + 1e-6)
```

### CLI Tool Patterns
```python
# Model loading for inference
model = OnsetsAndFrames().to(device)
checkpoint = torch.load('checkpoints/model_epoch_N.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Audio processing pipeline
audio_features = compute_log_mel(audio_path)  # Reuse from data_preprocessing
with torch.no_grad():
    outputs = model(audio_features.unsqueeze(0))
    predictions = torch.sigmoid(outputs['onset']) > threshold
```

### Desktop App State Management
```python
# Separate UI thread from model inference
class TranscriptionApp:
    def __init__(self):
        self.model = None  # Lazy load
        self.current_audio = None
        self.transcription_results = None
    
    def load_model_async(self, checkpoint_path):
        # Background model loading to prevent UI freezing
    
    def process_audio_file(self, file_path):
        # Non-blocking audio processing with progress callbacks
```

## Directory Conventions

### Core Components
- `checkpoints/`: Model states (extensive - 800+ epochs saved)
- `processed_maestro/`: Preprocessed .pt files by train/val/test split  
- `MAESTRO Data/`: Original MAESTRO dataset (ignored by git)
- `NN Graphs/`, `NN Saved Models/`: Visualization and export directories
- `sample audio/`, `SampleAudioWav/`: Test audio files

### Production Components
- ✅ `piano_transcriber/inference.py`: Shared inference engine 
- ✅ `piano_transcriber/cli.py`: Command-line tool implementation
- ✅ `piano_transcriber/gui/`: Desktop application code (PyQt6-based GUI)
- ✅ `piano_transcriber/web/`: FastAPI backend with modern web interface
- ✅ `dist/`: Built executables and distributions
- ✅ `examples/`: Sample usage scripts and demonstrations

## Development Workflow

### Research/Training Workflow
1. **Data Preparation**: Run [data_preprocessing.py](data_preprocessing.py) to create `.pt` files from MAESTRO
2. **Training**: Run [nn_pitch.py](nn_pitch.py) with interactive epoch control
3. **Visualization**: Use [data_visualizer.py](data_visualizer.py) for CQT/label plotting (depends on `polyphonic_pitch_detection` module)

### CLI Tool Development ✅ COMPLETED
1. ✅ **Inference Module**: Production-ready inference engine (`piano_transcriber/inference.py`)
2. ✅ **Audio I/O**: File loading with same preprocessing pipeline as training
3. ✅ **Output Generation**: MIDI and JSON output formats with proper note extraction
4. ✅ **CLI Interface**: Full `argparse` interface with batch processing, thresholds, device selection

### Desktop App Development ✅ COMPLETED
1. ✅ **GUI Framework**: PyQt6-based cross-platform application
2. ✅ **Core Integration**: Uses existing `PianoTranscriber` class from inference module
3. ✅ **UI Components**: File browser, progress bars, drag-and-drop, settings panel
4. ✅ **Threading**: `QThread` for non-blocking operations and UI responsiveness
5. ✅ **File Operations**: Drag-and-drop audio file loading with progress tracking
6. ✅ **Export Options**: Save transcriptions in MIDI/JSON formats

### Distribution Strategy ✅ COMPLETED
1. ✅ **PyInstaller Packaging**: Create standalone executables for Windows/Mac/Linux
2. ✅ **GitHub Releases**: Automated builds and downloadable applications
3. ✅ **Model Bundling**: Include optimized checkpoint in application package
4. ✅ **Documentation**: User guides and installation instructions

### Web Platform Development ✅ COMPLETED
1. ✅ **FastAPI Setup**: Complete web API with file upload, async processing, and download endpoints
2. ✅ **Modern Frontend**: Responsive HTML/CSS/JS interface with drag-and-drop and real-time updates
3. ✅ **Docker Infrastructure**: Full containerization with nginx proxy and production configuration
4. ✅ **Development Tools**: Local development server with auto-reload and comprehensive documentation
5. ✅ **Production Ready**: Health checks, error handling, file cleanup, and cloud deployment guides

## Key Dependencies & Imports
- **Core ML**: `torch`, `torchaudio`, `torch.utils.data.DataLoader`
- **Data Processing**: `pretty_midi`, `pandas`, `tqdm`
- **Visualization**: `librosa`, `matplotlib`, custom `polyphonic_pitch_detection`
- **Cross-module**: Models import from `nn_models`, datasets from `data_parser`
- **CLI Tools**: `argparse`, `pathlib`, `json`
- **Desktop App**: `tkinter`/`PyQt`, `pygame`/`pyaudio`, threading modules
- **Output Formats**: `mido` for MIDI generation, `numpy` for data export

## Important Gotchas

### Core System
- **MAESTRO Path**: Uses Windows-style backslashes in hardcoded paths
- **Memory Management**: Data stored as half/bool precision - always convert to float for training
- **Gradient Flow**: Frame branch intentionally detaches onset gradients
- **Validation Determinism**: Validation uses fixed crops (start=0) vs random training crops
- **Interactive Training**: No automated stopping - requires manual intervention

### Production Tools
- **Model Loading**: CLI/GUI tools need inference-only loading (no optimizer state)
- **Audio Compatibility**: Ensure same preprocessing pipeline (16kHz, 512 hop, 229 mels)
- **Threading**: Desktop app requires careful threading for UI responsiveness
- **Path Handling**: Use `pathlib` for cross-platform compatibility in new components
- **Error Handling**: Production tools need graceful failure modes and user-friendly error messages