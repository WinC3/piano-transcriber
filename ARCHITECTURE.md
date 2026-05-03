# Architecture: Piano Transcriber

## System Overview

Piano Transcriber uses the **Onsets and Frames** neural network architecture (Hawthorne et al., 2018, Google Magenta) to convert raw piano audio into MIDI notation. The system predicts which of 88 piano keys are pressed, when they're pressed, and how hard.

## Core Model: Three-Branch Architecture

The model has three prediction branches operating on the same feature extraction pipeline:

```
Input Audio (16kHz)
    ↓
Mel Spectrogram (229 bins, 30Hz–8kHz)
    ↓
CNN Feature Extraction (3 conv layers, 48 filters)
    ↓
    ├─→ Onset Detection Branch  → Sigmoid → Onset Probabilities
    ├─→ Frame Detection Branch  → Sigmoid → Frame Probabilities
    └─→ Velocity Regression     → ReLU   → Note Intensities
```

### Why Three Branches?

**Onset vs. Frame Detection**: These are fundamentally different problems:

- **Onsets** are transient events (sudden pressure on a key) — best detected by CNNs sensitive to spectral changes
- **Frames** are sustained notes (key held down) — require temporal modeling for polyphonic tracking
- Combining both improves F1 score by ~15% vs. either alone (compared to plain NN or CNN architectures)

**Velocity Estimation**: Predicts how hard each key was pressed (MIDI 1–127). Only computed where onsets occur (masked loss), preventing the model from learning useless velocity predictions for inactive keys.

## Technical Decisions

### 1. CNN Feature Extractor

- **3 convolutional layers** with (1,2) pooling pattern
- Reduces frequency resolution (important for discarding octave-level noise) while preserving temporal resolution
- Output: 48 channels of learned spectral features per time step

### 2. BiLSTM Temporal Modeling

- 128 hidden units in each direction (forward/backward)
- Captures polyphonic context: "if note A started, note B is less likely to start immediately after"
- Bidirectional processing allows the model to use future context when deciding about current frames

### 3. The `detach()` Trick (Critical Design)

```python
# Frame branch uses onset probabilities as input, but:
onset_probs = torch.sigmoid(onset_logits).detach()  # ← Stop gradients here
frame_input = torch.cat([lstm_output, onset_probs], dim=-1)
```

**Why?** Without `detach()`, frame loss gradients would backpropagate into the onset branch, corrupting onset predictions. With `detach()`, the frame branch can _use_ onset information without _training on_ onset loss.

### 4. Mel Spectrogram Design

- **229 mel bins** covering 30Hz–8kHz
- Covers full piano range (A0 at 27.5Hz to C8 at 4186Hz)
- Sufficient frequency resolution to distinguish adjacent semitones

## Training Strategy

- **Multi-task loss**:
  - Onset: Binary Cross-Entropy
  - Frame: Binary Cross-Entropy
  - Velocity: Mean Squared Error (masked by onset labels)
- **Data**: MAESTRO dataset (200+ hours of annotated piano recordings)
- **Chunking**: Train on 640-frame (~20 second) windows

## Inference Pipeline

For audio longer than the training window:

1. **Chunk audio** into overlapping 20-second windows (50% overlap)
2. **Process each chunk** independently through the model
3. **Average predictions** in overlapping regions to smooth boundaries
4. **Extract notes** by finding onset peaks and tracing sustained frames

This prevents boundary artifacts where notes might be cut off mid-prediction.

## Design Trade-offs

| Decision         | Pro                                                    | Con                                             |
| ---------------- | ------------------------------------------------------ | ----------------------------------------------- |
| Three branches   | Better performance, cleaner problem decomposition      | More parameters, complex loss weighting         |
| BiLSTM           | Temporal context, bidirectional                        | Slower than feedforward, requires full sequence |
| CNN + LSTM       | Leverages both local and temporal patterns             | No end-to-end optimization of feature learning  |
| Mel spectrograms | Human-perceptual frequency scale, lower dimensionality | Information loss vs. raw waveform               |

## Future Improvements

1. **Attention mechanisms**: Replace BiLSTM with Transformer for longer-range dependencies
2. **Multi-scale features**: Extract features at different time resolutions
3. **End-to-end waveform input**: Learn the spectrogram computation as part of the network
4. **Confidence scoring**: Output uncertainty estimates alongside predictions for more robust post-processing
