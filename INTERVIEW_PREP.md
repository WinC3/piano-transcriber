# Interview Preparation: Piano Transcriber Project

## Executive Summary for Henry

> "I built an end-to-end machine learning system that converts raw piano audio into MIDI notation. It started as a research experiment and evolved into a production-ready tool with a CLI, desktop app, web platform, and Docker deployment."

---

## 1. How to Walk Through the Project (Recommended Flow)

### Opening (2-3 min)

- "This is a system that listens to piano recordings and outputs sheet-music-equivalent MIDI data. It uses a neural network architecture called 'Onsets and Frames' which solves polyphonic piano transcription — detecting which of 88 keys are pressed, when, and how hard."

### Part A: The Problem & Research Phase (10-15 min)

Walk through the **git history bottom-up** to show your iterative problem-solving:

1. **Initial approach** (`9f1f7a0` → `dbe96ef`): Started with classical signal processing (Harmonic Product Spectrum) — explain why you tried this first and why it failed for polyphonic music.
2. **Pivot to neural networks** (`b6863b7` → `6393bb7`): Recognized the problem requires learning, not hand-crafted features. Built data pipeline from MAESTRO dataset.
3. **Architecture evolution** (`bb8f77d`): Switched from simple regression NN to CNN-based model with dramatically better results — shows willingness to pivot based on evidence.
4. **Full model** (`6edccde`): Implemented "Onsets and Frames" — explain the multi-task architecture (onset detection + frame detection + velocity estimation).

### Part B: The Architecture Deep-Dive (15-20 min)

Open `piano_transcriber/model/nn_models.py` and explain:

- **Why three branches?** Onset detection is a different problem than sustained-note detection. The paper found that combining them produces better results than either alone.
- **The `detach()` trick** (line ~113): Frame branch uses onset probabilities but `detach()` prevents gradient backprop — prevents the frame loss from corrupting the onset detector. This is a key design decision from the paper.
- **CNN → LSTM pipeline**: CNN extracts local spectral features; BiLSTM captures temporal context for polyphonic note tracking.

### Part C: Software Engineering (15-20 min)

Show how you productionized the research:

1. **Package structure** (`piano_transcriber/`): Clean separation of concerns — audio processing, model, inference engine, CLI, GUI, web.
2. **Inference engine** (`inference.py`): Chunking strategy for variable-length audio with overlap + averaging to avoid boundary artifacts.
3. **Multiple interfaces**: CLI for batch processing, PyQt6 GUI for end users, FastAPI web app for deployment.
4. **Containerization**: Docker + nginx reverse proxy for production deployment.
5. **Distribution**: PyInstaller executables, pip-installable package, Docker images.

### Closing (5 min)

- What you'd improve: tests, CI/CD pipeline, model quantization for mobile.
- What you learned: balancing research exploration vs. shipping working software.

---

## 2. Things to Polish Before the Interview

### HIGH PRIORITY (Do these)

#### A. Repository Name

Your repo is called `project-one` which is generic. Consider renaming it to `piano-transcriber` on GitHub (Settings → Repository name). Your `pyproject.toml` already uses this name.

#### B. Remove `nn_pitch.py` from `piano_transcriber/`

There's a stray training script at `piano_transcriber/nn_pitch.py` that doesn't belong in the installable package. It should be at the project root only (for research use). This is a minor cleanup but shows you understand package structure.

#### C. Add a Tests Directory

Even just 2-3 basic tests show engineering maturity:

- Test that the model instantiates correctly
- Test that `compute_log_mel` produces correct output shape
- Test that note extraction from predictions works

You don't need full coverage — just demonstrate you know testing matters.

#### D. Add a `.github/workflows/` CI file

A simple GitHub Actions workflow that runs `pip install -e .` and your tests shows you understand modern dev practices. Even if the tests are minimal, the infrastructure matters.

#### E. Clean Up Commit Messages (Optional but Recommended)

Your early commits are fine for a personal project, but some are vague:

- `fix some bugs` → what bugs?
- `rename` → rename what?
- `refactor` → what was refactored?

**However**: Rewriting history has risks. If you do this, use `git rebase -i --root` carefully. The later commits (from `c6e62fe` onward) are already much better. I'd recommend **leaving history as-is** and instead being ready to narrate: "You can see my early commits are scrappier — this was exploratory research. As the project matured, so did my process."

### MEDIUM PRIORITY (Nice to have)

#### F. Add an Architecture Diagram to README

A simple ASCII or Mermaid diagram showing:

```
Audio → Mel Spectrogram → CNN → BiLSTM → [Onset/Frame/Velocity] → MIDI
```

#### G. Add a `ARCHITECTURE.md`

A short (1-page) document explaining the model design decisions. This shows you can communicate technical depth in writing.

#### H. Pin Dependency Versions

Your `pyproject.toml` uses `>=` for all deps. Adding a `requirements.txt` with pinned versions (or a lockfile) shows production awareness.

### LOW PRIORITY (Don't bother unless you have time)

#### I. The `output.mid` in Root

Remove this test artifact from the repo — it's committed output data.

#### J. `build/` Directory

This should likely be in `.gitignore`. Check if it's tracked.

---

## 3. Technical Questions You Should Be Ready For

### Architecture Questions

- **"Why did you choose this particular architecture?"**
  → The "Onsets and Frames" paper (Google Magenta, 2018) showed that separating onset detection from frame detection significantly improves transcription accuracy. Onsets are transient events (good for CNN), while sustained notes require temporal modeling (good for LSTM).

- **"Why BiLSTM instead of Transformer?"**
  → The original paper used BiLSTM and it works well for this sequence length (~20s chunks). Transformers would be an interesting experiment for longer sequences but add complexity without guaranteed improvement for this task.

- **"Explain the detach() on line 113"**
  → Frame branch concatenates onset probabilities to its input, but we `detach()` to stop gradients flowing back through the onset branch. Without this, the frame loss would corrupt onset predictions. The onset branch should only be trained by onset loss.

- **"Why 229 mel bins?"**
  → Covers 30Hz-8kHz range, which spans the entire piano (A0=27.5Hz to C8=4186Hz). 229 bins provides sufficient frequency resolution to distinguish adjacent semitones across the full range.

### Software Engineering Questions

- **"How do you handle audio longer than your training sequence?"**
  → Chunking with 50% overlap, then averaging predictions in overlapping regions. This prevents boundary artifacts where notes might be cut off.

- **"Why Docker for deployment?"**
  → Reproducible environment with correct PyTorch/audio library versions. CPU-only build keeps image smaller. Nginx proxy handles SSL, rate limiting, and static file serving.

- **"How would you scale this?"**
  → Queue-based architecture (Redis + Celery), multiple worker containers, model serving with TorchServe or Triton for GPU batching.

- **"What's your testing strategy?"**
  → (Be honest if you don't have tests yet) "Currently I validate against known MIDI files from the MAESTRO test set. I'd add unit tests for the preprocessing pipeline and integration tests that verify round-trip accuracy."

### Process Questions

- **"Walk me through a bug you fixed"**
  → The `fix some bugs` commit — be ready to recall what that was. If you can't remember, pick the `switch to cnn model` pivot and explain how you diagnosed the regression approach wasn't working.

- **"What would you do differently?"**
  → "I'd write tests earlier, use conventional commits from the start, and set up CI before building the application layers. I also learned that research code and production code should be separated from day one."

---

## 4. Framing This as a Software Engineering Project

Even though this is ML-heavy, emphasize the **software engineering** aspects:

| ML Aspect             | Software Engineering Spin                          |
| --------------------- | -------------------------------------------------- |
| Training loop         | "Iterative development with measurable metrics"    |
| Data pipeline         | "ETL system processing 200+ hours of audio data"   |
| Model architecture    | "System design with clear component interfaces"    |
| CLI/GUI/Web           | "Multi-platform deployment with shared core logic" |
| Docker                | "Infrastructure as code, reproducible deployments" |
| Package structure     | "Clean API design, separation of concerns"         |
| Checkpoint management | "State management and versioning"                  |

### Key Software Engineering Principles to Highlight:

1. **DRY**: Single `PianoTranscriber` class used by CLI, GUI, and web app
2. **Separation of concerns**: Model, audio processing, inference, UI are all independent modules
3. **Interface design**: Clean public API (`transcribe_audio()` → predictions → `predictions_to_midi()`)
4. **Progressive enhancement**: Started with CLI, added GUI, then web — each layer builds on the same core
5. **Production considerations**: Error handling, background threading, Docker health checks, graceful shutdown

---

## 5. Demo Strategy

If you can do a live demo:

1. Have a short piano audio file ready (10-15 seconds)
2. Run the CLI: `piano-transcriber sample.wav -o demo.mid`
3. Open the MIDI in any MIDI player to show it produced real music
4. This takes 30 seconds and is very impressive visually

If you can't run inference live (no GPU, model too slow on CPU):

- Have a pre-generated MIDI file ready
- Show the web UI interface
- Walk through the code instead

---

## 6. Questions to Ask Henry

Prepare 2-3 thoughtful questions:

- "What does the engineering team's development workflow look like?"
- "What's the most interesting technical challenge the team has faced recently?"
- "How does the team balance shipping quickly vs. code quality?"

---

## Quick Checklist Before the Interview

- [ ] Can you run `piano-transcriber --help` successfully?
- [ ] Do you have a sample audio file + pre-generated MIDI output ready?
- [ ] Can you explain the `detach()` pattern without looking at notes?
- [ ] Can you draw the architecture on a whiteboard (or describe it verbally)?
- [ ] Have you rehearsed the git-history walkthrough once out loud?
- [ ] Is the GitHub repo public and clean?
- [ ] Do you know your F1 scores / model performance numbers?
