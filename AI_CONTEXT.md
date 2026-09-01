# AI_CONTEXT.md — Verified Repository State (Phase 0 Audit)

**Last Updated:** 2026-08-23  
**Status:** Verified via empirical audit of filesystem, dependencies, git history, and execution test runs.

---

## 1. Executive Summary

This document serves as the single authoritative source of truth regarding the current codebase, dataset contents, execution status, and environment setup for **`my_elephant_project`**. It resolves the ambiguities outlined in Phase 0 of `DEVELOPMENT_PLAN.md`.

---

## 2. Directory & File Layout Audit

- **Root Directory:** `C:\Users\hakki\MACONFLIC`
  - `DEVELOPMENT_PLAN.md`: Authoritative 4-month development plan and roadmap.
  - `AI_CONTEXT.md`: This file.
  - `my_elephant_project/`: Primary project workspace directory.
- **Git Branch:** `main` (clean working tree, up to date with `origin/main`).

### Verified Active Codebase Files (`my_elephant_project/`):
- `requirements.txt`: Specifies dependencies (`torch>=2.0.0`, `torchaudio>=2.0.0`, `librosa>=0.10.0`, `numpy>=1.24.0`, `scikit-learn>=1.2.0`, `matplotlib>=3.7.0`, `tqdm>=4.65.0`).
- `.gitignore`: Configured to exclude raw audio files (`.wav`, `.mp3`, `.flac`, `.ogg`), PyTorch models (`*.pt`, `*.pth`), and Python caches.
- `reference_repos/README.md`: Informational file for reference code.
- `src/preprocess.py`: Contains `AudioPreprocessor` (resampling to 22.05kHz, 3.0s padding/cropping, Log-Mel Spectrogram calculation) and `ElephantDataset` (PyTorch `Dataset` wrapper supporting `'detector'` and `'classifier'` modes).
- `src/train_detector.py`: Implements `ElephantDetectorCNN` (3 Conv blocks + BatchNorm + Adaptive AvgPool + Linear classifier) and binary training loop.
- `src/train_classifier.py`: Implements `ElephantCallClassifierCNN` (3 Conv blocks with higher channels + Dropout + Linear classifier) and 3-class call classification training loop.

---

## 3. Dataset Audit & Provenance

- **Canonical Dataset Location:** `my_elephant_project/dataset/`
- **Data Status:** **REAL AUDIO** (NOT dummy spectrograms or synthetic pink noise).
- **Total Audio Clips:** **579 `.wav` files**.

### Class Breakdown:

| Category | Subcategory | File Count | Audio Properties | Provenance |
|---|---|---|---|---|
| **Elephant (Target)** | `elephant/roar` | 90 `.wav` files | 44.1 kHz, ~6.0s, 32-bit float | Real elephant vocalizations |
| | `elephant/rumble` | 87 `.wav` files | 44.1 kHz, ~6.0s, 32-bit float | Real elephant vocalizations |
| | `elephant/trumpet` | 82 `.wav` files | 48.0 kHz, ~6.0s, 32-bit float | Real elephant vocalizations |
| **Non-Elephant (Background)** | `non_elephant/animals` | 120 `.wav` files | 44.1 kHz, 5.0s, 16-bit PCM | ESC-50 animal sound subset |
| | `non_elephant/human` | 120 `.wav` files | 44.1 kHz, 5.0s, 16-bit PCM | ESC-50 human sound subset |
| | `non_elephant/rain` | 40 `.wav` files | 44.1 kHz, 5.0s, 16-bit PCM | ESC-50 rain sound subset |
| | `non_elephant/wind` | 40 `.wav` files | 44.1 kHz, 5.0s, 16-bit PCM | ESC-50 wind sound subset |

**Totals:**
- Binary Detector Target Class (`elephant`): **259 audio files**
- Binary Detector Non-Target Class (`non_elephant`): **320 audio files**
- Multi-Class Call Classifier Dataset (`trumpet` / `roar` / `rumble`): **259 audio files**

---

## 4. Execution & Environment Audit

- **Python Runtime:** Python 3.11+ on Windows x64.
- **Installed Packages:**
  - `torch`: 2.1.0+cpu (Installed & working)
  - `librosa`: 0.10.1 (Installed & working)
  - `numpy`: 1.26.4 (Installed & working)
  - `scipy`: 1.11.4 (Installed & working)
  - `soundfile`: 0.12.1 (Installed & working)
  - `scikit-learn`: 1.8.0 (Installed & working)
  - `matplotlib`: 3.7.2 (Installed & working)
  - `tqdm`: 4.66.1 (Installed & working)
- **Missing Packages:** `torchaudio` is **NOT installed** in the active environment.
- **Fallback Verification:** `src/preprocess.py` gracefully detects that `torchaudio` is missing (`HAS_TORCHAUDIO = False`) and automatically falls back to `librosa` and `soundfile` for audio loading and STFT/Mel calculation.
- **Script Execution Verification:**
  - `preprocess.py`: Successfully loads raw audio files and generates tensors of shape `torch.Size([1, 64, 130])` (1 channel, 64 mel bins, 130 time steps).
  - `train_detector.py`: Successfully initializes dataset, creates train/val split, instantiates `ElephantDetectorCNN`, and runs forward/backward training passes on real CPU data.
  - `train_classifier.py`: Successfully instantiates `ElephantCallClassifierCNN` and processes multi-class elephant call data.
- **Saved Model Checkpoints:** **None**. No `.pt` or `.pth` files currently exist on disk.

---

## 5. Resolution of `DEVELOPMENT_PLAN.md` Conflict Table

| Conflict Item | Document 1 (Role Prompt) | Document 2 (Aspirational README) | Phase 0 Audit Truth |
|---|---|---|---|
| **Core Files** | `preprocess.py`, `train_detector.py`, `train_classifier.py` | `elephant_cnn_model.py` (only) | **Document 1 is REAL.** `src/preprocess.py`, `train_detector.py`, and `train_classifier.py` exist and function. No `elephant_cnn_model.py` exists. |
| **Dataset Path** | `dataset/elephant/*`, `dataset/non_elephant/*` | `data/positive/`, `data/negative/` | **Document 1 is REAL.** Path is `my_elephant_project/dataset/`. |
| **Dataset Type** | Real recordings | Dummy data / synthetic pink noise | **Document 1 is REAL.** 579 real `.wav` audio files exist in `dataset/`. No synthetic pink noise files found. |
| **Training Status** | Trained checkpoint produced | Never trained on real data | **Neither fully accurate.** Scripts can run and train on real data today, but **no saved model checkpoints (`.pt`) currently exist**. |

---

## 6. Critical Findings & Data Integrity Risks

1. **File Copy Duplication:** Duplicate files exist in elephant subfolders (e.g. `Roar01.wav` and `Roar01 copy.wav`, `Rumble01.wav` and `Rumble01 copy.wav`, `Trumpet02.wav` and `Trumpet02 copy.wav`). These must be cleaned or deduplicated during Phase 1 manifest creation.
2. **Data Leakage Risk:** Non-elephant ESC-50 clips share recording origin prefixes (e.g. `1-155858-A-25.wav`, `1-155858-B-25.wav`, `1-155858-C-25.wav`). Naive `random_split()` splits contiguous segments of the same source recording across train and validation sets, causing data leakage. Phase 1 must implement source recording-aware splitting.
3. **Sample Rate Variations:** Elephant trumpets are recorded at 48.0kHz, roars/rumbles at 44.1kHz, and ESC-50 files at 44.1kHz. `AudioPreprocessor` resamples all clips to a unified 22.05kHz.

---

## 7. Next Phase Readiness (Phase 1)

Phase 0 audit is **COMPLETE**. The repository state is fully documented and verified. No project files or structures were modified or restructured during this phase.

Proceeding to **Phase 1 (Real Dataset Pipeline)** requires:
- Creating `manifest.csv` with per-clip source IDs, durations, sample rates, and deduplication flags.
- Implementing recording-level train/val/test splitting to prevent data leakage.
