# BrainDrainDetector: Multimodal Cognitive Overload Detection

**M.Sc. Artificial Intelligence Project**  
*Authors: Theodora Pavlidou, Marilena Papasideri*

Multimodal detection of cognitive overload and socioemotional stress during dyadic debates using the K-EmoCon dataset.

Each 5-second window of a conversation is classified as **Safe (0)** or **Alarm (1)** based on self-reported arousal and valence. The system fuses audio (Wav2Vec2), E4 wristband physiological signals (EDA, HR, IBI), and NeuroSky EEG (θ, α, β).

## Table of Contents
- [Labels](#labels)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Running the Pipeline](#running-the-pipeline)
- [Experiments and Results](#experiments-and-results)
- [Modality Overlap Analysis](#modality-overlap-analysis)
- [Key Findings](#key-findings)
- [Evaluation](#evaluation)
- [Setup](#setup)

---

## Labels

Ground truth from self-annotations (`P{N}.self.csv`). Arousal and valence on a 1–5 scale.

| Label | Name | Rule |
|-------|------|------|
| 0 | Safe | Valence > 3 OR Arousal < 4 |
| 1 | Alarm | Valence ≤ 3 AND Arousal ≥ 4 |

Class distribution after filtering: ~83% Safe, ~17% Alarm (1,492 total windows, 27 participants).

---

## Architecture

- **Audio encoder:** Wav2Vec 2.0, pretrained, **frozen** during training. Fixed feature extractor.
- **Biosignal encoder:** Bidirectional GRU (h=128). Input: 50×6 tensor (5 s window, 6 channels). Signals are **z-score normalized per window**.
- **Fusion:** configurable via `model.fusion_mode`. See table below.
- **Head:** fully connected, 2 logits, CrossEntropyLoss with class weights.

### Fusion modes

| `fusion_mode` | Description |
|---------------|-------------|
| `cross_attn_pooled` (default) | Audio and pooled biosignal token in 256-d space, MultiHeadAttention with seq_len=1. Audio-anchored residual. |
| `balanced_cross_attn` | Addresses audio bias by learning weights ($w_a, w_b$) for a balanced residual, combined with 15% modality dropout. |
| `quality_aware_cross_attn` | Injects biosignal quality indices (BQI) into the attention mechanism to penalize noisy physiological tokens. |
| `sequence_cross_attn` | Audio (Q, 1 token) attends over full BiGRU output sequence (K/V, 50 biosignal tokens). Real temporal attention weights. |
| `gated_multimodal_unit` | Gate z = sigmoid(W·concat(a,b)) mixes audio and biosignal embeddings per feature. Dynamic modality contribution. |

---

## Dataset

K-EmoCon: 10-minute dyadic debates between 27 participants. Audio is stereo WAV; physiological signals come from Empatica E4 and NeuroSky MindWave headsets. Raw data is **not included** in this repo.

Place the dataset in `Data/` following the original K-EmoCon directory structure.

---

## Project Structure

```
BrainDrainDetector/
├── configs/                        # YAML experiment configurations
├── src/
│   ├── models/                     # Neural network components
│   ├── data/                       # Dataset and augmentation
│   ├── utils/                      # Metrics, plotting, quality parsing
│   ├── 01_build_labels.py          # Build Safe/Alarm labels from self-annotations
│   ├── 02_preprocess_audio.py      # Diarization, VAD, 5s window extraction
│   ├── 03_preprocess_physio.py     # E4 and NeuroSky windowing and quality check
│   ├── 04_build_tensors.py         # Save ready-to-use PyTorch tensors
│   ├── 05_train.py                 # LOSO cross-validation training
│   ├── 06_evaluate.py              # Metrics and plots
│   ├── 07_explain.py               # Attention map visualization
│   ├── 07_tune_alarm_threshold.py  # Post-hoc threshold sweep (no retraining)
│   └── 08_late_fusion.py           # Decision-level late fusion pipeline
├── notebooks/                      # Kaggle single-cell experiment scripts
├── scripts/                        # Result export and analysis utilities
├── report/                         # LaTeX source + compiled PDF
├── Data/                           # Raw K-EmoCon files — NOT in git
└── data_processed/                 # Processed PyTorch tensors — NOT in git
```

---

## Running the Pipeline

```bash
# Preprocessing
python src/01_build_labels.py       --config configs/base.yaml
python src/02_preprocess_audio.py   --config configs/base.yaml
python src/03_preprocess_physio.py  --config configs/base.yaml
python src/04_build_tensors.py      --config configs/base.yaml

# Training and evaluation
python src/05_train.py              --config configs/exp_baseline.yaml
python src/06_evaluate.py           --config configs/exp_baseline.yaml

# Optional post-hoc steps
python src/07_explain.py            --config configs/exp_baseline.yaml
python src/07_tune_alarm_threshold.py --config configs/exp_threshold_target_recall.yaml
python src/08_late_fusion.py        --config configs/exp_late_fusion_audio_e4_eeg.yaml
```

---

## Experiments and Results

All classification ablations use LOSO (27 folds), macro-F1 as selection metric, class-weighted loss, balanced sampling, batch size 8, and frozen Wav2Vec2. One architectural knob is changed per run.

### Biosignal Encoder Ablation

All runs use `cross_attn_pooled` fusion.

| Encoder | Macro-F1 | Alarm Recall | Alarm F1 |
|---------|----------|--------------|----------|
| BiGRU h=128 (default) | **59.5%** | **51.2%** | **56.0%** |
| BiGRU + inter-window GRU×5 | 53.16% | 29.2% | 40.28% |
| Hand-crafted feature MLP | 49.63% | 23.4% | 33.96% |
| 1D CNN front-end | 46.16% | 20.8% | 29.84% |
| BiGRU + inter-window LSTM×5 | 45.25% | 18.4% | 27.34% |

Simple BiGRU wins. CNN overfits. Inter-window temporal models find no useful pattern across consecutive 5-second windows.

### Fusion Architecture Comparison

| Model | Macro-F1 | Alarm Recall | Alarm F1 |
|-------|----------|--------------|----------|
| Cross-attn pooled | **59.5%** | **51.2%** | **56.0%** |
| Cross-attn sequence | 56.19% | 43.4% | 50.23% |
| GMU | 46.97% | 21.2% | 30.64% |
| Quality-aware cross-attn | 41.16% | 11.6% | 18.74% |
| Balanced cross-attn | 40.45% | 12.0% | 18.84% |

GMU underperforms: the gate collapses toward audio and the biosignal branch is not used effectively on this small noisy dataset.

### Dual-Tower Biosignal Encoder

Testing separate encoders for E4 (EDA/HR/IBI) and EEG (θ/α/β) towers, then concatenating into a 256-d vector.

| Model | Macro-F1 | Alarm Recall |
|-------|----------|--------------|
| Joint BiGRU (single encoder) | **59.5%** | **51.2%** |
| Dual-tower BiGRU | 48.06% | 24.2% |

More parameters, worse generalization with 27 subjects. Dual-tower does not help.

### Unimodal Ablations

| Modality | Balanced Acc | Macro-F1 | Alarm Recall |
|----------|-------------|----------|--------------|
| Audio only | 54.46% | ~54.4% | 24.2% |
| Bio only | 47.52% | ~47.3% | 9.6% |
| Fusion (pooled) | 59.85% | 59.5% | 51.2% |

Audio is stronger than bio alone. The fusion adds the most on Alarm Recall.

### Late Fusion — 2 Modalities (Audio + Bio)

Decision-level fusion of separately trained model probability outputs.

| Strategy | Macro-F1 | Alarm Recall | Alarm F1 |
|----------|----------|--------------|----------|
| Quality-weighted | 45.82% | 18.4% | 27.67% |
| Majority vote (OR) | 58.53% | 48.4% | 54.14% |
| Stacking LR | **58.74%** | 47.6% | **53.91%** |

### Late Fusion — 3 Modalities (Audio + E4 + EEG)

| Strategy | Macro-F1 | Alarm Recall | Alarm F1 |
|----------|----------|--------------|----------|
| Quality-weighted | 44.2% | 14.2% | 22.94% |
| Majority vote (OR) | 59.09% | 48.4% | 54.5% |
| Stacking LR | **61.86%** | **51.6%** | **57.78%** |

**Best overall result: Late fusion Stacking LR with 3 modalities (Macro-F1 = 61.86%).**

### Best Single-Model Run

Cross-attention pooled, BiGRU h=128, weighted loss, no augmentation.

| Split | Accuracy | Macro-F1 | Alarm Recall | Alarm F1 | Alarm Precision |
|-------|----------|----------|--------------|----------|-----------------|
| Balanced (50/50 test) | 59.8% | 59.5% | 51.2% | 56.0% | 61.84% |
| Real class distribution | 65.68% | 54.73% | 51.25% | 32.45% | 23.75% |

Under real class distribution Alarm Precision drops sharply (many false alarms on a rare class).

---

## Modality Overlap Analysis

Compared audio-only, bio-only, and fusion model predictions on the same 1,492 windows.

| Metric | Value |
|--------|-------|
| Audio fail rate | 25.0% |
| Bio fail rate | 26.74% |
| Fusion fail rate | 34.32% |
| Both unimodal correct | 62.27% |
| Synergy: both wrong, fusion right | 5.63% |
| Interference: both right, fusion wrong | **14.75%** |
| Oracle (if at least one correct) | 85.99% |
| Fusion follows audio when models disagree | **80.51%** |

The fusion is heavily audio-biased. There is more interference than synergy — fusion sometimes hurts. This motivated trying late fusion at decision level (Stacking LR) instead of learned feature fusion.

---

## Key Findings

**What worked:**
- Class-weighted loss + balanced sampling — essential for non-trivial Alarm recall.
- Frozen Wav2Vec2 — prevents catastrophic forgetting on a 27-subject dataset.
- Late fusion Stacking LR with 3 modalities — best overall Macro-F1 (61.86%).
- Cross-attn pooled with BiGRU — best single joint-model (59.5%).
- Macro-F1 as selection metric — prevents collapse to always-Safe prediction.

**What did not work:**
- CNN physio front-end — overfits on 27 subjects.
- Inter-window temporal models — no useful cross-window signal in 5-second clips.
- Dual-tower biosignal encoder — too many parameters, worse generalization.
- GMU — audio modality collapse; biosignals not learned.
- Quality-weighted fusion — too conservative; downweights noisy but informative signals.
- Offline augmentation — structural noise in dataset is not fixed by synthetic noise injection.

**Dataset challenges:**
- 27 participants, LOSO → tiny test sets, high fold variance.
- 17% Alarm rate. Alarm windows are hard: audio fails on 76%, bio fails on 90% of them.
- Some participants nearly perfectly predicted; others almost completely wrong (participant-level variance is larger than the model effect).
- K-EmoCon papers note noisy physiological signals as a known limitation.

---

## Evaluation

- **Primary metric:** Macro-F1
- **Key secondary:** Alarm Recall (safety-critical — missing an alarm is worse than a false alarm)
- **Validation:** LOSO cross-validation, 27 folds

---

## Setup

```powershell
git clone https://github.com/TheodoraPav/BrainDrainDetector.git
cd BrainDrainDetector

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Every new terminal session:

```powershell
.\.venv\Scripts\Activate.ps1
$env:HF_TOKEN = "hf_xxxxxxxx"
$env:HUGGING_FACE_HUB_TOKEN = $env:HF_TOKEN
```

PyAnnote models — accept on Hugging Face once:
- `pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`
- `pyannote/wespeaker-voxceleb-resnet34-LM`
- `pyannote/voice-activity-detection`

Python 3.10+ required.
