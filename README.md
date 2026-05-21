# BrainDrainDetector

Multimodal detection of cognitive overload and socioemotional stress using the K EmoCon dataset.

The system fuses audio, E4 wristband physiological signals, and NeuroSky EEG data to predict three states: Overloaded, Optimal, and Grey Zone.

---

## Project Structure

```
BrainDrainDetector/
├── configs/                    # YAML experiment configurations
├── src/
│   ├── models/                 # Neural network components
│   ├── data/                   # Dataset and augmentation
│   ├── utils/                  # Metrics, plotting, quality parsing
│   ├── 01_build_labels.py      # Build 3-class labels from annotations
│   ├── 02_preprocess_audio.py  # Stereo split, VAD filtering, SpecAugment
│   ├── 03_preprocess_physio.py # E4 and NeuroSky windowing and quality check
│   ├── 04_build_tensors.py     # Save ready-to-use PyTorch tensors
│   ├── 05_train.py             # LOSO cross-validation training
│   ├── 06_evaluate.py          # Metrics and plots
│   └── 07_explain.py           # Attention map visualization
├── report/                     # LaTeX academic report
├── notebooks/                  # Exploratory analysis (not part of pipeline)
├── Data/                       # Raw K EmoCon files — NOT in git
└── data_processed/             # Processed PyTorch tensors — NOT in git
```

---

## Labels

| Class | Name | Condition |
|-------|------|-----------|
| 0 | Optimal | Concentration == x AND Valence >= 3 AND no negative emotions |
| 1 | Overloaded | Valence <= 3 AND (Frustration/Confusion/Nervous == x OR Arousal >= 4) |
| 2 | Grey Zone | All other intermediate states |

At inference time, classes 0 and 2 merge into a safe state. Class 1 triggers an alert.

---

## Experiments

Three experiments compare augmentation strategies:

1. **Baseline** — no augmentation (`configs/exp_baseline.yaml`)
2. **Offline Augmentation** — noise added once during preprocessing (`configs/exp_offline_aug.yaml`)
3. **Online Augmentation** — dynamic noise injection per epoch (`configs/exp_online_aug.yaml`)

---

## Running the Pipeline

```bash
# Step 1 — build labels
python src/01_build_labels.py --config configs/base.yaml

# Step 2 — preprocess audio
python src/02_preprocess_audio.py --config configs/base.yaml

# Step 3 — preprocess physiological signals
python src/03_preprocess_physio.py --config configs/base.yaml

# Step 4 — build tensors
python src/04_build_tensors.py --config configs/base.yaml

# Step 5 — train (choose experiment config)
python src/05_train.py --config configs/exp_online_aug.yaml

# Step 6 — evaluate
python src/06_evaluate.py --config configs/exp_online_aug.yaml

# Step 7 — explain (attention maps)
python src/07_explain.py --config configs/exp_online_aug.yaml
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10 or higher is required.

---

## Data

The raw dataset (K EmoCon) is not included in this repository. Place the dataset in the `Data/` folder following the original K EmoCon directory structure. Processed tensors are saved to `data_processed/` after running the preprocessing scripts.

---

## Evaluation

Primary metric: Macro F1 Score. Secondary metrics: Cohen's Kappa, per-class Recall.
Validation strategy: Leave One Subject Out (LOSO) cross-validation.
