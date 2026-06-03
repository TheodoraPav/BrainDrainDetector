# BrainDrainDetector

Multimodal detection of cognitive overload and socioemotional stress using the K EmoCon dataset.

The system fuses audio, E4 wristband physiological signals, and NeuroSky EEG data to predict a binary alarm state: **Safe** (0) or **Alarm** (1).

---

## Project Structure

```
BrainDrainDetector/
├── configs/                    # YAML experiment configurations
├── src/
│   ├── models/                 # Neural network components
│   ├── data/                   # Dataset and augmentation
│   ├── utils/                  # Metrics, plotting, quality parsing
│   ├── 01_build_labels.py      # Build Safe/Alarm labels from self-annotations
│   ├── 02_preprocess_audio.py  # Diarization, VAD, 5s window extraction
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

Ground truth comes from **self-annotations only** (`P{N}.self.csv`). Emotion checkboxes in the CSV are ignored; each 5-second window uses **arousal** and **valence** (1–5) to assign a binary label:

| Label | Name | Rule (from arousal *A*, valence *V*) |
|-------|------|--------------------------------------|
| 0 | Safe | Not overloaded: either *V* > 3 or *A* < 4 (includes “optimal” and “grey zone” VA states) |
| 1 | Alarm | Overloaded: *V* ≤ 3 **and** *A* ≥ 4 |

Thresholds are in `configs/base.yaml` (`labels.overloaded_*`, `labels.optimal_*`). Step 01 may print an internal VA-zone breakdown (optimal / overloaded / grey) for logging; **training and evaluation use only Safe vs Alarm** (`model.num_classes: 2`).

---

## Audio Preprocessing

For each stereo debate WAV file (e.g. `p1.p2.wav`):

1. **Speaker diarization** runs on the mono mix `(left + right) / 2` (PyAnnote).
2. Each diarized speaker is mapped globally to the left or right participant (P1/P2).
3. The **other speaker is muted** on that participant's track.
4. Short gaps within one turn are filled (`diarization_min_gap_sec`, default 3s).
5. **VAD** runs once on each separated mono track (global speech timeline).
6. The fixed **5-second annotation grid** is walked; a window is kept when speech overlap is at least `vad_min_overlap_sec` (default 3s) **or** at least `vad_min_overlap_pct` (default 60%).

Config keys live under `data:` in `configs/base.yaml`.

**Hugging Face models** (accept once with your token): `speaker-diarization-3.1`, `segmentation-3.0`, `wespeaker-voxceleb-resnet34-LM`, `voice-activity-detection`, `segmentation`.

To export playable WAV files for manual listening checks:

```bash
python src/02_preprocess_audio.py --config configs/base.yaml --testing
```

This writes `data_processed/audio_preview/{participant}/*.wav` and `audio_preview/summary.csv`.

---

## Experiments

Two experiments compare augmentation strategies:

1. **Baseline** — no augmentation (`configs/exp_baseline.yaml`)
2. **Offline Augmentation** — noise added once during preprocessing (`configs/exp_offline_aug.yaml`)

---

## Fusion Mode

The fusion layer is selectable via `model.fusion_mode` in any YAML config:

| `fusion_mode` | Description |
|---------------|-------------|
| `cross_attn_pooled` (default) | Audio (1 token) attends over a single pooled biosignal token. Lightweight baseline. |
| `sequence_cross_attn` | Audio (1 token) attends over the BiGRU output sequence (T biosignal time steps). Produces real attention weights over time. Useful for visualizing where the model "looks" inside the biosignals. |

To switch modes, set the value in `configs/base.yaml` (or override in an experiment config) and rerun `05_train.py`. Step 7 (`07_explain.py`) automatically picks the matching plot.

---

## Audio Backbone (Frozen Wav2Vec2)

The default audio encoder is pretrained Wav2Vec 2.0. Its weights are **frozen** during training (`model.freeze_audio_backbone: true` in `configs/base.yaml`). The model extracts fixed audio features and only trains the BiGRU biosignal encoder, fusion layer, and classification head.

This keeps training stable on a medium-sized dataset and avoids catastrophic forgetting in the pretrained speech representations.

To fine-tune Wav2Vec2 as well, set `freeze_audio_backbone: false` (only applies when `audio_encoder: "wav2vec2"`).

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
python src/05_train.py --config configs/exp_baseline.yaml
python src/05_train.py --config configs/exp_offline_aug.yaml

# Step 6 — evaluate
python src/06_evaluate.py --config configs/exp_offline_aug.yaml

# Step 7 — explain (attention maps)
python src/07_explain.py --config configs/exp_offline_aug.yaml
```

---

## Setup (local, Windows)

### Μία φορά μόνο — εγκατάσταση

Αυτά **μένουν στον δίσκο**. Δεν τα ξανατρέχεις κάθε terminal.

```powershell
cd "c:\Users\theod\Desktop\Master\2ο εξάμηνο\Deep Learning\BrainDrainDetector"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Το `pip install` κατεβάζει torch, pyannote κ.λπ. **μέσα στο `.venv`**. Μία φορά αρκεί, εκτός αν αλλάξει το `requirements.txt`.

PyAnnote models (accept once on Hugging Face, same account as token):
1. Token: https://huggingface.co/settings/tokens
2. Accept: `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`, `pyannote/wespeaker-voxceleb-resnet34-LM`, `pyannote/voice-activity-detection`, `pyannote/segmentation`

### Κάθε φορά που ανοίγεις νέο terminal

Νέο terminal = «ξέχασε» το venv. **Δεν** ξανακάνεις `pip install`. Απλά:

```powershell
cd "c:\Users\theod\Desktop\Master\2ο εξάμηνο\Deep Learning\BrainDrainDetector"
.\.venv\Scripts\Activate.ps1

$env:HF_TOKEN = "hf_xxxxxxxx"
$env:HUGGING_FACE_HUB_TOKEN = $env:HF_TOKEN
```

Μετά τρέχεις scripts με `python` (τώρα δείχει στο `.venv`).

### Audio preprocess + WAV preview (testing)

```powershell
python src/02_preprocess_audio.py --config configs/base.yaml --testing
```

Output:
- `data_processed/audio/` — `.pt` tensors
- `data_processed/audio_preview/` — `.wav` για ακρόαση + `summary.csv`

**Σημαντικό:** Αν **δεν** κάνεις `Activate.ps1`, το `python` είναι του Windows και θα πει `No module named 'torch'`.

Python 3.10+ required.

---

## Data

The raw dataset (K EmoCon) is not included in this repository. Place the dataset in the `Data/` folder following the original K EmoCon directory structure. Processed tensors are saved to `data_processed/` after running the preprocessing scripts.

---

## Evaluation

Primary metric: Macro F1 Score. Secondary metrics: Cohen's Kappa, per-class Recall.
Validation strategy: Leave One Subject Out (LOSO) cross-validation.
