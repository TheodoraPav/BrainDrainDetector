# Kaggle Experiment Plan — BrainDrainDetector

## Notebook layout (2 cells)

| Cell | File | When to use |
|------|------|-------------|
| **Cell 1** | `kaggle_baseline_one_cell.py` | First time: full pipeline (or seed audio+diarization+physio from dataset). Also: `regression_va`, `va_separated_classify`. |
| **Cell 2** | `kaggle_classification_baseline_cell.py` | **All classification ablations** after Cell 1 (or seeded data). Only re-runs steps 05+06 by default. |

Always: **GPU T4 x1**, **HF_TOKEN** secret, dataset **BrainDrainDataset**.

---

## Already completed (`results/`)

| # | Run folder | F1 alarm | Fair? |
|---|------------|----------|-------|
| 1 | `classification_cross_attn_pooled_weighted_no_aug` | **0.185** | yes (macro_f1) — **best** |
| 2 | `classification_sequence_cross_attn_weighted_no_aug` | 0.142 | yes |
| 3 | `classification_cross_attn_pooled_dualtower_weighted_no_aug` | 0.126 | yes |
| 4 | `classification_sequence_cross_attn_cnn_weighted_no_aug` | 0.115 | **superseded** — see `_final` (macro_f1) |
| 5 | `classification_cross_attn_pooled_gru5_weighted_no_aug` | 0.105 | yes |
| 6 | `classification_cross_attn_pooled_cnn_weighted_no_aug` | 0.076 | yes |
| 7 | `classification_sequence_cross_attn_lstm5_weighted_no_aug` | 0.066 | **superseded** — run `seq_lstm5_fair` |
| 8 | `va_separated_classify_sequence_cross_attn_weighted_no_aug` | 0.077 | different task |

---

## Missing runs — Kaggle sessions to schedule

### ~~Session 1 — `seq_cnn_fair`~~ DONE

Saved: `results/results_classification_sequence_cross_attn_cnn_weighted_no_aug_final/`

---

### Session 2 — `seq_lstm5_fair` (Cell 2, ~2–4 h) **← NEXT**

**Goal:** Re-run sequence + inter-window LSTM with **macro_f1**, **batch 8**.

```
PRESET = "seq_lstm5_fair"
```

→ `results_classification_sequence_cross_attn_lstm5_weighted_no_aug.zip`

---

### Session 3 — `offline_aug` (Cell 2, ~2–4 h)

**Goal:** Baseline architecture + offline augmentation (step 04 builds `windows_aug/`).

```
PRESET = "offline_aug"
```

→ `results_classification_cross_attn_pooled_weighted_aug.zip` (new name)

---

### Session 4 — `regression_va` (Cell 1, ~4–6 h)

**Goal:** Continuous A/V regression + derived alarm metrics.

In **Cell 1** set:

```
TASK_MODE = "regression_va"
FUSION_MODE = "cross_attn_pooled"
DUAL_TOWER_BIOSIGNAL = False
```

Auto steps: `01, 04, 05, 06`.

→ `results_regression_va_cross_attn_pooled_weighted_no_aug.zip`

---

## Fair comparison rules (Cell 2)

Keep **fixed** across classification ablations:

- `SELECTION_METRIC = "macro_f1"`
- `BATCH_SIZE = 8`
- `EARLY_STOPPING_PATIENCE = 8`, `MIN_EPOCHS = 5`
- `AUGMENTATION_ENABLED = False` (except offline_aug preset)
- `weighted_loss = True`, `balanced_sampling = True`
- Wav2Vec2 **frozen**, embedding cache **on**

Change **one** architecture knob per run (fusion / CNN / dual-tower / temporal).

---

## Dataset seed (recommended)

Upload to Kaggle dataset:

```
data_processed/data_processed/audio_diarization/   # segments.csv per debate
data_processed/data_processed/physio/              # optional — saves step 03
```

Cell 1 `SEED_DATASET_FOLDERS = ["audio", "audio_diarization", "physio"]` — step 02 runs VAD only (~15–30 min), not full pyannote.

---

## After each session

1. Download zip from **Output** tab.
2. Extract into `results/results_<name>/` locally.
3. Compare `data_processed/loso_results.pt` → `summary` → `f1_alarm_mean`.

---

## Do NOT use for fair classification

- `kaggle_rerun_experiment_cell.py` — defaults `selection_metric` to `recall_alarm` for classification. **Deprecated** for arch ablations; use Cell 2 instead.
