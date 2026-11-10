
# CS781 Phase-4: Legal Outcome Classification (CJPE) – Final Views & PEFT Ablations

**Course:** CS781 Capstone  
**Phase:** Phase-4 (Downstream Adaptation – Court Judgment Prediction / Outcome Classification)  
**Task:** Binary (or multi-class) legal outcome classification of Indian court judgments  
**Backbones:** All four Phase-1 foundation models (`custom_mha`, `mla_moe`, `gqa_alibi`, `gqa_conv`)

This repository consolidates the complete Phase-4 pipelines for every Phase-1 backbone. Each architecture section reloads the corresponding Phase-1 checkpoint + tokenizer, constructs multiple document-representation **views** (flat head/tail/hybrid and hierarchical multi-chunk), attaches a classification head, and runs a systematic matrix of parameter-efficient fine-tuning (PEFT) methods, pooling strategies, loss functions and knowledge-distillation experiments. Training is resumable; decision thresholds are tuned on the development set to maximise macro-F1; final predictions are packaged into official submission zips.

---

## Table of Contents
1. [Task Definition](#1-task-definition)
2. [Data Pipeline](#2-data-pipeline)
3. [Document Representation Views](#3-document-representation-views)
4. [Model Architecture (Phase-4)](#4-model-architecture-phase-4)
5. [Parameter-Efficient Fine-Tuning Methods](#5-parameter-efficient-fine-tuning-methods)
6. [Loss Functions & Pooling](#6-loss-functions--pooling)
7. [Knowledge Distillation](#7-knowledge-distillation)
8. [Training Methodology](#8-training-methodology)
9. [Threshold Tuning & Inference](#9-threshold-tuning--inference)
10. [Experimental Matrix](#10-experimental-matrix)
11. [How to Run](#11-how-to-run)
12. [Evaluation Metrics & Analysis](#12-evaluation-metrics--analysis)
13. [Submission Generation](#13-submission-generation)
14. [File Layout & Artifacts](#14-file-layout--artifacts)
15. [Reproducibility Notes](#15-reproducibility-notes)

---

## 1. Task Definition

### 1.1 Input / Output
- **Input:** A full court judgment (variable length, often several thousand tokens).
- **Output:** A discrete outcome label (binary or multi-class depending on the CJPE split; the code treats the problem as classification with `NL` classes and reports macro-F1).

### 1.2 Source
Hugging Face dataset `Exploration-Lab/CS-781-Capstone`:

```
Phase_4_CJPE/
├── Train/ …
├── Dev/   …
└── Test/  …
```

Documents are already tokenised offline (or tokenised on the fly) with the Phase-1 BPE of the active backbone and cached under a shared `cache/` directory so that all architectures that share a tokenizer reuse the same token ids.

---

## 2. Data Pipeline

### 2.1 Loading & Caching
```python
def hf(fn): return hf_hub_download(HF_REPO, fn, repo_type="dataset", token=HF_TOKEN)
# Documents are read with a streaming / ijson-friendly loader.
# Token ids are written to CACHE so that subsequent runs (and other backbones
# that share the same tokenizer) never re-tokenise.
```

### 2.2 Dataset Objects
```python
class DocDS(torch.utils.data.Dataset):
    # Returns (doc_index, token_sequence_or_list_of_chunks, label)
```

A collate function packs either a single flat sequence or a list of hierarchical chunks into padded tensors together with lengths / attention masks and gold labels.

### 2.3 Sampling Controls
```python
TRAIN_N    = None   # stratified subset of training docs (None = all)
DEV_FREQ_N = 1500   # docs used for in-training evaluation
```

---

## 3. Document Representation Views

Because judgments routinely exceed the Phase-1 context length (1024 tokens), Phase-4 defines a family of **views** that map a long document onto one or more fixed-length sequences:

| View name     | Kind  | Length | Selection rule                                      | Micro-batch |
|---------------|-------|--------|-----------------------------------------------------|-------------|
| `head512`     | flat  | 512    | first 512 tokens                                    | 16          |
| `tail512`     | flat  | 512    | last 512 tokens                                     | 16          |
| `tail1024`    | flat  | 1024   | last 1024 tokens                                    | 8           |
| `ht1024`      | flat  | 1024   | first 256 + last 768 tokens                         | 8           |
| `hier8`       | hier  | 8×512  | first K/4 + last 3K/4 chunks (cut from the end)     | 2           |
| `hier8_tail`  | hier  | 8×512  | last 8 chunks                                       | 2           |
| `hier8_even`  | hier  | 8×512  | evenly spaced 8 chunks                              | 2           |
| `hier4`       | hier  | 4×256  | 1 head + 3 tail chunks                              | 8           |
| `hier4_tail`  | hier  | 4×256  | last 4 chunks                                       | 8           |

**Flat views** produce a single sequence per document.  
**Hierarchical views** produce `K` overlapping / contiguous chunks; each chunk is encoded independently by the backbone; the resulting chunk vectors are then pooled by a small transformer / position-aware aggregator before the final classifier.

```python
def flat_seq(ids, v): …          # head / tail / ht selection + BOS/EOS
def hier_chunks(ids, v): …       # sliding windows from the end + selection policy
```

---

## 4. Model Architecture (Phase-4)

### 4.1 High-level Stack
```
Document  →  View (flat or hierarchical chunks)
                │
                ▼
┌──────────────────────────────────────────────┐
│  Phase-1 Backbone (bidirectional encoding)   │
│  • Token embedding                           │
│  • 8 × Transformer blocks                    │
│  • Final RMSNorm                             │
└──────────────────────────────────────────────┘
                │  hidden states
                ▼
Pooling (last / mean / last+mean)
                │
                ▼
( hierarchical only ) Chunk-position embedding
                + Transformer / attention pooler
                │
                ▼
Classification head  →  logits [B, NL]
```

### 4.2 Backbone Wrapper
The same `LMBackbone` / `load_phase1` utilities used in Phase-3 are reused.  
Optional layer subsetting (`layers_keep`) permits construction of thinner student models for distillation.

### 4.3 Classification Head
```python
# Typical construction inside build_model(cfg)
pool = cfg.get("pool", "last_mean")          # last | mean | last_mean
# Projection + dropout + linear classifier
# For hierarchical views an extra TransformerEncoder layer
# (or simple attention pool) aggregates the K chunk vectors.
```

Soft-prompt tokens can be prepended when the configuration requests them.

---

## 5. Parameter-Efficient Fine-Tuning Methods

| Method | Description | Typical hyper-parameters |
|--------|-------------|--------------------------|
| **Full FT** | All backbone + head parameters | `lr=2e-5`, `head_lr=1e-3` |
| **LoRA** | Low-rank adapters on attention & FFN projections | `r=8/16`, `alpha=2r` |
| **rsLoRA** | Rank-stabilised LoRA (scale \(1/\sqrt{r}\)) | same as LoRA |
| **LoRMA** | Multiplicative low-rank adapters (output-side) | `r=8/16` |
| **AdaLoRA** | Adaptive rank via SVD sensitivity; ranks redistributed during training | `r=12`, `alpha=24`, target rank schedule |
| **BitFit** | Only bias terms + LayerNorm scales | very few parameters |
| **Bottleneck Adapters** | Serial down/up-projection residuals after each block | bottleneck dim 64–128 |
| **Linear probe** | Head only (backbone frozen) | `head_lr=2e-3` |

Injection targets cover the same Linear modules used in Phase-2/3 (`q`, `k`, `v`, `o`, `w13`, `w2`, …).  MoE expert weights are left frozen.

```python
METHOD_HP = {
    "adalora": dict(r=12, alpha=24, …),
    "lorma":   dict(r=8, alpha=16),
    "bitfit":  dict(),
    …
}
```

---

## 6. Loss Functions & Pooling

| Loss | Notes |
|------|-------|
| **Weighted CE** (`wce`) | Inverse-frequency or effective-number class weights |
| **Focal** | \(\gamma=2\), optional \(\alpha\) |
| **Label-smoothed CE** | Smoothing 0.1 (mainly for KD students) |

Pooling options for the final token representation:
- `last` – last non-pad token  
- `mean` – mean of non-pad tokens  
- `last_mean` – concatenation of last and mean (then projected)

---

## 7. Knowledge Distillation

A high-capacity teacher (full FT or strong PEFT model on a long view) produces soft logits.  
Students are typically 4-layer truncations of the same backbone, trained with:

```
L = α · CE(student, gold) + (1-α) · T² · KL(student/T ‖ teacher/T)
```

Typical settings: \(T=2.0\), \(\alpha=0.5\).  
Distilled students appear in the results under names such as `kd_student_…` and are included in the final submission selection when they rank highly.

---

## 8. Training Methodology

### 8.1 Optimiser & Schedule
- AdamW with separate parameter groups (backbone / PEFT vs. head)
- Linear warm-up → cosine decay
- Gradient clipping at 1.0
- Mixed precision (`bfloat16` preferred)
- Gradient accumulation to reach `EFF_BS` (default 32 documents per optimiser step)

### 8.2 Step Budget
```python
MAX_STEPS  = 1000          # optimiser steps per experiment
EVAL_EVERY = 200           # evaluate on DEV_FREQ_N docs
CKPT_EVERY = 50            # write resume checkpoint
```

### 8.3 Resumption
Every run writes:
```
ckpts/<exp_name>_best.pt     # best macro-F1 weights
ckpts/<exp_name>_ckpt.pt     # full trainer state for resumption
```
If a `_best.pt` already exists the training loop is skipped (Mode A).

### 8.4 Throughput Measurement
A utility `sec_per_step(cfg)` times a few micro-batches so that wall-clock estimates can be reported alongside parameter counts.

---

## 9. Threshold Tuning & Inference

After training (or after loading a checkpoint) the model produces class probabilities on the development set.  
A simple grid search finds the decision threshold that maximises macro-F1:

```python
def tune_threshold(probs, labels):
    grid = np.arange(0.05, 0.96, 0.01)
    f1s  = [f1_score(labels, (probs >= t).astype(int), average="macro") for t in grid]
    best = int(np.argmax(f1s))
    return float(grid[best]), float(f1s[best])
```

The tuned threshold is then applied to the test-set probabilities to obtain the final hard predictions.

Raw logits / probabilities are saved under `preds/` for later ensemble or analysis.

---

## 10. Experimental Matrix

The notebook executes a curated matrix that covers the required ablations.  
Typical experiment names follow the pattern:

```
<method>_<view>
e.g.  adalora_tail512 , lorma_ht1024 , bitfit_hier8_tail ,
      full_ht1024 , kd_student_hier4 , …
```

Groups recorded in `results.json`:
- `method` – pure PEFT / full-FT comparison
- `view` – effect of different document representations
- `loss` – CE vs. focal vs. weighted
- `kd` – distillation students
- `pool` – pooling strategy ablations

All results are aggregated into a pandas `results_df` and visualised (learning curves, F1 bar charts, efficiency scatter plots, confusion matrices).

---

## 11. How to Run

### 11.1 Prerequisites
```bash
pip install torch tqdm ijson huggingface_hub scikit-learn pandas matplotlib numpy
```

### 11.2 Path Setup
1. Place Phase-1 tokenizers and checkpoints under `PHASE1_DIR` (default `/kaggle/working/cs781_phase1`).
2. Run the shared path-resolution cell; confirm `exists=True` for the desired backbone.

### 11.3 Per-Backbone Workflow
```python
# 1. Activate backbone
BACKBONE = "gqa_conv"          # or custom_mha / mla_moe / gqa_alibi
# … run activation cell …

# 2. Load Phase-1 classes (tokenizer + LegalLM / DualPathLM / …)
# 3. Define VIEWS and METHOD_HP
# 4. Build & train (or load) a configuration
cfg = dict(BASE, method="adalora", view="tail512", **METHOD_HP["adalora"])
train_run(cfg)                 # skips training when *_best.pt exists

# 5. Threshold tuning + test prediction + submission zip
#    (performed automatically by the final cells)
```

### 11.4 Two Execution Modes
- **Mode A (default)** – pre-computed `*_best.pt` checkpoints are present → training is skipped, evaluation / threshold tuning / submission generation run immediately.
- **Mode B** – delete or rename existing checkpoints → full training from the Phase-1 initialisation is performed.

---

## 12. Evaluation Metrics & Analysis

### 12.1 Primary Metric
**Macro-F1** on the development set (used for model selection and threshold tuning).  
Test-set macro-F1 is reported after the tuned threshold is applied.

### 12.2 Secondary Analyses
- Per-class precision / recall / F1  
- Confusion matrices  
- Parameter count vs. macro-F1 efficiency plots  
- Wall-clock seconds-per-step measurements  
- Comparison of flat vs. hierarchical views

---

## 13. Submission Generation

For every experiment that finishes successfully the notebook writes:

```
preds/submission_<exp_name>/
├── prediction.json          # [{"id": …, "pred": …}, …]
└── (zipped as submission_<exp_name>.zip)
```

A final selection cell can keep only the top-k distinct techniques (including a KD student when available) for the official hand-in.

---

## 14. File Layout & Artifacts

```
phase4-<backbone>/
├── cache/                       # shared token-id cache
├── <arch>/                      # e.g. arch1 … arch4
│   ├── ckpts/
│   │   ├── <exp>_best.pt
│   │   └── <exp>_ckpt.pt
│   ├── preds/
│   │   ├── <exp>_dev_tuned.npy
│   │   ├── <exp>_test_tuned.npy
│   │   └── submission_<exp>.zip
│   └── results.json
└── plots/                       # learning curves, efficiency, confusion, …
```

---

## 15. Reproducibility Notes

- Global seed `SEED = 42` is set for Python, NumPy and PyTorch (including CUDA).
- Token caches are deterministic given the same tokenizer and cleaning function.
- Class weights (when used) are computed once from the training fold and frozen.
- Checkpoints store full model state, optimiser state, GradScaler state and step counter → exact resumption.
- Threshold grid search is deterministic.
- The same view definitions, label inventory and evaluation script are used for every backbone, guaranteeing fair comparison.

---