
# CS781 Phase-5: Legal Statute Identification (LSI) – Multi-Label Classification

**Course:** CS781 Capstone  
**Phase:** Phase-5 (Downstream Adaptation – Legal Statute Identification)  
**Task:** Multi-label prediction of applicable statutes / legal provisions for a given judgment or case description  
**Backbones:** All four Phase-1 foundation models (`custom_mha`, `mla_moe`, `gqa_alibi`, `gqa_conv`)

This repository consolidates the complete Phase-5 pipelines for every Phase-1 backbone. Each experiment reloads the matching Phase-1 checkpoint and tokenizer, encodes long documents as multiple fixed-length chunks, applies a **label-wise attention head** (each statute has its own query over chunks), and trains under a variety of PEFT regimes (full FT, LoRMA, AdaLoRA, KD students, …). Decision thresholds are tuned globally and per-label on the development set; the better strategy is applied to the test set and packaged into official submission zips.

---

## Table of Contents
1. [Task Definition & Label Schema](#1-task-definition--label-schema)
2. [Data Pipeline](#2-data-pipeline)
3. [Document Encoding (Chunking)](#3-document-encoding-chunking)
4. [Model Architecture (Phase-5)](#4-model-architecture-phase-5)
5. [Parameter-Efficient Fine-Tuning Methods](#5-parameter-efficient-fine-tuning-methods)
6. [Loss Function](#6-loss-function)
7. [Training Methodology](#7-training-methodology)
8. [Inference, Threshold Tuning & Metrics](#8-inference-threshold-tuning--metrics)
9. [Required Experiments](#9-required-experiments)
10. [How to Run](#10-how-to-run)
11. [Evaluation & Analysis](#11-evaluation--analysis)
12. [Submission Generation](#12-submission-generation)
13. [File Layout & Artifacts](#13-file-layout--artifacts)
14. [Reproducibility Notes](#14-reproducibility-notes)

---

## 1. Task Definition & Label Schema

### 1.1 Input / Output
- **Input:** A legal document (judgment, case description, or cue text). Documents are long; they are never truncated to a single 512/1024-token window.
- **Output:** A variable-sized set of statute / provision labels that apply to the document (multi-label).

### 1.2 Label Inventory
Labels are loaded from the Phase-5 data files (typically on the order of **100 statutes**).

```python
label_list   # ordered list of statute names / IDs
NUM_LABELS   # len(label_list)  (≈ 100)
Y            # binary matrix [N_docs, NUM_LABELS]
```

Each document may be associated with zero, one, or many labels.

### 1.3 Why a Dedicated Tokenizer per Backbone?
Each Phase-1 model was trained with its own byte-level BPE (different merge tables → different vocabulary sizes).  
Using a mismatched tokenizer would either produce out-of-range token ids or feed the model a completely different segmentation from the one it saw during pre-training.  
Consequently the code keeps **one tokenizer file per backbone** and always loads the matching one when `BACKBONE` changes. The embedding-matrix size is taken from the checkpoint; an assert verifies consistency with `tokenizer.vocab_size`.

---

## 2. Data Pipeline

### 2.1 Source
Hugging Face dataset `Exploration-Lab/CS-781-Capstone` (Phase-5 LSI splits).  
The loader supports both “full text” and “cue” modes; cue texts are already condensed and therefore use a simpler chunk-selection policy.

### 2.2 Cleaning
The identical conservative legal text cleaner developed in Phase-1 is applied to every document (NFKC, quote/dash normalisation, de-hyphenation, page-number removal, etc.).

### 2.3 Token Cache
Tokenisation is expensive.  Results are cached on disk:

```
phase5/cache/tok_{backbone}_{mode}_{split}.pkl
```

Subsequent runs (and any other experiment that shares the same backbone + mode + split) simply unpickle the list of token-id arrays.

### 2.4 Dataset Class
```python
class LSIDataset(Dataset):
    def __init__(self, toks_list, labels, mode="headtail", n_chunks=4):
        ...
    # Returns token chunks + multi-hot label vector
```

A custom collate function packs a batch of variable-length chunk lists into a dense tensor `[B, C, L]` together with a chunk-level validity mask.

---

## 3. Document Encoding (Chunking)

Long documents are split into fixed-length chunks of `CHUNK_LEN` tokens (typically 512).  
A selection policy then keeps at most `n_chunks` of them:

| Mode       | Policy |
|------------|--------|
| `headtail` | first \(\lfloor n/4\rfloor\) chunks + last remaining chunks |
| `cue`      | first \(n\) chunks (cue text is already short) |
| `even`     | evenly spaced chunks (optional ablation) |

Each selected chunk is wrapped with `[BOS] … [EOS]` and fed independently through the backbone.  
The resulting chunk vectors form a grid of shape `[B, C, d]` that is later consumed by the label-attention head.

```python
def pick_chunks(n_tok, n_chunks, mode): ...
def make_batch(toks_list, tok, n_chunks, mode): ...  # → ids [B,C,L], mask [B,C]
```

---

## 4. Model Architecture (Phase-5)

### 4.1 High-level Stack
```
Document tokens
      │
      ▼
Chunk selection (head-tail / cue / even)
      │
      ▼
┌────────────────────────────────────────────┐
│  Phase-1 Backbone (shared across chunks)   │
│  • Token embedding                         │
│  • 8 × Transformer blocks (bidirectional)  │
│  • Final RMSNorm                           │
└────────────────────────────────────────────┘
      │  chunk vectors  [B, C, d]
      ▼
Label-wise Attention Head
  • each of the N labels has its own query
  • produces one document vector per label
      │
      ▼
Linear classifier (or bias only)  →  logits [B, N]
```

### 4.2 Label-Attention Head
```python
class LabelAttnHead(nn.Module):
    """Each statute attends over chunks with its own query → its own document vector."""
    def __init__(self, d, n):
        self.att  = nn.Linear(d, n)          # produces attention logits per label
        self.w    = nn.Parameter(...)        # per-label projection
        self.bias = nn.Parameter(torch.zeros(n))

    def forward(self, grid, mask):
        # grid: [B, C, d], mask: [B, C]
        a = self.att(grid).masked_fill(~mask[..., None], -inf).softmax(dim=1)
        # a: [B, C, N]
        return (einsum("bcl,bcd->bld", a, grid) * self.w).sum(-1) + self.bias
```

This design lets each statute focus on the chunks that are most relevant to it, which is especially useful when a long judgment discusses many different legal issues.

### 4.3 Pooling Alternatives
A simpler mean / last-token pool over chunks can be selected via a configuration flag; the label-attention head is the default and strongest option.

### 4.4 Backbone Loading
```python
def load_lm(backbone):
    # Dispatches to the correct class (LegalLM / DualPathLM / ALiBi-GQA / MLA-MoE)
    # and loads the matching Phase-1 checkpoint + tokenizer.
```

Gradient checkpointing can be enabled per experiment to fit larger micro-batches.

---

## 5. Parameter-Efficient Fine-Tuning Methods

| Method | Description | Typical use in required experiments |
|--------|-------------|-------------------------------------|
| **Full FT** | All backbone + head parameters | `exp2_full_gqa_conv` (also serves as KD teacher) |
| **LoRMA** | Multiplicative low-rank adapters | `exp1_lorma_custom_mha` |
| **AdaLoRA** | Adaptive rank via SVD sensitivity | `exp3_adalora_mla_moe` |
| **KD student** | 4-layer (or half-depth) student distilled from a strong teacher | `exp4_kd_gqa_alibi_half` |
| **LoRA / rsLoRA / BitFit / Adapters** | Available in the generic builder for extra ablations | optional |

Injection targets are the same Linear modules used in earlier phases (attention projections and FFN / Dual-Path matrices).  MoE experts stay frozen.

---

## 6. Loss Function

Because the task is multi-label and highly imbalanced, training uses **binary cross-entropy with logits** (optionally with pos-weight derived from label frequencies, or an asymmetric / focal variant):

```python
loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=…)
```

Class / label priors can be supplied to the model constructor and incorporated into the bias initialisation of the label-attention head.

---

## 7. Training Methodology

### 7.1 Experiment Descriptor
```python
@dataclass
class LSIExp:
    name: str
    backbone: str          # custom_mha | mla_moe | gqa_conv | gqa_alibi
    method: str            # full | lorma | adalora | kd | …
    lr: float
    head_lr: float
    epochs: int
    batch_docs: int
    grad_accum: int
    grad_ckpt: bool = False
    # … plus optional KD teacher path, rank, etc.
```

### 7.2 Optimiser & Schedule
- AdamW with two parameter groups (backbone / PEFT vs. head)
- Linear warm-up followed by cosine decay
- Gradient clipping
- Mixed precision (`bfloat16` when available)
- Gradient accumulation to reach a stable effective batch size

### 7.3 Resumption
```
phase5/ckpt/<exp_name>_latest.pt   # full trainer state
phase5/ckpt/<exp_name>_best.pt     # best validation macro-F1
```
Re-running the same experiment automatically resumes from the latest checkpoint.

### 7.4 Typical Hyper-parameters
```python
lr          = 3e-5
head_lr     = 1e-3
epochs      = 1–2
batch_docs  = 4
grad_accum  = 4
n_chunks    = 4
CHUNK_LEN   = 512
```

---

## 8. Inference, Threshold Tuning & Metrics

### 8.1 Probability Generation
```python
@torch.no_grad()
def lsi_predict(model, toks, tok, exp, bs=8):
    # Returns sigmoid probabilities [N_docs, NUM_LABELS]
```

Probabilities for the development and test sets are written to disk:

```
phase5/dev_<exp_name>.npy
phase5/test_<exp_name>.npy
```

### 8.2 Threshold Strategies
Two strategies are evaluated on the development set:

1. **Global threshold** – a single scalar \(t\) applied to every label.  
2. **Per-label thresholds** – an independent \(t_j\) for each statute.

```python
def tune_global_threshold(Y, probs): ...
def tune_per_label_thresholds(Y, probs): ...
```

The strategy that yields the higher macro-F1 on the development set is selected and applied to the test probabilities.

### 8.3 Primary Metric
**Macro-F1** (label-wise F1 averaged over all statutes).  
Micro-F1 and per-label F1 vectors are also recorded for analysis.

---

## 9. Required Experiments

The course rubric asks for a minimal set of distinct techniques.  A typical required list is:

| Experiment name              | Backbone    | Method   | Notes |
|------------------------------|-------------|----------|-------|
| `exp2_full_gqa_conv`         | gqa_conv    | full FT  | Strongest single model; also KD teacher |
| `exp1_lorma_custom_mha`      | custom_mha  | LoRMA    | Multiplicative low-rank |
| `exp3_adalora_mla_moe`       | mla_moe     | AdaLoRA  | Adaptive rank |
| `exp4_kd_gqa_alibi_half`     | gqa_alibi   | KD       | Half-depth student distilled from exp2 |

Additional ablations (different chunk policies, pooling variants, longer training) can be added by simply appending more `LSIExp` entries.

---

## 10. How to Run

### 10.1 Prerequisites
```bash
pip install torch tqdm ijson huggingface_hub scikit-learn numpy pandas
```

### 10.2 Path Setup
```python
DATASET_DIR = "/kaggle/working/phase1_models"   # folder containing all .pt + tokenizer files
BACKBONE    = "gqa_conv"                        # default; each experiment overrides it
```

Place the four Phase-1 checkpoints and the four tokenizer files (`legal_bpe_*.json` / `*.pkl`) inside `DATASET_DIR`.

### 10.3 Execution
```python
# 1. Run the shared path / tokenizer / backbone-class cells
# 2. Define the REQUIRED list of LSIExp objects
# 3. Launch training (resumable)
for exp in REQUIRED:
    lsi_run(exp)

# 4. Threshold tuning + submission packaging
#    (cells that load the .npy probability arrays and call make_final_submission)
```

If a checkpoint already exists for an experiment, `lsi_run` skips training and only regenerates predictions if needed.

---

## 11. Evaluation & Analysis

- Macro-F1 / Micro-F1 on the development set for every experiment  
- Per-label F1 vectors (helps identify statutes that remain hard)  
- Comparison of global vs. per-label threshold strategies  
- Optional plots: label-frequency histogram, F1 vs. label frequency, confusion at the label level  

All numeric results are stored as `result_<exp_name>.json` under the Phase-5 output directory.

---

## 12. Submission Generation

After threshold selection the code writes, for each experiment:

```
phase5/retuned_submissions/submission_<exp_name>/
├── prediction.json          # {doc_id: [label_name, …]}
└── submission_tuned.zip
```

`prediction.json` maps every test document ID to the list of statute names whose probability exceeded the chosen threshold(s).

---

## 13. File Layout & Artifacts

```
phase5/
├── cache/
│   └── tok_{backbone}_{mode}_{split}.pkl
├── ckpt/
│   ├── <exp>_latest.pt
│   └── <exp>_best.pt
├── dev_<exp>.npy
├── test_<exp>.npy
├── result_<exp>.json
└── retuned_submissions/
    └── submission_<exp>/
        ├── prediction.json
        └── submission_tuned.zip
```

---

## 14. Reproducibility Notes

- Global seed `SEED = 42` is set for Python, NumPy and PyTorch.  
- Token caches are deterministic given the same tokenizer and cleaning function.  
- Chunk selection (`pick_chunks`) is deterministic.  
- Label priors / pos-weights are computed once from the training fold and frozen.  
- Checkpoints contain full model, optimiser and step counter state → exact resumption.  
- Threshold grid search is deterministic; the same development set is used for every experiment.  
- Each backbone always uses its own tokenizer, guaranteeing that embedding indices stay in-range and match pre-training.

---