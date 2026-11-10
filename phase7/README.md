
# CS781 Phase-7: Legal Case Retrieval (Precedent / Similar-Case Retrieval)

**Course:** CS781 Capstone  
**Phase:** Phase-7 (Downstream Retrieval – Dense Legal Case Retrieval)  
**Task:** Retrieve the most relevant prior cases / precedents for a given query judgment  
**Backbones:** All four Phase-1 foundation models (`custom_mha`, `mla_moe`, `gqa_alibi`, `gqa_conv`)

This repository consolidates the complete Phase-7 pipelines for every Phase-1 backbone. Each experiment reloads the matching Phase-1 checkpoint and tokenizer, encodes queries and candidate cases into dense vectors (with optional multi-chunk MaxSim scoring), and trains under a variety of parameter-efficient regimes (full fine-tuning, LoRMA, knowledge-distillation students, …). Retrieval quality is measured by Precision / Recall / F1 at rank 7; the best scoring variant (mean-pooling vs MaxSim) is selected on the development set and used to produce the official submission.

---

## Table of Contents
1. [Task Definition](#1-task-definition)
2. [Data Pipeline](#2-data-pipeline)
3. [Document & Query Encoding](#3-document--query-encoding)
4. [Scoring Functions](#4-scoring-functions)
5. [Model Architecture (Phase-7)](#5-model-architecture-phase-7)
6. [Parameter-Efficient Fine-Tuning Methods](#6-parameter-efficient-fine-tuning-methods)
7. [Training Objective](#7-training-objective)
8. [Knowledge Distillation](#8-knowledge-distillation)
9. [Experimental Protocol](#9-experimental-protocol)
10. [How to Run](#10-how-to-run)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Submission Generation](#12-submission-generation)
13. [File Layout & Artifacts](#13-file-layout--artifacts)
14. [Reproducibility Notes](#14-reproducibility-notes)

---

## 1. Task Definition

### 1.1 Input / Output
- **Query:** A legal judgment (or a condensed version of it).
- **Candidate pool:** A large collection of prior cases.
- **Output:** A ranked list of the top-k most relevant candidates (official metric uses cut-off 7).

### 1.2 Evaluation Metric
Primary metric reported by the course server and used for model selection:

- **Precision@7, Recall@7, F1@7**

Both a simple mean-pooled cosine score and a late-interaction MaxSim score are computed; the variant that yields the higher development F1@7 is retained for the test submission.

---

## 2. Data Pipeline

### 2.1 Source
Hugging Face dataset `Exploration-Lab/CS-781-Capstone` (Phase-7 retrieval splits).  
The loader yields:

```python
TOK["train"/"dev"/"test"] = {
    "q": list of query token-id arrays,
    "c": list of candidate token-id arrays,
    ...
}
```

### 2.2 Cleaning & Tokenisation
The identical conservative legal text cleaner developed in Phase-1 is applied.  
Each backbone uses **its own BPE tokenizer** (different merge tables / vocabulary sizes). Tokenisation results are cached on disk so that repeated experiments never re-encode the corpus.

### 2.3 Negative Sampling
During training a mixture of negatives is used:

- in-batch negatives,
- randomly sampled hard negatives,
- optional BM25-retrieved negatives (`NEG_MODE = "bm25"`).

The negative strategy is controlled by a global flag and can be ablated.

---

## 3. Document & Query Encoding

Long documents are split into fixed-length chunks (default `INFER_CHUNKS` / `DEV_EVAL_CHUNKS`).  
Each chunk is passed through the backbone; the resulting token representations are pooled according to the chosen strategy:

| Pooling mode | Description |
|--------------|-------------|
| `mean`       | mean of non-padding token vectors |
| `last`       | vector of the final non-padding token |
| `mean_last`  | concatenation of mean and last (then projected) |

```python
def encode_corpus(model, token_lists, n_chunks, pool_modes):
    # returns a dict mode → (vectors, masks) for every requested pooling
```

For efficiency the same encoder is used for both queries and candidates; only the subsequent scoring function differs.

---

## 4. Scoring Functions

Two complementary scoring functions are implemented:

### 4.1 Mean-pooled Cosine
```python
def mean_scores(q_vec, q_mask, c_vec, c_mask):
    # L2-normalise → matrix of cosine similarities
```

### 4.2 Late-Interaction MaxSim (ColBERT-style)
```python
def maxsim_scores(q_tok, q_mask, c_tok, c_mask):
    # for every query token take the maximum similarity
    # over all candidate tokens, then average
```

Both score matrices are evaluated on the development set; the higher-F1@7 variant is selected automatically inside `finalize`.

---

## 5. Model Architecture (Phase-7)

```
Query / Candidate text
        │
        ▼
Chunking (fixed-length windows)
        │
        ▼
┌──────────────────────────────────────────────┐
│  Phase-1 Backbone (shared encoder)           │
│  • Token embedding                           │
│  • 8 × Transformer blocks                    │
│  • Final RMSNorm                             │
└──────────────────────────────────────────────┘
        │  token / chunk vectors
        ▼
Pooling (mean / last / mean_last)
        │
        ▼
Dense representation(s)  →  scoring (cosine or MaxSim)
```

The backbone can be used frozen (zero-shot), fully fine-tuned, or adapted with PEFT modules.  
For the KD student a shallower (half-depth) copy of the backbone is trained while a strong full-depth teacher supplies embedding targets.

---

## 6. Parameter-Efficient Fine-Tuning Methods

| Method | Description | Typical experiment name |
|--------|-------------|-------------------------|
| **Zero-shot** | Frozen Phase-1 encoder | `zeroshot` |
| **Full FT** | All parameters updated | `full` |
| **LoRMA** | Multiplicative low-rank adapters on attention & FFN | `lorma` |
| **Partial FT** | Only top layers / selected modules | `partial` |
| **KD Student** | Distilled shallower student | `kd_student` |

```python
METHOD_CFG = {
    "full":   dict(lr=…, epochs=…),
    "lorma":  dict(r=8, alpha=16, lr=…),
    "kd":     dict(lr=…, epochs=…),
    …
}
```

Only the parameters returned by `trainable_state(model, names)` are optimised and stored in checkpoints, keeping adapter checkpoints small.

---

## 7. Training Objective

Two loss families are supported (selected by `LOSS_KIND`):

- **InfoNCE** (default) – in-batch contrastive loss with temperature.
- **Triplet** – classic triplet margin loss with hard-negative mining.

```python
def step_loss(model, batch, teacher=None):
    # encodes queries & candidates, computes contrastive / triplet loss
    # when teacher is supplied an additional MSE / cosine distillation term
    # is added on the embedding space
```

Gradient clipping, mixed-precision (`bfloat16` preferred) and a cosine learning-rate schedule with linear warm-up are used throughout.

---

## 8. Knowledge Distillation

1. A strong teacher (usually the best full-FT or LoRMA run) is selected by development F1@7.
2. Teacher embeddings for the whole training set are pre-computed and cached:
   ```python
   teacher_tensors = build_teacher_tensors(teacher_exp, teacher_method)
   # → {"q": …, "c": …} stored under P7_CKPT/teacher__*.pt
   ```
3. A student (often a half-depth backbone) is trained with a combination of the original contrastive loss and an embedding-space distillation loss toward the teacher vectors.

```python
run_experiment("kd_student", "kd", teacher=teacher_tensors)
```

---

## 9. Experimental Protocol

### 9.1 Required Experiments
The course rubric asks for at least:

| Experiment   | Method   | Notes |
|--------------|----------|-------|
| `full`       | full FT  | Upper-bound performance |
| `lorma`      | LoRMA    | Parameter-efficient multiplicative adapters |
| `kd_student` | KD       | Distilled student from the strongest teacher |

Additional ablations (random vs BM25 negatives, InfoNCE vs triplet, different pooling) can be added by simply changing the global flags and calling `run_experiment` with a new name.

### 9.2 Training Loop
```python
def train_model(model, exp, cfg, teacher=None):
    # AdamW + cosine schedule + GradScaler
    # resumes automatically from ckpt_path(exp, "last")
    # keeps the best checkpoint by quick dev F1@7
```

### 9.3 Finalisation
```python
def finalize(model, exp, method, total, trainable, history):
    # encodes dev & test
    # scores with both mean and MaxSim
    # selects the better variant
    # writes submission and records a RESULTS row
```

---

## 10. How to Run

### 10.1 Prerequisites
```bash
pip install torch tqdm ijson huggingface_hub numpy scikit-learn pandas
```

### 10.2 Path Setup
```python
DATASET_DIR = "/kaggle/working/phase1_models"   # all .pt + tokenizer files
BACKBONE    = "mla_moe"                         # or custom_mha / gqa_conv / gqa_alibi
```

Place the four Phase-1 checkpoints and the four tokenizer files inside `DATASET_DIR`. The helper `_first` automatically picks the first existing path for each backbone.

### 10.3 Execution
```python
# 1. Run shared imports, path resolution, tokenizer & backbone-class cells
# 2. (Optional) zero-shot baseline
# run_experiment("zeroshot", "zeroshot")

# 3. Required PEFT / full-FT runs
for exp, method in [("full", "full"), ("lorma", "lorma")]:
    run_experiment(exp, method)

# 4. Build teacher tensors from the strongest run and train the KD student
teacher_exp = max(candidates, key=lambda e: RESULTS[tagged(e)]["dev_f1"])
teacher_tensors = build_teacher_tensors(teacher_exp, …)
run_experiment("kd_student", "kd", teacher=teacher_tensors)
```

If a result for a given experiment name already exists in `RESULTS` and the corresponding test scores are on disk, the run is skipped (unless `force=True`).

---

## 11. Evaluation Metrics

For every experiment the following numbers are recorded:

- Development Precision@7 / Recall@7 / F1@7 (both mean-pooled and MaxSim)
- Which scoring variant was selected as best
- Wall-clock seconds to encode the development set
- Peak GPU memory (GB)
- Total vs trainable parameter counts
- Full training history (loss & dev F1 per epoch)

All of the above are stored in a single `RESULTS` dictionary and periodically flushed to disk.

---

## 12. Submission Generation

```python
def write_submission(tag, test_scores):
    # produces a ranked list of candidate IDs for every test query
    # writes prediction.json and packages it into a zip
```

The submission that corresponds to the higher development F1@7 (mean vs MaxSim) is the one retained for official evaluation.

---

## 13. File Layout & Artifacts

```
phase7/   (or cs781_phase7/)
├── ckpt/
│   ├── <exp>_best.pt
│   ├── <exp>_last.pt
│   └── teacher__<exp>.pt
├── scores/
│   ├── <exp>_dev.npy
│   └── <exp>_test.npy
├── results.json
└── submissions/
    └── submission_<exp>.zip
phase1_models/
├── legal_lm_*.pt
├── legal_bpe_*.json / *.pkl
└── …
```

---

## 14. Reproducibility Notes

- Global seed `SEED = 42` is set for Python, NumPy and PyTorch.  
- Token caches are deterministic given the same tokenizer and cleaning function.  
- Negative sampling can be made deterministic by fixing the RNG state before each epoch.  
- Checkpoints store full trainable state, optimiser state, scheduler state and epoch counter → exact resumption.  
- Both scoring functions are pure NumPy / PyTorch and produce bit-identical results for identical embeddings.  
- Each backbone always uses its own tokenizer, guaranteeing that embedding indices stay in-range and match pre-training.

---