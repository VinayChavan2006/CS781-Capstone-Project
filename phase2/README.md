
# CS781 Phase-2: Legal Named Entity Recognition (LNER)

**Course:** CS781 Capstone  
**Phase:** Phase-2 (Downstream Adaptation – Legal NER)  
**Task:** Token-level Named Entity Recognition on Indian legal documents  
**Backbones:** All four Phase-1 foundation models (`custom_mha`, `mla_moe`, `gqa_alibi`, `gqa_conv`)

This repository consolidates the complete Phase-2 pipelines for every Phase-1 backbone. Each architecture section reloads the corresponding Phase-1 checkpoint + tokenizer, attaches a token-classification head, and runs a full suite of parameter-efficient and full fine-tuning experiments (LoRA / LoRMA / BitFit-style partial updates / Adapters / Full FT / Knowledge Distillation). Training is resumable; evaluation uses BIO-constrained Viterbi decoding; submissions are produced in the official format.

---

## Table of Contents
1. [Task Definition & Label Schema](#1-task-definition--label-schema)
2. [Data Pipeline](#2-data-pipeline)
3. [Model Architecture (Phase-2)](#3-model-architecture-phase-2)
4. [Parameter-Efficient Fine-Tuning Methods](#4-parameter-efficient-fine-tuning-methods)
5. [Training Methodology](#5-training-methodology)
6. [Decoding: Constrained Viterbi](#6-decoding-constrained-viterbi)
7. [Experimental Setup](#7-experimental-setup)
8. [How to Run](#8-how-to-run)
9. [Evaluation Metrics & Analysis](#9-evaluation-metrics--analysis)
10. [File Layout & Artifacts](#10-file-layout--artifacts)
11. [Reproducibility Notes](#11-reproducibility-notes)

---

## 1. Task Definition & Label Schema

### 1.1 Input / Output
- **Input:** Raw legal document text (statutes, judgments, tribunal orders).
- **Output:** Character-offset entity spans with type labels.

### 1.2 Entity Types
Labels are read from `Phase_2_LNER/Train/labels.txt` (exact list provided by the course).  
Typical Indian-legal entity types include (illustrative):

- `COURT`, `PETITIONER`, `RESPONDENT`, `JUDGE`, `LAWYER`
- `STATUTE`, `SECTION`, `ARTICLE`, `CASE_NUMBER`
- `DATE`, `ORG`, `GPE`, `MONEY`, …

### 1.3 BIO Encoding
Every entity type \(L\) produces two tags:
```
B-L   beginning of entity of type L
I-L   inside / continuation of entity of type L
```
plus the outside tag:
```
O
```
Total number of classes:
```python
NL = 1 + 2 * |LABELS|
BIO = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B", "I")]
```

Token-level gold labels are derived by projecting character spans onto the BPE tokenisation of the document (first sub-token of a span receives `B-…`, subsequent sub-tokens receive `I-…`).

---

## 2. Data Pipeline

### 2.1 Source
Hugging Face dataset `Exploration-Lab/CS-781-Capstone`:

```
Phase_2_LNER/
├── Train/
│   ├── labels.txt
│   ├── fold1-text.json   + fold1-gold.json
│   ├── fold2-text.json   + fold2-gold.json
│   └── …
└── Test/
    └── fold3-text.json          # official test set (no gold)
```

- Fold 1 → training  
- Fold 2 → validation  
- Fold 3 → test (submission)

### 2.2 Document → Token Windows
Because documents can exceed the Phase-1 context length (1024 tokens), each document is sliced into overlapping windows:

```python
WIN    = 512          # or 768 / 1024 depending on backbone memory
STRIDE = WIN - 64     # 64-token overlap
```

A special `[BOS]` token (id = 2) is prepended to every window.  
Padding uses id = 0; label padding uses `-100` (ignored by cross-entropy).

### 2.3 Collation
```python
ids  : LongTensor [B, L]     # token ids, left-padded with PAD
y    : LongTensor [B, L]     # BIO ids, -100 for pad / non-first sub-tokens
attn : BoolTensor [B, L]     # True for real tokens
```

---

## 3. Model Architecture (Phase-2)

### 3.1 High-level Stack
```
Input tokens
    │
    ▼
┌──────────────────────────────────────┐
│  Phase-1 Backbone (frozen or tuned)  │
│  • Token embedding                   │
│  • 8 × Transformer blocks            │
│  • Final RMSNorm                     │
└──────────────────────────────────────┘
    │  hidden states  [B, T, d]
    ▼
Dropout
    │
    ▼
┌──────────────────────────────────────┐
│  Token Classification Head           │
│  • Linear   :  d → NL                │
│  • or MLP   :  d → d → GELU → d → NL │
└──────────────────────────────────────┘
    │
    ▼
Logits [B, T, NL]  →  Viterbi / argmax  →  BIO tags  →  entity spans
```

### 3.2 Backbone Wrapper
```python
class Backbone(nn.Module):
    def __init__(self, lm, causal=False, adapter_dim=0):
        ...
    def forward(self, ids, attn_mask):
        # runs the Phase-1 model in non-causal (bidirectional) mode
        # for NER, returning final-layer hidden states
```

- The original causal attention mask is replaced by a **bidirectional** mask (plus padding mask) so that every token can attend to the whole window.
- Optional **Adapter** modules (bottleneck residual) can be inserted after every block when `adapter_dim > 0`.

### 3.3 Classification Head
Two head variants are supported:

| Head type | Architecture |
|-----------|--------------|
| `linear`  | `Linear(d, NL)` |
| `mlp`     | `Linear(d, d) → GELU → Dropout → Linear(d, NL)` |

The head is always trained (even under PEFT regimes).


---

## 4. Parameter-Efficient Fine-Tuning Methods

All methods share the same `NERModel` interface; only the set of trainable parameters changes.

| Method       | Trainable components                                      | Typical # trainable params |
|--------------|-----------------------------------------------------------|----------------------------|
| **Full FT**  | Entire backbone + head                                    | ~66–78 M                   |
| **LoRA**     | Low-rank adapters on selected Linear layers + head        | 0.5–2 M                    |
| **LoRMA**    | Multiplicative low-rank adapters (output-side) + head     | 0.5–2 M                    |
| **Adapter**  | Bottleneck residual adapters after every block + head     | ~1–3 M                     |
| **Partial**  | Last \(k\) transformer blocks + final norm + head         | ~15–25 M                   |
| **BitFit-style** | Bias terms only (when implemented) + head              | < 0.1 M                    |

### 4.1 LoRA / LoRMA Injection Targets

(Exact set depends on the backbone; attention projections and SwiGLU / Dual-Path FFN matrices are covered.)

```python
class LoRALinear(nn.Module):
    # y = W x + (x Aᵀ Bᵀ) * (α / r)
    ...

class LoRMALinear(nn.Module):
    # y = W x + ( (W x) Aᵀ Bᵀ ) * (α / r)   # multiplicative
    ...
```

### 4.2 Adapter Module
```python
class Adapter(nn.Module):
    def __init__(self, d, r=64):
        self.down = nn.Linear(d, r)
        self.up   = nn.Linear(r, d)          # zero-init
    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))
```

---

## 5. Training Methodology

### 5.1 Loss
Token-level cross-entropy with optional class weighting (to counteract the heavy “O” class):

```python
def class_weights(docs, power=0.5, cap=10.0):
    # w_c = clip( (count_O / count_c)^power , 0, cap )
    # w_O = 1.0
```

Knowledge-distillation variant (when a teacher is supplied):
```
L = α · CE(student, gold) + (1-α) · T² · KL(student / T, teacher / T)
```

### 5.2 Optimiser & Schedule
- AdamW, two parameter groups:
  - backbone / PEFT parameters → `lr`
  - classification head → `head_lr` (usually 5–10× higher)
- Linear warm-up (10 % of total steps) → cosine decay
- Gradient clipping at 1.0
- Mixed precision (`bfloat16` preferred)

### 5.3 Resumption
Every experiment writes:
```
checkpoints/<experiment_name>/
├── epoch_XX.pt          # model + optimiser + scaler + step
└── best.pt              # best validation macro-F1
```
Re-running the same cell automatically resumes from the latest checkpoint.

### 5.4 Typical Hyper-parameters
```python
EPOCHS          = 4–8
BATCH_SIZE      = 4–8
LR (backbone)   = 1e-5 – 5e-5
HEAD_LR         = 1e-3
WEIGHT_DECAY    = 0.01
MAX_GRAD_NORM   = 1.0
LoRA rank r     = 8 or 16
Adapter dim     = 64
```

---

## 6. Decoding: Constrained Viterbi

A hard BIO transition matrix is enforced at inference time:

```python
allow[prev, cur] = False  if  cur is I-X and prev is not B-X / I-X
```

Viterbi DP:
```python
dp[t, c] = max_p  dp[t-1, p] + trans[p, c] + logp[t, c]
```

After obtaining the best tag sequence, consecutive `B-X`/`I-X` tokens are merged into character-offset spans using the original tokenisation offsets.

Both **Viterbi** and simple **argmax** decoding are implemented; Viterbi is the default for final submissions.

---

## 7. How to Run

### 7.1 Prerequisites
```bash
pip install torch tqdm ijson huggingface_hub scikit-learn numpy
```

### 7.2 Path Setup
1. Place Phase-1 tokenizers and checkpoints under `DATASET_DIR` (default `/kaggle/working/cs781_phase1`).
2. Run the **Shared path resolution** cell; it must print `exists=True` for the desired backbone.

### 7.3 Per-Backbone Workflow
```python
# 1. Activate backbone
BACKBONE = "gqa_conv"          # or custom_mha / mla_moe / gqa_alibi
# … run activation cell …

# 2. Load Phase-1 classes (tokenizer + LegalLM / DualPathLM / …)
# 3. Load LNER data + build BIO label maps
# 4. Build model under a chosen regime
model = build(kind="lora", head="linear", r=8)

# 5. Train (resumable)
train(model, name="gqa_conv_lora_r8", epochs=6, lr=2e-5, head_lr=1e-3)

# 6. Predict on test set
preds = predict_docs(model, test_docs, decode="viterbi")

# 7. Write submission
json.dump(preds, open("prediction.json", "w"))
```

### 8.4 Bundling
A final cell packages weights, resume checkpoints, plots, and predictions into a zip archive ready for submission / archival.

---

## 9. Evaluation Metrics & Analysis

### 9.1 Primary Metric
**Entity-level micro / macro F1** (exact span + type match) on Fold-2.

### 9.2 Secondary Metrics
- Token-level precision / recall / F1 (per BIO class)
- Confusion matrices
- Error analysis (missing spans, boundary errors, type confusions)

### 9.3 Visualisations
- Training / validation loss & F1 curves
- Parameter-efficiency vs. F1 scatter plot across all regimes
- Per-entity-type F1 bar charts

---

## 10. File Layout & Artifacts

```
phase2-<backbone>/
├── checkpoints/
│   ├── full_ft/
│   ├── lora_r8/
│   └── …
├── predictions/
│   └── test_viterbi.json
├── plots/
│   ├── learning_curves.png
│   └── performance_comparison.png
└── submission.zip
```

---

## 11. Reproducibility Notes

- All random seeds fixed (`SEED = 42`).
- Document-to-window conversion is deterministic.
- Class weights are computed once from the training fold and frozen.
- Checkpoints store full model state, optimiser state, GradScaler state and epoch/step counters → exact resumption.
- The same BIO transition matrix is used for every backbone, guaranteeing fair decoding comparison.

---