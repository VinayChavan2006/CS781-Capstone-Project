```markdown
# CS781 Phase-3: Legal Rhetorical Role (RR) Classification

**Course:** CS781 Capstone  
**Phase:** Phase-3 (Downstream Adaptation – Rhetorical Role / Document Classification)  
**Task:** Sentence-level Rhetorical Role labelling of Indian legal judgments  
**Domains:** CL (Case Law) and IT (Income Tax / Tribunal)  
**Backbones:** All four Phase-1 foundation models (`custom_mha`, `mla_moe`, `gqa_alibi`, `gqa_conv`)

This repository consolidates the complete Phase-3 pipelines for every Phase-1 backbone. Each architecture section reloads the corresponding Phase-1 checkpoint + tokenizer, wraps the decoder as a bidirectional sentence encoder, attaches a classification head (optionally with document-level BiGRU context), and runs a full matrix of parameter-efficient fine-tuning (PEFT), loss-function, context-modelling and knowledge-distillation experiments. Training is resumable; evaluation reports macro-F1 on both domains; the best models are packaged into official submission zips.

---

## Table of Contents
1. [Task Definition & Label Schema](#1-task-definition--label-schema)
2. [Data Pipeline](#2-data-pipeline)
3. [Model Architecture (Phase-3)](#3-model-architecture-phase-3)
4. [Context Modelling Variants](#4-context-modelling-variants)
5. [Parameter-Efficient Fine-Tuning Methods](#5-parameter-efficient-fine-tuning-methods)
6. [Loss Functions](#6-loss-functions)
7. [Knowledge Distillation](#7-knowledge-distillation)
8. [Training Methodology](#8-training-methodology)
9. [Experimental Matrix](#9-experimental-matrix)
10. [How to Run](#10-how-to-run)
11. [Evaluation Metrics & Analysis](#11-evaluation-metrics--analysis)
12. [Submission Generation](#12-submission-generation)
13. [File Layout & Artifacts](#13-file-layout--artifacts)
14. [Reproducibility Notes](#14-reproducibility-notes)

---

## 1. Task Definition & Label Schema

### 1.1 Input / Output
- **Input:** A legal document already segmented into sentences.
- **Output:** One rhetorical-role label per sentence.

### 1.2 Domains
| Domain code | Description |
|-------------|-------------|
| **CL** | Case-law judgments (Supreme Court / High Courts) |
| **IT** | Income-tax / tribunal orders |

Each domain has its own train / dev / test split.

### 1.3 Label Vocabulary
Labels are loaded from `Phase_3_RR/Train/label_vocab.json`.  
Typical rhetorical roles (exact list is data-driven):

```
Fact, ArgumentPetitioner, ArgumentRespondent,
Statute, PrecedentReliedUpon, PrecedentNotReliedUpon,
RatioOfTheDecision, RulingByLowerCourt, RulingByPresentCourt,
Issue, None, …
```

```python
LABELS = [...]                     # ordered list from label_vocab.json
L2I    = {l: i for i, l in enumerate(LABELS)}
NL     = len(LABELS)
```

Every sentence receives exactly one of these discrete labels.

---

## 2. Data Pipeline

### 2.1 Source
Hugging Face dataset `Exploration-Lab/CS-781-Capstone`:

```
Phase_3_RR/
├── Train/
│   ├── label_vocab.json
│   ├── CL_train.json / CL_dev.json
│   └── IT_train.json / IT_dev.json
└── Test/
    ├── CL_test.json
    └── IT_test.json
```

### 2.2 Document Loading
```python
def load_docs(path, domain, labelled=True):
    # Supports both dict-of-docs and list-of-docs JSON layouts.
    # Text is cleaned with the same conservative legal cleaner used in Phase-1.
    # Sentences are obtained either from a pre-split list or by splitting on newlines.
    # Gold labels (when present) are mapped through L2I.
```

### 2.3 Tokenisation
Each sentence is encoded independently with the Phase-1 BPE tokenizer of the active backbone (24 k / 30 k / 32 k vocabulary).  
A special `[BOS]` token is prepended.  Sentences longer than the chosen context window are truncated; shorter ones are padded.

### 2.4 Batching
A collate function packs variable-length sentences into a single tensor together with:
- `ids` – token ids  
- `st`, `en` – start / end offsets of each sentence inside the packed sequence  
- `sm` – attention / padding mask  
- `y` – gold label indices (or `-100` for test)

---

## 3. Model Architecture (Phase-3)

### 3.1 High-level Stack
```
Sentence tokens
      │
      ▼
┌─────────────────────────────────────────────┐
│  Phase-1 Backbone (LMBackbone wrapper)      │
│  • Token embedding                          │
│  • 8 × Transformer blocks (bidirectional)   │
│  • Final RMSNorm                            │
└─────────────────────────────────────────────┘
      │  hidden states [B, T, d]
      ▼
Pooling (last-token / mean / last+mean)
      │
      ▼
Projection (LayerNorm → Dropout → Linear → GELU)
      │
      ▼
Optional Document-level BiGRU context
      │
      ▼
Classification head  →  logits [B, NL]
```

### 3.2 Backbone Wrapper
```python
class LMBackbone(nn.Module):
    def __init__(self, lm):
        self.wte, self.blocks, self.ln_f = lm.wte, lm.blocks, lm.ln_f
        self.dim = lm.cfg["dim"]          # or N_EMBD

    def hidden(self, x, gc=False):
        # Runs every block; optional gradient checkpointing
        for b in self.blocks:
            x = checkpoint(b.train_fwd, x) if gc else b(x)[0]
        return self.ln_f(x)
```

The original causal mask of the Phase-1 decoder is replaced by a **bidirectional** mask (plus padding mask) so that every token inside a sentence can attend to the whole sentence.

### 3.3 Classification Head (`RRModel`)
```python
class RRModel(nn.Module):
    def __init__(self, bb, n_labels, head="bigru", pool="last_mean",
                 n_prompt=0, drop=0.1):
        ...
        self.proj = nn.Sequential(LayerNorm, Dropout, Linear(d→d), GELU)
        self.ctx  = nn.GRU(d, d//2, bidirectional=True) if head=="bigru" else None
        self.cls  = nn.Sequential(Dropout, Linear(d, n_labels))
        # optional soft prompt tokens
        if n_prompt:
            self.prompt = nn.Parameter(...)
```

Pooling options:
- `last` – final token  
- `mean` – mean of non-pad tokens  
- `last_mean` – concatenation of last + mean (then projected)

When `head="bigru"` the projected sentence vectors of a whole document are passed through a bidirectional GRU; the GRU outputs are fed to the final linear classifier.  This realises the **Ctx-Doc BiGRU** setting.

### 3.4 Soft Prompts
An optional set of continuous prompt tokens can be prepended to every sentence (or every document).  They are initialised from random rows of the embedding matrix and trained end-to-end.

---

## 4. Context Modelling Variants

| Name | Description |
|------|-------------|
| `sent` | Each sentence is encoded in isolation (default). |
| `ctx_doc_bigru` | All sentences of a document are encoded, then a BiGRU mixes document-level context before classification. |
| `ctx_window` | A sliding window of neighbouring sentences is concatenated (used in some ablation runs). |

The BiGRU context model is the strongest single-model variant for both CL and IT.

---

## 5. Parameter-Efficient Fine-Tuning Methods

All methods share the same `RRModel` interface; only the set of trainable parameters changes.

| Method | Trainable components | Typical # trainable params |
|--------|----------------------|----------------------------|
| **Full FT** (`m_full`) | Entire backbone + head | ~66–78 M |
| **LoRA** (`m_lora`) | Low-rank adapters on selected Linear layers + head | 0.5–2 M |
| **rsLoRA** (`m_rslora`) | Rank-stabilised LoRA (scaling by \(1/\sqrt{r}\)) + head | 0.5–2 M |
| **QLoRA** (`m_qlora`) | 4-bit quantised base + LoRA adapters + head | < 1 M |
| **LoRMA** (`m_lorma`) | Multiplicative low-rank adapters (output-side) + head | 0.5–2 M |
| **Linear probe** | Head only (backbone frozen) | ~10 k |

Injection targets cover attention projections (`q`, `k`, `v`, `o`, `qkv`, …) and the feed-forward / Dual-Path matrices (`w13`, `w2`, …).  Expert modules inside MoE layers are left untouched.

---

## 6. Loss Functions

| Name | Formula / Notes |
|------|-----------------|
| **Cross-Entropy** (`loss_ce`) | Standard multi-class CE |
| **Focal Loss** (`loss_focal`) | \(\mathrm{FL}(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t\) with \(\gamma=2\), \(\alpha\) inverse-frequency or uniform |
| **Label-smoothed CE** | Optional smoothing 0.1 (used in some KD runs) |

Class weights can be computed from the training fold and applied to any of the above losses.

---

## 7. Knowledge Distillation

A strong teacher (usually a fully fine-tuned or high-capacity PEFT model) produces soft logits.  
The student is trained with a mixture:

```
L = α · CE(student, gold) + (1-α) · T² · KL( student/T  ‖  teacher/T )
```

Typical hyper-parameters: \(T=2.0\), \(\alpha=0.5\).  
The distilled student is stored under the experiment name `kd_student_logits` and is always included in the final submission set.

---

## 8. Training Methodology

### 8.1 Optimiser & Schedule
- AdamW with two (or three) parameter groups:
  - backbone / PEFT parameters → base `lr`
  - classification head → higher `head_lr` (usually 5–10×)
  - optional prompt tokens → same as head
- Linear warm-up (≈10 % of steps) followed by cosine decay
- Gradient clipping at 1.0
- Mixed precision (`bfloat16` preferred)

### 8.2 Resumption
Every experiment writes:
```
ckpts/<experiment_name>_ckpt.pt   # model + optimiser + scaler + epoch/step
ckpts/<experiment_name>.pt        # best validation macro-F1 weights
```
Re-running the same cell automatically resumes from the latest checkpoint.

### 8.3 Wall-clock Cap
A soft limit `MAX_MIN_PER_RUN` (default 45 min) aborts an experiment gracefully and still records the best checkpoint obtained so far.

### 8.4 Typical Hyper-parameters
```python
EPOCHS          = 4–8
BATCH_SIZE      = 8–16 (sentence-level)
LR (backbone)   = 1e-5 – 5e-5
HEAD_LR         = 1e-3
WEIGHT_DECAY    = 0.01
LoRA / rsLoRA r = 8 or 16
Focal γ         = 2.0
```

---

## 9. Experimental Matrix

The notebook executes a curated matrix that is required by the course rubric.  
The exact set of runs differs slightly per backbone (see the top-level documentation cell), but the union covers:

**Methods**  
`m_full`, `m_lora`, `m_rslora`, `m_qlora`, `m_lorma`, `linear_probe`

**Losses**  
`loss_ce`, `loss_focal`

**Context**  
`sent`, `ctx_doc_bigru`

**Distillation**  
`kd_student_logits`

**Longer training**  
`m_full_ep8`, `m_rslora_ep8`

**Ensemble**  
A simple TAPT-style ensemble of the strongest models is also produced.

All results are collected into a single `results_df` (columns: name, group, method, ctx, loss, F1_CL, F1_IT, F1_mean, …) and visualised with learning curves, per-class F1 bars, confusion matrices and efficiency scatter plots.

---

## 10. How to Run

### 10.1 Prerequisites
```bash
pip install torch tqdm ijson huggingface_hub scikit-learn pandas matplotlib numpy
```

### 10.2 Path Setup
1. Place Phase-1 tokenizers and checkpoints under `DATASET_DIR` / `PHASE1_DIR` (default `/kaggle/working/cs781_phase1`).
2. Run the shared path-resolution cell; it must print `exists=True` for the desired backbone.

### 10.3 Per-Backbone Workflow
```python
# 1. Activate backbone
BACKBONE = "gqa_conv"          # or custom_mha / mla_moe / gqa_alibi
# … run activation cell …

# 2. Load Phase-1 classes (tokenizer + LegalLM / DualPathLM / …)
# 3. Load RR data + label maps
# 4. Build model under a chosen regime
model = build_rr(method="rslora", head="bigru", pool="last_mean", r=8)

# 5. Train (resumable)
run_exp(name="m_rslora", model=model, epochs=6, lr=2e-5, head_lr=1e-3)

# 6. Predict on both test domains
predict_test(model, name="m_rslora")

# 7. Aggregate results & generate submission zips
```

### 10.4 Two Execution Pathways
- **Pathway A (recommended)** – use the supplied Phase-1 checkpoints; simply “Run All”.
- **Pathway B** – train tokenizers / Phase-1 models from scratch first, then proceed with the Phase-3 cells.

---

## 11. Evaluation Metrics & Analysis

### 11.1 Primary Metric
**Macro-F1** averaged across the two domains:
```
F1_mean = (F1_CL + F1_IT) / 2
```
Macro-F1 is also reported per domain and per class.

### 11.2 Secondary Analyses
- Confusion matrices (CL / IT)
- Within-group confusion share (Precedent*, Arguments, Reasoning/Ruling groups)
- Learning curves (train loss, dev macro-F1)
- Parameter-efficiency scatter (trainable params vs. F1_mean)
- Error inspection (most frequent true→pred confusions)

---

## 12. Submission Generation

After all experiments finish, the notebook automatically selects the four strongest runs that use distinct techniques (always including the KD student) and writes:

```
submission_<name>/
├── prediction.json      # {doc_id: [label, …]} for both CL & IT test sets
└── submission.zip
```

The selection logic can be overridden by manually editing the `SUBMIT` list.

---

## 13. File Layout & Artifacts

```
phase3-<backbone>/
├── ckpts/
│   ├── m_full.pt / m_full_ckpt.pt
│   ├── m_rslora.pt / m_rslora_ckpt.pt
│   ├── kd_student_logits.pt
│   └── …
├── test_preds/
│   ├── m_full_CL.json / m_full_IT.json
│   └── …
├── results.json
├── curves.png / confusion_*.png / perclass_*.png / efficiency.png
└── submission_*/
    ├── prediction.json
    └── submission.zip
```

---

## 14. Reproducibility Notes

- Global seed `SEED = 42` is set for Python, NumPy and PyTorch (including CUDA).
- Document order and sentence order inside a document are deterministic.
- Class weights (when used) are computed once from the training fold and frozen.
- Checkpoints store full model state, optimiser state, GradScaler state and epoch/step counters → exact resumption.
- The same label vocabulary and evaluation script are used for every backbone, guaranteeing fair comparison.

---
