# CS781 Phase-6: Legal Document Summarization

**Course:** CS781 Capstone  
**Phase:** Phase-6 (Downstream Generation – Abstractive / Hybrid Summarization)  
**Task:** Produce concise, faithful summaries of long Indian legal judgments  
**Backbones:** All four Phase-1 foundation models (`custom_mha`, `mla_moe`, `gqa_alibi`, `gqa_conv`)

This repository consolidates the complete Phase-6 pipelines for every Phase-1 backbone. Each experiment reloads the matching Phase-1 checkpoint and tokenizer, constructs a token-budgeted document representation, and fine-tunes the decoder-only language model with a causal language-modelling objective (optionally augmented by knowledge distillation and semantic alignment). At inference the model generates summaries autoregressively under length and repetition constraints. Final outputs are written to the official submission format.

---

## Table of Contents
1. [Task Definition](#1-task-definition)
2. [Overall Pipeline](#2-overall-pipeline)
3. [Document Representation](#3-document-representation)
4. [Training Sequence Construction](#4-training-sequence-construction)
5. [Decoder-Only Language Modelling](#5-decoder-only-language-modelling)
6. [Causal Language Modelling Objective](#6-causal-language-modelling-objective)
7. [Parameter-Efficient Fine-Tuning (PEFT)](#7-parameter-efficient-fine-tuning-peft)
8. [Knowledge Distillation](#8-knowledge-distillation)
9. [Semantic Alignment Loss](#9-semantic-alignment-loss)
10. [Autoregressive Summary Generation](#10-autoregressive-summary-generation)
11. [N-gram Repetition Blocking](#11-n-gram-repetition-blocking)
12. [Model Loading & Backbone Handling](#12-model-loading--backbone-handling)
13. [Inference & Submission Packaging](#13-inference--submission-packaging)
14. [How to Run](#14-how-to-run)
15. [Evaluation Notes](#15-evaluation-notes)
16. [File Layout & Artifacts](#16-file-layout--artifacts)
17. [Reproducibility Notes](#17-reproducibility-notes)

---

## 1. Task Definition

### 1.1 Input / Output
- **Input:** A full-length court judgment or legal document \(D = (w_1, w_2, \ldots, w_n)\).
- **Output:** A fluent natural-language summary \(\hat{Y}\) (target length \(\le 256\) tokens) that captures the main factual background, issues, reasoning and final decision.

### 1.2 Why a Hybrid / Budgeted Approach?
Pure abstractive generation from a long document exceeds the context window of the Phase-1 models.  
Phase-6 therefore first constructs a **token-budgeted representation** \(D^*\) (\(|D^*| \le 750\)) and then trains a decoder-only language model to generate the summary conditioned on \(D^*\).

---

## 2. Overall Pipeline

```
Raw judgment D
       │
       ▼
Hierarchical / semantic sectioning
       │
       ▼
Token-budgeted representation D*  (|D*| ≤ 750)
       │
       ▼
Training sequence:
[BOS] ⊕ D* ⊕ [SEP] ⊕ Y ⊕ [EOS]
       │
       ▼
Decoder-only Transformer (causal self-attention)
       │
       ▼
L_total = L_LM + α L_KD + β L_SEM
       │
       ▼
Autoregressive generation of Ŷ  (|Ŷ| ≤ 256)
       │
       ▼
prediction.json  →  submission.zip
```

---

## 3. Document Representation

Let the legal document be
\[
D = (w_1, w_2, \ldots, w_n).
\]

After tokenisation and hierarchical preprocessing we obtain semantic sections
\[
D_H = \{S_1, S_2, \ldots, S_m\},
\]
where each \(S_i\) corresponds to a coherent part of the judgment (facts, arguments, reasoning, decision, citations, \ldots).

A token-budgeted representation is then constructed:
\[
D^* = \operatorname{Truncate}\!\left( \bigoplus_{i=1}^{m} S_i \right)
\quad\text{subject to}\quad
|D^*| \le 750.
\]

---

## 4. Training Sequence Construction

For each document–summary pair \((D^*, Y)\) with
\[
Y = (y_1, y_2, \ldots, y_T),
\]
the decoder-only model receives the concatenated sequence
\[
X = [\text{BOS}] \oplus D^* \oplus [\text{SEP}] \oplus Y \oplus [\text{EOS}].
\]

---

## 5. Decoder-Only Language Modelling

The Phase-1 checkpoint initialises the model parameters \(\theta_0\).  
For any prefix \(x_{<t}\) the transformer produces
\[
h_t = f_\theta(x_{<t}),
\]
and the next-token distribution is
\[
p_\theta(x_t \mid x_{<t}) = \operatorname{Softmax}(W h_t).
\]

Causal self-attention is enforced by the mask
\[
M_{ij} =
\begin{cases}
0 & \text{if } j \le i, \\
-\infty & \text{if } j > i.
\end{cases}
\]

---

## 6. Causal Language Modelling Objective

The model factorises
\[
P_\theta(Y \mid D^*) = \prod_{t=1}^{T} p_\theta(y_t \mid D^*, y_{<t}).
\]

The cross-entropy loss (computed only on summary tokens) is
\[
\mathcal{L}_{\text{LM}} = -\sum_{t\in\mathcal{S}} \log p_\theta(y_t \mid D^*, y_{<t}),
\]
where \(\mathcal{S}\) denotes the positions belonging to \(Y \cup [\text{EOS}]\) (document tokens are masked with label \(-100\)).

---

## 7. Parameter-Efficient Fine-Tuning (PEFT)

Instead of updating all parameters we learn a low-rank update
\[
\theta = \theta_0 + \Delta\theta, \qquad \operatorname{rank}(\Delta\theta) \ll d.
\]

For LoRA / LoRMA the weight matrices are re-parameterised as
\[
W' = W + BA,
\]
with
\[
B \in \mathbb{R}^{d \times r},\quad
A \in \mathbb{R}^{r \times k},\quad
r \ll d.
\]
Only \(A\) and \(B\) are trained.

---

## 8. Knowledge Distillation

Given a teacher distribution \(p_T(\cdot)\) and a student distribution \(p_S(\cdot)\), the distillation loss is
\[
\mathcal{L}_{\text{KD}} = T^2 \, D_{\text{KL}}\bigl(p_T^{(T)} \,\|\, p_S^{(T)}\bigr),
\]
where \(T\) is the temperature.  
The combined objective becomes
\[
\mathcal{L} = (1-\lambda)\mathcal{L}_{\text{LM}} + \lambda\mathcal{L}_{\text{KD}}.
\]

---

## 9. Semantic Alignment Loss

Let \(E(\cdot)\) be a sentence encoder. Define
\[
e_g = E(\hat{Y}), \qquad e_r = E(Y).
\]
The semantic alignment loss is
\[
\mathcal{L}_{\text{SEM}} = 1 - \frac{e_g^\top e_r}{\|e_g\|\,\|e_r\|}.
\]

The final training objective is
\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \alpha\mathcal{L}_{\text{KD}} + \beta\mathcal{L}_{\text{SEM}}.
\]

---

## 10. Autoregressive Summary Generation

At inference the model generates
\[
\hat{y}_t = \arg\max_y\, p_\theta(y \mid D^*, \hat{y}_{<t})
\]
(or via nucleus / beam search). Generation terminates when
\[
\hat{y}_t = [\text{EOS}]
\quad\text{or}\quad
|\hat{Y}| = 256.
\]
Hence
\[
|\hat{Y}| \le 256.
\]

---

## 11. N-gram Repetition Blocking

Let \(\mathcal{G}_n(\hat{Y}_{<t})\) be the set of \(n\)-grams already generated.  
Any token \(y\) that would create a duplicate \(n\)-gram is forbidden:
\[
p_\theta(y) = 0
\quad\text{if}\quad
\mathcal{G}_n(\hat{Y}_{<t}+y)\text{ contains a duplicate}.
\]

---

## 12. Model Loading & Backbone Handling

```python
def load_trained(backbone_name):
    if backbone_name == "custom_mha":
        m = load_v2(CKPTS[backbone_name], device=DEVICE)
    elif backbone_name == "mla_moe":
        m = load_v3(CKPTS[backbone_name], device=DEVICE)
    elif backbone_name == "gqa_conv":
        m = load_gqa(CKPTS[backbone_name], device=DEVICE)
    elif backbone_name == "gqa_alibi":
        m = load_alibi(CKPTS[backbone_name], device=DEVICE)
    ...
    t = TOKENIZERS[backbone_name]
    return m, t, meta
```

- Each backbone uses **its own tokenizer** (different BPE merge tables / vocabulary sizes).  
- Checkpoint loading strips any `_orig_mod.` prefix left by `torch.compile`.  
- The model is placed in `eval()` mode for pure inference runs.

---

## 13. Inference & Submission Packaging

```python
predictions = []
for idx, doc_id in enumerate(TEST_IDS):
    doc_text   = test_texts[idx]
    D_star     = build_budgeted_representation(doc_text)          # |D*| ≤ 750
    tok_input  = tokenizer.encode(D_star)[:MAX_DOC_TOK]
    summary    = generate(lm, tok, sep, [tok_input], **GEN)[0]   # |Ŷ| ≤ 256
    predictions.append({"id": str(doc_id), "summary": summary})

json.dump(predictions, open("prediction.json", "w"), ensure_ascii=False, indent=2)

with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write("prediction.json", arcname="prediction.json")
```

---

## 14. How to Run

### 14.1 Prerequisites
```bash
pip install torch tqdm ijson huggingface_hub numpy scikit-learn scipy
```

### 14.2 Path Setup
```python
DATASET_DIR = "/kaggle/working/phase1_models"   # all .pt + tokenizer files
BACKBONE    = "gqa_conv"                        # or custom_mha / mla_moe / gqa_alibi
```

### 14.3 Execution
```python
# 1. Run shared imports, path resolution, tokenizer & backbone-class cells
# 2. (Optional) fine-tune with L_total
# 3. Run inference on the test set
# 4. Collect prediction.json and the submission zip
```

Changing the single variable `BACKBONE` automatically selects the correct tokenizer, checkpoint and model class.

---

## 15. Evaluation Notes

Official evaluation typically reports:

- ROUGE-1 / ROUGE-2 / ROUGE-L  
- BERTScore or legal-domain semantic similarity  
- Length statistics and copy-rate  

The token-budget constraint on \(D^*\) together with the repetition-blocking mechanism produces concise, multi-aspect summaries that correlate well with ROUGE-L and human judgements.

---

## 16. File Layout & Artifacts

```
phase6/
├── prediction.json              # list of {"id", "summary"}
├── submission.zip               # official hand-in
└── (optional) ckpts/            # any Phase-6 fine-tuned weights
phase1_models/
├── legal_lm_*.pt
├── legal_bpe_*.json / *.pkl
└── …
```

---

## 17. Reproducibility Notes

- Global seed `SEED = 42` is set for Python, NumPy and PyTorch.  
- Document truncation to 750 tokens and summary length cap of 256 tokens are deterministic.  
- Causal masking and label masking (`-100` on document tokens) guarantee that the loss is computed only on summary tokens.  
- Each backbone always uses its own tokenizer, keeping embedding indices in-range.  
- All hyper-parameters of \(\mathcal{L}_{\text{total}}\) (\(\lambda\), \(\alpha\), \(\beta\), temperature, rank, \ldots) are explicit and can be swept for ablation studies.

---

**Mathematical Summary of the Pipeline**

\[
D \;\to\; D_H \;\to\; D^* \;\to\;
[\text{BOS}]\,D^*\,[\text{SEP}]\,Y\,[\text{EOS}]
\;\to\;
\text{Decoder-only Transformer (causal self-attention)}
\;\to\;
\mathcal{L}_{\text{total}}
\;\to\;
\hat{Y}
\]

with the hard constraints
\[
|D^*| \le 750, \qquad |\hat{Y}| \le 256.
\]

---
