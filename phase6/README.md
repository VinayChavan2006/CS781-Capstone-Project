
# CS781 Phase-6: Legal Document Summarization

**Course:** CS781 Capstone  
**Phase:** Phase-6 (Downstream Generation – Abstractive / Hybrid Summarization)  
**Task:** Produce concise, faithful summaries of long Indian legal judgments  
**Backbones:** All four Phase-1 foundation models (`custom_mha`, `mla_moe`, `gqa_alibi`, `gqa_conv`)

This repository consolidates the complete Phase-6 pipelines for every Phase-1 backbone. Each experiment reloads the matching Phase-1 checkpoint and tokenizer, applies a **hybrid extractive pre-compression** stage (TF-IDF + K-Means aspect clustering + MILP sentence selection under a character budget), then performs **abstractive generation** with the causal language model. Decoding uses nucleus sampling with repetition control; final summaries are written to the official submission format.

---

## Table of Contents
1. [Task Definition](#1-task-definition)
2. [Overall Pipeline](#2-overall-pipeline)
3. [Hybrid Extractive Pre-compression](#3-hybrid-extractive-pre-compression)
4. [Abstractive Generation](#4-abstractive-generation)
5. [Model Loading & Backbone Handling](#5-model-loading--backbone-handling)
6. [Decoding Configuration](#6-decoding-configuration)
7. [Training / Adaptation (Optional)](#7-training--adaptation-optional)
8. [Inference & Submission Packaging](#8-inference--submission-packaging)
9. [How to Run](#9-how-to-run)
10. [Evaluation Notes](#10-evaluation-notes)
11. [File Layout & Artifacts](#11-file-layout--artifacts)
12. [Reproducibility Notes](#12-reproducibility-notes)

---

## 1. Task Definition

### 1.1 Input / Output
- **Input:** A full-length court judgment or legal document (often several thousand tokens).
- **Output:** A fluent natural-language summary (target length typically 60–128 tokens) that captures the main factual background, issues, reasoning and final decision.

### 1.2 Why a Hybrid Approach?
Pure abstractive generation from a 1 k-token context window loses information when the source document is much longer.  
Pure extractive selection preserves factual accuracy but produces disjointed text.  
Phase-6 therefore adopts a **two-stage hybrid** design that is common in the legal-summarization literature:

1. **Extractive pre-compression** – select a small set of representative sentences that cover distinct topical aspects while respecting a character budget.  
2. **Abstractive generation** – condition the Phase-1 causal LM on the compressed text and let it produce a coherent, fluent summary.

---

## 2. Overall Pipeline

```
Raw judgment text
       │
       ▼
┌──────────────────────────────────────────────┐
│  Hybrid Extractive Core-Context Extraction   │
│  • Sentence splitting                        │
│  • TF-IDF + K-Means aspect clustering        │
│  • Extended-LexRank-style semantic scoring   │
│  • MILP: pick ≤1 sentence per cluster        │
│    under a character budget                  │
└──────────────────────────────────────────────┘
       │  compressed text (≤ ~1200 characters)
       ▼
Tokenizer (backbone-specific BPE)
       │
       ▼
┌──────────────────────────────────────────────┐
│  Phase-1 Causal LM (frozen or lightly tuned) │
│  Prompt = [BOS] + compressed tokens + SEP    │
│  Generate with nucleus sampling + rep. penalty│
└──────────────────────────────────────────────┘
       │
       ▼
Post-processing (strip prompt, length trim)
       │
       ▼
prediction.json  →  submission.zip
```

---

## 3. Hybrid Extractive Pre-compression

The function `hybrid_extract_core_context` implements the extractive stage.

### 3.1 Sentence Segmentation
```python
sentences = [s.strip() for s in re.split(r'(?<=[.;?!])\s+', text)
             if len(s.strip()) > 15]
```
Very short fragments are discarded.

### 3.2 Aspect Clustering
- Vectorise sentences with `TfidfVectorizer(stop_words='english')`.
- Run `KMeans(n_clusters=num_clusters, random_state=42)` (default `num_clusters=4`).
- Each cluster is interpreted as a distinct topical aspect of the judgment (facts, issues, reasoning, decision, …).

### 3.3 Semantic Scoring (Extended LexRank style)
For every sentence \(i\) belonging to cluster \(c\):

\[
\text{score}_i = 0.7\cdot\text{cos}(v_i, \text{centroid}_c)
              + 0.3\cdot\bigl(1 - \tfrac{|len_i - \overline{len}|}{len_i + \overline{len}}\bigr)
\]

The length penalty encourages sentences of typical length and discourages both very short and extremely long outliers.

### 3.4 MILP Sentence Selection
A mixed-integer linear program selects a subset of sentences:

- **Objective:** maximise total semantic score \(\sum_i \text{score}_i\cdot x_i\).
- **Character-budget constraint:** \(\sum_i \text{len}_i\cdot x_i \le \text{max_summary_chars}\) (default 1200).
- **Diversity constraint:** at most one sentence per cluster (\(\sum_{i\in c}x_i\le 1\)).
- **Integrality:** \(x_i\in\{0,1\}\).

Solved with `scipy.optimize.milp`.  
If the solver fails, a greedy per-cluster argmax fallback is used.  
Selected sentences are concatenated in original document order, preserving discourse flow.

```python
def hybrid_extract_core_context(text, num_clusters=4, max_summary_chars=1200):
    ...
```

---

## 4. Abstractive Generation

### 4.1 Prompt Construction
```python
prefix = [tok.bos_id] + list(doc_ids) + tok.encode(sep_token)
```
- `doc_ids` = BPE encoding of the compressed text (truncated to `MAX_DOC_TOK`, default 1024).  
- `sep_token` is a simple newline (or a learned separator) that signals the model to start generating the summary.

### 4.2 Autoregressive Decoding
The Phase-1 model is called repeatedly; at each step only the last `block_size` tokens are kept as context (sliding-window generation).  
Generation stops on `EOS` or when `max_new` tokens have been produced.

### 4.3 Post-processing
The decoder output is split on the separator token; only the part after the separator is retained as the summary.  
Leading/trailing whitespace is stripped.

---

## 5. Model Loading & Backbone Handling

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

- Each backbone uses **its own tokenizer** (different BPE merge tables / vocab sizes).  
- Checkpoint loading strips any `_orig_mod.` prefix left by `torch.compile`.  
- The model is placed in `eval()` mode; gradients are disabled for pure inference runs.

---

## 6. Decoding Configuration

Default generation hyper-parameters (can be overridden per experiment):

```python
GEN = dict(
    max_new   = MAX_SUM_TOK - 1,   # typically 127
    min_new   = 60,                # discourage ultra-short outputs
    no_repeat = 3,                 # no-repeat n-gram window
    rep_pen   = 1.15,              # repetition penalty
    do_sample = True,              # nucleus sampling
    top_p     = 0.9,
    temperature = 0.7,
    top_k     = 50,
)
```

- **Nucleus sampling** (`top_p=0.9`) yields more diverse and natural summaries than pure greedy decoding.  
- **Repetition penalty** and **no-repeat n-gram** suppress the degenerative loops that causal LMs sometimes produce on highly formulaic legal text.  
- A modest `min_new` forces the model to generate a reasonably informative summary rather than a single boilerplate sentence.

---

## 7. Training / Adaptation (Optional)

The core Phase-6 notebook focuses on **zero- / few-shot inference** with the frozen Phase-1 models.  
When additional supervised summarization data are available, the same PEFT machinery developed in Phases 2–5 (LoRA, LoRMA, adapters, …) can be applied:

- Treat the compressed document as the source and the reference summary as the target.  
- Fine-tune with a standard language-modelling loss on the summary tokens only (source tokens masked).  
- The extractive stage remains unchanged; only the abstractive parameters are updated.

Any such fine-tuned checkpoint is loaded through the same `load_trained` pathway simply by pointing `CKPTS[backbone]` at the new weights.

---

## 8. Inference & Submission Packaging

```python
predictions = []
for idx, doc_id in enumerate(TEST_IDS):
    doc_text = test_texts[idx]
    compressed = hybrid_extract_core_context(doc_text, num_clusters=4,
                                             max_summary_chars=1200)
    tok_input  = np.asarray(tok.encode(clean_text(compressed))[:MAX_DOC_TOK])
    summary    = generate(lm, tok, sep, [tok_input], **GEN)[0]
    predictions.append({"id": str(doc_id), "summary": summary})

json.dump(predictions, open("prediction.json", "w"), ensure_ascii=False, indent=2)

with zipfile.ZipFile("submission_lda.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write("prediction.json", arcname="prediction.json")
```

The resulting `submission_lda.zip` (or any other experiment-specific name) is the artifact handed in for official evaluation.

---

## 9. How to Run

### 9.1 Prerequisites
```bash
pip install torch tqdm ijson huggingface_hub numpy scikit-learn scipy
```

### 9.2 Path Setup
```python
DATASET_DIR = "/kaggle/working/phase1_models"   # all .pt + tokenizer files
BACKBONE    = "gqa_conv"                        # or custom_mha / mla_moe / gqa_alibi
```

Place the four Phase-1 checkpoints and the four tokenizer files inside `DATASET_DIR`.

### 9.3 Execution
```python
# 1. Run shared imports, path resolution, tokenizer & backbone-class cells
# 2. (Optional) set RUN_PHASE2_TRAIN / RUN_PHASE3_TRAIN if you also want
#    to execute earlier-phase fine-tuning from this notebook
# 3. Execute the hybrid-extraction + generation cell
# 4. Collect prediction.json and the submission zip
```

For a pure inference run it is sufficient to execute the notebook top-to-bottom after the checkpoints have been copied into `DATASET_DIR`.

### 9.4 Switching Backbones
Change the single variable `BACKBONE`; the correct tokenizer, checkpoint and model class are selected automatically.

---

## 10. Evaluation Notes

Official evaluation (performed by the course server) typically reports:

- **ROUGE-1 / ROUGE-2 / ROUGE-L** (overlap with reference summaries)  
- **BERTScore** or legal-domain semantic similarity  
- Length statistics and copy-rate (fraction of summary tokens that appear in the source)

Because the extractive stage already enforces diversity and a length budget, the generated summaries tend to be concise and cover multiple aspects of the judgment, which correlates well with ROUGE-L and human preference.

---

## 11. File Layout & Artifacts

```
phase6/
├── prediction.json              # list of {"id", "summary"}
├── submission_lda.zip           # official hand-in
└── (optional) ckpts/            # any Phase-6 fine-tuned weights
phase1_models/
├── legal_lm_*.pt
├── legal_bpe_*.json / *.pkl
└── …
```

---

## 12. Reproducibility Notes

- Global seed `SEED = 42` is set for Python, NumPy, PyTorch and the K-Means initialisation.  
- TF-IDF + K-Means + MILP are deterministic given the same input text and seed.  
- Nucleus sampling is seeded through the global PyTorch RNG; identical hardware and library versions therefore produce identical summaries.  
- Each backbone always uses its own tokenizer, guaranteeing that embedding indices stay in-range.  
- The character budget and cluster count are explicit function arguments and can be swept for ablation studies.

---