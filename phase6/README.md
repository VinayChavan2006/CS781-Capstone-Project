# CS781 Phase-6: Legal Document Summarization

**Course:** CS781 Capstone  
**Phase:** Phase-6 — Downstream Generation: Abstractive / Hybrid Summarization  
**Task:** Produce concise, faithful summaries of long Indian legal judgments.  
**Backbones:** All four Phase-1 foundation models: `custom_mha`, `mla_moe`, `gqa_alibi`, and `gqa_conv`.

This repository consolidates the complete Phase-6 pipelines for every Phase-1 backbone. Each experiment reloads the matching Phase-1 checkpoint and tokenizer, constructs a token-budgeted document representation, and fine-tunes the decoder-only language model using a causal language-modelling objective, optionally augmented with knowledge distillation and semantic alignment.

At inference, the model generates summaries autoregressively under length and repetition constraints. Final predictions are written in the official submission format.

## Table of Contents

1. [Task Definition](#1-task-definition)
2. [Overall Pipeline](#2-overall-pipeline)
3. [Document Representation](#3-document-representation)
4. [Training Sequence Construction](#4-training-sequence-construction)
5. [Decoder-Only Language Modelling](#5-decoder-only-language-modelling)
6. [Causal Language Modelling Objective](#6-causal-language-modelling-objective)
7. [Parameter-Efficient Fine-Tuning](#7-parameter-efficient-fine-tuning-peft)
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
18. [Mathematical Summary](#18-mathematical-summary-of-the-pipeline)

## 1. Task Definition

### 1.1 Input / Output

**Input:** A full-length court judgment or legal document:

$$
D = (w_1, w_2, \ldots, w_n)
$$

**Output:** A fluent natural-language summary \(\hat{Y}\) of at most 256 tokens, capturing:

- Factual background
- Legal issues
- Judicial reasoning
- Relevant precedents and citations
- Final decision

### 1.2 Why a Hybrid / Budgeted Approach?

Pure abstractive generation from a long document can exceed the context window of the Phase-1 models.

Phase-6 therefore constructs a token-budgeted representation \(D^*\) satisfying:

$$
|D^*| \leq 750
$$

The resulting representation is then used to train a decoder-only language model to generate the corresponding legal summary.

## 2. Overall Pipeline

```text
Raw judgment D
       |
       v
Hierarchical / semantic sectioning
       |
       v
Token-budgeted representation D*  (|D*| <= 750)
       |
       v
Training sequence:
[BOS] + D* + [SEP] + Y + [EOS]
       |
       v
Decoder-only Transformer (causal self-attention)
       |
       v
Combined training objective:
L_total = L_LM + alpha * L_KD + beta * L_SEM
       |
       v
Autoregressive summary generation
       |
       v
Generated summary Y_hat  (|Y_hat| <= 256)
       |
       v
prediction.json
       |
       v
submission.zip
```

The overall objective combines language modelling, knowledge distillation, and semantic alignment when these optional components are enabled.

## 3. Document Representation

Let the original legal document be represented as:

$$
D = (w_1, w_2, \ldots, w_n)
$$

After tokenisation and hierarchical preprocessing, the document is divided into semantic sections:

$$
D_H = \{S_1, S_2, \ldots, S_m\}
$$

Each section \(S_i\) corresponds to a coherent part of the judgment, such as:

- Factual background
- Arguments presented by the parties
- Legal issues and questions of law
- Judicial reasoning
- Relevant precedents and citations
- Final judgment or decision

A token-budgeted representation is then constructed by concatenating the selected semantic sections and applying truncation:

$$
D^* = \text{Truncate}(S_1 \oplus S_2 \oplus \cdots \oplus S_m)
$$

Subject to the maximum document token budget:

$$
|D^*| \leq 750
$$

Here, \(\oplus\) denotes sequence concatenation.

This representation preserves the most relevant information from the original judgment while ensuring that the input remains within the model's context constraints.

## 4. Training Sequence Construction

For each document-summary pair \((D^*, Y)\), let the reference summary be:

$$
Y = (y_1, y_2, \ldots, y_T)
$$

The decoder-only language model receives a concatenated sequence consisting of the document representation, a separator token, and the target summary:

$$
X = [\text{BOS}] \oplus D^* \oplus [\text{SEP}] \oplus Y \oplus [\text{EOS}]
$$

| Component | Description |
|---|---|
| `[BOS]` | Beginning-of-sequence token |
| \(D^*\) | Token-budgeted legal document |
| `[SEP]` | Separates the document from the target summary |
| \(Y\) | Ground-truth reference summary |
| `[EOS]` | End-of-sequence token |

The model learns to predict the summary tokens conditioned on the preceding document and generated summary tokens.

## 5. Decoder-Only Language Modelling

The Phase-1 checkpoint initializes the model parameters:

$$
\theta_0
$$

For any position \(t\), the decoder-only Transformer computes a hidden representation using the preceding tokens:

$$
h_t = f_\theta(x_{<t})
$$

The next-token probability distribution is obtained through a linear projection followed by softmax:

$$
p_\theta(x_t \mid x_{<t}) = \text{Softmax}(W h_t)
$$

### 5.1 Causal Self-Attention

Causal self-attention ensures that each token can attend only to itself and preceding tokens, preventing information leakage from future positions.

The causal attention mask is defined as:

$$
M_{ij} =
\begin{cases}
0, & \text{if } j \leq i \\
-\infty, & \text{if } j > i
\end{cases}
$$

This guarantees that predictions at position \(i\) depend only on tokens at positions \(j \leq i\).

## 6. Causal Language Modelling Objective

The decoder-only model learns the conditional probability of generating the reference summary given the document representation.

The conditional probability is factorized autoregressively:

$$
P_\theta(Y \mid D^*) =
\prod_{t=1}^{T}
p_\theta(y_t \mid D^*, y_{<t})
$$

The language modelling objective is the negative log-likelihood of the target summary tokens:

$$
L_{\text{LM}} =
-\sum_{t \in S}
\log p_\theta(y_t \mid D^*, y_{<t})
$$

where \(S\) denotes the positions corresponding to the target summary and the end-of-sequence token.

### 6.1 Label Masking

The loss is computed only on summary tokens.

- Document tokens are assigned the ignore index `-100`.
- Separator and document-prefix tokens are excluded from the language modelling loss.
- Summary tokens and `[EOS]` contribute to the cross-entropy objective.

```text
Input:
[BOS] D* [SEP] Y [EOS]

Labels:
-100  -100  -100  Y  [EOS]
```

This ensures that the model learns to generate the summary rather than reconstruct the input judgment.

## 7. Parameter-Efficient Fine-Tuning (PEFT)

Instead of updating all parameters of the pretrained Phase-1 backbone, Phase-6 can use Parameter-Efficient Fine-Tuning (PEFT).

The updated model parameters are represented as:

$$
\theta = \theta_0 + \Delta\theta
$$

where:

- \(\theta_0\) represents the original pretrained parameters.
- \(\Delta\theta\) represents the learned parameter update.

For LoRA / LoRMA-style low-rank adaptation, a weight matrix is reparameterized as:

$$
W' = W + BA
$$

where:

$$
B \in \mathbb{R}^{d \times r},
\qquad
A \in \mathbb{R}^{r \times k}
$$

Here:

- \(d\) is the output dimension.
- \(k\) is the input dimension.
- \(r\) is the low-rank dimension.
- \(r \ll \min(d, k)\).

The effective weight update is:

$$
\Delta W = BA
$$

Only the low-rank adapter matrices \(A\) and \(B\) are trained, while the original pretrained weights remain frozen.

This significantly reduces the number of trainable parameters and memory requirements.

## 8. Knowledge Distillation

Knowledge distillation optionally transfers knowledge from a teacher model to the Phase-6 student model.

Let:

- \(p_T\) denote the teacher probability distribution.
- \(p_S\) denote the student probability distribution.
- \(\tau\) denote the distillation temperature.

The softened probability distributions are:

$$
p_T(\tau) = \text{Softmax}\left(\frac{z_T}{\tau}\right)
$$

$$
p_S(\tau) = \text{Softmax}\left(\frac{z_S}{\tau}\right)
$$

where \(z_T\) and \(z_S\) represent the teacher and student logits, respectively.

The knowledge distillation loss is defined using Kullback-Leibler divergence:

$$
L_{\text{KD}} =
\tau^2 D_{\text{KL}}\left(p_T(\tau) \parallel p_S(\tau)\right)
$$

The KL divergence is given by:

$$
D_{\text{KL}}(P \parallel Q)
=
\sum_i P(i) \log \frac{P(i)}{Q(i)}
$$

The combined language modelling and distillation objective becomes:

$$
L =
(1 - \lambda)L_{\text{LM}}
+
\lambda L_{\text{KD}}
$$

where \(\lambda \in [0, 1]\) controls the contribution of the distillation objective.

## 9. Semantic Alignment Loss

To encourage semantic consistency between the generated summary and the reference summary, an optional semantic alignment loss is introduced.

Let \(E(\cdot)\) denote a sentence encoder.

The generated and reference summary embeddings are:

$$
e_g = E(\hat{Y})
$$

$$
e_r = E(Y)
$$

The semantic alignment loss is defined as one minus cosine similarity:

$$
L_{\text{SEM}}
=
1 -
\frac{e_g^\top e_r}
{\|e_g\|_2 \|e_r\|_2}
$$

This objective encourages the generated summary to preserve the semantic meaning of the reference summary, even when the exact wording differs.

### 9.1 Final Training Objective

The complete Phase-6 training objective combines language modelling, knowledge distillation, and semantic alignment:

$$
L_{\text{total}}
=
L_{\text{LM}}
+
\alpha L_{\text{KD}}
+
\beta L_{\text{SEM}}
$$

where:

- \(\alpha\) controls the contribution of knowledge distillation.
- \(\beta\) controls the contribution of semantic alignment.
- \(L_{\text{LM}}\) provides the primary autoregressive generation objective.

These additional objectives are optional and can be enabled or disabled through their respective hyperparameters.

## 10. Autoregressive Summary Generation

During inference, the trained model generates a summary token by token, conditioned on the token-budgeted document representation.

At each decoding step, the next token can be selected using greedy decoding:

$$
\hat{y}_t =
\arg\max_y
p_\theta(y \mid D^*, \hat{y}_{<t})
$$

Alternatively, nucleus sampling or beam search can be used.

Generation terminates when either of the following conditions is satisfied:

- The model generates the `[EOS]` token.
- The maximum generation length is reached.

The maximum summary length is constrained by:

$$
|\hat{Y}| \leq 256
$$

This ensures that generated summaries remain concise and suitable for the downstream evaluation requirements.

## 11. N-gram Repetition Blocking

To reduce repetitive text generation, an n-gram repetition blocking mechanism is applied during inference.

Let:

$$
G_n(\hat{Y}_{<t})
$$

denote the set of n-grams already generated in the current summary.

A candidate token \(y\) is blocked if appending it would create an n-gram that has already appeared.

Formally:

$$
p_\theta(y) = 0
$$

whenever:

$$
G_n(\hat{Y}_{<t} \oplus y)
$$

contains a duplicate n-gram.

This mechanism discourages repeated phrases and improves the fluency and readability of generated legal summaries.

## 12. Model Loading & Backbone Handling

Each Phase-6 experiment reloads the corresponding Phase-1 checkpoint and tokenizer.

```python
def load_trained(backbone_name):

    if backbone_name == "custom_mha":
        m = load_v2(
            CKPTS[backbone_name],
            device=DEVICE
        )

    elif backbone_name == "mla_moe":
        m = load_v3(
            CKPTS[backbone_name],
            device=DEVICE
        )

    elif backbone_name == "gqa_conv":
        m = load_gqa(
            CKPTS[backbone_name],
            device=DEVICE
        )

    elif backbone_name == "gqa_alibi":
        m = load_alibi(
            CKPTS[backbone_name],
            device=DEVICE
        )

    else:
        raise ValueError(
            f"Unknown backbone: {backbone_name}"
        )

    t = TOKENIZERS[backbone_name]

    return m, t, meta
```

### 12.1 Backbone-Specific Handling

The following considerations are maintained:

- Each backbone uses its own tokenizer because the BPE merge tables and vocabulary sizes may differ.
- Checkpoint loading removes any `_orig_mod.` prefix introduced by `torch.compile`.
- The corresponding architecture-specific loading function is selected automatically.
- Models are moved to the configured computation device.
- The model is placed in `eval()` mode during inference to disable training-specific behaviour.

### 12.2 Supported Backbones

| Backbone | Loading Function |
|---|---|
| `custom_mha` | `load_v2()` |
| `mla_moe` | `load_v3()` |
| `gqa_conv` | `load_gqa()` |
| `gqa_alibi` | `load_alibi()` |

## 13. Inference & Submission Packaging

The inference pipeline processes every test document, constructs its token-budgeted representation, generates a summary, and stores the prediction.

### 13.1 Prediction Generation

```python
predictions = []

for idx, doc_id in enumerate(TEST_IDS):

    doc_text = test_texts[idx]

    # Construct token-budgeted representation
    D_star = build_budgeted_representation(
        doc_text
    )

    # Tokenize and enforce document token budget
    tok_input = tokenizer.encode(
        D_star
    )[:MAX_DOC_TOK]

    # Generate summary
    summary = generate(
        lm,
        tok,
        sep,
        [tok_input],
        **GEN
    )

    predictions.append({
        "id": str(doc_id),
        "summary": summary
    })
```

### 13.2 Save Predictions

The generated summaries are stored in `prediction.json`:

```python
import json

with open(
    "prediction.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        predictions,
        f,
        ensure_ascii=False,
        indent=2
    )
```

The output follows the required structure:

```json
[
    {
        "id": "document_id_1",
        "summary": "Generated legal judgment summary."
    },
    {
        "id": "document_id_2",
        "summary": "Another generated legal judgment summary."
    }
]
```

### 13.3 Create Submission ZIP

The final submission archive is created as follows:

```python
import zipfile

with zipfile.ZipFile(
    "submission.zip",
    "w",
    zipfile.ZIP_DEFLATED
) as zf:

    zf.write(
        "prediction.json",
        arcname="prediction.json"
    )
```

The final archive contains:

```text
submission.zip
└── prediction.json
```

## 14. How to Run

### 14.1 Prerequisites

Install the required Python dependencies:

```bash
pip install torch tqdm ijson huggingface_hub numpy scikit-learn scipy
```

### 14.2 Path Setup

Configure the dataset directory and select the desired backbone:

```python
DATASET_DIR = "/kaggle/working/phase1_models"

BACKBONE = "gqa_conv"
```

Supported backbone options:

```python
BACKBONE = "custom_mha"
BACKBONE = "mla_moe"
BACKBONE = "gqa_alibi"
BACKBONE = "gqa_conv"
```

Changing the `BACKBONE` variable automatically selects the corresponding Phase-1 checkpoint, tokenizer, and model architecture.

### 14.3 Execution Steps

The complete execution procedure is:

1. Run the shared import and configuration cells.
2. Resolve checkpoint and tokenizer paths.
3. Load the selected backbone and its corresponding tokenizer.
4. Construct the hierarchical and token-budgeted document representations.
5. Optionally fine-tune the model using the combined training objective.
6. Load the trained model for inference.
7. Generate summaries for all test documents.
8. Save the predictions to `prediction.json`.
9. Package the JSON file into `submission.zip`.

## 15. Evaluation Notes

The generated summaries are evaluated using standard summarization metrics and semantic evaluation techniques.

| Metric | Description |
|---|---|
| ROUGE-1 | Measures unigram overlap between generated and reference summaries |
| ROUGE-2 | Measures bigram overlap between generated and reference summaries |
| ROUGE-L | Measures the longest common subsequence between generated and reference summaries |
| BERTScore | Evaluates semantic similarity using contextual embeddings |
| Legal-domain semantic similarity | Measures semantic consistency in the legal domain |
| Length statistics | Measures the generated summary length |
| Copy rate | Measures the extent of direct copying from the source document |

### 15.1 Expected Benefits

The Phase-6 design aims to achieve:

- Concise summaries through the 256-token generation limit.
- Better context utilization through semantic document sectioning.
- Reduced input complexity through the 750-token document budget.
- More fluent generation through autoregressive decoding.
- Reduced repetition through n-gram blocking.
- Improved semantic consistency through optional knowledge distillation and semantic alignment.

The actual performance gains must be established empirically through evaluation and ablation studies.

## 16. File Layout & Artifacts

The expected repository structure is:

```text
phase6/
│
├── prediction.json
│   └── Generated summaries in the official format
│
├── submission.zip
│   └── Final submission archive
│
└── ckpts/
    └── Optional Phase-6 fine-tuned checkpoints

phase1_models/
│
├── legal_lm_*.pt
│   └── Phase-1 model checkpoints
│
├── legal_bpe_*.json
│   └── Tokenizer configurations
│
├── legal_bpe_*.pkl
│   └── Serialized tokenizer artifacts
│
└── ...
```

### 16.1 Important Artifacts

| Artifact | Purpose |
|---|---|
| Phase-1 `.pt` checkpoints | Initialize the corresponding Phase-6 backbone |
| Tokenizer files | Tokenize documents and summaries |
| Phase-6 checkpoints | Store fine-tuned model parameters, if applicable |
| `prediction.json` | Store generated test predictions |
| `submission.zip` | Final packaged submission |

## 17. Reproducibility Notes

### 17.1 Random Seed

A global random seed is configured:

```python
SEED = 42
```

The seed is applied to:

- Python random number generation
- NumPy random number generation
- PyTorch random number generation

### 17.2 Deterministic Processing

- Document truncation follows a fixed 750-token budget.
- Summary generation is constrained to a maximum of 256 tokens.
- The same backbone-specific tokenizer is used for training and inference.
- Causal masking prevents future-token information leakage.
- Label masking (`-100`) ensures that the language modelling loss is computed only on summary tokens and `[EOS]`.

### 17.3 Backbone Consistency

Every experiment uses its corresponding Phase-1 checkpoint and tokenizer.

This prevents vocabulary mismatches and ensures that token IDs remain compatible with the respective embedding layers.

### 17.4 Hyperparameter Configuration

All important hyperparameters are explicitly configurable, including:

- Knowledge distillation weight \(\lambda\)
- Knowledge distillation temperature \(\tau\)
- Semantic alignment weight \(\alpha\) or \(\beta\), depending on the objective convention
- LoRA / LoRMA rank \(r\)
- Maximum document token budget
- Maximum summary generation length
- N-gram repetition blocking size
- Learning rate and training batch size

These parameters can be varied systematically during ablation studies.

## 18. Mathematical Summary of the Pipeline

The complete Phase-6 pipeline can be expressed mathematically as:

$$
D
\longrightarrow
D_H
\longrightarrow
D^*
$$

$$
D^*
\longrightarrow
[\text{BOS}] \oplus D^* \oplus [\text{SEP}] \oplus Y \oplus [\text{EOS}]
\longrightarrow
\text{Decoder-only Transformer}
\longrightarrow
L_{\text{total}}
\longrightarrow
\hat{Y}
$$

Subject to the two hard constraints:

$$
|D^*| \leq 750
$$

$$
|\hat{Y}| \leq 256
$$

The overall objective is:

$$
L_{\text{total}}
=
L_{\text{LM}}
+
\alpha L_{\text{KD}}
+
\beta L_{\text{SEM}}
$$

Thus, Phase-6 transforms long Indian legal judgments into concise, semantically faithful natural-language summaries using a token-budgeted document representation and autoregressive decoder-only generation, initialized from the four Phase-1 foundation model backbones.