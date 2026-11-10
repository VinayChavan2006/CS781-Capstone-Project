# Phase 1 – Pre-training the Legal Backbones

This phase focuses on the foundation stage of the capstone project: training and evaluating four legal-domain decoder-only language model backbones from scratch or from pre-trained checkpoints.

## Objective

The main goal of Phase 1 is to build strong base language models for legal text generation and understanding. The models are adapted to legal corpora and are later reused in downstream phases for classification and extraction tasks.

## Architectures included

| Backbone | Architecture summary |
|---|---|
| `custom_mha` | Multi-head attention with RoPE, RMSNorm, and SwiGLU |
| `mla_moe` | Multi-head latent attention with MoE routing and specialized feed-forward blocks |
| `gqa_conv` | GQA with dual-path FFN and causal depthwise convolution |
| `gqa_alibi` | GQA with ALiBi positional bias and SwiGLU layers |

## Methodology

The Phase 1 workflow includes:

- legal text preprocessing and cleaning
- BPE tokenizer training for each backbone
- tokenization into binary shards for efficient memory-mapped training
- autoregressive language-model pre-training
- checkpoint saving and resume support
- generation and decoding-strategy comparison

The notebooks are designed to support two execution modes:

1. Load from pre-trained weights
2. Train from scratch if no checkpoint is available

## Implementation details

The notebook pipeline is implemented in PyTorch and includes:

- streaming ingestion of legal documents (using `ijson` when needed)
- vocab building using byte-pair encoding
- shard serialization to disk for efficient training
- custom blocks for attention, gating, FFN, RMSNorm, and positional encodings
- KV-cache-based generation for faster inference
- decoding strategy analysis for greedy, Top-k, nucleus sampling, and beam search

The code also includes path management for different checkpoint locations, which allows training to resume from saved states without reinitializing the whole model.

## Experimental setup

The notebook uses a standard configuration flow:

- set the data directory and checkpoint/output folder
- ensure the tokenizer file exists for the active backbone
- optionally skip training if pre-trained weights are already available
- run generation and evaluation cells after loading a checkpoint

Common setup variables include:

- `DATASET_DIR`
- `OUT_DIR`
- `BACKBONE` / architecture-specific activation cells
- tokenizer path mappings for each backbone

## How to run

1. Open `phase1/phase1.ipynb`.
2. Run the shared path resolution cell.
3. Choose the backbone you want to work with.
4. If pre-trained checkpoints are already available, keep training cells commented and skip to generation/evaluation.
5. If no checkpoint exists, uncomment the appropriate training cell and execute the training block.
6. Run the evaluation and generation cells after training or loading is complete.

## Expected outputs

- tokenizer files (`.json`, `.pkl`)
- checkpoint files (`.pt`)
- generated text from decoding strategies
- evaluation summaries and comparison metrics

## Notes

This phase is the base foundation for all subsequent phases. The later tasks depend on the checkpoint quality, tokenizer compatibility, and architecture-specific design choices made here.
