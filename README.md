# CS781 Capstone Project

This repository contains the full end-to-end experimental pipeline for a legal-domain language modeling and downstream adaptation project. The notebooks are organized as a staged sequence of phases, each building on the previous one and representing a part of the overall capstone submission.

## Repository structure

- `phase1/` – pre-training of four legal language model backbones
- `phase2/` – LNER adaptation using phase-1 checkpoints
- `phase3/` – rhetorical role and document classification experiments
- `phase4/` – final view / PEFT ablation analysis and submission preparation
- `phase5/` – merged phase 1–4 pipeline for combined training and evaluation
- `phase6/` – extended merged experimental workflow and reproducibility setup
- `phase7/` – final consolidated pipeline and submission-oriented experimentation

## Project overview

The project investigates four custom decoder-only transformer backbones for legal text processing:

- `custom_mha`
- `mla_moe`
- `gqa_conv`
- `gqa_alibi`

These models are trained or adapted for a legal-document understanding pipeline involving:

- language modeling on legal corpora
- named entity recognition (LNER)
- rhetorical role classification
- document classification / legal outcome classification
- PEFT-based adaptation and ablation analysis

## Core model architectures

| Backbone | Main design ideas |
|---|---|
| `custom_mha` | RoPE, full multi-head attention, SwiGLU, RMSNorm |
| `mla_moe` | Multi-head latent attention, mixture-of-experts routing, decoupled RoPE |
| `gqa_conv` | GQA with dual-path FFN, causal depthwise convolution, RoPE |
| `gqa_alibi` | GQA with ALiBi positional bias and SwiGLU-based blocks |

## Methodology

The overall workflow follows a staged research pipeline:

1. Pre-train legal-domain backbones on corpora with custom tokenizer setup and checkpointing.
2. Reuse the relevant backbone for downstream tasks using transfer learning.
3. Adapt the models using parameter-efficient fine-tuning strategies and task-specific heads.
4. Evaluate performance with task metrics, threshold tuning, and predictions.
5. Produce submission-ready outputs and organized deliverables.

## Implementation details

The notebooks use a PyTorch-based implementation and include:

- BPE tokenizer training and loading
- custom model classes for each backbone
- checkpoint resume support
- data preprocessing and sharding
- greedy / top-k / nucleus / beam decoding analysis
- PEFT methods such as LoRA, LoRMA, BitFit, AdaLoRA, and adapters
- document view and hierarchical pooling strategies for downstream tasks

## Experimental setup

Most experiments are designed to run either:

- using pre-trained checkpoint files already present in the workspace, or
- by uncommenting training cells and running the training pipeline from scratch

Common configuration patterns include:

- setting `DATASET_DIR`
- selecting the active `BACKBONE`
- ensuring the correct tokenizer checkpoint pair is loaded
- verifying the model checkpoint path before evaluation or submission

## Running the notebooks

To reproduce the project:

1. Open the desired phase notebook in the corresponding folder.
2. Check the initialization cells for dataset and checkpoint paths.
3. Load the correct backbone tokenizer and model.
4. Run the notebook cells in sequence or use `Run All`.
5. If you have no checkpoint, uncomment training cells in the relevant section and train first.

## Notes

This repository is organized around modular notebook-based experimentation. Each phase is self-contained and documents a specific contribution, while the full project reflects a multi-phase legal language-modeling and downstream transfer learning pipeline.

For more detail, see the README in each phase folder.
