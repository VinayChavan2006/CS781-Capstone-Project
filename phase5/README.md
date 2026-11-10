# Phase 5 – Merged Pipeline for Phases 1–4

Phase 5 is a merged experimentation pipeline that brings together the earlier phases into a unified workflow. It connects legal-language-model pretraining, downstream adaptation, and evaluation under a single reproducible setup.

## Objective

The main purpose of this phase is to operationalize the entire pipeline in a more streamlined, reusable form. Instead of treating Model Loading, LNER, classification, and PEFT as isolated blocks, this phase consolidates the project into a single integrated flow.

## Model backbones

The merged workflow supports the following models:

- `custom_mha`
- `mla_moe`
- `gqa_conv`
- `gqa_alibi`

Each backbone has its own tokenizer and checkpoint mapping. This is essential because each tokenizer is trained for a specific model and vocabulary size.

## Methodology

The methodology in this phase emphasizes:

- consistent backbone loading across all tasks
- path-aware configuration and checkpoint validation
- training phases for different stages while preserving tokenizer/model compatibility
- modular learning flow with task-specific training toggles
- artifact generation for later benchmarking and submission preparation

## Implementation details

The notebook includes:

- initialization of model, dataset, and output directories
- per-backbone tokenizer path resolution
- environment setup for Hugging Face or external dataset access
- main configuration variables such as `DATASET_DIR`, `BACKBONE`, and output directories
- integrated phase logic for training and evaluation

The notebook also makes explicit why different tokenizers are needed per backbone: the model and embedding matrix shape are tied to the tokenizer vocabulary used during pretraining.

## Experimental setup

The notebook expects:

- a folder with tokenizer files and model checkpoints
- a selected `BACKBONE`
- output directories for Phase 2 and Phase 3 artifacts
- optional Hugging Face login if using external model or dataset hosting

## How to run

1. Open `phase5/phase5.ipynb`.
2. Confirm or adjust the dataset and tokenizer paths.
3. Select a valid `BACKBONE`.
4. Run the initial configuration cells.
5. Execute the workflow in sequence, enabling `RUN_PHASE2_TRAIN` and `RUN_PHASE3_TRAIN` as needed.
6. Review generated outputs and logs to analyze performance.

## Expected outputs

- checkpoint and tokenizer validation logs
- intermediate training artifacts
- downstream task results from merged end-to-end training
- model selection and experiment summaries

## Notes

Phase 5 is a central integration stage in the project. It demonstrates the end-to-end idea of using a legally specialized foundation model and then carrying it through multiple downstream tasks in a consistent experimental framework.
