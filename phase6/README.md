# Phase 6 – Extended Merged Workflow and Reproducibility Setup

Phase 6 continues the merged model workflow with an emphasis on modularity, reproducibility, and more robust experimental execution.

## Objective

This phase refines the integrated training pipeline from earlier phases, ensuring consistent use of the correct backbone, tokenizer, and output folders. It is intended to make the full workflow easier to reproduce and easier to extend.

## Supported architectures

The project continues to support four backbones:

- `custom_mha`
- `mla_moe`
- `gqa_conv`
- `gqa_alibi`

The workflow keeps a dedicated tokenizer path for each model, ensuring that the checkpoint and tokenization configuration remain aligned.

## Methodology

The main methodological goal of Phase 6 is to provide a more reproducible and maintainable implementation of the merged pipeline. It includes:

- deterministic seed setting
- path and output configuration
- output organization for all phases
- consistent backbone-to-tokenizer matching
- modular training and evaluation toggles

## Implementation details

The notebook includes:

- dataset folder preparation
- environment initialization for model execution
- configuration variables for dataset and backbone selection
- checkpoint validation and path resolution
- merged training/evaluation logic that is easier to reuse for different experiments

## Experimental setup

Before running the notebook:

1. Place the relevant tokenizer files in the target dataset directory.
2. Ensure the correct model checkpoint is available for the selected backbone.
3. Set the `BACKBONE` variable to the proper model.
4. Confirm output directories exist or are created automatically.

## How to run

1. Open `phase6/phase6.ipynb`.
2. Run the initialization cells.
3. Configure the dataset and backbone settings.
4. Execute the notebook cells in order.
5. Keep training or evaluation flags set according to whether you are loading existing inference artifacts or running a new training pipeline.

## Expected outputs

- organized experimental output folders
- training logs and model diagnostics
- task results and model comparisons
- reproducible artifacts for later analysis

## Notes

Phase 6 is focused on robustness and repeatability. It is an important stage for making the full project easier to understand, maintain, and extend in later work.
