# Phase 2 – LNER Adaptation on the Pre-trained Backbones

Phase 2 extends the legal language-model backbones from Phase 1 to the named entity recognition task.

## Objective

The goal of this phase is to adapt each pre-trained legal LM to a token-level labeling problem using task-specific heads and transfer-learning strategies. This includes evaluation of multiple training strategies such as LoRA, BitFit, full fine-tuning, and knowledge distillation-inspired adaptations.

## Model setup

This phase reuses the four legal backbones from Phase 1:

- `custom_mha`
- `mla_moe`
- `gqa_conv`
- `gqa_alibi`

Each notebook section loads the corresponding tokenizer and checkpoint from the Phase 1 folder or a compatible output directory.

## Methodology

The experimentation in this phase uses:

- pre-trained legal backbones as frozen or partially trainable encoders
- downstream LNER heads on top of contextual token representations
- parameter-efficient fine-tuning (PEFT) and full adaptation experiments
- checkpoint resumption support for long training runs
- analysis of training curves and result reporting

The code is structured so that the notebook can either:

- evaluate using pre-trained checkpoints, or
- train newer LNER models from the selected backbone

## Implementation details

The notebook includes:

- path standardization for model and tokenizer artifacts
- model reloading for each backbone using the matching Phase 1 checkpoint
- token-level labeling logic for the legal NER objective
- training loops with checkpoint saving and resume
- result aggregation and output generation

The experimental setup also ensures that the tokenizer and checkpoint stay consistent for each architecture, which is essential for correct inference and evaluation.

## Experimental setup

Before running Phase 2:

1. Ensure the Phase 1 checkpoints are available.
2. Set `DATASET_DIR` and model paths appropriately.
3. Confirm the tokenizer exists for the active backbone.
4. Select the correct backbone and retain the matching Phase 1 checkpoint.

The notebook includes configuration cells that validate the presence of tokenizer and model artifacts before the LNER training/evaluation pipeline begins.

## How to run

1. Open `phase2/phase2.ipynb`.
2. Run the shared initialization cell to set paths.
3. Select the architecture you want to investigate.
4. If checkpoints are present, load them and skip the training setup.
5. If not, uncomment the relevant training cells and run the training block.
6. Execute the evaluation and reporting cells afterward.

## Expected outputs

- trained LNER checkpoints for each architecture
- result tables and analysis summaries
- token-level predictions or merged submission outputs
- logs for comparison across adaptation methods

## Notes

Phase 2 is a bridge between general legal language modeling and task-specific legal information extraction. It demonstrates how backbone quality from Phase 1 transfers to a structured sequence-labeling task.
