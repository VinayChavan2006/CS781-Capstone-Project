# Phase 4 – Final View Representation and PEFT Ablations

Phase 4 focuses on final-view representation design and parameter-efficient fine-tuning analysis. It consolidates the legal-language backbones into a more robust downstream classification pipeline with ablation experiments and final submission generation.

## Objective

The goal of this phase is to compare different contextual representations and PEFT strategies for the legal outcome classification task. The notebook explores both flat and hierarchical view pipelines while evaluating the final adaptation quality of each backbone.

## Core design elements

The phase includes several modeling axes:

- flat document views
- head/tail hybrid views
- hierarchical chunk-based pooling
- LoRMA, AdaLoRA, BitFit, and adapter-based tuning
- knowledge distillation scenarios
- threshold optimization for final decisions

## Architectures used

The same four backbones are evaluated here:

- `custom_mha`
- `mla_moe`
- `gqa_conv`
- `gqa_alibi`

Each backbone is combined with different representation schemes and PEFT configurations to measure which setup yields the best legal classification performance.

## Methodology

This phase strengthens the research pipeline by:

- comparing contextual views and pooling strategies
- evaluating adaptation quality under tight parameter budgets
- recording summary tables of F1 scores across settings
- tuning thresholds for final prediction generation
- assembling final deliverables and compressed submission bundles

## Implementation details

The notebook includes:

- hierarchical and flat-view encoders
- final model selection logic
- threshold-grid tuning
- output generation for `prediction.json`-style submissions
- packaging and readme generation for deliverables
- compatibility checks for existing checkpoint paths

The code is written so that the system can run in evaluation mode immediately when checkpoints are available, or in training mode when new experiments must be configured.

## Experimental setup

To reproduce the results:

1. Load the expected benchmark/checkpoint files.
2. Ensure the active architecture is compatible with its tokenizer and trained weights.
3. Prefer the default evaluation pathway when checkpoints are already present.
4. If training is required, configure the experimental dictionary and call the training routine.
5. Run threshold tuning and generate the final submission artifact.

## How to run

1. Open `phase4/phase4.ipynb`.
2. Initialize the workspace and deliverable directories.
3. Run the architecture activation block for the model you want to analyze.
4. Execute the notebook in sequence or use `Run All`.
5. If using the default submission mode, continue directly to the evaluation and submission cells.

## Expected outputs

- final-view experiment results
- F1 tables and ablation summaries
- tuned decision thresholds
- zipped or packaged submission outputs
- project-level documentation for deliverables

## Important: Hugging Face token setup

Wherever the code asks for `HF_TOKEN`, replace it with your own personal Hugging Face token before executing the notebook.

Use your own token for secret retrieval, environment configuration, or login calls; do not commit or expose the real value in the repository.

## Notes

Phase 4 consolidates many experimental threads from the earlier phases into a final ablation and submission-ready stage. It is the closest phase to the final capstone deliverable and evaluation iteration.
