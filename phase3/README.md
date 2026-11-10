# Phase 3 – Rhetorical Role and Document Classification

Phase 3 moves from generation and extraction into downstream classification. The notebook focuses on document-level and rhetorical-role classification using the legal backbones created in Phase 1.

## Objective

This phase adapts the legal language models to classification tasks that require understanding legal argument structure, rhetorical role signals, and document-level outcomes. The project compares multiple fine-tuning and adaptation strategies across the four backbones.

## Architectures and tasks

The four backbone architectures are reused here:

- `custom_mha`
- `mla_moe`
- `gqa_conv`
- `gqa_alibi`

The downstream tasks include:

- rhetorical role classification
- document classification
- legal outcome / judgment classification
- PEFT-based adaptation and task-specific classification heads

## Methodology

This phase uses a broad adaptation matrix covering:

- full fine-tuning
- LoRA / LoRMA / rsLoRA variants
- BitFit and lightweight adaptation
- knowledge distillation
- focal loss and cross-entropy objectives
- context-document hybrid representations using BiGRU-style setups
- threshold tuning and result analysis

The repository also includes ensemble based behavior for selected submissions and multiple experimental configurations.

## Implementation details

The implementation includes:

- backbone reloading with Phase 1 checkpoints
- sequence and document pooling strategies
- classification heads for rhetorical role or legal outcome prediction
- multiple objective functions and training configurations
- validation and threshold optimization
- result aggregation in DataFrame-friendly form
- submission-ready output generation

The notebook is designed to support both quick evaluation from stored checkpoints and full training pipelines from scratch.

## Experimental setup

The notebook expects:

- valid Phase 1 checkpoint files
- tokenizers for each backbone
- dataset paths for classification tasks
- correct `BACKBONE` assignment and activation cell sequence

The code includes checks to ensure the model and tokenizer pair are compatible and to minimize common compatibility mistakes.

## How to run

1. Open `phase3/phase3.ipynb`.
2. Confirm the copying/setup cells are active if using pre-trained models.
3. If using fresh training, disable or skip those copy steps and train the required backbone first.
4. Select the architecture block and run the notebook section.
5. Execute adaptation and training cells to produce classifier checkpoints.
6. Run evaluation and submission generation cells after the training loop completes.

## Expected outputs

- fine-tuned classification models
- validation metrics and confusion-based analysis
- threshold-selected predictions
- predictions and submission files for downstream evaluation

## Notes

Phase 3 is where the project shifts from generic legal language modeling to concrete classification decisions. It is a critical stage for demonstrating practical utility on legal datasets and downstream legal text understanding.
