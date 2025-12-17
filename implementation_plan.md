# Optimizing LoRA‑BERT Training & Experiment Suite

## Goal Description
Create a reproducible research package for the LoRA‑weighted BERT project. This includes:
1. **Paper‑ready artifacts** (README, LICENSE, requirements, config, citation). 
2. **Two dataset loaders** (STS‑B and MRPC from the GLUE benchmark). 
3. **Three model wrappers** (BERT‑base, RoBERTa‑base, DistilBERT‑base) that each use the LoRA‑augmented linear layer.
4. **Experiment runner** (`run_experiments.py`) to train/evaluate each model on each dataset with configurable hyper‑parameters.
5. **Verification steps** to ensure scripts run end‑to‑end.

## User Review Required
- Confirm the choice of datasets (STS‑B & MRPC) and models (BERT, RoBERTa, DistilBERT). 
- Approve the inclusion of a `config.yaml` file for hyper‑parameter management. 
- Accept the plan to generate a `requirements.txt` that pins `torch`, `transformers`, and `datasets`.
- Confirm that a permissive `MIT` license is acceptable (can be changed later).

## Proposed Changes
### 1. Paper Artifacts (new files)
- `README.md` – project overview, installation, usage, and citation.
- `LICENSE` – MIT license text.
- `requirements.txt` – pinned dependencies.
- `config.yaml` – default hyper‑parameters (learning‑rate, epochs, batch‑size, lambda_reg, etc.).
- `CITATION.cff` – citation metadata for Zenodo/DOI.

### 2. Dataset Loading Modules (`datasets/`)
- `sts_b.py` – loads the STS‑B dataset via `datasets.load_dataset('glue', 'stsb')` and returns tokenized inputs/labels.
- `mrpc.py` – loads the MRPC dataset via `datasets.load_dataset('glue', 'mrpc')`.
- Both modules expose a `get_dataset(split, tokenizer, max_length=128)` function.

### 3. Model Wrappers (`models/`)
- `bert_lora.py` – imports `BertForSequenceClassification` and injects LoRA using the existing `inject_lora_bert` function.
- `roberta_lora.py` – similar wrapper for `RobertaForSequenceClassification`.
- `distilbert_lora.py` – wrapper for `DistilBertForSequenceClassification`.
- Each wrapper provides a `build_model()` function returning the LoRA‑augmented model.

### 4. Experiment Runner (`run_experiments.py`)
- Parses command‑line arguments or reads `config.yaml`.
- Loops over **models × datasets**, creates model, loads data, trains with the LoRA training loop (including mixed‑precision, scheduler, optional unfreeze). 
- Logs results to `results/` as JSON (accuracy, loss, training time).
- Saves the best checkpoint per experiment.

### 5. Minor Code Adjustments
- Refactor the existing LoRA utilities into a package `lora_utils/` so they can be imported by the new model wrappers.
- Add a small dropout after LoRA output (configurable).
- Replace heavy per‑sample gradient storage with on‑the‑fly regularization (as described earlier).
- Parameterize `scale_factor`, `lambda_reg`, `r`, and dropout via `config.yaml`.

## Verification Plan
1. **Static checks** – run `flake8`/`black` on all new files.
2. **Dependency install** – `pip install -r requirements.txt` should succeed.
3. **Dataset sanity** – load each dataset and print a few examples.
4. **Model sanity** – instantiate each model wrapper and perform a single forward pass.
5. **Full run** – execute `python run_experiments.py --config config.yaml` on a subset (e.g., 1 epoch) to ensure no runtime errors.
6. **Result validation** – compare validation accuracy against a baseline (expect within ±2 % of original BERT baseline).

---
*All files will be created under the repository root (`e:\label_wise_regularization_lora`).*
