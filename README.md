# LoRA‑Weighted BERT

A lightweight library that injects LoRA (Low‑Rank Adaptation) layers into BERT, RoBERTa, and DistilBERT for efficient fine‑tuning.

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from models.bert_lora import build_model
from _datasets.sts_b import get_dataset

model = build_model()
train_dataset, val_dataset = get_dataset('train'), get_dataset('validation')
# use run_experiments.py for full training
```

## Citation
If you use this code, please cite the accompanying paper (see `CITATION.cff`).
