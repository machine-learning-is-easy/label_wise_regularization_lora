import json
import os

"""
This script generates a self-contained Jupyter Notebook (`experiments_full.ipynb`) for running LoRA experiments.
It reads the source code from the local Python files and embeds them directly into the notebook cells.
This ensures the notebook is always up-to-date with the latest code and comments.
"""

def read_file(path):
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return ""
    with open(path, 'r', encoding='utf-8') as f:
        # Prepend a comment indicating the source
        return f"# Source: {path}\n" + f.read()

# ---------------------------------------------------------
# Read Source Code from Files
# ---------------------------------------------------------

# Modeling Utils
modeling_source = read_file('calora_utils/modeling.py')

# Datasets
datasets_source = read_file('_datasets/mrpc.py') + "\n\n" + \
                  read_file('_datasets/sts_b.py')
# We need to adjust dataset functions because in the files they are named `get_dataset`
# In the notebook we need them to be distinct e.g. `get_dataset_mrpc` and `get_dataset_stsb`
# or we can keep them as is if we put them in different cells, but the notebook design 
# in previous steps had them in one block. 
# Actually, the previous hardcoded strings had `get_dataset_mrpc` and `get_dataset_stsb`.
# The files `_datasets/mrpc.py` and `sts_b.py` both have `def get_dataset(...)`.
# If I simply concatenate them, I'll have two `get_dataset` functions and the second will overwrite the first.
# So I need to rename them or wrap them.

# Approach: Read and Rename
def read_and_rename_dataset(path, suffix):
    content = read_file(path)
    return content.replace("def get_dataset(", f"def get_dataset_{suffix}(")

datasets_source = read_and_rename_dataset('_datasets/mrpc.py', 'mrpc') + "\n\n" + \
                  read_and_rename_dataset('_datasets/sts_b.py', 'stsb')


# Models
# Similarly, models have `build_model`. I need to rename them.
def read_and_rename_model(path, suffix):
    content = read_file(path)
    return content.replace("def build_model(", f"def build_model_{suffix}(")

models_source = read_and_rename_model('models/bert_lora.py', 'bert') + "\n\n" + \
                read_and_rename_model('models/roberta_lora.py', 'roberta') + "\n\n" + \
                read_and_rename_model('models/distilbert_lora.py', 'distilbert')

# Train Loop
# The train loop is in `run_experiments.py`.
# But `run_experiments.py` has a main function and imports.
# I want just `evaluate` and `train` functions.
# I can extract them.
def extract_train_functions(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Simple extraction based on known structure or just copying the whole file but stripping imports/main
    # For now, let's just grab the functions we know.

    
    content = ""
    recording = False
    
    # Logic: Start recording at `def evaluate` and stop before `def main`?
    # Or just read the whole file and manually filter lines?
    # Since I just scrutinized `run_experiments.py`, I know the structure.
    # `evaluate` starts at line ~20. `train` starts at ~48. `main` starts at ~109.
    
    captured_lines = []
    capture = False
    for line in lines:
        if line.startswith("def evaluate("):
            capture = True
        if line.startswith("def main("):
            capture = False
        
        if capture:
            captured_lines.append(line)
            
    return "".join(captured_lines)

# However, `run_experiments.py` uses `config['lambda_reg']`.
# In the notebook, `train` expects `config`.
# It matches. The imports in `run_experiments.py` (like torch, nn, tqdm) need to be present.
# I'll rely on the imports cell in the notebook.

train_source = extract_train_functions('run_experiments.py')


# ---------------------------------------------------------
# Build Notebook
# ---------------------------------------------------------

cells = []

# Cell 1: Header
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# All-in-One Experiment Runner\n",
        "This notebook contains all necessary functions and classes to run the LoRA experiments.\n",
        "The code below is automatically loaded from the project source files."
    ]
})

# Cell 2: Imports
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import yaml\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "from torch.utils.data import DataLoader\n",
        "from tqdm.notebook import tqdm\n",
        "import json\n",
        "import os\n",
        "from datetime import datetime\n",
        "from transformers import AutoTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, DistilBertForSequenceClassification\n",
        "from datasets import load_dataset"
    ]
})

# Cell 3: Config
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Configuration\n",
        "config = {\n",
        "    'seed': 42,\n",
        "    'batch_size': 32,\n",
        "    'max_epochs': 5,\n",
        "    'learning_rate': 2e-4,\n",
        "    'lambda_reg': 0.01,\n",
        "    'scale_factor': 1.0,\n",
        "    'lora_rank': 4,\n",
        "    'lora_alpha': 1.0,\n",
        "    'dropout': 0.1,\n",
        "    'warmup_steps': 100,\n",
        "    'unfreeze_layers_after': 2\n",
        "}\n",
        "\n",
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "print(f\"Using device: {device}\")\n",
        "os.makedirs(\"results\", exist_ok=True)"
    ]
})

# Cell 4: Modeling Utils
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": modeling_source.strip().splitlines(keepends=True)
})

# Cell 5: Datasets
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": datasets_source.strip().splitlines(keepends=True)
})

# Cell 6: Models
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": models_source.strip().splitlines(keepends=True)
})

# Cell 7: Train/Eval
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": train_source.strip().splitlines(keepends=True)
})

# Cell 8: Main Loop
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Define experiments mapping\n",
        "experiments_map = [\n",
        "    ('bert-base-uncased', 'sts_b', build_model_bert),\n",
        "    ('bert-base-uncased', 'mrpc', build_model_bert),\n",
        "    ('roberta-base', 'sts_b', build_model_roberta),\n",
        "    ('roberta-base', 'mrpc', build_model_roberta),\n",
        "    ('distilbert-base-uncased', 'sts_b', build_model_distilbert),\n",
        "    ('distilbert-base-uncased', 'mrpc', build_model_distilbert),\n",
        "]\n",
        "\n",
        "results = []\n",
        "\n",
        "print(\"Starting Experiments...\")\n",
        "\n",
        "for model_name, dataset_name, build_fn in experiments_map:\n",
        "    print(f\"\\n🚀 Running {model_name} on {dataset_name}\")\n",
        "    \n",
        "    # Update config for current run (optional, logging purposes)\n",
        "    config['model_name'] = model_name\n",
        "    config['dataset_name'] = dataset_name\n",
        "    \n",
        "    tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
        "    \n",
        "    # Load Data\n",
        "    if dataset_name == 'sts_b':\n",
        "        is_regression = True\n",
        "        num_labels = 1\n",
        "        train_data = get_dataset_stsb('train', tokenizer)\n",
        "        val_data = get_dataset_stsb('validation', tokenizer)\n",
        "    else:\n",
        "        is_regression = False\n",
        "        num_labels = 2\n",
        "        train_data = get_dataset_mrpc('train', tokenizer)\n",
        "        val_data = get_dataset_mrpc('validation', tokenizer)\n",
        "        \n",
        "    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)\n",
        "    val_loader = DataLoader(val_data, batch_size=config['batch_size'])\n",
        "    \n",
        "    # Build Model\n",
        "    model = build_fn(\n",
        "        model_name=model_name,\n",
        "        num_labels=num_labels,\n",
        "        r=config['lora_rank'],\n",
        "        alpha=config['lora_alpha'],\n",
        "        scale_factor=config['scale_factor'],\n",
        "        dropout=config['dropout']\n",
        "    )\n",
        "    \n",
        "    # Train\n",
        "    metric = train(model, train_loader, val_loader, config, device, is_regression)\n",
        "    \n",
        "    results.append({\n",
        "        'model': model_name,\n",
        "        'dataset': dataset_name,\n",
        "        'metric': metric,\n",
        "        'type': 'MSE' if is_regression else 'Accuracy'\n",
        "    })\n",
        "    \n",
        "    # Clean up to save memory\n",
        "    del model\n",
        "    torch.cuda.empty_cache()\n",
        "    \n",
        "# Save results\n",
        "with open('results/experiment_results_notebook.json', 'w') as f:\n",
        "    json.dump(results, f, indent=4)\n",
        "    \n",
        "print(\"\\n✅ All experiments completed!\")"
    ]
})


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('experiments_full.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)
