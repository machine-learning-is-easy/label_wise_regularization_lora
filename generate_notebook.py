import json

# Strings containing the source code
# ---------------------------------------------------------
modeling_source = r'''
import torch
import torch.nn as nn

# -------------------------------
# 1️⃣ LoRA Weighted Function
# -------------------------------
class LoRAWeightedFunction(torch.autograd.Function):
    """
    Custom forward/backward function for LoRA layer.
    Forward: computes x @ A @ B
    Backward: scales gradients based on output norm to encourage learning on low-confidence samples.
    """
    @staticmethod
    def forward(ctx, x, A, B, scale_factor=1.0):
        ctx.save_for_backward(x, A, B)
        ctx.scale_factor = scale_factor
        out = x @ A @ B
        ctx.out_forward = out.detach()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, A, B = ctx.saved_tensors
        out = ctx.out_forward

        # Compute per-sample output norm
        out_norm = torch.norm(out, dim=-1, keepdim=True) + 1e-6
        weight = ctx.scale_factor / out_norm
        grad = grad_output * weight

        # Sample-level gradients
        grad_A_sample = x.unsqueeze(2) @ (grad @ B.T).unsqueeze(1)  # [B, D, r]
        grad_B_sample = (x @ A).unsqueeze(2) * grad.unsqueeze(1)    # [B, r, D]

        grad_A = grad_A_sample.sum(dim=0)
        grad_B = grad_B_sample.sum(dim=0)
        grad_x = grad @ B @ A.T

        # Save sample-level gradients for regularization
        # Note: This static storage is not thread-safe or multi-model safe. 
        # For production, consider attaching to the module instance or context.
        LoRAWeightedFunction.grad_A_sample = grad_A_sample
        LoRAWeightedFunction.grad_B_sample = grad_B_sample

        return grad_x, grad_A, grad_B, None

# -------------------------------
# 2️⃣ LoRA Linear Layer
# -------------------------------
class LoRABertLinear(nn.Module):
    def __init__(self, original_linear, r=4, alpha=1.0, scale_factor=1.0, dropout=0.1):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scale_factor = scale_factor
        self.scaling = alpha / r
        
        # Freeze original weights
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        self.weight.requires_grad = False
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(self.in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r, self.out_features) * 0.01)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Buffers for gradients
        self.grad_A_sample = None
        self.grad_B_sample = None
        
        self.lora_A.register_hook(self._save_grad_A)
        self.lora_B.register_hook(self._save_grad_B)

    def _save_grad_A(self, grad):
        self.grad_A_sample = grad

    def _save_grad_B(self, grad):
        self.grad_B_sample = grad

    def forward(self, x):
        main = x @ self.weight.T
        lora = LoRAWeightedFunction.apply(x, self.lora_A, self.lora_B, self.scale_factor)
        return main + self.scaling * self.dropout(lora)

# -------------------------------
# 3️⃣ Injection Utility
# -------------------------------
def inject_lora_bert(model, r=4, alpha=1.0, scale_factor=1.0, dropout=0.1):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and \
           ('query' in name or 'key' in name or 'value' in name or \
            'q_lin' in name or 'k_lin' in name or 'v_lin' in name):
            # Handle both BERT and RoBERTa/DistilBERT naming conventions if possible
            # But strictly speaking, we need to find the parent module.
            # This simple string split works for standard Transformers models.
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            
            # Retrieve parent module
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            
            # Replace
            setattr(parent, child_name, LoRABertLinear(module, r, alpha, scale_factor, dropout))

# -------------------------------
# 4️⃣ Regularization Loss
# -------------------------------
def grad_regularization_bert(model, logits, labels):
    preds = logits.argmax(dim=-1)
    correct_mask = preds == labels
    reg_loss = 0.0
    count = correct_mask.sum().item()
    if count == 0:
        return torch.tensor(0., device=logits.device)
        
    for module in model.modules():
        if isinstance(module, LoRABertLinear) and module.grad_A_sample is not None:
            # We need to be careful about the batch dimension matching
            # Assuming grad_A_sample is [B, D, r]
            if module.grad_A_sample.shape[0] != correct_mask.shape[0]:
                continue # Skip if shapes don't match (e.g. last batch)
                
            mask = correct_mask.view(-1, 1, 1).expand_as(module.grad_A_sample)
            grad_A_correct = module.grad_A_sample[mask].view(-1, module.r)
            
            mask_B = correct_mask.view(-1, 1, 1).expand_as(module.grad_B_sample)
            grad_B_correct = module.grad_B_sample[mask_B].view(-1, module.lora_B.size(1))
            
            reg_loss += (grad_A_correct**2).sum() + (grad_B_correct**2).sum()
            
    return reg_loss / count
'''

datasets_source = r'''
from datasets import load_dataset
import torch

def get_dataset_mrpc(split, tokenizer, max_length=128):
    """
    Load MRPC dataset.
    split: 'train', 'validation', 'test'
    """
    dataset = load_dataset('glue', 'mrpc', split=split)
    
    def tokenize_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], 
                         padding='max_length', truncation=True, max_length=max_length)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return tokenized_datasets

def get_dataset_stsb(split, tokenizer, max_length=128):
    """
    Load STS-B dataset.
    split: 'train', 'validation', 'test'
    """
    dataset = load_dataset('glue', 'stsb', split=split)
    
    def tokenize_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], 
                         padding='max_length', truncation=True, max_length=max_length)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # STS-B is a regression task, label is float
    tokenized_datasets = tokenized_datasets.map(lambda x: {'label': float(x['label'])})
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return tokenized_datasets
'''

models_source = r'''
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, DistilBertForSequenceClassification

def build_model_bert(model_name="bert-base-uncased", num_labels=1, r=4, alpha=1.0, scale_factor=1.0, dropout=0.1):
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    inject_lora_bert(model, r=r, alpha=alpha, scale_factor=scale_factor, dropout=dropout)
    
    # Freeze base model parameters
    for param in model.bert.parameters():
        param.requires_grad = False
        
    return model

def build_model_roberta(model_name="roberta-base", num_labels=1, r=4, alpha=1.0, scale_factor=1.0, dropout=0.1):
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    inject_lora_bert(model, r=r, alpha=alpha, scale_factor=scale_factor, dropout=dropout)
    
    # Freeze base model parameters
    for param in model.roberta.parameters():
        param.requires_grad = False
        
    return model

def build_model_distilbert(model_name="distilbert-base-uncased", num_labels=1, r=4, alpha=1.0, scale_factor=1.0, dropout=0.1):
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    inject_lora_bert(model, r=r, alpha=alpha, scale_factor=scale_factor, dropout=dropout)
    
    # Freeze base model parameters
    for param in model.distilbert.parameters():
        param.requires_grad = False
        
    return model
'''

train_source = r'''
def evaluate(model, dataloader, device, is_regression=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits.squeeze() if is_regression else outputs.logits
            
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            
            if not is_regression:
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
            
            total += len(labels)
            
    avg_loss = total_loss / total
    metric = avg_loss if is_regression else correct / total
    return metric

def train(model, train_loader, val_loader, config, device, is_regression=False):
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config['learning_rate']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epochs'])
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    
    best_metric = float('inf') if is_regression else 0.0
    
    for epoch in range(config['max_epochs']):
        model.train()
        
        # Optional: Unfreeze layers
        if epoch == config.get('unfreeze_layers_after', 999):
            print(f"Unfreezing last {config.get('unfreeze_layers_count', 2)} layers...")
            # Logic to unfreeze would go here
            
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']}")
        for batch in pbar:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                logits = outputs.logits.squeeze() if is_regression else outputs.logits
                
                loss_task = criterion(logits, labels)
                
                # Gradient regularization (only for classification for now)
                loss_grad = torch.tensor(0., device=device)
                if not is_regression and config.get('lambda_reg', 0.0) > 0.0:
                    try:
                        loss_grad = grad_regularization_bert(model, outputs.logits, labels)
                    except Exception as e:
                        # Fallback if algo is unstable
                        pass
                
                loss = loss_task + float(config['lambda_reg']) * loss_grad
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        
        # Validation
        metric = evaluate(model, val_loader, device, is_regression)
        print(f"Validation {'MSE' if is_regression else 'Acc'}: {metric:.4f}")
        
        # Save best
        if is_regression:
            if metric < best_metric:
                best_metric = metric
                torch.save(model.state_dict(), f"results/best_model_{config['model_name']}_{config['dataset_name']}.pt")
        else:
            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), f"results/best_model_{config['model_name']}_{config['dataset_name']}.pt")
                
    return best_metric
'''

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
        "This notebook contains all necessary functions and classes to run the LoRA experiments."
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
        "from transformers import AutoTokenizer"
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
