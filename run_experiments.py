import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from datetime import datetime

# Import our packages
from _datasets import sts_b, mrpc
from models import bert_lora, roberta_lora, distilbert_lora
from lora_utils.modeling import grad_regularization_bert

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate(model, dataloader, device, is_regression=False):
    """
    Evaluate the model on a given dataset.
    
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        device (str): Device to run the evaluation on ('cuda' or 'cpu').
        is_regression (bool): Whether the task is regression (True) or classification (False).
        
    Returns:
        float: The evaluation metric (MSE for regression, Accuracy for classification).
    """
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
    """
    Train the model.
    
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        config (dict): Configuration dictionary.
        device (str): Device to run the training on.
        is_regression (bool): Whether the task is regression.
        
    Returns:
        float: The best metric achieved on the validation set.
    """
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
            # Logic to unfreeze would go here (simplified for now)
            
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
                if not is_regression:
                    loss_grad = grad_regularization_bert(model, outputs.logits, labels)
                
                loss = loss_task + float(config['lambda_reg']) * loss_grad
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        
        # Validation
        metric = evaluate(model, val_loader, device, is_regression)
        print(f"Validation {'MSE' if is_regression else 'Acc'}: {metric:.4f}")
        
        # Save best model if metric improves
        if is_regression:
            if metric < best_metric:
                best_metric = metric
                torch.save(model.state_dict(), f"results/best_model_{config['model_name']}_{config['dataset_name']}.pt")
        else:
            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), f"results/best_model_{config['model_name']}_{config['dataset_name']}.pt")
                
    return best_metric

def main():
    """
    Main function to run the experiments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs("results", exist_ok=True)
    
    # Define experiments list: (model_name, dataset_name, model_module)
    experiments = [
        ('bert-base-uncased', 'sts_b', bert_lora),
        ('bert-base-uncased', 'mrpc', bert_lora),
        ('roberta-base', 'sts_b', roberta_lora),
        ('roberta-base', 'mrpc', roberta_lora),
        ('distilbert-base-uncased', 'sts_b', distilbert_lora),
        ('distilbert-base-uncased', 'mrpc', distilbert_lora),
    ]
    
    results = []
    
    from transformers import AutoTokenizer
    
    for model_name, dataset_name, model_module in experiments:
        print(f"\n🚀 Running {model_name} on {dataset_name}")
        
        # Update config for current run
        config['model_name'] = model_name
        config['dataset_name'] = dataset_name
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load Data based on dataset name
        if dataset_name == 'sts_b':
            dataset_module = sts_b
            is_regression = True
            num_labels = 1
        else:
            dataset_module = mrpc
            is_regression = False
            num_labels = 2
            
        train_data = dataset_module.get_dataset('train', tokenizer)
        val_data = dataset_module.get_dataset('validation', tokenizer)
        
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'])
        
        # Build Model with LoRA
        model = model_module.build_model(
            model_name=model_name,
            num_labels=num_labels,
            r=config['lora_rank'],
            alpha=config['lora_alpha'],
            scale_factor=config['scale_factor'],
            dropout=config['dropout']
        )
        
        # Train the model
        metric = train(model, train_loader, val_loader, config, device, is_regression)
        
        results.append({
            'model': model_name,
            'dataset': dataset_name,
            'metric': metric,
            'type': 'MSE' if is_regression else 'Accuracy'
        })
        
    # Save experiment results to a JSON file
    with open('results/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n✅ All experiments completed!")

if __name__ == '__main__':
    main()
