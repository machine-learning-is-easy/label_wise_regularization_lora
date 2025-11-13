import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import pandas as pd
from weighted_lora_module import inject_lora_bert, grad_regularization_bert

# ============================================================
# ⚙️ Utility Functions
# ============================================================
def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total

def prepare_data(dataset_name, tokenizer, text_key="text", max_length=128, n_train=2000, n_val=500):
    dataset = load_dataset(dataset_name)
    if "train" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.2)
    if "label" not in dataset["train"].column_names:
        dataset = dataset.rename_column("labels", "label")
    if text_key not in dataset["train"].column_names:
        text_key = "content" if "content" in dataset["train"].column_names else dataset["train"].column_names[0]

    def tokenize_fn(batch):
        return tokenizer(batch[text_key], truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    train_data = tokenized["train"].select(range(min(n_train, len(tokenized["train"]))))
    val_data = tokenized["test"].select(range(min(n_val, len(tokenized["test"]))))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)
    return train_loader, val_loader, len(set(dataset["train"]["label"]))

def finetune_lora(model, train_loader, val_loader, device="cuda", lr=1e-4, epochs=2, lambda_reg=0.01, weighted=False):
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"{'Weighted' if weighted else 'Normal'} Epoch {epoch+1}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            loss_task = criterion(logits, labels)

            if weighted:
                loss_grad = grad_regularization_bert(model, logits, labels)
                loss = loss_task + lambda_reg * loss_grad
            else:
                loss = loss_task

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return evaluate(model, val_loader, device)

# ============================================================
# 🧩 Experiment Setup
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

datasets = [
    ("sst2", "text"),
    ("imdb", "text"),
    ("amazon_polarity", "content"),
    ("ag_news", "text"),
    ("dbpedia_14", "content"),
]

models = [
    "bert-base-uncased",
    "roberta-base",
    "distilbert-base-uncased",
]

results = []

# ============================================================
# 🚀 Main Loop
# ============================================================
for model_name in models:
    print(f"\n🚀 Testing model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for dataset_name, text_key in datasets:
        print(f"\n📊 Dataset: {dataset_name}")
        train_loader, val_loader, num_labels = prepare_data(dataset_name, tokenizer, text_key=text_key)

        # ---------------- Normal LoRA ----------------
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["query", "value"],
            lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_CLS
        )
        normal_lora = get_peft_model(base_model, lora_config)
        acc_normal = finetune_lora(normal_lora, train_loader, val_loader, device=device, weighted=False)

        # ---------------- Weighted LoRA ----------------
        custom_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        inject_lora_bert(custom_model)
        for p in custom_model.base_model.parameters():
            p.requires_grad = False

        acc_weighted = finetune_lora(custom_model, train_loader, val_loader, device=device, weighted=True)

        results.append({
            "Model": model_name,
            "Dataset": dataset_name,
            "Normal LoRA Acc": round(acc_normal * 100, 2),
            "Weighted LoRA Acc": round(acc_weighted * 100, 2)
        })

# ============================================================
# 📈 Results Summary
# ============================================================
df = pd.DataFrame(results)
print("\n===================== Final Accuracy Table =====================")
print(df)

# Optional: Save results
df.to_csv("lora_comparison_results.csv", index=False)