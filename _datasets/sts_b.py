from datasets import load_dataset
import torch

def get_dataset(split, tokenizer, max_length=128):
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
