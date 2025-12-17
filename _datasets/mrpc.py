from datasets import load_dataset
import torch

def get_dataset(split, tokenizer, max_length=128):
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
