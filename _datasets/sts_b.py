from datasets import load_dataset
import torch

def get_dataset(split, tokenizer, max_length=128):
    """
    Load and preprocess the STS-B dataset.
    
    Args:
        split (str): One of 'train', 'validation', 'test'.
        tokenizer (PreTrainedTokenizer): Tokenizer to process the text.
        max_length (int): Maximum sequence length.
        
    Returns:
        Dataset: The tokenized dataset with 'input_ids', 'attention_mask', and 'label'.
                 Labels are converted to floats for regression.
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
