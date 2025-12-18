from transformers import RobertaForSequenceClassification
from lora_utils.modeling import inject_lora_bert

def build_model(model_name="roberta-base", num_labels=1, r=4, alpha=1.0, scale_factor=1.0, dropout=0.1):
    """
    Build a RoBERTa model with LoRA layers injected.
    
    Args:
        model_name (str): Name of the pre-trained RoBERTa model.
        num_labels (int): Number of output labels.
        r (int): LoRA rank.
        alpha (float): LoRA alpha.
        scale_factor (float): Scale factor for weighted gradients..
        dropout (float): Dropout probability.
        
    Returns:
        nn.Module: The modified RoBERTa model with LoRA layers.
    """
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    inject_lora_bert(model, r=r, alpha=alpha, scale_factor=scale_factor, dropout=dropout)
    
    # Freeze base model parameters
    for param in model.roberta.parameters():
        param.requires_grad = False
        
    return model
