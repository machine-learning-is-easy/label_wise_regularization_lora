from transformers import DistilBertForSequenceClassification
from calora_utils.modeling import inject_calora_bert

def build_model(model_name="distilbert-base-uncased", num_labels=1, r=4, alpha=1.0, scale_factor=1.0, dropout=0.1):
    """
    Build a DistilBERT model with LoRA layers injected.
    
    Args:
        model_name (str): Name of the pre-trained DistilBERT model.
        num_labels (int): Number of output labels.
        r (int): LoRA rank.
        alpha (float): LoRA alpha.
        scale_factor (float): Scale factor for weighted gradients.
        dropout (float): Dropout probability.
        
    Returns:
        nn.Module: The modified DistilBERT model with LoRA layers.
    """
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    inject_calora_bert(model, r=r, alpha=alpha, scale_factor=scale_factor, dropout=dropout)
    
    # Freeze base model parameters
    for param in model.distilbert.parameters():
        param.requires_grad = False
        
    return model
