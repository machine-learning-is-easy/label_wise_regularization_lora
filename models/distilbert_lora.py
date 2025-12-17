from transformers import DistilBertForSequenceClassification
from lora_utils.modeling import inject_lora_bert

def build_model(model_name="distilbert-base-uncased", num_labels=1, r=4, alpha=1.0, scale_factor=1.0, dropout=0.1):
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    inject_lora_bert(model, r=r, alpha=alpha, scale_factor=scale_factor, dropout=dropout)
    
    # Freeze base model parameters
    for param in model.distilbert.parameters():
        param.requires_grad = False
        
    return model
