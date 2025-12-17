from transformers import RobertaForSequenceClassification
from lora_utils.modeling import inject_lora_bert

def build_model(model_name="roberta-base", num_labels=1, r=4, alpha=1.0, scale_factor=1.0, dropout=0.1):
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    inject_lora_bert(model, r=r, alpha=alpha, scale_factor=scale_factor, dropout=dropout)
    
    # Freeze base model parameters
    for param in model.roberta.parameters():
        param.requires_grad = False
        
    return model
