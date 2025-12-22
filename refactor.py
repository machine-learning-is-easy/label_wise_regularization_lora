import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# -------------------------------
# 1️⃣ CA-LoRA Weighted Function
# -------------------------------
class CALoRAWeightedFunction(torch.autograd.Function):
    """
    Custom forward/backward function for CA-LoRA layer.
    
    Forward: Computes x @ A @ B
    Backward: Performs sample-level gradient scaling. If the output norm is close to 0, 
              the gradient is scaled up to encourage learning on these samples.
    """
    @staticmethod
    def forward(ctx, x, A, B, scale_factor=1.0):
        ctx.save_for_backward(x, A, B)
        ctx.scale_factor = scale_factor
        out = x @ A @ B  # LoRA forward computation
        ctx.out_forward = out.detach()  # Detach for gradient scaling logic
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, A, B = ctx.saved_tensors
        out = ctx.out_forward

        # Compute per-sample output norm to avoid division by zero
        out_norm = torch.norm(out, dim=-1, keepdim=True) + 1e-6
        weight = ctx.scale_factor / out_norm  # Scale gradient: smaller output norm -> larger weight
        grad = grad_output * weight

        # Sample-level gradients
        # x: [B, L, D], grad: [B, L, D]
        grad_A = x.transpose(-2, -1) @ (grad @ B.T)
        grad_A = grad_A.sum(dim=0)
        
        grad_B = (x @ A).transpose(-2, -1) @ grad
        grad_B = grad_B.sum(dim=0)
        
        grad_x = grad @ B @ A.T

        return grad_x, grad_A, grad_B, None

# -------------------------------
# 2️⃣ CA-LoRA Linear Layer (Replaces BERT Linear)
# -------------------------------
class CALoRABertLinear(nn.Module):
    """
    CA-LoRA Layer Wrapper:
    - Keeps original weights frozen.
    - Adds LoRA A/B low-rank parameters that are trainable.
    """
    def __init__(self, original_linear, r=4, alpha=1.0, scale_factor=1.0):
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

        # Initialize LoRA low-rank parameters
        self.lora_A = nn.Parameter(torch.randn(self.in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r, self.out_features) * 0.01)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # Original weight forward pass
        main = x @ self.weight.T
        # LoRA forward pass
        lora = CALoRAWeightedFunction.apply(x, self.lora_A, self.lora_B, self.scale_factor)
        return main + self.scaling * self.dropout(lora)

# -------------------------------
# 3️⃣ Inject CA-LoRA into BERT
# -------------------------------
def inject_calora_bert(model, r=4, alpha=1.0):
    """
    Iterate through the BERT model and replace query/key/value Linear layers 
    in Self-Attention with CALoRABertLinear.
    """
    for name, module in model.named_modules():
        # Only replace query/key/value in self-attention
        if isinstance(module, nn.Linear) and 'attention.self.query' in name:
            parent = dict(model.named_modules())[name.rsplit('.',1)[0]]
            setattr(parent, 'query', CALoRABertLinear(module, r, alpha))
        if isinstance(module, nn.Linear) and 'attention.self.key' in name:
            parent = dict(model.named_modules())[name.rsplit('.',1)[0]]
            setattr(parent, 'key', CALoRABertLinear(module, r, alpha))
        if isinstance(module, nn.Linear) and 'attention.self.value' in name:
            parent = dict(model.named_modules())[name.rsplit('.',1)[0]]
            setattr(parent, 'value', CALoRABertLinear(module, r, alpha))

# -------------------------------
# 4️⃣ Load Pre-trained BERT
# -------------------------------
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
inject_calora_bert(model)

# Freeze original BERT parameters, only train LoRA
for param in model.bert.parameters():
    param.requires_grad = False

# -------------------------------
# 5️⃣ Data Preparation
# -------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
texts = ["Hello world", "LoRA test", "Transformers rocks", "PyTorch is great"]
labels = torch.tensor([0, 1, 0, 1])

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# -------------------------------
# 6️⃣ Training Loop
# -------------------------------
for epoch in range(3):
    optimizer.zero_grad()
    outputs = model(**inputs)
    logits = outputs.logits
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    optimizer.step()
    print("One step")
