import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# -------------------------------
# 1️⃣ LoRA Weighted Function
# -------------------------------
class LoRAWeightedFunction(torch.autograd.Function):
    """
    Custom forward/backward function for LoRA layer.
    
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
        # We need to being careful with dimensions for matrix multiplication
        grad_A_sample = x.unsqueeze(2) @ (grad @ B.T).unsqueeze(1)  # [B, D, r]
        grad_B_sample = (x @ A).unsqueeze(2) * grad.unsqueeze(1)    # [B, r, D]

        grad_A = grad_A_sample.sum(dim=0)
        grad_B = grad_B_sample.sum(dim=0)
        grad_x = grad @ B @ A.T

        # Save sample-level gradients for regularization use
        LoRAWeightedFunction.grad_A_sample = grad_A_sample
        LoRAWeightedFunction.grad_B_sample = grad_B_sample

        return grad_x, grad_A, grad_B, None  # None for scale_factor as it doesn't need gradient

# -------------------------------
# 2️⃣ LoRA Linear Layer (Replaces BERT Linear)
# -------------------------------
class LoRABertLinear(nn.Module):
    """
    LoRA Layer Wrapper:
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

        # Storage for sample-level gradients
        self.grad_A_sample = None
        self.grad_B_sample = None

        # Register hooks to save gradients
        self.lora_A.register_hook(self._save_grad_A)
        self.lora_B.register_hook(self._save_grad_B)

    def _save_grad_A(self, grad):
        self.grad_A_sample = grad  # [B, D, r]

    def _save_grad_B(self, grad):
        self.grad_B_sample = grad  # [B, r, D]

    def forward(self, x):
        # Original weight forward pass
        main = x @ self.weight.T
        # LoRA forward pass
        lora = LoRAWeightedFunction.apply(x, self.lora_A, self.lora_B, self.scale_factor)
        return main + self.scaling * lora

# -------------------------------
# 3️⃣ Inject LoRA into BERT
# -------------------------------
def inject_lora_bert(model, r=4, alpha=1.0):
    """
    Iterate through the BERT model and replace query/key/value Linear layers 
    in Self-Attention with LoRABertLinear.
    """
    for name, module in model.named_modules():
        # Only replace query/key/value in self-attention
        if isinstance(module, nn.Linear) and 'attention.self.query' in name:
            parent = dict(model.named_modules())[name.rsplit('.',1)[0]]
            setattr(parent, 'query', LoRABertLinear(module, r, alpha))
        if isinstance(module, nn.Linear) and 'attention.self.key' in name:
            parent = dict(model.named_modules())[name.rsplit('.',1)[0]]
            setattr(parent, 'key', LoRABertLinear(module, r, alpha))
        if isinstance(module, nn.Linear) and 'attention.self.value' in name:
            parent = dict(model.named_modules())[name.rsplit('.',1)[0]]
            setattr(parent, 'value', LoRABertLinear(module, r, alpha))

# -------------------------------
# 4️⃣ Load Pre-trained BERT
# -------------------------------
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
inject_lora_bert(model)

# Freeze original BERT parameters, only train LoRA
for param in model.bert.parameters():
    param.requires_grad = False

# -------------------------------
# 5️⃣ Gradient Regularization Function
# -------------------------------
def grad_regularization_bert(model, logits, labels):
    """
    Calculate the sum of squared LoRA parameter gradients for correctly classified samples.
    This encourages the model to have small gradients for samples it is already confident about.
    """
    preds = logits.argmax(dim=-1)
    correct_mask = preds == labels
    reg_loss = 0.0
    count = correct_mask.sum().item()
    if count == 0:
        return torch.tensor(0., device=logits.device)
    for module in model.modules():
        if isinstance(module, LoRABertLinear) and module.grad_A_sample is not None:
            mask = correct_mask.view(-1,1,1).expand_as(module.grad_A_sample)
            grad_A_correct = module.grad_A_sample[mask].view(-1, module.r)
            grad_B_correct = module.grad_B_sample[mask].view(-1, module.lora_B.size(1))
            reg_loss += (grad_A_correct**2).sum() + (grad_B_correct**2).sum()
    return reg_loss / count

# -------------------------------
# 6️⃣ Data Preparation
# -------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
texts = ["Hello world", "LoRA test", "Transformers rocks", "PyTorch is great"]
labels = torch.tensor([0, 1, 0, 1])

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
lambda_reg = 0.01
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# -------------------------------
# 7️⃣ Training Loop
# -------------------------------
for epoch in range(3):
    optimizer.zero_grad()
    # During forward pass, store_grad=True (implicit via hooks) saves sample-level gradients
    outputs = model(**inputs)
    logits = outputs.logits
    # Task loss
    loss_task = nn.CrossEntropyLoss()(logits, labels)
    # LoRA gradient regularization loss for correctly classified samples
    loss_grad = grad_regularization_bert(model, logits, labels)
    # Total loss
    loss = loss_task + lambda_reg * loss_grad
    loss.backward()
    optimizer.step()
    print("One step")
    # print(f"Epoch {epoch+1}: task_loss={loss_task.item():.4f}, grad_loss={loss_grad.item():.4f}")
