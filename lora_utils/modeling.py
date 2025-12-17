import torch
import torch.nn as nn

# -------------------------------
# 1️⃣ LoRA Weighted Function
# -------------------------------
class LoRAWeightedFunction(torch.autograd.Function):
    """
    Custom forward/backward function for LoRA layer.
    Forward: computes x @ A @ B
    Backward: scales gradients based on output norm to encourage learning on low-confidence samples.
    """
    @staticmethod
    def forward(ctx, x, A, B, scale_factor=1.0):
        ctx.save_for_backward(x, A, B)
        ctx.scale_factor = scale_factor
        out = x @ A @ B
        ctx.out_forward = out.detach()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, A, B = ctx.saved_tensors
        out = ctx.out_forward

        # Compute per-sample output norm
        out_norm = torch.norm(out, dim=-1, keepdim=True) + 1e-6
        weight = ctx.scale_factor / out_norm
        grad = grad_output * weight

        # Sample-level gradients
        grad_A_sample = x.unsqueeze(2) @ (grad @ B.T).unsqueeze(1)  # [B, D, r]
        grad_B_sample = (x @ A).unsqueeze(2) * grad.unsqueeze(1)    # [B, r, D]

        grad_A = grad_A_sample.sum(dim=0)
        grad_B = grad_B_sample.sum(dim=0)
        grad_x = grad @ B @ A.T

        # Save sample-level gradients for regularization
        # Note: This static storage is not thread-safe or multi-model safe. 
        # For production, consider attaching to the module instance or context.
        LoRAWeightedFunction.grad_A_sample = grad_A_sample
        LoRAWeightedFunction.grad_B_sample = grad_B_sample

        return grad_x, grad_A, grad_B, None

# -------------------------------
# 2️⃣ LoRA Linear Layer
# -------------------------------
class LoRABertLinear(nn.Module):
    def __init__(self, original_linear, r=4, alpha=1.0, scale_factor=1.0, dropout=0.1):
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
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(self.in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r, self.out_features) * 0.01)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Buffers for gradients
        self.grad_A_sample = None
        self.grad_B_sample = None
        
        self.lora_A.register_hook(self._save_grad_A)
        self.lora_B.register_hook(self._save_grad_B)

    def _save_grad_A(self, grad):
        self.grad_A_sample = grad

    def _save_grad_B(self, grad):
        self.grad_B_sample = grad

    def forward(self, x):
        main = x @ self.weight.T
        lora = LoRAWeightedFunction.apply(x, self.lora_A, self.lora_B, self.scale_factor)
        return main + self.scaling * self.dropout(lora)

# -------------------------------
# 3️⃣ Injection Utility
# -------------------------------
def inject_lora_bert(model, r=4, alpha=1.0, scale_factor=1.0, dropout=0.1):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and \
           ('query' in name or 'key' in name or 'value' in name or \
            'q_lin' in name or 'k_lin' in name or 'v_lin' in name):
            # Handle both BERT and RoBERTa/DistilBERT naming conventions if possible
            # But strictly speaking, we need to find the parent module.
            # This simple string split works for standard Transformers models.
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            
            # Retrieve parent module
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            
            # Replace
            setattr(parent, child_name, LoRABertLinear(module, r, alpha, scale_factor, dropout))

# -------------------------------
# 4️⃣ Regularization Loss
# -------------------------------
def grad_regularization_bert(model, logits, labels):
    preds = logits.argmax(dim=-1)
    correct_mask = preds == labels
    reg_loss = 0.0
    count = correct_mask.sum().item()
    if count == 0:
        return torch.tensor(0., device=logits.device)
        
    for module in model.modules():
        if isinstance(module, LoRABertLinear) and module.grad_A_sample is not None:
            # We need to be careful about the batch dimension matching
            # Assuming grad_A_sample is [B, D, r]
            if module.grad_A_sample.shape[0] != correct_mask.shape[0]:
                continue # Skip if shapes don't match (e.g. last batch)
                
            mask = correct_mask.view(-1, 1, 1).expand_as(module.grad_A_sample)
            grad_A_correct = module.grad_A_sample[mask].view(-1, module.r)
            
            mask_B = correct_mask.view(-1, 1, 1).expand_as(module.grad_B_sample)
            grad_B_correct = module.grad_B_sample[mask_B].view(-1, module.lora_B.size(1))
            
            reg_loss += (grad_A_correct**2).sum() + (grad_B_correct**2).sum()
            
    return reg_loss / count
