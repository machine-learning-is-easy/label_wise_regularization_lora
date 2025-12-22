import torch
import torch.nn as nn

# -------------------------------
# 1️⃣ CA-LoRA Weighted Function
# -------------------------------
class CALoRAWeightedFunction(torch.autograd.Function):
    """
    Custom forward/backward function for CA-LoRA layer.
    
    This function implements the core logic of label-wise regularization (Confidence-Aware LoRA).
    In the backward pass, it scales the gradients based on the inverse of the 
    output norm. This means samples with lower confidence (smaller output norm)
    will have their gradients scaled up, while high-confidence samples will have
    smaller gradients.
    """
    @staticmethod
    def forward(ctx, x, A, B, scale_factor=1.0):
        """
        Forward pass: computes x @ A @ B
        """
        ctx.save_for_backward(x, A, B)
        ctx.scale_factor = scale_factor
        out = x @ A @ B
        ctx.out_forward = out.detach()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: computes gradients with sample-wise scaling.
        """
        x, A, B = ctx.saved_tensors
        out = ctx.out_forward

        # Compute per-sample output norm
        out_norm = torch.norm(out, dim=-1, keepdim=True) + 1e-6
        weight = ctx.scale_factor / out_norm
        grad = grad_output * weight

        # Standard LoRA gradients with the scaled 'grad'
        # Matrix shapes:
        # x: [Batch, Seq, In]
        # A: [In, r]
        # B: [r, Out]
        # grad: [Batch, Seq, Out]
        
        # grad_A = x^T @ (grad @ B^T)
        grad_A = x.transpose(-2, -1) @ (grad @ B.T)
        grad_A = grad_A.sum(dim=0) # Sum over batch
        
        # grad_B = (x @ A)^T @ grad
        grad_B = (x @ A).transpose(-2, -1) @ grad
        grad_B = grad_B.sum(dim=0) # Sum over batch
        
        grad_x = grad @ B @ A.T

        return grad_x, grad_A, grad_B, None

# -------------------------------
# 2️⃣ CA-LoRA Linear Layer
# -------------------------------
class CALoRABertLinear(nn.Module):
    """
    CA-LoRA Linear Layer that replaces a standard nn.Linear layer.
    
    It freezes the original weights and adds trainable LoRA matrices A and B.
    It uses CALoRAWeightedFunction for the forward pass of the LoRA path to 
    enable the gradient scaling logic.
    """
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

    def forward(self, x):
        main = x @ self.weight.T
        lora = CALoRAWeightedFunction.apply(x, self.lora_A, self.lora_B, self.scale_factor)
        return main + self.scaling * self.dropout(lora)

# -------------------------------
# 3️⃣ Injection Utility
# -------------------------------
def inject_calora_bert(model, r=4, alpha=1.0, scale_factor=1.0, dropout=0.1):
    """
    Inject CA-LoRA layers into a BERT-based model.
    
    It targets the query, key, and value projection layers in the self-attention mechanism.
    
    Args:
        model (nn.Module): The model to modify.
        r (int): LoRA rank.
        alpha (float): LoRA alpha scaling.
        scale_factor (float): Factor for the weighted gradient scaling.
        dropout (float): Dropout probability.
    """
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
            setattr(parent, child_name, CALoRABertLinear(module, r, alpha, scale_factor, dropout))
