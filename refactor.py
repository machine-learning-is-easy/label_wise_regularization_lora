import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# -------------------------------
# 1️⃣ LoRA Weighted Function
# -------------------------------
class LoRAWeightedFunction(torch.autograd.Function):
    """
    自定义前向/反向函数，用于 LoRA 层。
    前向：计算 x @ A @ B
    反向：对梯度进行样本级调整，如果前向输出接近0，会放大梯度
    """
    @staticmethod
    def forward(ctx, x, A, B, scale_factor=1.0):
        ctx.save_for_backward(x, A, B)
        ctx.scale_factor = scale_factor
        out = x @ A @ B  # LoRA 前向计算
        ctx.out_forward = out.detach()  # 用于梯度缩放
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, A, B = ctx.saved_tensors
        out = ctx.out_forward

        # 计算每个样本的输出范数，避免除零
        out_norm = torch.norm(out, dim=-1, keepdim=True) + 1e-6
        weight = ctx.scale_factor / out_norm  # 越小的输出放大梯度
        grad = grad_output * weight

        # 样本级梯度
        grad_A_sample = x.unsqueeze(2) @ (grad @ B.T).unsqueeze(1)  # [B, D, r]
        grad_B_sample = (x @ A).unsqueeze(2) * grad.unsqueeze(1)    # [B, r, D]

        grad_A = grad_A_sample.sum(dim=0)
        grad_B = grad_B_sample.sum(dim=0)
        grad_x = grad @ B @ A.T

        # 保存样本级梯度供正则使用
        LoRAWeightedFunction.grad_A_sample = grad_A_sample
        LoRAWeightedFunction.grad_B_sample = grad_B_sample

        return grad_x, grad_A, grad_B, None  # None 对应 scale_factor 无需梯度

# -------------------------------
# 2️⃣ LoRA 线性层（替换 BERT Linear）
# -------------------------------
class LoRABertLinear(nn.Module):
    """
    LoRA 层封装：
    - 保留原始权重，不训练
    - 添加 LoRA A/B 低秩参数，训练它们
    """
    def __init__(self, original_linear, r=4, alpha=1.0, scale_factor=1.0):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scale_factor = scale_factor
        self.scaling = alpha / r

        # 原始权重保持不训练
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        self.weight.requires_grad = False

        # LoRA 低秩参数初始化
        self.lora_A = nn.Parameter(torch.randn(self.in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r, self.out_features) * 0.01)

        # 用于存储样本级梯度
        self.grad_A_sample = None
        self.grad_B_sample = None

        # 注册 hook 保存梯度
        self.lora_A.register_hook(self._save_grad_A)
        self.lora_B.register_hook(self._save_grad_B)

    def _save_grad_A(self, grad):
        self.grad_A_sample = grad  # [B, D, r]

    def _save_grad_B(self, grad):
        self.grad_B_sample = grad  # [B, r, D]

    def forward(self, x):
        # 原始权重前向
        main = x @ self.weight.T
        # LoRA 前向
        lora = LoRAWeightedFunction.apply(x, self.lora_A, self.lora_B, self.scale_factor)
        return main + self.scaling * lora

# -------------------------------
# 3️⃣ LoRA 注入 BERT
# -------------------------------
def inject_lora_bert(model, r=4, alpha=1.0):
    """
    遍历 BERT 模型，将 Self-Attention 的 query/key/value Linear 替换为 LoRALinear
    """
    for name, module in model.named_modules():
        # 这里只替换自注意力层的 query/key/value
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
# 4️⃣ 加载预训练 BERT
# -------------------------------
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
inject_lora_bert(model)

# 冻结原始 BERT 参数，只训练 LoRA
for param in model.bert.parameters():
    param.requires_grad = False

# -------------------------------
# 5️⃣ 梯度正则函数
# -------------------------------
def grad_regularization_bert(model, logits, labels):
    """
    对正确分类的样本，计算 LoRA 参数梯度平方和
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
# 6️⃣ 数据准备
# -------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
texts = ["Hello world", "LoRA test", "Transformers rocks", "PyTorch is great"]
labels = torch.tensor([0, 1, 0, 1])

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
lambda_reg = 0.01
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# -------------------------------
# 7️⃣ 训练循环
# -------------------------------
for epoch in range(3):
    optimizer.zero_grad()
    # forward 时 store_grad=True 保存样本级梯度
    outputs = model(**inputs)
    logits = outputs.logits
    # 任务 loss
    loss_task = nn.CrossEntropyLoss()(logits, labels)
    # LoRA 正确分类梯度正则 loss
    loss_grad = grad_regularization_bert(model, logits, labels)
    # 总 loss
    loss = loss_task + lambda_reg * loss_grad
    loss.backward()
    optimizer.step()
    print("One step")
    # print(f"Epoch {epoch+1}: task_loss={loss_task.item():.4f}, grad_loss={loss_grad.item():.4f}")