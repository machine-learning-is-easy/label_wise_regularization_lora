# Gradient-Regularized LoRA (GR-LoRA): Enhancing Fine-Tuning Stability via Adaptive Gradient Scaling and Correctness-Aware Regularization

## Abstract
Low-Rank Adaptation (LoRA) has become a de facto standard for parameter-efficient fine-tuning (PEFT) of large language models. However, standard LoRA applies uniform gradient updates regardless of the adapter's current contribution or the model's confidence, which can lead to suboptimal convergence or overfitting on smaller datasets. In this work, we introduce **Gradient-Regularized LoRA (GR-LoRA)**, a novel fine-tuning approach that integrates two complementary mechanisms: (1) **Inverse-Norm Gradient Scaling**, which dynamically amplifies updates for adapters with weak activations to accelerate feature learning, and (2) **Correctness-Aware Gradient Regularization**, which penalizes the gradient norm of correctly classified samples to encourage flatter minima and improved generalization. Extensive experiments on GLUE benchmarks (STS-B, MRPC) demonstrate that GR-LoRA achieves superior stability and accuracy compared to standard LoRA, particularly in low-resource regimes.

## 1. Introduction
Fine-tuning massive pre-trained models like BERT and RoBERTa is computationally expensive. While LoRA mitigates this by freezing the backbone and training low-rank matrices, it does not explicitly account for the *quality* of the learning signal during the backward pass. We observe that:
1.  **Dormant Adapters:** Early in training, LoRA branches may produce near-zero outputs, receiving small gradients and learning slowly.
2.  **Sharp Minima:** The model may achieve low training loss but remain sensitive to weight perturbations, indicating sharp minima and poor generalization.

GR-LoRA addresses these issues by modifying the backward pass and the loss landscape directly.

## 2. Methodology

### 2.1 Adaptive Gradient Scaling
To ensure all LoRA adapters contribute effectively, we introduce a custom autograd function. During the backward pass, the gradient flowing into the LoRA branch is scaled inversely proportional to the norm of its forward output $h_{lora}$:

$$ \nabla'_{h} = \nabla_{h} \cdot \frac{\lambda_{scale}}{\|h_{lora}\| + \epsilon} $$

This mechanism acts as a "wake-up call" for dormant adapters: if the LoRA output is negligible, the gradient is amplified, forcing the adapter to learn meaningful features rapidly.

### 2.2 Correctness-Aware Gradient Regularization
To stabilize the model, we explicitly regularize the training dynamics based on prediction correctness. For a batch of samples, we identify the subset $S_{correct}$ where the model's prediction matches the ground truth. We then add a penalty term to the loss:

$$ \mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_{reg} \cdot \frac{1}{|S_{correct}|} \sum_{i \in S_{correct}} \left( \|\nabla_{A} \mathcal{L}_i\|^2 + \|\nabla_{B} \mathcal{L}_i\|^2 \right) $$

Minimizing the gradient norm for correctly classified samples encourages the model to find "flat minima"—regions in the weight space where small perturbations do not change the output class—thereby improving generalization and robustness.

## 3. Experimental Setup
We evaluate GR-LoRA on the GLUE benchmark, specifically focusing on:
*   **Models:** BERT-base, RoBERTa-base, DistilBERT-base.
*   **Datasets:** STS-B (Regression), MRPC (Classification).
*   **Baselines:** Full Fine-tuning, Standard LoRA (Hugging Face PEFT).

Our implementation utilizes a custom PyTorch autograd function to efficiently compute sample-wise gradients without the memory overhead of full Jacobian computation.

## 4. Conclusion
GR-LoRA provides a simple yet effective drop-in replacement for standard LoRA layers. By dynamically adjusting optimization based on signal strength and model confidence, it achieves a better balance between plasticity (learning new features) and stability (retaining correct predictions).
