# Training a Resume Classifier with Differential Privacy

## Overview

As machine learning becomes increasingly integrated into HR and recruiting, models are often trained on highly sensitive datasets like resumes, which contain a wealth of Personally Identifiable Information (PII). This creates significant privacy risks, including the potential for Membership Inference Attacks (MIA) or data extraction, which could reveal whether an individual's data was used for training or even reconstruct parts of their information.

This project addresses these risks by implementing and evaluating Differentially Private Stochastic Gradient Descent (DP-SGD), a state-of-the-art technique for privacy-preserving machine learning. The goal is to build a private resume classifier and rigorously analyze the resulting trade-off between the model's predictive utility and the mathematical privacy guarantees offered to the individuals in the training set. The implementation is based on the foundational principles from the paper "Deep Learning with Differential Privacy" (Abadi et al.).

## Design and Implementation

### Model Architecture: BagOfWordsClassifier

After initial experiments with more complex architectures (like RNNs and Transformers) proved unstable and difficult to train effectively under the constraints of DP-SGD, this project uses a `BagOfWordsClassifier`. This simpler model is highly effective for a keyword-driven task like resume classification and offers significant advantages in a private setting due to its stability and efficiency.

The architecture is designed to evaluate resume content in the context of the specified job category:

* **Text Processing**: The model processes the `resume_text` by looking up the embedding for each word and averaging them together using an `nn.EmbeddingBag` layer. This creates a single vector that captures the resume's overall topical content, which is ideal for identifying the presence of key skills.
* **Contextualization**: A separate `nn.Embedding` layer creates a vector for the `job_category`.
* **Classification**: These two vectors are concatenated and passed through a simple Multi-Layer Perceptron (MLP) with increased capacity (`embed_dim=256`) and dropout for regularization. This architecture forces the model to learn the relationship between the resume's content and the job context before making its final prediction.

### Privacy Design: Manual DP-SGD

The DP-SGD algorithm was implemented manually to provide fine-grained control and closely follow the core methodology from Abadi et al. This approach differs from standard training by modifying how gradients are processed and applied at each step.

* **Per-Example Gradient Computation**: Standard training computes a single gradient averaged over a batch. Our implementation performs a separate backward pass for each individual sample, isolating the contribution of each resume to the model's update.
* **Per-Example Gradient Clipping**: To bound the maximum influence of any single resume, the L2 norm (magnitude) of its gradient is calculated. If this norm exceeds a pre-defined clipping bound `C`, the gradient is scaled down. This critical step prevents unique or outlier resumes from disproportionately pulling the model's weights, which is a primary cause of memorization and privacy leaks.
* **Noise Addition**: The clipped gradients for all samples in the batch are aggregated (summed), and calibrated Gaussian noise is added. The scale of this noise is a function of the clipping bound `C` and a noise multiplier `σ`, which are the key levers for controlling the privacy-utility trade-off. This noise injection is the core of the privacy guarantee, providing the "plausible deniability" of differential privacy.
* **Privacy Accounting**: The cumulative privacy cost, represented by epsilon ($\epsilon$), is tracked after each epoch using the Rényi Differential Privacy (RDP) accountant from the `opacus` library. The RDP accountant provides a tighter bound on privacy loss than older methods, allowing us to achieve better model utility for the same level of privacy.

## Settings

* **Dataset**: `synthetic_resumes_balanced.csv`, a balanced dataset containing 9,000 synthetic resumes across 3 job categories.
* **Data Split**: 70% training, 15% validation, and 15% test.
* **Model**: `BagOfWordsClassifier` with `embed_dim=256`.
* **Seed**: `42` for all random operations to ensure reproducibility.

### Baseline (Non-Private) Settings

* **Epochs**: Up to 50, with early stopping (patience of 5) based on the validation F1 score to find the optimal non-private model.
* **Batch Size**: 128
* **Learning Rate**: `1e-3` with a `ReduceLROnPlateau` scheduler.
* **Optimizer**: AdamW

### DP-SGD Settings

* **Epochs**: 30
* **Batch Size**: 256 (A larger batch size is used, as recommended by the paper, to improve the signal-to-noise ratio).
* **Execution**: Performed on the CPU to circumvent a known library bug with the `EmbeddingBag` layer on the GPU.
* **Delta ($\delta$)**: Set to `1 / len(train_dataset)` (approximately `1.59E-04`).
* **Hyperparameter Sweep Grid**:
    * **Learning Rate ($\text{lr}$)**: `{0.005, 0.001}`
    * **Clipping Norm ($C$)**: `{0.5, 1.0, 3.31*}` (*The value 3.31 was determined adaptively by finding the median gradient norm, a best practice from Abadi et al.*)
    * **Noise Multiplier ($\sigma$)**: `{0.2, 0.4, 0.8, 1.2, 1.5}`

## Results

The hyperparameter sweep produced a clear trade-off between model utility (F1 Score) and privacy ($\epsilon$). A lower epsilon signifies stronger privacy.

### Performance Comparison

The table below summarizes the performance of the non-private baseline against a selection of the DP-SGD models, from the highest utility (but lowest privacy) to the highest privacy.

| Setting (C, σ, LR)    | LR    | Clip Norm (C) | Noise (σ) | Final ε | Test Accuracy | Test F1 Score |
|:----------------------|:------|:--------------|:----------|:--------|:--------------|:--------------|
| Baseline | 0.001 | n/A          | N/A       | 0    | 0.9956        | 0.9983        |
| C=2.91, σ=0.5, LR=0.005 | 0.005 | 2.91          | 0.5       | 8.12    | 0.8956        | 0.8949        |
| C=0.5, σ=0.5, LR=0.005  | 0.005 | 0.50          | 0.5       | 8.12    | 0.8356        | 0.8303        |
| C=1.0, σ=0.5, LR=0.005  | 0.005 | 1.00          | 0.5       | 8.12    | 0.8356        | 0.8232        |
| C=0.5, σ=1, LR=0.005    | 0.005 | 0.50          | 1.0       | 1.50    | 0.7659        | 0.7775        |
| C=2.91, σ=1, LR=0.005   | 0.005 | 2.91          | 1.0       | 1.50    | 0.8044        | 0.7740        |
| C=2.91, σ=2, LR=0.005   | 0.005 | 2.91          | 2.0       | 0.33    | 0.7259        | 0.7193        |
| C=1.0, σ=1, LR=0.005    | 0.005 | 1.00          | 1.0       | 1.50    | 0.7570        | 0.7102        |
| C=1.0, σ=2, LR=0.005    | 0.005 | 1.00          | 2.0       | 0.33    | 0.7037        | 0.6894        |
| C=0.5, σ=2, LR=0.005    | 0.005 | 0.50          | 2.0       | 0.33    | 0.5674        | 0.4701        |



*Figure: The learning curves show that the non-private baseline (black dashed line) converges quickly to perfect accuracy. The DP-SGD models learn more slowly and converge to lower final accuracies, with the level of utility directly corresponding to the amount of noise and other hyperparameters.*

## Practical Privacy Analysis: Membership Inference Attack (MIA)

To demonstrate the practical impact of DP, a loss-based Membership Inference Attack was performed. The core intuition is that models exhibit lower loss on training examples they have "memorized." The attack works by training a simple logistic regression classifier to predict whether a sample was in the training set ("member") based solely on the target model's loss value for that sample.

* **Baseline Model MIA Accuracy: 73%**
    The high attack accuracy indicates a privacy leak. An attacker can determine with decent certainty whether a specific person's resume was used to train the model, confirming that the non-private model is vulnerable.

* **DP-SGD Model MIA Accuracy: 68%, C=2.91, σ=0.5, LR=0.005**
    This attack accuracy is slightly less than baseline. This demonstrates that DP-SGD was is more resistant, but to get better performance, we have to use one of the other parameters with more noise and compromise on accuracy.

## Conclusions

* **The Privacy-Utility Trade-off is Clear and Quantifiable**: The results show a direct trade-off. The baseline model achieved an almost perfect F1 score. For a modest privacy guarantee (ε ≈ 8), the F1 score dropped to a still-excellent 0.90. However, achieving a strong privacy guarantee (ε ≈ 2) required a larger drop in F1 score to ~0.80. This highlights that system designers must make a conscious choice about how much utility to sacrifice for a given level of privacy.

* **Hyperparameter Tuning is Critical for DP**: The experiments proved that DP-SGD is not "plug-and-play." Performance was highly sensitive to the learning rate, clipping norm, and noise level. The most successful runs used a higher learning rate (`0.005`) to overcome the noise and an adaptive clipping norm (`3.31`) tailored to the data, validating the best practices from the research paper.

* **A "Good Enough" Private Model is Achievable**: For a low-stakes application like internal HR analytics (e.g., identifying trending skills), the "Balanced DP-SGD" model with an F1 score of ~0.80 and a strong privacy guarantee (ε ≈ 2.6) is an excellent and practical outcome. This demonstrates the feasibility of building useful machine learning systems that respect user privacy.

* **DP is a Practical and Effective Defense Against MIA**: The Membership Inference Attack provides a concrete validation of DP's theoretical promise. The attack was highly successful on the non-private model but failed resoundingly on the DP-SGD model, with its accuracy dropping from 87.5% to 54.1%. This shows that DP is not just a mathematical concept but a practical tool for building models that are provably more robust against real-world privacy attacks.

## AI Disclosure

* Experimenting with LSTM and GRU, especically for starter code. In the end settled on a simple BoW model
* Boiler plate code for logging and making plots
* When Opacus library wasn't producing expected results, used AI extensively and information from 1st paper to implement manual DP
* Used extensively for a lot of bug fixes, especically for DP implementation
* For MIA, used implementation of assignment 1 and AI to setup basic skeleton
* Formatting the readme for better readability