# Overview
This repository contains a Jupyter notebook that implements an adversarial attack pipeline on an image to text or vision language model and  image based and prompt-based defenses. 

# Design and Implementation
## Attack design
The attack implements a gradient based differentiable pixel perturbation method that optimizes a loss objective to cause the model to output a target text or to change its high level prediction The attack iteratively perturbs the image within a bounded L infinity budget and uses projected gradient steps to enforce the constraint The notebook supports both targeted and untargeted variants and logs intermediate model outputs per iteration

## Defense design
Gaussian Blur : Gaussian Blur is an image-based preprocessing defense that smooths out high-frequency details in an input image before it is passed to the model. Since most adversarial perturbations used in multimodal attacks are subtle, high-frequency pixel-level changes, applying a slight blur (e.g., radius 0.8) helps remove or weaken these perturbations without heavily distorting the overall image. This reduces the effectiveness of attacks that attempt to alter the model’s visual perception or inject misleading cues through imperceptible noise, thereby improving robustness against image-level adversarial manipulations.

System-Prompt Reinforcement : System-Prompt Reinforcement is a text-based defense that wraps user and image inputs with a strong system-level instruction layer. It explicitly tells the multimodal model to ignore any hidden, injected, or manipulative instructions that may appear within text or image content. By reasserting the model’s safe role and filtering the interpretation scope to only verified user intent, this defense mitigates prompt-injection and cross-modal attacks—where attackers hide malicious commands inside captions, OCR text, or visual elements—to maintain consistent, safe, and policy aligned behavior.

# Metrics and Results

## Attack Demo

![Input Image](results/input.png)

![Adv Response](results/response.png)

![pgd](results/pgd.png)

## Targeted attack success
An individual attack instance is counted as a success if the model output contains the target phrase or token within the model response matching normalized whitespace and case insensitive comparison

## Loose match success
An auxiliary success criterion considers partial overlap by token Jaccard similarity where overlap at or above 0.6 is considered a success

## Attack success rate
Attack success rate is defined as the fraction of test images for which the attack meets the success criterion on the original model without defense

## Post defense attack success rate
Post defense attack success rate is the fraction of test images for which the same adversarial perturbation still causes the model to meet the success criterion after the defense is applied

## Quantitative Results 

![Quantitative Results](results/quant.png)

After Implementing the defences, the attack success rate dropped to zero.

# Discussion

## Limitations in attack or defense
Single defense limitations
A single low cost image transform is easy to implement but provides limited robustness under adaptive adversaries An attacker aware of the defense can incorporate the transformation into the attack loop using expectation over transformation or differentiable approximations and thus recover high success rates
Perceptual impact
Some defenses such as aggressive compression or heavy smoothing degrade image quality and may impact downstream model performance on benign inputs creating a tradeoff between robustness and utility
Model dependence
Attack and defense outcomes depend heavily on the target model architecture tokenization and pre processing pipeline Results reported here are specific to the experimental model and dataset and do not necessarily generalize across models
Evaluation limitations
Using a single test set size and a single random seed can overstate stability of metrics The notebook reports point estimates and simple confidence intervals should be computed for robust claims

## Open challenges
Adaptive adversaries
Designing defenses that remain robust when adversaries are fully aware of the defense and optimize against it is an open challenge
Transferability
Attacks that transfer across models and defenses remain a practical concern for real world deployment
Perceptual fidelity versus robustness
Finding defenses that maintain high perceptual fidelity and preserve accuracy on benign samples while providing strong robustness is difficult
Certified guarantees
Moving from empirical defenses to defenses with provable guarantees under realistic threat models is an active research direction

## Possible future improvements
Ensemble of defenses
Combine multiple complementary transformations and randomized pipelines to increase cost and complexity for adaptive attackers
Adversarial training and model level defenses
Incorporate defended samples into model training or fine tuning to improve model robustness without heavy pre processing
Expectation over transformation attacks and defenses
Evaluate defenses under expectation over transformation attacks and design defenses that explicitly account for adaptive optimization
Certified defenses and randomized smoothing
Explore randomized smoothing and other methods that can provide certified robustness bounds under certain norms
Comprehensive evaluation
Run ablation studies vary budgets seeds and test set sizes and report confidence intervals and statistical significance for all reported metrics
