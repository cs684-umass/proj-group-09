## Overview
This repository contains a Jupyter notebook that implements an adversarial attack pipeline on an image to text or vision language model and a single simple image based defense. 

## Design and Implementation
# Attack design
The attack implements a gradient based differentiable pixel perturbation method that optimizes a loss objective to cause the model to output a target text or to change its high level prediction The attack iteratively perturbs the image within a bounded L infinity budget and uses projected gradient steps to enforce the constraint The notebook supports both targeted and untargeted variants and logs intermediate model outputs per iteration

# Defense design
Implements a Gaussian blur–based image preprocessing defense that smooths the adversarial noise before feeding the image back to the model.

## Metrics and Results

# Targeted attack success
An individual attack instance is counted as a success if the model output contains the target phrase or token within the model response matching normalized whitespace and case insensitive comparison

# Loose match success
An auxiliary success criterion considers partial overlap by token Jaccard similarity where overlap at or above 0.6 is considered a success

# Attack success rate
Attack success rate is defined as the fraction of test images for which the attack meets the success criterion on the original model without defense

# Post defense attack success rate
Post defense attack success rate is the fraction of test images for which the same adversarial perturbation still causes the model to meet the success criterion after the defense is applied


## Discussion
# Limitations in attack or defense
Single defense limitations
A single low cost image transform is easy to implement but provides limited robustness under adaptive adversaries An attacker aware of the defense can incorporate the transformation into the attack loop using expectation over transformation or differentiable approximations and thus recover high success rates
Perceptual impact
Some defenses such as aggressive compression or heavy smoothing degrade image quality and may impact downstream model performance on benign inputs creating a tradeoff between robustness and utility
Model dependence
Attack and defense outcomes depend heavily on the target model architecture tokenization and pre processing pipeline Results reported here are specific to the experimental model and dataset and do not necessarily generalize across models
Evaluation limitations
Using a single test set size and a single random seed can overstate stability of metrics The notebook reports point estimates and simple confidence intervals should be computed for robust claims

# Open challenges
Adaptive adversaries
Designing defenses that remain robust when adversaries are fully aware of the defense and optimize against it is an open challenge
Transferability
Attacks that transfer across models and defenses remain a practical concern for real world deployment
Perceptual fidelity versus robustness
Finding defenses that maintain high perceptual fidelity and preserve accuracy on benign samples while providing strong robustness is difficult
Certified guarantees
Moving from empirical defenses to defenses with provable guarantees under realistic threat models is an active research direction

# Possible future improvements
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
