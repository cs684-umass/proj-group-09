
# AI Recruiting System - Membership Inference Attack Analysis

A comprehensive privacy analysis of an AI-powered resume screening system, implementing and evaluating membership inference attacks and training data extraction techniques.

## Overview

This project demonstrates the privacy vulnerabilities inherent in machine learning models used for AI recruiting. Through detailed analysis of a resume classification system, we explore how an attacker might infer training data membership and attempt to extract sensitive information from the model.

## Dataset

### Use Case
The system evaluates resume suitability for different job categories in an AI recruiting pipeline, classifying resumes as "suitable" (1) or "not suitable" (0) for specific positions.

### Sample Size & Label Distribution
- **Total Samples**: 200 synthetic resumes
- **Job Categories**: 4 categories with 50 resumes each
  - Software Engineer: 50 samples (25 suitable, 25 unsuitable)
  - Data Scientist: 50 samples (25 suitable, 25 unsuitable) 
  - Marketing Professional: 50 samples (25 suitable, 25 unsuitable)
  - Sales Professional: 50 samples (25 suitable, 25 unsuitable)

### Generation Method
Resumes were synthetically generated using large language models with category-specific templates and constraints:

**Software Engineer Templates**:
- Suitable: Focus on Python, Java, React, software development experience
- Unsuitable: Data science, marketing, or sales backgrounds

**Data Scientist Templates**:
- Suitable: Python, R, ML libraries, statistical modeling experience
- Unsuitable: Software engineering, marketing, or sales backgrounds

**Marketing Professional Templates**:
- Suitable: SEO, social media, campaign management experience
- Unsuitable: Technical programming or data science backgrounds

**Sales Professional Templates**:
- Suitable: CRM systems, lead generation, relationship building experience
- Unsuitable: Technical or marketing-focused backgrounds

### Example Resume Entries

**Software Engineer (Suitable)**:
```
Roy Patterson - roy.patterson@icloud.com
Software Engineer with a strong background in programming and software development. 
Proficient in programming languages such as Python, Java, and React. Adept at designing 
and developing scalable software solutions to meet complex business needs.
```

**Data Scientist (Suitable)**:
```
Anna Davis - anna.davis@icloud.com  
Data Scientist with expertise in data analysis, machine learning, and statistical modeling.
Proficient in Python, R, and ML libraries, with a proven track record in delivering 
high-quality data-driven insights.
```

**Marketing Professional (Unsuitable for Software Engineer)**:
```
Tyler Jackson - tyler.jackson@hotmail.com
Marketing Professional with 5 years of experience in SEO, social media, and campaign 
management. Proven ability to drive growth through creative and data-driven strategies.
```

## Attack Implementation

### Membership Inference Attack

#### Design Choices
- **Loss-based approach**: Leverages the principle that models typically have lower loss on training data
- **Calibration methodology**: Uses 50% of data for threshold optimization, 50% for evaluation
- **Statistical analysis**: Comprehensive ROC curve analysis and performance metrics

#### Implementation Details
1. **Data Split**: Training/test data split into calibration and evaluation sets
2. **Loss Computation**: Per-example cross-entropy losses calculated for all samples
3. **Threshold Optimization**: Best threshold found using calibration set (accuracy: 79.0%)
4. **Attack Execution**: Threshold applied to evaluation set for membership prediction

#### Key Technical Components
```python
# Core attack methodology
def membership_inference_attack(model, train_data, test_data, tokenizer):
    # Split data for calibration and evaluation
    cal_losses = compute_losses(calibration_data)
    optimal_threshold = optimize_threshold(cal_losses, cal_labels)

    # Execute attack on evaluation set
    eval_losses = compute_losses(evaluation_data)
    predictions = (eval_losses <= optimal_threshold)
    return predictions, metrics
```

### Training Data Extraction Attack

#### Multi-Method Approach
1. **Template-based extraction**: Leverages knowledge of resume structure
2. **Query-based extraction**: Systematic model querying for pattern detection  
3. **Gradient-based extraction**: Optimization-based content reconstruction

#### Implementation Highlights
- **Resume-specific vocabulary**: Targeted keyword extraction for recruiting domain
- **Template matching**: Structured approach using job category templates
- **Confidence scoring**: Multi-metric evaluation of extraction success

## Results & Metrics

### Membership Inference Attack Performance

#### Core Metrics
- **AUC Score**: 0.694 (moderate attack strength)
- **Attack Accuracy**: 80.0%  
- **Optimal Threshold**: 0.7214

#### Detailed Performance
- **True Positive Rate (TPR)**: 96.3% - Successfully identified members
- **False Positive Rate (FPR)**: 85.0% - Non-members misclassified as members  
- **Precision**: 81.9%
- **Specificity**: 15.0%
- **F1-Score**: 88.5%

#### Privacy Risk Assessment
- **Risk Level**: MEDIUM
- **Vulnerability**: An attacker can determine resume membership with 80.0% accuracy
- **Impact**: 96.2% of training resumes can be correctly identified

### Training Data Extraction Results

#### Extraction Attempts
- **Total Extracted Samples**: 6
- **Template-based**: 4 potential samples
- **Query-based**: 0 high-confidence patterns  
- **Gradient-based**: 2 reconstructed samples

#### Success Analysis
- **Exact Matches**: 0
- **Partial Matches**: 0
- **Semantic Matches**: 0
- **Precision**: 0.000
- **Recall**: 0.000

### Loss Analysis Statistics

#### Training vs Test Loss Distributions
**Calibration Set**:
- Members (Train): Mean=0.6843, Std=0.0216
- Non-Members (Test): Mean=0.6865, Std=0.0193

**Evaluation Set**:
- Members (Train): Mean=0.6877, Std=0.0191  
- Non-Members (Test): Mean=0.7022, Std=0.0212

#### Privacy Leakage Indicators
- **Calibration Loss Gap**: 0.0022 (Non-member - Member)
- **Evaluation Loss Gap**: 0.0145 (Non-member - Member)
- **Loss Range Overlap**: 75.54%

## Vulnerability Analysis

### Attack Strength vs Reconstruction Quality

#### Mismatch Analysis
- **MIA Strength (AUC)**: 0.694
- **Extraction Success**: 0.000  
- **Mismatch Score**: 0.694
- **Relationship Type**: MODERATE_MISMATCH

#### Why MIA Succeeded
- Statistical patterns in confidence scores
- Loss differences between train/test distributions
- Model overfitting to training data characteristics

#### Why Extraction Failed
- Classification model architecture (no text generation capability)
- No direct content memorization mechanism
- Template matching unsuccessful due to synthetic data uniformity

### Privacy Implications

#### Risk Assessment
- **Membership Privacy**: PROTECTED (despite successful inference)
- **Content Privacy**: PROTECTED (no successful extraction)
- **Overall Risk**: HIGH (due to membership inference capability)

#### Practical Impact for AI Recruiting
- Attackers could identify which resumes were used for training
- Potential discrimination concerns if training data membership is revealed
- Privacy violations for job candidates whose resumes were in training set
- Regulatory compliance issues under GDPR/CCPA frameworks

## Model Training Summary

### Final Training Metrics
- **Final Training Loss**: 0.6251
- **Final Validation Loss**: 0.7112  
- **Best Validation Loss**: 0.6943
- **Final Accuracy**: 47.5%

### Training Characteristics
- **Model**: DistilBERT-base-uncased for sequence classification
- **Epochs**: 5
- **Batch Size**: 8  
- **Learning Rate**: Default transformer settings
- **Optimization**: AdamW optimizer with weight decay

## Defensive Recommendations

### Immediate Measures
1. **Differential Privacy**: Implement noise injection during training
2. **Data Sanitization**: Remove or anonymize sensitive information
3. **Query Rate Limiting**: Implement monitoring and restrictions on model access
4. **Regular Auditing**: Systematic evaluation of model outputs for memorization

### Long-term Solutions
1. **Federated Learning**: Distribute training across multiple parties
2. **Secure Aggregation**: Cryptographic protection of model updates
3. **Privacy-Preserving ML**: Adopt techniques like homomorphic encryption
4. **Data Minimization**: Reduce training data exposure through careful curation

## Technical Implementation

### Key Dependencies
```python
# Core ML framework
torch>=1.9.0
transformers>=4.0.0

# Analysis and visualization  
scikit-learn>=0.24.0
matplotlib>=3.0.0
numpy>=1.19.0

# Attack implementation
tqdm>=4.60.0
```

### File Structure
```
├── 690f_assignment1_final.ipynb  # Main Code
├── images                        # Results plotting and charts
└── final_dataset.json            # Generated Resumes
```

## Conclusions

The analysis reveals a **moderate privacy vulnerability** in the AI recruiting system. While membership inference attacks achieved 80% accuracy, training data extraction was unsuccessful. This suggests the model exhibits statistical overfitting without direct content memorization.

**Key Takeaways**:
- Classification models can leak membership information through loss patterns
- Synthetic training data may provide some protection against content extraction
- Privacy risks in AI recruiting require proactive mitigation strategies
- Regular privacy auditing should be standard practice for production ML systems

![image info](assignment-1/code/assets/loss.jpeg)
![image info](assignment-1/code/assets/log_roc.jpeg)
![image info](assignment-1/code/assets/auc.jpeg)
