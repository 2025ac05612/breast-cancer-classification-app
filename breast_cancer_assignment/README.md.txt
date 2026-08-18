# Breast Cancer Diagnostic Classification Web App

This repository contains an end-to-end Machine Learning project developed as part of the M.Tech (AIML/DSE) Machine Learning Assignment 2. The objective is to build, evaluate, and deploy five distinct machine learning models to classify breast tumors as either Malignant or Benign.

## a. Problem Statement
Breast cancer is one of the most common cancers among women worldwide. Early detection significantly increases the chances of survival. Using clinical data containing physical characteristics of cell nuclei from breast mass aspirates, this project implements 5 classification algorithms to automatically and accurately diagnose tumors.

## b. Dataset Description
- **Dataset Source:** Breast Cancer Wisconsin (Diagnostic) Dataset.
- **Attributes (Features):** 30 continuous features describing characteristics of the cell nuclei (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension).
- **Target Variable:** Binary classification (`target`):
  - `0` for Malignant (cancerous)
  - `1` for Benign (non-cancerous)
- **Dataset Size:** 569 instances (455 training instances, 114 test instances).

## c. GitHub Repository Link
https://github.com/2025ac05612/breast-cancer-classification-app.git

## d. Models Used & Comparison Table
We evaluated 5 classic machine learning models on a stratified 20% test partition (114 samples). The features were standard-scaled before training and evaluation.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| **kNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| **Random Forest (Ensemble)** | 0.9561 | 0.9939 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

## e. Observations on Model Performance
- **Logistic Regression:** Achieved the highest performance across all evaluation criteria, including an Accuracy of 98.25% and an outstanding MCC of 0.9623. Since breast cancer datasets are often highly linearly separable after scaling, the linear decision boundary of Logistic Regression proved highly effective and stable without overfitting.
- **Decision Tree:** Displayed the lowest accuracy (91.23%) and MCC (0.8174). Individual decision trees are prone to high variance and overfitting on continuous datasets with small sample sizes.
- **kNN:** Performed strongly at 95.61% accuracy. Feature scaling played a key role here since kNN relies entirely on distance metrics.
- **Naive Bayes (Gaussian):** Provided respectable results (92.98% accuracy) under the assumption that continuous features follow normal distributions, but struggled slightly relative to discriminative models like Logistic Regression.
- **Random Forest (Ensemble):** Tied with kNN for second-best accuracy (95.61%) and generated a high AUC score (0.9939). The ensemble of multiple randomized decision trees significantly reduced variance and corrected the overfitting observed in the single Decision Tree.

### **Final Implication**
Logistic Regression is the overall winner. It achieved the highest scores on all major metrics, showing superior generalization capabilities on the unseen test set while maintaining high precision and recall (reducing false negatives, which is critical in healthcare settings).
