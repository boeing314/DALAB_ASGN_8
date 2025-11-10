# DA5401 A8: Ensemble Learning for Complex Regression Modeling on Bike Share Data

## Overview
This project applies and compares multiple **ensemble learning techniques** — Bagging, Boosting, and Stacking — to predict bike rental demand using the **Bike Sharing Dataset**.  
The objective is to demonstrate how combining models can reduce bias and variance, improving prediction accuracy over single models.

---

## Dataset
**Dataset:** [UCI Machine Learning Repository – Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)  
**Attributes:** Hourly data containing weather, time, and rental counts  
**Target Variable:** `cnt` (total count of bikes rented)

**Key preprocessing steps:**
- Dropped irrelevant columns: `instant`, `dteday`, `casual`, `registered`
- One-hot encoded categorical features (`season`, `weathersit`, `mnth`, `hr`, etc.)
- Performed an 80/20 train-test split

---

## ⚙️ Models Implemented

### **Part A — Baseline Models**
| Model | Description | RMSE |
|--------|--------------|------|
| Decision Tree Regressor (max_depth=6) | Single tree baseline | 118.46 |
| Linear Regression | Linear model baseline | **100.45** |

>  Linear Regression gave the lowest RMSE and was used as the baseline.

---

### **Part B — Ensemble Techniques**
| Model | Purpose | RMSE | Observation |
|--------|----------|------|--------------|
| Bagging Regressor | Reduces variance using bootstrapped Decision Trees | 101.46 | No improvement due to shallow base trees |
| Gradient Boosting Regressor | Sequentially reduces bias by correcting residuals | **44.43** | Major improvement in accuracy |

---

### **Part C — Stacking for Optimal Performance**
**Base Learners (Level-0):**
- K-Nearest Neighbors Regressor  
- Bagging Regressor  
- Gradient Boosting Regressor  

**Meta-Learner (Level-1):**
- Ridge Regression (tuned over multiple alpha values)

| Model | RMSE |
|--------|------|
| **Stacking Regressor** | **44.36** |

> Stacking slightly outperformed Boosting, showing optimal bias-variance trade-off.

---

## Key Insights
- **Bagging** failed to improve results because the base trees were already low-variance and underfit the data.
- **Boosting** effectively reduced bias by sequentially correcting prediction errors, drastically improving RMSE.
- **Stacking** achieved the best performance by combining diverse models (KNN, Bagging, Boosting) and learning optimal weights through Ridge Regression.

---

## Final RMSE Comparison

| Model | RMSE |
|--------|------|
| Baseline (Linear Regression) | 100.45 |
| Bagging Regressor | 101.46 |
| Gradient Boosting Regressor | 44.43 |
| Stacking Regressor | **44.36** |

> Both **Boosting and Stacking** perform similarly because they effectively reduce bias and handle complex non-linearities, but stacking benefits slightly from model diversity and meta-learning.

---

## Tools and Libraries
- Python 3.11  
- pandas, numpy  
- scikit-learn  
- matplotlib (for visualizations)  

---
