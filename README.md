# 📉 Bankruptcy Prediction in Taiwan 🇹🇼

This project uses financial indicators to predict company bankruptcy in Taiwan using a full machine learning pipeline. It includes preprocessing, class imbalance handling, Random Forest training, hyperparameter tuning, and feature importance analysis.

---

## 🚀 Project Highlights

- ✅ Loaded and cleaned financial data with 95 features  
- ⚖️ Addressed class imbalance using `RandomOverSampler`  
- 🌲 Trained a `RandomForestClassifier` with `GridSearchCV`  
- 🧪 Evaluated model performance using:
  - Accuracy  
  - Confusion matrix  
  - Classification report  
- 🔍 Analyzed feature importance (Gini Index)  
- 💾 Saved the trained model using `pickle`

---

## 🧠 Dataset

- **Source:** UCI Machine Learning Repository  
- **Target column:** `bankrupt` (binary)  
- **Shape:** 6819 companies × 96 columns  
- Highly imbalanced dataset (bankrupt companies ≈ 0.6%)

---

