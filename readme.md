# Bank Marketing Campaign — Machine Learning Analysis

This project analyzes the **Bank Marketing dataset** and builds several machine learning models to
predict whether a client will subscribe to a term deposit (`y` variable).  
The notebook includes data preprocessing, feature engineering, PCA, feature selection, model training,
evaluation, and association rule mining.

---

## 🚀 Project Overview

The notebook performs:

### ✔ Data Loading & Exploration
- Reads `bank-additional.csv` (semicolon-separated)
- Checks dataset shape, info, null values, distributions

### ✔ Feature Engineering
- Splits features into numerical / categorical / ordinal groups
- Label encoding for binary columns
- One-hot encoding for categorical columns
- Feature scaling using StandardScaler

### ✔ Dimensionality Reduction
- PCA applied to numerical features  
- Retains 95% variance  
- PCA components merged back into the final dataset

### ✔ Feature Selection
- Chi-square test on categorical features  
- Extracts Top 10 most important features  
- Visualized in barplots

### ✔ Visualization
- Target distribution  
- Numerical feature distributions  
- Categorical counts  
- PCA explained variance visualization

### ✔ Machine Learning Models
Trains and compares:

1. **Linear Regression (converted to classifier)**
2. **Logistic Regression**
3. **K-Means clustering (treated as classifier)**
4. **Random Forest Classifier**

Evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix  

Also includes **5-fold cross-validation** for Logistic Regression.

### ✔ Association Rule Mining
Using `mlxtend`:
- Generates frequent itemsets with FP-Growth  
- Extracts association rules related to `y = 1`  
- Useful for marketing insights

---

## 📁 Project Structure
.
├── bank_analysis.ipynb
├── README.md
├── requirements.txt
└── .gitignore

---

## 📥 Dataset Download (IMPORTANT)

The dataset **is NOT included** in this repository due to licensing restrictions.

Download from UCI Machine Learning Repository:

🔗 https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

After downloading, place:

bank-additional.csv

in the **project root folder** (same folder as the notebook).

---

## ▶ Running the Notebook

Install dependencies:

```bash
pip install -r requirements.txt


Launch Jupyter:

jupyter notebook

Open:

bank_analysis.ipynb


##  🛠 Tools & Libraries Used
	•	Python
	•	Jupyter Notebook
	•	Pandas, NumPy
	•	Scikit-learn
	•	Seaborn, Matplotlib
	•	MLxtend
	•	PCA, Chi-Square Feature Selection
	•	K-Fold Cross Validation
	•	FP-Growth & Association Rules
