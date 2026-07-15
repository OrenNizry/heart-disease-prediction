# Heart Disease Prediction Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OrenNizry/heart-disease-prediction/blob/main/heart_disease_prediction.ipynb)

> **Author:** Oren Nizry (208708784)  
> **Language:** Python 3 · Jupyter Notebook  
> **Dataset:** `heart.csv` — Cleveland Heart Disease dataset

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
  - [1. Data Exploration](#1-data-exploration)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Visualization & Correlation](#4-visualization--correlation)
  - [5. Train / Validation / Test Split](#5-train--validation--test-split)
  - [6. Model Selection](#6-model-selection)
  - [7. Hyperparameter Tuning](#7-hyperparameter-tuning)
  - [8. Overfitting Check](#8-overfitting-check)
  - [9. Final Evaluation](#9-final-evaluation)
- [Results](#results)
- [Libraries](#libraries)
- [How to Run](#how-to-run)

---

## Overview

A supervised machine learning project that builds a binary classification model to predict whether a patient has heart disease (`target = 1`) or not (`target = 0`).

Three classifiers are compared — **K-Nearest Neighbors**, **Random Forest**, and **Decision Tree** — and the best model is tuned with `GridSearchCV` and evaluated on a held-out test set using accuracy, precision, recall, F1 score, and a confusion matrix.

---

## Dataset

The dataset (`heart.csv`) is based on the **Cleveland Heart Disease** dataset from the UCI Machine Learning Repository.

| Feature | Description |
|---|---|
| `age` | Age in years (converted to `age_group` — see below) |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia type (1–3) |
| `target` | **Label** — 1 = heart disease, 0 = no heart disease |

---

## Pipeline

### 1. Data Exploration

```python
print(data.head())
data.info()
data.describe()
```

Initial inspection of shape, data types, and basic statistics.

---

### 2. Data Cleaning

```python
# Check and remove duplicate rows
data = data.drop_duplicates()

# Verify no missing values
print(data.isnull().sum())
```

Duplicate rows were identified and removed. No null values were found in the dataset.

---

### 3. Feature Engineering

The continuous `age` column was bucketed into 4 ordinal groups, then the original column was dropped:

```python
def categorize_age(age):
    if age < 20:   return 0   # Under 20
    elif age < 40: return 1   # 20–39
    elif age < 60: return 2   # 40–59
    else:          return 3   # 60+

data['age_group'] = data['age'].apply(categorize_age)
data = data.drop('age', axis=1)
```

This reduces noise from treating age as a continuous linear predictor and groups patients into clinically meaningful bands.

---

### 4. Visualization & Correlation

- **Histograms** of all numerical features to understand distributions
- **Heatmap** of the full correlation matrix to identify strongly correlated feature pairs and their relationship with `target`

```python
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
```

---

### 5. Train / Validation / Test Split

Following the course convention: **70% train / 15% validation / 15% test**

```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

K-fold cross-validation (`cross_val_score`) was also applied during model selection to compensate for the limited dataset size after deduplication.

---

### 6. Model Selection

Three classifiers were trained and compared on the validation set:

| Model | Notes |
|---|---|
| `KNeighborsClassifier` | Distance-based, sensitive to scale |
| **`RandomForestClassifier`** | Ensemble of decision trees — best performer |
| `DecisionTreeClassifier` | Single tree, prone to overfitting |

All three are scored on **Accuracy, Precision, Recall, and F1**, then plotted as grouped bar charts for direct comparison.

---

### 7. Hyperparameter Tuning

`GridSearchCV` (5-fold CV) was applied to the winning **Random Forest** model to find the optimal combination of:

```python
param_grid = {
    'n_estimators':      [25, 50, 100, 200],
    'max_depth':         [6, 10, 20, 30],
    'min_samples_split': [2, 4, 6, 8]
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
```

---

### 8. Overfitting Check

Training accuracy vs. validation accuracy is plotted for all three models:

```python
train_vs_val_df.plot(kind='bar')
plt.title('Training vs Validation Accuracy Comparison')
```

A large gap between training and validation accuracy signals overfitting. The tuned Random Forest was selected as the best balance between performance and generalization.

---

### 9. Final Evaluation

The tuned Random Forest is evaluated on the **held-out test set** (never seen during training or tuning):

```python
y_test_pred = best_rf.predict(X_test)
```

Metrics reported:

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted positives, how many are truly positive |
| **Recall** | Of all actual positives, how many were caught |
| **F1 Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Visualized as an annotated heatmap |

---

## Results

The **Random Forest** classifier (after GridSearchCV tuning) outperformed KNN and Decision Tree across all metrics on both the validation and test sets.

The final confusion matrix confirms reliable discrimination between disease-positive and disease-negative patients with minimal false negatives — critical in a medical prediction context.

---

## Libraries

| Library | Usage |
|---|---|
| `pandas` | Data loading, manipulation, deduplication |
| `numpy` | Numerical operations |
| `matplotlib` | Histograms, bar charts, confusion matrix plots |
| `seaborn` | Correlation heatmap |
| `scikit-learn` | Models, splitting, cross-validation, GridSearchCV, metrics |

---

## How to Run

**Option 1 — Google Colab (no setup needed):**

Click the badge at the top → [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OrenNizry/heart-disease-prediction/blob/main/heart_disease_prediction.ipynb)

Upload `heart.csv` to the Colab session storage when prompted.

**Option 2 — Local Jupyter:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook heart_disease_prediction.ipynb
```

Ensure `heart.csv` is in the same directory as the notebook.
