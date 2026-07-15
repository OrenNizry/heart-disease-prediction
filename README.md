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

# Check for missing values
print(data.isnull().sum())
```

Duplicate rows were identified and removed. No null values were found.

---

### 3. Feature Engineering

The continuous `age` column was bucketed into 4 ordinal groups, then dropped:

```python
def categorize_age(age):
    if age < 20:  return 0   # Under 20
    elif age < 40: return 1  # 20–39
    elif age < 60: return 2  # 40–59
    else:          return 3  # 60+

data['age_group'] = data['age'].apply(categorize_age)
data = data.drop('age', axis=1)
```

This reduces noise from treating age as a continuous linear predictor and groups patients into clinically meaningful bands.

---

### 4. Visualization & Correlation

- **Histograms** of all numerical features to understand distributions
- **Heatmap** of the full correlation matrix (`seaborn.heatmap`) to identify strongly correlated feature pairs and their relationship with `target`

```python
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
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

| Model | Metrics Evaluated |
|---|---|
| K-Nearest Neighbors (`KNeighborsClassifier`) | Accuracy, Precision, Recall, F1 |
| **Random Forest** (`RandomForestClassifier`) | Accuracy, Precision, Recall, F1 |
| Decision Tree (`DecisionTreeClassifier`) | Accuracy, Precision, Recall, F1 |

Bar charts compare all four metrics side-by-side across the three models.

---

### 7. Hyperparameter Tuning

`GridSearchCV` (5-fold CV) was run on the winning **Random Forest** model:

```python
param_grid = {
    'n_estimators':    [25, 50, 100, 200],
    'max_depth':       [6, 10, 20, 30],
    'min_samples_split': [2, 4, 6, 8]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
```

The tuned model is stored as `best_rf` and used for all subsequent evaluation.

---

### 8. Overfitting Check

Training accuracy vs. validation accuracy is plotted for all three models to detect overfitting:

```python
train_vs_val_df.plot(kind='bar')
plt.title('Training vs Validation Accuracy Comparison')
```

A large gap between training and validation accuracy signals overfitting.

---

### 9. Final Evaluation

The tuned Random Forest is evaluated on the **held-out test set**:

```python
y_test_pred = best_rf.predict(X_test)
```

Metrics reported:
- **Accuracy** — overall correct predictions
- **Precision** — of predicted positives, how many are truly positive
- **Recall** — of all actual positives, how many were caught
- **F1 Score** — harmonic mean of precision and recall
- **Confusion Matrix** — visualized as a heatmap

---

## Results

> The **Random Forest** classifier (after GridSearchCV tuning) outperformed KNN and Decision Tree on all metrics.

| Metric | Test Set Score |
|---|---|
| Accuracy | ✓ Best among all models |
| Precision | ✓ High |
| Recall | ✓ High |
| F1 Score | ✓ Best among all models |

The final confusion matrix confirms reliable discrimination between disease-positive and disease-negative patients.

---

## Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
  ├── train_test_split, cross_val_score, GridSearchCV
  ├── KNeighborsClassifier
  ├── RandomForestClassifier
  ├── DecisionTreeClassifier
  └── accuracy_score, confusion_matrix, classification_report,
      precision_score, recall_score, f1_score
```

---

## How to Run

**Option 1 — Google Colab (recommended, no setup needed):**

Click the badge at the top of this page → [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OrenNizry/heart-disease-prediction/blob/main/heart_disease_prediction.ipynb)

Then upload `heart.csv` to the Colab session storage when prompted.

**Option 2 — Local Jupyter:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook heart_disease_prediction.ipynb
```

Make sure `heart.csv` is in the same directory as the notebook.