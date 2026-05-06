# 🫀 Liver Disease Prediction — ML Ensemble Model

A machine learning project that predicts the likelihood of liver disease using clinical lab data. It combines multiple ensemble techniques including Random Forest, XGBoost, LightGBM, CatBoost, AdaBoost, Gradient Boosting, and a Stacking Classifier for maximum predictive accuracy.

---

## 📁 Project Structure

```
liver-disease-prediction/
│
├── data/
│   ├── raw/                    # Original dataset (e.g., ILPD dataset)
│   └── processed/              # Cleaned and transformed data
│
├── notebooks/
│   └── liver_disease_eda.ipynb # Exploratory Data Analysis
│
├── src/
│   ├── preprocess.py           # Data cleaning, encoding, scaling
│   ├── features.py             # Feature engineering
│   ├── train.py                # Model training and ensemble logic
│   └── evaluate.py             # Metrics, plots, SHAP analysis
│
├── models/
│   └── *.pkl                   # Saved trained models (joblib)
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── shap_summary.png
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

This project uses the **Indian Liver Patient Dataset (ILPD)** from the UCI Machine Learning Repository.

| Feature | Description |
|---|---|
| Age | Age of the patient |
| Gender | Male / Female |
| Total Bilirubin | Lab value |
| Direct Bilirubin | Lab value |
| Alkaline Phosphotase | Enzyme level |
| Alamine Aminotransferase (ALT) | Liver enzyme |
| Aspartate Aminotransferase (AST) | Liver enzyme |
| Total Proteins | Protein levels |
| Albumin | Protein fraction |
| Albumin/Globulin Ratio | Derived ratio |
| **Target** | 1 = Liver Patient, 2 = No Disease |

> 📌 **Download:** [UCI ILPD Dataset](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset))

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/liver-disease-prediction.git
cd liver-disease-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
imbalanced-learn
shap
matplotlib
seaborn
joblib
optuna
```

---

## 🚀 Usage

### Run the Full Pipeline

```python
python src/train.py
```

### Predict on New Data

```python
from src.train import predict_liver_disease

result = predict_liver_disease({
    "age": 45,
    "gender": "Male",
    "total_bilirubin": 1.2,
    "direct_bilirubin": 0.4,
    "alkaline_phosphotase": 230,
    "alamine_aminotransferase": 35,
    "aspartate_aminotransferase": 40,
    "total_proteins": 7.5,
    "albumin": 3.8,
    "albumin_globulin_ratio": 1.2
})

print(result)  # {'prediction': 'Liver Disease', 'probability': 0.83}
```

---

## 🤖 Models Used

| Model | Type |
|---|---|
| Logistic Regression | Baseline |
| Decision Tree | Baseline |
| Random Forest | Ensemble (Bagging) |
| AdaBoost | Ensemble (Boosting) |
| Gradient Boosting | Ensemble (Boosting) |
| XGBoost | Ensemble (Boosting) |
| LightGBM | Ensemble (Boosting) |
| CatBoost | Ensemble (Boosting) |
| **Voting Classifier** | Soft Voting Ensemble |
| **Stacking Classifier** | Meta-learner Ensemble |

---

## 📈 Evaluation Metrics

The models are evaluated using:

- **Accuracy** — Overall correctness
- **Precision** — Of predicted positives, how many are correct
- **Recall / Sensitivity** — Of actual positives, how many are caught
- **Specificity** — Of actual negatives, how many are correctly identified
- **F1 Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Area under the ROC curve
- **Confusion Matrix** — Visual breakdown of predictions

> ⚠️ In medical ML, **Recall (Sensitivity)** is the most critical metric — missing a liver patient is more dangerous than a false positive.

---

## 🔬 Feature Engineering

The following derived features are engineered for clinical significance:

```python
# De Ritis Ratio — elevated in alcoholic liver disease
df['ast_alt_ratio'] = df['aspartate_aminotransferase'] / df['alamine_aminotransferase']

# Protein balance indicator
df['protein_albumin_ratio'] = df['total_proteins'] / df['albumin']

# Log transform skewed features
for col in ['total_bilirubin', 'direct_bilirubin', 'alkaline_phosphotase',
            'alamine_aminotransferase', 'aspartate_aminotransferase']:
    df[f'log_{col}'] = np.log1p(df[col])
```

---

## ⚖️ Handling Class Imbalance

The dataset is imbalanced (~71% liver patients). We address this using:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

Or via model-level balancing:

```python
RandomForestClassifier(class_weight='balanced', random_state=42)
```

---

## 🧠 SHAP Explainability

SHAP (SHapley Additive exPlanations) is used to explain model predictions:

```python
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

This is essential for clinical trust and regulatory compliance in medical AI.

---

## 🧪 Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
print(f"AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 💾 Saving & Loading Models

```python
import joblib

# Save
joblib.dump(stacking_clf, 'models/stacking_classifier.pkl')

# Load
model = joblib.load('models/stacking_classifier.pkl')
```

---

## 🛠️ Recommended Improvements (Roadmap)

- [ ] Hyperparameter tuning with **Optuna**
- [ ] Add **Precision-Recall curve** plots
- [ ] REST API deployment with **FastAPI**
- [ ] Docker containerization
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Add **calibration plots** for probability reliability
- [ ] Integrate **MLflow** for experiment tracking

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [scikit-learn](https://scikit-learn.org/)
- [SHAP Library](https://shap.readthedocs.io/)
- [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), [CatBoost](https://catboost.ai/)

