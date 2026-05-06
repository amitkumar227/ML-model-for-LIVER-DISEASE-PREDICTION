import sys
!{sys.executable} -m pip install xgboost lightgbm catboost imbalanced-learn shap optuna
!python liver_disease_prediction.py


# ── 0. Install Dependencies ──────────────────────────────────
pip install xgboost lightgbm catboost imbalanced-learn shap optuna

# ── 1. Imports ───────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import optuna
import joblib
from io import StringIO

from sklearn.model_selection import(
train_test_split, StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import(
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, VotingClassifier, StackingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay, PrecisionRecallDisplay
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 2. Load Dataset ──────────────────────────────────────────
COLUMNS = [
    "age", "gender", "total_bilirubin", "direct_bilirubin",
    "alkaline_phosphotase", "alamine_aminotransferase",
    "aspartate_aminotransferase", "total_proteins",
    "albumin", "albumin_globulin_ratio", "target"
]

def load_data(path: str = None) -> pd.DataFrame:
    """ Load the ILPD dataset.
    If path is None, downloads from UCI repository.
    Target: 1 = Liver Patient, 2 = No Disease → remapped to 1/0.  """
    if path:
        df = pd.read_csv(path, header=None, names=COLUMNS)
    else:
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
        )
        df = pd.read_csv(url, header=None, names=COLUMNS)

    # Remap target: 1 → 1 (disease), 2 → 0 (healthy)
    df["target"] = (df["target"] == 1).astype(int)
    return df


# ── 3. Exploratory Summary ───────────────────────────────────
def summarize(df: pd.DataFrame) -> None:
    print("=" * 60)
    print(f"Shape         : {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nClass balance :\n{df['target'].value_counts(normalize=True).round(3)}")
    print("=" * 60)


# ── 4. Feature Engineering ───────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clinically meaningful derived features.
    """
    df = df.copy()

    # 4a. Encode gender
    df["gender"] = LabelEncoder().fit_transform(df["gender"].astype(str))

    # 4b. Age bins (clinical grouping)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 40, 60, 120],
        labels=[0, 1, 2, 3]          # pediatric / adult / middle / elderly
    ).astype(int)

    # 4c. De Ritis ratio — elevated in alcoholic & acute liver disease
    df["ast_alt_ratio"] = (
        df["aspartate_aminotransferase"] /
        (df["alamine_aminotransferase"] + 1e-6)
    )

    # 4d. Protein / Albumin ratio — reflects synthetic liver function
    df["protein_albumin_ratio"] = (
        df["total_proteins"] / (df["albumin"] + 1e-6)
    )

    # 4e. Bilirubin ratio (direct / total)
    df["bilirubin_ratio"] = (
        df["direct_bilirubin"] / (df["total_bilirubin"] + 1e-6)
    )

    # 4f. Log-transform right-skewed lab values
    skewed_cols = [
        "total_bilirubin", "direct_bilirubin",
        "alkaline_phosphotase", "alamine_aminotransferase",
        "aspartate_aminotransferase"
    ]
    for col in skewed_cols:
        df[f"log_{col}"] = np.log1p(df[col])

    return df


# ── 5. Preprocessing Pipeline ────────────────────────────────
def build_preprocessor(feature_cols: list) -> ColumnTransformer:
    """
    Impute → Scale. Prevents data leakage when used inside CV.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, feature_cols)
    ])


# ── 6. Base Models ───────────────────────────────────────────
def get_base_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_depth=8, random_state=42, n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=150, learning_rate=0.05, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            scale_pos_weight=2.4,          # handles imbalance
            use_label_encoder=False,
            eval_metric="logloss", random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            class_weight="balanced", random_state=42,
            verbose=-1, n_jobs=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=4,
            auto_class_weights="Balanced",
            random_seed=42, verbose=0
        ),
    }


# ── 7. Optuna Hyperparameter Tuning (XGBoost) ───────────────
def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                 n_trials: int = 30) -> XGBClassifier:
    """
    Tune XGBoost with Optuna. Returns best model.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
            "max_depth":       trial.suggest_int("max_depth", 3, 8),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-5, 10, log=True),
            "scale_pos_weight":trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = XGBClassifier(
        **study.best_params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    best.fit(X_train, y_train)
    print(f"\n✅ Best XGBoost AUC (CV): {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")
    return best


# ── 8. Cross-Validated Evaluation ────────────────────────────
def cv_evaluate(name: str, model, X: np.ndarray,
                y: np.ndarray, cv) -> dict:
    """
    Return mean ± std for multiple metrics via StratifiedKFold.
    """
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    row = {"Model": name}
    for metric in scoring:
        key = f"test_{metric}"
        row[metric.capitalize()] = f"{results[key].mean():.4f} ± {results[key].std():.4f}"
    return row


# ── 9. Ensemble: Soft Voting + Stacking ──────────────────────
def build_ensembles(base_models: dict,
                    tuned_xgb: XGBClassifier) -> dict:
    """
    Soft VotingClassifier and StackingClassifier
    with a calibrated meta-learner.
    """
    # Replace XGBoost in base with tuned version
    estimators = [
        (name.replace(" ", "_"), model)
        for name, model in base_models.items()
        if name != "XGBoost"
    ]
    estimators.append(("XGBoost_tuned", tuned_xgb))

    voting_clf = VotingClassifier(
        estimators=estimators,
        voting="soft",        # ✅ soft voting for probability averaging
        n_jobs=-1
    )

    # Stacking: tree-based learners feed a calibrated LR meta-learner
    stack_estimators = [
        ("rf",   base_models["Random Forest"]),
        ("xgb",  tuned_xgb),
        ("lgbm", base_models["LightGBM"]),
        ("cat",  base_models["CatBoost"]),
        ("gb",   base_models["Gradient Boosting"]),
    ]
    meta_learner = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        cv=5, method="isotonic"
    )
    stacking_clf = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=meta_learner,
        cv=5,                 # ✅ cross-validated predictions for meta-learner
        stack_method="predict_proba",
        n_jobs=-1
    )
    return {"Soft Voting": voting_clf, "Stacking": stacking_clf}


# ── 10. Metrics & Plots ──────────────────────────────────────
def evaluate_on_test(name: str, model,
                     X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model":       name,
        "Accuracy":    accuracy_score(y_test, y_pred),
        "Precision":   precision_score(y_test, y_pred, zero_division=0),
        "Recall":      recall_score(y_test, y_pred),
        "Specificity": recall_score(y_test, y_pred, pos_label=0),
        "F1":          f1_score(y_test, y_pred),
        "ROC-AUC":     roc_auc_score(y_test, y_prob),
    }


def plot_roc_pr(models: dict, X_test: np.ndarray,
                y_test: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test,
                                       name=name, ax=axes[0])
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test,
                                              name=name, ax=axes[1])
    axes[0].set_title("ROC Curves", fontsize=13, fontweight="bold")
    axes[1].set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("outputs/roc_pr_curves.png", dpi=150)
    plt.show()


def plot_confusion_matrices(models: dict, X_test: np.ndarray,
                             y_test: np.ndarray) -> None:
    n = len(models)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    axes = axes.flatten()
    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Healthy", "Disease"],
                    yticklabels=["Healthy", "Disease"], ax=ax)
        ax.set_title(name, fontsize=9, fontweight="bold")
    for ax in axes[len(models):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrices.png", dpi=150)
    plt.show()


def plot_calibration(models: dict, X_test: np.ndarray,
                     y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    for name, model in models.items():
        prob_pos = model.predict_proba(X_test)[:, 1]
        fraction_pos, mean_pred = calibration_curve(
            y_test, prob_pos, n_bins=10
        )
        ax.plot(mean_pred, fraction_pos, marker="o", label=name)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Plots", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/calibration.png", dpi=150)
    plt.show()


# ── 11. SHAP Explainability ──────────────────────────────────
def shap_analysis(model, X_train: np.ndarray,
                  feature_names: list, model_name: str = "XGBoost") -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    plt.figure()
    shap.summary_plot(shap_values, X_train,
                      feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary — {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"outputs/shap_{model_name.replace(' ', '_')}.png", dpi=150)
    plt.show()

    plt.figure()
    shap.summary_plot(shap_values, X_train,
                      feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"outputs/shap_bar_{model_name.replace(' ', '_')}.png", dpi=150)
    plt.show()


# ── 12. Main Pipeline ─────────────────────────────────────────
def predict_liver_disease(patient: dict = None) -> dict:
    """
    End-to-end pipeline. If `patient` dict is passed,
    also returns prediction for that single patient.
    """
    import os
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models",  exist_ok=True)

    # ── 12.1 Load & Summarize ────────────────────────────────
    print("\n📦 Loading data...")
    df = load_data()           # Replace with load_data("your_file.csv") locally
    summarize(df)

    # ── 12.2 Feature Engineering ─────────────────────────────
    print("🔧 Engineering features...")
    df = engineer_features(df)

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    # ── 12.3 Train/Test Split (stratified) ───────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 12.4 Preprocessing (fit on train only → no leakage) ──
    print("⚙️  Preprocessing...")
    preprocessor = build_preprocessor(list(range(X.shape[1])))
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)

    # ── 12.5 SMOTE (only on training data) ───────────────────
    print("⚖️  Applying SMOTE to balance training set...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"   Before SMOTE: {np.bincount(y_train)}")
    print(f"   After  SMOTE: {np.bincount(y_train_bal)}")

    # ── 12.6 Stratified KFold ─────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── 12.7 Base Model CV Evaluation ────────────────────────
    print("\n🔍 Cross-validating base models...")
    base_models = get_base_models()
    cv_rows = []
    for name, model in base_models.items():
        print(f"   {name}...", end=" ")
        row = cv_evaluate(name, model, X_train_bal, y_train_bal, skf)
        cv_rows.append(row)
        print("done")

    cv_df = pd.DataFrame(cv_rows).set_index("Model")
    print("\n📊 Cross-Validation Results (mean ± std):")
    print(cv_df.to_string())

    # ── 12.8 Optuna Tuning (XGBoost) ─────────────────────────
    print("\n🔎 Tuning XGBoost with Optuna (30 trials)...")
    tuned_xgb = tune_xgboost(X_train_bal, y_train_bal, n_trials=30)

    # ── 12.9 Train All Base Models ────────────────────────────
    print("\n🏋️  Training all base models on balanced data...")
    for name, model in base_models.items():
        model.fit(X_train_bal, y_train_bal)

    # ── 12.10 Ensembles ───────────────────────────────────────
    print("🤝 Building ensemble models...")
    ensembles = build_ensembles(base_models, tuned_xgb)
    for name, model in ensembles.items():
        print(f"   Fitting {name}...")
        model.fit(X_train_bal, y_train_bal)

    # ── 12.11 Test Set Evaluation ─────────────────────────────
    all_models = {**base_models, "XGBoost_Tuned": tuned_xgb, **ensembles}
    print("\n📈 Evaluating on test set...")
    test_rows = [evaluate_on_test(n, m, X_test, y_test)
                 for n, m in all_models.items()]
    test_df = (
        pd.DataFrame(test_rows)
        .set_index("Model")
        .sort_values("ROC-AUC", ascending=False)
    )
    print("\n🏆 Test Set Results (sorted by ROC-AUC):")
    print(test_df.round(4).to_string())

    # ── 12.12 Plots ───────────────────────────────────────────
    print("\n📊 Generating plots...")
    top_models = {
        "XGBoost (Tuned)": tuned_xgb,
        "Soft Voting":     ensembles["Soft Voting"],
        "Stacking":        ensembles["Stacking"],
        "Random Forest":   base_models["Random Forest"],
        "LightGBM":        base_models["LightGBM"],
    }
    plot_roc_pr(top_models, X_test, y_test)
    plot_confusion_matrices(top_models, X_test, y_test)
    plot_calibration(top_models, X_test, y_test)

    # ── 12.13 SHAP ────────────────────────────────────────────
    print("🔬 Running SHAP analysis on XGBoost...")
    shap_analysis(tuned_xgb, X_train_bal, feature_cols, "XGBoost_Tuned")

    # ── 12.14 Best Model Classification Report ────────────────
    best_name = test_df["ROC-AUC"].idxmax()
    best_model = all_models[best_name]
    print(f"\n🥇 Best Model: {best_name}")
    print(classification_report(
        y_test, best_model.predict(X_test),
        target_names=["Healthy", "Liver Disease"]
    ))

    # ── 12.15 Save Models & Preprocessor ─────────────────────
    print("💾 Saving models...")
    joblib.dump(best_model,   f"models/best_model_{best_name}.pkl")
    joblib.dump(tuned_xgb,    "models/xgboost_tuned.pkl")
    joblib.dump(ensembles["Stacking"], "models/stacking_clf.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    # ── 12.16 Inference on New Patient ───────────────────────
    if patient:
        print("\n🩺 Predicting for new patient...")
        pat_df = pd.DataFrame([patient])
        pat_df.columns = COLUMNS[:-1]           # drop 'target'
        pat_eng = engineer_features(
            pd.concat([pat_df, pat_df])          # dummy row to allow cuts
        ).iloc[[0]][feature_cols]
        pat_scaled = preprocessor.transform(pat_eng.values)
        prob = best_model.predict_proba(pat_scaled)[0][1]
        pred = int(prob >= 0.5)
        result = {
            "prediction":  "Liver Disease" if pred else "Healthy",
            "probability": round(prob, 4),
            "model_used":  best_name
        }
        print(f"   Result: {result}")
        return result

    print("\n✅ Pipeline complete. Outputs saved to /outputs and /models.")
    return {"status": "complete", "best_model": best_name}


# ── 13. Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    # Example single-patient prediction
    sample_patient = {
        "age": 52, "gender": "Male",
        "total_bilirubin": 1.8, "direct_bilirubin": 0.6,
        "alkaline_phosphotase": 260, "alamine_aminotransferase": 48,
        "aspartate_aminotransferase": 55, "total_proteins": 6.8,
        "albumin": 3.2, "albumin_globulin_ratio": 0.9
    }
    result = predict_liver_disease(patient=sample_patient)
    print(result)
