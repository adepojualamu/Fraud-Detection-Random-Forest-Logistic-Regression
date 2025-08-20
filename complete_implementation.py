# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import shap
import time
import os

# Set random seed
np.random.seed(42)

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("../data/creditcard.csv")

# Feature engineering
df['Time'] = df['Time'] / 3600  # Convert seconds to hours

# Normalize 'Amount'
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# Split features & target
X = df.drop('Class', axis=1)
y = df['Class']

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Class Imbalance Handling
smote = SMOTE(random_state=42)
undersampler = RandomUnderSampler(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Save model evaluation reports
results_file = open("outputs/model_results.txt", "w")

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)

    start_inf = time.time()
    model.predict(X_test[:1000])
    inf_time = (time.time() - start_inf) / 1000

    # Write to file
    results_file.write(f"\n=== {model_name} ===\n")
    results_file.write(report)
    results_file.write(f"\nAUC-ROC: {auc_roc:.4f}\n")
    results_file.write(f"AUC-PR: {auc_pr:.4f}\n")
    results_file.write(f"Training Time: {train_time:.2f}s\n")
    results_file.write(f"Inference Time: {inf_time:.5f}ms\n\n")

    return model

# Initialize models
lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(random_state=42)

# Baseline Models
lr_baseline = evaluate_model(lr, X_train, y_train, X_test, y_test, "LR Baseline")
rf_baseline = evaluate_model(rf, X_train, y_train, X_test, y_test, "RF Baseline")

# SMOTE Models
lr_smote = evaluate_model(LogisticRegression(max_iter=1000, random_state=42), X_train_smote, y_train_smote, X_test, y_test, "LR + SMOTE")
rf_smote = evaluate_model(RandomForestClassifier(random_state=42), X_train_smote, y_train_smote, X_test, y_test, "RF + SMOTE")

# Class Weighted Models
lr_weighted = evaluate_model(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42), X_train, y_train, X_test, y_test, "LR + Class Weighting")
rf_weighted = evaluate_model(RandomForestClassifier(class_weight='balanced_subsample', random_state=42), X_train, y_train, X_test, y_test, "RF + Class Weighting")

# Hyperparameter Tuning
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
grid_lr = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42), param_grid_lr, scoring='average_precision', cv=StratifiedKFold(5), n_jobs=-1)
grid_lr.fit(X_train, y_train)

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}
random_rf = RandomizedSearchCV(RandomForestClassifier(class_weight='balanced_subsample', random_state=42), param_grid_rf, n_iter=10, scoring='average_precision', cv=StratifiedKFold(5), n_jobs=-1)
random_rf.fit(X_train, y_train)

results_file.write(f"\nBest Logistic Regression Parameters: {grid_lr.best_params_}\n")
results_file.write(f"Best Random Forest Parameters: {random_rf.best_params_}\n")

# Interpretability - Logistic Regression Coefficients
lr_coef = pd.DataFrame({
    'Feature': X_train.columns,
    'Weight': grid_lr.best_estimator_.coef_[0]
}).sort_values('Weight', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Weight', y='Feature', data=lr_coef)
plt.title("Logistic Regression Feature Weights")
plt.tight_layout()
plt.savefig("outputs/lr_feature_weights.png")

# Interpretability - SHAP for Random Forest
explainer = shap.TreeExplainer(random_rf.best_estimator_)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values[1], X_test, plot_type="dot", show=False)
plt.title("SHAP Feature Importance (Fraud Class)")
plt.tight_layout()
plt.savefig("outputs/shap_summary.png")

# ROC Curves
plt.figure()
RocCurveDisplay.from_estimator(grid_lr.best_estimator_, X_test, y_test, name="Tuned LR")
RocCurveDisplay.from_estimator(random_rf.best_estimator_, X_test, y_test, name="Tuned RF")
plt.title("ROC Curve Comparison")
plt.savefig("outputs/roc_curve.png")

# Precision-Recall Curves
plt.figure()
PrecisionRecallDisplay.from_estimator(grid_lr.best_estimator_, X_test, y_test, name="Tuned LR")
PrecisionRecallDisplay.from_estimator(random_rf.best_estimator_, X_test, y_test, name="Tuned RF")
plt.title("Precision-Recall Curve Comparison")
plt.savefig("outputs/pr_curve.png")

results_file.close()