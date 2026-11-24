"""Credit card fraud detection: Logistic Regression vs Random Forest.

Run from the repository root:

    python src/fraud_detection.py                 # full run on data/creditcard.csv
    python src/fraud_detection.py --sample 0.1    # quick smoke test on 10% of the data

Design notes
------------
* Every model is an ``imblearn`` Pipeline (preprocessing -> optional resampler -> classifier).
  Scaling and resampling are therefore fitted on training folds only, which prevents the
  train/test leakage you get from scaling or oversampling the full dataset up front.
* Models are compared on PR-AUC (average precision). With ~0.17% fraud, ROC-AUC looks
  excellent even for weak models, so it is reported but not used for selection.
* The decision threshold is tuned on out-of-fold training predictions, never on the test set.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import RobustScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
SCALED_COLUMNS = ["Time", "Amount"]  # V1-V28 are already PCA outputs
TARGET = "Class"


# --------------------------------------------------------------------------- data
def load_data(path: Path, sample: float | None, seed: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset first (see README, 'Get the data')."
        )
    df = pd.read_csv(path)
    missing = {TARGET, *SCALED_COLUMNS} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")

    # The Kaggle file contains ~1k exact duplicate rows; left in, copies land on both
    # sides of the split and make the test set look easier than it is.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {before:,} rows, dropped {before - len(df):,} duplicates.")

    df["Time"] = df["Time"] / 3600  # seconds -> hours since first transaction

    if sample is not None:
        df, _ = train_test_split(df, train_size=sample, stratify=df[TARGET], random_state=seed)
        df = df.reset_index(drop=True)
        print(f"Using a stratified {sample:.0%} sample: {len(df):,} rows.")
    return df


def build_preprocessor() -> ColumnTransformer:
    # RobustScaler: Amount is heavy-tailed, and outliers distort mean/std scaling.
    return ColumnTransformer(
        [("scale", RobustScaler(), SCALED_COLUMNS)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def make_pipeline(model, sampler=None) -> Pipeline:
    steps = [("prep", build_preprocessor())]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", model))
    return Pipeline(steps)


# ------------------------------------------------------------------------ metrics
def best_f1_threshold(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    p, r = precision[:-1], recall[:-1]
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def evaluate(name, pipe, X_test, y_test, threshold=0.5, train_time=None, reports=None) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )

    batch = X_test.iloc[:1000]
    start = time.perf_counter()
    pipe.predict_proba(batch)
    infer_ms = (time.perf_counter() - start) / len(batch) * 1000  # ms per transaction

    if reports is not None:
        reports.append(
            f"=== {name} (threshold={threshold:.3f}) ===\n"
            + classification_report(y_test, pred, target_names=["Legit", "Fraud"], zero_division=0)
        )
    return {
        "Model": name,
        "Threshold": round(threshold, 3),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC": average_precision_score(y_test, proba),
        "Train (s)": train_time,
        "Inference (ms/txn)": infer_ms,
    }


# ------------------------------------------------------------------- experiments
def run_baselines(X_train, y_train, X_test, y_test, seed, results, reports):
    """LR and RF under four imbalance strategies: none, SMOTE, under-sampling, class weights."""

    def lr(**kw):
        return LogisticRegression(max_iter=1000, random_state=seed, **kw)

    def rf(**kw):
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1, **kw)

    experiments = {
        "LR baseline": make_pipeline(lr()),
        "RF baseline": make_pipeline(rf()),
        "LR + SMOTE": make_pipeline(lr(), SMOTE(random_state=seed)),
        "RF + SMOTE": make_pipeline(rf(), SMOTE(random_state=seed)),
        "LR + under-sampling": make_pipeline(lr(), RandomUnderSampler(random_state=seed)),
        "RF + under-sampling": make_pipeline(rf(), RandomUnderSampler(random_state=seed)),
        "LR + class weights": make_pipeline(lr(class_weight="balanced")),
        "RF + class weights": make_pipeline(rf(class_weight="balanced_subsample")),
    }
    for name, pipe in experiments.items():
        print(f"Fitting {name} ...")
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - start
        results.append(evaluate(name, pipe, X_test, y_test, train_time=train_time, reports=reports))


def tune_models(X_train, y_train, cv, seed, n_jobs):
    """Tune on PR-AUC. Class weighting is used for both (cheapest strategy that worked well)."""
    lr_search = GridSearchCV(
        make_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        ),
        # L2 only: with ~30 dense PCA features L1 sparsity adds little, and the `penalty`
        # argument is deprecated in recent scikit-learn.
        param_grid={"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]},
        scoring="average_precision",
        cv=cv,
        n_jobs=n_jobs,
    )
    rf_search = RandomizedSearchCV(
        make_pipeline(
            RandomForestClassifier(class_weight="balanced_subsample", random_state=seed, n_jobs=1)
        ),
        param_distributions={
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [5, 10, 20, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        },
        n_iter=10,
        scoring="average_precision",
        cv=cv,
        n_jobs=n_jobs,
        random_state=seed,
    )
    tuned = {}
    for name, search in {"Tuned LR": lr_search, "Tuned RF": rf_search}.items():
        print(f"Tuning {name} ...")
        start = time.perf_counter()
        search.fit(X_train, y_train)
        tuned[name] = (search, time.perf_counter() - start)
        print(f"  best CV PR-AUC={search.best_score_:.4f}  params={search.best_params_}")
    return tuned


# ----------------------------------------------------------------------- plotting
def plot_lr_weights(pipe, out_path: Path, top_n: int = 15):
    names = pipe.named_steps["prep"].get_feature_names_out()
    weights = pd.DataFrame({"Feature": names, "Weight": pipe.named_steps["clf"].coef_[0]})
    top = weights.reindex(weights["Weight"].abs().sort_values(ascending=False).index).head(top_n)
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Weight", y="Feature", data=top, color="steelblue")
    plt.title(f"Logistic Regression: top {top_n} coefficients (by magnitude)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_shap(pipe, X_test, out_path: Path, seed: int, n_rows: int = 2000):
    """SHAP on a subsample: exact TreeSHAP on all ~57k test rows is needlessly slow."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    sample = X_test.sample(min(n_rows, len(X_test)), random_state=seed)
    transformed = pd.DataFrame(pipe.named_steps["prep"].transform(sample), columns=names)
    values = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(
        transformed, check_additivity=False
    )
    # Older SHAP returns a list per class; newer returns (rows, features, classes).
    if isinstance(values, list):
        values = values[1]
    elif getattr(values, "ndim", 2) == 3:
        values = values[:, :, 1]
    plt.figure()
    shap.summary_plot(values, transformed, plot_type="dot", show=False)
    plt.title("SHAP feature impact (fraud class)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_curves(models: dict, X_test, y_test, output_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.set_title("ROC curve comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.set_title("Precision-Recall curve comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curve.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- output
def to_markdown(df: pd.DataFrame) -> str:
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "creditcard.csv")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--sample", type=float, default=None,
                   help="Use a stratified fraction of the data (0-1) for quick runs.")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.sample is not None and not 0 < args.sample < 1:
        p.error("--sample must be between 0 and 1 (exclusive)")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data, args.sample, args.seed)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )
    print(f"Train: {len(X_train):,} rows ({y_train.sum()} fraud) | "
          f"Test: {len(X_test):,} rows ({y_test.sum()} fraud)")

    results, reports = [], []
    run_baselines(X_train, y_train, X_test, y_test, args.seed, results, reports)

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    tuned = tune_models(X_train, y_train, cv, args.seed, args.n_jobs)

    best_params = []
    best_models = {}
    for name, (search, tune_time) in tuned.items():
        model = search.best_estimator_
        best_models[name] = model
        best_params.append(f"{name}: {search.best_params_}")
        results.append(evaluate(name, model, X_test, y_test, train_time=tune_time, reports=reports))

        # Threshold chosen on out-of-fold TRAIN predictions so the test set stays untouched.
        oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba",
                                n_jobs=args.n_jobs)[:, 1]
        threshold = best_f1_threshold(y_train, oof)
        results.append(evaluate(f"{name} (tuned threshold)", model, X_test, y_test,
                                threshold=threshold, train_time=tune_time, reports=reports))

    summary = pd.DataFrame(results)
    summary.to_csv(args.output_dir / "results.csv", index=False)
    (args.output_dir / "results.md").write_text(to_markdown(summary) + "\n")
    (args.output_dir / "model_results.txt").write_text(
        "\n".join(reports) + "\nBest parameters\n" + "\n".join(best_params) + "\n"
    )

    plot_lr_weights(best_models["Tuned LR"], args.output_dir / "lr_feature_weights.png")
    plot_shap(best_models["Tuned RF"], X_test, args.output_dir / "shap_summary.png", args.seed)
    plot_curves(best_models, X_test, y_test, args.output_dir)

    print("\n" + to_markdown(summary))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
