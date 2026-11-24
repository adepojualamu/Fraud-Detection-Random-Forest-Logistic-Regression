# Credit Card Fraud Detection: Random Forest vs Logistic Regression

Detects fraudulent credit card transactions with Logistic Regression and Random Forest, using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 frauds, about 0.17%).

The project compares four ways of handling the extreme class imbalance, tunes both models, and explains their predictions.

## Features

- **Imbalance strategies compared:** none, SMOTE, random under-sampling, class weighting
- **Two model families:** Logistic Regression and Random Forest, each as baseline and tuned
- **Hyperparameter tuning:** `GridSearchCV` (LR) and `RandomizedSearchCV` (RF), scored on PR-AUC
- **Threshold tuning:** decision threshold chosen on out-of-fold training predictions, never on the test set
- **Evaluation:** precision, recall, F1, ROC-AUC, PR-AUC, training time and per-transaction inference time
- **Interpretability:** Logistic Regression coefficients and SHAP values for the Random Forest

## Project layout

```
.
├── src/fraud_detection.py   # full pipeline (run this)
├── data/                    # put creditcard.csv here (not tracked by git)
├── outputs/                 # generated: metrics tables, plots, reports
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/adepojualamu/Fraud-Detection-Random-Forest-Logistic-Regression.git
cd Fraud-Detection-Random-Forest-Logistic-Regression
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

1. Create a Kaggle account and [set up an API token](https://www.kaggle.com/docs/api).
2. Download and unpack the dataset into `data/`:

```bash
pip install kaggle
mkdir -p data
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p data
```

You should end up with `data/creditcard.csv`. You can also download it manually from the dataset page and place it there.

## Usage

```bash
python src/fraud_detection.py                  # full run (the tuning step takes a while)
python src/fraud_detection.py --sample 0.1     # quick check on a 10% stratified sample
```

| Option | Default | Meaning |
|---|---|---|
| `--data` | `data/creditcard.csv` | Path to the dataset |
| `--output-dir` | `outputs/` | Where results are written |
| `--sample` | off | Stratified fraction (0 to 1) of the data, for fast runs |
| `--cv-folds` | `5` | Cross-validation folds for tuning |
| `--n-jobs` | `-1` | Parallel jobs for the searches |
| `--seed` | `42` | Random seed |

### Outputs

| File | Contents |
|---|---|
| `results.csv` / `results.md` | Metrics for every model (the `.md` table is ready to paste below) |
| `model_results.txt` | Per-model classification reports and best hyperparameters |
| `roc_curve.png`, `pr_curve.png` | Tuned LR vs tuned RF |
| `lr_feature_weights.png` | Top Logistic Regression coefficients |
| `shap_summary.png` | SHAP feature impact for the Random Forest |

## Results

> Run the script on the full dataset and paste the contents of `outputs/results.md` here.
> The numbers depend on the dataset and hardware, so none are quoted until they come from a real run.

<!-- paste outputs/results.md here -->

## Design decisions

- **Leak-free preprocessing.** Scaling and resampling live inside an `imblearn` `Pipeline`, so they are fitted on training data only, including inside each cross-validation fold. Scaling or oversampling the full dataset first leaks information from the test set or validation folds and inflates scores.
- **PR-AUC over ROC-AUC.** With about 0.17% fraud, ROC-AUC looks excellent even for weak models because true negatives dominate. PR-AUC (average precision) reflects how well frauds are ranked, so it is the tuning metric. ROC-AUC is still reported.
- **Threshold tuning.** The default 0.5 cutoff is rarely right for imbalanced data. The F1-optimal threshold is found on out-of-fold training predictions and then applied once to the test set. Real deployments should pick the threshold from the relative cost of a missed fraud versus a false alarm.
- **Duplicates removed.** The Kaggle file contains about 1,000 exact duplicate rows. Left in, copies land in both train and test and flatter the results.
- **RobustScaler for `Time` and `Amount`.** `Amount` is heavy-tailed, and outliers distort mean/std scaling. `V1` to `V28` are already PCA outputs and are left as they are. `Time` is converted from seconds to hours.
- **Resampling only on training data.** The test set keeps the real class distribution.

## Limitations

- The features are anonymised PCA components, so SHAP and coefficients say which components matter, not which real-world attributes.
- The data covers two days of European card transactions from 2013. Fraud patterns drift, so a production system would need monitoring and retraining.
- This is a single random split. For a rigorous comparison, repeat with several seeds or use a time-based split.

## Disclaimer

This project is for educational purposes only. Do not use it for real-world financial decisions without proper validation.
