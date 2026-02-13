# Predictive Modeling (Customer Churn)

This folder contains the deep ANN churn prediction workflow built in PyTorch.

## Data inputs
Pre-split, preprocessed files from `Data_Preparation`:
- `X_train_raw.csv`, `X_test_raw.csv`
- `y_train.csv`, `y_test.csv`

## How to run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python Predictive_Modeling/deep_ann_torch.py
```

## Model design
- Network: 256→128→64 MLP with BatchNorm + ReLU + Dropout, logistic output head.
- Loss: `BCEWithLogitsLoss` with `pos_weight` (neg/pos ratio) to counter 26.5% churn imbalance.
- Optimizer/Schedule: AdamW + CosineAnnealingLR, early-stop on validation ROC-AUC.
- Thresholding: sweep and auto-select threshold that maximizes validation F2 to emphasize recall.

## Why recall-focused
- Business cost of missing a churner is higher than over-contacting a non-churner, so recall/PR-AUC/F2 are prioritized over plain accuracy.
- F2 weights recall more than precision, aligning with retention goals.

## Why threshold tuning
- The optimal operating point is business-dependent. We search thresholds (0.1–0.9) and pick the one with highest validation F2, then report both default (0.50) and best-threshold metrics.

## Outputs
- Metrics printed to console (train/val/test), classification report, threshold sweep, auto-selected threshold results.
- Figures saved to `Predictive_Modeling/figures/`: ROC, PR, confusion matrix (test).
- Model checkpoint + scaler stats: `Predictive_Modeling/models/deep_ann_torch.pt`.

## Latest run snapshot (device: MPS)
- Default thr=0.50 (test): acc 0.713 | prec 0.475 | recall 0.754 | f1 0.583 | f2 0.675 | roc_auc 0.807 | pr_auc 0.602
- Auto thr=0.38 (val F2 max): acc 0.649 | prec 0.422 | recall 0.864 | f1 0.567 | f2 0.714 | roc_auc 0.807 | pr_auc 0.602
- Threshold sweep shows recall up to ~0.93 at thr=0.20 with expected drop in precision.

## Quick takeaways
- Recall can be pushed above 0.86 via threshold tuning while keeping ROC-AUC ~0.81 and PR-AUC ~0.60.
- Use lower thresholds when reducing missed churn is paramount; raise thresholds if outreach cost is a constraint.
