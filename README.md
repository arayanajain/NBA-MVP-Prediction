# NBA MVP Prediction

Predicting NBA MVP vote share with a walk-forward regression model that has better accuracy than the Stanford CS229 baseline.

## What this is

The Stanford CS229 project tried to predict MVP winners with Naive Bayes (~67% accuracy) and logistic regression (~50% seasonal accuracy) using only the top-3 finalists and a fixed train/test split on 4 seasons.

This notebook does the same task with a stricter setup:

- Continuous **vote-share regression** instead of binary classification
- 32 features (box score + advanced + team context) instead of 10 basic stats
- Expanding-window walk-forward validation across **36 seasons** (1990–2025)
- Baselines: prior-season share and Win Shares-only Ridge

## Results

Held-out season winner accuracy (did the model pick the actual MVP?):

| Model                      | Accuracy   |
|---------------------------|------------|
| Random Forest             | **69.4%** (25/36) |
| Gradient Boosting         | 66.7% (24/36) |
| Ridge Regression          | 63.9% (23/36) |
| Win Shares-only Ridge     | 58.3% (21/36) |
| Prior-season share        | 27.8% (10/36) |

Paper numbers for reference: ~50% (logreg) / ~67% (Naive Bayes) on only 4 seasons.

## How to run

Open `NBA_MVP_Prediction.ipynb` in Colab or Jupyter. The data is pulled automatically from a public CSV of historical MVP vote recipients.

```bash
pip install pandas numpy scikit-learn matplotlib scipy
