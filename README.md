# NBA MVP Prediction

A machine learning project that predicts NBA Most Valuable Player (MVP) vote share using historical player statistics and evaluates models through expanding-window walk-forward testing.

## 1. Project Overview

The NBA MVP award is decided through media voting. This project uses regression models to estimate each player's MVP vote share and ranks players within each season. The player with the highest predicted share is selected as the predicted MVP.

**Project objectives:**
- Predict MVP vote share using historical NBA statistics.
- Compare multiple machine learning models.
- Evaluate predictions chronologically across 36 held-out seasons.
- Compare performance against simple historical baselines.

## 2. Dataset

**Dataset:** Historical NBA MVP voting statistics

**Source repository:** https://github.com/DavidYoo912/nba_mvp

**Direct CSV download:** https://raw.githubusercontent.com/DavidYoo912/nba_mvp/main/data/master_table.csv

The dataset was scraped from Basketball-Reference and contains historical MVP voting records. Each row represents a player-season.

The dataset includes:
- Basic statistics: points, rebounds, assists, steals and blocks.
- Shooting statistics and percentages.
- Advanced metrics such as Player Efficiency Rating (PER), Win Shares, Box Plus/Minus (BPM) and Value Over Replacement Player (VORP).
- Team context, including wins and win percentage.
- MVP vote share, which is the prediction target.

After cleaning, the dataset contains 724 player-season records from 1980–2025, with 45 original columns.

**Important limitation:** The dataset includes players who received MVP votes, not every NBA player. Therefore, the model selects among recorded MVP vote recipients rather than all eligible players.

## 3. Methodology

The target variable is `Share`, representing the fraction of possible MVP voting points received by a player.

The model predicts vote share for each player and ranks players within each season.

The project uses 32 numerical predictors covering box-score statistics, advanced metrics and team performance. Voting outcomes and identifiers are excluded from the predictors, and missing values are handled through model pipelines.

### Models and baselines

- Ridge Regression
- Random Forest Regressor
- HistGradientBoosting Regressor
- Win Shares-only Ridge baseline
- Historical vote-share baseline

### Evaluation strategy

The project uses **expanding-window walk-forward validation**. For each test season, the model trains only on earlier seasons. The evaluation covers 36 held-out seasons, from 1990 through 2025.

The primary metric is **season-winner accuracy (Top-1 accuracy)**: whether the player with the highest predicted vote share matches the actual MVP.

Vote-share MAE and RMSE are also evaluated.

## 4. Results

The executed notebook reports the following results:

| Model | Correct picks | Accuracy |
|---|---:|---:|
| Random Forest | 25/36 | 69.44% |
| HistGradientBoosting | 24/36 | 66.67% |
| Ridge Regression | 23/36 | 63.89% |
| Win Shares-only Ridge | 21/36 | 58.33% |
| Historical vote-share baseline | 10/36 | 27.78% |

Random Forest achieves the highest season-winner accuracy in this experiment. HistGradientBoosting has lower vote-share prediction error, so the best model depends on the evaluation metric.

## 5. Stanford CS229 Comparison

The project also discusses reported results from a Stanford CS229 NBA MVP prediction project. These figures are included as contextual references, not as a direct head-to-head comparison.

The candidate pools, prediction targets and evaluation protocols differ. Therefore, the numerical accuracy figures alone do not establish that one approach is definitively better than the other.

## 6. How to Run

1. Open `NBA_MVP_Prediction.ipynb` in Google Colab or Jupyter Notebook. Replace this filename with the exact notebook filename in the repository if it differs.
2. Run the notebook cells in order.
3. The notebook downloads the dataset from the public CSV source.
4. Review the cleaning, model evaluation, accuracy table and prediction audit outputs.

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib scipy
```

## 7. Limitations and Future Work

- The dataset contains MVP vote recipients, not all eligible NBA players.
- Completed-season statistics are used, so this is a retrospective reconstruction rather than a preseason forecast.
- Historical accuracy does not guarantee future performance.
- A fair comparison with Stanford would require evaluating both approaches on the same seasons and candidate pool.

Possible future improvements include testing additional features, evaluating alternative candidate-selection methods and conducting prospective predictions before MVP voting concludes.
