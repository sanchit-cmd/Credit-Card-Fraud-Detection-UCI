# Default of Credit Card Clients

An exploratory machine-learning project using the UCI Default of Credit Card
Clients dataset. The notebook predicts whether a customer will default on a
credit-card payment next month and examines payment behavior associated with
default risk.

## Project contents

- [`main.ipynb`](./main.ipynb) — data exploration, model training and evaluation,
  visualizations, and risk-driver analysis.
- [`UCI_Credit_Card.csv`](./UCI_Credit_Card.csv) — dataset used by the notebook
  (30,000 customer records).
- [`requirements.txt`](./requirements.txt) — pinned Python dependencies.

## Getting started

Use Python with `pip` and Jupyter Notebook installed. From the project directory,
install the listed dependencies and start Jupyter:

```bash
python -m pip install -r requirements.txt
python -m notebook
```

Open `main.ipynb` and run the cells from top to bottom. Keep
`UCI_Credit_Card.csv` in the project directory; the notebook loads it using a
relative path.

## Notebook workflow

The analysis:

1. Loads the CSV and checks its shape, columns, data types, missing values, and
   duplicate rows.
2. Explores the target, `default.payment.next.month`, and its class imbalance.
3. Removes the customer ID and target from the predictors, then creates a
   stratified 75/25 train/test split with `random_state=42`.
4. Scales features for Logistic Regression and trains two classifiers:
   Logistic Regression and XGBoost.
5. Evaluates accuracy, classification reports, confusion matrices, and ROC AUC,
   and compares the models.
6. Examines feature importance and trends in payment delays, bill balances,
   payment amounts, age, and sex.
7. Demonstrates predicted default probabilities (PD) and an example exposure at
   default (EAD) field in the notebook.

## Results

The notebook's recorded test-set results are:

| Model | Accuracy | ROC AUC |
| --- | ---: | ---: |
| Logistic Regression | 0.8096 | 0.7155 |
| XGBoost | 0.8191 | 0.7772 |

In this run, XGBoost had the higher ROC AUC. The notebook's analysis also
highlights recent payment delays as an important risk signal and compares
payment and balance trends between customers who did and did not default.

## Notes

The target is imbalanced, so accuracy alone does not describe how well a model
identifies defaulters; review the classification reports and ROC AUC as well.
Results are tied to the notebook's particular split and model settings. This
project is an educational analysis, not a validated credit decision system or
financial advice. The expected-loss section is illustrative and does not
constitute a complete Basel-compliant calculation.
