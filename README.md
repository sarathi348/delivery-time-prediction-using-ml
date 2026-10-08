# Delivery Time Prediction using Machine Learning Regression Models

Project-Based Learning (PBL) – Machine Learning, Chennai Institute of Technology
Team: Sarathi R (2104251040896), Sudharshan Lakshmipathy (2104251040984)

Predicts delivery time in minutes from information known at order time and
compares Linear Regression, Decision Tree, Random Forest and Gradient Boosting.

## Files

| File | Purpose |
|---|---|
| `simulate.py` | Rule-based generator for the 12,120-row dataset (with deliberate faults) |
| `features.py` | Feature engineering (24 raw inputs → 34 engineered inputs) |
| `experiment.py` | Cleaning, 70/15/15 split, Iteration 1 and 2, tuning, evaluation, model saving |
| `figs.py` | Report figures 5.1, 5.2, 5.3, 6.1, 6.2 |
| `predict.py` | Command-line prediction interface |

## Run

```
pip install -r requirements.txt
python simulate.py        # writes delivery_data.csv
python experiment.py      # writes results.json, delivery_model.joblib, test_predictions.csv
python figs.py            # writes the five PNG figures
python predict.py --distance 7.5 --traffic High --weather Rain --area Metropolitan \
                  --hour 19 --day 5 --stops 1 --prep 15 --experience 2.5 --rating 4.4
```

Seed is 42 everywhere. Grid search takes a few minutes on two cores.

## Note on data

The data is simulated. Reported accuracy shows how well the models recover the
generator's rules, not accuracy on a real delivery service.
