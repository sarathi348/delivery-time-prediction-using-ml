"""Cleaning, two training iterations, evaluation and model saving."""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

from features import basic, engineer

SEED = 42
TARGET = "delivery_time_min"
EXPECTED = ["order_id", "order_hour", "day_of_week", "distance_km", "traffic_level",
            "weather", "vehicle_type", "area_type", "courier_experience_yrs",
            "courier_rating", "prep_time_min", "multiple_deliveries", "festival", TARGET]


def metrics(y, p):
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


def clean(df, log):
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    log["raw_rows"] = len(df)
    n = len(df); df = df.drop_duplicates("order_id")
    log["duplicates_removed"] = n - len(df)
    n = len(df); df = df[df["distance_km"] > 0]
    log["negative_distance_removed"] = n - len(df)
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    cap = q3 + 3 * (q3 - q1)
    n = len(df); df = df[df[TARGET] <= cap]
    log["outlier_cap_minutes"] = float(cap)
    log["outliers_removed"] = n - len(df)
    log["clean_rows"] = len(df)
    return df.reset_index(drop=True)


def models(tuned, X, y):
    """Return fitted models; tuned=False uses scikit-learn defaults."""
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    out, best = {}, {}
    out["Linear Regression"] = LinearRegression().fit(X, y)
    specs = {
        "Decision Tree": (DecisionTreeRegressor(random_state=SEED),
                          {"max_depth": [6, 8, 12], "min_samples_leaf": [10, 20]}),
        "Random Forest": (RandomForestRegressor(random_state=SEED, n_jobs=-1),
                          {"n_estimators": [200], "max_features": [0.5, 1.0],
                           "min_samples_leaf": [1, 3]}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=SEED),
                              {"n_estimators": [300], "learning_rate": [0.05, 0.1],
                               "max_depth": [3, 4], "subsample": [0.8]}),
    }
    times = {"Linear Regression": 0.0}
    for name, (est, grid) in specs.items():
        t = time.time()
        if tuned:
            gs = GridSearchCV(est, grid, cv=cv, n_jobs=-1,
                              scoring="neg_root_mean_squared_error").fit(X, y)
            out[name], best[name] = gs.best_estimator_, gs.best_params_
        else:
            out[name] = est.fit(X, y)
        times[name] = time.time() - t
    return out, best, times


def main():
    log = {}
    raw = pd.read_csv("delivery_data.csv")
    df = clean(raw, log)
    log["missing_values_after_cleaning"] = {k: int(v) for k, v in
                                            df.isna().sum().items() if v}

    train, rest = train_test_split(df, test_size=0.30, random_state=SEED, shuffle=True)
    val, test = train_test_split(rest, test_size=0.50, random_state=SEED)
    log["split"] = {"train": len(train), "val": len(val), "test": len(test)}

    # Fill values come from the training part only (no leakage)
    fill = {"courier_rating": float(train["courier_rating"].median()),
            "weather": train["weather"].mode()[0],
            "traffic_level": train["traffic_level"].mode()[0]}
    log["fill_values"] = fill
    y_tr, y_va, y_te = train[TARGET], val[TARGET], test[TARGET]

    results = {"cleaning": log}
    results["eda"] = {
        "target": {"min": float(df[TARGET].min()), "max": float(df[TARGET].max()),
                   "mean": float(df[TARGET].mean()), "median": float(df[TARGET].median()),
                   "std": float(df[TARGET].std()),
                   "q1": float(df[TARGET].quantile(.25)), "q3": float(df[TARGET].quantile(.75))},
        "corr_with_target": df.select_dtypes("number").corr()[TARGET].drop(TARGET).round(3).to_dict(),
        "mean_by_traffic": df.groupby("traffic_level")[TARGET].mean().round(1).to_dict(),
        "mean_by_weather": df.groupby("weather")[TARGET].mean().round(1).to_dict(),
        "mean_by_vehicle": df.groupby("vehicle_type")[TARGET].mean().round(1).to_dict(),
    }

    # ---------------- Iteration 1: raw columns, default settings ----------
    Xtr1, Xva1 = basic(train, fill), basic(val, fill)
    Xva1 = Xva1.reindex(columns=Xtr1.columns, fill_value=0.0)
    m1, _, _ = models(False, Xtr1, y_tr)
    mean_model = DummyRegressor().fit(Xtr1, y_tr)
    it1 = {"inputs": Xtr1.shape[1],
           "Mean predictor": {"val": metrics(y_va, mean_model.predict(Xva1))}}
    for n, m in m1.items():
        it1[n] = {"train": metrics(y_tr, m.predict(Xtr1)), "val": metrics(y_va, m.predict(Xva1))}
    results["iteration1"] = it1

    # ---------------- Iteration 2: engineered features + tuning ------------
    Xtr = engineer(train, fill)
    cols = list(Xtr.columns)
    Xva = engineer(val, fill).reindex(columns=cols, fill_value=0.0)
    Xte = engineer(test, fill).reindex(columns=cols, fill_value=0.0)
    m2, best, times = models(True, Xtr, y_tr)
    mean_model = DummyRegressor().fit(Xtr, y_tr)
    it2 = {"inputs": len(cols), "best_params": best, "train_seconds": times,
           "Mean predictor": {"val": metrics(y_va, mean_model.predict(Xva)),
                              "test": metrics(y_te, mean_model.predict(Xte))}}
    for n, m in m2.items():
        it2[n] = {"train": metrics(y_tr, m.predict(Xtr)),
                  "val": metrics(y_va, m.predict(Xva)),
                  "test": metrics(y_te, m.predict(Xte))}
    results["iteration2"] = it2

    # ---------------- Model selection on validation RMSE -------------------
    names = list(m2)
    chosen = min(names, key=lambda n: it2[n]["val"]["RMSE"])
    final = m2[chosen]
    results["selected_model"] = chosen

    pred = final.predict(Xte)
    err = pred - y_te.values
    results["final_test"] = {
        **metrics(y_te, pred),
        "within_3": float((abs(err) <= 3).mean()), "within_5": float((abs(err) <= 5).mean()),
        "within_10": float((abs(err) <= 10).mean()), "mean_signed_error": float(err.mean())}
    cv = cross_val_score(final, Xtr, y_tr, scoring="neg_root_mean_squared_error",
                         cv=KFold(5, shuffle=True, random_state=SEED))
    results["final_cv_rmse"] = {"folds": (-cv).round(2).tolist(),
                                "mean": float(-cv.mean()), "std": float(cv.std())}

    e = test.assign(abs_err=abs(err))
    bands = pd.cut(e["distance_km"], [0, 3, 6, 10, 25], include_lowest=True,
                   labels=["0-3", "3-6", "6-10", "10-25"])
    results["mae_by_distance"] = e.groupby(bands, observed=True)["abs_err"].mean().round(2).to_dict()
    results["mae_by_traffic"] = (e.groupby("traffic_level")["abs_err"].mean().round(2)
                                 .reindex(["Low", "Medium", "High", "Jam"]).to_dict())

    pi = permutation_importance(final, Xte, y_te, n_repeats=5, random_state=SEED,
                                scoring="neg_root_mean_squared_error", n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    results["permutation_importance"] = imp.head(8).round(2).to_dict()

    # ---------------- Save artefacts ---------------------------------------
    df.to_csv("delivery_clean.csv", index=False)
    pd.DataFrame({"actual": y_te.values, "predicted": pred}).to_csv("test_predictions.csv", index=False)
    joblib.dump({"model": final, "columns": cols, "fill": fill, "name": chosen},
                "delivery_model.joblib")
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ---------------- Console summary --------------------------------------
    print(json.dumps(log, indent=2, default=str))
    print("\nIteration 1 validation RMSE:",
          {n: round(it1[n]["val"]["RMSE"], 2) for n in ["Mean predictor"] + names})
    print("Iteration 2 validation RMSE:",
          {n: round(it2[n]["val"]["RMSE"], 2) for n in ["Mean predictor"] + names})
    print("\nTest set:")
    for n in ["Mean predictor"] + names:
        t = it2[n]["test"]
        print(f"  {n:18s} MAE {t['MAE']:.2f}  RMSE {t['RMSE']:.2f}  R2 {t['R2']:.3f}")
    print("\nSelected:", chosen, results["final_test"])


if __name__ == "__main__":
    main()
