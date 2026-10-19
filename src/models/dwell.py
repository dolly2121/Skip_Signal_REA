"""Dwell time regression. Predicts how long a buyer looks at a home.
Target is log transformed because dwell time is very skewed."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.training_data import FEATURES

GOLD = Path("data/gold")
ARTIFACTS = Path("artifacts")
REPORTS = Path("reports")
QUICK_EXIT_S = 5


def run():
    fit = pd.read_parquet(GOLD / "examples_fit.parquet")
    test = pd.read_parquet(GOLD / "examples_test.parquet")
    fit, test = fit[fit.action != "leave"], test[test.action != "leave"]

    y_fit = np.log1p(fit.dwell_s)
    y_true = test.dwell_s.to_numpy()
    qe_true = y_true < QUICK_EXIT_S

    models = {
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               Ridge(alpha=1.0)),
        "Gradient boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          random_state=42)),
    }

    median_pred = np.full(len(y_true), np.median(fit.dwell_s))
    rows = [{"model": "Median (no model)",
             "MAE_s": round(mean_absolute_error(y_true, median_pred), 2),
             "quick_exit_F1": 0.0}]
    fitted = {}
    for name, m in models.items():
        m.fit(fit[FEATURES], y_fit)
        pred = np.expm1(m.predict(test[FEATURES]))
        rows.append({"model": name,
                     "MAE_s": round(mean_absolute_error(y_true, pred), 2),
                     "quick_exit_F1": round(f1_score(qe_true, pred < QUICK_EXIT_S), 3)})
        fitted[name] = m

    table = pd.DataFrame(rows)
    best = table[table.model.isin(models)].sort_values("MAE_s").iloc[0].model

    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(fitted[best], ARTIFACTS / "dwell.joblib")
    table.to_csv(REPORTS / "dwell_models.csv", index=False)

    print(table.to_string(index=False))
    print(f"\nPicked: {best} (lowest MAE)")


if __name__ == "__main__":
    run()