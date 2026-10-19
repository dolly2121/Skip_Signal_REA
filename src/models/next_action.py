"""Next action model: skip, view, save, enquire or leave.
Picks the decision tree if it is within 2 macro F1 points of the best,
because product teams can read a tree."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from src.training_data import FEATURES

GOLD = Path("data/gold")
ARTIFACTS = Path("artifacts")
REPORTS = Path("reports")
TREE = "Decision tree (depth 5)"
TREE_MARGIN = 0.02


def imp():
    return SimpleImputer(strategy="median")


def run():
    fit = pd.read_parquet(GOLD / "examples_fit.parquet")
    test = pd.read_parquet(GOLD / "examples_test.parquet")
    Xf, yf = fit[FEATURES], fit.action
    Xt, yt = test[FEATURES], test.action

    models = {
        "Always most common": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_pipeline(
            imp(), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        TREE: make_pipeline(
            imp(), DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                          random_state=42)),
        "Random forest": make_pipeline(
            imp(), RandomForestClassifier(n_estimators=200, max_depth=10,
                                          min_samples_leaf=20, class_weight="balanced",
                                          n_jobs=-1, random_state=42)),
        "Gradient boosting": make_pipeline(
            imp(), HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                  class_weight="balanced",
                                                  random_state=42)),
    }

    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(Xf, yf)
        p = m.predict(Xt)
        rows.append({
            "model": name,
            "accuracy": round(accuracy_score(yt, p), 3),
            "macro_F1": round(f1_score(yt, p, average="macro"), 3),
            "enquire_precision": round(precision_score(yt == "enquire", p == "enquire",
                                                       zero_division=0), 3),
            "enquire_recall": round(recall_score(yt == "enquire", p == "enquire"), 3),
        })
        fitted[name] = m

    table = pd.DataFrame(rows)
    real = table[table.model != "Always most common"]
    best_f1 = real.macro_F1.max()
    tree_f1 = real.set_index("model").loc[TREE, "macro_F1"]
    pick = TREE if best_f1 - tree_f1 <= TREE_MARGIN else real.sort_values(
        "macro_F1").iloc[-1].model

    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(fitted[pick], ARTIFACTS / "next_action.joblib")
    table.to_csv(REPORTS / "next_action_models.csv", index=False)

    tree = fitted[TREE][-1]
    (REPORTS / "next_action_tree.txt").write_text(
        export_text(tree, feature_names=FEATURES, max_depth=3))

    print(table.to_string(index=False))
    print(f"\nPicked: {pick}  (tree macro F1 {tree_f1}, best {best_f1})")


if __name__ == "__main__":
    run()