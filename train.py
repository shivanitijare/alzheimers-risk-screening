"""
Alzheimer's risk classification: training pipeline.

Reproduces the Chem 277A Project II modelling work as a reusable sklearn
Pipeline so that (a) encoders are fit on training data only, and (b) a single
raw patient record can be scored at inference time by the Streamlit app.

Tuned hyperparameters are the ones selected by the notebook's GridSearchCV
runs (5-fold stratified CV on the training split).

Run:  python train.py            (fast: fits final models with known best params)
      python train.py --search   (slow: re-runs the full GridSearchCV)
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "alzheimers_prediction_dataset.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

TARGET = "Alzheimer's Diagnosis"

NUMERIC = ["Age", "Education Level", "BMI", "Cognitive Test Score"]
ORDINAL = [
    "Physical Activity Level",
    "Depression Level",
    "Air Pollution Exposure",
    "Social Engagement Level",
    "Income Level",
    "Stress Levels",
]
NOMINAL = [
    "Gender",
    "Smoking Status",
    "Alcohol Consumption",
    "Diabetes",
    "Hypertension",
    "Cholesterol Level",
    "Family History of Alzheimer's",
    "Sleep Quality",
    "Dietary Habits",
    "Employment Status",
    "Marital Status",
    "Genetic Risk Factor (APOE-ε4 allele)",
    "Urban vs Rural Living",
]
HIGH_CARD = ["Country"]

# Best parameters found by the notebook's GridSearchCV (5-fold stratified).
BEST_PARAMS = {
    "Logistic Regression": {"C": 0.01, "l1_ratio": 0.0},
    "Decision Tree": {"criterion": "gini", "max_depth": 3, "min_samples_split": 2},
    "Random Forest": {"max_depth": 10, "min_samples_split": 5, "n_estimators": 200},
    "SVM": {"C": 0.1},
}

# Test accuracies reported in the notebook, for side-by-side comparison.
NOTEBOOK_TEST_ACC = {
    "Logistic Regression": 0.7129,
    "Decision Tree": 0.7297,
    "Random Forest": 0.7277,
    "SVM": 0.7160,
}

SEARCH_GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__l1_ratio": [0.0, 0.5, 1.0]},
    "Decision Tree": {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [3, 5, 7, 10, None],
        "clf__min_samples_split": [2, 5, 10],
    },
    "Random Forest": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    },
    "SVM": {"clf__C": [0.1, 1.0, 10.0]},
}


def load_data():
    """Load the dataset and normalise the curly apostrophes in column names."""
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.replace("’", "'", regex=False)
    return df


def build_preprocessor():
    """Encode each feature group according to its type.

    One-hot (drop first) for binary/nominal, explicit ordinal encoding for the
    Low/Medium/High scales, binary encoding for high-cardinality Country, and
    standard scaling for the four numeric columns. Wrapped in a
    ColumnTransformer so it is fit on training folds only.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            (
                "ord",
                OrdinalEncoder(
                    categories=[["Low", "Medium", "High"]] * len(ORDINAL),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ORDINAL,
            ),
            (
                "nom",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                NOMINAL,
            ),
            ("country", BinaryEncoder(cols=HIGH_CARD), HIGH_CARD),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_estimator(name, params):
    """Return an unfitted classifier for the given model name."""
    if name == "Logistic Regression":
        # sklearn >=1.8 deprecates `penalty`; l1_ratio alone selects the mix
        # (0.0 == ridge, 1.0 == lasso), which is what the tuned value encodes.
        return LogisticRegression(
            solver="saga",
            max_iter=2000,
            random_state=RANDOM_STATE,
            **params,
        )
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    if name == "Random Forest":
        return RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE, **params)
    if name == "SVM":
        # LinearSVC has no predict_proba, so calibrate it to make the
        # threshold analysis comparable across all four models.
        return CalibratedClassifierCV(
            LinearSVC(
                class_weight="balanced",
                max_iter=2000,
                dual="auto",
                random_state=RANDOM_STATE,
                **params,
            ),
            method="sigmoid",
            cv=3,
        )
    raise ValueError(name)


def evaluate(name, pipe, X_test, y_test):
    """Score a fitted pipeline on the held-out test set."""
    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, tpr, thr = roc_curve(y_test, proba)
    return {
        "model": name,
        "best_params": BEST_PARAMS[name],
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "notebook_accuracy": NOTEBOOK_TEST_ACC[name],
        "roc": {
            "fpr": [round(v, 5) for v in fpr[:: max(1, len(fpr) // 250)].tolist()],
            "tpr": [round(v, 5) for v in tpr[:: max(1, len(tpr) // 250)].tolist()],
        },
    }


def build(search=False):
    """Fit every model, evaluate, and write all artifacts to disk.

    Importable so the app can self-heal: if the committed pickles were written
    by a different scikit-learn version than the host installs, the app calls
    this instead of failing to start.
    """
    args = argparse.Namespace(search=search)

    t0 = time.time()
    df = load_data()
    y = df[TARGET].map({"No": 0, "Yes": 1})
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"train {X_train.shape}  test {X_test.shape}")
    print(f"class balance (train): {y_train.value_counts(normalize=True).round(3).to_dict()}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = [], {}

    for name in BEST_PARAMS:
        t = time.time()
        if args.search:
            pipe = Pipeline(
                [("prep", build_preprocessor()), ("clf", make_estimator(name, {}))]
            )
            search = GridSearchCV(
                pipe, SEARCH_GRIDS[name], cv=cv, scoring="accuracy", n_jobs=-1
            )
            search.fit(X_train, y_train)
            pipe = search.best_estimator_
            print(f"{name}: searched best={search.best_params_} cv={search.best_score_:.4f}")
        else:
            pipe = Pipeline(
                [
                    ("prep", build_preprocessor()),
                    ("clf", make_estimator(name, BEST_PARAMS[name])),
                ]
            )
            pipe.fit(X_train, y_train)

        res = evaluate(name, pipe, X_test, y_test)
        results.append(res)
        fitted[name] = pipe
        print(
            f"{name}: acc={res['accuracy']:.4f} (notebook {res['notebook_accuracy']:.4f}) "
            f"recall={res['recall']:.4f} auc={res['roc_auc']:.4f} "
            f"[{time.time() - t:.1f}s]"
        )

    # Feature importance from the tuned Random Forest, on encoded feature names.
    rf = fitted["Random Forest"]
    feat_names = rf.named_steps["prep"].get_feature_names_out()
    importances = (
        pd.Series(rf.named_steps["clf"].feature_importances_, index=feat_names)
        .sort_values(ascending=False)
    )

    # Mutual information on the encoded training matrix.
    Xt = rf.named_steps["prep"].transform(X_train)
    mi = pd.Series(
        mutual_info_classif(Xt, y_train, random_state=RANDOM_STATE), index=feat_names
    ).sort_values(ascending=False)

    # Held-out probabilities for the threshold / screening-policy analysis.
    probas = {n: fitted[n].predict_proba(X_test)[:, 1].astype(np.float32) for n in fitted}

    joblib.dump(fitted, ART / "models.joblib", compress=3)
    joblib.dump(
        {
            "y_test": y_test.to_numpy(dtype=np.int8),
            "probas": probas,
            "importances": importances,
            "mutual_info": mi,
            "feature_names": list(feat_names),
        },
        ART / "eval.joblib",
        compress=3,
    )

    # Raw-schema reference so the app can build a valid single-patient record.
    schema = {
        "numeric": {
            c: {
                "min": float(X[c].min()),
                "max": float(X[c].max()),
                "median": float(X[c].median()),
            }
            for c in NUMERIC
        },
        "ordinal": {c: ["Low", "Medium", "High"] for c in ORDINAL},
        "nominal": {c: sorted(X[c].dropna().unique().tolist()) for c in NOMINAL},
        "high_card": {c: sorted(X[c].dropna().unique().tolist()) for c in HIGH_CARD},
        "column_order": list(X.columns),
    }
    (ART / "schema.json").write_text(json.dumps(schema, indent=2))
    (ART / "results.json").write_text(json.dumps(results, indent=2))

    # Descriptive statistics for the overview panel.
    desc = {
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "target_balance": {
            k: float(v) for k, v in df[TARGET].value_counts(normalize=True).items()
        },
        "missing_total": int(df.isna().sum().sum()),
        "numeric_by_class": {
            c: {
                "No": df.loc[df[TARGET] == "No", c].describe().round(3).to_dict(),
                "Yes": df.loc[df[TARGET] == "Yes", c].describe().round(3).to_dict(),
            }
            for c in NUMERIC
        },
        "positive_rate_by_category": {
            c: (df.assign(_p=y).groupby(c, observed=True)["_p"].mean().round(4).to_dict())
            for c in NOMINAL + ORDINAL
        },
        "age_bins": (
            df.assign(_p=y, _b=pd.cut(df["Age"], bins=range(50, 101, 5)))
            .groupby("_b", observed=True)["_p"]
            .agg(["mean", "count"])
            .round(4)
            .reset_index()
            .astype({"_b": str})
            .to_dict(orient="records")
        ),
    }
    (ART / "descriptive.json").write_text(json.dumps(desc, indent=2, default=str))

    t = time.time()
    ceiling = ceiling_study(X_train, X_test, y_train, y_test)
    ceiling["best_tuned"] = {
        "label": max(results, key=lambda r: r["accuracy"])["model"],
        "accuracy": max(r["accuracy"] for r in results),
    }
    ceiling["naive_baseline"] = float((y_train == 0).mean())
    (ART / "ceiling.json").write_text(json.dumps(ceiling, indent=2))
    print(f"ceiling study [{time.time() - t:.1f}s]")

    print(f"\nartifacts written to {ART}  [total {time.time() - t0:.1f}s]")
    print(importances.head(10).round(4).to_string())


def ceiling_study(X_train, X_test, y_train, y_test):
    """Test whether the ~73% plateau is the algorithm or the feature set.

    Three cheaper models and a learning curve. If a stronger algorithm and more
    data both fail to move the number, the limit is the data, and that is a
    finding rather than a failure.
    """
    from sklearn.compose import ColumnTransformer as CT
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import learning_curve

    rows = []

    def score(label, pipe, cols, note):
        pipe.fit(X_train[cols], y_train)
        pred = pipe.predict(X_test[cols])
        prob = pipe.predict_proba(X_test[cols])[:, 1]
        rows.append(
            {
                "label": label,
                "accuracy": float(accuracy_score(y_test, pred)),
                "roc_auc": float(roc_auc_score(y_test, prob)),
                "n_features": len(cols),
                "note": note,
            }
        )

    # One variable only.
    score(
        "Age alone",
        Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
        ["Age"],
        "A single feature, logistic regression.",
    )

    # The three features both selection methods agreed on.
    top3 = ["Age", "Genetic Risk Factor (APOE-ε4 allele)", "Family History of Alzheimer's"]
    score(
        "Top 3 features",
        Pipeline(
            [
                (
                    "p",
                    CT(
                        [
                            ("n", StandardScaler(), ["Age"]),
                            ("c", OneHotEncoder(drop="first"), top3[1:]),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        ),
        top3,
        "Age, APOE-ε4 and family history only.",
    )

    # A stronger algorithm than anything in the original notebook.
    score(
        "Gradient boosting",
        Pipeline(
            [
                ("prep", build_preprocessor()),
                ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=300)),
            ]
        ),
        list(X_train.columns),
        "All 24 features, the algorithm that usually wins on tabular data.",
    )

    # Does more data help?
    rf = Pipeline(
        [
            ("prep", build_preprocessor()),
            ("clf", make_estimator("Random Forest", BEST_PARAMS["Random Forest"])),
        ]
    )
    sizes, _, val = learning_curve(
        rf,
        X_train,
        y_train,
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
        cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
        n_jobs=-1,
    )
    curve = [
        {"n": int(s), "accuracy": float(v)} for s, v in zip(sizes, val.mean(axis=1))
    ]
    return {"models": rows, "learning_curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true", help="re-run full GridSearchCV")
    build(search=ap.parse_args().search)


if __name__ == "__main__":
    main()
