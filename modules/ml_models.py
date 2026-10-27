import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import *

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

VARIANTS = {False: "без fuzzy_risk", True: "с fuzzy_risk"}


def get_features(df, with_fuzzy):
    cols = NUM_COLS + DERIVED_COLS
    if with_fuzzy:
        cols = cols + ["fuzzy_risk"]
    X = df[cols].copy()
    for s in SECTORS:
        X["sector_" + s] = (df["sector"] == s).astype(int)
    return X


def get_split(df, with_fuzzy):
    X = get_features(df, with_fuzzy)
    y = df["risk_class"].values
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def make_pipe(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def make_mlp(params=None):
    p = {"hidden_layer_sizes": (64, 32), "alpha": 0.001, "learning_rate_init": 0.001, "activation": "relu"}
    if params:
        p.update(params)
    return MLPClassifier(max_iter=500, early_stopping=True, random_state=SEED, **p)


def tune_mlp(X_train, y_train, n_trials=N_TRIALS):
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)

    def objective(trial):
        layers = trial.suggest_int("n_layers", 1, 2)
        size1 = trial.suggest_int("size1", 16, 128)
        size2 = trial.suggest_int("size2", 8, 64)
        sizes = (size1,) if layers == 1 else (size1, size2)
        params = {
            "hidden_layer_sizes": sizes,
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        }
        pipe = make_pipe(make_mlp(params))
        return cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro").mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    sizes = (best["size1"],) if best["n_layers"] == 1 else (best["size1"], best["size2"])
    return {"hidden_layer_sizes": sizes, "alpha": best["alpha"],
            "learning_rate_init": best["learning_rate_init"], "activation": best["activation"]}


def calc_metrics(y_true, pred, proba):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "roc_auc_ovr": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
        "mcc": matthews_corrcoef(y_true, pred),
    }


def draw_confusion(cms):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (key, cm) in zip(axes.ravel(), cms.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_title(key, fontsize=9)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xlabel("предсказано")
        ax.set_ylabel("факт")
    path = os.path.join(FIG_DIR, "confusion_matrices.png")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return path


def train_all(df, n_trials=N_TRIALS):
    models = {}
    rows = []
    cms = {}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    for with_fuzzy in (False, True):
        X_train, X_test, y_train, y_test = get_split(df, with_fuzzy)
        mlp_params = tune_mlp(X_train, y_train, n_trials)
        estimators = {
            "MLP": make_mlp(mlp_params),
            "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=SEED),
        }
        for name, est in estimators.items():
            pipe = make_pipe(est)
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)

            row = {"model": name, "variant": VARIANTS[with_fuzzy]}
            row.update(calc_metrics(y_test, pred, proba))
            row["cv_f1_mean"] = cv_scores.mean()
            row["cv_f1_std"] = cv_scores.std()
            rows.append(row)

            key = f"{name} | {VARIANTS[with_fuzzy]}"
            cms[key] = confusion_matrix(y_test, pred)
            models[key] = {"name": name, "variant": VARIANTS[with_fuzzy], "with_fuzzy": with_fuzzy,
                           "pipe": pipe, "features": list(X_train.columns)}
            print("готово:", key, "F1 =", round(row["f1_macro"], 3))

    table = pd.DataFrame(rows)
    table.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    draw_confusion(cms)

    best_row = table.sort_values(["f1_macro", "mcc"], ascending=False).iloc[0]
    best_key = f"{best_row['model']} | {best_row['variant']}"
    joblib.dump(models[best_key], BEST_MODEL_PATH)
    joblib.dump(models, ALL_MODELS_PATH)
    return table, models, best_key


def load_best():
    return joblib.load(BEST_MODEL_PATH)


def load_all():
    return joblib.load(ALL_MODELS_PATH)
