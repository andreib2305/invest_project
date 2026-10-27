import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cleaning import add_derived
from .config import *
from .fuzzy_model import compute_fuzzy
from .ml_models import get_features


def prepare(df):
    df = add_derived(df.copy())
    return df.join(compute_fuzzy(df))


def predict(bundle, df):
    X = get_features(df, bundle["with_fuzzy"])[bundle["features"]]
    proba = bundle["pipe"].predict_proba(X)
    return proba.argmax(axis=1), proba


def evaluate_project(bundle, project):
    df = prepare(pd.DataFrame([project]))
    pred, proba = predict(bundle, df)
    result = {"fin_risk": float(df["fin_risk"][0]), "mkt_risk": float(df["mkt_risk"][0]),
              "org_risk": float(df["org_risk"][0]), "fuzzy_risk": float(df["fuzzy_risk"][0]),
              "class": int(pred[0]), "proba": [float(p) for p in proba[0]]}
    return result


def what_if(bundle, project, changes):
    new_project = dict(project)
    for key, value in changes.items():
        new_project[key] = value
    before = evaluate_project(bundle, project)
    after = evaluate_project(bundle, new_project)
    return before, after


def monte_carlo(bundle, project, n=N_MC, noise=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    runs = pd.DataFrame([project] * n)
    for col in NUM_COLS:
        values = runs[col].astype(float) * rng.normal(1, noise, n)
        lo, hi = RANGES[col]
        values = values.clip(lo, hi)
        if col in INT_COLS:
            values = values.round()
        runs[col] = values
    runs = prepare(runs)
    pred, proba = predict(bundle, runs)
    runs["pred_class"] = pred

    base = evaluate_project(bundle, project)
    shares = {c: float((pred == c).mean()) for c in (0, 1, 2)}
    summary = {
        "base_class": base["class"],
        "base_fuzzy": base["fuzzy_risk"],
        "fuzzy_mean": float(runs["fuzzy_risk"].mean()),
        "fuzzy_std": float(runs["fuzzy_risk"].std()),
        "fuzzy_p5": float(runs["fuzzy_risk"].quantile(0.05)),
        "fuzzy_p95": float(runs["fuzzy_risk"].quantile(0.95)),
        "class_shares": shares,
        "stability": shares[base["class"]],
    }
    return summary, runs


def draw_monte_carlo(summary, runs, name="monte_carlo.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(runs["fuzzy_risk"], bins=25, color="steelblue")
    axes[0].axvline(summary["base_fuzzy"], color="red", label="исходный проект")
    axes[0].set_title("fuzzy_risk в 500 прогонах")
    axes[0].set_xlabel("fuzzy_risk")
    axes[0].legend()
    shares = summary["class_shares"]
    axes[1].bar([CLASS_NAMES[c] for c in shares], list(shares.values()), color=["green", "orange", "red"])
    axes[1].set_title("Доли предсказанных классов")
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path
