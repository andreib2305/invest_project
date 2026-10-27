import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from .config import *
from .ml_models import get_features


def get_shap(bundle, df, n=200):
    X = get_features(df, bundle["with_fuzzy"])[bundle["features"]]
    sample = X.sample(min(n, len(X)), random_state=SEED)
    scaler = bundle["pipe"][:-1]
    model = bundle["pipe"][-1]
    Xs = scaler.transform(sample)

    if isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(Xs)
        base = np.array(explainer.expected_value).ravel()
    else:
        sample = sample.iloc[:60]
        Xs = Xs[:60]
        background = shap.sample(Xs, 40, random_state=SEED)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        values = explainer.shap_values(Xs, nsamples=100)
        base = np.array(explainer.expected_value).ravel()

    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    return sample, np.array(values), base


def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def make_shap_plots(bundle, df):
    sample, values, base = get_shap(bundle, df)
    features = list(sample.columns)
    paths = []

    mean_abs = np.abs(values).mean(axis=(0, 2))
    order = np.argsort(mean_abs)
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(features)[order], mean_abs[order], color="steelblue")
    plt.title("SHAP: важность признаков (summary)")
    plt.xlabel("среднее |SHAP|")
    paths.append(save_fig("shap_summary.png"))

    predicted = bundle["pipe"].predict(sample)
    high = np.where(predicted == 2)[0]
    i = int(high[0]) if len(high) else 0
    cls = int(predicted[i])
    exp = shap.Explanation(values=values[i, :, cls], base_values=base[cls],
                           data=sample.iloc[i].values, feature_names=features)
    plt.figure()
    shap.plots.waterfall(exp, show=False)
    paths.append(save_fig("shap_waterfall.png"))

    top = np.argsort(mean_abs)[::-1][:10][::-1]
    per_class = np.abs(values).mean(axis=0)
    y = np.arange(len(top))
    plt.figure(figsize=(8, 6))
    for c, color in zip(range(3), ["green", "orange", "red"]):
        plt.barh(y + (c - 1) * 0.27, per_class[top, c], height=0.27, color=color, label=CLASS_NAMES[c])
    plt.yticks(y, np.array(features)[top])
    plt.title("SHAP по классам (топ-10 признаков)")
    plt.legend()
    paths.append(save_fig("shap_classes.png"))

    importance = pd.Series(mean_abs, index=features).sort_values(ascending=False)
    project_id = int(df.loc[sample.index[i], "project_id"])
    return paths, importance, project_id
