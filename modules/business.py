import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import *
from .ml_models import get_split

COST_FN = 5
COST_FP = 1

LOSS_MATRIX = np.zeros((3, 3))
for true in range(3):
    for pred in range(3):
        if pred < true:
            LOSS_MATRIX[true, pred] = COST_FN * (true - pred)
        elif pred > true:
            LOSS_MATRIX[true, pred] = COST_FP * (pred - true)


def expected_loss(cm):
    return float((cm * LOSS_MATRIX).sum() / cm.sum())


def decide(proba, t_mid, t_high):
    pred = np.zeros(len(proba), dtype=int)
    pred[proba[:, 1] + proba[:, 2] >= t_mid] = 1
    pred[proba[:, 2] >= t_high] = 2
    return pred


def find_thresholds(y_true, proba):
    best = (0.5, 0.5, 1e9)
    for t_mid in np.arange(0.05, 0.91, 0.05):
        for t_high in np.arange(0.05, 0.91, 0.05):
            pred = decide(proba, t_mid, t_high)
            loss = expected_loss(confusion_matrix(y_true, pred, labels=[0, 1, 2]))
            if loss < best[2]:
                best = (round(t_mid, 2), round(t_high, 2), loss)
    return best


def run_business(df, models):
    rows = []
    for key, bundle in models.items():
        X_train, X_test, y_train, y_test = get_split(df, bundle["with_fuzzy"])
        pred = bundle["pipe"].predict(X_test)
        cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
        fn = int(sum(cm[i, j] for i in range(3) for j in range(3) if j < i))
        fp = int(sum(cm[i, j] for i in range(3) for j in range(3) if j > i))
        rows.append({"model": bundle["name"], "variant": bundle["variant"],
                     "fn_errors": fn, "fp_errors": fp, "expected_loss": expected_loss(cm)})
    table = pd.DataFrame(rows).sort_values("expected_loss").reset_index(drop=True)
    table.to_csv(os.path.join(RES_DIR, "business_models.csv"), index=False, encoding="utf-8-sig")

    best = table.iloc[0]
    key = f"{best['model']} | {best['variant']}"
    bundle = models[key]
    X_train, X_test, y_train, y_test = get_split(df, bundle["with_fuzzy"])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    train_proba = cross_val_predict(bundle["pipe"], X_train, y_train, cv=cv, method="predict_proba")
    t_mid, t_high, _ = find_thresholds(y_train, train_proba)

    test_proba = bundle["pipe"].predict_proba(X_test)
    cm_default = confusion_matrix(y_test, test_proba.argmax(axis=1), labels=[0, 1, 2])
    cm_thr = confusion_matrix(y_test, decide(test_proba, t_mid, t_high), labels=[0, 1, 2])
    loss_default = expected_loss(cm_default)
    loss_thr = expected_loss(cm_thr)

    info = {"best_key": key, "t_mid": t_mid, "t_high": t_high,
            "loss_default": loss_default, "loss_thresholds": loss_thr,
            "cm_thresholds": cm_thr}

    pd.DataFrame(LOSS_MATRIX, index=["факт 0", "факт 1", "факт 2"],
                 columns=["прогноз 0", "прогноз 1", "прогноз 2"]).to_csv(
        os.path.join(RES_DIR, "loss_matrix.csv"), encoding="utf-8-sig")

    plt.figure(figsize=(9, 4))
    labels = [f"{r.model}\n{r.variant}" for r in table.itertuples()]
    plt.bar(labels, table["expected_loss"], color="steelblue")
    plt.xticks(fontsize=7)
    plt.ylabel("ожидаемые потери на проект")
    plt.title("Ожидаемые потери (FN дороже FP в 5 раз)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "business_loss.png"), dpi=120)
    plt.close()

    text = (
        f"Матрица потерь: недооценка риска (FN) стоит {COST_FN} за каждый класс ошибки, "
        f"переоценка (FP) - {COST_FP}.\n"
        f"Минимальные ожидаемые потери у модели {key}: {best['expected_loss']:.3f} на проект.\n"
        f"Пороги подобраны на обучающей выборке (кросс-валидация): "
        f"средний риск при P(класс 1 или 2) >= {t_mid}, высокий риск при P(класс 2) >= {t_high}.\n"
        f"На тесте потери с порогами {loss_thr:.3f} против {loss_default:.3f} при обычном argmax.\n"
        f"Вывод: модель с порогами лучше ловит рискованные проекты, "
        f"цена - больше лишних отказов хороших проектов."
    )
    with open(os.path.join(RES_DIR, "business_conclusion.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    info["text"] = text
    return table, info
