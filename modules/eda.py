import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import *


def save(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def run_eda(df):
    paths = []

    cols = NUM_COLS + DERIVED_COLS + ["risk_class"]
    corr = df[cols].corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
    plt.title("Корреляции признаков")
    paths.append(save("eda_corr.png"))

    box_cols = ["profitability", "payback_period", "market_volatility",
                "regulatory_risk", "capability_score", "risk_density"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.ravel(), box_cols):
        sns.boxplot(data=df, x="risk_class", y=col, ax=ax)
        ax.set_title(col)
    paths.append(save("eda_boxplots.png"))

    hist_cols = ["project_cost", "expected_profit", "payback_period", "team_experience",
                 "market_volatility", "regulatory_risk"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.ravel(), hist_cols):
        ax.hist(df[col], bins=30, color="steelblue")
        ax.set_title(col)
    paths.append(save("eda_hist.png"))

    plt.figure(figsize=(6, 4))
    df["risk_class"].value_counts().sort_index().plot(kind="bar", color=["green", "orange", "red"])
    plt.title("Распределение классов риска")
    plt.xlabel("risk_class")
    plt.ylabel("количество")
    paths.append(save("eda_classes.png"))

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="profitability", y="payback_period", hue="risk_class",
                    palette=["green", "orange", "red"], s=15)
    plt.title("Рентабельность и срок окупаемости")
    paths.append(save("eda_scatter.png"))

    plt.figure(figsize=(7, 4))
    pd_table = df.groupby(["sector", "risk_class"]).size().unstack(fill_value=0)
    pd_table.div(pd_table.sum(axis=1), axis=0).plot(kind="bar", stacked=True,
                                                    color=["green", "orange", "red"], ax=plt.gca())
    plt.title("Доли классов риска по секторам")
    plt.xticks(rotation=0)
    paths.append(save("eda_sector.png"))

    top = corr["risk_class"].drop("risk_class").sort_values(key=abs, ascending=False)
    return paths, top
