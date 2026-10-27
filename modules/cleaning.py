import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from .config import *


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def fill_missing(df, method="median"):
    df = df.copy()
    if method == "knn":
        df[NUM_COLS] = KNNImputer(n_neighbors=5).fit_transform(df[NUM_COLS])
    else:
        for col in NUM_COLS:
            df[col] = df[col].fillna(df[col].median())
    return df


def cap_outliers(df):
    df = df.copy()
    rows = []
    for col in NUM_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low = max(q1 - 1.5 * iqr, RANGES[col][0])
        high = min(q3 + 1.5 * iqr, RANGES[col][1])
        count = int(((df[col] < low) | (df[col] > high)).sum())
        df[col] = df[col].clip(low, high)
        rows.append({"column": col, "low": round(low, 2), "high": round(high, 2), "capped": count})
    return df, pd.DataFrame(rows)


def add_derived(df):
    df = df.copy()
    df["profitability"] = df["expected_profit"] / df["project_cost"] * 100
    df["roi_index"] = df["expected_profit"] / (df["project_cost"] / df["payback_period"])
    df["risk_density"] = (df["market_volatility"] + df["regulatory_risk"] + df["competition_level"] * 10) / 3
    df["capability_score"] = df["team_experience"] * df["technology_readiness"] * df["strategic_fit"]
    return df


def clean_and_save(method="median"):
    raw = read_csv(RAW_PATH)
    missing = raw[NUM_COLS].isna().sum()
    df = fill_missing(raw, method)
    df, report = cap_outliers(df)
    df = add_derived(df)
    df.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")
    report["missing_before"] = report["column"].map(missing).values
    report.to_csv(os.path.join(RES_DIR, "cleaning_report.csv"), index=False, encoding="utf-8-sig")
    return df, report
