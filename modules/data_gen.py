import numpy as np
import pandas as pd

from .config import *


def norm(x, name):
    lo, hi = RANGES[name]
    return (x - lo) / (hi - lo)


def make_clean_data(n=N_ROWS, seed=SEED):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()
    df["project_id"] = np.arange(1, n + 1)
    df["sector"] = rng.choice(SECTORS, n, p=[0.3, 0.3, 0.2, 0.2])

    cost = np.clip(rng.lognormal(2.8, 0.9, n), 1, 100)
    profitability = np.clip(rng.normal(70, 35, n), 5, 180)
    profit = np.clip(cost * profitability / 100, 0.5, 150)
    payback = np.clip(rng.normal(5, 2.2, n) - (profitability - 70) / 40, 1, 10)

    df["project_cost"] = cost.round(2)
    df["expected_profit"] = profit.round(2)
    df["payback_period"] = payback.round(1)
    df["team_experience"] = np.clip(rng.gamma(4, 2.5, n), 0, 20).round(1)
    df["market_volatility"] = np.clip(rng.normal(45, 20, n) + (df["sector"] == "ИТ") * 10, 0, 100).round(1)
    df["technology_readiness"] = np.clip(np.round(rng.normal(6, 1.8, n)), 1, 9).astype(int)
    df["competition_level"] = np.clip(np.round(rng.normal(5.5, 2, n)), 1, 10).astype(int)
    df["strategic_fit"] = np.clip(np.round(rng.normal(6, 2, n)), 1, 10).astype(int)
    df["regulatory_risk"] = np.clip(rng.normal(35, 20, n) + (df["sector"] == "Энергетика") * 15, 0, 100).round(1)

    df["risk_class"] = make_risk_class(df, rng)
    return df


# Формула риска: взвешенная сумма нормированных признаков (0..1) + небольшой шум.
# Самые большие веса у окупаемости (0.22) и рентабельности (0.20) - для инвестора это главное.
# Дальше идут рыночные риски (волатильность 0.15, регуляторный 0.12, конкуренция 0.08)
# и организационные факторы (опыт 0.08, технологии 0.07, соответствие стратегии 0.05).
# Энергетика и логистика получают небольшую надбавку (капиталоемкие, жесткое регулирование).
# Классы режутся по квантилям, чтобы получить дисбаланс примерно 60 / 25 / 15.
def make_risk_class(df, rng):
    prof = df["expected_profit"] / df["project_cost"] * 100
    prof_n = np.clip(prof / 150, 0, 1)
    sector_add = df["sector"].map({"Производство": 0.0, "ИТ": 0.0, "Логистика": 0.01, "Энергетика": 0.03})

    score = (
        0.22 * norm(df["payback_period"], "payback_period")
        + 0.20 * (1 - prof_n)
        + 0.15 * norm(df["market_volatility"], "market_volatility")
        + 0.12 * norm(df["regulatory_risk"], "regulatory_risk")
        + 0.08 * norm(df["competition_level"], "competition_level")
        + 0.08 * (1 - norm(df["team_experience"], "team_experience"))
        + 0.07 * (1 - norm(df["technology_readiness"], "technology_readiness"))
        + 0.05 * (1 - norm(df["strategic_fit"], "strategic_fit"))
        + sector_add
        + rng.normal(0, 0.025, len(df))
    )
    q60, q85 = np.quantile(score, [0.60, 0.85])
    return np.where(score <= q60, 0, np.where(score <= q85, 1, 2))


def make_dirty(df, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    n = len(df)

    for col in ["team_experience", "market_volatility", "regulatory_risk"]:
        idx = rng.choice(n, int(n * 0.05), replace=False)
        df.loc[idx, col] = np.nan

    for col in ["project_cost", "expected_profit", "payback_period"]:
        idx = rng.choice(n, int(n * 0.03), replace=False)
        df.loc[idx, col] = (df.loc[idx, col] * rng.uniform(3, 6, len(idx))).round(2)

    return df


def generate_and_save():
    clean = make_clean_data()
    raw = make_dirty(clean)
    raw.to_csv(RAW_PATH, index=False, encoding="utf-8-sig")
    return raw
