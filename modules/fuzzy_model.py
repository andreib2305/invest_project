import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz

from .config import *

TERMS = ["низкий", "средний", "высокий"]
X_OUT = np.arange(0, 101, 1)

UNIVERSE = {
    "profitability": (0, 200),
    "payback_period": (1, 10),
    "market_volatility": (0, 100),
    "competition_level": (1, 10),
    "regulatory_risk": (0, 100),
    "team_experience": (0, 20),
    "technology_readiness": (1, 9),
    "strategic_fit": (1, 10),
    "fin_risk": (0, 100),
    "mkt_risk": (0, 100),
    "org_risk": (0, 100),
}

OUT_MF = [("trap", [0, 0, 20, 45]), ("tri", [25, 50, 75]), ("trap", [55, 80, 100, 100])]

MF = {
    "profitability": [("trap", [0, 0, 15, 40]), ("tri", [25, 60, 100]), ("trap", [80, 110, 200, 200])],
    "payback_period": [("trap", [1, 1, 2, 4]), ("tri", [3, 5, 7]), ("trap", [6, 8, 10, 10])],
    "market_volatility": [("trap", [0, 0, 20, 40]), ("tri", [30, 50, 70]), ("trap", [60, 80, 100, 100])],
    "competition_level": [("trap", [1, 1, 3, 5]), ("tri", [3, 5.5, 8]), ("trap", [6, 8, 10, 10])],
    "regulatory_risk": [("trap", [0, 0, 20, 40]), ("tri", [30, 50, 70]), ("trap", [60, 80, 100, 100])],
    "team_experience": [("trap", [0, 0, 3, 7]), ("tri", [4, 10, 16]), ("trap", [13, 17, 20, 20])],
    "technology_readiness": [("trap", [1, 1, 3, 5]), ("tri", [3, 5, 7]), ("trap", [6, 7.5, 9, 9])],
    "strategic_fit": [("trap", [1, 1, 3, 5]), ("tri", [3, 5.5, 8]), ("trap", [6, 8, 10, 10])],
    "fin_risk": OUT_MF,
    "mkt_risk": OUT_MF,
    "org_risk": OUT_MF,
}

L, M, H = 0, 1, 2

FIN_RULES = [
    ([("profitability", L), ("payback_period", L)], M),
    ([("profitability", L), ("payback_period", M)], H),
    ([("profitability", L), ("payback_period", H)], H),
    ([("profitability", M), ("payback_period", L)], L),
    ([("profitability", M), ("payback_period", M)], M),
    ([("profitability", M), ("payback_period", H)], H),
    ([("profitability", H), ("payback_period", L)], L),
    ([("profitability", H), ("payback_period", M)], L),
    ([("profitability", H), ("payback_period", H)], M),
]

def make_grid_rules(names, risk_up, weights, low_max, high_min):
    rules = []
    for terms in itertools.product([L, M, H], repeat=len(names)):
        total = 0
        for t, up, w in zip(terms, risk_up, weights):
            total += w * (t if up else 2 - t)
        out = L if total <= low_max else (H if total >= high_min else M)
        rules.append((list(zip(names, terms)), out))
    return rules


MKT_RULES = make_grid_rules(["market_volatility", "regulatory_risk", "competition_level"],
                            [True, True, True], [1, 1, 1], 2, 4)

ORG_RULES = make_grid_rules(["team_experience", "technology_readiness", "strategic_fit"],
                            [False, False, False], [1, 1, 1], 2, 4)

FINAL_RULES = make_grid_rules(["fin_risk", "mkt_risk", "org_risk"],
                              [True, True, True], [2, 1, 1], 3, 5)

SYSTEMS = [("fin_risk", FIN_RULES), ("mkt_risk", MKT_RULES), ("org_risk", ORG_RULES)]


def membership(name, values, params=None):
    lo, hi = UNIVERSE[name]
    x = np.clip(np.asarray(values, dtype=float), lo, hi)
    result = []
    for kind, p in MF[name]:
        result.append(fuzz.trimf(x, p) if kind == "tri" else fuzz.trapmf(x, p))
    return np.array(result)


def run_system(rules, get):
    cache = {}
    n = len(get(rules[0][0][0][0]))
    agg = np.zeros((3, n))
    strengths = []
    for conds, out in rules:
        s = np.ones(n)
        for var, term in conds:
            if var not in cache:
                cache[var] = membership(var, get(var))
            s = np.minimum(s, cache[var][term])
        agg[out] = np.maximum(agg[out], s)
        strengths.append(s)

    out_mf = np.array([fuzz.trimf(X_OUT, p) if k == "tri" else fuzz.trapmf(X_OUT, p) for k, p in OUT_MF])
    total = np.minimum(out_mf[:, None, :], agg[:, :, None]).max(axis=0)
    area = total.sum(axis=1)
    centroid = (total * X_OUT).sum(axis=1) / np.where(area == 0, 1, area)
    centroid = np.where(area == 0, 50, centroid)
    return centroid, strengths


def compute_fuzzy(df):
    sub = {}
    for name, rules in SYSTEMS:
        sub[name], _ = run_system(rules, lambda v: df[v].values)
    final, _ = run_system(FINAL_RULES, lambda v: sub[v])
    res = pd.DataFrame(sub, index=df.index)
    res["fuzzy_risk"] = final
    return res


def add_fuzzy(df):
    df = df.drop(columns=[c for c in FUZZY_COLS if c in df.columns])
    return df.join(compute_fuzzy(df))


def rule_text(conds, out, out_name):
    left = " И ".join(f"{v} = {TERMS[t]}" for v, t in conds)
    return f"ЕСЛИ {left} ТО {out_name} = {TERMS[out]}"


def rules_table():
    rows = []
    for name, rules in SYSTEMS:
        for i, (conds, out) in enumerate(rules, 1):
            rows.append({"level": 1, "system": name, "rule": i, "text": rule_text(conds, out, name)})
    for i, (conds, out) in enumerate(FINAL_RULES, 1):
        rows.append({"level": 2, "system": "fuzzy_risk", "rule": i, "text": rule_text(conds, out, "fuzzy_risk")})
    return pd.DataFrame(rows)


def trace_rules(project, min_strength=0.05):
    one = pd.DataFrame([project])
    rows = []
    sub = {}
    for name, rules in SYSTEMS:
        sub[name], strengths = run_system(rules, lambda v: one[v].values)
        for i, ((conds, out), s) in enumerate(zip(rules, strengths), 1):
            rows.append({"level": 1, "system": name, "rule": i, "strength": float(s[0]),
                         "text": rule_text(conds, out, name)})
    final, strengths = run_system(FINAL_RULES, lambda v: sub[v])
    for i, ((conds, out), s) in enumerate(zip(FINAL_RULES, strengths), 1):
        rows.append({"level": 2, "system": "fuzzy_risk", "rule": i, "strength": float(s[0]),
                     "text": rule_text(conds, out, "fuzzy_risk")})
    table = pd.DataFrame(rows)
    table = table[table["strength"] >= min_strength].sort_values(["level", "strength"], ascending=[True, False])
    values = {k: float(v[0]) for k, v in sub.items()}
    values["fuzzy_risk"] = float(final[0])
    return table.reset_index(drop=True), values


def draw_surface(df, x_name="profitability", y_name="payback_period"):
    xs = np.linspace(5, 150, 30)
    ys = np.linspace(1, 10, 30)
    gx, gy = np.meshgrid(xs, ys)
    base = df[NUM_COLS + DERIVED_COLS].median()
    grid = pd.DataFrame([base] * gx.size).reset_index(drop=True)
    grid[x_name] = gx.ravel()
    grid[y_name] = gy.ravel()
    z = compute_fuzzy(grid)["fuzzy_risk"].values.reshape(gx.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(gx, gy, z, cmap="viridis")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel("fuzzy_risk")
    ax.set_title("Поверхность fuzzy_risk")
    path = os.path.join(FIG_DIR, "fuzzy_surface.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path, (xs, ys, z)


def sensitivity(df):
    inputs = ["profitability", "payback_period", "market_volatility", "competition_level",
              "regulatory_risk", "team_experience", "technology_readiness", "strategic_fit"]
    base_mean = compute_fuzzy(df)["fuzzy_risk"].mean()
    rows = []
    for col in inputs:
        res = {}
        for k in (0.8, 1.2):
            tmp = df.copy()
            tmp[col] = tmp[col] * k
            res[k] = compute_fuzzy(tmp)["fuzzy_risk"].mean() - base_mean
        rows.append({"input": col, "minus_20": res[0.8], "plus_20": res[1.2],
                     "range": abs(res[1.2] - res[0.8])})
    table = pd.DataFrame(rows).sort_values("range")

    plt.figure(figsize=(8, 5))
    y = np.arange(len(table))
    plt.barh(y - 0.2, table["minus_20"], height=0.4, label="-20%")
    plt.barh(y + 0.2, table["plus_20"], height=0.4, label="+20%")
    plt.yticks(y, table["input"])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("изменение среднего fuzzy_risk")
    plt.title("Чувствительность (±20% по каждому входу)")
    plt.legend()
    path = os.path.join(FIG_DIR, "fuzzy_sensitivity.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    table.to_csv(os.path.join(RES_DIR, "sensitivity.csv"), index=False, encoding="utf-8-sig")
    return path, table.sort_values("range", ascending=False)


def fuzzy_and_save(df):
    df = add_fuzzy(df)
    df.to_csv(FINAL_PATH, index=False, encoding="utf-8-sig")
    return df
