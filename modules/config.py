import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
RES_DIR = os.path.join(BASE_DIR, "../results")
FIG_DIR = os.path.join(RES_DIR, "figures")
MODEL_DIR = os.path.join(BASE_DIR, "../models")

RAW_PATH = os.path.join(DATA_DIR, "data_raw.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "data_clean.csv")
FINAL_PATH = os.path.join(DATA_DIR, "data_final.csv")
COMPARISON_PATH = os.path.join(RES_DIR, "model_comparison.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
ALL_MODELS_PATH = os.path.join(MODEL_DIR, "all_models.joblib")
PDF_PATH = os.path.join(RES_DIR, "report.pdf")
XLSX_PATH = os.path.join(RES_DIR, "report.xlsx")

for folder in (DATA_DIR, RES_DIR, FIG_DIR, MODEL_DIR):
    os.makedirs(folder, exist_ok=True)

SEED = 42
N_ROWS = 2500
N_TRIALS = 30
N_MC = 500

SECTORS = ["Производство", "ИТ", "Логистика", "Энергетика"]

RANGES = {
    "project_cost": (1, 100),
    "expected_profit": (0.5, 150),
    "payback_period": (1, 10),
    "team_experience": (0, 20),
    "market_volatility": (0, 100),
    "technology_readiness": (1, 9),
    "competition_level": (1, 10),
    "strategic_fit": (1, 10),
    "regulatory_risk": (0, 100),
}

NUM_COLS = list(RANGES.keys())
INT_COLS = ["technology_readiness", "competition_level", "strategic_fit"]
DERIVED_COLS = ["profitability", "roi_index", "risk_density", "capability_score"]
FUZZY_COLS = ["fin_risk", "mkt_risk", "org_risk", "fuzzy_risk"]

CLASS_NAMES = {0: "низкий риск", 1: "средний риск", 2: "высокий риск"}
