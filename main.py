import sys

import pandas as pd

from modules import whatif, business, data_gen, explain, cleaning, reports, eda, ml_models, fuzzy_model
from modules.config import *

FIELDS = [
    ("project_cost", "Стоимость проекта, млн руб.", 20),
    ("expected_profit", "Ожидаемая прибыль, млн руб.", 15),
    ("payback_period", "Срок окупаемости, лет", 5),
    ("team_experience", "Опыт команды, лет", 10),
    ("market_volatility", "Волатильность рынка (0-100)", 45),
    ("technology_readiness", "Готовность технологии (1-9)", 6),
    ("competition_level", "Уровень конкуренции (1-10)", 5),
    ("strategic_fit", "Соответствие стратегии (1-10)", 6),
    ("regulatory_risk", "Регуляторный риск (0-100)", 35),
]


def need(path):
    if not os.path.exists(path):
        print("Нет файла", path, "- сначала выполните предыдущие пункты меню.")
        return False
    return True


def ask_number(text, default, lo, hi):
    while True:
        s = input(f"{text} [{default}]: ").strip()
        if s == "":
            return default
        try:
            value = float(s.replace(",", "."))
        except ValueError:
            print("Нужно число.")
            continue
        if lo <= value <= hi:
            return value
        print(f"Допустимый диапазон: {lo} - {hi}")


def ask_project():
    project = {}
    print("Сектор:", ", ".join(f"{i + 1}={s}" for i, s in enumerate(SECTORS)))
    n = int(ask_number("Номер сектора", 1, 1, len(SECTORS)))
    project["sector"] = SECTORS[n - 1]
    for key, label, default in FIELDS:
        lo, hi = RANGES[key]
        value = ask_number(label, default, lo, hi)
        project[key] = int(value) if key in INT_COLS else value
    return project


def print_result(r):
    print(f"  финансовый индекс: {r['fin_risk']:.1f}")
    print(f"  рыночный индекс:   {r['mkt_risk']:.1f}")
    print(f"  организационный:   {r['org_risk']:.1f}")
    print(f"  fuzzy_risk:        {r['fuzzy_risk']:.1f}")
    print(f"  класс: {r['class']} ({CLASS_NAMES[r['class']]})")
    print("  вероятности:", [round(p, 3) for p in r["proba"]])


def pick_project(df):
    pid = input("project_id проекта (Enter - случайный): ").strip()
    if pid.isdigit() and int(pid) in set(df["project_id"]):
        row = df[df["project_id"] == int(pid)].iloc[0]
    else:
        row = df.sample(1, random_state=None).iloc[0]
    print("Выбран проект", int(row["project_id"]), "сектор", row["sector"])
    project = {"sector": row["sector"]}
    for col in NUM_COLS:
        project[col] = row[col]
    return project


def step_generate():
    raw = data_gen.generate_and_save()
    print("Сохранено:", RAW_PATH, raw.shape)
    print("Пропуски:", raw[NUM_COLS].isna().sum()[lambda s: s > 0].to_dict())
    print("Доли классов:", raw["risk_class"].value_counts(normalize=True).sort_index().round(2).to_dict())


def step_clean():
    if not need(RAW_PATH):
        return
    df, report = cleaning.clean_and_save()
    print(report.to_string(index=False))
    print("Пропуски заполнены медианой: данные несимметричны и есть выбросы, медиана к ним устойчива.")
    print("Выбросы ограничены по границам IQR (1.5 * IQR).")
    print("Сохранено:", CLEAN_PATH)


def step_eda():
    if not need(CLEAN_PATH):
        return
    df = cleaning.read_csv(CLEAN_PATH)
    paths, top = eda.run_eda(df)
    print("Графики:")
    for p in paths:
        print(" ", p)
    print("Корреляция признаков с risk_class:")
    print(top.round(3).to_string())


def step_fuzzy():
    if not need(CLEAN_PATH):
        return
    df = cleaning.read_csv(CLEAN_PATH)
    df = fuzzy_model.fuzzy_and_save(df)
    print("fuzzy_risk по классам (среднее):", df.groupby("risk_class")["fuzzy_risk"].mean().round(1).to_dict())
    print("Корреляция fuzzy_risk и risk_class:", round(df["fuzzy_risk"].corr(df["risk_class"]), 3))
    path, _ = fuzzy_model.draw_surface(df)
    print("3D поверхность:", path)
    path, table = fuzzy_model.sensitivity(df)
    print("Чувствительность:", path)
    print(table.round(2).to_string(index=False))
    rules = fuzzy_model.rules_table()
    print("Правил 1 уровня:", int((rules["level"] == 1).sum()), "2 уровня:", int((rules["level"] == 2).sum()))


def step_train():
    if not need(FINAL_PATH):
        return
    df = cleaning.read_csv(FINAL_PATH)
    table, models, best = ml_models.train_all(df, N_TRIALS)
    print(table.round(3).to_string(index=False))
    print("Лучшая модель:", best)
    print("Сохранено:", COMPARISON_PATH, BEST_MODEL_PATH)


def step_table():
    if not need(COMPARISON_PATH):
        return
    print(pd.read_csv(COMPARISON_PATH, encoding="utf-8-sig").round(3).to_string(index=False))


def step_shap():
    if not (need(FINAL_PATH) and need(BEST_MODEL_PATH)):
        return
    df = cleaning.read_csv(FINAL_PATH)
    bundle = ml_models.load_best()
    print("Считаю SHAP для модели", bundle["name"], bundle["variant"], "...")
    paths, importance, project_id = explain.make_shap_plots(bundle, df)
    for p in paths:
        print(" ", p)
    print("Топ-5 признаков:")
    print(importance.head(5).round(3).to_string())
    row = df[df["project_id"] == project_id].iloc[0]
    table, values = fuzzy_model.trace_rules(row.to_dict())
    print(f"Трассировка правил для проекта {project_id}:")
    print(table[["level", "system", "rule", "strength", "text"]].to_string(index=False))
    print("Индексы:", {k: round(v, 1) for k, v in values.items()})


def step_whatif():
    if not (need(FINAL_PATH) and need(BEST_MODEL_PATH)):
        return
    df = cleaning.read_csv(FINAL_PATH)
    bundle = ml_models.load_best()
    project = pick_project(df)
    changes = {}
    for i in range(2):
        print("Какой параметр менять? " + ", ".join(f"{j + 1}={f[0]}" for j, f in enumerate(FIELDS)) + ", 0=хватит")
        n = int(ask_number("Номер", 0, 0, len(FIELDS)))
        if n == 0:
            break
        key = FIELDS[n - 1][0]
        lo, hi = RANGES[key]
        value = ask_number(f"Новое значение {key} (сейчас {project[key]})", project[key], lo, hi)
        changes[key] = int(value) if key in INT_COLS else value
    if not changes:
        print("Ничего не изменено.")
        return
    before, after = whatif.what_if(bundle, project, changes)
    print("ДО:")
    print_result(before)
    print("ПОСЛЕ:")
    print_result(after)


def step_monte():
    if not (need(FINAL_PATH) and need(BEST_MODEL_PATH)):
        return
    df = cleaning.read_csv(FINAL_PATH)
    bundle = ml_models.load_best()
    project = pick_project(df)
    summary, runs = whatif.monte_carlo(bundle, project, N_MC)
    path = whatif.draw_monte_carlo(summary, runs)
    print(f"fuzzy_risk: среднее {summary['fuzzy_mean']:.1f}, std {summary['fuzzy_std']:.1f}, "
          f"5-95%: {summary['fuzzy_p5']:.1f} - {summary['fuzzy_p95']:.1f}")
    print("Доли классов:", {k: round(v, 3) for k, v in summary["class_shares"].items()})
    print(f"Устойчивость класса: {summary['stability']:.1%}")
    print("График:", path)


def step_business():
    if not (need(FINAL_PATH) and need(ALL_MODELS_PATH)):
        return
    df = cleaning.read_csv(FINAL_PATH)
    table, info = business.run_business(df, ml_models.load_all())
    print("Матрица потерь (строки - факт, столбцы - прогноз):")
    print(business.LOSS_MATRIX)
    print(table.round(3).to_string(index=False))
    print(info["text"])


def step_new_project():
    if not need(BEST_MODEL_PATH):
        return
    bundle = ml_models.load_best()
    project = ask_project()
    result = whatif.evaluate_project(bundle, project)
    print_result(result)
    print("  рекомендация:", reports.recommend(result["class"], result["fin_risk"],
                                               result["mkt_risk"], result["org_risk"]))
    table, _ = fuzzy_model.trace_rules(whatif.prepare(pd.DataFrame([project])).iloc[0].to_dict())
    print("Сработавшие правила:")
    print(table[["level", "system", "strength", "text"]].head(8).to_string(index=False))


def step_reports():
    if not (need(FINAL_PATH) and need(BEST_MODEL_PATH) and need(COMPARISON_PATH)):
        return
    df = cleaning.read_csv(FINAL_PATH)
    bundle = ml_models.load_best()
    metrics = pd.read_csv(COMPARISON_PATH, encoding="utf-8-sig")
    biz_path = os.path.join(RES_DIR, "business_conclusion.txt")
    biz = open(biz_path, encoding="utf-8").read() if os.path.exists(biz_path) else "Бизнес-часть не запускалась."
    example = df.iloc[0]
    project = {"sector": example["sector"]}
    for col in NUM_COLS:
        project[col] = example[col]
    text = reports.whatif_text(bundle, project)
    print("Excel:", reports.make_excel(df, bundle, metrics))
    print("PDF:", reports.make_pdf(df, bundle, metrics, text, biz))


MENU = [
    ("Сгенерировать данные", step_generate),
    ("Очистить данные", step_clean),
    ("Разведочный анализ (графики)", step_eda),
    ("Нечеткая модель (fuzzy_risk, 3D, чувствительность)", step_fuzzy),
    ("Обучить модели (6 штук, Optuna)", step_train),
    ("Показать таблицу метрик", step_table),
    ("SHAP и трассировка правил", step_shap),
    ("What-if сценарий", step_whatif),
    ("Монте-Карло (500 прогонов)", step_monte),
    ("Бизнес: матрица потерь и пороги", step_business),
    ("Новый проект (ввод и оценка)", step_new_project),
    ("Отчеты PDF и Excel", step_reports),
]


def run_all():
    for func in (step_generate, step_clean, step_eda, step_fuzzy, step_train,
                 step_shap, step_business, step_reports):
        print("\n----", func.__name__, "----")
        func()


def main():
    if "--all" in sys.argv:
        run_all()
        return
    while True:
        print("\n---- Оценка инвестиционных проектов ----")
        for i, (name, _) in enumerate(MENU, 1):
            print(f"{i:2}. {name}")
        print(" 0. Выход")
        choice = input("Пункт меню: ").strip()
        if choice == "0":
            break
        if choice.isdigit() and 1 <= int(choice) <= len(MENU):
            MENU[int(choice) - 1][1]()
        else:
            print("Нет такого пункта.")


if __name__ == "__main__":
    main()
