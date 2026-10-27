import datetime

import matplotlib
import pandas as pd
from openpyxl.styles import Font
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import *
from .fuzzy_model import rules_table
from .whatif import predict, prepare, what_if


def recommend(pred_class, fin, mkt, org):
    weakest = max([("финансовая часть (рентабельность, окупаемость)", fin),
                   ("рыночные и регуляторные риски", mkt),
                   ("готовность команды и технологии", org)], key=lambda x: x[1])[0]
    if pred_class == 0:
        return "Рекомендуется к одобрению. Риск низкий."
    if pred_class == 1:
        return f"Одобрить с условиями. Слабое место: {weakest}."
    return f"Отказ или доработка проекта. Главная причина: {weakest}."


def make_predictions(bundle, df):
    from .ml_models import get_split
    X_train, X_test, y_train, y_test = get_split(df, bundle["with_fuzzy"])
    test = df.loc[X_test.index].copy()
    pred, proba = predict(bundle, test)
    out = test[["project_id", "sector", "fuzzy_risk", "fin_risk", "mkt_risk", "org_risk", "risk_class"]].copy()
    out["pred_class"] = pred
    out["p_low"] = proba[:, 0]
    out["p_mid"] = proba[:, 1]
    out["p_high"] = proba[:, 2]
    out["recommendation"] = [recommend(p, f, m, o) for p, f, m, o in
                             zip(out["pred_class"], out["fin_risk"], out["mkt_risk"], out["org_risk"])]
    return out.reset_index(drop=True)


def make_excel(df, bundle, metrics, path=XLSX_PATH):
    preds = make_predictions(bundle, df)
    rules = rules_table()
    recs = preds[["project_id", "sector", "pred_class", "fuzzy_risk", "recommendation"]]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
        rules.to_excel(writer, sheet_name="Rules", index=False)
        metrics.to_excel(writer, sheet_name="Metrics", index=False)
        preds.to_excel(writer, sheet_name="Predictions", index=False)
        recs.to_excel(writer, sheet_name="Recommendations", index=False)
        for ws in writer.book.worksheets:
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in list(col)[:50])
                ws.column_dimensions[col[0].column_letter].width = min(width + 2, 60)
    return path


def register_fonts():
    folder = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(folder, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(folder, "DejaVuSans-Bold.ttf")))


def table_from_df(df, font_size=7, col_widths=None):
    data = [list(df.columns)] + df.values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ]))
    return t


def add_image(story, name, width=16):
    path = os.path.join(FIG_DIR, name)
    if os.path.exists(path):
        from reportlab.lib.utils import ImageReader
        w, h = ImageReader(path).getSize()
        story.append(Image(path, width=width * cm, height=width * cm * h / w))
        story.append(Spacer(1, 0.3 * cm))


def make_pdf(df, bundle, metrics, whatif_example, business_text, path=PDF_PATH):
    register_fonts()
    h1 = ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=15, spaceAfter=8)
    h2 = ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=12, spaceAfter=6, spaceBefore=6)
    text = ParagraphStyle("text", fontName="DejaVu", fontSize=9, leading=13)
    title = ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=20, leading=26, alignment=1)

    story = [Spacer(1, 6 * cm),
             Paragraph("Интеллектуальная система оценки инвестиционных проектов", title),
             Spacer(1, 1 * cm),
             Paragraph("Нечеткая логика и нейронные сети", ParagraphStyle("c", parent=text, alignment=1, fontSize=12)),
             Spacer(1, 1 * cm),
             Paragraph(f"Дата: {datetime.date.today():%d.%m.%Y}", ParagraphStyle("c2", parent=text, alignment=1)),
             PageBreak()]

    story.append(Paragraph("1. Данные", h1))
    shares = df["risk_class"].value_counts(normalize=True).sort_index().round(3)
    story.append(Paragraph(f"Проектов: {len(df)}. Доли классов (0/1/2): "
                           f"{shares[0]:.0%} / {shares[1]:.0%} / {shares[2]:.0%}.", text))
    story.append(Spacer(1, 0.3 * cm))
    show_cols = ["project_id", "sector", "project_cost", "expected_profit", "payback_period",
                 "profitability", "fuzzy_risk", "risk_class"]
    story.append(table_from_df(df[show_cols].head(12).round(1), 7))
    story.append(Spacer(1, 0.3 * cm))
    add_image(story, "eda_corr.png", 13)
    add_image(story, "eda_boxplots.png", 15)
    story.append(PageBreak())

    story.append(Paragraph("2. Нечеткая модель", h1))
    story.append(Paragraph("Три подиндекса (финансовый, рыночный, организационный) считаются отдельными "
                           "нечеткими системами, затем из них собирается итоговый fuzzy_risk (0-100). "
                           "Примеры правил:", text))
    rules = rules_table()
    sample_rules = pd.concat([rules[rules["system"] == s].head(3) for s in
                              ["fin_risk", "mkt_risk", "org_risk", "fuzzy_risk"]])
    for r in sample_rules.itertuples():
        story.append(Paragraph(f"[{r.system} #{r.rule}] {r.text}", text))
    story.append(Paragraph(f"Всего правил: {len(rules)} (первый уровень {int((rules['level'] == 1).sum())}, "
                           f"второй {int((rules['level'] == 2).sum())}). Полный список - в Excel, лист Rules.", text))
    story.append(Spacer(1, 0.3 * cm))
    add_image(story, "fuzzy_surface.png", 12)
    add_image(story, "fuzzy_sensitivity.png", 13)
    story.append(PageBreak())

    story.append(Paragraph("3. Метрики моделей", h1))
    m = metrics.copy()
    for c in m.columns[2:]:
        m[c] = m[c].round(3)
    story.append(table_from_df(m, 6))
    story.append(Spacer(1, 0.3 * cm))
    add_image(story, "confusion_matrices.png", 16)
    story.append(PageBreak())

    story.append(Paragraph("4. Объяснение модели (SHAP)", h1))
    add_image(story, "shap_summary.png", 12)
    add_image(story, "shap_waterfall.png", 14)
    add_image(story, "shap_classes.png", 12)
    story.append(PageBreak())

    story.append(Paragraph("5. Пример what-if", h1))
    story.append(Paragraph(whatif_example, text))
    story.append(Spacer(1, 0.3 * cm))
    add_image(story, "monte_carlo.png", 16)
    story.append(Paragraph("6. Бизнес-часть", h1))
    for line in business_text.split("\n"):
        story.append(Paragraph(line, text))
    add_image(story, "business_loss.png", 13)

    story.append(Paragraph("7. Выводы", h1))
    best = metrics.sort_values(["f1_macro", "mcc"], ascending=False).iloc[0]
    story.append(Paragraph(
        f"Лучшая модель по F1-macro: {best['model']} ({best['variant']}), F1 = {best['f1_macro']:.3f}, "
        f"MCC = {best['mcc']:.3f}. Нечеткая модель дает объяснимый индекс риска и показывает, "
        f"какие правила сработали. SHAP показывает вклад признаков в каждое решение, "
        f"what-if и Монте-Карло позволяют проверить сценарии и устойчивость оценки.", text))

    SimpleDocTemplate(path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm).build(story)
    return path


def whatif_text(bundle, project):
    changes = {"payback_period": max(1, project["payback_period"] - 2),
               "market_volatility": min(100, project["market_volatility"] + 20)}
    before, after = what_if(bundle, project, changes)
    return (f"Проект: сектор {project['sector']}, окупаемость {project['payback_period']} лет, "
            f"волатильность {project['market_volatility']}. Изменения: окупаемость -2 года, волатильность +20. "
            f"fuzzy_risk: {before['fuzzy_risk']:.1f} -> {after['fuzzy_risk']:.1f}. "
            f"Класс: {before['class']} -> {after['class']}. "
            f"Вероятности до: {[round(p, 2) for p in before['proba']]}, "
            f"после: {[round(p, 2) for p in after['proba']]}.")
