# core/report.py
# 把报警事件 CSV 导出成带"明细 + 汇总 + 图表"的 Excel 工作簿。
# 只依赖 openpyxl（打包 exe 也能用，不加 pandas 那套重依赖）。
from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill

HEADER_FILL = PatternFill("solid", fgColor="1E88E5")
HEADER_FONT = Font(color="FFFFFF", bold=True)
CENTER = Alignment(horizontal="center", vertical="center")


def _read_events(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            ts = (r.get("时间") or "").strip()
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                dt = None
            try:
                conf = float(r.get("最高置信度") or 0)
            except ValueError:
                conf = 0.0
            try:
                cnt = int(float(r.get("命中数量") or 0))
            except ValueError:
                cnt = 0
            rows.append({
                "dt": dt, "time": ts, "cls": (r.get("类别") or "").strip(),
                "conf": conf, "count": cnt, "img": (r.get("抓拍图片") or "").strip(),
            })
    return rows


def _style_header(ws, ncols: int) -> None:
    for c in range(1, ncols + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER


def export_events_excel(csv_path, out_path) -> Path:
    """导出工作簿：Sheet1 明细 + Sheet2 汇总与图表。返回输出路径。"""
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    events = _read_events(csv_path) if csv_path.exists() else []

    wb = openpyxl.Workbook()

    # ---------------- Sheet 1：事件明细 ----------------
    ws = wb.active
    ws.title = "事件明细"
    headers = ["时间", "类别", "最高置信度", "命中数量", "抓拍图片"]
    ws.append(headers)
    for ev in events:
        row = [ev["time"], ev["cls"], ev["conf"], ev["count"], ev["img"]]
        ws.append(row)
        # 抓拍路径做成可点链接（只保留文件名显示，路径放超链接）
        if ev["img"]:
            cell = ws.cell(row=ws.max_row, column=5)
            cell.hyperlink = ev["img"]
            cell.style = "Hyperlink"
    _style_header(ws, len(headers))
    for col, width in zip("ABCDE", (20, 10, 12, 10, 46)):
        ws.column_dimensions[col].width = width

    # ---------------- Sheet 2：统计汇总 ----------------
    ws2 = wb.create_sheet("统计汇总")
    ws2["A1"] = "报警事件汇总"
    ws2["A1"].font = Font(bold=True, size=14)

    total = len(events)
    ws2["A3"] = "事件总数"; ws2["B3"] = total
    ws2["A4"] = "类别分布"
    by_cls = Counter(e["cls"] for e in events)
    r = 5
    for cls, n in sorted(by_cls.items()):
        ws2.cell(row=r, column=1, value=cls)
        ws2.cell(row=r, column=2, value=n)
        r += 1

    # 时间范围
    valid = [e["dt"] for e in events if e["dt"]]
    if valid:
        ws2["A3"].offset(0, 2).value = "时间范围"
        ws2["A4"].offset(0, 2).value = f"{min(valid):%Y-%m-%d %H:%M} ~ {max(valid):%Y-%m-%d %H:%M}"

    # 按小时分布（图表数据源）
    hourly = Counter(e["dt"].strftime("%m-%d %H:00") for e in events if e["dt"])
    chart_title_cell = ws2.cell(row=r + 2, column=1, value="按小时分布")
    chart_title_cell.font = Font(bold=True)
    hr_start = r + 3
    for i, (h, n) in enumerate(sorted(hourly.items())):
        ws2.cell(row=hr_start + i, column=1, value=h)
        ws2.cell(row=hr_start + i, column=2, value=n)

    if hourly:
        chart = BarChart()
        chart.type = "col"
        chart.title = "每小时报警次数"
        chart.y_axis.title = "次数"
        chart.x_axis.title = "小时"
        data = Reference(ws2, min_col=2, min_row=hr_start, max_row=hr_start + len(hourly) - 1)
        cats = Reference(ws2, min_col=1, min_row=hr_start, max_row=hr_start + len(hourly) - 1)
        chart.add_data(data, titles_from_data=False)
        chart.set_categories(cats)
        chart.height = 8
        chart.width = 18
        ws2.add_chart(chart, f"D{r + 2}")

    for col, width in zip("ABCDE", (18, 14, 20, 20, 20)):
        ws2.column_dimensions[col].width = width

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_path))
    return out_path
