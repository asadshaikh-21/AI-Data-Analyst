"""Turn a query result + (LLM or heuristic) chart hint into a JSON chart spec for Chart.js."""
from __future__ import annotations

import json

import pandas as pd

CHART_TYPES = {"bar", "line", "pie", "scatter"}
TIME_HINTS = ("date", "month", "year", "week", "day", "quarter", "time")


def df_to_table(df: pd.DataFrame, limit: int = 200) -> dict:
    data = json.loads(df.head(limit).to_json(orient="split", date_format="iso"))
    rows = [[v[:10] if isinstance(v, str) and v.endswith("T00:00:00.000") else v for v in r] for r in data["data"]]
    return {"columns": data["columns"], "rows": rows, "total_rows": len(df)}


def _num(v):
    return None if pd.isna(v) else float(v)


def build_chart(df: pd.DataFrame, spec: dict | None) -> dict | None:
    if df is None or df.empty or df.shape[1] < 2:
        return None
    spec = spec or {}
    if str(spec.get("type", "")).lower() == "none":
        return None
    cols = list(df.columns)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    ctype = str(spec.get("type", "")).lower()
    x = spec.get("x") if spec.get("x") in cols else None
    ys = spec.get("y") or []
    ys = [ys] if isinstance(ys, str) else ys
    ys = [y for y in ys if y in numeric and y != x]
    if x is None:
        non_num = [c for c in cols if c not in numeric]
        x = non_num[0] if non_num else cols[0]
    if not ys:
        ys = [c for c in numeric if c != x][:3]
    if not ys:
        return None
    if ctype not in CHART_TYPES:
        is_time = pd.api.types.is_datetime64_any_dtype(df[x]) or any(h in x.lower() for h in TIME_HINTS)
        ctype = "line" if is_time else "bar"
    df = df.head(200)
    title = spec.get("title") or f"{', '.join(ys)} by {x}"
    if ctype == "scatter":
        if x not in numeric:
            ctype = "bar"
        else:
            pts = [{"x": _num(a), "y": _num(b)} for a, b in zip(df[x], df[ys[0]])]
            return {"type": "scatter", "title": title, "x_label": x, "y_labels": [ys[0]],
                    "series": [{"name": ys[0], "data": pts}]}
    labels = [str(v)[:10] if pd.api.types.is_datetime64_any_dtype(df[x]) else str(v) for v in df[x]]
    if ctype == "pie":
        ys = ys[:1]
    return {"type": ctype, "title": title, "x_label": x, "y_labels": ys, "labels": labels,
            "series": [{"name": y, "data": [_num(v) for v in df[y]]} for y in ys]}
