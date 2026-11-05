"""Explainable statistical anomaly detection.

A value is flagged only if it breaches the IQR fences AND its robust z-score (MAD-based) exceeds a
threshold, which keeps false positives low. If a low-cardinality text column explains most of a
numeric column's variance (e.g. unit_price varies by product), values are judged *within* that group
so legitimately expensive items are not flagged.
"""
from __future__ import annotations

import pandas as pd


def _is_id_like(name: str, s: pd.Series) -> bool:
    return name.endswith("id") or (
        pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing and s.nunique() == len(s))


def _is_text(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)
                or pd.api.types.is_bool_dtype(s))


def _best_group(df: pd.DataFrame, col: str, candidates: list[str]) -> str | None:
    # Winsorise first so a single extreme value cannot hide the grouping structure.
    s = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))
    d = df.assign(**{col: s})
    total = ((s - s.mean()) ** 2).sum()
    if not total:
        return None
    best, best_eta = None, 0.5
    for g in candidates:
        within = ((s - d.groupby(g)[col].transform("mean")) ** 2).sum()
        eta = 1 - within / total
        if eta > best_eta:
            best, best_eta = g, eta
    return best


def _flag(s: pd.Series, z_thresh: float, iqr_k: float, scope: str = "") -> list[tuple]:
    """Return [(index, value, score, reason)] for one series."""
    s = s.dropna()
    if len(s) < 8 or s.nunique() < 5:
        return []
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    med = s.median()
    mad = (s - med).abs().median()
    out = []
    for idx, val in s[(s < lo) | (s > hi)].items():
        if mad > 0:
            z = 0.6745 * (val - med) / mad
            if abs(z) < z_thresh:
                continue
            score = abs(z)
            why = (f"{val:,.2f} is {abs(z):.1f} robust z-scores {'above' if z > 0 else 'below'} the "
                   f"{scope}median ({med:,.2f}) and outside the IQR fence [{lo:,.2f}, {hi:,.2f}]")
        else:
            score = abs(val - med) / (iqr or 1)
            why = f"{val:,.2f} is outside the {scope}IQR fence [{lo:,.2f}, {hi:,.2f}] (median {med:,.2f})"
        out.append((idx, float(val), float(score), why))
    return out


def detect_anomalies(df: pd.DataFrame, table: str, per_column: int = 5,
                     z_thresh: float = 3.5, iqr_k: float = 1.5) -> list[dict]:
    text_cols = [c for c in df.columns if _is_text(df[c])]
    groupers = [c for c in text_cols if not c.endswith("id") and 2 <= df[c].nunique() <= 30
                and len(df) / df[c].nunique() >= 8]
    ctx_cols = text_cols[:3]
    found: list[dict] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s) or _is_id_like(col, s):
            continue
        group = _best_group(df, col, groupers) if groupers and s.notna().all() else None
        hits = []
        if group:
            for gval, gdf in df.groupby(group):
                hits += _flag(gdf[col], z_thresh, iqr_k, scope=f"{group}='{gval}' ")
        else:
            hits = _flag(s, z_thresh, iqr_k)
        rows = [{"table": table, "column": col, "row": int(df.index.get_loc(i)), "value": v,
                 "score": round(sc, 2), "reason": why, "context": {c: str(df.at[i, c]) for c in ctx_cols}}
                for i, v, sc, why in hits]
        rows.sort(key=lambda r: -r["score"])
        found.extend(rows[:per_column])
    found.sort(key=lambda r: -r["score"])
    return found


def scan(datasets: dict, only: list[str] | None = None, limit: int = 25) -> list[dict]:
    out: list[dict] = []
    for name, ds in datasets.items():
        if only and name not in only:
            continue
        out.extend(detect_anomalies(ds.df, name))
    out.sort(key=lambda r: -r["score"])
    return out[:limit]
