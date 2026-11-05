"""Heuristic planner used when no LLM key is configured (demo / CI / LLM outage).

It understands the common question shapes (top-N, lowest, trend, anomalies, summary,
"generate SQL") against whatever schema is loaded. It is intentionally simple: the LLM path
is the primary product; this keeps the app usable and testable without network access.
"""
from __future__ import annotations

import re

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10}
MEASURE_PREF = ["revenue", "sales", "amount", "total", "profit", "price", "quantity", "qty"]


def _stem(w: str) -> str:
    if w.endswith("ies"):
        return w[:-3] + "y"
    return w[:-1] if w.endswith("s") and not w.endswith("ss") else w


def _tokens(q: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z]+", q.lower())]


def _top_n(q: str, default: int) -> int:
    m = re.search(r"\b(\d+)\b", q)
    if m:
        return max(1, min(50, int(m.group(1))))
    for w, n in NUM_WORDS.items():
        if re.search(rf"\b{w}\b", q.lower()):
            return n
    return default


def _measure(session, toks):
    best = None
    for ds in session.datasets.values():
        for c in ds.columns:
            if c["kind"] != "numeric" or c["name"].endswith("id"):
                continue
            words = c["name"].split("_")
            score = 10 if any(w in toks for w in words) else 0
            for i, p in enumerate(MEASURE_PREF):
                if p in c["name"]:
                    score += len(MEASURE_PREF) - i
                    break
            if best is None or score > best[0]:
                best = (score, ds.name, c["name"])
    return best[1:] if best else (None, None)


def _dim(session, toks, mtable):
    cands = []
    for ds in session.datasets.values():
        for c in ds.columns:
            if c["kind"] != "text":
                continue
            score = sum(1 for w in c["name"].split("_") if w in toks)
            if not score:
                continue
            score += (0.5 if c["name"].endswith("name") else 0) + (0.2 if ds.name == mtable else 0)
            score -= 0.3 if c["name"].endswith("id") else 0
            cands.append((score, ds.name, c["name"]))
    return max(cands)[1:] if cands else (None, None)


def _shared_col(session, a, b):
    ca = {c["name"] for c in session.datasets[a].columns}
    cb = {c["name"] for c in session.datasets[b].columns}
    both = sorted(ca & cb, key=lambda c: (not c.endswith("id"), c))
    return both[0] if both else None


def plan(session, question: str) -> dict:
    q = question.lower()
    toks = _tokens(question)
    if not session.datasets:
        return {"intent": "chat", "answer": "Upload a CSV first (or load the sample data), then ask me anything about it.",
                "reasoning": "No datasets are loaded."}
    if re.search(r"anomal|outlier|unusual|suspicious|irregular", q):
        return {"intent": "anomaly", "reasoning": "Scanning numeric columns with IQR fences and robust z-scores (MAD).",
                "sql": None, "chart": {"type": "none"}}
    if re.search(r"\bsql\b|\bpandas\b|generate code", q):
        prev = next((h for h in reversed(session.history) if h.get("sql")), None)
        if prev:
            return {"intent": "code", "reasoning": "Re-using the SQL from the previous analysis.",
                    "sql": prev["sql"], "pandas_code": prev.get("pandas_code"), "chart": prev.get("chart")}
        return {"intent": "chat", "answer": "Ask an analysis question first; then I can show the SQL behind it.",
                "reasoning": "No previous analysis to generate code for."}

    mtable, measure = _measure(session, toks)
    if not measure:
        t = next(iter(session.datasets))
        return {"intent": "summary", "reasoning": "No numeric measure found, so I summarised row counts.",
                "sql": f'SELECT COUNT(*) AS row_count FROM "{t}"', "chart": {"type": "none"}}

    agg_fn = "AVG" if re.search(r"average|avg|mean", q) else "SUM"
    if re.search(r"how many|count|number of", q):
        agg = 'COUNT(*)'
        alias = "count"
    else:
        agg = f'ROUND({agg_fn}(m."{measure}"), 2)'
        alias = f'{"avg" if agg_fn == "AVG" else "total"}_{measure}'

    if re.search(r"trend|monthly|per month|by month|over time|month", q):
        dcol = next((c["name"] for c in session.datasets[mtable].columns if c["kind"] == "datetime"), None)
        if dcol:
            sql = (f"SELECT strftime(date_trunc('month', m.\"{dcol}\"), '%Y-%m') AS month, {agg} AS {alias} "
                   f'FROM "{mtable}" m GROUP BY 1 ORDER BY 1')
            return {"intent": "query", "reasoning": f"Monthly {alias} computed with date_trunc on {dcol}.",
                    "sql": sql, "chart": {"type": "line", "x": "month", "y": [alias], "title": f"Monthly {alias}"},
                    "pandas_code": (f"df.groupby(df['{dcol}'].dt.to_period('M'))['{measure}']"
                                    f".{'mean' if agg_fn == 'AVG' else 'sum'}().reset_index()")}

    dtable, dim = _dim(session, toks, mtable)
    if dim:
        asc = bool(re.search(r"underperform|lowest|worst|bottom|least|weak|poor", q))
        n = _top_n(q, 5 if not asc else 5)
        order = "ASC" if asc else "DESC"
        if dtable == mtable:
            frm, d = f'"{mtable}" m', f'm."{dim}"'
        else:
            key = _shared_col(session, mtable, dtable)
            if not key:
                frm, d = f'"{mtable}" m', None
            else:
                frm = f'"{mtable}" m JOIN "{dtable}" d ON m."{key}" = d."{key}"'
                d = f'd."{dim}"'
        if d:
            sql = f"SELECT {d} AS {dim}, {agg} AS {alias} FROM {frm} GROUP BY 1 ORDER BY 2 {order} LIMIT {n}"
            ctype = "pie" if re.search(r"\bpie\b|share|proportion", q) else "bar"
            return {"intent": "query",
                    "reasoning": f"Grouped {measure} by {dim}, ranked {'ascending' if asc else 'descending'}, kept {n}.",
                    "sql": sql, "chart": {"type": ctype, "x": dim, "y": [alias], "title": f"{alias} by {dim}"},
                    "pandas_code": f"df.groupby('{dim}')['{measure}'].{'mean' if agg_fn == 'AVG' else 'sum'}()"
                                   f".sort_values(ascending={asc}).head({n})"}
    return {"intent": "summary", "reasoning": f"Computed overall {alias}.",
            "sql": f'SELECT {agg} AS {alias} FROM "{mtable}" m', "chart": {"type": "none"}}


def narrative(question: str, df, total_rows: int) -> str:
    if df is None or df.empty:
        return "The query returned no rows."
    first = df.iloc[0]
    cols = list(df.columns)
    if len(cols) >= 2 and df.shape[0] >= 1:
        out = f"**{first[cols[0]]}** is first in this ranking with **{first[cols[1]]:,}**." if isinstance(first[cols[1]], (int, float)) \
            else f"**{first[cols[0]]}** is first in this ranking."
        if df.shape[0] > 1 and isinstance(df.iloc[-1][cols[1]], (int, float)):
            last = df.iloc[-1]
            out += f"\n- Lowest shown: {last[cols[0]]} at {last[cols[1]]:,}"
        out += f"\n- {total_rows} row(s) returned"
        return out + "\n\n_Offline mode: set an LLM API key for richer business insights._"
    return f"Result: **{first[cols[0]]}**"


def anomaly_narrative(anoms: list[dict]) -> str:
    if not anoms:
        return "No statistical anomalies found: every numeric column sits inside its IQR fences."
    by_col: dict[str, int] = {}
    for a in anoms:
        by_col[f"{a['table']}.{a['column']}"] = by_col.get(f"{a['table']}.{a['column']}", 0) + 1
    top = anoms[0]
    lines = [f"Flagged **{len(anoms)}** suspicious value(s) across {len(by_col)} column(s)."]
    lines += [f"- `{k}`: {v} value(s)" for k, v in by_col.items()]
    lines.append(f"Most extreme: `{top['table']}.{top['column']}` = {top['value']:,.2f} (row {top['row']}). "
                 f"Each flag needs BOTH an IQR-fence breach and a robust z-score above 3.5.")
    return "\n".join(lines)
