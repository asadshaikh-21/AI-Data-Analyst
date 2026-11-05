"""Orchestrator: plan -> validate -> execute (with self-repair) -> chart -> explain."""
from __future__ import annotations

import json
import logging
import re
import time

import duckdb

from . import anomaly, offline
from .charts import build_chart, df_to_table
from .data_store import Session
from .llm import LLM, LLMError
from .sql_guard import SQLValidationError, validate_sql

log = logging.getLogger("analyst.agent")
MAX_REPAIRS = 2

PLAN_SYSTEM = """You are an expert data analyst. The user's CSV files are loaded as DuckDB tables.
Respond with ONE JSON object and nothing else:
{"intent": "query|anomaly|summary|code|chat",
 "reasoning": "2-3 sentences on how you will answer and why (shown to the user)",
 "sql": "a single read-only DuckDB SELECT/WITH query, or null",
 "pandas_code": "equivalent pandas snippet (df = table dataframe), or null",
 "chart": {"type": "bar|line|pie|scatter|none", "x": "result column", "y": ["result columns"], "title": "..."},
 "anomaly_tables": ["table names"] or null,
 "answer": "only for intent=chat: the direct answer"}
Rules:
- Use ONLY the tables/columns listed below. DuckDB dialect (date_trunc, strftime, ROUND, etc.). Never write DDL/DML.
- Alias result columns clearly; ORDER and LIMIT rankings; JOIN tables on shared keys for multi-file questions.
- Time trends: aggregate with date_trunc and use a line chart. Rankings/comparisons: bar. Shares: pie.
- Follow-ups ("and by region?", "now only 2025") must reuse the earlier turns' context.
- "Generate SQL/pandas" for a previous analysis: intent=code, repeat that SQL and include pandas_code.
- Anomaly/outlier requests: intent=anomaly (the system runs statistical detection).
- If the data cannot answer the question, intent=chat and say what is missing. Never invent numbers.
"""

ANSWER_SYSTEM = """You are a senior business analyst. Using ONLY the numbers in the provided result, answer
the user's question: lead with the direct answer, then 2-4 short bullet insights (comparisons, share of total,
notable gaps), then one caveat if relevant. Be concise; markdown allowed. Do not mention JSON or SQL internals."""

ANOMALY_SYSTEM = """You are a data-quality analyst. You are given statistically flagged anomalies. Explain what was
flagged and why (the method: IQR fences + robust z-score), group by column, and give 2-3 plausible hypotheses to
verify (data entry error, bulk order, refund, etc.), clearly labelled as hypotheses. Be concise; markdown allowed."""


class PlanError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise PlanError("No JSON object in LLM output")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise PlanError(str(e)) from e


def _context(session: Session) -> str:
    if not session.history:
        return ""
    turns = [f"Q: {h['question']}\nSQL: {h.get('sql') or '-'}\nA: {h['answer'][:300]}" for h in session.history[-5:]]
    return "Previous turns:\n" + "\n---\n".join(turns) + "\n\n"


def _llm_plan(session: Session, question: str, llm: LLM, repair: tuple[str, str] | None = None) -> dict:
    content = f"{_context(session)}Question: {question}"
    if repair:
        content += f"\n\nYour previous SQL failed.\nSQL: {repair[0]}\nError: {repair[1]}\nReturn corrected JSON."
    system = PLAN_SYSTEM + "\nSchema:\n" + session.schema_text()
    for attempt in range(2):
        try:
            return _extract_json(llm.complete(system, [{"role": "user", "content": content}]))
        except PlanError:
            content += "\n\nReturn ONLY valid JSON."
    raise PlanError("LLM did not return valid JSON")


def _fmt_result(df, n: int = 30) -> str:
    return df.head(n).to_csv(index=False) + (f"\n({len(df)} rows total)" if len(df) > n else "")


def run_turn(session: Session, question: str, llm: LLM | None) -> dict:
    t0 = time.perf_counter()
    steps, warnings = [], []
    mode = llm.name if llm else "offline"

    def step(name, start, **extra):
        steps.append({"step": name, "ms": round((time.perf_counter() - start) * 1000), **extra})

    s = time.perf_counter()
    plan = None
    if llm:
        try:
            plan = _llm_plan(session, question, llm)
        except (LLMError, PlanError) as e:
            warnings.append(f"LLM unavailable ({e}); used offline heuristics.")
            log.warning("llm plan failed: %s", e)
            llm, mode = None, "offline"
    if plan is None:
        plan = offline.plan(session, question)
    step("plan", s, intent=plan.get("intent"))

    intent = plan.get("intent", "query")
    sql, df, error, anoms = None, None, None, []

    if intent == "anomaly":
        s = time.perf_counter()
        anoms = anomaly.scan(session.datasets, plan.get("anomaly_tables"))
        step("anomaly_scan", s, found=len(anoms))
    elif plan.get("sql"):
        for attempt in range(MAX_REPAIRS + 1):
            s = time.perf_counter()
            try:
                sql = validate_sql(plan["sql"])
                df = session.run_sql(sql)
                step("execute_sql", s, rows=len(df), attempt=attempt)
                error = None
                break
            except (SQLValidationError, duckdb.Error) as e:
                error = str(e).splitlines()[0]
                step("execute_sql_failed", s, error=error)
                if not llm or attempt == MAX_REPAIRS:
                    break
                try:
                    plan = {**plan, **_llm_plan(session, question, llm, (plan["sql"], error))}
                except (LLMError, PlanError):
                    break

    chart = build_chart(df, plan.get("chart")) if df is not None else None
    table = df_to_table(df) if df is not None else None

    s = time.perf_counter()
    if error:
        answer = f"I couldn't run a valid query for that ({error}). Try rephrasing or naming the columns."
    elif intent == "chat" or (df is None and not anoms and intent != "anomaly"):
        answer = plan.get("answer") or plan.get("reasoning") or "I couldn't answer that from the loaded data."
    elif intent == "anomaly":
        answer = None
        if llm:
            try:
                answer = llm.complete(ANOMALY_SYSTEM, [{"role": "user", "content":
                    f"Question: {question}\nFlagged anomalies (JSON):\n{json.dumps(anoms[:15], default=str)}"}])
            except LLMError as e:
                warnings.append(f"LLM explanation failed ({e}).")
        answer = answer or offline.anomaly_narrative(anoms)
    else:
        answer = None
        if llm:
            try:
                answer = llm.complete(ANSWER_SYSTEM, [{"role": "user", "content":
                    f"Question: {question}\nSQL: {sql}\nResult:\n{_fmt_result(df)}"}])
            except LLMError as e:
                warnings.append(f"LLM explanation failed ({e}).")
        answer = answer or offline.narrative(question, df, len(df))
    step("explain", s)

    session.history.append({"question": question, "answer": answer, "sql": sql,
                            "pandas_code": plan.get("pandas_code"), "chart": plan.get("chart")})
    session.history[:] = session.history[-20:]
    total = round((time.perf_counter() - t0) * 1000)
    log.info("session=%s mode=%s intent=%s ms=%d", session.id, mode, intent, total)
    return {"answer": answer, "reasoning": plan.get("reasoning"), "intent": intent, "sql": sql,
            "pandas_code": plan.get("pandas_code"), "table": table, "chart": chart,
            "anomalies": anoms, "warnings": warnings, "mode": mode, "steps": steps, "latency_ms": total}
