# AI Data Analyst

Upload one or more CSV files and talk to your data in plain English. The app writes and **executes SQL**, draws **charts**, flags **anomalies with explanations**, shows its **reasoning**, and remembers the **conversation** so follow-ups work.

> Built for the Digital Back Office *AI Engineer Assignment*. The focus is a production-minded application around the LLM (validation, sandboxed execution, self-repair, graceful degradation, tests), not just an API call.
 **Demo video:** `https://www.loom.com/share/821ee684c55e4a648b369264d4ecf8a6`

| Chat + chart | Anomaly scan |
|---|---|
| `docs/screenshot-chat.png` | `docs/screenshot-anomalies.png` |

_(Drop your screenshots into `docs/` and keep the names above.)_

## Features

| Requirement | How it is done |
|---|---|
| Upload & validate 1+ CSVs | Extension/size/empty/parse checks, delimiter + encoding sniffing, column-name sanitising, de-duplication, date and currency-string type coercion |
| Natural-language Q&A | LLM plans a query → DuckDB runs it → LLM explains the real result (never invents numbers) |
| Business insights | Second LLM pass over the actual result rows: direct answer, bullets, caveat |
| Charts | Bar / Line / Pie / Scatter, chosen by the LLM and validated/auto-corrected against the result columns (Chart.js) |
| SQL / Pandas code | Every answer exposes the executed SQL and an equivalent Pandas snippet; "Generate SQL for this analysis" reuses the last query |
| Anomaly detection + why | IQR fence **and** robust z-score (MAD) must both fire; group-aware (a laptop's price isn't an anomaly among mice). Every flag carries a plain-English reason |
| Reasoning | "How I got this" panel + per-step trace (plan → execute → explain, with timings) |
| Conversation context | Last 5 turns (question, SQL, answer) are fed back to the planner, so "now by region" / "show the SQL" work |
| **Bonus** | Multi-file joins, data-quality score, caching, export report (Markdown), logging/observability, offline fallback mode, Docker, tests |

## Architecture

```mermaid
flowchart LR
    UI["Browser UI<br/>(chat, Chart.js, tables)"] -->|REST/JSON| API["FastAPI<br/>app/main.py"]
    API --> SM["SessionManager<br/>(per-session state)"]
    API --> AG["Agent<br/>app/agent.py"]
    SM --> DS["Data store<br/>validate · profile · quality"]
    DS --> DB[("DuckDB in-memory<br/>one table per CSV")]
    AG -->|1. plan JSON| LLM["LLM client<br/>Anthropic / OpenAI<br/>retry + cache"]
    AG -->|fallback| OFF["Offline heuristic planner"]
    AG -->|2. validate| GUARD["SQL guard<br/>read-only SELECT only"]
    GUARD -->|3. execute| DB
    DB -->|result| AG
    AG -->|repair loop on error<br/>(max 2)| LLM
    AG --> CH["Chart builder"]
    AG --> AN["Anomaly detector<br/>IQR + MAD, group-aware"]
    AN --> DB
    AG -->|4. explain real rows| LLM
```

**Request flow:** question + schema + recent history → LLM returns a JSON plan (intent, reasoning, SQL, pandas, chart hint) → SQL is validated as a single read-only statement → executed in DuckDB (15 s timeout) → on error the message is fed back to the LLM for up to 2 repairs → result is charted and summarised from the *actual rows*.

### Key design decisions
- **LLM output is untrusted.** `sql_guard.py` allows exactly one `SELECT`/`WITH`, blocks DDL/DML and file/network functions (`read_csv`, `COPY`, `PRAGMA`, …). Data lives in an in-memory DuckDB per session.
- **LLM for language, code for numbers.** Statistics, anomalies and aggregations are computed deterministically; the model only plans and narrates.
- **Never fully down.** No API key or an LLM outage degrades to an offline heuristic planner (top-N, lowest, monthly trend, anomalies, SQL recap) and tells the user so.
- **Provider-agnostic.** Plain-HTTP clients for Anthropic and OpenAI-compatible APIs; swap via env vars.
- **Session isolation.** Each browser session has its own DuckDB connection and history (LRU-evicted at 100 sessions).

## Quick start

```bash
git clone <this repo> && cd ai-data-analyst
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY or OPENAI_API_KEY (optional)
export $(grep -v '^#' .env | xargs)
uvicorn app.main:app --reload   # http://localhost:8000
```

Click **Load sample data**, then try the example chips.

### Docker
```bash
cp .env.example .env            # add a key (optional)
docker compose up --build       # http://localhost:8000
```

### Configuration (`.env`)
| Var | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `auto` / `anthropic` / `openai` / `offline` |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | – / `claude-sonnet-5-5` | |
| `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` | – / `gpt-4o-mini` | any OpenAI-compatible endpoint |
| `MAX_UPLOAD_MB` | `25` | per file |
| `QUERY_TIMEOUT_S` | `15` | SQL timeout |

## API
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/sessions` | new session |
| POST | `/api/sessions/{id}/upload` | multipart `files` (many CSVs) |
| POST | `/api/sessions/{id}/load-samples` | load `sample_data/` |
| GET | `/api/sessions/{id}/datasets` | schema, profile, quality |
| POST | `/api/sessions/{id}/chat` | `{"question": "..."}` → answer, sql, table, chart, anomalies, steps |
| GET | `/api/sessions/{id}/anomalies` | full anomaly scan |
| GET | `/api/sessions/{id}/report` | Markdown report of the session |
| GET | `/health` | status + active LLM |

## Tests
```bash
pytest -q        # 31 tests, no network needed (forces offline mode)
```
Covers the SQL guard (allow/deny matrix), upload validation, type coercion, DuckDB round-trip, anomaly precision, chart fallback and all six example questions through the API.

## Project layout
```
app/
  main.py        FastAPI routes
  agent.py       plan → validate → execute → repair → explain
  llm.py         Anthropic/OpenAI clients, retry, response cache
  offline.py     heuristic planner (no-key fallback)
  sql_guard.py   read-only SQL enforcement
  data_store.py  sessions, CSV validation, profiling, quality, DuckDB
  anomaly.py     IQR + MAD, group-aware
  charts.py      chart spec builder
  static/index.html   UI
sample_data/     sales.csv (1.4k rows, planted anomalies) + customers.csv
scripts/make_sample_data.py   reproducible generator
tests/
```

## Assumptions & notes
- Dates like `01/02/2024` are parsed month-first; ISO dates are unambiguous.
- Sessions are in-memory (no auth/persistence). For multi-user production: add auth, persist sessions in Redis/Postgres, and run queries in a worker.
- Anomaly detection is univariate/statistical; it will not catch a negative revenue that sits inside a wide IQR fence — that case is surfaced by the **data-quality check** instead (see the quality badge per dataset).
- The LLM sees the schema, 3 sample values per column and at most 30 result rows — never the full dataset.
- Offline mode is deliberately limited; it exists for demos, CI and outages.

## Possible next steps
Streaming responses (SSE), forecasting (statsmodels/Prophet), auth, an LLM-eval harness over a golden question set, saved dashboards.
