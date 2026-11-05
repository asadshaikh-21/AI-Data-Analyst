"""FastAPI app: sessions, uploads, chat, anomaly scan, report export."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, anomaly
from .agent import run_turn
from .data_store import MAX_UPLOAD_MB, SessionManager, ValidationError
from .llm import get_llm

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("analyst.api")

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).parent / "static"
SAMPLE_DIR = ROOT / "sample_data"

app = FastAPI(title="AI Data Analyst", version=__version__)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
sessions = SessionManager()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def _session(sid: str):
    try:
        return sessions.get(sid)
    except KeyError:
        raise HTTPException(404, "Session not found. Refresh the page to start a new one.")


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse({"detail": "Internal error. Check server logs."}, status_code=500)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    llm = get_llm()
    return {"status": "ok", "version": __version__, "llm": f"{llm.name}:{llm.model}" if llm else "offline"}


@app.post("/api/sessions")
def create_session():
    return {"session_id": sessions.create().id}


@app.post("/api/sessions/{sid}/upload")
def upload(sid: str, files: list[UploadFile] = File(...)):
    s = _session(sid)
    results = []
    for f in files:
        raw = f.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
        try:
            results.append({"filename": f.filename, "ok": True, "dataset": s.add_dataset(f.filename or "", raw).summary()})
        except ValidationError as e:
            results.append({"filename": f.filename, "ok": False, "error": str(e)})
    return {"results": results, "datasets": [d.summary() for d in s.datasets.values()]}


@app.post("/api/sessions/{sid}/load-samples")
def load_samples(sid: str):
    s = _session(sid)
    for p in sorted(SAMPLE_DIR.glob("*.csv")):
        s.add_dataset(p.name, p.read_bytes())
    return {"datasets": [d.summary() for d in s.datasets.values()]}


@app.get("/api/sessions/{sid}/datasets")
def datasets(sid: str):
    return {"datasets": [d.summary() for d in _session(sid).datasets.values()]}


@app.post("/api/sessions/{sid}/chat")
def chat(sid: str, req: ChatRequest):
    s = _session(sid)
    if not s.datasets:
        raise HTTPException(400, "Upload at least one CSV first.")
    return run_turn(s, req.question.strip(), get_llm())


@app.get("/api/sessions/{sid}/anomalies")
def anomalies(sid: str):
    s = _session(sid)
    return {"anomalies": anomaly.scan(s.datasets)}


@app.get("/api/sessions/{sid}/report", response_class=PlainTextResponse)
def report(sid: str):
    s = _session(sid)
    out = ["# AI Data Analyst report\n", "## Datasets"]
    for d in s.datasets.values():
        out.append(f"- **{d.name}** ({len(d.df)} rows, quality {d.quality['score']}/100)")
    out.append("\n## Analysis log")
    for i, h in enumerate(s.history, 1):
        out.append(f"\n### {i}. {h['question']}\n\n{h['answer']}")
        if h.get("sql"):
            out.append(f"\n```sql\n{h['sql']}\n```")
    return PlainTextResponse("\n".join(out), headers={"Content-Disposition": "attachment; filename=report.md"})
