"""Sessions, CSV upload/validation, profiling, data-quality checks, DuckDB execution."""
from __future__ import annotations

import io
import logging
import os
import re
import threading
import time
import uuid
import warnings
from dataclasses import dataclass, field

import duckdb
import pandas as pd

log = logging.getLogger("analyst.data")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_FILES = int(os.getenv("MAX_FILES_PER_SESSION", "10"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "100"))
QUERY_TIMEOUT_S = float(os.getenv("QUERY_TIMEOUT_S", "15"))

DATE_RE = re.compile(
    r"^\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?$|^\d{1,2}/\d{1,2}/\d{2,4}$"
)
CURRENCY_RE = re.compile(r"[$€£₹,\s%]")


class ValidationError(ValueError):
    """Raised for unusable uploads (shown to the user)."""


def sanitize_identifier(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip()).strip("_").lower()
    s = s or "col"
    return "c_" + s if s[0].isdigit() else s


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]) \
                or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        s = df[col].dropna().astype(str).str.strip()
        if s.empty:
            continue
        sample = s.head(50)
        if sample.map(lambda v: bool(DATE_RE.match(v))).all():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() >= 0.9 * df[col].notna().sum():
                df[col] = parsed
                continue
        cleaned = s.map(lambda v: CURRENCY_RE.sub("", v))
        nums = pd.to_numeric(cleaned, errors="coerce")
        if nums.notna().mean() >= 0.95 and s.str.contains(r"\d").mean() >= 0.95:
            df[col] = pd.to_numeric(
                df[col].astype(str).map(lambda v: CURRENCY_RE.sub("", v)), errors="coerce"
            )
        else:
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)  # trim whitespace
    return df


def load_csv(filename: str, raw: bytes) -> pd.DataFrame:
    if not filename.lower().endswith(".csv"):
        raise ValidationError(f"'{filename}' is not a .csv file.")
    if not raw:
        raise ValidationError(f"'{filename}' is empty.")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValidationError(f"'{filename}' exceeds the {MAX_UPLOAD_MB} MB limit.")
    df = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            raise ValidationError(f"Could not parse '{filename}': {e}") from e
    if df is None or df.shape[1] == 0:
        raise ValidationError(f"Could not read any columns from '{filename}'.")
    df = df.dropna(how="all")
    if df.empty:
        raise ValidationError(f"'{filename}' has headers but no data rows.")
    df.columns = _dedupe([sanitize_identifier(c) for c in df.columns])
    return _coerce_types(df)


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "text"


def profile(df: pd.DataFrame) -> list[dict]:
    out = []
    for c in df.columns:
        s = df[c]
        kind = _kind(s)
        info = {
            "name": c, "kind": kind, "nulls": int(s.isna().sum()),
            "null_pct": round(float(s.isna().mean()) * 100, 1),
            "unique": int(s.nunique()),
            "sample": [str(v) for v in s.dropna().unique()[:3]],
        }
        if kind in ("numeric", "datetime") and s.notna().any():
            info["min"], info["max"] = str(s.min()), str(s.max())
        out.append(info)
    return out


def quality_report(df: pd.DataFrame) -> dict:
    issues, penalty = [], 0.0
    dups = int(df.duplicated().sum())
    if dups:
        issues.append(f"{dups} duplicate row(s)")
        penalty += min(15, dups / len(df) * 100)
    for c in df.columns:
        s = df[c]
        null_pct = s.isna().mean() * 100
        if null_pct > 20:
            issues.append(f"'{c}' is {null_pct:.0f}% empty")
            penalty += 5
        elif null_pct > 0:
            issues.append(f"'{c}' has {int(s.isna().sum())} missing value(s)")
            penalty += 1
        if s.nunique(dropna=True) == 1 and len(df) > 1:
            issues.append(f"'{c}' is constant")
            penalty += 2
        if _kind(s) == "numeric" and re.search(r"price|qty|quantity|revenue|amount|sales", c):
            neg = int((s < 0).sum())
            if neg:
                issues.append(f"'{c}' has {neg} negative value(s)")
                penalty += 3
        if _kind(s) == "text":
            vals = s.dropna().astype(str)
            if len(vals) and (vals != vals.str.strip()).any():
                issues.append(f"'{c}' has leading/trailing whitespace")
                penalty += 1
    return {"score": max(0, round(100 - penalty)), "issues": issues[:12]}


@dataclass
class Dataset:
    name: str
    filename: str
    df: pd.DataFrame
    columns: list[dict]
    quality: dict

    def summary(self) -> dict:
        return {"name": self.name, "filename": self.filename, "rows": len(self.df),
                "columns": self.columns, "quality": self.quality}


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    datasets: dict[str, Dataset] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _conn: duckdb.DuckDBPyConnection = field(default=None, repr=False)

    def __post_init__(self):
        self._conn = duckdb.connect(":memory:")

    def add_dataset(self, filename: str, raw: bytes) -> Dataset:
        if len(self.datasets) >= MAX_FILES and sanitize_identifier(filename.rsplit(".", 1)[0]) not in self.datasets:
            raise ValidationError(f"Max {MAX_FILES} files per session.")
        df = load_csv(filename, raw)
        name = sanitize_identifier(os.path.splitext(os.path.basename(filename))[0])
        with self._lock:
            if name in self.datasets:
                self._conn.unregister(name)
            self._conn.register(name, df)
            ds = Dataset(name, filename, df, profile(df), quality_report(df))
            self.datasets[name] = ds
        log.info("session=%s loaded table=%s rows=%d", self.id, name, len(df))
        return ds

    def schema_text(self) -> str:
        lines = []
        for ds in self.datasets.values():
            cols = []
            for c in ds.columns:
                extra = ""
                if "min" in c:
                    extra = f" range {c['min']}..{c['max']}"
                elif c["sample"]:
                    extra = " e.g. " + ", ".join(c["sample"])
                cols.append(f"  - {c['name']} ({c['kind']}){extra}")
            lines.append(f"Table {ds.name} ({len(ds.df)} rows):\n" + "\n".join(cols))
        return "\n".join(lines)

    def run_sql(self, sql: str, max_rows: int = 1000) -> pd.DataFrame:
        # Registered frames are visible only on the registering connection (not on cursors),
        # so queries on one session are serialised under its lock.
        with self._lock:
            timer = threading.Timer(QUERY_TIMEOUT_S, self._conn.interrupt)
            timer.start()
            try:
                return self._conn.execute(sql).fetchdf().head(max_rows)
            finally:
                timer.cancel()


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        with self._lock:
            if len(self._sessions) >= MAX_SESSIONS:
                oldest = min(self._sessions.values(), key=lambda s: s.last_used)
                del self._sessions[oldest.id]
            s = Session()
            self._sessions[s.id] = s
            return s

    def get(self, sid: str) -> Session:
        s = self._sessions.get(sid)
        if s is None:
            raise KeyError(sid)
        s.last_used = time.time()
        return s
