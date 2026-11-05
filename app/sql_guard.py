"""Read-only SQL validation. LLM-generated SQL is never trusted."""
from __future__ import annotations

import re

FORBIDDEN = {
    "insert", "update", "delete", "drop", "create", "alter", "attach", "detach",
    "copy", "pragma", "install", "load", "export", "import", "call", "set",
    "reset", "truncate", "vacuum", "checkpoint", "grant", "revoke",
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "read_text", "read_blob", "glob", "parquet_scan", "sniff_csv", "httpfs",
}


class SQLValidationError(ValueError):
    """Raised when a query is not a single read-only SELECT."""


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"--[^\n]*", " ", sql)


def validate_sql(sql: str) -> str:
    """Return a cleaned single SELECT/WITH statement or raise SQLValidationError."""
    if not sql or not sql.strip():
        raise SQLValidationError("Empty SQL.")
    no_comments = _strip_comments(sql)
    # Blank out string literals so keywords/semicolons inside them are ignored.
    scan = re.sub(r"'(?:[^']|'')*'", "''", no_comments)
    statements = [s for s in scan.split(";") if s.strip()]
    if len(statements) != 1:
        raise SQLValidationError("Only a single SQL statement is allowed.")
    first = re.match(r"\s*\(?\s*([a-z_]+)", statements[0].lower())
    if not first or first.group(1) not in {"select", "with"}:
        raise SQLValidationError("Only SELECT queries are allowed.")
    bad = FORBIDDEN.intersection(re.findall(r"[a-z_]+", statements[0].lower()))
    if bad:
        raise SQLValidationError(f"Forbidden keyword(s): {', '.join(sorted(bad))}.")
    return no_comments.strip().rstrip(";").strip()
