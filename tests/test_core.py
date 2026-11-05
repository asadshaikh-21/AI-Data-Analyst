import numpy as np
import pandas as pd
import pytest

from app.anomaly import detect_anomalies
from app.charts import build_chart
from app.data_store import Session, ValidationError, load_csv
from app.sql_guard import SQLValidationError, validate_sql


# ---------- SQL guard ----------
@pytest.mark.parametrize("sql", [
    "SELECT * FROM t", "select 'a;b' as x", "WITH a AS (SELECT 1) SELECT * FROM a;", "SELECT replace(x,'a','b') FROM t",
])
def test_guard_allows(sql):
    assert validate_sql(sql)


@pytest.mark.parametrize("sql", [
    "", "DROP TABLE t", "SELECT 1; DROP TABLE t", "DELETE FROM t", "SELECT * FROM read_csv('/etc/passwd')",
    "COPY t TO 'x.csv'", "PRAGMA database_list", "SELECT 1 -- hi\n; INSERT INTO t VALUES (1)",
])
def test_guard_blocks(sql):
    with pytest.raises(SQLValidationError):
        validate_sql(sql)


# ---------- upload / validation ----------
def test_load_csv_ok_and_types():
    raw = b"Order Date,Region Name,Amount\n2024-01-01,N,$1,200\n"  # malformed on purpose -> still parses or errors cleanly
    raw = b'Order Date,Region Name,Amount\n2024-01-01,N,"$1,200"\n2024-01-02,S,"$300"\n'
    df = load_csv("a.csv", raw)
    assert list(df.columns) == ["order_date", "region_name", "amount"]
    assert str(df["order_date"].dtype).startswith("datetime")
    assert df["amount"].tolist() == [1200, 300]


@pytest.mark.parametrize("name,raw", [("a.txt", b"x,y\n1,2"), ("a.csv", b""), ("a.csv", b"x,y\n")])
def test_load_csv_rejects(name, raw):
    with pytest.raises(ValidationError):
        load_csv(name, raw)


def test_duplicate_columns_deduped():
    df = load_csv("a.csv", b"a,a,b\n1,2,3\n")
    assert list(df.columns) == ["a", "a_1", "b"]


def test_session_sql_roundtrip():
    s = Session()
    s.add_dataset("Sales Data.csv", b"region,rev\nN,10\nS,20\nN,5\n")
    assert "sales_data" in s.datasets
    out = s.run_sql('SELECT region, SUM(rev) AS t FROM sales_data GROUP BY 1 ORDER BY 2 DESC')
    assert out.iloc[0].tolist() == ["S", 20] or out.iloc[0]["t"] == 20


# ---------- anomalies ----------
def test_anomaly_flags_outlier_with_reason():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"amount": np.append(rng.normal(100, 5, 100), 5000), "name": ["x"] * 101})
    res = detect_anomalies(df, "t")
    assert res and res[0]["value"] == 5000 and "robust z" in res[0]["reason"]


def test_anomaly_ignores_clean_and_ids():
    df = pd.DataFrame({"order_id": range(100), "v": np.linspace(1, 10, 100)})
    assert detect_anomalies(df, "t") == []


# ---------- charts ----------
def test_chart_auto_line_for_dates():
    df = pd.DataFrame({"month": ["2024-01", "2024-02"], "total": [1.0, 2.0]})
    c = build_chart(df, {"type": "bogus", "x": "nope", "y": ["nope"]})
    assert c["type"] == "line" and c["series"][0]["data"] == [1.0, 2.0]


def test_chart_none_for_single_column():
    assert build_chart(pd.DataFrame({"n": [1]}), {}) is None


def test_anomaly_group_aware_ignores_expensive_group_but_flags_true_outlier():
    rng = np.random.default_rng(1)
    cheap = rng.normal(10, 1, 60)
    dear = rng.normal(1000, 20, 60)
    cheap[5] = 5000  # real outlier inside the cheap group
    df = pd.DataFrame({"product": ["mouse"] * 60 + ["laptop"] * 60, "price": np.append(cheap, dear)})
    res = detect_anomalies(df, "t")
    assert [r["value"] for r in res] == [5000.0]
