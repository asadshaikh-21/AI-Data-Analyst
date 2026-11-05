from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _session():
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(f"/api/sessions/{sid}/load-samples")
    assert r.status_code == 200 and len(r.json()["datasets"]) == 2
    return sid


def chat(sid, q):
    r = client.post(f"/api/sessions/{sid}/chat", json={"question": q})
    assert r.status_code == 200, r.text
    return r.json()


def test_health():
    assert client.get("/health").json()["llm"] == "offline"


def test_upload_validation_error_is_reported():
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(f"/api/sessions/{sid}/upload", files=[("files", ("bad.txt", b"x", "text/plain"))])
    assert r.json()["results"][0]["ok"] is False


def test_chat_requires_data_and_valid_session():
    sid = client.post("/api/sessions").json()["session_id"]
    assert client.post(f"/api/sessions/{sid}/chat", json={"question": "hi"}).status_code == 400
    assert client.post("/api/sessions/nope/chat", json={"question": "hi"}).status_code == 404


def test_region_revenue_and_sql_followup():
    sid = _session()
    r = chat(sid, "Which region generated the highest revenue?")
    assert r["sql"] and r["table"]["rows"] and r["chart"]["type"] == "bar"
    again = chat(sid, "Generate SQL for this analysis.")
    assert again["sql"] == r["sql"]  # conversation context is kept


def test_trend_is_line_chart():
    r = chat(_session(), "Show monthly sales trends.")
    assert r["chart"]["type"] == "line" and len(r["chart"]["labels"]) >= 12


def test_top_customers_joins_tables():
    r = chat(_session(), "What are the top five customers?")
    assert "JOIN" in r["sql"] and len(r["table"]["rows"]) == 5


def test_underperforming_products_ascending():
    r = chat(_session(), "Which products are underperforming?")
    vals = [row[1] for row in r["table"]["rows"]]
    assert vals == sorted(vals)


def test_anomalies_found_with_explanations():
    r = chat(_session(), "Detect anomalies in the dataset.")
    assert r["anomalies"] and all(a["reason"] for a in r["anomalies"])


def test_report_export():
    sid = _session()
    chat(sid, "Which region generated the highest revenue?")
    text = client.get(f"/api/sessions/{sid}/report").text
    assert "highest revenue" in text and "```sql" in text
