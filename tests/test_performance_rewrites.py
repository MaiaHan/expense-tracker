"""The batched/vectorized rewrites must behave exactly like the simple versions they replaced."""
import random

import pandas as pd

import main
from conftest import upload


def reference_fraud_reasons(df: pd.DataFrame) -> list[list[str]]:
    """The original O(n^2) implementation, kept as the reference answer."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    out = []
    for _, row in df.iterrows():
        reasons = []
        if len(df[(df["date"] == row["date"]) & (df["description"] == row["description"])]) > 2:
            reasons.append("Duplicate transaction: same merchant same day")
        merchant_avg = df[df["description"] == row["description"]]["amount"].mean()
        if row["amount"] > merchant_avg * 3 and row["amount"] > 50:
            reasons.append(f"Unusual amount: ${row['amount']:.2f} vs avg ${merchant_avg:.2f}")
        recent = df[(df["description"] == row["description"])
                    & (df["date"] >= row["date"] - pd.Timedelta(days=7)) & (df["date"] <= row["date"])]
        if len(recent) > 5:
            reasons.append(f"High frequency: {len(recent)} transactions in 7 days")
        out.append(reasons)
    return out


def test_vectorized_fraud_matches_reference():
    rng = random.Random(42)
    merchants = [f"SHOP {i}" for i in range(8)]
    start = pd.Timestamp("2026-01-01")
    df = pd.DataFrame({
        "date": [start + pd.Timedelta(days=rng.randint(0, 40)) for _ in range(400)],
        "description": [rng.choice(merchants) for _ in range(400)],
        "amount": [round(rng.choice([rng.uniform(5, 40), rng.uniform(200, 900)]), 2) for _ in range(400)],
    }).sort_values("date", kind="stable").reset_index(drop=True)

    expected = reference_fraud_reasons(df)
    actual = main.fraud_flags(df)["reasons"].tolist()
    assert actual == expected
    assert sum(1 for r in expected if r) > 50  # the data really exercises all three rules
    assert {kind for r in expected for reason in r for kind in [reason.split(":")[0]]} == {
        "Duplicate transaction", "Unusual amount", "High frequency"}


def test_fraud_endpoint(client, llm):
    rows = "".join(f"11/0{d}/2026,11/0{d}/2026,COFFEE CART,4.50,Restaurants\n" for d in (1, 1, 1, 2, 3, 4))
    r = client.post("/fraud").json()
    assert "error" in r
    upload(client, rows)
    r = client.post("/fraud").json()
    assert r["total_transactions"] == 6
    # the 3 charges on Nov 1 are duplicates; only the Nov 4 charge has 6 in its 7-day window
    assert r["flagged_count"] == 4
    nov4 = [t for t in r["flagged_transactions"] if t["date"] == "2026-11-04"]
    assert nov4[0]["reasons"] == ["High frequency: 6 transactions in 7 days"]


def test_large_upload_spans_many_insert_batches(client):
    # 5,000 rows: more than SQLAlchemy's 1,000-row insert pages, and more bind parameters
    # than SQLite allows in one statement
    rows = "".join(f"11/{1 + i % 28:02d}/2026,11/01/2026,STORE {i},{i}.25,Merchandise\n" for i in range(5000))
    r = upload(client, rows)
    assert (r["rows"], r["inserted"], r["duplicates"]) == (5000, 5000, 0)
    r = upload(client, rows + "11/30/2026,11/30/2026,NEW STORE,9.99,Merchandise\n")
    assert (r["rows"], r["inserted"], r["duplicates"]) == (5001, 1, 5000)
    amounts = sorted(t["amount"] for t in client.get("/transactions").json())
    assert amounts[0] == 0.25 and isinstance(amounts[0], float)
