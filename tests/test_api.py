"""API tests: upload/dedupe, classification, spending summaries, correction rules."""
import sqlite3

from sqlalchemy import create_engine

import database
from conftest import by_description, upload

STATEMENT = """11/01/2026,11/02/2026,SUBWAY 41234 SEATTLE WA,12.50,Restaurants
11/01/2026,11/02/2026,SUBWAY 41234 SEATTLE WA,12.50,Restaurants
11/03/2026,11/04/2026,AMAZON MKTPL 1234567890123,40.00,Merchandise
11/05/2026,11/06/2026,INTERNET PAYMENT - THANK YOU,-100.00,Payments and Credits
bad-date,11/06/2026,CHIPOTLE,9.00,Restaurants
11/07/2026,11/08/2026,AMAZON MKTPLACE PMTS AMZN.COM/BILLWA,-25.00,Payments and Credits
11/08/2026,11/08/2026,CASHBACK BONUS REDEMPTION PYMT/STMT CRDT,-5.00,Awards and Rebate Credits
"""


def shop_or_food(item):
    return "Shopping" if "AMAZON" in item["description"] else "Food & Dining"


# ── upload ──

def test_upload_keeps_identical_purchases_and_skips_invalid_rows(client):
    r = upload(client, STATEMENT)
    assert (r["rows"], r["inserted"], r["duplicates"], r["skipped_invalid"]) == (6, 6, 0, 1)


def test_reupload_only_inserts_new_rows(client):
    upload(client, STATEMENT)
    r = upload(client, STATEMENT + "11/10/2026,11/11/2026,CHIPOTLE 0999,15.00,Restaurants\n")
    assert (r["inserted"], r["duplicates"]) == (1, 6)


# ── classify ──

def test_classify_uses_rules_then_llm_and_skips_done_rows(client, llm):
    llm.categorize = shop_or_food
    upload(client, STATEMENT)
    client.post("/classify")
    t = by_description(client)
    assert t["INTERNET PAYMENT - THANK YOU"]["predicted_category"] == "Card Payment"
    assert t["CASHBACK BONUS REDEMPTION PYMT/STMT CRDT"]["predicted_category"] == "Rewards & Cashback"
    assert t["AMAZON MKTPLACE PMTS AMZN.COM/BILLWA"]["predicted_category"] == "Shopping"  # refund
    assert "INTERNET PAYMENT - THANK YOU" not in llm.asked

    calls = len(llm.requests)
    client.post("/classify")
    assert len(llm.requests) == calls  # nothing pending, no LLM call


def test_chat_summary_excludes_card_payments_and_nets_refunds(client, llm):
    llm.categorize = shop_or_food
    upload(client, STATEMENT)
    client.post("/classify")
    answer = client.post("/chat", params={"question": "how much?"}).json()["answer"]
    assert '"Shopping": 15.0' in answer  # 40 purchase - 25 refund
    assert "Card Payment" not in answer and "Rewards" not in answer


def test_reserved_category_names_cannot_be_added(client):
    assert "error" in client.post("/categories/add", params={"name": "Card Payment"}).json()


# ── corrections and merchant rules ──

DRIVERS = """11/01/2026,11/02/2026,LYFT *RIDE ALICE 8005550100 VA,18.00,Travel/ Entertainment
11/02/2026,11/03/2026,LYFT *RIDE BOB 8005550100 VA,22.00,Travel/ Entertainment
11/04/2026,11/05/2026,SUBWAY 41234 SEATTLE WA,12.50,Restaurants
"""
ALICE, BOB, SUBWAY = (
    "LYFT *RIDE ALICE 8005550100 VA", "LYFT *RIDE BOB 8005550100 VA", "SUBWAY 41234 SEATTLE WA")


def correct(client, description, **params):
    tid = by_description(client)[description]["id"]
    return client.post(f"/transactions/{tid}/category", params=params).json()


def classified(client, llm, rows=DRIVERS):
    llm.categorize = lambda it: "Food & Dining" if "SUBWAY" in it["description"] else "Other"
    upload(client, rows)
    client.post("/classify")


def test_rule_with_short_pattern_fixes_all_matching_merchants(client, llm):
    classified(client, llm)
    r = correct(client, ALICE, category="Transportation", pattern=" lyft ")
    assert r["rule"]["pattern"] == "LYFT" and r["updated"] == 2
    t = by_description(client)[BOB]
    assert (t["predicted_category"], t["category_source"]) == ("Transportation", "user_rule")


def test_rule_applies_to_future_uploads_without_llm(client, llm):
    classified(client, llm)
    correct(client, ALICE, category="Transportation", pattern="LYFT")
    llm.requests.clear()
    upload(client, "11/10/2026,11/11/2026,LYFT *RIDE CAROL 8005550100 VA,15.00,Travel/ Entertainment\n")
    client.post("/classify")
    assert llm.asked == []
    assert by_description(client)["LYFT *RIDE CAROL 8005550100 VA"]["predicted_category"] == "Transportation"


def test_manual_fix_survives_reclassify(client, llm):
    classified(client, llm)
    correct(client, SUBWAY, category="Entertainment", apply_to_merchant=False)
    client.post("/classify", params={"reclassify": True})
    t = by_description(client)[SUBWAY]
    assert (t["predicted_category"], t["category_source"]) == ("Entertainment", "manual")


def test_correction_errors(client, llm):
    classified(client, llm)
    assert "error" in correct(client, ALICE, category="Not A Category")
    assert "error" in correct(client, ALICE, category="Transportation", pattern="UBER")
    assert "error" in client.post("/transactions/missing/category", params={"category": "Other"}).json()
    assert "error" in client.delete("/rules/999").json()


def test_deleting_rules_falls_back_to_broader_rule_then_llm(client, llm):
    classified(client, llm)
    correct(client, ALICE, category="Transportation", pattern="LYFT")
    correct(client, BOB, category="Travel", pattern="LYFT *RIDE BOB")
    assert by_description(client)[BOB]["predicted_category"] == "Travel"  # more specific wins

    rules = {r["pattern"]: r["id"] for r in client.get("/rules").json()["rules"]}
    client.delete(f"/rules/{rules['LYFT *RIDE BOB']}")
    assert by_description(client)[BOB]["predicted_category"] == "Transportation"

    r = client.delete(f"/rules/{rules['LYFT']}").json()
    assert r["cleared"] == 2 and by_description(client)[BOB]["predicted_category"] is None
    llm.requests.clear()
    client.post("/classify")
    assert len(llm.asked) == 2


# ── schema upgrade ──

def test_init_db_adds_missing_column_to_old_database(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE transactions (id VARCHAR PRIMARY KEY, date VARCHAR, description VARCHAR, "
                "amount FLOAT, category VARCHAR, predicted_category VARCHAR, created_at DATETIME)")
    con.execute("INSERT INTO transactions (id, description, amount) VALUES ('a', 'OLD ROW', 1.0)")
    con.commit()
    con.close()

    monkeypatch.setattr(database, "engine", create_engine(f"sqlite:///{path}"))
    database.init_db()
    con = sqlite3.connect(path)
    assert "category_source" in [row[1] for row in con.execute("PRAGMA table_info(transactions)")]
    assert con.execute("SELECT description FROM transactions").fetchall() == [("OLD ROW",)]
