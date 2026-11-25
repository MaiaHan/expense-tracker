"""Load test the API on synthetic statements.

    python benchmarks/bench.py --rows 100000 --merchants 2000
    DATABASE_URL=postgresql://expense:expense@localhost:5432/expense_bench python benchmarks/bench.py ...

Runs the FastAPI app in-process against DATABASE_URL (the database is wiped first).
The LLM is faked: each call sleeps like the real one did on our labeled data
(~0.5s + 0.11s per item, scaled by --llm-scale), so no API calls and no cost.
"""
import argparse
import io
import json
import os
import random
import sys
import time
import types
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "bench-not-a-real-key")
os.environ.setdefault("TAVILY_API_KEY", "bench-not-a-real-key")

CHAINS = ["SAFEWAY", "TRADER JOE S", "WHOLEFDS", "COSTCO WHSE", "TARGET", "WALGREENS", "CVS/PHARMACY",
          "STARBUCKS STORE", "MCDONALD'S", "CHIPOTLE", "SUBWAY", "SHELL OIL", "CHEVRON", "ARCO",
          "HOME DEPOT", "BEST BUY", "UNIQLO USA", "AMC THEATRES", "UNITED AIRLINES", "MARRIOTT"]
LOCAL_KINDS = ["CAFE", "NOODLE HOUSE", "TACO SHOP", "MARKET", "BAKERY", "SALON", "AUTO REPAIR", "BOOKS",
               "PHO", "SUSHI BAR", "PIZZA", "DELI", "HARDWARE", "FLOWERS", "YOGA STUDIO"]
LOCAL_NAMES = ["BLUE", "GOLDEN", "LUCKY", "NORTH", "OLD TOWN", "SUNNY", "MAPLE", "HARBOR", "RIVER",
               "CEDAR", "PINE", "ROSE", "STAR", "UNION", "PARK", "LAKE", "HILL", "BAY", "RED", "GREEN"]
CITIES = ["SEATTLE WA", "BELLEVUE WA", "PORTLAND OR", "AUSTIN TX", "DENVER CO", "BOSTON MA"]


def make_merchants(n: int, rng: random.Random) -> list[str]:
    merchants = set()
    while len(merchants) < n:
        if rng.random() < 0.4:
            merchants.add(f"{rng.choice(CHAINS)} #{rng.randint(100, 9999)} {rng.choice(CITIES)}")
        else:
            merchants.add(f"{rng.choice(LOCAL_NAMES)} {rng.choice(LOCAL_NAMES)} {rng.choice(LOCAL_KINDS)} "
                          f"{rng.choice(CITIES)}")
    return sorted(merchants)


def make_csv(rows: int, merchants: list[str], rng: random.Random) -> str:
    start = date(2024, 1, 1)
    weights = [1 / (i + 1) for i in range(len(merchants))]  # a few merchants get most of the spend
    out = io.StringIO()
    out.write("Trans. Date,Post Date,Description,Amount,Category\n")
    for _ in range(rows):
        d = start + timedelta(days=rng.randint(0, 729))
        r = rng.random()
        if r < 0.02:
            desc, amount = "INTERNET PAYMENT - THANK YOU", -round(rng.uniform(200, 3000), 2)
        elif r < 0.03:
            desc, amount = "CASHBACK BONUS REDEMPTION PYMT/STMT CRDT", -round(rng.uniform(1, 30), 2)
        else:
            desc = rng.choices(merchants, weights)[0]
            amount = round(rng.lognormvariate(3, 1), 2)
            if r < 0.04:
                amount = -amount  # refund
        out.write(f"{d:%m/%d/%Y},{d:%m/%d/%Y},{desc},{amount:.2f},Merchandise\n")
    return out.getvalue()


def fake_llm(scale: float):
    calls = []

    def create(model, messages, response_format=None):
        content = messages[-1]["content"]
        if "bank CSV file" in content:
            out = {"description_col": "Description", "amount_col": "Amount", "date_col": "Trans. Date",
                   "type_col": None, "category_col": "Category"}
        elif content.startswith("Categorize these transactions"):
            items = json.loads(content.split(": ", 1)[1])
            time.sleep(scale * (0.5 + 0.11 * len(items)))
            calls.append(len(items))
            out = {str(it["id"]): "Other" for it in items}
        else:
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))])
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(
            content=json.dumps(out)))])
    return create, calls


def timed(label, fn, rows=None):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    rate = f"  {rows / elapsed:>10,.0f} rows/s" if rows else ""
    print(f"  {label:<34}{elapsed:>9.2f}s{rate}")
    return result, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--merchants", type=int, default=2_000)
    parser.add_argument("--llm-scale", type=float, default=0.1,
                        help="multiplier on simulated LLM latency (1.0 = like the real API)")
    parser.add_argument("--fraud-max-rows", type=int, default=20_000,
                        help="skip /fraud above this many rows")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import database
    import main as app_main
    from fastapi.testclient import TestClient

    with database.Session(database.engine) as session:
        for model in (database.Transaction, database.MerchantRule, database.Category):
            session.query(model).delete()
        session.commit()
    database.init_db()

    create, calls = fake_llm(args.llm_scale)
    app_main.client.chat.completions.create = create
    app_main.tavily.search = lambda q, max_results=3: {"results": []}
    client = TestClient(app_main.app)

    rng = random.Random(args.seed)
    csv = make_csv(args.rows, make_merchants(args.merchants, rng), rng)
    print(f"DB: {database.engine.url.get_backend_name()} | {args.rows:,} rows, ~{args.merchants:,} merchants "
          f"| LLM latency scale {args.llm_scale}")

    def post_upload():
        r = client.post("/upload", files={"file": ("bench.csv", csv, "text/csv")})
        assert r.status_code == 200, r.text[:300]
        return r.json()

    r, _ = timed("upload (new)", post_upload, args.rows)
    print(f"    inserted {r.get('inserted'):,}, duplicates {r.get('duplicates'):,}")
    r, _ = timed("upload again (all duplicates)", post_upload, args.rows)
    assert r["inserted"] == 0, r

    def classify():
        r = client.post("/classify")
        assert r.status_code == 200, r.text[:300]
    _, t = timed("classify", classify, args.rows)
    print(f"    {len(calls)} LLM calls, {sum(calls):,} merchants "
          f"(simulated LLM time if sequential: {sum(args.llm_scale * (0.5 + 0.11 * n) for n in calls):.1f}s)")

    timed("GET /transactions", lambda: client.get("/transactions").json(), args.rows)
    timed("chat summary query", lambda: client.post("/chat", params={"question": "hi"}).json())
    timed("deals (top merchants)", lambda: client.post("/deals").json())
    if args.rows <= args.fraud_max_rows:
        timed("fraud", lambda: client.post("/fraud").json(), args.rows)
    else:
        print(f"  {'fraud':<34}  skipped (> --fraud-max-rows)")


if __name__ == "__main__":
    main()
