import os
import io
import json
from typing import Optional
import numpy as np
import pandas as pd
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from sqlalchemy import func, or_, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session
from database import (
    init_db, get_categories, engine, Transaction, Category, MerchantRule, make_transaction_id,
    NON_SPENDING_CATEGORIES,
)
from classifier import clean_description, merchant_key, match_user_rule, predict_categories

# Load API keys from .env file
load_dotenv()

# Initialize database when server starts
init_db()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
app = FastAPI()

def dialect_insert(table):
    """INSERT that supports on_conflict_do_nothing for the database in use."""
    if engine.dialect.name == "postgresql":
        return postgresql.insert(table)
    return sqlite.insert(table)

TRANSACTION_FIELDS = ["id", "date", "description", "amount", "category", "predicted_category", "category_source"]


def all_transactions(session: Session) -> list[dict]:
    """Every stored transaction as a JSON-ready dict, oldest first.

    Selects plain columns instead of loading ORM objects, which is several times faster
    for large tables.
    """
    columns = [getattr(Transaction, f) for f in TRANSACTION_FIELDS]
    rows = session.query(*columns).order_by(Transaction.date, Transaction.id).all()
    return [
        {**dict(zip(TRANSACTION_FIELDS, row)),
         "date": row.date.isoformat() if row.date else None,
         "amount": None if row.amount is None else float(row.amount)}  # SQLite may hand back ints
        for row in rows
    ]


def load_transactions_df(session: Session) -> pd.DataFrame:
    """All stored transactions as a DataFrame (empty DataFrame if none)."""
    return pd.DataFrame(all_transactions(session), columns=TRANSACTION_FIELDS)


# Endpoints are plain `def`, not `async def`: they do blocking work (DB, OpenAI client, pandas),
# which inside async endpoints would stall the event loop for every other request.
# FastAPI runs plain def endpoints in a thread pool instead.

# /upload endpoint
@app.post("/upload")
def upload(file: UploadFile = File(...)):
    # Read the uploaded CSV file
    contents = file.file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Show AI the column names and first 3 rows to figure out the structure
    sample = df.head(3).to_string()
    columns = list(df.columns)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""Here are the columns and sample rows from a bank CSV file:
Columns: {columns}
Sample rows:
{sample}

Identify which column represents:
1. Transaction description / merchant name
2. Amount / money spent
3. Date (the transaction date, not the posting date)
4. Income or expense type (if exists)
5. Spending category assigned by the bank (if exists)

Return ONLY a JSON object like this:
{{
    "description_col": "column name here",
    "amount_col": "column name here",
    "date_col": "column name here",
    "type_col": "column name or null if doesn't exist",
    "category_col": "column name or null if doesn't exist"
}}"""
            }
        ],
        response_format={"type": "json_object"}
    )

    # Parse AI's column mapping
    col_map = json.loads(response.choices[0].message.content)

    # Make sure the required columns the AI picked actually exist
    for key in ("description_col", "amount_col", "date_col"):
        if col_map.get(key) not in df.columns:
            return {"error": f"Could not detect {key.replace('_col', '')} column", "detected_mapping": col_map}

    # Filter only expenses if type column exists
    if col_map.get("type_col") and col_map["type_col"] in df.columns:
        type_col = col_map["type_col"]
        # Try to filter expenses (handle different naming conventions)
        expense_keywords = ["expense", "debit", "withdrawal", "dr"]
        mask = df[type_col].astype(str).str.lower().str.contains("|".join(expense_keywords), na=False)
        if mask.sum() > 0:
            df = df[mask]

    # Normalize into our schema: ISO date, numeric amount, string description
    category_col = col_map.get("category_col")
    normalized = pd.DataFrame({
        "date": pd.to_datetime(df[col_map["date_col"]], errors="coerce"),
        "description": df[col_map["description_col"]].astype(str).str.strip(),
        "amount": pd.to_numeric(df[col_map["amount_col"]], errors="coerce"),
        "category": df[category_col] if category_col in df.columns else None,
    })
    valid = normalized.dropna(subset=["date", "amount"])
    skipped = len(normalized) - len(valid)
    valid = valid.assign(date=valid["date"].dt.date, amount=valid["amount"].round(2))

    # Deterministic IDs so re-uploads (or overlapping statements) don't duplicate rows
    occurrence = valid.groupby(["date", "description", "amount"]).cumcount()
    records = [
        {
            "id": make_transaction_id(d.isoformat(), desc, float(amount), int(n)),
            "date": d,
            "description": desc,
            "amount": float(amount),
            "category": None if pd.isna(cat) else str(cat),
        }
        for d, desc, amount, cat, n in zip(valid["date"], valid["description"], valid["amount"],
                                           valid["category"], occurrence)
    ]

    # INSERT ... ON CONFLICT DO NOTHING RETURNING id: the database skips IDs it already has and
    # reports the ones it inserted. Passing a list of rows lets SQLAlchemy batch them into
    # multi-row statements ("insertmanyvalues"), which also stays under bind-parameter limits.
    with Session(engine) as session:
        if records:
            stmt = (dialect_insert(Transaction)
                    .on_conflict_do_nothing(index_elements=["id"])
                    .returning(Transaction.id))
            inserted = len(session.execute(stmt, records).all())
        else:
            inserted = 0
        session.commit()

    return {
        "rows": len(records),
        "inserted": inserted,
        "duplicates": len(records) - inserted,
        "skipped_invalid": skipped,
        "columns": columns,
        "detected_mapping": col_map
    }

# /categories endpoint - Get all categories
@app.get("/categories")
def get_all_categories():
    categories = get_categories()
    return {"categories": categories}

# Add a custom category
@app.post("/categories/add")
def add_category(name: str):
    if name in NON_SPENDING_CATEGORIES:
        return {"error": f"'{name}' is a reserved category"}
    with Session(engine) as session:
        existing = session.query(Category).filter_by(name=name).first()
        if existing:
            return {"error": f"Category '{name}' already exists"}
        session.add(Category(name=name))
        session.commit()
    return {"success": True, "message": f"Category '{name}' added"}

# Delete a category
@app.delete("/categories/delete")
def delete_category(name: str):
    with Session(engine) as session:
        category = session.query(Category).filter_by(name=name).first()
        if not category:
            return {"error": f"Category '{name}' not found"}
        session.delete(category)
        session.commit()
    return {"success": True, "message": f"Category '{name}' deleted"}


def load_user_rules(session: Session) -> list[tuple[str, str]]:
    return [(r.pattern, r.category) for r in session.query(MerchantRule).all()]


# /classify endpoint
@app.post("/classify")
def classify(reclassify: bool = False):
    """Categorize stored transactions: user rules, then built-in rules, then the LLM.

    By default only transactions without a category are processed, so classifying again
    after a new upload doesn't pay for already-labeled rows.
    Pass reclassify=true to redo everything except transactions the user fixed by hand.
    """
    categories = get_categories()

    with Session(engine) as session:
        if session.query(Transaction).count() == 0:
            return {"error": "No data uploaded yet"}

        # Only the columns we need, not full ORM objects
        query = session.query(Transaction.id, Transaction.description, Transaction.amount, Transaction.category)
        if reclassify:
            query = query.filter(or_(Transaction.category_source.is_(None),
                                     Transaction.category_source != "manual"))
        else:
            query = query.filter(Transaction.predicted_category.is_(None))
        pending = query.all()

        if pending:
            result = predict_categories(client, [(p.description, p.amount, p.category) for p in pending],
                                        categories, user_rules=load_user_rules(session))
            # Bulk UPDATE by primary key (one executemany instead of an UPDATE per ORM object)
            session.execute(update(Transaction), [
                {"id": p.id, "predicted_category": category, "category_source": source}
                for p, category, source in zip(pending, result["categories"], result["sources"])
            ])
            print(f"Classified {len(pending)} transactions "
                  f"({result['sources'].count('user_rule')} by your rules, {sum(result['by_rule'])} by built-in rules), "
                  f"{result['llm_merchants']} unique merchants in {result['llm_calls']} LLM calls")

        session.commit()

        return all_transactions(session)


@app.get("/transactions")
def list_transactions():
    with Session(engine) as session:
        return all_transactions(session)


def apply_user_rules(session: Session) -> int:
    """Re-apply all user rules to stored transactions (hand-fixed ones are left alone)."""
    rules = load_user_rules(session)
    updated = 0
    for t in session.query(Transaction).filter(or_(Transaction.category_source.is_(None),
                                                   Transaction.category_source != "manual")):
        category = match_user_rule(t.description, rules)
        if category and (category != t.predicted_category or t.category_source != "user_rule"):
            t.predicted_category = category
            t.category_source = "user_rule"
            updated += 1
    return updated


@app.post("/transactions/{transaction_id}/category")
def correct_category(transaction_id: str, category: str, apply_to_merchant: bool = True,
                           pattern: Optional[str] = None):
    """Fix a transaction's category.

    apply_to_merchant=true (default) saves a rule so every transaction whose merchant name
    contains `pattern` gets this category, now and in future uploads. `pattern` defaults to
    this transaction's merchant name; shorten it (e.g. "LYFT") to cover name variations.
    apply_to_merchant=false changes only this one transaction.
    """
    if category not in set(get_categories()) | set(NON_SPENDING_CATEGORIES):
        return {"error": f"Unknown category '{category}'"}

    with Session(engine) as session:
        t = session.get(Transaction, transaction_id)
        if not t:
            return {"error": f"Transaction '{transaction_id}' not found"}

        if not apply_to_merchant:
            t.predicted_category = category
            t.category_source = "manual"
            session.commit()
            return {"success": True, "updated": 1, "rule": None}

        pattern = " ".join((pattern or merchant_key(t.description)).upper().split())
        if not pattern:
            return {"error": "Pattern is empty"}
        if pattern not in merchant_key(t.description):
            return {"error": f"Pattern '{pattern}' doesn't match this transaction's merchant "
                             f"'{merchant_key(t.description)}'"}

        rule = session.query(MerchantRule).filter_by(pattern=pattern).first()
        if rule:
            rule.category = category
        else:
            rule = MerchantRule(pattern=pattern, category=category)
            session.add(rule)
        session.flush()

        # This transaction follows the rule from now on, even if it was hand-fixed before
        if t.category_source == "manual":
            t.category_source = None
        updated = apply_user_rules(session)
        session.commit()
        return {"success": True, "updated": updated, "rule": rule.to_dict()}


@app.get("/rules")
def list_rules():
    with Session(engine) as session:
        return {"rules": [r.to_dict() for r in session.query(MerchantRule).order_by(MerchantRule.pattern)]}


@app.delete("/rules/{rule_id}")
def delete_rule(rule_id: int):
    """Delete a rule. Transactions it labeled are cleared and get re-categorized on the next /classify
    (by another matching rule, if any, otherwise by the built-in rules / LLM)."""
    with Session(engine) as session:
        rule = session.get(MerchantRule, rule_id)
        if not rule:
            return {"error": f"Rule {rule_id} not found"}
        pattern = rule.pattern
        session.delete(rule)
        session.flush()

        cleared = 0
        for t in session.query(Transaction).filter(Transaction.category_source == "user_rule"):
            if pattern in merchant_key(t.description):
                t.predicted_category = None
                t.category_source = None
                cleared += 1
        reapplied = apply_user_rules(session)
        session.commit()
        return {"success": True, "cleared": cleared, "reassigned_by_other_rules": reapplied}

# Endpoint to find deals based on spending
@app.post("/deals")
def find_deals():
    with Session(engine) as session:
        if session.query(Transaction).count() == 0:
            return {"error": "No data uploaded yet"}

        # Top merchants by spending (purchases only, refunds/payments are negative).
        # Sum per description in SQL, then merge descriptions that clean to the same merchant.
        rows = (session.query(Transaction.description, func.sum(Transaction.amount))
                .filter(Transaction.amount > 0).group_by(Transaction.description).all())

    spend = pd.DataFrame(rows, columns=["description", "amount"])
    spend["merchant"] = spend["description"].map(clean_description)
    top_merchants = spend.groupby("merchant")["amount"].sum().nlargest(5).index.tolist()

    deals_found = {}

    # Search for deals for each top merchant
    for merchant in top_merchants:
        if not merchant or merchant == "Unknown":
            continue

        # Use Tavily to search for current deals
        search_results = tavily.search(f"{merchant} deals coupons discounts 2026", max_results=3)

        # Extract deal information
        deals = []
        for result in search_results.get("results", []):
            deals.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", "")[:200]  # First 200 characters
            })

        deals_found[merchant] = deals

    return {"deals": deals_found}

# Endpoint for chatting about expenses
@app.post("/chat")
def chat(question: str):
    with Session(engine) as session:
        if session.query(Transaction).count() == 0:
            return {"error": "No data uploaded yet"}

        # Spending by the AI-predicted category (not the bank's own category column).
        # Card payments and cashback aren't spending, so leave them out; refunds stay in and net out.
        category_expr = func.coalesce(Transaction.predicted_category, "Uncategorized")
        rows = (
            session.query(category_expr, func.sum(Transaction.amount))
            .filter(or_(
                Transaction.predicted_category.is_(None),
                Transaction.predicted_category.notin_(NON_SPENDING_CATEGORIES),
            ))
            .group_by(category_expr)
            .all()
        )
    summary = {category: round(float(total), 2) for category, total in rows}

    # Ask ChatGPT about the spending
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"You are a financial advisor. Here is the user's spending summary by category: {json.dumps(summary)}"
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return {"answer": response.choices[0].message.content}

def fraud_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based fraud flags, vectorized per merchant (O(n log n) instead of a scan per row).

    Adds a "reasons" column (list of strings) to a copy of df[date, description, amount].
    """
    df = df[["date", "description", "amount"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    by_merchant = df.groupby("description")["amount"]

    # Rule 1: duplicate — same merchant more than twice on the same day
    same_day = df.groupby(["date", "description"])["amount"].transform("size")
    # Rule 2: unusual amount — over 3x the merchant's average (and over $50)
    merchant_avg = by_merchant.transform("mean")
    # Rule 3: high frequency — more than 5 charges at the same merchant within the last 7 days
    # (counting [date - 7 days, date], same-day charges included), via binary search on sorted dates
    in_window = pd.Series(0, index=df.index)
    for _, idx in df.groupby("description").groups.items():
        dates = df.loc[idx, "date"].sort_values()
        values = dates.values
        right = np.searchsorted(values, values, side="right")
        left = np.searchsorted(values, (dates - pd.Timedelta(days=7)).values, side="left")
        in_window.loc[dates.index] = right - left

    reasons = [[] for _ in range(len(df))]
    for pos in np.flatnonzero((same_day > 2).to_numpy()):
        reasons[pos].append("Duplicate transaction: same merchant same day")
    unusual = (df["amount"] > merchant_avg * 3) & (df["amount"] > 50)
    for pos in np.flatnonzero(unusual.to_numpy()):
        reasons[pos].append(f"Unusual amount: ${df['amount'].iat[pos]:.2f} vs avg ${merchant_avg.iat[pos]:.2f}")
    for pos in np.flatnonzero((in_window > 5).to_numpy()):
        reasons[pos].append(f"High frequency: {in_window.iat[pos]} transactions in 7 days")
    df["reasons"] = reasons
    return df


@app.post("/fraud")
def detect_fraud():
    with Session(engine) as session:
        df = load_transactions_df(session)

    if df.empty:
        return {"error": "No data uploaded yet"}

    flagged = fraud_flags(df)
    flagged = flagged[flagged["reasons"].map(len) > 0]
    return {
        "total_transactions": len(df),
        "flagged_count": len(flagged),
        "flagged_transactions": [
            {"date": str(row.date.date()), "merchant": row.description, "amount": row.amount,
             "flagged": True, "reasons": row.reasons}
            for row in flagged.itertuples()
        ],
    }
