import os
import io
import json
import pandas as pd
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from database import (
    init_db, get_categories, engine, Transaction, Category, make_transaction_id,
    NON_SPENDING_CATEGORIES,
)
from classifier import clean_description, predict_categories

# Load API keys from .env file
load_dotenv()

# Initialize database when server starts
init_db()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
app = FastAPI()

def load_transactions_df(session: Session) -> pd.DataFrame:
    """All stored transactions as a DataFrame (empty DataFrame if none)."""
    rows = [t.to_dict() for t in session.query(Transaction).order_by(Transaction.date).all()]
    return pd.DataFrame(rows, columns=["id", "date", "description", "amount", "category", "predicted_category"])


# /upload endpoint
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Read the uploaded CSV file
    contents = await file.read()
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
    valid = valid.assign(date=valid["date"].dt.strftime("%Y-%m-%d"))

    # Build deterministic IDs so re-uploads (or overlapping statements) don't duplicate rows
    occurrence = valid.groupby(["date", "description", "amount"]).cumcount()
    records = []
    for (_, row), n in zip(valid.iterrows(), occurrence):
        records.append(Transaction(
            id=make_transaction_id(row["date"], row["description"], row["amount"], n),
            date=row["date"],
            description=row["description"],
            amount=float(row["amount"]),
            category=None if pd.isna(row["category"]) else str(row["category"]),
        ))

    with Session(engine) as session:
        ids = [r.id for r in records]
        existing = {
            tid for (tid,) in session.query(Transaction.id).filter(Transaction.id.in_(ids)).all()
        } if ids else set()
        new_records = [r for r in records if r.id not in existing]
        session.add_all(new_records)
        session.commit()

    return {
        "rows": len(records),
        "inserted": len(new_records),
        "duplicates": len(records) - len(new_records),
        "skipped_invalid": skipped,
        "columns": columns,
        "detected_mapping": col_map
    }

# /categories endpoint - Get all categories
@app.get("/categories")
async def get_all_categories():
    categories = get_categories()
    return {"categories": categories}

# Add a custom category
@app.post("/categories/add")
async def add_category(name: str):
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
async def delete_category(name: str):
    with Session(engine) as session:
        category = session.query(Category).filter_by(name=name).first()
        if not category:
            return {"error": f"Category '{name}' not found"}
        session.delete(category)
        session.commit()
    return {"success": True, "message": f"Category '{name}' deleted"}


# /classify endpoint
@app.post("/classify")
async def classify(reclassify: bool = False):
    """Categorize stored transactions with the LLM.

    By default only transactions without a predicted category are sent to the LLM,
    so classifying again after a new upload doesn't pay for already-labeled rows.
    Pass reclassify=true to redo everything (e.g. after changing categories).
    """
    categories = get_categories()

    with Session(engine) as session:
        if session.query(Transaction).count() == 0:
            return {"error": "No data uploaded yet"}

        query = session.query(Transaction)
        if not reclassify:
            query = query.filter(Transaction.predicted_category.is_(None))
        pending = query.all()

        if pending:
            result = predict_categories(client, [(t.description, t.amount, t.category) for t in pending], categories)
            for t, category in zip(pending, result["categories"]):
                t.predicted_category = category
            print(f"Classified {len(pending)} transactions ({sum(result['by_rule'])} by rule), "
                  f"{result['llm_merchants']} unique merchants in {result['llm_calls']} LLM calls")

        session.commit()

        return [t.to_dict() for t in session.query(Transaction).order_by(Transaction.date).all()]

# Endpoint to find deals based on spending
@app.post("/deals")
async def find_deals():
    with Session(engine) as session:
        if session.query(Transaction).count() == 0:
            return {"error": "No data uploaded yet"}

        # Get top merchants by spending (purchases only, refunds/payments are negative)
        rows = session.query(Transaction.description, Transaction.amount).filter(Transaction.amount > 0).all()

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
async def chat(question: str):
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
    summary = {category: round(total, 2) for category, total in rows}

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

@app.post("/fraud")
async def detect_fraud():
    with Session(engine) as session:
        df_fraud = load_transactions_df(session)

    if df_fraud.empty:
        return {"error": "No data uploaded yet"}

    df_fraud["date"] = pd.to_datetime(df_fraud["date"])

    flags = []

    for index, row in df_fraud.iterrows():
        reasons = []

        # Rule 1: 重复交易 — 同一天同一商家超过2次
        same_day_same_merchant = df_fraud[
            (df_fraud["date"] == row["date"]) &
            (df_fraud["description"] == row["description"])
        ]
        if len(same_day_same_merchant) > 2:
            reasons.append("Duplicate transaction: same merchant same day")

        # Rule 2: 金额异常 — 超过该商家平均金额3倍
        merchant_avg = df_fraud[
            df_fraud["description"] == row["description"]
        ]["amount"].mean()
        if row["amount"] > merchant_avg * 3 and row["amount"] > 50:
            reasons.append(f"Unusual amount: ${row['amount']:.2f} vs avg ${merchant_avg:.2f}")

        # Rule 3: 高频消费 — 7天内同一商家超过5次
        week_ago = row["date"] - pd.Timedelta(days=7)
        recent_same_merchant = df_fraud[
            (df_fraud["description"] == row["description"]) &
            (df_fraud["date"] >= week_ago) &
            (df_fraud["date"] <= row["date"])
        ]
        if len(recent_same_merchant) > 5:
            reasons.append(f"High frequency: {len(recent_same_merchant)} transactions in 7 days")

        flags.append({
            "date": str(row["date"].date()),
            "merchant": row["description"],
            "amount": row["amount"],
            "flagged": len(reasons) > 0,
            "reasons": reasons
        })

    flagged = [f for f in flags if f["flagged"]]

    return {
        "total_transactions": len(flags),
        "flagged_count": len(flagged),
        "flagged_transactions": flagged
    }
