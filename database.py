import os
import re
import hashlib
from sqlalchemy import create_engine, inspect, text, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase, Session
from datetime import datetime

# SQLite by default; override with DATABASE_URL (e.g. for tests or Postgres later)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///expense_tracker.db")
engine = create_engine(DATABASE_URL)

class Base(DeclarativeBase):
    pass

# Table for storing transactions
class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String, primary_key=True)    # deterministic hash, see make_transaction_id
    date = Column(String)                    # ISO format: YYYY-MM-DD
    description = Column(String)
    amount = Column(Float)
    category = Column(String)                # category provided by the bank CSV, if any
    predicted_category = Column(String)      # category assigned by us (see category_source)
    category_source = Column(String)         # "manual" | "user_rule" | "rule" | "llm"
    created_at = Column(DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "date": self.date,
            "description": self.description,
            "amount": self.amount,
            "category": self.category,
            "predicted_category": self.predicted_category,
            "category_source": self.category_source,
        }

# Table for storing user's custom categories
class Category(Base):
    __tablename__ = "categories"
    name = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.now)

# Categories the user taught us: any transaction whose merchant name contains `pattern`
# gets `category`. Patterns are stored normalized (see classifier.merchant_key).
class MerchantRule(Base):
    __tablename__ = "merchant_rules"
    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern = Column(String, unique=True, nullable=False)
    category = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {"id": self.id, "pattern": self.pattern, "category": self.category}

# Default categories
DEFAULT_CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Transportation",
    "Shopping",
    "Entertainment",
    "Software & Subscriptions",
    "Insurance",
    "Utilities",
    "Healthcare",
    "Travel",
    "Other"
]

# Reserved categories for money that isn't spending. Excluded from spending totals.
# Refunds are NOT in here: they keep the merchant's category so they net out that spending.
CARD_PAYMENT = "Card Payment"          # paying off the card bill
REWARDS = "Rewards & Cashback"         # cashback / statement credits
NON_SPENDING_CATEGORIES = [CARD_PAYMENT, REWARDS]

# Deterministic rules, applied to negative amounts before asking the LLM.
# Deliberately specific: refunds look like "AMAZON MKTPLACE PMTS", so never match a bare "PMT".
_PAYMENT_RE = re.compile(
    r"PAYMENT\s*-\s*THANK YOU|AUTO-?PAY|AUTOMATIC PAYMENT|ONLINE PAYMENT|INTERNET PAYMENT"
    r"|MOBILE PAYMENT|PAYMENT RECEIVED",
    re.IGNORECASE,
)
_REWARDS_RE = re.compile(r"CASH\s?BACK|REWARD|REBATE|STATEMENT CREDIT|STMT CRDT", re.IGNORECASE)

def rule_based_category(description: str, amount: float):
    """Return a reserved category if the transaction is clearly not spending, else None."""
    if amount is None or amount >= 0:
        return None
    if _REWARDS_RE.search(description or ""):
        return REWARDS
    if _PAYMENT_RE.search(description or ""):
        return CARD_PAYMENT
    return None

def make_transaction_id(date: str, description: str, amount: float, occurrence: int) -> str:
    """Same transaction -> same ID, so re-uploading a statement doesn't create duplicates.

    `occurrence` is the n-th time this exact (date, description, amount) appears in the
    uploaded file. It keeps two genuine identical purchases on the same day (e.g. two
    coffees) as separate rows, while a re-upload of the same file still maps to the same IDs.
    """
    key = f"{date}|{description}|{amount:.2f}|{occurrence}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

def init_db():
    # Create all tables
    Base.metadata.create_all(engine)

    # create_all doesn't add new columns to an existing table, so add them by hand
    columns = {c["name"] for c in inspect(engine).get_columns("transactions")}
    if "category_source" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE transactions ADD COLUMN category_source VARCHAR"))

    # Add default categories if none exist
    with Session(engine) as session:
        existing = session.query(Category).count()
        if existing == 0:
            for cat in DEFAULT_CATEGORIES:
                session.add(Category(name=cat))
            session.commit()
            print("Default categories added")

def get_categories():
    with Session(engine) as session:
        categories = session.query(Category).all()
        return [c.name for c in categories]
