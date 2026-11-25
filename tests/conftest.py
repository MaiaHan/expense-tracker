"""Shared test setup: a throwaway SQLite DB and a fake LLM, so tests never call real APIs."""
import json
import os
import sys
import types

import pytest

# Must be set before importing the app: the DB engine and API clients are created at import time.
# SQLite by default; set TEST_DATABASE_URL to run the suite against Postgres (its tables get wiped).
os.environ["DATABASE_URL"] = os.getenv("TEST_DATABASE_URL") or (
    "sqlite:///" + os.path.join(os.path.dirname(__file__), ".test.db"))
os.environ["OPENAI_API_KEY"] = "test-not-a-real-key"
os.environ["TAVILY_API_KEY"] = "test-not-a-real-key"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.environ["DATABASE_URL"].startswith("sqlite:///") and os.path.exists(os.environ["DATABASE_URL"][10:]):
    os.remove(os.environ["DATABASE_URL"][10:])

import database  # noqa: E402
import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _response(content: str):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])


class FakeLLM:
    """Stands in for client.chat.completions.create.

    `categorize(item)` decides the category for one {"id", "description", ...} item;
    return None to simulate the LLM skipping it. `requests` records every categorize payload.
    """
    COLUMN_MAPPING = {"description_col": "Description", "amount_col": "Amount", "date_col": "Trans. Date",
                      "type_col": None, "category_col": "Category"}

    def __init__(self):
        self.categorize = lambda item: "Other"
        self.requests = []
        # deals: identify(merchant) -> {"brand", "chain"}; extract(brand, pages) -> list of deal dicts
        self.identify = lambda merchant: {"brand": None, "chain": False}
        self.extract = lambda brand, pages: []
        self.identified = []

    @property
    def asked(self):
        return [item["description"] for batch in self.requests for item in batch]

    def create(self, model, messages, response_format=None):
        # Same rule as the real API: JSON mode requires the word "json" somewhere in the messages
        if response_format == {"type": "json_object"}:
            assert any("json" in m["content"].lower() for m in messages), \
                "OpenAI rejects json_object requests whose messages don't mention JSON"
        content = messages[-1]["content"]
        if "bank CSV file" in content:
            return _response(json.dumps(self.COLUMN_MAPPING))
        if content.startswith("Categorize these transactions"):
            items = json.loads(content.split(": ", 1)[1])
            self.requests.append(items)
            answers = {str(it["id"]): self.categorize(it) for it in items}
            return _response(json.dumps({k: v for k, v in answers.items() if v is not None}))
        if content.startswith("Identify these merchants"):
            items = json.loads(content.split(": ", 1)[1])
            self.identified.extend(it["merchant"] for it in items)
            return _response(json.dumps({str(it["id"]): self.identify(it["merchant"]) for it in items}))
        if content.startswith("Pages: "):
            brand = messages[0]["content"].split("Extract deals and promo codes for ", 1)[1].split(" from the web", 1)[0]
            return _response(json.dumps({"deals": self.extract(brand, json.loads(content[len("Pages: "):]))}))
        return _response("fake answer to: " + messages[0]["content"])


class FakeTavily:
    """pages[brand] = list of {"url", "title", "raw_content"}; raise_for = brands whose search fails."""

    def __init__(self):
        self.pages = {}
        self.raise_for = set()
        self.queries = []

    def search(self, query, **kwargs):
        self.queries.append(query)
        self.kwargs = kwargs
        brand = next((b for b in sorted(set(self.pages) | self.raise_for, key=len, reverse=True)
                      if query.startswith(b + " ")), None)
        if brand in self.raise_for:
            raise RuntimeError("search service unavailable")
        return {"results": self.pages.get(brand, [])}


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(main.client.chat.completions, "create", fake.create)
    return fake


@pytest.fixture
def web(monkeypatch):
    fake = FakeTavily()
    monkeypatch.setattr(main, "tavily", fake)
    return fake


@pytest.fixture
def client(llm):
    """API client on an empty database with default categories."""
    with database.Session(database.engine) as session:
        for model in (database.Transaction, database.MerchantRule, database.Category,
                      database.Deal, database.DealSearch, database.MerchantProfile):
            session.query(model).delete()
        session.commit()
    database.init_db()
    return TestClient(main.app)


HEADER = "Trans. Date,Post Date,Description,Amount,Category\n"


def upload(client, rows: str):
    """Upload CSV rows (without header) in Discover's format."""
    return client.post("/upload", files={"file": ("statement.csv", HEADER + rows, "text/csv")}).json()


def by_description(client) -> dict:
    return {t["description"]: t for t in client.get("/transactions").json()}
