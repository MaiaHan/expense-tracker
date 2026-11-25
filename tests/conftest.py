"""Shared test setup: a throwaway SQLite DB and a fake LLM, so tests never call real APIs."""
import json
import os
import sys
import types

import pytest

# Must be set before importing the app: the DB engine and API clients are created at import time
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(os.path.dirname(__file__), ".test.db")
os.environ["OPENAI_API_KEY"] = "test-not-a-real-key"
os.environ["TAVILY_API_KEY"] = "test-not-a-real-key"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.path.exists(os.environ["DATABASE_URL"][len("sqlite:///"):]):
    os.remove(os.environ["DATABASE_URL"][len("sqlite:///"):])

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

    @property
    def asked(self):
        return [item["description"] for batch in self.requests for item in batch]

    def create(self, model, messages, response_format=None):
        content = messages[-1]["content"]
        if "bank CSV file" in content:
            return _response(json.dumps(self.COLUMN_MAPPING))
        if content.startswith("Categorize these transactions"):
            items = json.loads(content.split(": ", 1)[1])
            self.requests.append(items)
            answers = {str(it["id"]): self.categorize(it) for it in items}
            return _response(json.dumps({k: v for k, v in answers.items() if v is not None}))
        return _response("fake answer to: " + messages[0]["content"])


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(main.client.chat.completions, "create", fake.create)
    monkeypatch.setattr(main.tavily, "search", lambda q, max_results=3: {"results": []})
    return fake


@pytest.fixture
def client(llm):
    """API client on an empty database with default categories."""
    with database.Session(database.engine) as session:
        for model in (database.Transaction, database.MerchantRule, database.Category):
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
