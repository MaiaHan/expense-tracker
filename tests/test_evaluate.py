"""evaluate.py end to end with a fake LLM on a tiny labels file."""
import sys
import types

import pytest

import evaluate
from conftest import FakeLLM

LABELS = """date,description,amount,bank_category,true_category,label_source
2026-01-01,SAFEWAY 1234,50.0,Supermarkets,Groceries,bank
2026-01-02,LYFT *RIDE ALICE 8005550100 VA,18.0,Travel/ Entertainment,Transportation,manual
2026-01-03,INTERNET PAYMENT - THANK YOU,-100.0,Payments and Credits,Card Payment,manual
2026-02-01,SAFEWAY 1234,30.0,Supermarkets,Groceries,bank
2026-02-02,LYFT *RIDE ALICE 8005550100 VA,25.0,Travel/ Entertainment,Transportation,manual
2026-02-03,LYFT *RIDE BOB 8005550100 VA,20.0,Travel/ Entertainment,Transportation,manual
"""


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    fake = FakeLLM()
    fake.categorize = lambda it: "Groceries" if "SAFEWAY" in it["description"] else "Other"
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake.create)))
    monkeypatch.setattr(evaluate, "OpenAI", lambda api_key=None: client)
    path = tmp_path / "labels.csv"
    path.write_text(LABELS)

    def _run(*flags):
        monkeypatch.setattr(sys, "argv", ["evaluate.py", str(path), "--default-categories", *flags])
        evaluate.main()
        return capsys.readouterr().out, fake
    return _run


def test_reports_accuracy_split_by_rules_and_label_source(run):
    out, fake = run()
    assert "Overall accuracy:        50.0%  (3/6 transactions)" in out
    assert "rules only:            100.0%" in out
    assert "LLM, manual labels:" in out and "0.0%" in out
    assert all("bank_category" not in item for batch in fake.requests for item in batch)


def test_bank_hint_flag_sends_bank_category(run):
    _, fake = run("--bank-hint")
    assert all("bank_category" in item for batch in fake.requests for item in batch)


def test_simulated_corrections(run):
    out, _ = run("--simulate-corrections", "--split-date", "2026-02-01")
    # January: LYFT* ALICE is wrong -> one rule for that exact merchant name.
    # February: it fixes ALICE, but not BOB (different name), so 1/3 -> 2/3
    assert "user fixes 1 -> 1 merchant rules" in out
    assert "3 transactions, 1 matched a rule" in out
    assert "later accuracy without rules: 33.3%" in out
    assert "later accuracy with rules:    66.7%" in out
