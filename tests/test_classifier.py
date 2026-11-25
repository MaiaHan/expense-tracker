"""Unit tests for the categorization rules and pipeline (no DB, fake LLM)."""
import types

from conftest import FakeLLM
from classifier import clean_description, match_user_rule, merchant_key, predict_categories
from database import CARD_PAYMENT, REWARDS, make_transaction_id, rule_based_category

CATEGORIES = ["Food & Dining", "Groceries", "Shopping", "Transportation", "Other"]


def fake_client(fake: FakeLLM):
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake.create)))


# ── built-in rules ──

def test_card_payment_and_cashback_rules():
    assert rule_based_category("INTERNET PAYMENT - THANK YOU", -100) == CARD_PAYMENT
    assert rule_based_category("CASHBACK BONUS REDEMPTION PYMT/STMT CRDT", -5) == REWARDS


def test_refund_is_not_a_card_payment():
    # Discover labels refunds "Payments and Credits" and their descriptions contain "PMTS"
    assert rule_based_category("AMAZON MKTPLACE PMTS AMZN.COM/BILLWA", -25) is None


def test_rules_ignore_purchases():
    assert rule_based_category("INTERNET PAYMENT - THANK YOU", 100) is None


def test_transaction_id_is_deterministic_and_counts_duplicates():
    a = make_transaction_id("2026-11-01", "SUBWAY", 12.5, 0)
    assert a == make_transaction_id("2026-11-01", "SUBWAY", 12.50, 0)
    assert a != make_transaction_id("2026-11-01", "SUBWAY", 12.5, 1)


def test_clean_description_strips_noise():
    assert clean_description("CITY METRO TRANSITAPPLE PAY ENDING IN 0000") == "CITY METRO TRANSIT"
    assert clean_description("LYFT *RIDE BOB 8005550100 VA") == "LYFT *RIDE BOB  VA"
    assert merchant_key("LYFT *RIDE BOB 8005550100 VA") == "LYFT *RIDE BOB VA"


# ── user rules ──

def test_user_rule_matches_substring_and_longest_wins():
    rules = [("LYFT", "Transportation"), ("LYFT *RIDE CAROL", "Shopping")]
    assert match_user_rule("LYFT *RIDE BOB 8005550100 VA", rules) == "Transportation"
    assert match_user_rule("lyft *ride carol 8005550100", rules) == "Shopping"
    assert match_user_rule("UBER TRIP", rules) is None


# ── pipeline ──

def test_pipeline_order_and_llm_dedupe():
    fake = FakeLLM()
    fake.categorize = lambda it: "Food & Dining" if "SUBWAY" in it["description"] else "Other"
    transactions = [
        ("LYFT *RIDE BOB 8005550100 VA", 20.0),
        ("INTERNET PAYMENT - THANK YOU", -100.0),
        ("SUBWAY 41234", 12.5),
        ("SUBWAY 41234", 9.0),
    ]
    result = predict_categories(fake_client(fake), transactions, CATEGORIES,
                                user_rules=[("LYFT", "Transportation")])
    assert result["categories"] == ["Transportation", CARD_PAYMENT, "Food & Dining", "Food & Dining"]
    assert result["sources"] == ["user_rule", "rule", "llm", "llm"]
    assert fake.asked == ["SUBWAY 41234"]  # each merchant asked once; rule-labeled rows never asked
    assert result["llm_calls"] == 1


def test_invalid_skipped_or_unsafe_llm_answers_become_other():
    fake = FakeLLM()
    answers = {"MADE UP": "Not A Category", "SKIPPED": None, "PAYPAL": CARD_PAYMENT}
    fake.categorize = lambda it: answers[it["description"]]
    result = predict_categories(fake_client(fake), [("MADE UP", 1.0), ("SKIPPED", 1.0), ("PAYPAL", 20.0)],
                                CATEGORIES)
    assert result["categories"] == ["Other", "Other", "Other"]
    assert result["unanswered"] == 1


def test_rules_for_deleted_categories_are_ignored():
    fake = FakeLLM()
    result = predict_categories(fake_client(fake), [("LYFT *RIDE X", 5.0)], CATEGORIES,
                                user_rules=[("LYFT", "Deleted Category")])
    assert result["sources"] == ["llm"]


def test_bank_hint_only_sent_when_enabled():
    fake = FakeLLM()
    predict_categories(fake_client(fake), [("ACME HEALTH E-BILL", 90.0, "Medical Services")], CATEGORIES)
    predict_categories(fake_client(fake), [("ACME HEALTH E-BILL", 90.0, "Medical Services")], CATEGORIES,
                       use_bank_hint=True)
    assert "bank_category" not in fake.requests[0][0]
    assert fake.requests[1][0]["bank_category"] == "Medical Services"
