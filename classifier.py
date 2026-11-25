"""Transaction categorization: deterministic rules first, then the LLM.

Shared by the /classify endpoint and evaluate.py, so the evaluation measures
exactly the code path the app uses.
"""
from __future__ import annotations

import re
import json
from database import CARD_PAYMENT, NON_SPENDING_CATEGORIES, rule_based_category

MODEL = "gpt-4o-mini"
# Max number of merchant descriptions sent to the LLM in one request
CLASSIFY_BATCH_SIZE = 100


def clean_description(text):
    text = re.sub(r'APPLE PAY ENDING IN \w+', '', str(text))
    text = re.sub(r'\d{10,}', '', text)
    text = re.sub(r'\d{3}-\d{3}-\d{4}', '', text)
    return text.strip()


def merchant_key(description: str) -> str:
    """Normalized merchant text that user rules are matched against."""
    return " ".join(clean_description(description).upper().split())


def match_user_rule(description: str, rules: list[tuple[str, str]]):
    """Category of the most specific (longest) rule pattern contained in the merchant name."""
    key = merchant_key(description)
    best = None
    for pattern, category in rules:
        if pattern and pattern in key and (best is None or len(pattern) > len(best[0])):
            best = (pattern, category)
    return best[1] if best else None


def categorize_descriptions(client, items: list[tuple[str, str | None]], categories: list[str],
                            use_bank_hint: bool = False) -> tuple[dict, dict]:
    """Ask the LLM to categorize (description, bank_category) items, in batches.

    Items are sent numbered and the LLM answers by number, so a merchant name it
    rewrites slightly (e.g. collapsed spaces) can't silently fall back to "Other".
    Returns ({item: category}, stats) where stats counts LLM calls and unanswered items.
    """
    hint_rule = (
        'Some items include "bank_category", the category the card issuer assigned. It is often '
        'right for unfamiliar merchant names, but it is coarse (e.g. software subscriptions are '
        'often labeled "Merchandise", transit and rideshare "Travel/ Entertainment") - use it as a '
        "hint, not the answer.\n"
    ) if use_bank_hint else ""
    result = {}
    stats = {"llm_calls": 0, "unanswered": 0}
    for start in range(0, len(items), CLASSIFY_BATCH_SIZE):
        batch = items[start:start + CLASSIFY_BATCH_SIZE]
        payload = []
        for i, (description, bank_category) in enumerate(batch):
            entry = {"id": i, "description": description}
            if use_bank_hint and bank_category:
                entry["bank_category"] = bank_category
            payload.append(entry)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expense categorizer.
You will receive a list of credit card transactions, each with an id and a merchant description.
Categorize each one into ONE of these categories: {', '.join(categories)}.
Use "{CARD_PAYMENT}" only for payments toward the credit card bill itself.
Refunds from a merchant are NOT card payments: give them the merchant's normal category.
{hint_rule}Return ONLY a JSON object mapping each id (as a string) to its category.
Example: {{"0": "Food & Dining", "1": "Transportation"}}"""
                },
                {
                    "role": "user",
                    "content": f"Categorize these transactions: {json.dumps(payload)}"
                }
            ],
            response_format={"type": "json_object"}
        )
        stats["llm_calls"] += 1
        answers = json.loads(response.choices[0].message.content)
        for i, item in enumerate(batch):
            if str(i) in answers:
                result[item] = answers[str(i)]
            else:
                stats["unanswered"] += 1
    return result, stats


def predict_categories(client, transactions: list[tuple], categories: list[str],
                       use_bank_hint: bool = False, user_rules: list[tuple[str, str]] | None = None) -> dict:
    """Categorize (description, amount[, bank_category]) tuples.

    Order: the user's merchant rules, then built-in card payment / cashback rules, then the LLM.
    Returns {"categories": [...], "sources": ["user_rule" | "rule" | "llm", ...], "by_rule": [bool, ...],
             "llm_calls": int, "llm_merchants": int, "unanswered": int}.
    """
    llm_categories = categories + [CARD_PAYMENT]
    allowed = set(categories) | set(NON_SPENDING_CATEGORIES)
    # Ignore rules pointing at a category the user has since deleted
    user_rules = [(p, cat) for p, cat in (user_rules or []) if cat in allowed]

    predicted, sources = [], []
    for t in transactions:
        category = match_user_rule(t[0], user_rules)
        if category:
            predicted.append(category); sources.append("user_rule")
            continue
        # Clear-cut card payments / cashback are labeled by rule, no LLM needed
        category = rule_based_category(t[0], t[1])
        predicted.append(category); sources.append("rule" if category else "llm")

    # Everything else goes to the LLM, once per unique (cleaned description, bank category)
    def bank_category(t):
        value = t[2] if len(t) > 2 else None
        return value if isinstance(value, str) and value.strip() else None

    keys = [(clean_description(t[0]), bank_category(t) if use_bank_hint else None) for t in transactions]
    to_ask = sorted({k for k, p in zip(keys, predicted) if p is None}, key=lambda k: (k[0], k[1] or ""))
    if to_ask:
        llm_result, stats = categorize_descriptions(client, to_ask, llm_categories, use_bank_hint)
    else:
        llm_result, stats = {}, {"llm_calls": 0, "unanswered": 0}

    for i, t in enumerate(transactions):
        if predicted[i] is not None:
            continue
        category = llm_result.get(keys[i], "Other")
        if category not in llm_categories:
            category = "Other"
        # A purchase (positive amount) can never be a card payment
        if category == CARD_PAYMENT and t[1] >= 0:
            category = "Other"
        predicted[i] = category

    return {"categories": predicted, "sources": sources, "by_rule": [s == "rule" for s in sources],
            "llm_merchants": len(to_ask), **stats}
