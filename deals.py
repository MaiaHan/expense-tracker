"""Find coupons and deals for the chains you actually visit.

Pipeline per brand: web search -> LLM extracts structured deals -> deterministic verification.
LLMs readily invent plausible promo codes, so a deal is kept only if its code and an evidence
quote appear verbatim in the page it came from. That proves the page says it, not that the
code still works: coupon sites list plenty of dead codes.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import date

from classifier import MODEL

SEARCH_RESULTS = 5            # pages per brand
PAGE_CHARS = 8000             # page text sent to the LLM, per page
MIN_QUOTE_CHARS = 15
DISCOUNT_TYPES = {"percent", "amount", "bogo", "free_item", "other"}
NOT_A_CODE = {"", "NONE", "N/A", "NA", "NULL", "NO CODE", "NO CODE NEEDED", "AUTOMATIC"}


def _ask_json(client, system: str, user: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ── 1. which merchants are chains worth searching ──

def identify_merchants(client, merchants: list[str]) -> dict[str, dict]:
    """{merchant: {"brand": str | None, "kind": str | None, "is_chain": bool}} for card-statement names."""
    if not merchants:
        return {}
    payload = [{"id": i, "merchant": m} for i, m in enumerate(merchants)]
    answers = _ask_json(
        client,
        """You identify merchants from credit card statement descriptions.
For each item give the brand name people would search for (e.g. "MCDONALD'S F1234 SEATTLE WA" -> "McDonald's",
"TST* PISCO DC WASHINGTON DC" -> "Pisco"), what kind of business it is in a few words (e.g. "fast food
restaurant", "gas station", "grocery store", "online marketplace"), and whether it is a national or
regional chain or online service that might publish promo codes (true) or a single local business (false).
Return ONLY a JSON object mapping each id (as a string) to {"brand": "...", "kind": "...", "chain": true/false};
use null for brand or kind if you can't tell.""",
        f"Identify these merchants: {json.dumps(payload)}",
    )
    result = {}
    for i, merchant in enumerate(merchants):
        answer = answers.get(str(i)) or {}
        brand = answer.get("brand")
        brand = brand.strip() if isinstance(brand, str) and brand.strip() else None
        kind = answer.get("kind")
        kind = kind.strip() if isinstance(kind, str) and kind.strip() else None
        result[merchant] = {"brand": brand, "kind": kind, "is_chain": bool(answer.get("chain")) and brand is not None}
    return result


# ── 2. search ──

def search_pages(tavily, brand: str, kind: str | None = None) -> list[dict]:
    """Recent web pages about the brand's promo codes, with their full text.

    The kind of business ("ARCO gas station") keeps same-name companies out of the results;
    country only *boosts* US pages, so verify_deal still rejects non-US ones.
    """
    response = tavily.search(
        f"{brand} {kind} promo code coupon" if kind else f"{brand} promo code coupon",
        country="united states",
        exclude_domains=["tavily.com"],   # its company profile pages show up for every brand
        max_results=SEARCH_RESULTS,
        time_range="month",
        include_raw_content="text",
    )
    pages = []
    for result in response.get("results", []):
        text = result.get("raw_content") or result.get("content") or ""
        if text.strip() and result.get("url"):
            pages.append({"url": result["url"], "title": result.get("title", ""), "text": text})
    return pages


# ── 3. extract ──

def extract_deals(client, brand: str, pages: list[dict], today: date, kind: str | None = None) -> list[dict]:
    """Candidate deals as the LLM read them. Not trustworthy until verify_deal passes."""
    if not pages:
        return []
    payload = [{"page": i, "title": p["title"], "text": p["text"][:PAGE_CHARS]} for i, p in enumerate(pages)]
    answer = _ask_json(
        client,
        f"""Today is {today.isoformat()}. Extract deals and promo codes for {brand} from the web pages.
{brand} is {f"a {kind}" if kind else "a business"} in the United States. Only include deals the page states
explicitly for this {brand}, usable in the US. Ignore deals for other businesses with a similar name or
abbreviation, and deals in other countries or currencies. Skip discounts on one specific product (e.g.
"43% off a pair of sneakers"); keep store-wide, category-wide, membership or first-order deals.
If {brand} doesn't offer such deals, return none.
Never guess or complete a code.
For each deal return:
  "page": the page number it came from,
  "code": the promo code copied exactly, or null if no code is needed,
  "description": a short plain description (e.g. "20% off orders over $30"),
  "discount_type": one of percent, amount, bogo, free_item, other,
  "discount_value": the number (20 for 20% or $20), or null,
  "min_spend": minimum purchase in dollars, or null,
  "expires_on": "YYYY-MM-DD" if the page gives an expiry date, else null,
  "conditions": other restrictions in a few words, or null,
  "quote": one sentence copied verbatim from the page that states this deal, including its code and amount.
Return ONLY a JSON object {{"deals": [...]}}; an empty list if the pages have no concrete deals.""",
        f"Pages: {json.dumps(payload)}",
    )
    deals = answer.get("deals", [])
    return deals if isinstance(deals, list) else []


# ── 4. verify ──

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                         "–": "-", "—": "-", " ": " "}))
    return " ".join(text.lower().split())


def _number(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


_FOREIGN_CURRENCY = re.compile(r"[£€¥₹]|\b(?:gbp|eur|cad|aud|inr|nzd)\b", re.IGNORECASE)


def _compact(text: str) -> str:
    """Lowercase letters and digits only, for matching brand names ("Trader Joe's" -> "traderjoes")."""
    return re.sub(r"[^a-z0-9]", "", _normalize(text))


def _number_in_quote(discount_type: str, value: float, quote: str) -> bool:
    number = f"{value:g}"
    if discount_type == "percent":
        return re.search(rf"(?<![\d.]){re.escape(number)}\s*(?:%|percent)", quote) is not None
    if discount_type == "amount":   # "$10", "$10.00", or cents: "10¢", "10 cents" (fuel rewards)
        return re.search(rf"\$\s*{re.escape(number)}(?:\.00)?(?![\d])"
                         rf"|(?<![\d.]){re.escape(number)}\s*(?:¢|cents?\b)", quote) is not None
    return True


def verify_deal(candidate: dict, pages: list[dict], today: date, brand: str | None = None
                ) -> tuple[dict | None, str | None]:
    """Return (clean deal, None) if the page backs it up, else (None, rejection reason).

    Checks, all deterministic: the quote is on the page; the code is in the quote; there is a code
    or a concrete discount, and the stated discount and minimum spend match the quote; the brand is named in the page title, URL or quote; the deal and
    page are in US dollars; it hasn't expired.
    """
    if not isinstance(candidate, dict):
        return None, "malformed"
    page_index = candidate.get("page")
    if not isinstance(page_index, int) or not 0 <= page_index < len(pages):
        return None, "unknown page"
    page = pages[page_index]
    page_text = _normalize(page["text"])

    description = (candidate.get("description") or "").strip()
    quote = (candidate.get("quote") or "").strip()
    if not description:
        return None, "no description"
    if len(quote) < MIN_QUOTE_CHARS or _normalize(quote) not in page_text:
        return None, "quote not in page"

    code = (candidate.get("code") or "").strip()
    if code.upper() in NOT_A_CODE:
        code = None
    if code is not None:
        if len(code) < 3 or not re.search(rf"(?<![a-z0-9]){re.escape(_normalize(code))}(?![a-z0-9])", page_text):
            return None, "code not in page"
        if not re.search(rf"(?<![a-z0-9]){re.escape(_normalize(code))}(?![a-z0-9])", _normalize(quote)):
            return None, "code not in quote"

    if brand and _compact(brand) not in _compact(f"{page['title']} {page['url']} {quote}"):
        return None, "brand not on page"
    if _FOREIGN_CURRENCY.search(quote):
        return None, "non-US deal"
    if len(_FOREIGN_CURRENCY.findall(page["text"])) > page["text"].count("$"):
        return None, "non-US page"

    discount_type = candidate.get("discount_type")
    discount_value = _number(candidate.get("discount_value"))
    if code is None and discount_value is None and discount_type not in ("free_item", "bogo"):
        return None, "no concrete discount"   # e.g. "a students program with exclusive savings"
    if discount_value is not None and not _number_in_quote(discount_type, discount_value, _normalize(quote)):
        return None, "discount not in quote"
    min_spend = _number(candidate.get("min_spend"))
    if min_spend is not None and not _number_in_quote("amount", min_spend, _normalize(quote)):
        return None, "min spend not in quote"

    expires_on = None
    if candidate.get("expires_on"):
        try:
            expires_on = date.fromisoformat(str(candidate["expires_on"]))
        except ValueError:
            expires_on = None
    if expires_on and expires_on < today:
        return None, "expired"

    return {
        "code": code,
        "description": description,
        "discount_type": discount_type if discount_type in DISCOUNT_TYPES else "other",
        "discount_value": discount_value,
        "min_spend": min_spend,
        "expires_on": expires_on,
        "conditions": (candidate.get("conditions") or None),
        "quote": quote,
        "source_url": page["url"],
        "source_title": page["title"],
        "dedupe_key": code.upper() if code else _normalize(description),
    }, None


def find_deals_for_brand(client, tavily, brand: str, today: date, kind: str | None = None,
                         debug_dir: str | None = None) -> dict:
    """Search, extract and verify. Returns {"deals": [...], "candidates": n, "rejected": {reason: n}}.

    With debug_dir, the pages, candidates and verdicts are saved as JSON so verification changes
    can be re-checked on the same pages without paying for another search.
    """
    pages = search_pages(tavily, brand, kind)
    candidates = extract_deals(client, brand, pages, today, kind)
    kept, rejected, verdicts = {}, {}, []
    for candidate in candidates:
        deal, reason = verify_deal(candidate, pages, today, brand)
        verdicts.append(reason or "kept")
        if deal is None:
            rejected[reason] = rejected.get(reason, 0) + 1
        else:
            kept.setdefault(deal["dedupe_key"], deal)  # same code on several pages -> keep the first
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, re.sub(r"[^A-Za-z0-9]+", "_", brand) + ".json"), "w") as f:
            json.dump({"brand": brand, "kind": kind, "today": today.isoformat(), "pages": pages,
                       "candidates": candidates, "verdicts": verdicts}, f, indent=1, default=str)
    return {"deals": list(kept.values()), "pages": len(pages), "candidates": len(candidates), "rejected": rejected}
