"""Deals: verification against the source page, merchant ranking, caching and storage."""
from datetime import date

import main
from conftest import FakeLLM, upload
from deals import find_deals_for_brand, verify_deal

TODAY = date(2026, 11, 20)
PAGE = {
    "url": "https://example.com/mcd",
    "title": "McDonald's deals",
    "text": "Use code “BIGMAC20” for 20% off  orders over $15. Offer ends 12/31/2026.\n"
            "Free fries on Fridays with any app purchase.",
}


def deal(**overrides):
    base = {"page": 0, "code": "BIGMAC20", "description": "20% off orders over $15", "discount_type": "percent",
            "discount_value": 20, "min_spend": 15, "expires_on": "2026-12-31", "conditions": None,
            "quote": 'Use code "BIGMAC20" for 20% off orders over $15.'}
    return {**base, **overrides}


# ── verification ──

def test_deal_backed_by_page_is_kept_despite_quote_and_space_differences():
    kept, reason = verify_deal(deal(), [PAGE], TODAY, "McDonald's")
    assert reason is None
    assert kept["code"] == "BIGMAC20" and kept["expires_on"] == date(2026, 12, 31)
    assert kept["source_url"] == PAGE["url"] and kept["dedupe_key"] == "BIGMAC20"


def test_invented_code_is_rejected():
    assert verify_deal(deal(code="BIGMAC25"), [PAGE], TODAY) == (None, "code not in page")


def test_code_must_match_whole_token():
    assert verify_deal(deal(code="BIGMAC2"), [PAGE], TODAY) == (None, "code not in page")


def test_paraphrased_quote_is_rejected():
    assert verify_deal(deal(quote="Get 20 percent off with BIGMAC20"), [PAGE], TODAY) == (None, "quote not in page")


def test_expired_deal_is_rejected_and_bad_dates_are_dropped():
    assert verify_deal(deal(expires_on="2026-11-01"), [PAGE], TODAY) == (None, "expired")
    kept, _ = verify_deal(deal(expires_on="end of year"), [PAGE], TODAY)
    assert kept["expires_on"] is None


def test_codeless_deal_and_placeholder_codes():
    kept, _ = verify_deal(deal(code="N/A", description="Free fries on Fridays", discount_type="weird",
                               min_spend=None, quote="Free fries on Fridays with any app purchase."), [PAGE], TODAY)
    assert kept["code"] is None and kept["discount_type"] == "other"
    assert kept["dedupe_key"] == "free fries on fridays"


def test_unknown_page_and_malformed_candidates():
    assert verify_deal(deal(page=3), [PAGE], TODAY) == (None, "unknown page")
    assert verify_deal("not a dict", [PAGE], TODAY) == (None, "malformed")


def test_same_code_on_two_pages_is_kept_once():
    llm = FakeLLM()
    llm.extract = lambda brand, pages: [deal(page=0), deal(page=1), deal(code="FAKE99")]
    client = type("C", (), {})()
    client.chat = type("Chat", (), {})()
    client.chat.completions = type("Comp", (), {"create": staticmethod(llm.create)})()
    web = type("W", (), {"search": staticmethod(lambda q, **kw: {"results": [
        {"url": "https://a.example", "title": "McDonald's coupons", "raw_content": PAGE["text"]},
        {"url": "https://b.example", "title": "McDonald's deals", "raw_content": PAGE["text"]},
    ]})})()
    result = find_deals_for_brand(client, web, "McDonald's", TODAY)
    assert [d["code"] for d in result["deals"]] == ["BIGMAC20"]
    assert result["candidates"] == 3 and result["rejected"] == {"code not in page": 1}


# ── API ──

STATEMENT = (
    "10/01/2026,10/01/2026,SAFEWAY #1234 SEATTLE WA,500.00,Supermarkets\n"
    + "".join(f"10/{d:02d}/2026,10/{d:02d}/2026,MCDONALD'S F1111 SEATTLE WA,8.50,Restaurants\n" for d in (2, 3, 4))
    + "".join(f"10/{d:02d}/2026,10/{d:02d}/2026,MCDONALD'S F2222 BELLEVUE WA,9.00,Restaurants\n" for d in (5, 6))
    + "".join(f"10/{d:02d}/2026,10/{d:02d}/2026,LUCKY PHO SEATTLE WA,14.00,Restaurants\n" for d in range(7, 13))
    + "06/01/2026,06/01/2026,OLD CHAIN #9 SEATTLE WA,5.00,Restaurants\n" * 9  # outside the 90-day window
)


def setup_world(client, llm, web):
    upload(client, STATEMENT)
    brands = {"MCDONALD'S": "McDonald's", "SAFEWAY": "Safeway", "OLD CHAIN": "Old Chain", "LUCKY PHO": "Lucky Pho"}
    llm.identify = lambda m: next(({"brand": b, "kind": "restaurant", "chain": "PHO" not in m}
                                   for prefix, b in brands.items() if m.startswith(prefix)), {"brand": None, "chain": False})
    llm.extract = lambda brand, pages: [deal()] if brand == "McDonald's" else []
    web.pages["McDonald's"] = [{"url": PAGE["url"], "title": PAGE["title"], "raw_content": PAGE["text"]}]
    web.pages["Safeway"] = [{"url": "https://s.example", "title": "s", "raw_content": "no deals today"}]


def test_refresh_ranks_chains_by_recent_visits(client, llm, web, monkeypatch):
    monkeypatch.setattr(main, "deal_today", lambda: TODAY)
    setup_world(client, llm, web)
    r = client.post("/deals/refresh", params={"limit": 1}).json()
    # McDonald's: 5 visits over two stores beats Safeway's one big purchase; the local pho shop
    # has more visits but isn't a chain; Old Chain's visits are older than 90 days
    assert [(b["brand"], b["visits"], b["status"]) for b in r["brands"]] == [("McDonald's", 5, "searched")]
    assert web.queries == ["McDonald's restaurant promo code coupon"]
    assert web.kwargs["country"] == "united states"
    assert r["brands"][0]["kept"] == 1
    assert "OLD CHAIN #9 SEATTLE WA" not in llm.identified

    deals = client.get("/deals").json()["deals"]
    assert list(deals) == ["McDonald's"] and deals["McDonald's"][0]["code"] == "BIGMAC20"


def test_refresh_uses_cache_and_force(client, llm, web, monkeypatch):
    monkeypatch.setattr(main, "deal_today", lambda: TODAY)
    setup_world(client, llm, web)
    client.post("/deals/refresh", params={"limit": 2})
    identified, queries = len(llm.identified), len(web.queries)

    r = client.post("/deals/refresh", params={"limit": 2}).json()
    assert {b["status"] for b in r["brands"]} == {"cached"}
    assert len(web.queries) == queries and len(llm.identified) == identified  # nothing re-done

    r = client.post("/deals/refresh", params={"limit": 2, "force": True}).json()
    assert {b["status"] for b in r["brands"]} == {"searched"}
    assert len(llm.identified) == identified  # merchant identification stays cached
    assert len(client.get("/deals").json()["deals"]["McDonald's"]) == 1  # upserted, not duplicated


def test_expired_deals_are_hidden(client, llm, web, monkeypatch):
    monkeypatch.setattr(main, "deal_today", lambda: TODAY)
    setup_world(client, llm, web)
    client.post("/deals/refresh", params={"limit": 1})
    monkeypatch.setattr(main, "deal_today", lambda: date(2027, 1, 1))
    assert client.get("/deals").json()["deals"] == {}


def test_one_failing_search_doesnt_sink_the_rest(client, llm, web, monkeypatch):
    monkeypatch.setattr(main, "deal_today", lambda: TODAY)
    setup_world(client, llm, web)
    web.raise_for.add("Safeway")
    r = client.post("/deals/refresh", params={"limit": 2}).json()
    status = {b["brand"]: b["status"] for b in r["brands"]}
    assert status == {"McDonald's": "searched", "Safeway": "error"}


def test_refresh_without_data(client, llm, web):
    assert "error" in client.post("/deals/refresh").json()


def test_bills_and_subscriptions_are_not_searched(client, llm, web, monkeypatch):
    monkeypatch.setattr(main, "deal_today", lambda: TODAY)
    setup_world(client, llm, web)
    upload(client, "".join(f"10/{d:02d}/2026,10/{d:02d}/2026,APPLE.COM/BILL 866-712-7753 CA,2.99,Merchandise\n"
                           for d in range(13, 23)))
    llm.categorize = lambda it: "Software & Subscriptions" if "APPLE" in it["description"] else "Food & Dining"
    client.post("/classify")
    llm.identify = (lambda inner: lambda m: {"brand": "Apple", "chain": True} if "APPLE" in m else inner(m))(llm.identify)
    r = client.post("/deals/refresh", params={"limit": 5}).json()
    assert "Apple" not in [b["brand"] for b in r["brands"]]  # 10 visits, but it's a subscription
    assert not any("APPLE" in m for m in llm.identified)


# ── stricter checks added after the first real run ──

def test_code_must_be_in_the_quote_not_just_somewhere_on_the_page():
    # a real run kept code "WENTY", a fragment found elsewhere on the page
    candidate = deal(code="BIGMAC20", quote="Free fries on Fridays with any app purchase.",
                     discount_type="free_item")
    assert verify_deal(candidate, [PAGE], TODAY) == (None, "code not in quote")


def test_page_about_another_business_is_rejected():
    # a real run took Taco John's "TJ" app code for Trader Joe's
    page = {"url": "https://www.facebook.com/tacojohns/posts/1", "title": "Taco John's",
            "text": "Grab this deal when you enter code TRIPLE555 at checkout in the TJ app or online."}
    candidate = deal(code="TRIPLE555", description="Deal in the TJ app", discount_type="other", discount_value=None,
                     quote="Grab this deal when you enter code TRIPLE555 at checkout in the TJ app or online.")
    assert verify_deal(candidate, [page], TODAY, "Trader Joe's") == (None, "brand not on page")


def test_non_us_deals_and_pages_are_rejected():
    # a real run matched US gas station ARCO to a UK company with prices in pounds
    uk_page = {"url": "https://example.com/arco.co.uk", "title": "Arco Discount Codes",
               "text": "Get £50 off orders. Use code 10G9300 from Arco. Delivery from £4.95, free over £99."}
    pounds = deal(code=None, description="£50 off", discount_type="amount", discount_value=50,
                  quote="Get £50 off orders.")
    assert verify_deal(pounds, [uk_page], TODAY, "ARCO") == (None, "non-US deal")
    code_only = deal(code="10G9300", description="25% off", discount_type="percent", discount_value=None,
                     quote="Use code 10G9300 from Arco.")
    assert verify_deal(code_only, [uk_page], TODAY, "ARCO") == (None, "non-US page")


def test_discount_must_match_the_quote():
    # a real run described "$30 Off First Online Order" as 30% off
    page = {"url": "https://safeway.example", "title": "Safeway codes", "text": "$30 Off First Online Order $75+"}
    wrong = deal(code=None, description="30% off first online order", discount_type="percent",
                 discount_value=30, quote="$30 Off First Online Order $75+")
    assert verify_deal(wrong, [page], TODAY, "Safeway") == (None, "discount not in quote")
    right = {**wrong, "discount_type": "amount", "min_spend": 75, "description": "$30 off first online order $75+"}
    assert verify_deal(right, [page], TODAY, "Safeway")[1] is None


def test_research_replaces_deals_no_longer_found(client, llm, web, monkeypatch):
    monkeypatch.setattr(main, "deal_today", lambda: TODAY)
    setup_world(client, llm, web)
    client.post("/deals/refresh", params={"limit": 1})
    llm.extract = lambda brand, pages: []
    client.post("/deals/refresh", params={"limit": 1, "force": True})
    assert client.get("/deals").json()["deals"] == {}


def test_min_spend_must_be_in_the_quote():
    # a real run described "20% Off Air Purifier" as "20% off orders over $30"
    page = {"url": "https://amazon.example", "title": "Amazon coupons", "text": "20% Off Air Purifier for Allergies"}
    candidate = deal(code=None, description="20% off orders over $30", discount_value=20, min_spend=30,
                     quote="20% Off Air Purifier for Allergies")
    assert verify_deal(candidate, [page], TODAY, "Amazon") == (None, "min spend not in quote")


def test_brand_spelling_change_still_replaces_old_deals(client, llm, web, monkeypatch):
    # a real run identified "IKEA" as "Ikea" the second time and left the old deals behind
    monkeypatch.setattr(main, "deal_today", lambda: TODAY)
    setup_world(client, llm, web)
    client.post("/deals/refresh", params={"limit": 1})
    assert list(client.get("/deals").json()["deals"]) == ["McDonald's"]

    from database import MerchantProfile, Session, engine
    with Session(engine) as session:
        for p in session.query(MerchantProfile).filter(MerchantProfile.brand == "McDonald's"):
            p.brand = "MCDONALD'S"
        session.commit()
    web.pages["MCDONALD'S"] = web.pages["McDonald's"]
    llm.extract = lambda brand, pages: [deal(code=None, description="Free fries on Fridays", discount_type="free_item",
                                             min_spend=None, quote="Free fries on Fridays with any app purchase.")]
    r = client.post("/deals/refresh", params={"limit": 1}).json()
    assert r["brands"][0]["status"] == "cached"  # same brand, different spelling: still cached
    client.post("/deals/refresh", params={"limit": 1, "force": True})
    deals = client.get("/deals").json()["deals"]
    assert list(deals) == ["MCDONALD'S"] and [d["code"] for d in deals["MCDONALD'S"]] == [None]


def test_search_skips_tavily_profile_pages(client, llm, web, monkeypatch):
    monkeypatch.setattr(main, "deal_today", lambda: TODAY)
    setup_world(client, llm, web)
    client.post("/deals/refresh", params={"limit": 1})
    assert web.kwargs["exclude_domains"] == ["tavily.com"]


def test_cents_count_as_amounts():
    # a real run wrongly rejected ARCO's "bonus 10¢/gal in Rewards"
    page = {"url": "https://www.facebook.com/arco/videos/1", "title": "Arco Member Day",
            "text": "Members, remember! Don't miss out on a bonus 10¢/gal in Rewards every month on the 15th!"}
    candidate = deal(code=None, description="Bonus 10¢/gal in Rewards on the 15th", discount_type="amount",
                     discount_value=10, min_spend=None,
                     quote="Members, remember! Don't miss out on a bonus 10¢/gal in Rewards every month on the 15th!")
    assert verify_deal(candidate, [page], TODAY, "ARCO")[1] is None


def test_vague_deals_without_code_or_amount_are_rejected():
    page = {"url": "https://amazon.example", "title": "Amazon promo codes",
            "text": "Amazon offers a students discount program that can provide exclusive savings."}
    candidate = deal(code=None, description="Students discount program", discount_type="percent",
                     discount_value=None, min_spend=None,
                     quote="Amazon offers a students discount program that can provide exclusive savings.")
    assert verify_deal(candidate, [page], TODAY, "Amazon") == (None, "no concrete discount")
