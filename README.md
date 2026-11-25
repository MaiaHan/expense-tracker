# AI Expense Tracker

Upload a credit card statement CSV and get every transaction categorized: rules first, then an LLM, and it learns from your corrections. Spending summaries leave out card payments and cashback, and refunds net out against the category they came from. Correctness is measured on 318 hand-labeled real transactions.

**Stack:** FastAPI · SQLAlchemy · PostgreSQL (SQLite for quick local runs) · Docker Compose · OpenAI `gpt-4o-mini` · Streamlit · pytest

## How categorization works

```
CSV upload ──► column detection (LLM) ──► normalize + dedupe ──► SQLite
                                                                   │
/classify, for each uncategorized transaction:                     ▼
  1. your merchant rules      "merchant contains LYFT" → Transportation
  2. built-in rules           card payments, cashback (negative amounts only)
  3. LLM                      one request per 100 unique merchants, answered by item id
```

- **Dedupe on re-upload.** Each transaction ID is a hash of date, description, amount and its position among identical rows. Re-uploading or uploading overlapping statements never duplicates anything, and two real identical purchases on the same day stay as two rows.
- **Payments vs. refunds.** Card issuers can put both under one label (Discover: "Payments and Credits"). Payments to the card and cashback are matched by narrow rules and left out of spending totals. Refunds keep their merchant's category, so a $40 purchase plus a $25 refund shows up as $15 of Shopping.
- **Learning from corrections.** Fixing a transaction saves a rule: merchant name contains a pattern → category. The rule relabels past transactions and applies to future uploads without an LLM call. When several rules match, the most specific (longest) one wins. One-off fixes are never overwritten by reclassification.
- **The LLM answers by id.** It doesn't echo merchant names back, so a slightly rewritten name can't silently turn into "Other". This also made responses about 3× faster, since the output is much shorter.

## Deals

`deals.py` finds promo codes for the chains you actually go to:

1. **Pick merchants by visits, not spend.**
   - It counts visits over the last 90 days of your data, so frequent lunch spots win over one large grocery run.
   - Bills and subscriptions are skipped.
2. **Identify each merchant once** with the LLM and cache the result. For example, `ARCO#01234 SEATTLE WA` becomes ARCO, a gas station, and a chain. Single local businesses are skipped, and several stores of one chain count together.
3. **Search** with Tavily for `ARCO gas station promo code`. Adding the kind of business keeps same-name companies out. US results are boosted, and full page text is fetched.
4. **Extract** structured deals with the LLM: code, discount, minimum spend, expiry, conditions, and a verbatim quote as evidence.
5. **Verify each deal deterministically.** A deal is kept only if all of these hold:
   - the quote is on the page
   - the code is inside the quote
   - there is a code or a concrete discount, and the discount and minimum spend in the description appear in the quote
   - the brand appears in the page title, URL or quote
   - the deal and the page are priced in US dollars
   - it hasn't expired

   Rejections are counted by reason. A re-search replaces a brand's old deals.

Verification proves the page says it, not that the code still works: coupon sites list plenty of dead codes. The UI shows the quote and source link next to each code.

**Iterating on real results.** I ran it on my own statement and checked every kept deal by hand. Each check above was added in response to a failure in one of these runs, and each has a regression test built from that failure.

| Run | Deals kept | Wrong business or garbled | Description not faithful to source | Vague or single-product |
|---|---|---|---|---|
| 1: text-presence checks only | 13 | 5: a UK company named Arco for the ARCO gas station; Taco John's "TJ" app code for Trader Joe's; code fragment "WENTY" | 1: "$30 off" described as 30% | 3 |
| 2: + business kind, brand, currency, code-in-quote checks | 8 | 0 | 1 | 6 |
| 3: + min-spend and concrete-discount checks, skip single-product deals | 4 | 0 | 0 | 1 |

In run 3 the cents check had first rejected a real offer ("bonus 10¢/gal in Rewards", ARCO). The fix was verified by replaying the saved pages (`DEALS_DEBUG_DIR`) without searching again. The sample is small, and fewer deals also means some real ones are missed.

## Evaluation

`evaluate.py` runs the same pipeline as the app against a labeled CSV. The dataset is 12 months of my own credit card transactions (318 rows, 116 unique merchants), hand-checked. It isn't in the repo for privacy.

| | Result |
|---|---|
| Overall accuracy | **92.5 – 94.3%** across 3 runs (LLM output varies by about ±2%) |
| Per unique merchant | 85 – 89% |
| Built-in payment / cashback rules | 100% (23 / 23) |

**Labels.** Labels started from the issuer's own categories, but 94 of 318 (30%) had to be fixed by hand. For example, software subscriptions are filed as "Merchandise", and transit and rideshare as "Travel/ Entertainment". The `label_source` column records which labels were fixed, and results are reported separately for those rows.

**Experiment: giving the LLM the issuer's category as a hint made it worse.** Overall accuracy dropped from 92.5% to 83.3%. On the hand-fixed rows it fell from 83% to 35%: the model mostly copied the issuer's label, even with an explicit warning that the label is coarse. On rows where the issuer's category was already right it reached 100%, which looks great but only because those labels came from the issuer in the first place. The hint is off by default and still available as `--bank-hint`.

**Simulated corrections (`--simulate-corrections`).** The data is split at the median date. A simulated user fixes every mistake in the first half, using exact merchant names and no hand-shortened patterns, which gives 14 rules. On the second half, errors went from 7 to 3 (95.6% → 98.1%). The remaining errors are merchants that first appear in the second half, which rules can't cover. Only 4 transactions matched a rule, so this number is directional.

```bash
python evaluate.py data/labels.csv                          # baseline
python evaluate.py data/labels.csv --simulate-corrections   # + learning from corrections
python evaluate.py data/labels.csv --bank-hint              # the issuer-category experiment
```

The labels CSV needs `description, amount, true_category`. Optional columns are `date`, `bank_category` and `label_source`.

## Performance

`benchmarks/bench.py` load-tests the API in-process on synthetic statements (PostgreSQL 16 in Docker on an Apple M5 laptop). The LLM is simulated: each call sleeps like the real API did on the labeled data (0.5s + 0.11s per merchant), scaled by 0.1 so runs stay short.

| PostgreSQL | before | after |
|---|---|---|
| Upload 100k rows | **crashed**: dedupe check sent all IDs in one `IN (...)` (Postgres max is 65,535 parameters) | 3.0s (33k rows/s) |
| Upload 10k rows | 0.53s | 0.36s |
| Classify 10k rows (495 merchants, 5 LLM calls) | 6.2s | 2.7s |
| `/fraud` 10k rows | 10.6s, O(n²) | **0.17s** (62×) |
| `/fraud` 100k rows | — | 1.3s |

What changed:
- **Upload.** `INSERT … ON CONFLICT DO NOTHING RETURNING id`, sent through SQLAlchemy's batched multi-row inserts. The database skips existing IDs itself, so there's no separate duplicate lookup.
- **Classify.**
  - LLM batches run 4 at a time. They're I/O-bound.
  - Results are written back with one bulk UPDATE by primary key instead of loading ORM objects.
- **Fraud rules.** Group-by transforms replace a per-row scan. The 7-day window count uses binary search over each merchant's sorted dates. A test checks the output is identical to the old implementation on random data.
- **Endpoints are plain `def`, not `async def`.** They call the blocking OpenAI client and database driver, which inside `async def` would stall the event loop for every other request.
- **Schema.**
  - `amount` is `NUMERIC(12,2)` (exact sums) and `date` is `DATE`.
  - Indexes on `date` and `predicted_category`.
  - Existing databases are upgraded on startup.

```bash
docker compose up -d db
docker compose exec db psql -U expense -d expense_tracker -c "CREATE DATABASE expense_bench"
DATABASE_URL=postgresql://expense:expense@localhost:5432/expense_bench \
  python benchmarks/bench.py --rows 100000 --merchants 2000 --fraud-max-rows 100000
```

## Running it

Create a `.env` with your keys (it is read at runtime and never copied into the image):

```
OPENAI_API_KEY=...
TAVILY_API_KEY=...     # only for the deals page
```

**With Docker** (PostgreSQL + API + UI):

```bash
docker compose up --build
# UI: http://localhost:8501   API docs: http://localhost:8000/docs
```

**Without Docker** (SQLite file, `expense_tracker.db`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload      # API on :8000
streamlit run app.py           # UI on :8501
```

`DATABASE_URL` selects the database (e.g. `postgresql://user:pass@host/db`). Tables are created on startup, and columns added in later versions are added to existing databases automatically.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests                                   # SQLite

docker compose up -d db                                  # or against Postgres
docker compose exec db psql -U expense -d expense_tracker -c "CREATE DATABASE expense_test"
TEST_DATABASE_URL=postgresql://expense:expense@localhost:5432/expense_test python -m pytest tests
```

The 51 tests cover:
- the built-in rules and the categorization pipeline
- upload dedupe, including uploads spanning many insert batches
- spending summaries
- correction rules (specificity, deletion fallback, manual fixes surviving reclassify)
- the vectorized fraud rules matching the original implementation
- deal verification, using cases taken from real runs (wrong business, foreign currency, code outside the quote, mismatched discount, cents, vague deals)
- deal ranking by visits, caching, brand-spelling changes, and one failing search not affecting the rest
- the startup schema upgrade for older databases
- `evaluate.py`

The LLM is faked, so the tests make no network calls. They pass on both SQLite and PostgreSQL.

## API

| Endpoint | |
|---|---|
| `POST /upload` | CSV upload. The LLM maps the columns; rows are normalized and deduplicated. |
| `POST /classify?reclassify=` | Categorize pending transactions. `reclassify=true` redoes all of them except manual fixes. |
| `GET /transactions` | All transactions, each with how it was labeled (`manual`, `user_rule`, `rule`, `llm`). |
| `POST /transactions/{id}/category` | Fix one transaction, or save a merchant rule (`apply_to_merchant`, `pattern`). |
| `GET /rules`, `DELETE /rules/{id}` | List or delete merchant rules. Deleting one falls back to broader rules, then the LLM. |
| `GET/POST/DELETE /categories…` | Manage custom categories. Reserved names are rejected. |
| `POST /chat` | Ask questions about your spending, using a category summary that excludes non-spending categories. |
| `POST /deals/refresh?limit=&force=` | Find deals at the chains you visit most (see below). Brands searched in the last 24h are skipped unless `force=true`. |
| `GET /deals` | Unexpired deals grouped by brand. |
| `POST /fraud` | Rule-based flags: more than 2 charges at the same merchant in one day, an amount over 3× the merchant's average, or more than 5 charges at the same merchant within 7 days. |

## Next

- Use the issuer category only as a fallback when the model answers "Other", or turn it into candidate categories instead of a single label.
- Run the eval on statements from other banks.
- Measure deal precision on a hand-checked sample, and add sources beyond web search (e.g. promo emails).
- Stream large uploads instead of reading the whole CSV into memory; paginate `GET /transactions`.
