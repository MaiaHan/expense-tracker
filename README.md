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

The 24 tests cover:
- the built-in rules and the categorization pipeline
- upload dedupe
- spending summaries
- correction rules (specificity, deletion fallback, manual fixes surviving reclassify)
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
| `POST /deals` | Web search (Tavily) for coupons at your top 5 merchants. |
| `POST /fraud` | Rule-based flags: more than 2 charges at the same merchant in one day, an amount over 3× the merchant's average, or more than 5 charges at the same merchant within 7 days. |

## Next

- Use the issuer category only as a fallback when the model answers "Other", or turn it into candidate categories instead of a single label.
- Run the eval on statements from other banks.
- Coupon search: extract structured deals (merchant, code, discount, expiry) instead of raw search results.
- Make `/fraud` vectorized; it is currently O(n²).
