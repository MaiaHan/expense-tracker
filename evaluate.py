"""Measure categorization accuracy against hand-checked labels.

Usage:
    python evaluate.py data/labels.csv [--out data/eval_predictions.csv] [--bank-hint]

The labels CSV needs columns: description, amount, true_category. Optional columns:
  bank_category  - the card issuer's category, given to the LLM with --bank-hint
  label_source   - "bank" if the label was mapped from bank_category, "manual" if it was
                   corrected by hand; accuracy is reported for each group, because the
                   bank hint makes "bank" rows easier than they really are
It runs the same rules + LLM pipeline as /classify (see classifier.py), so it calls
the OpenAI API and needs OPENAI_API_KEY in .env.
"""
import argparse
import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from database import DEFAULT_CATEGORIES, NON_SPENDING_CATEGORIES, get_categories, init_db
from classifier import MODEL, clean_description, predict_categories


def per_category_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat in sorted(set(df["true_category"]) | set(df["predicted"])):
        tp = ((df["true_category"] == cat) & (df["predicted"] == cat)).sum()
        n_true = (df["true_category"] == cat).sum()
        n_pred = (df["predicted"] == cat).sum()
        rows.append({
            "category": cat,
            "n": n_true,
            "precision": tp / n_pred if n_pred else float("nan"),
            "recall": tp / n_true if n_true else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", help="CSV with description, amount, true_category")
    parser.add_argument("--out", help="optional path to save per-transaction predictions")
    parser.add_argument("--default-categories", action="store_true",
                        help="use the built-in default categories instead of the ones saved in the DB")
    parser.add_argument("--bank-hint", action="store_true",
                        help="give the bank's category to the LLM as a hint (experiment: off in the app, "
                             "it dropped accuracy on rows where the bank's category is wrong)")
    args = parser.parse_args()

    load_dotenv()
    labels = pd.read_csv(args.labels)
    missing = {"description", "amount", "true_category"} - set(labels.columns)
    if missing:
        raise SystemExit(f"labels file is missing columns: {sorted(missing)}")

    if args.default_categories:
        categories = list(DEFAULT_CATEGORIES)
    else:
        init_db()
        categories = get_categories()

    unknown = set(labels["true_category"]) - set(categories) - set(NON_SPENDING_CATEGORIES)
    if unknown:
        raise SystemExit(f"labels use categories the classifier can't pick: {sorted(unknown)}. "
                         f"Available: {categories}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    bank = labels["bank_category"] if "bank_category" in labels.columns else [None] * len(labels)
    transactions = list(zip(labels["description"].astype(str), labels["amount"].astype(float), bank))
    use_hint = args.bank_hint
    start = time.time()
    result = predict_categories(client, transactions, categories, use_bank_hint=use_hint)
    elapsed = time.time() - start

    df = labels.assign(
        predicted=result["categories"],
        by_rule=result["by_rule"],
        merchant=labels["description"].map(clean_description),
    )
    df["correct"] = df["predicted"] == df["true_category"]
    llm_rows = df[~df["by_rule"]]
    merchants = llm_rows.drop_duplicates("merchant")

    print(f"Model: {MODEL} | categories: {len(categories)} | bank hint: {'on' if use_hint else 'off'} | "
          f"{result['llm_merchants']} unique merchants in {result['llm_calls']} LLM calls, {elapsed:.1f}s")
    if result["unanswered"]:
        print(f"WARNING: LLM gave no answer for {result['unanswered']} merchants (counted as Other)")
    print()
    print(f"Overall accuracy:        {df['correct'].mean():.1%}  ({df['correct'].sum()}/{len(df)} transactions)")
    print(f"  rules only:            {df[df['by_rule']]['correct'].mean():.1%}  ({df['by_rule'].sum()} transactions)")
    print(f"  LLM only:              {llm_rows['correct'].mean():.1%}  ({len(llm_rows)} transactions)")
    print(f"  LLM, per merchant:     {merchants['correct'].mean():.1%}  ({len(merchants)} unique merchants)")
    if "label_source" in df.columns:
        for source, group in llm_rows.groupby("label_source"):
            print(f"  LLM, {source + ' labels:':<18}{group['correct'].mean():.1%}  ({len(group)} transactions)")
    print()

    print("Per category:")
    print(per_category_report(df).to_string(index=False, float_format=lambda x: f"{x:.0%}"))

    mistakes = (df[~df["correct"]]
                .groupby(["merchant", "true_category", "predicted"]).size()
                .reset_index(name="count").sort_values("count", ascending=False))
    print(f"\nMistakes ({len(mistakes)} merchants):")
    print(mistakes.to_string(index=False) if len(mistakes) else "  none")

    if args.out:
        df.drop(columns=["merchant"]).to_csv(args.out, index=False)
        print(f"\nPredictions saved to {args.out}")


if __name__ == "__main__":
    main()
