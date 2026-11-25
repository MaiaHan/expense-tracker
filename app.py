import streamlit as st
import requests
import pandas as pd
import json
import os

# API base URL (docker-compose points this at the api service)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Card payments and cashback aren't spending (refunds are: they net out their category)
NON_SPENDING = ["Card Payment", "Rewards & Cashback"]
SOURCE_LABELS = {"manual": "you (this one)", "user_rule": "your rule", "rule": "built-in rule", "llm": "AI"}

st.set_page_config(
    page_title="AI Expense Tracker",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AI Expense Tracker")
st.caption("Upload your bank CSV, get AI-powered insights and deal recommendations")

# Sidebar for navigation
page = st.sidebar.selectbox("Navigate", ["Upload & Classify", "Transactions", "Deals", "Chat", "Settings"])

# ─── Upload & Classify ───
if page == "Upload & Classify":
    st.header("Upload Your Bank CSV")
    
    uploaded_file = st.file_uploader("Choose your bank CSV file", type="csv")
    
    if uploaded_file:
        # Upload to backend
        with st.spinner("Uploading..."):
            response = requests.post(
                f"{API_URL}/upload",
                files={"file": uploaded_file}
            )
            result = response.json()
        
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(
                f"Read {result['rows']} transactions: {result['inserted']} new, "
                f"{result['duplicates']} already stored"
                + (f", {result['skipped_invalid']} skipped (missing date/amount)" if result["skipped_invalid"] else "")
            )
            st.json(result["detected_mapping"])
            
            # Classify button
            if st.button("Classify with AI"):
                with st.spinner("AI is categorizing your expenses..."):
                    classify_response = requests.post(f"{API_URL}/classify")
                    transactions = classify_response.json()
                
                if "error" in transactions:
                    st.error(transactions["error"])
                else:
                    df = pd.DataFrame(transactions)
                    
                    # Show summary chart
                    st.subheader("Spending by Category")
                    # Net out refunds — positive expenses minus negative refunds
                    spending = df[~df["predicted_category"].isin(NON_SPENDING)]
                    category_summary = spending.groupby("predicted_category")["amount"].sum()
                    category_summary = category_summary.sort_values(ascending=False)

                    # Only show categories with positive net spending
                    category_summary = category_summary[category_summary > 0]
                    st.bar_chart(category_summary)
                    
                    # Show transactions table
                    st.subheader("Transactions")
                    st.dataframe(df)

# ─── Transactions: review and fix categories ───
elif page == "Transactions":
    st.header("🧾 Transactions")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))

    transactions = requests.get(f"{API_URL}/transactions").json()
    if not transactions:
        st.info("No transactions yet. Upload a CSV first.")
    else:
        df = pd.DataFrame(transactions)
        df["labeled by"] = df["category_source"].map(SOURCE_LABELS).fillna("")
        only_other = st.checkbox("Only show transactions labeled Other")
        view = df[df["predicted_category"] == "Other"] if only_other else df
        st.dataframe(view[["date", "description", "amount", "predicted_category", "labeled by"]],
                     width="stretch", hide_index=True)

        st.subheader("Fix a category")
        categories = requests.get(f"{API_URL}/categories").json().get("categories", []) + NON_SPENDING
        rows_by_id = df.set_index("id")
        labels = {
            tid: f"{r['date']}  {r['description']}  ${r['amount']:.2f}  ({r['predicted_category'] or 'unclassified'})"
            for tid, r in rows_by_id.loc[view["id"]].iterrows()
        }
        choice = st.selectbox("Transaction", list(labels), format_func=lambda tid: labels.get(tid, tid))
        if choice is not None:
            current = rows_by_id.at[choice, "predicted_category"]
            new_category = st.selectbox("New category", categories,
                                        index=categories.index(current) if current in categories else 0)
            apply_to_merchant = st.checkbox("Remember this for the merchant (all past and future transactions)", value=True)
            pattern = ""
            if apply_to_merchant:
                pattern = st.text_input(
                    "Merchant name contains",
                    key=f"pattern_{choice}",
                    help="Leave empty to use the full merchant name. Shorten it to cover variations, "
                         "e.g. LYFT matches every LYFT *RIDE <driver> variation.",
                )
            if st.button("Save"):
                params = {"category": new_category, "apply_to_merchant": apply_to_merchant}
                if pattern.strip():
                    params["pattern"] = pattern
                result = requests.post(f"{API_URL}/transactions/{choice}/category", params=params).json()
                if "error" in result:
                    st.error(result["error"])
                else:
                    rule = result["rule"]
                    st.session_state.flash = (
                        f"Rule saved: merchant contains '{rule['pattern']}' → {rule['category']} "
                        f"({result['updated']} transactions updated)" if rule else "Updated this transaction"
                    )
                    st.rerun()

# ─── Deals ───
elif page == "Deals":
    st.header("💸 Deals")
    st.caption("Promo codes for the chains you visit most. Every code shown was found word-for-word "
               "on its source page, but coupon sites list expired codes too, so check before relying on one.")

    col1, col2 = st.columns([1, 1])
    limit = col1.number_input("Chains to search", min_value=1, max_value=10, value=5)
    force = col2.checkbox("Search again even if searched in the last 24 hours")
    if st.button("Find deals"):
        with st.spinner("Searching the web and checking each deal against its source..."):
            result = requests.post(f"{API_URL}/deals/refresh", params={"limit": limit, "force": force}).json()
        if "error" in result:
            st.error(result["error"])
        else:
            for b in result["brands"]:
                visits = f"{b['visits']} visit{'s' if b['visits'] != 1 else ''}"
                if b["status"] == "searched":
                    rejected = ", ".join(f"{n} {reason}" for reason, n in b["rejected"].items()) or "none"
                    st.write(f"**{b['brand']}** ({visits}): {b['kept']} deals kept "
                             f"from {b['candidates']} found on {b['pages']} pages · rejected: {rejected}")
                elif b["status"] == "cached":
                    st.write(f"**{b['brand']}** ({visits}): searched in the last 24 hours")
                else:
                    st.warning(f"{b['brand']}: {b.get('error')}")

    def md(text):
        """Escape $ so Streamlit doesn't render "$30 off ... $75+" as a LaTeX formula."""
        return str(text).replace("$", "\\$")

    deals = requests.get(f"{API_URL}/deals").json().get("deals", {})
    if not deals:
        st.info("No deals yet. Upload a statement, then click Find deals.")
    for brand, brand_deals in deals.items():
        st.subheader(brand)
        for deal in brand_deals:
            with st.container(border=True):
                left, right = st.columns([3, 1])
                left.markdown(f"**{md(deal['description'])}**")
                if deal["code"]:
                    right.code(deal["code"], language=None)
                details = []
                if deal["min_spend"]:
                    details.append(f"min spend \\${deal['min_spend']:g}")
                if deal["conditions"]:
                    details.append(md(deal["conditions"]))
                details.append(f"expires {deal['expires_on']}" if deal["expires_on"] else "no expiry listed")
                left.caption(" · ".join(details))
                st.caption(f"“{md(deal['quote'])}” — [{md(deal['source_title'] or 'source')}]({deal['source_url']})")

# ─── Chat ───
elif page == "Chat":
    st.header("💬 Chat About Your Expenses")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your expenses..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_URL}/chat",
                    params={"question": prompt}
                )
                answer = response.json().get("answer", "Sorry, I couldn't answer that.")
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# ─── Settings ───
elif page == "Settings":
    st.header("⚙️ Category Settings")
    
    # Show current categories
    response = requests.get(f"{API_URL}/categories")
    categories = response.json().get("categories", [])
    
    st.subheader("Current Categories")
    for cat in categories:
        col1, col2 = st.columns([4, 1])
        col1.write(cat)
        if col2.button("Delete", key=f"del_{cat}"):
            requests.delete(f"{API_URL}/categories/delete", params={"name": cat})
            st.rerun()
    
    # Merchant rules learned from corrections
    st.subheader("Merchant Rules")
    rules = requests.get(f"{API_URL}/rules").json().get("rules", [])
    if not rules:
        st.caption("No rules yet. Fix a category on the Transactions page to create one.")
    for rule in rules:
        col1, col2 = st.columns([4, 1])
        col1.write(f"Merchant contains **{rule['pattern']}** → {rule['category']}")
        if col2.button("Delete", key=f"rule_{rule['id']}"):
            requests.delete(f"{API_URL}/rules/{rule['id']}")
            st.rerun()

    # Add new category
    st.subheader("Add New Category")
    new_cat = st.text_input("Category name")
    if st.button("Add Category"):
        if new_cat:
            response = requests.post(f"{API_URL}/categories/add", params={"name": new_cat})
            result = response.json()
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Added '{new_cat}' successfully!")
                st.rerun()
