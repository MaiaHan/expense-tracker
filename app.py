import streamlit as st
import requests
import pandas as pd
import json

# API base URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Expense Tracker",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AI Expense Tracker")
st.caption("Upload your bank CSV, get AI-powered insights and deal recommendations")

# Sidebar for navigation
page = st.sidebar.selectbox("Navigate", ["Upload & Classify", "Deals", "Chat", "Settings"])

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
                    # Card payments and cashback aren't spending; refunds stay in and net out
                    NON_SPENDING = ["Card Payment", "Rewards & Cashback"]
                    spending = df[~df["predicted_category"].isin(NON_SPENDING)]
                    category_summary = spending.groupby("predicted_category")["amount"].sum()
                    category_summary = category_summary.sort_values(ascending=False)

                    # Only show categories with positive net spending
                    category_summary = category_summary[category_summary > 0]
                    st.bar_chart(category_summary)
                    
                    # Show transactions table
                    st.subheader("Transactions")
                    st.dataframe(df)

# ─── Deals ───
elif page == "Deals":
    st.header("💸 Deal Recommendations")
    st.caption("Based on your top merchants, here are current deals and coupons")
    
    if st.button("Find Deals"):
        with st.spinner("Searching for deals..."):
            response = requests.post(f"{API_URL}/deals")
            result = response.json()
        
        if "error" in result:
            st.error(result["error"])
        else:
            deals = result.get("deals", {})
            if not deals:
                st.warning("No deals found. Make sure you've uploaded and classified your data first.")
            else:
                for merchant, merchant_deals in deals.items():
                    st.subheader(f"🏪 {merchant}")
                    for deal in merchant_deals:
                        with st.expander(deal.get("title", "Deal")):
                            st.write(deal.get("snippet", ""))
                            st.markdown(f"[View Deal]({deal.get('url', '')})")

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
