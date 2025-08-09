import streamlit as st  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from transformers import pipeline
import requests
import time  # For retry delay
import torch
import seaborn as sns  # For nicer chart colors
import re

# App layout setup
st.set_page_config(page_title="SmartSales Advisor", layout="wide")

# Header styling
st.markdown(
    """
    <style>
    .big-title {
        font-size:40px !important;
        color: #004aad;
        font-weight: bold;
    }
    .subtitle {
        font-size:18px !important;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">📊 SmartSales AI Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Understand your business like never before — forecasts, insights, and AI-powered guidance in one click.</div>', unsafe_allow_html=True)

# Cached training function
@st.cache_data
def train_prophet(daily_revenue):
    model = Prophet()
    model.fit(daily_revenue)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return model, forecast

# Cached metrics calculation
@st.cache_data
def calculate_metrics(df):
    daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
    daily_revenue = daily_revenue.rename(columns={'date': 'ds', 'revenue': 'y'})

    last_week = daily_revenue['y'].iloc[-14:-7].sum()
    this_week = daily_revenue['y'].iloc[-7:].sum()
    delta = ((this_week - last_week) / last_week) * 100 if last_week != 0 else 0
    aov = (df['revenue'] / df['quantity']).mean()

    top_product = df.groupby('product')['revenue'].sum().idxmax()
    dow_stats = df.groupby(df['date'].dt.day_name())['revenue'].mean()
    best_days = dow_stats.sort_values(ascending=False).head(2).index.tolist()
    worst_day = dow_stats.idxmin()

    summary_input = f"""
    As an AI sales advisor, summarize the following sales performance and give actionable advice in 3-4 full sentences:

    Last week's revenue: ${last_week:.2f}
    This week's revenue: ${this_week:.2f}
    Growth rate: {delta:.2f}%
    Top-selling product: {top_product}
    Best sales days: {', '.join(best_days)}
    Lowest sales day: {worst_day}
    Average order value: ${aov:.2f}
    """

    return daily_revenue, last_week, this_week, delta, aov, top_product, best_days, worst_day, summary_input

# Tabs
tab1, tab2 = st.tabs(["📤 Upload & Forecast", "🤖 AI Insights"])

with tab1:
    uploaded_file = st.file_uploader("Upload your sales data CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Error handling for missing columns
        if not {'date', 'revenue', 'quantity', 'product'}.issubset(df.columns):
            st.error("Your CSV must include columns: date, revenue, quantity, product.")
            st.stop()

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')

        # ✅ Store in session state for Tab 2
        st.session_state["sales_df"] = df

        st.subheader("📊 Preview of Your Data")
        st.dataframe(df.head())


        # Calculate metrics
        daily_revenue, last_week, this_week, delta, aov, top_product, best_days, worst_day, summary_input = calculate_metrics(df)
        st.session_state["summary_input"] = summary_input

        # 📌 Styled Key Metrics
        st.subheader("📌 Key Business Metrics")
        st.markdown(
            """
            <style>
            .metric-card {
                background-color: #f0f4ff;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                text-align: center;
                margin-bottom: 10px;
            }
            .metric-title { font-size: 16px; color: #333; font-weight: bold; }
            .metric-value { font-size: 24px; color: #004aad; font-weight: bold; margin-top: 5px; }
            .metric-delta { font-size: 14px; font-weight: bold; }
            </style>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">📅 Last Week Revenue</div>
                    <div class="metric-value">${last_week:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            delta_color = "green" if delta >= 0 else "red"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">📅 This Week Revenue</div>
                    <div class="metric-value">${this_week:,.2f}</div>
                    <div class="metric-delta" style="color:{delta_color};">{delta:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">💰 Avg Order Value</div>
                    <div class="metric-value">${aov:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        import seaborn as sns

        # 📊 Weekly Revenue
        weekly_revenue = df.resample('W-SUN', on='date')['revenue'].sum().reset_index()
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**📆 Weekly Revenue:**")
            st.dataframe(weekly_revenue)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=weekly_revenue, x='date', y='revenue', ax=ax, color="#004aad")
            ax.set_title("Weekly Revenue Trend", fontsize=14)
            st.pyplot(fig)

        # 💰 AOV by Product
        aov_by_product = (df.groupby('product').apply(lambda x: (x['revenue'] / x['quantity']).mean()))
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**💰 AOV by Product:**")
            st.dataframe(aov_by_product.rename("AOV"))
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=aov_by_product.values, y=aov_by_product.index, palette="Blues_r", ax=ax)
            ax.set_title("Average Order Value by Product", fontsize=14)
            st.pyplot(fig)

        # 📦 Units Sold
        units_sold = df.groupby('product')['quantity'].sum()
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**📦 Units Sold per Product:**")
            st.dataframe(units_sold.rename("Units Sold"))
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=units_sold.values, y=units_sold.index, palette="coolwarm", ax=ax)
            ax.set_title("Units Sold per Product", fontsize=14)
            st.pyplot(fig)

        # 📈 Revenue Share
        revenue_share = df.groupby('product')['revenue'].sum()
        revenue_share_percent = (revenue_share / revenue_share.sum() * 100).round(2)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**📈 Revenue Share (%):**")
            st.dataframe(revenue_share_percent.rename("Revenue %"))
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.pie(revenue_share_percent, labels=revenue_share_percent.index, autopct='%1.1f%%', startangle=90,
                   colors=sns.color_palette("pastel"))
            ax.set_title("Revenue Share by Product", fontsize=14)
            st.pyplot(fig)

        # 📅 Avg Revenue by Day of Week
        avg_revenue_by_dow = df.groupby(df['date'].dt.day_name())['revenue'].mean()
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**📅 Average Revenue by Day of Week:**")
            st.dataframe(avg_revenue_by_dow.rename("Avg Revenue"))
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=avg_revenue_by_dow.index, y=avg_revenue_by_dow.values, palette="viridis", ax=ax)
            ax.set_title("Average Revenue by Day of Week", fontsize=14)
            st.pyplot(fig)

        # 📈 Automatic Forecasting
        dataset_days = (df['date'].max() - df['date'].min()).days
        if dataset_days < 90:
            forecast_days = 30
        elif dataset_days < 180:
            forecast_days = 60
        elif dataset_days < 365:
            forecast_days = 90
        else:
            forecast_days = 180

        forecast_months = forecast_days // 30

        st.subheader("📈 Forecasting with Prophet")
        with st.spinner("Training model... please wait"):
            model = Prophet()
            model.fit(daily_revenue)
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

        fig1 = model.plot(forecast, figsize=(8, 4))
        st.pyplot(fig1)

        # Explanation
        st.info(f"We forecasted for {forecast_months} months ({forecast_days} days) based on your dataset size ({dataset_days} days of sales data). "
                "Smaller datasets give less reliable long-term predictions, so we adjusted the forecast period accordingly.")


import re

with tab2:
    if "summary_input" in st.session_state and "sales_df" in st.session_state:
        st.subheader("💡 AI-Generated Business Insights")

        api_key = st.secrets.get("TOGETHER_API_KEY")
        df = st.session_state["sales_df"]

        # ✅ Helper to clean numbering artifacts
        def clean_numbered_list(text):
            lines = text.splitlines()
            cleaned_lines = []
            for line in lines:
                if re.match(r"^\s*\d+[\*\.\)]\s*$", line.strip()):  # skip orphan numbers
                    continue
                cleaned_lines.append(line)
            return "\n".join(cleaned_lines).strip()

        # ✅ Generate insights only once per dataset (when file is first uploaded)
        if "ai_insight" not in st.session_state and api_key:
            summary_text = st.session_state["summary_input"]

            ai_prompt = f"""
            You are an AI sales advisor. Using the given sales metrics, write a natural, insightful 3-4 sentence summary 
            that avoids repetition and gives actionable advice.

            Metrics: {summary_text}

            Guidelines:
            - Do not repeat days if they appear twice.
            - Replace bullet-like text with a smooth narrative.
            - Interpret the growth rate (say if sales are increasing or decreasing).
            - End with at least one practical suggestion to improve sales.
            - If you list numbered points, stop numbering after the last real point.
            """

            with st.spinner("Generating insights..."):
                try:
                    response = requests.post(
                        "https://api.together.xyz/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "mistralai/Mistral-7B-Instruct-v0.2",
                            "messages": [
                                {"role": "system", "content": "You are an AI sales analyst."},
                                {"role": "user", "content": ai_prompt}
                            ],
                            "temperature": 0,
                            "max_tokens": 250
                        }
                    )
                    if response.status_code == 200:
                        raw_output = response.json()["choices"][0]["message"]["content"]
                        st.session_state["ai_insight"] = clean_numbered_list(raw_output)
                    else:
                        st.session_state["ai_insight"] = f"⚠️ API Error {response.status_code}: {response.text}"
                except Exception as e:
                    st.session_state["ai_insight"] = f"⚠️ Error generating insights: {e}"

        # ✅ Always show saved insight without regenerating
        st.success(st.session_state.get("ai_insight", "⚠️ No insights generated."))

        # ------------------- CHAT SECTION -------------------
        st.subheader("🤖 Ask the AI Sales Advisor")
        st.markdown("**Not sure what to ask? Try these:**")
        example_questions = [
            "What was my best sales month?",
            "Which products should I focus on next month?",
            "How can I increase sales by 10%?",
            "Which day of the week performs best?",
            "Do I have any declining products?"
        ]
        cols = st.columns(len(example_questions))
        for idx, q in enumerate(example_questions):
            if cols[idx].button(q):
                st.session_state["user_question_prefill"] = q

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        user_question = st.text_input(
            "Ask a question about your sales data:",
            value=st.session_state.get("user_question_prefill", "")
        )

        def answer_with_pandas(question, df):
            q = question.lower()
            try:
                if "best sales month" in q:
                    df['month'] = df['date'].dt.to_period('M')
                    best_month = df.groupby('month')['revenue'].sum().idxmax()
                    return f"Your best sales month was {best_month} with total revenue of ${df.groupby('month')['revenue'].sum().max():,.2f}."
                elif "day of the week" in q and "best" in q:
                    best_day = df.groupby(df['date'].dt.day_name())['revenue'].mean().idxmax()
                    return f"The best performing day of the week is {best_day}."
                elif "top product" in q or "best selling product" in q:
                    top_product = df.groupby('product')['revenue'].sum().idxmax()
                    return f"Your top-selling product is {top_product}."
            except:
                pass
            return None

        def ask_together_ai(api_key, question, df):
            summary = {
                "total_revenue": df['revenue'].sum(),
                "top_product": df.groupby('product')['revenue'].sum().idxmax(),
                "best_day": df.groupby(df['date'].dt.day_name())['revenue'].mean().idxmax(),
                "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
                "num_products": df['product'].nunique()
            }
            prompt = f"""
            You are an AI sales advisor. Use the following dataset summary to answer the question.

            Dataset Summary: {summary}

            Question: {question}

            Be specific and provide actionable business advice.
            If you list numbered points, stop numbering after the last real point.
            """

            url = "https://api.together.xyz/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            body = {
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "messages": [
                    {"role": "system", "content": "You are an AI sales analyst."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 256
            }
            response = requests.post(url, headers=headers, json=body)
            if response.status_code == 200:
                raw_output = response.json()['choices'][0]['message']['content']
                return clean_numbered_list(raw_output)
            else:
                return f"⚠️ API Error {response.status_code}: {response.text}"

        if api_key and user_question.strip():
            last_q = st.session_state["chat_history"][-1]["question"] if st.session_state["chat_history"] else None
            if user_question != last_q:
                with st.spinner("🤖 Thinking..."):
                    direct_answer = answer_with_pandas(user_question, df)
                    if direct_answer:
                        ai_response = direct_answer
                    else:
                        ai_response = ask_together_ai(api_key, user_question, df)

                st.session_state["chat_history"].append({
                    "question": user_question,
                    "answer": ai_response
                })

            st.session_state["user_question_prefill"] = ""  # clear prefill

        if st.session_state["chat_history"]:
            st.markdown("### 🗨️ Conversation History")
            for turn in st.session_state["chat_history"]:
                st.markdown(f"**You:** {turn['question']}")
                st.markdown(f"**🤖 Advisor:** {turn['answer']}")

        st.markdown("### Was this advice helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                st.success("Thank you for your feedback!")
        with col2:
            if st.button("👎 No"):
                st.warning("Thanks – your feedback helps us improve.")
