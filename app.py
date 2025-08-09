import streamlit as st  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from transformers import pipeline
import requests
import time  # For retry delay
import torch
from transformers import pipeline


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

        st.subheader("📊 Preview of Your Data")
        st.dataframe(df.head())

        # Calculate metrics first
        daily_revenue, last_week, this_week, delta, aov, top_product, best_days, worst_day, summary_input = calculate_metrics(df)
        st.session_state["summary_input"] = summary_input

        # 📌 Styled Key Business Metrics
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
            .metric-title {
                font-size: 16px;
                color: #333;
                font-weight: bold;
            }
            .metric-value {
                font-size: 24px;
                color: #004aad;
                font-weight: bold;
                margin-top: 5px;
            }
            .metric-delta {
                font-size: 14px;
                font-weight: bold;
            }
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

        # 📊 Extended Business Metrics
        st.subheader("📊 Extended Business Metrics")
        weekly_revenue = df.resample('W-SUN', on='date')['revenue'].sum().reset_index()
        st.markdown("**📆 Weekly Revenue:**")
        st.dataframe(weekly_revenue)

        aov_by_product = (df.groupby('product').apply(lambda x: (x['revenue'] / x['quantity']).mean()))
        st.markdown("**💰 AOV by Product:**")
        st.dataframe(aov_by_product.rename("AOV"))

        units_sold = df.groupby('product')['quantity'].sum()
        st.markdown("**📦 Units Sold per Product:**")
        st.dataframe(units_sold.rename("Units Sold"))

        revenue_share = df.groupby('product')['revenue'].sum()
        revenue_share_percent = (revenue_share / revenue_share.sum() * 100).round(2)
        st.markdown("**📈 Revenue Share (%):**")
        st.dataframe(revenue_share_percent.rename("Revenue %"))

        avg_revenue_by_dow = df.groupby(df['date'].dt.day_name())['revenue'].mean()
        st.markdown("**📅 Average Revenue by Day of Week:**")
        st.dataframe(avg_revenue_by_dow.rename("Avg Revenue"))

        # 📈 Forecasting LAST
        st.subheader("📈 Forecasting with Prophet")
        with st.spinner("Training model... please wait"):
            model, forecast = train_prophet(daily_revenue)
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

with tab2:
    if "summary_input" in st.session_state:
        st.subheader("💡 AI-Generated Business Insights")

        with st.spinner("Generating insights..."):
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            suggestion = summarizer(st.session_state['summary_input'], max_length=120, min_length=40, do_sample=False)[0]['summary_text']
        st.success(suggestion)

        st.subheader("🤖 Ask the AI Sales Advisor")

        # Default suggestions
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

        # Chat history init
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Text input
        user_question = st.text_input(
            "Ask a question about your sales data:",
            value=st.session_state.get("user_question_prefill", "")
        )

        def ask_together_ai_with_memory(api_key, retries=3, delay=2):
            # Build conversation with memory
            history_messages = [
                {"role": "system", "content": "You are an AI sales advisor helping small businesses based on the provided sales metrics."},
                {"role": "system", "content": st.session_state['summary_input']}
            ]
            for turn in st.session_state["chat_history"][-5:]:
                history_messages.append({"role": "user", "content": turn["question"]})
                history_messages.append({"role": "assistant", "content": turn["answer"]})
            history_messages.append({"role": "user", "content": user_question})

            url = "https://api.together.xyz/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            body = {
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "messages": history_messages,
                "temperature": 0.7,
                "max_tokens": 256
            }

            for attempt in range(retries):
                try:
                    response = requests.post(url, headers=headers, json=body)
                    if response.status_code == 429:
                        st.warning("⚠️ Rate limit hit. Retrying...")
                        time.sleep(delay)
                        continue
                    if response.status_code == 200:
                        return response.json()['choices'][0]['message']['content']
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                        time.sleep(delay)
                except requests.exceptions.RequestException as e:
                    st.error(f"⚠️ Request failed: {e}")
                    time.sleep(delay)
            return "⚠️ Could not get a response after multiple attempts."

         api_key = st.secrets["TOGETHER_API_KEY"]  # Will error if secret not set

        if api_key and user_question:
            with st.spinner("🤖 Thinking... Generating advice..."):
                ai_response = ask_together_ai_with_memory(api_key)

            # Incomplete / ambiguous response handling
            vague_phrases = ["i don't know", "not sure", "cannot", "unsure", "not enough information", "need more details"]
            if len(ai_response.strip()) < 20 or any(phrase in ai_response.lower() for phrase in vague_phrases):
                ai_response = "⚠️ Please clarify your question so I can give you better advice."

            # Store in chat history
            st.session_state["chat_history"].append({"question": user_question, "answer": ai_response})

        # Show conversation history
        if st.session_state["chat_history"]:
            st.markdown("### 🗨️ Conversation History")
            for turn in st.session_state["chat_history"]:
                st.markdown(f"**You:** {turn['question']}")
                st.markdown(f"**🤖 Advisor:** {turn['answer']}")

        # Feedback
        st.markdown("### Was this advice helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                st.success("Thank you for your feedback!")
        with col2:
            if st.button("👎 No"):
                st.warning("Thanks – your feedback helps us improve.")



