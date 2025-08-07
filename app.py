import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from transformers import pipeline
import requests

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

# Upload CSV
tab1, tab2 = st.tabs(["📤 Upload & Forecast", "🤖 AI Insights"])

with tab1:
    uploaded_file = st.file_uploader("Upload your sales data CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if 'date' not in df.columns:
            st.error("Your CSV must include a 'date' column.")
            st.stop()

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')

        st.subheader("📊 Preview of Your Data")
        st.dataframe(df.head())

        st.subheader("📈 Forecasting with Prophet")
        daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
        daily_revenue = daily_revenue.rename(columns={'date': 'ds', 'revenue': 'y'})

        model = Prophet()
        model.fit(daily_revenue)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        last_week = daily_revenue['y'].iloc[-14:-7].sum()
        this_week = daily_revenue['y'].iloc[-7:].sum()
        aov = (df['revenue'] / df['quantity']).mean()

        top_product = df.groupby('product')['revenue'].sum().idxmax()
        dow_stats = df.groupby(df['date'].dt.day_name())['revenue'].mean()
        best_days = dow_stats.sort_values(ascending=False).head(2).index.tolist()
        worst_day = dow_stats.idxmin()

        # KPI metrics
        st.subheader("📌 Key Business Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📅 Last Week Revenue", f"${last_week:,.2f}")
        with col2:
            delta = ((this_week - last_week) / last_week) * 100 if last_week != 0 else 0
            st.metric("📅 This Week Revenue", f"${this_week:,.2f}", delta=f"{delta:.2f}%")
        with col3:
            st.metric("💰 Avg Order Value", f"${aov:.2f}")

        st.session_state['summary_input'] = f"""
        As an AI sales advisor, summarize the following sales performance and give actionable advice in 3-4 full sentences:

        Last week's revenue: ${last_week:.2f}\n
        This week's revenue: ${this_week:.2f}\n
        Growth rate: {delta:.2f}%\n
        Top-selling product: {top_product}\n
        Best sales days: {', '.join(best_days)}\n
        Lowest sales day: {worst_day}\n
        Average order value: ${aov:.2f}
        """

with tab2:
    if 'summary_input' in st.session_state:
        st.subheader("💡 AI-Generated Business Insights")

        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        suggestion = summarizer(st.session_state['summary_input'], max_length=120, min_length=40, do_sample=False)[0]['summary_text']
        st.success(suggestion)

        st.subheader("🤖 Ask the AI Sales Advisor")
        user_question = st.text_input("Ask a question about your sales data:")

        if user_question:
            def ask_together_ai(prompt, api_key):
                url = "https://api.together.xyz/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                body = {
                    "model": "mistralai/Mistral-7B-Instruct-v0.2",
                    "messages": [
                        {"role": "system", "content": "You are an AI sales advisor helping small businesses."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 256
                }
                response = requests.post(url, headers=headers, json=body)
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                else:
                    return "⚠️ Could not get a response."

            api_key = st.secrets["TOGETHER_API_KEY"] if "TOGETHER_API_KEY" in st.secrets else st.text_input("Enter your Together.ai API key")

            if api_key:
                full_prompt = f"{st.session_state['summary_input']}\n\nQuestion: {user_question}"
                ai_response = ask_together_ai(full_prompt, api_key)
                st.markdown(f"**🤖 Advisor:** {ai_response}")

        st.markdown("### Was this advice helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                st.success("Thank you for your feedback!")
        with col2:
            if st.button("👎 No"):
                st.warning("Thanks – your feedback helps us improve.")
