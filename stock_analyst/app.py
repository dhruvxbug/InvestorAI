from __future__ import annotations

import asyncio

import streamlit as st

from agents.orchestrator import AnalysisOrchestrator
from config.settings import validate_environment
from utils.formatter import report_to_markdown

st.set_page_config(page_title="InvestorAI — Multi-Agent Stock Analyst", layout="wide")
st.title("📈 InvestorAI — Indian Stock Multi-Agent Analyst")

env = validate_environment()
if not env["anthropic_key_configured"]:
    st.error("ANTHROPIC_API_KEY missing in stock_analyst/.env")
if not env["exa_key_configured"]:
    st.error("EXA_API_KEY missing in stock_analyst/.env")

user_ticker = st.text_input("Enter NSE/BSE ticker (e.g. RELIANCE, INFY, HDFCBANK)", "RELIANCE")

if st.button("Run Analysis", type="primary"):
    try:
        with st.spinner("Running specialist agents in parallel..."):
            orchestrator = AnalysisOrchestrator()
            report, json_path, md_path = asyncio.run(orchestrator.analyze_stock(user_ticker))

        st.success("Analysis complete")
        st.subheader("Final Recommendation")
        c1, c2, c3 = st.columns(3)
        c1.metric("Long-Term Signal", report.long_term.signal.value)
        c2.metric("Short-Term Signal", report.short_term.signal.value)
        c3.metric("Current Price", f"₹{report.current_price:.2f}")

        st.markdown(report_to_markdown(report))
        st.caption(f"Saved JSON: {json_path}")
        st.caption(f"Saved Markdown: {md_path}")

    except Exception as exc:
        st.exception(exc)
