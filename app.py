"""
📈 Stock Analyst AI — Streamlit frontend
Section 9 of SYS_PROMPT.md

Layout:
  Sidebar  : ticker input, analysis type, Analyse button, recent stocks
  Main     : signal banner → key summary → long/short columns
             → scores chart → expandable detail sections → export
"""

from __future__ import annotations

import asyncio
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from agents.orchestrator import AnalysisOrchestrator
from config.settings import validate_environment
from models.report_schema import StockReport
from utils.formatter import report_to_markdown

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📈 Stock Analyst AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS injection ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Signal banners */
    .sig-banner {
        padding: 1.1rem 1.4rem;
        border-radius: 12px;
        color: #fff;
        margin-bottom: 1.2rem;
    }
    .sig-green  { background: linear-gradient(135deg, #1b5e20 0%, #43a047 100%); }
    .sig-yellow { background: linear-gradient(135deg, #bf360c 0%, #fb8c00 100%); }
    .sig-red    { background: linear-gradient(135deg, #b71c1c 0%, #e53935 100%); }

    /* Signal badges (inline) */
    .badge       { border-radius: 6px; padding: 3px 10px; font-weight: 700; }
    .badge-buy   { background: #e8f5e9; color: #1b5e20; }
    .badge-hold  { background: #fff8e1; color: #e65100; }
    .badge-sell  { background: #ffebee; color: #b71c1c; }

    /* Clean divider */
    hr { margin: 1.2rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Pure-Python helpers ──────────────────────────────────────────────────────


def _signal_css(signal: str) -> str:
    """Map signal string → CSS class for the banner."""
    upper = (signal or "").upper()
    if upper in {"STRONG BUY", "BUY"}:
        return "sig-green"
    if upper in {"HOLD", "ACCUMULATE", "WAIT"}:
        return "sig-yellow"
    return "sig-red"


def _badge_html(signal: str) -> str:
    upper = (signal or "").upper()
    if "BUY" in upper:
        cls = "badge-buy"
    elif "SELL" in upper or "AVOID" in upper or "REDUCE" in upper:
        cls = "badge-sell"
    else:
        cls = "badge-hold"
    return f'<span class="badge {cls}">{signal}</span>'


def _pct_delta(target: float, entry: float) -> str:
    if not entry:
        return ""
    pct = (target - entry) / entry * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _fmt_float(val: Any, fmt: str = ".2f", prefix: str = "", suffix: str = "") -> str:
    """Safely format a numeric value."""
    if val is None:
        return "N/A"
    try:
        return f"{prefix}{float(val):{fmt}}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def _scores_chart(report: StockReport) -> go.Figure:
    categories = ["Technical", "Fundamental", "Sentiment", "Management", "Valuation"]
    values = [
        report.scores.technical,
        report.scores.fundamental,
        report.scores.sentiment,
        report.scores.management,
        report.scores.valuation,
    ]
    colors = [
        "#43a047" if v > 7 else "#fb8c00" if v >= 4 else "#e53935" for v in values
    ]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=categories,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}/10" for v in values],
            textposition="auto",
            hovertemplate="%{y}: %{x:.1f}/10<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 10], title="Score out of 10", tickfont=dict(size=11)),
        yaxis=dict(title=""),
        height=250,
        margin=dict(l=10, r=20, t=10, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,240,240,0.3)",
    )
    return fig


# ─── Session state ────────────────────────────────────────────────────────────
for _k, _v in [
    ("recent_stocks", []),
    ("last_report", None),
    ("last_markdown", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─── Environment check ────────────────────────────────────────────────────────
_env = validate_environment()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock Analyst AI")
    st.caption("Multi-Agent AI · NSE / BSE · Powered by Claude")
    st.divider()

    ticker_input: str = st.text_input(
        "Stock Ticker",
        placeholder="HDFCBANK, INFY, RELIANCE",
        help="Enter NSE or BSE ticker. Exchange suffix (.NS / .BO) is added automatically.",
    )

    analysis_type: str = st.selectbox(
        "Analysis Type",
        ["Full Analysis", "Quick Technical", "Sentiment Only"],
        help=(
            "Full Analysis — runs all 5 specialist agents.\n"
            "Quick Technical — stock data + technical indicators only.\n"
            "Sentiment Only — news + management intelligence only."
        ),
    )

    if not _env["anthropic_key_configured"]:
        st.warning("⚠️ ANTHROPIC_API_KEY not set in .env")
    if not _env["exa_key_configured"]:
        st.warning("⚠️ EXA_API_KEY not set in .env")

    analyse_clicked = st.button(
        "🔍 Analyse Stock",
        type="primary",
        use_container_width=True,
        disabled=not ticker_input.strip(),
    )

    if st.session_state.recent_stocks:
        st.divider()
        st.markdown("**🕘 Recently Analysed**")
        for _s in reversed(st.session_state.recent_stocks[-8:]):
            st.markdown(f"&nbsp;&nbsp;`{_s}`")

# ─── Page header ─────────────────────────────────────────────────────────────
st.title("📈 Stock Analyst AI")
st.caption(
    "Specialist AI agents analyse **fundamentals**, **technicals**, **news sentiment**, "
    "and **management quality** — synthesised into clear ₹ BUY / SELL signals."
)

# ─── Run analysis ─────────────────────────────────────────────────────────────
if analyse_clicked:
    raw_ticker = ticker_input.strip().upper()
    if not raw_ticker:
        st.warning("Enter a ticker symbol in the sidebar first.")
        st.stop()

    with st.status("🔄 Running analysis…", expanded=True) as _status:
        st.write("🔍 Fetching stock data…")
        st.write("📊 Running technical analysis…")
        st.write("🌐 Scanning news and sentiment…")
        st.write("👔 Checking management signals…")
        st.write("🧠 Synthesising with AI…")

        try:
            _orchestrator = AnalysisOrchestrator()
            _report, _json_path, _md_path = asyncio.run(
                _orchestrator.analyze_stock(raw_ticker)
            )
            _markdown = report_to_markdown(_report)

            st.session_state.last_report = _report
            st.session_state.last_markdown = _markdown

            if raw_ticker not in st.session_state.recent_stocks:
                st.session_state.recent_stocks.append(raw_ticker)

            _status.update(
                label="✅ Analysis complete!", state="complete", expanded=False
            )
        except Exception as _exc:
            _status.update(label="❌ Analysis failed", state="error", expanded=True)
            st.error(f"**Error:** {_exc}")
            st.exception(_exc)
            st.stop()

# ─── Display results ──────────────────────────────────────────────────────────
if st.session_state.last_report is None:
    # Welcome placeholder shown before first analysis
    st.info(
        "👈 Enter a ticker in the sidebar and click **Analyse Stock** to start.\n\n"
        "Example tickers: `HDFCBANK`, `RELIANCE`, `INFY`, `TCS`, `BAJFINANCE`"
    )
    st.stop()

rpt: StockReport = st.session_state.last_report
md_text: str = st.session_state.last_markdown

# ── 1. Signal Banner ──────────────────────────────────────────────────────────
lt_signal = rpt.long_term.signal.value
st.markdown(
    f"""
    <div class="sig-banner {_signal_css(lt_signal)}">
        <h2 style="margin:0;font-size:1.9rem;letter-spacing:.5px;">🎯 {lt_signal}</h2>
        <p style="margin:6px 0 0;font-size:1rem;opacity:.92;">
            <b>{rpt.ticker}</b> &mdash; {rpt.company_name}
            &nbsp;|&nbsp; Price: ₹{rpt.current_price:,.2f}
            &nbsp;|&nbsp; Overall Score: <b>{rpt.scores.overall:.1f}/10</b>
            &nbsp;|&nbsp; Confidence: <b>{rpt.long_term.confidence_pct}%</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 2. Key Summary ────────────────────────────────────────────────────────────
_bullets = "\n".join(f"• {item}" for item in rpt.key_summary)
st.info(f"**⚡ Key Summary**\n\n{_bullets}")

# ── 3. Long-Term | Short-Term columns ────────────────────────────────────────
col_lt, col_st = st.columns(2, gap="large")

with col_lt:
    lt = rpt.long_term
    st.subheader("📅 Long-Term (1–3 Years)")
    st.markdown(_badge_html(lt.signal.value), unsafe_allow_html=True)
    st.markdown(" ")

    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Entry Price", f"₹{lt.entry_price:,.2f}")
    r1c2.metric(
        "1-Year Target",
        f"₹{lt.target_1_year:,.2f}",
        _pct_delta(lt.target_1_year, lt.entry_price),
    )
    r1c3.metric(
        "3-Year Target",
        f"₹{lt.target_3_year:,.2f}",
        _pct_delta(lt.target_3_year, lt.entry_price),
    )

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("Stop Loss", f"₹{lt.stop_loss_price:,.2f}")
    r2c2.metric("Exp. CAGR", f"{lt.expected_cagr:.1f}%")
    r2c3.metric("Risk Level", lt.risk_level.value)

    with st.expander("📝 Reasoning, Catalysts & Exit Triggers"):
        st.markdown(lt.reasoning)
        st.markdown(f"**Entry Condition:** {lt.entry_condition}")
        if lt.exit_triggers:
            st.markdown("**Exit if:**")
            for _t in lt.exit_triggers:
                st.markdown(f"- {_t}")
        if lt.key_catalysts:
            st.markdown("**Key Catalysts:**")
            for _c in lt.key_catalysts:
                st.markdown(f"- {_c}")
        if lt.key_risks:
            st.markdown("**Key Risks:**")
            for _r in lt.key_risks:
                st.markdown(f"- {_r}")

with col_st:
    short = rpt.short_term
    st.subheader("⚡ Short-Term (2–6 Weeks)")
    st.markdown(_badge_html(short.signal.value), unsafe_allow_html=True)
    st.markdown(" ")

    s1c1, s1c2, s1c3 = st.columns(3)
    s1c1.metric("Entry", f"₹{short.entry_price:,.2f}")
    s1c2.metric(
        "Target 1 (40%)",
        f"₹{short.target_1:,.2f}",
        _pct_delta(short.target_1, short.entry_price),
    )
    s1c3.metric(
        "Target 2 (40%)",
        f"₹{short.target_2:,.2f}",
        _pct_delta(short.target_2, short.entry_price),
    )

    s2c1, s2c2, s2c3 = st.columns(3)
    s2c1.metric("Hard Stop", f"₹{short.stop_loss:,.2f}")
    s2c2.metric("Setup Type", short.trade_setup_type)
    s2c3.metric("Confidence", f"{short.confidence_pct}%")

    if short.target_3 is not None:
        st.metric(
            "Target 3 (trail 20%)",
            f"₹{short.target_3:,.2f}",
            _pct_delta(short.target_3, short.entry_price),
        )

    st.caption(
        f"**R/R:** {short.risk_reward_ratio} &nbsp;&nbsp;|&nbsp;&nbsp;"
        f"**Hold:** {short.holding_period}"
    )

    with st.expander("📝 Trade Setup & Reasoning"):
        st.markdown(short.reasoning)
        st.markdown(f"**Entry Condition:** {short.entry_condition}")

st.divider()

# ── 4. Scores ─────────────────────────────────────────────────────────────────
st.subheader("📊 Signal Scorecard")
chart_col, agree_col = st.columns([3, 1])

with chart_col:
    st.plotly_chart(_scores_chart(rpt), use_container_width=True, key="scores_chart")

with agree_col:
    st.markdown("**Agent Agreement**")
    _total = (
        rpt.scores.agents_bullish
        + rpt.scores.agents_bearish
        + rpt.scores.agents_neutral
    )
    st.metric("🟢 Bullish", rpt.scores.agents_bullish)
    st.metric("🟡 Neutral", rpt.scores.agents_neutral)
    st.metric("🔴 Bearish", rpt.scores.agents_bearish)
    if _total:
        _bull_pct = rpt.scores.agents_bullish / _total
        st.progress(
            _bull_pct,
            text=f"{int(_bull_pct * 100)}% agents bullish",
        )

st.divider()

# ── 5. Expandable detail sections ────────────────────────────────────────────
raw_tech = rpt.raw_technical or {}
raw_fund = rpt.raw_fundamental or {}
raw_sent = rpt.raw_sentiment or {}
raw_mgmt = rpt.raw_management or {}

with st.expander("📊 Technical Analysis"):
    ta_c1, ta_c2, ta_c3 = st.columns(3)
    ta_c1.metric("RSI (14)", _fmt_float(raw_tech.get("rsi"), ".1f"))
    ta_c2.metric("MACD", _fmt_float(raw_tech.get("macd"), ".3f"))
    ta_c3.metric("ADX (14)", _fmt_float(raw_tech.get("adx"), ".1f"))

    tb_c1, tb_c2, tb_c3 = st.columns(3)
    tb_c1.metric("SMA 20", _fmt_float(raw_tech.get("sma_20"), ",.2f", "₹"))
    tb_c2.metric("SMA 50", _fmt_float(raw_tech.get("sma_50"), ",.2f", "₹"))
    tb_c3.metric("SMA 200", _fmt_float(raw_tech.get("sma_200"), ",.2f", "₹"))

    tc_c1, tc_c2, tc_c3 = st.columns(3)
    tc_c1.metric("ATR (14)", _fmt_float(raw_tech.get("atr"), ",.2f", "₹"))
    tc_c2.metric("BB Upper", _fmt_float(raw_tech.get("bb_upper"), ",.2f", "₹"))
    tc_c3.metric("BB Lower", _fmt_float(raw_tech.get("bb_lower"), ",.2f", "₹"))

    patterns = raw_tech.get("patterns_detected") or []
    if patterns:
        st.success("**Patterns detected:** " + "  ·  ".join(patterns))

    supports = raw_tech.get("support_levels") or []
    resistances = raw_tech.get("resistance_levels") or []
    if supports:
        st.markdown(
            "**Support levels:** " + "  |  ".join(f"₹{v:,.2f}" for v in supports)
        )
    if resistances:
        st.markdown(
            "**Resistance levels:** " + "  |  ".join(f"₹{v:,.2f}" for v in resistances)
        )

    _sig_reason = raw_tech.get("signal_reasoning", "")
    if _sig_reason:
        st.caption(f"Signal reasoning: {_sig_reason}")

with st.expander("💰 Fundamental Analysis"):
    _pe = raw_fund.get("pe_ratio") or {}
    fa_c1, fa_c2, fa_c3, fa_c4 = st.columns(4)
    fa_c1.metric("P/E (Trailing)", _fmt_float(_pe.get("trailing"), ".1f", suffix="x"))
    fa_c2.metric("P/E (Forward)", _fmt_float(_pe.get("forward"), ".1f", suffix="x"))
    fa_c3.metric("P/B Ratio", _fmt_float(raw_fund.get("pb_ratio"), ".2f", suffix="x"))
    fa_c4.metric("EV/EBITDA", _fmt_float(raw_fund.get("ev_ebitda"), ".1f", suffix="x"))

    fb_c1, fb_c2, fb_c3, fb_c4 = st.columns(4)
    fb_c1.metric("ROE", _fmt_float(raw_fund.get("roe"), ".1f", suffix="%"))
    fb_c2.metric(
        "Revenue Growth", _fmt_float(raw_fund.get("revenue_growth"), ".1f", suffix="%")
    )
    fb_c3.metric(
        "Net Margin", _fmt_float(raw_fund.get("profit_margin"), ".1f", suffix="%")
    )
    fb_c4.metric(
        "Debt / Equity",
        _fmt_float(raw_fund.get("debt_to_equity"), ".2f", suffix="x"),
    )

    fc_c1, fc_c2, fc_c3, fc_c4 = st.columns(4)
    fc_c1.metric("Valuation", raw_fund.get("valuation_label", "N/A"))
    fc_c2.metric("Fin. Health", raw_fund.get("financial_health", "N/A"))
    _iv = raw_fund.get("intrinsic_value")
    _up = raw_fund.get("upside_pct")
    fc_c3.metric(
        "DCF Value",
        _fmt_float(_iv, ",.2f", "₹"),
        _fmt_float(_up, ".1f", suffix="% upside") if _up is not None else None,
    )
    fc_c4.metric(
        "Dividend Yield",
        _fmt_float(raw_fund.get("dividend_yield"), ".2f", suffix="%"),
    )

    _strengths = raw_fund.get("strengths") or []
    _concerns = raw_fund.get("concerns") or []
    if _strengths:
        st.success("**Strengths:** " + "  ·  ".join(_strengths))
    if _concerns:
        st.warning("**Concerns:** " + "  ·  ".join(_concerns))

with st.expander("📰 News & Sentiment"):
    se_c1, se_c2, se_c3 = st.columns(3)
    se_c1.metric("Overall Sentiment", raw_sent.get("overall_sentiment", "N/A"))
    se_c2.metric("Weighted Score", _fmt_float(raw_sent.get("sentiment_score"), ".2f"))
    _avg_tp = raw_sent.get("avg_target_price")
    se_c3.metric("Avg Analyst Target", _fmt_float(_avg_tp, ",.2f", "₹"))

    sa_c1, sa_c2, sa_c3 = st.columns(3)
    sa_c1.metric("✅ Positive", raw_sent.get("positive_count", 0))
    sa_c2.metric("⬜ Neutral", raw_sent.get("neutral_count", 0))
    sa_c3.metric("❌ Negative", raw_sent.get("negative_count", 0))

    _themes = raw_sent.get("sentiment_themes") or []
    if _themes:
        st.markdown("**Top Themes:** " + "  ·  ".join(_themes))

    _analyst = raw_sent.get("analyst_consensus") or {}
    if _analyst:
        _ac1, _ac2, _ac3, _ac4 = st.columns(4)
        _ac1.metric("Strong Buy", _analyst.get("strong_buy", 0))
        _ac2.metric("Buy", _analyst.get("buy", 0))
        _ac3.metric("Hold", _analyst.get("hold", 0))
        _ac4.metric("Sell", _analyst.get("sell", 0))

    _articles = (raw_sent.get("articles") or [])[:10]
    if _articles:
        st.markdown("**Recent Articles:**")
        for _art in _articles:
            _sent = _art.get("sentiment", "NEUTRAL")
            _icon = (
                "✅" if _sent == "POSITIVE" else "❌" if _sent == "NEGATIVE" else "⬜"
            )
            _imp = _art.get("impact_level", "LOW")
            _title = _art.get("headline") or "—"
            _url = _art.get("url") or ""
            _src = _art.get("source") or ""
            _link = f"[{_title}]({_url})" if _url else _title
            st.markdown(f"{_icon} {_link}  \n`{_src}` · **{_imp}** impact")

with st.expander("👔 Management & Insider Intelligence"):
    mg_c1, mg_c2, mg_c3 = st.columns(3)
    mg_c1.metric(
        "Mgmt Score", _fmt_float(raw_mgmt.get("management_score"), ".1f", suffix="/10")
    )
    mg_c2.metric("Quality Label", raw_mgmt.get("management_label", "N/A"))
    _ia = (raw_mgmt.get("insider_activity") or "N/A").title()
    mg_c3.metric("Insider Activity", _ia)

    mm_c1, mm_c2, mm_c3 = st.columns(3)
    _pht = (raw_mgmt.get("promoter_holding_trend") or "N/A").title()
    mm_c1.metric("Promoter Trend", _pht)
    _mg = (raw_mgmt.get("management_guidance") or "N/A").title()
    mm_c2.metric("Guidance", _mg)
    mm_c3.metric("Red Flags", len(raw_mgmt.get("red_flags") or []))

    _lc = raw_mgmt.get("leadership_changes") or []
    if _lc:
        st.warning("**Leadership Changes:** " + " | ".join(_lc[:3]))

    _sp = raw_mgmt.get("strategic_plans") or []
    if _sp:
        st.info("**Strategic Plans:**")
        for _plan in _sp[:3]:
            st.markdown(f"- {_plan}")

    _cats = raw_mgmt.get("upcoming_catalysts") or []
    if _cats:
        st.markdown("**Upcoming Catalysts:**")
        for _cat in _cats[:4]:
            _prob = _cat.get("positive_probability", "?")
            _evt = _cat.get("event", "—")
            st.markdown(f"- {_evt}  *(probability: {_prob})*")

    _rf = raw_mgmt.get("red_flags") or []
    if _rf:
        st.error("**⚠️ Red Flags:**")
        for _r in _rf:
            st.markdown(f"- {_r}")

st.divider()

# ── 6. Export ─────────────────────────────────────────────────────────────────
st.subheader("📥 Export")

exp_c1, exp_c2, exp_c3 = st.columns(3)

with exp_c1:
    st.download_button(
        label="📄 Download Full Report",
        data=md_text.encode("utf-8"),
        file_name=f"{rpt.ticker}_report_{rpt.analysis_date}.md",
        mime="text/markdown",
        use_container_width=True,
        help="Downloads the complete Markdown report (open in any Markdown viewer or convert to PDF).",
    )

with exp_c2:
    _json_bytes = rpt.model_dump_json(indent=2).encode("utf-8")
    st.download_button(
        label="📊 Download Raw JSON",
        data=_json_bytes,
        file_name=f"{rpt.ticker}_data_{rpt.analysis_date}.json",
        mime="application/json",
        use_container_width=True,
        help="Downloads all agent outputs and scores as a structured JSON file.",
    )

with exp_c3:
    # Show the clean ticker in a copyable code block for adding to Groww watchlist
    _clean_ticker = rpt.ticker.replace(".NS", "").replace(".BO", "")
    if st.button(
        "📋 Copy for Groww Watchlist",
        use_container_width=True,
        help="Copy the ticker symbol to search for on Groww.",
    ):
        st.toast(f"Ticker: {_clean_ticker}", icon="✅")
    st.code(_clean_ticker, language=None)
