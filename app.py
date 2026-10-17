import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from src.seo_audit import audit_page, recommendations
from src.pagespeed import run_pagespeed
from src.free_chatbot import answer_dashboard_question

# Optional RAG imports
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL ANIMATION
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(ellipse at 8% 5%, rgba(91, 117, 238, 0.28), transparent 34%),
            radial-gradient(ellipse at 92% 18%, rgba(30, 190, 171, 0.22), transparent 33%),
            radial-gradient(ellipse at 55% 100%, rgba(152, 175, 255, 0.20), transparent 42%),
            linear-gradient(145deg, #e9efff 0%, #f0f2ff 46%, #e5f6f2 100%);
        background-attachment: fixed;
        color: #17233c;
        font-family: Inter, "Segoe UI", sans-serif;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: transparent !important;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1440px;
        background: transparent !important;
        border: 0;
        border-radius: 26px;
        padding: 2.1rem 2.1rem 2.8rem;
        box-shadow: none;
        backdrop-filter: none;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eef2ff 0%, #f3f8ff 58%, #eaf8f5 100%);
    }

    h1, h2, h3, h4, p, label, [data-testid="stCaptionContainer"] {
        color: #17233c;
    }

    h2, h3 {
        position: relative;
        padding-left: 13px;
        letter-spacing: -0.025em;
        animation: sectionEnter 0.55s ease both;
    }

    h2::before, h3::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.16em;
        bottom: 0.12em;
        width: 4px;
        border-radius: 99px;
        background: linear-gradient(180deg, #5b70ed, #28bba9);
    }

    [data-testid="stDateInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        border-radius: 12px;
        border-color: #d8e1f1;
        background: rgba(255, 255, 255, 0.92);
        color: #17233c;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff, #f6f9ff);
        border: 1px solid #e5ebf6;
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 8px 24px rgba(39, 62, 108, 0.07);
        animation: cardEnter 0.7s ease both;
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetricValue"] {
        color: #17233c !important;
    }

    [data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        border-left: 4px solid #6477eb;
    }

    [data-testid="stMetric"]::after {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        opacity: 0;
        background: linear-gradient(112deg, transparent 18%, rgba(112, 151, 255, 0.12) 49%, transparent 78%);
        transform: translateX(-100%);
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-7px) scale(1.025) !important;
        border-color: #67cabb;
        box-shadow: 0 18px 34px rgba(54, 82, 157, 0.19), 0 0 0 1px rgba(99, 119, 235, 0.10) !important;
        z-index: 2;
    }

    [data-testid="stMetric"]:hover::after {
        opacity: 1;
        animation: metricShine 0.75s ease-out;
    }

    [data-testid="stMetric"]:hover [data-testid="stMetricValue"] {
        color: #4f63d5 !important;
        transition: color 0.2s ease;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid #dce5f5;
        color: #223557;
        background: linear-gradient(135deg, #ffffff, #f3f7ff);
        font-weight: 650;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        border-color: #91a9f8;
        box-shadow: 0 8px 20px rgba(79, 105, 210, 0.16);
    }

    .stButton > button[kind="primary"] {
        color: #ffffff;
        border: 0;
        background: linear-gradient(110deg, #4968e8, #7355d9 58%, #188f9d);
        box-shadow: 0 8px 22px rgba(83, 94, 204, 0.22);
    }

    .st-key-mm_auth_stage {
        box-sizing: border-box;
        min-height: 440px;
        margin: 1.8rem auto 0;
        padding: 2.5rem 1.25rem;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: 1px solid rgba(255,255,255,0.7);
        border-radius: 28px;
        background:
            radial-gradient(ellipse at 12% 12%, rgba(104, 236, 205, 0.38), transparent 35%),
            radial-gradient(ellipse at 90% 88%, rgba(113, 151, 255, 0.28), transparent 36%),
            linear-gradient(132deg, #17345e 0%, #236d78 54%, #39b99f 100%);
        box-shadow: 0 20px 45px rgba(31, 58, 109, 0.19);
    }

    .st-key-mm_auth_card {
        box-sizing: border-box;
        width: min(100%, 520px) !important;
        margin: 0 auto !important;
        padding: 2.25rem 2rem 1.8rem;
        border: 1px solid rgba(226, 232, 241, 0.95);
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.98);
        box-shadow: 0 16px 38px rgba(13, 33, 65, 0.2);
        text-align: center;
    }

    .mm-auth-title {
        margin: 0 0 0.45rem;
        color: #17233c;
        font-size: clamp(1.55rem, 3vw, 2rem);
        font-weight: 760;
        letter-spacing: -0.035em;
        line-height: 1.25;
    }

    .mm-auth-copy {
        margin: 0 0 1.1rem;
        color: #53627d;
        font-size: 0.96rem;
        line-height: 1.55;
    }

    .st-key-mm_auth_card [data-testid="st-key-mm_google_login"],
    .st-key-mm_auth_card .st-key-mm_google_login {
        width: 100% !important;
        display: block !important;
    }

    .st-key-mm_auth_card [data-testid="st-key-mm_google_login"] [data-testid="stButton"],
    .st-key-mm_auth_card .st-key-mm_google_login [data-testid="stButton"] {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
    }

    .st-key-mm_auth_card [data-testid="st-key-mm_google_login"] button,
    .st-key-mm_auth_card .st-key-mm_google_login button {
        width: min(100%, 380px) !important;
        flex: 0 1 380px !important;
        min-height: 48px;
        margin: 0 auto !important;
        position: relative;
        border: 1px solid #dadce0 !important;
        border-radius: 9px !important;
        color: #3c4043 !important;
        background: #ffffff !important;
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.16) !important;
        font-size: 0.96rem;
        font-weight: 600;
    }

    .st-key-mm_auth_card [data-testid="st-key-mm_google_login"] button:hover,
    .st-key-mm_auth_card .st-key-mm_google_login button:hover {
        border-color: #c5c9ce !important;
        background: #f8faff !important;
        box-shadow: 0 2px 5px rgba(60, 64, 67, 0.2) !important;
        transform: translateY(-1px);
    }

    .st-key-mm_auth_card [data-testid="st-key-mm_google_login"] button::before,
    .st-key-mm_auth_card .st-key-mm_google_login button::before {
        content: "G";
        position: absolute;
        left: 18px;
        top: 50%;
        transform: translateY(-50%);
        background: conic-gradient(from -45deg, #4285f4 0 25%, #34a853 25% 50%, #fbbc05 50% 75%, #ea4335 75% 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: Arial, sans-serif;
        font-size: 1.2rem;
        font-weight: 750;
    }

    [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e5ebf5;
    }

    [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        color: #53627d;
    }

    [data-testid="stDataFrame"], [data-testid="stFileUploader"] section {
        border-radius: 16px;
        border-color: #e3eaf5;
        background: rgba(255, 255, 255, 0.78);
    }

    [data-testid="stPlotlyChart"] {
        overflow: hidden;
        animation: sectionEnter 0.65s ease both;
        border: 1px solid rgba(224, 232, 245, 0.95);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.84);
        box-shadow: 0 8px 25px rgba(39, 62, 108, 0.06);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    [data-testid="stPlotlyChart"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 32px rgba(39, 62, 108, 0.13);
    }

    [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid #e3eaf5;
        box-shadow: 0 7px 22px rgba(39, 62, 108, 0.05);
        transition: box-shadow 0.25s ease, transform 0.25s ease;
    }

    [data-testid="stDataFrame"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(39, 62, 108, 0.11);
    }

    hr {
        height: 1px;
        border: 0;
        background: linear-gradient(90deg, transparent, #cbd7ef 18%, #bde9e2 82%, transparent);
    }

    .stAlert {
        animation: fadeIn 0.7s ease-out;
        border-radius: 14px;
    }

    .st-key-marketmind_chat_launcher {
        position: fixed !important;
        right: 26px !important;
        bottom: 82px !important;
        z-index: 9999 !important;
        width: 68px !important;
        min-width: 68px !important;
    }

    .st-key-marketmind_chat_launcher [data-testid="stButton"] {
        position: relative;
        z-index: 1;
    }

    .st-key-marketmind_chat_launcher button {
        width: 62px !important;
        height: 62px !important;
        min-height: 62px !important;
        padding: 0 !important;
        border: 2px solid rgba(255, 255, 255, 0.94) !important;
        border-radius: 50% !important;
        color: #fff !important;
        background: linear-gradient(145deg, #526ce8, #7755d7 58%, #159a9f) !important;
        box-shadow: 0 10px 26px rgba(57, 72, 164, 0.34) !important;
        font-size: 1.65rem !important;
        animation: chatBob 2.8s ease-in-out infinite;
    }

    .st-key-marketmind_chat_launcher button:hover {
        transform: translateY(-4px) scale(1.06) !important;
        box-shadow: 0 14px 30px rgba(57, 72, 164, 0.42) !important;
    }

    .st-key-marketmind_chat_launcher::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 50%;
        background: rgba(102, 120, 238, 0.32);
        animation: chatPulse 2.2s ease-out infinite;
        pointer-events: none;
    }

    .st-key-marketmind_chat_launcher::after {
        content: "";
        position: absolute;
        top: 3px;
        right: 3px;
        z-index: 3;
        width: 13px;
        height: 13px;
        border: 2px solid #fff;
        border-radius: 50%;
        background: #37d69a;
        box-shadow: 0 2px 7px rgba(17, 120, 90, 0.25);
        pointer-events: none;
    }

    .st-key-marketmind_clear_chat button {
        min-width: 84px !important;
        white-space: nowrap !important;
        padding-left: 8px !important;
        padding-right: 8px !important;
        font-size: 0.82rem !important;
    }

    .st-key-marketmind_chat_panel {
        position: fixed !important;
        right: 26px !important;
        top: 88px !important;
        bottom: 82px !important;
        z-index: 9999 !important;
        width: min(390px, calc(100vw - 32px)) !important;
        max-height: none !important;
        height: auto !important;
        overflow-y: auto !important;
        padding: 18px !important;
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        border-radius: 22px !important;
        background: linear-gradient(150deg, rgba(255,255,255,0.98), rgba(241,247,255,0.98) 65%, rgba(232,250,247,0.98)) !important;
        box-shadow: 0 20px 54px rgba(25, 42, 82, 0.28) !important;
        backdrop-filter: blur(16px);
        animation: chatPanelIn 0.24s ease-out both;
    }

    .st-key-marketmind_chat_history {
        max-height: 350px !important;
        overflow-y: auto !important;
        padding-right: 4px;
    }

    @keyframes sectionEnter {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes chatPulse {
        0% { transform: scale(0.88); opacity: 0.7; }
        75%, 100% { transform: scale(1.55); opacity: 0; }
    }

    @keyframes chatBob {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-4px); }
    }

    @keyframes chatPanelIn {
        from { opacity: 0; transform: translateY(12px) scale(0.97); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    .mm-hero {
        position: relative;
        overflow: hidden;
        margin: 0.1rem 0 1.7rem;
        padding: 28px 34px 30px;
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 24px;
        color: #fff;
        background: linear-gradient(112deg, #14284d 0%, #294b8d 48%, #256d79 100%);
        box-shadow: 0 20px 45px rgba(31, 58, 109, 0.22);
        animation: heroEnter 0.85s cubic-bezier(.2,.75,.25,1) both;
    }

    .mm-hero::before, .mm-hero::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        pointer-events: none;
    }

    .mm-hero::before {
        width: 320px;
        height: 320px;
        top: -210px;
        right: 7%;
        background: rgba(117, 154, 255, 0.23);
        filter: blur(2px);
    }

    .mm-hero::after {
        width: 210px;
        height: 210px;
        right: -65px;
        bottom: -150px;
        background: rgba(85, 234, 209, 0.23);
        filter: blur(4px);
    }

    .mm-brand-row {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        gap: 15px;
    }

    .mm-logo {
        display: grid;
        place-items: center;
        width: 58px;
        height: 58px;
        flex: 0 0 58px;
        border: 1px solid rgba(255,255,255,0.48);
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(255,255,255,0.25), rgba(255,255,255,0.08));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.28), 0 8px 20px rgba(8,20,45,0.2);
        animation: logoGlow 3.5s ease-in-out infinite;
    }

    .mm-logo svg { width: 36px; height: 36px; }
    .mm-kicker { color: #c7dcff; font-size: 0.70rem; font-weight: 750; letter-spacing: 0.17em; }
    .mm-brand { color: #fff; font-size: clamp(1.55rem, 3vw, 2.05rem); font-weight: 800; letter-spacing: -0.045em; line-height: 1.15; }
    .mm-brand span { color: #75f0dc; }
    .mm-pill {
        margin-left: auto;
        padding: 8px 12px;
        border: 1px solid rgba(255,255,255,0.24);
        border-radius: 999px;
        color: #e5f2ff;
        background: rgba(255,255,255,0.10);
        font-size: 0.73rem;
        font-weight: 650;
        letter-spacing: 0.04em;
    }
    .mm-tagline { position: relative; z-index: 1; margin: 22px 0 4px; color: #fff; font-size: clamp(1.25rem, 2.5vw, 1.75rem); font-weight: 720; letter-spacing: -0.035em; }
    .mm-description { position: relative; z-index: 1; max-width: 760px; margin: 0; color: #dce8fb; font-size: 0.98rem; line-height: 1.65; }

    @keyframes heroEnter {
        from { opacity: 0; transform: translateY(-14px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes logoGlow {
        0%, 100% { box-shadow: inset 0 1px 0 rgba(255,255,255,0.28), 0 8px 20px rgba(8,20,45,0.2); }
        50% { box-shadow: inset 0 1px 0 rgba(255,255,255,0.38), 0 0 24px rgba(117,240,220,0.34); }
    }

    @keyframes metricShine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    @keyframes cardEnter {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] { padding: 1.4rem 1rem 2rem; }
        .mm-hero { padding: 22px 20px 24px; }
        .mm-logo { width: 50px; height: 50px; flex-basis: 50px; }
        .mm-pill { display: none; }
        .st-key-mm_auth_stage { min-height: 380px; margin-top: 1.2rem; padding: 1.4rem 0.8rem; }
        .st-key-mm_auth_card { padding: 1.7rem 1rem 1.4rem; }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
    }

    </style>
    """,
    unsafe_allow_html=True
)


def get_app_secret(name, default=""):
    """Read deployment secrets without requiring a local secrets file."""
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


@st.cache_resource
def _pagespeed_rate_state():
    return {"timestamps": [], "lock": threading.Lock()}


def allow_pagespeed_request():
    """Limit aggregate PageSpeed calls per app process to control public quota use."""
    try:
        max_calls = max(1, int(get_app_secret("PAGESPEED_MAX_PER_HOUR", "60")))
    except (TypeError, ValueError):
        max_calls = 60
    now = time.time()
    state = _pagespeed_rate_state()
    with state["lock"]:
        state["timestamps"] = [stamp for stamp in state["timestamps"] if now - stamp < 3600]
        if len(state["timestamps"]) >= max_calls:
            return False
        state["timestamps"].append(now)
    return True


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

DATA_FILE = DATA_DIR / "marketing_kpi_data.csv"

MODEL_FILE = BASE_DIR / "models" / "revenue_model.pkl"

CHROMA_DIR = DATA_DIR / "chroma_db"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_numeric(df, columns):
    """
    Convert numeric columns safely.
    """

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def prepare_dataset(data):
    """
    Standardize and validate uploaded marketing data.

    Required:
        Date
        Campaign
        Channel
        Spend
        Revenue
        Impressions
        Conversions

    Optional:
        Clicks
        CTR
    """

    df = data.copy()

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Clean column names
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    required_columns = [
        "Date",
        "Campaign",
        "Channel",
        "Spend",
        "Revenue",
        "Impressions",
        "Conversions"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # Date
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Numeric columns
    numeric_columns = [
        "Spend",
        "Revenue",
        "Impressions",
        "Conversions",
        "Clicks",
        "CTR"
    ]

    df = clean_numeric(
        df,
        numeric_columns
    )

    # Remove invalid dates
    df = df.dropna(
        subset=["Date"]
    ).copy()

    # --------------------------------------------------------
    # CLICKS / CTR HANDLING
    # --------------------------------------------------------

    has_clicks = "Clicks" in df.columns

    has_ctr = "CTR" in df.columns

    if not has_clicks:

        if has_ctr:

            # If CTR is given as percentage,
            # calculate clicks from impressions.
            df["Clicks"] = (
                df["Impressions"]
                * df["CTR"]
                / 100
            )

            df["Clicks"] = (
                df["Clicks"]
                .round()
            )

            has_clicks = True

        else:

            # No clicks information available.
            df["Clicks"] = np.nan

    else:

        df["Clicks"] = df["Clicks"].where(
            df["Clicks"] >= 0
        )

    # --------------------------------------------------------
    # CTR
    # --------------------------------------------------------

    if has_clicks:

        df["CTR_Calc"] = np.where(
            (
                df["Impressions"].notna()
                & (df["Impressions"] > 0)
                & df["Clicks"].notna()
            ),
            (
                df["Clicks"]
                / df["Impressions"]
                * 100
            ),
            np.nan
        )

    else:

        df["CTR_Calc"] = np.nan

    # --------------------------------------------------------
    # CONVERSION RATE
    # --------------------------------------------------------

    df["Conversion_Rate_Calc"] = np.where(
        (
            df["Clicks"].notna()
            & (df["Clicks"] > 0)
        ),
        (
            df["Conversions"]
            / df["Clicks"]
            * 100
        ),
        np.nan
    )

    return df


def safe_divide(numerator, denominator):
    """
    Safe division.
    """

    return np.where(
        denominator > 0,
        numerator / denominator,
        np.nan
    )


def relative_position(value, series, higher_is_better=True):
    """
    Returns relative performance from 0 to 1.

    0 = weakest
    1 = strongest
    """

    clean_series = pd.Series(series).dropna()

    if clean_series.empty:
        return 0.5

    minimum = clean_series.min()
    maximum = clean_series.max()

    if maximum == minimum:
        return 0.5

    position = (
        value - minimum
    ) / (
        maximum - minimum
    )

    if not higher_is_better:
        position = 1 - position

    return float(
        np.clip(position, 0, 1)
    )


def calculate_health_scores(campaign_data):
    """
    Dataset-relative Health Score.

    No hard-coded ROAS values such as:
        165
        175
        180

    Instead, every campaign is compared against
    other campaigns in the currently active dataset.
    """

    result = campaign_data.copy()

    scores = []

    for _, row in result.iterrows():

        components = []
        weights = []

        # ----------------------------------------------------
        # ROAS
        # ----------------------------------------------------

        if pd.notna(row["ROAS"]):

            roas_score = (
                relative_position(
                    row["ROAS"],
                    result["ROAS"],
                    higher_is_better=True
                )
                * 100
            )

            components.append(roas_score)
            weights.append(0.40)

        # ----------------------------------------------------
        # CTR
        # ----------------------------------------------------

        if pd.notna(row["CTR"]):

            ctr_score = (
                relative_position(
                    row["CTR"],
                    result["CTR"],
                    higher_is_better=True
                )
                * 100
            )

            components.append(ctr_score)
            weights.append(0.20)

        # ----------------------------------------------------
        # CONVERSION RATE
        # ----------------------------------------------------

        if pd.notna(row["Conversion_Rate"]):

            conversion_score = (
                relative_position(
                    row["Conversion_Rate"],
                    result["Conversion_Rate"],
                    higher_is_better=True
                )
                * 100
            )

            components.append(conversion_score)
            weights.append(0.25)

        # ----------------------------------------------------
        # CPA
        # Lower CPA = Better
        # ----------------------------------------------------

        if pd.notna(row["CPA"]):

            cpa_score = (
                relative_position(
                    row["CPA"],
                    result["CPA"],
                    higher_is_better=False
                )
                * 100
            )

            components.append(cpa_score)
            weights.append(0.15)

        # ----------------------------------------------------
        # Weighted score
        # ----------------------------------------------------

        if components:

            total_weight = sum(weights)

            score = sum(
                value * weight
                for value, weight
                in zip(components, weights)
            ) / total_weight

        else:

            score = 50

        scores.append(
            round(
                float(
                    np.clip(score, 0, 100)
                ),
                1
            )
        )

    result["Health_Score"] = scores

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    def get_status(score):

        if score >= 75:
            return "Strong"

        elif score >= 50:
            return "Needs Optimization"

        else:
            return "Needs Attention"

    result["Health_Status"] = (
        result["Health_Score"]
        .apply(get_status)
    )

    return result

def generate_recommendations(
    row,
    campaign_data
):
    """
    Health Score aligned recommendation engine.

    Health Score:
        75-100  -> Strong
        50-74   -> Optimization Recommended
        25-49   -> Needs Attention
        0-24    -> Immediate Attention
    """

    actions = []

    # --------------------------------------------------------
    # HEALTH SCORE
    # --------------------------------------------------------

    health_score = float(
        row.get("Health_Score", 50)
    )

    # --------------------------------------------------------
    # DATASET BENCHMARKS
    # --------------------------------------------------------

    median_roas = campaign_data["ROAS"].median()

    median_cpa = campaign_data["CPA"].median()

    median_ctr = (
        campaign_data["CTR"].median()
        if campaign_data["CTR"].notna().any()
        else np.nan
    )

    median_conversion = (
        campaign_data["Conversion_Rate"].median()
        if campaign_data["Conversion_Rate"].notna().any()
        else np.nan
    )

    # ========================================================
    # 1. CRITICAL CAMPAIGN
    # Health Score < 25
    # ========================================================

    if health_score < 25:

        actions.append(
            "Reduce or pause inefficient spending and "
            "review campaign efficiency before scaling."
        )

        # CTR
        if (
            pd.notna(row["CTR"])
            and pd.notna(median_ctr)
            and row["CTR"] < median_ctr * 0.90
        ):

            actions.append(
                "Test new ad creatives, headlines "
                "and calls-to-action."
            )

        # Conversion Rate
        if (
            pd.notna(row["Conversion_Rate"])
            and pd.notna(median_conversion)
            and row["Conversion_Rate"]
            < median_conversion * 0.90
        ):

            actions.append(
                "Improve landing page experience, "
                "offer quality and conversion flow."
            )

        # CPA
        if (
            pd.notna(row["CPA"])
            and pd.notna(median_cpa)
            and row["CPA"] > median_cpa * 1.10
        ):

            actions.append(
                "Optimize audience targeting and "
                "acquisition efficiency."
            )

    # ========================================================
    # 2. NEEDS ATTENTION
    # Health Score 25-49
    # ========================================================

    elif health_score < 50:

        actions.append(
            "Optimize campaign performance before "
            "increasing the budget."
        )

        if (
            pd.notna(row["ROAS"])
            and pd.notna(median_roas)
            and row["ROAS"] < median_roas * 0.90
        ):

            actions.append(
                "Review spending efficiency and "
                "campaign profitability."
            )

        if (
            pd.notna(row["CTR"])
            and pd.notna(median_ctr)
            and row["CTR"] < median_ctr * 0.90
        ):

            actions.append(
                "Test new ad creatives, headlines "
                "and calls-to-action."
            )

        if (
            pd.notna(row["Conversion_Rate"])
            and pd.notna(median_conversion)
            and row["Conversion_Rate"]
            < median_conversion * 0.90
        ):

            actions.append(
                "Review landing page, offer quality "
                "and conversion flow."
            )

        if (
            pd.notna(row["CPA"])
            and pd.notna(median_cpa)
            and row["CPA"] > median_cpa * 1.10
        ):

            actions.append(
                "Optimize audience targeting and "
                "acquisition efficiency."
            )

    # ========================================================
    # 3. OPTIMIZATION RECOMMENDED
    # Health Score 50-74
    # ========================================================

    elif health_score < 75:

        actions.append(
            "Maintain the current budget and "
            "continue monitoring performance."
        )

        if (
            pd.notna(row["CTR"])
            and pd.notna(median_ctr)
            and row["CTR"] < median_ctr * 0.90
        ):

            actions.append(
                "Test new ad creatives and improve "
                "ad messaging."
            )

        if (
            pd.notna(row["Conversion_Rate"])
            and pd.notna(median_conversion)
            and row["Conversion_Rate"]
            < median_conversion * 0.90
        ):

            actions.append(
                "Improve landing page experience "
                "and conversion flow."
            )

        if (
            pd.notna(row["CPA"])
            and pd.notna(median_cpa)
            and row["CPA"] > median_cpa * 1.10
        ):

            actions.append(
                "Optimize audience targeting to "
                "reduce acquisition cost."
            )

    # ========================================================
    # 4. STRONG CAMPAIGN
    # Health Score >= 75
    # ========================================================

    else:

        actions.append(
            "Consider gradually scaling budget while "
            "monitoring ROAS and campaign efficiency."
        )

        if (
            pd.notna(row["CTR"])
            and pd.notna(median_ctr)
            and row["CTR"] < median_ctr * 0.90
        ):

            actions.append(
                "Continue testing creatives to "
                "maintain strong engagement."
            )

        if (
            pd.notna(row["Conversion_Rate"])
            and pd.notna(median_conversion)
            and row["Conversion_Rate"]
            < median_conversion * 0.90
        ):

            actions.append(
                "Monitor and improve conversion flow "
                "while scaling carefully."
            )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    actions = list(
        dict.fromkeys(actions)
    )

    return actions


def campaign_reasons(
    row,
    campaign_data
):
    """
    Dataset-relative Why analysis.
    """

    reasons = []

    median_roas = campaign_data["ROAS"].median()

    if pd.notna(row["ROAS"]):

        if row["ROAS"] < median_roas * 0.90:

            reasons.append(
                "ROAS is below the dataset benchmark, "
                "indicating weaker revenue efficiency."
            )

        elif row["ROAS"] > median_roas * 1.10:

            reasons.append(
                "ROAS is above the dataset benchmark, "
                "indicating stronger revenue efficiency."
            )

    median_ctr = (
        campaign_data["CTR"].median()
        if campaign_data["CTR"].notna().any()
        else np.nan
    )

    if (
        pd.notna(row["CTR"])
        and pd.notna(median_ctr)
    ):

        if row["CTR"] < median_ctr * 0.90:

            reasons.append(
                "CTR is below the dataset benchmark, "
                "which may indicate creative or messaging opportunities."
            )

        elif row["CTR"] > median_ctr * 1.10:

            reasons.append(
                "CTR is above the dataset benchmark, "
                "suggesting stronger ad engagement."
            )

    median_conversion = (
        campaign_data["Conversion_Rate"].median()
        if campaign_data["Conversion_Rate"].notna().any()
        else np.nan
    )

    if (
        pd.notna(row["Conversion_Rate"])
        and pd.notna(median_conversion)
    ):

        if (
            row["Conversion_Rate"]
            < median_conversion * 0.90
        ):

            reasons.append(
                "Conversion rate is below the dataset benchmark, "
                "so the landing page, offer or audience may need review."
            )

    median_cpa = campaign_data["CPA"].median()

    if (
        pd.notna(row["CPA"])
        and pd.notna(median_cpa)
        and row["CPA"] > median_cpa * 1.10
    ):

        reasons.append(
            "CPA is above the dataset benchmark, "
            "indicating higher acquisition cost."
        )

    if not reasons:

        reasons.append(
            "The campaign is performing within the "
            "normal range of the active dataset."
        )

    return reasons


def format_percentage(value):
    """
    Display percentage safely.
    """

    if pd.isna(value):
        return "N/A"

    return f"{value:.2f}%"


def format_currency(value):
    return f"₹{value:,.2f}"


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <section class="mm-hero">
      <div class="mm-brand-row">
        <div class="mm-logo" aria-label="MarketMind AI logo">
          <svg viewBox="0 0 48 48" role="img" aria-hidden="true">
            <defs><linearGradient id="mmg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#8ef5de"/><stop offset="1" stop-color="#9bb5ff"/></linearGradient></defs>
            <path d="M7 37.5h34" stroke="rgba(255,255,255,.55)" stroke-width="2.5" stroke-linecap="round"/>
            <rect x="9" y="24" width="7" height="13.5" rx="2.2" fill="url(#mmg)"/>
            <rect x="21" y="17" width="7" height="20.5" rx="2.2" fill="url(#mmg)"/>
            <rect x="33" y="10" width="7" height="27.5" rx="2.2" fill="url(#mmg)"/>
            <path d="m10 19 10-6 8 2 10-8" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M33 7h5v5" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <div>
          <div class="mm-kicker">MARKETING INTELLIGENCE PLATFORM</div>
          <h1 class="mm-brand">MarketMind <span>AI</span></h1>
        </div>
        <div class="mm-pill">✦ PERFORMANCE · SEO · INSIGHTS</div>
      </div>
      <div class="mm-tagline">Turn marketing data into your next best move.</div>
      <p class="mm-description">Understand campaign performance, uncover what is changing, and find clear next steps for growth.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

# Account-based sign in. Google creates the account on a user's first sign-in.
auth_settings = get_app_secret("auth", {})
auth_configured = hasattr(auth_settings, "get") and all(
    auth_settings.get(key)
    for key in ("redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url")
)
if not auth_configured:
    st.error("Google sign-in is not configured. Add the [auth] settings from `.streamlit/secrets.toml.example` to your private Streamlit secrets.")
    st.info("For local use, configure your Google OAuth Client ID, Client Secret, and redirect URL. Before deployment, update the redirect URL to your hosted app address.")
    st.stop()

if not st.user.is_logged_in:
    with st.container(key="mm_auth_stage"):
        with st.container(key="mm_auth_card"):
            st.markdown(
                "<div class='mm-auth-title'>Welcome to MarketMind AI</div>"
                "<p class='mm-auth-copy'>Sign in to analyze your marketing data and get personalized recommendations.</p>",
                unsafe_allow_html=True,
            )
            st.button(
                "Continue with Google",
                key="mm_google_login",
                on_click=st.login,
                type="secondary",
                use_container_width=True,
            )
            st.markdown(
                "<p style='color:#687792;font-size:.84rem;margin:.75rem 0 0;'>"
                "Your Google account creates your MarketMind account on first sign-in.</p>",
                unsafe_allow_html=True,
            )
    st.stop()

with st.sidebar:
    st.caption(f"Signed in as {st.user.get('email', st.user.get('name', 'Google user'))}")
    st.button("Sign out", on_click=st.logout, use_container_width=True)


# ============================================================
# UPLOAD DATA
# ============================================================

st.markdown("---")

st.subheader(
    "📥 Upload Your Own Marketing Data"
)

uploaded_file = st.file_uploader(
    "Upload your marketing dataset",
    type=["csv", "xlsx"]
)
st.caption("Privacy: Uploaded CSV files are processed during your session and are not saved to disk by this app. The free chatbot processes dashboard questions locally. Website URLs submitted for PageSpeed checks are sent to Google.")


# ============================================================
# LOAD DEFAULT DATA
# ============================================================

try:

    default_df = pd.read_csv(
        DATA_FILE
    )

    default_df = prepare_dataset(
        default_df
    )

except Exception as e:

    st.error(
        "❌ Default marketing dataset could not be loaded."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# ACTIVE DATASET
# ============================================================

if uploaded_file is not None:

    try:

        if uploaded_file.name.lower().endswith(".csv"):

            active_df = pd.read_csv(
                uploaded_file
            )

        else:

            active_df = pd.read_excel(
                uploaded_file
            )

        active_df = prepare_dataset(
            active_df
        )

        st.success(
            "✅ Marketing data uploaded successfully! "
            "All analysis below is now based on the uploaded dataset."
        )

        st.caption(
            f"📌 Active Dataset: {uploaded_file.name} "
            f"| {len(active_df)} rows"
        )

    except Exception as e:

        st.error(
            f"❌ Unable to process uploaded file: {e}"
        )

        st.stop()

else:

    active_df = default_df

    st.info(
        "Using the default MarketMind AI marketing dataset."
    )


# ============================================================
# UPLOADED DATA PREVIEW
# ============================================================

if uploaded_file is not None:

    st.subheader(
        "📊 Uploaded Data Preview"
    )

    st.dataframe(
        active_df.head(10),
        use_container_width=True
    )

    st.subheader(
        "📋 Dataset Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            active_df.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            active_df.shape[1]
        )

    with col3:

        st.metric(
            "Missing Values",
            int(
                active_df.isnull()
                .sum()
                .sum()
            )
        )
# ============================================================
# DATE FILTER
# ============================================================

st.markdown("---")
st.subheader("📅 Date Filter")

min_date = active_df["Date"].min().date()
max_date = active_df["Date"].max().date()
calendar_max_date = max(max_date, pd.Timestamp.today().date())

# First time app open aagumbodhu full date range select aagum
if "mm_date_range" not in st.session_state:
    st.session_state.mm_date_range = (min_date, max_date)
else:
    previous_start, previous_end = st.session_state.mm_date_range

    # Pudhu dataset upload pannina, date range-ai dataset bounds-kulla vechukkum
    if previous_end < min_date or previous_start > calendar_max_date:
        st.session_state.mm_date_range = (min_date, max_date)
    elif previous_start < min_date or previous_end > calendar_max_date:
        st.session_state.mm_date_range = (
            max(previous_start, min_date),
            min(previous_end, calendar_max_date),
        )

# Quick date range buttons
quick_cols = st.columns(4)
quick_ranges = [
    ("Full range", (min_date, max_date)),
    ("Last 7 days", (max(min_date, max_date - pd.Timedelta(days=6)), max_date)),
    ("Last 30 days", (max(min_date, max_date - pd.Timedelta(days=29)), max_date)),
    ("Last 90 days", (max(min_date, max_date - pd.Timedelta(days=89)), max_date)),
]

for col, (label, selected_range) in zip(quick_cols, quick_ranges):
    if col.button(label, use_container_width=True, key=f"mm_range_{label}"):
        st.session_state.mm_date_range = selected_range

st.caption(
    f"Data available: {min_date.strftime('%d %b %Y')} – "
    f"{max_date.strftime('%d %b %Y')} · "
    f"Date picker available through {calendar_max_date.strftime('%d %b %Y')}"
)

selected_range = st.date_input(
    "Choose start and end dates",
    value=st.session_state.mm_date_range,
    min_value=min_date,
    max_value=calendar_max_date,
    key="mm_date_range",
    format="DD/MM/YYYY",
)

if not isinstance(selected_range, tuple) or len(selected_range) != 2:
    st.info("Calendar-la start date-um end date-um select pannunga.")
    st.stop()

start_date, end_date = selected_range

if start_date > end_date:
    st.error("⚠️ Start Date cannot be after End Date.")
    filtered_df = active_df.copy()
else:
    filtered_df = active_df[
        (active_df["Date"] >= pd.Timestamp(start_date))
        & (active_df["Date"] <= pd.Timestamp(end_date))
    ].copy()

st.caption(
    f"Showing data from {start_date.strftime('%d %b %Y')} "
    f"to {end_date.strftime('%d %b %Y')}"
)

if filtered_df.empty:
    st.warning("No data available for the selected date range.")
    st.stop()

# ============================================================
# MARKETING KPIs
# ============================================================

total_spend = filtered_df["Spend"].sum()

total_revenue = filtered_df["Revenue"].sum()

total_conversions = filtered_df["Conversions"].sum()

total_impressions = filtered_df["Impressions"].sum()

total_clicks = (
    filtered_df["Clicks"].sum()
    if filtered_df["Clicks"].notna().any()
    else np.nan
)


# ============================================================
# OVERALL CTR
# ============================================================

if (
    pd.notna(total_clicks)
    and total_impressions > 0
):

    overall_ctr = (
        total_clicks
        / total_impressions
        * 100
    )

else:

    overall_ctr = np.nan


# ============================================================
# OVERALL ROAS
# ============================================================

if total_spend > 0:

    overall_roas = (
        total_revenue
        / total_spend
    )

else:

    overall_roas = np.nan


# ============================================================
# KPI DISPLAY
# ============================================================

daily_revenue = (
    filtered_df
    .groupby("Date", as_index=False)
    ["Revenue"]
    .sum()
)

st.markdown(
    "## 📌 Marketing KPIs"
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "💰 Total Spend",
        format_currency(total_spend)
    )

with col2:

    st.metric(
        "💵 Revenue",
        format_currency(total_revenue)
    )

with col3:

    st.metric(
        "🎯 Conversions",
        f"{total_conversions:,.0f}"
    )

with col4:

    st.metric(
        "📈 CTR",
        format_percentage(overall_ctr)
    )

with col5:

    st.metric(
        "🚀 ROAS",
        f"{overall_roas:.2f}"
        if pd.notna(overall_roas)
        else "N/A"
    )


if pd.isna(overall_ctr):

    st.caption(
        "ℹ️ CTR is unavailable because Clicks data "
        "was not provided in the active dataset."
    )


# ============================================================
# DAILY REVENUE
# ============================================================

st.markdown("---")

st.subheader(
    "📈 Daily Revenue Trend"
)

fig_revenue = px.line(
    daily_revenue,
    x="Date",
    y="Revenue",
    title="Daily Revenue"
)

fig_revenue.update_layout(
    hovermode="x unified"
)

st.plotly_chart(
    fig_revenue,
    use_container_width=True
)


# ============================================================
# CAMPAIGN PERFORMANCE
# ============================================================

st.markdown("---")

st.subheader(
    "🎯 Campaign Performance"
)

campaign_data = (
    filtered_df
    .groupby("Campaign")
    .agg(
        Spend=("Spend", "sum"),
        Revenue=("Revenue", "sum"),
        Impressions=("Impressions", "sum"),
        Clicks=("Clicks", "sum"),
        Conversions=("Conversions", "sum")
    )
    .reset_index()
)


# ============================================================
# CAMPAIGN KPI CALCULATIONS
# ============================================================

campaign_data["ROAS"] = safe_divide(
    campaign_data["Revenue"],
    campaign_data["Spend"]
)


campaign_data["CTR"] = safe_divide(
    campaign_data["Clicks"],
    campaign_data["Impressions"]
) * 100


campaign_data["Conversion_Rate"] = safe_divide(
    campaign_data["Conversions"],
    campaign_data["Clicks"]
) * 100


campaign_data["CPA"] = safe_divide(
    campaign_data["Spend"],
    campaign_data["Conversions"]
)


# ============================================================
# CHARTS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    fig_campaign_revenue = px.bar(
        campaign_data,
        x="Campaign",
        y="Revenue",
        title="Revenue by Campaign"
    )

    st.plotly_chart(
        fig_campaign_revenue,
        use_container_width=True
    )

with col2:

    fig_campaign_roas = px.bar(
        campaign_data,
        x="Campaign",
        y="ROAS",
        title="ROAS by Campaign"
    )

    st.plotly_chart(
        fig_campaign_roas,
        use_container_width=True
    )


# ============================================================
# CHANNEL PERFORMANCE
# ============================================================

st.markdown("---")

st.subheader(
    "📣 Channel Performance"
)

channel_data = (
    filtered_df
    .groupby("Channel")
    .agg(
        Spend=("Spend", "sum"),
        Revenue=("Revenue", "sum"),
        Impressions=("Impressions", "sum"),
        Clicks=("Clicks", "sum"),
        Conversions=("Conversions", "sum")
    )
    .reset_index()
)

channel_data["ROAS"] = safe_divide(
    channel_data["Revenue"],
    channel_data["Spend"]
)

channel_data["CTR"] = safe_divide(
    channel_data["Clicks"],
    channel_data["Impressions"]
) * 100

channel_data["Conversion_Rate"] = safe_divide(
    channel_data["Conversions"],
    channel_data["Clicks"]
) * 100


col1, col2 = st.columns(2)

with col1:

    fig_channel_revenue = px.bar(
        channel_data,
        x="Channel",
        y="Revenue",
        title="Revenue by Channel"
    )

    st.plotly_chart(
        fig_channel_revenue,
        use_container_width=True
    )

with col2:

    fig_channel_roas = px.bar(
        channel_data,
        x="Channel",
        y="ROAS",
        title="ROAS by Channel"
    )

    st.plotly_chart(
        fig_channel_roas,
        use_container_width=True
    )


# ============================================================
# BRAND AWARENESS PERFORMANCE
# ============================================================

st.markdown("---")
st.subheader("📣 Brand Awareness Performance")
st.caption("Impressions show ad exposure; they do not represent unique people reached.")
awareness_campaigns = campaign_data.sort_values("Impressions", ascending=False)
awareness_channels = channel_data.sort_values("Impressions", ascending=False)
awareness_col1, awareness_col2 = st.columns(2)
with awareness_col1:
    awareness_campaign_chart = px.bar(
        awareness_campaigns,
        x="Campaign",
        y="Impressions",
        title="Ad Impressions by Campaign",
        color="Impressions",
        color_continuous_scale="Teal",
    )
    st.plotly_chart(awareness_campaign_chart, use_container_width=True)
with awareness_col2:
    awareness_channel_chart = px.bar(
        awareness_channels,
        x="Channel",
        y="Impressions",
        title="Ad Impressions by Channel",
        color="Impressions",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(awareness_channel_chart, use_container_width=True)


# ============================================================
# HEALTH SCORE
# ============================================================

campaign_data = calculate_health_scores(
    campaign_data
)


# ============================================================
# CAMPAIGN HEALTH SCORE
# ============================================================

st.markdown("---")

st.subheader(
    "❤️ Campaign Health Score"
)

st.caption(
    "Health Score is calculated relative to the "
    "currently active dataset rather than fixed industry values."
)

st.dataframe(
    campaign_data[
        [
            "Campaign",
            "Health_Score",
            "Health_Status"
        ]
    ],
    use_container_width=True
)


# ============================================================
# BEST / WORST CAMPAIGNS
# ============================================================

best_campaign = campaign_data.loc[
    campaign_data["ROAS"].idxmax()
]

worst_campaign = campaign_data.loc[
    campaign_data["ROAS"].idxmin()
]

best_campaign_name = (
    best_campaign["Campaign"]
)

worst_campaign_name = (
    worst_campaign["Campaign"]
)

average_health_score = (
    campaign_data["Health_Score"].mean()
)


# ============================================================
# CAMPAIGN DEEP DIVE
# ============================================================

st.markdown("---")

st.header(
    "🔎 Campaign Deep Dive"
)

selected_campaign = st.selectbox(
    "Select a campaign",
    campaign_data["Campaign"].tolist()
)

selected_data = campaign_data[
    campaign_data["Campaign"]
    == selected_campaign
].iloc[0]


st.subheader(
    f"📊 {selected_campaign} Performance"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "ROAS",
        f"{selected_data['ROAS']:.2f}"
    )

with col2:

    st.metric(
        "CTR",
        format_percentage(
            selected_data["CTR"]
        )
    )

with col3:

    st.metric(
        "Conversion Rate",
        format_percentage(
            selected_data[
                "Conversion_Rate"
            ]
        )
    )

with col4:

    st.metric(
        "CPA",
        format_currency(
            selected_data["CPA"]
        )
        if pd.notna(
            selected_data["CPA"]
        )
        else "N/A"
    )


# ============================================================
# WHY DID THIS HAPPEN?
# ============================================================

st.subheader(
    "🔍 Campaign Analysis - Why?"
)

reasons = campaign_reasons(
    selected_data,
    campaign_data
)

for reason in reasons:

    st.info(
        "• " + reason
    )


# ============================================================
# WHAT'S NEXT?
# ============================================================

st.subheader(
    "🔮 Campaign Outlook"
)

selected_health = (
    selected_data["Health_Score"]
)


if selected_health >= 75:

    st.success(
        "The campaign is performing strongly "
        "relative to the current dataset. "
        "Consider controlled scaling while monitoring "
        "ROAS and other available KPIs."
    )

elif selected_health >= 50:

    st.warning(
        "The campaign has optimization opportunities. "
        "Improve weaker KPIs before significantly "
        "increasing the budget."
    )

else:

    st.error(
        "The campaign is performing below the "
        "current dataset benchmark. Review efficiency "
        "before scaling."
    )


# ============================================================
# WHAT SHOULD I DO?
# ============================================================

st.subheader(
    "🎯 Recommended Action"
)

actions = generate_recommendations(
    selected_data,
    campaign_data
)

for index, action in enumerate(
    actions,
    start=1
):

    st.write(
        f"**{index}.** {action}"
    )


# ============================================================
# CAMPAIGN KPI TABLE
# ============================================================

st.markdown("---")

st.subheader(
    "📋 Campaign KPI Table"
)

display_campaign_data = campaign_data.copy()

st.dataframe(
    display_campaign_data.round(2),
    use_container_width=True
)


# ============================================================
# CAMPAIGN RECOMMENDATIONS
# ============================================================

st.markdown("---")

st.header(
    "🎯 Campaign Recommendations"
)

for _, row in campaign_data.iterrows():

    campaign = row["Campaign"]

    health_score = row["Health_Score"]

    health_status = row["Health_Status"]

    campaign_actions = generate_recommendations(
        row,
        campaign_data
    )

    final_actions = " ".join(
        campaign_actions
    )

    if health_status == "Strong":

        st.success(
            f"""
🟢 **{campaign} — Strong Campaign**

Health Score: **{health_score:.1f}/100**

{final_actions}
"""
        )

    elif health_status == "Needs Optimization":

        st.warning(
            f"""
🟡 **{campaign} — Optimization Recommended**

Health Score: **{health_score:.1f}/100**

{final_actions}
"""
        )

    else:

        st.error(
            f"""
🔴 **{campaign} — Needs Attention**

Health Score: **{health_score:.1f}/100**

{final_actions}
"""
        )


# ============================================================
# MARKETMIND INSIGHT
# ============================================================

st.markdown("---")

st.subheader(
    "🤖 MarketMind Insight"
)

st.success(
    f"""
🏆 **Best Campaign:** {best_campaign_name}

ROAS: **{best_campaign["ROAS"]:.2f}**

⚠️ **Lowest ROAS Campaign:** {worst_campaign_name}

ROAS: **{worst_campaign["ROAS"]:.2f}**
"""
)


# ============================================================
# ON-PAGE & TECHNICAL SEO
# ============================================================

st.markdown("---")
st.header("🔎 On-page & Technical SEO Audit")
st.write(
    "Enter a public page URL to check its on-page SEO and technical setup, then review "
    "the prioritized fixes."
)

with st.form("seo_audit_form"):
    own_url = st.text_input("Your website page URL", placeholder="https://example.com")
    run_seo_audit = st.form_submit_button("🔎 Run SEO audit", type="primary")

if run_seo_audit:
    if not own_url.strip():
        st.error("Enter your website page URL to start the audit.")
    else:
        try:
            with st.spinner("Checking on-page and technical SEO…"):
                main_audit = audit_page(own_url)
            st.session_state["latest_seo_audit"] = {
                "url": main_audit["url"],
                "checks": main_audit["checks"],
                "title": main_audit["title"],
                "description": main_audit["description"],
                "response_seconds": round(main_audit["response_seconds"], 2),
                "page_kb": round(main_audit["page_kb"], 1),
                "images": main_audit["images"],
                "images_with_alt": main_audit["images_with_alt"],
                "internal_links": main_audit["internal_links"],
            }
            main_score = round(100 * sum(main_audit["checks"].values()) / len(main_audit["checks"]))
            st.subheader("Audit results")
            metric_cols = st.columns(4)
            metric_cols[0].metric("On-page & technical score", f"{main_score}/100")
            metric_cols[1].metric("HTTP status", main_audit["status"])
            metric_cols[2].metric("Response time", f"{main_audit['response_seconds']:.2f}s")
            metric_cols[3].metric("Internal links", main_audit["internal_links"])

            technical_check_names = {
                "HTTPS enabled", "Mobile viewport configured", "Canonical URL present",
                "Robots.txt reachable", "Sitemap.xml reachable", "Response under 2 seconds",
                "HTML page under 500 KB",
            }
            on_page_checks, technical_checks = st.tabs(["On-page SEO", "Technical SEO"])
            for tab, selected_checks in (
                (on_page_checks, {k: v for k, v in main_audit["checks"].items() if k not in technical_check_names}),
                (technical_checks, {k: v for k, v in main_audit["checks"].items() if k in technical_check_names}),
            ):
                with tab:
                    checks_df = pd.DataFrame(
                        [{"Check": name, "Result": "Pass" if passed else "Needs attention"}
                         for name, passed in selected_checks.items()]
                    )
                    st.dataframe(checks_df, use_container_width=True, hide_index=True)
            st.caption(
                f"Title: {main_audit['title'] or 'Missing'} · "
                f"Meta description: {main_audit['description'] or 'Missing'} · "
                f"Images with alt text: {main_audit['images_with_alt']}/{main_audit['images']} · "
                f"External links: {main_audit['external_links']} · "
                f"HTML size: {main_audit['page_kb']:.1f} KB"
            )

            st.subheader("Prioritized action plan")
            for item in recommendations(main_audit):
                st.markdown(f"**{item['priority']} · {item['issue']}** — {item['action']}")
        except Exception as exc:
            st.error(f"Could not audit this page: {exc}")

st.markdown("### 📱 Mobile and 💻 Laptop/desktop PageSpeed")
st.caption("Run each Google PageSpeed Insights check separately. Google may limit anonymous requests; adding your own API key gives your project its own quota.")
pagespeed_url = st.text_input("Website URL for speed test", placeholder="https://example.com", key="pagespeed_url")
pagespeed_key = get_app_secret("PAGESPEED_API_KEY")
if pagespeed_key:
    st.caption("PageSpeed API key is configured privately in the app's server secrets.")
else:
    pagespeed_key = st.text_input(
        "Google PageSpeed API key (optional)",
        type="password",
        help="For a public deployment, configure PAGESPEED_API_KEY in server secrets instead of asking every visitor to enter a key.",
        key="pagespeed_api_key",
    )

mobile_col, desktop_col = st.columns(2)
with mobile_col:
    check_mobile = st.button("📱 Check mobile speed", use_container_width=True)
with desktop_col:
    check_desktop = st.button("💻 Check laptop/desktop speed", use_container_width=True)

if check_mobile or check_desktop:
    selected_strategy = "mobile" if check_mobile else "desktop"
    if not pagespeed_url.strip():
        st.error("Enter your public website URL before running the speed check.")
    else:
        try:
            if not allow_pagespeed_request():
                raise RuntimeError("This app has reached its PageSpeed request limit for the hour. Please try again later.")
            with st.spinner(f"Google is checking the {selected_strategy} version…"):
                result = run_pagespeed(pagespeed_url.strip(), selected_strategy, pagespeed_key.strip())
            st.session_state[f"pagespeed_{selected_strategy}_result"] = result
            st.session_state[f"pagespeed_{selected_strategy}_url"] = pagespeed_url.strip()
            st.session_state.pop(f"pagespeed_{selected_strategy}_error", None)
        except Exception as exc:
            st.session_state[f"pagespeed_{selected_strategy}_error"] = str(exc)

def show_pagespeed_result(strategy):
    result = st.session_state.get(f"pagespeed_{strategy}_result")
    if result and st.session_state.get(f"pagespeed_{strategy}_url") == st.session_state.get("pagespeed_url", ""):
        st.markdown(f"#### {'📱 Mobile' if strategy == 'mobile' else '💻 Laptop/desktop'} results")
        cols = st.columns(4)
        for col, (key, label) in zip(cols, [
            ("performance", "Speed"),
            ("accessibility", "Accessibility"),
            ("best-practices", "Best practices"),
            ("seo", "SEO"),
        ]):
            value = result["scores"].get(key)
            col.metric(label, f"{value}/100" if value is not None else "N/A")

        if strategy == "mobile" and result["mobile_checks"]:
            st.markdown("**Mobile friendliness**")
            st.dataframe(pd.DataFrame(result["mobile_checks"]), use_container_width=True, hide_index=True)
        if result["metrics"]:
            st.markdown("**Loading metrics**")
            metric_cols = st.columns(len(result["metrics"]))
            names = {"largest-contentful-paint": "LCP", "cumulative-layout-shift": "CLS", "interaction-to-next-paint": "INP"}
            for col, (key, value) in zip(metric_cols, result["metrics"].items()):
                col.metric(names.get(key, key), value)
        if result["opportunities"]:
            st.markdown("**Recommended speed improvements**")
            for item in result["opportunities"]:
                detail = f" — {item['display_value']}" if item["display_value"] else ""
                st.markdown(f"- **{item['title']}**{detail}")
        else:
            st.success("No major Lighthouse issues were reported in this run.")

    error = st.session_state.get(f"pagespeed_{strategy}_error")
    if error:
        st.warning(f"{strategy.title()} check couldn't run: {error}")

show_pagespeed_result("mobile")
show_pagespeed_result("desktop")


# ============================================================
# AI REVENUE FORECAST
# ============================================================

st.markdown("---")

st.header(
    "🔮 AI Revenue Forecast"
)

st.write(
    "MarketMind AI uses a machine learning model "
    "to estimate future revenue based on recent "
    "marketing performance."
)

prediction = None

try:

    if MODEL_FILE.exists():

        model = joblib.load(
            MODEL_FILE
        )

        latest_data = (
            filtered_df
            .sort_values("Date")
            .iloc[-1]
        )

        latest_impressions = (
            latest_data["Impressions"]
        )

        latest_clicks = (
            latest_data["Clicks"]
        )

        latest_conversions = (
            latest_data["Conversions"]
        )

        latest_spend = (
            latest_data["Spend"]
        )

        latest_revenue = (
            latest_data["Revenue"]
        )

        if (
            pd.notna(latest_clicks)
            and latest_impressions > 0
        ):

            latest_ctr = (
                latest_clicks
                / latest_impressions
                * 100
            )

        else:

            latest_ctr = 0

        if (
            pd.notna(latest_clicks)
            and latest_clicks > 0
        ):

            latest_conversion_rate = (
                latest_conversions
                / latest_clicks
                * 100
            )

        else:

            latest_conversion_rate = 0

        latest_roas = (
            latest_revenue
            / latest_spend
            if latest_spend > 0
            else 0
        )

        prediction_input = pd.DataFrame(
            {
                "Impressions": [
                    latest_impressions
                ],

                "Clicks": [
                    latest_clicks
                    if pd.notna(latest_clicks)
                    else 0
                ],

                "Conversions": [
                    latest_conversions
                ],

                "Spend": [
                    latest_spend
                ],

                "CTR": [
                    latest_ctr
                ],

                "Conversion_Rate": [
                    latest_conversion_rate
                ],

                "ROAS": [
                    latest_roas
                ],

                "Day": [
                    latest_data["Date"].day
                ],

                "Month": [
                    latest_data["Date"].month
                ],

                "DayOfWeek": [
                    latest_data["Date"].dayofweek
                ]
            }
        )

        prediction = model.predict(
            prediction_input
        )[0]

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🔮 Predicted Next Revenue",
                f"₹{prediction:,.2f}"
            )

        with col2:

            st.metric(
                "📅 Based on Latest Data",
                str(
                    latest_data["Date"].date()
                )
            )

    else:

        st.warning(
            "Revenue prediction model not found. "
            "Run revenue_prediction.py first."
        )

except Exception as e:

    st.warning(
        "Revenue prediction could not be generated "
        "for this uploaded dataset."
    )

    st.caption(
        f"Details: {e}"
    )


# ============================================================
# WHAT-IF SCENARIO
# ============================================================

st.markdown("---")

st.header(
    "🎯 What-If Scenario Simulator"
)

st.write(
    "Change the marketing budget and see the "
    "estimated revenue impact."
)

current_roas = (
    total_revenue / total_spend
    if total_spend > 0
    else 0
)

budget_change = st.slider(
    "Marketing Budget Change (%)",
    min_value=-50,
    max_value=100,
    value=20,
    step=5
)

scenario_spend = (
    total_spend
    * (
        1
        + budget_change / 100
    )
)

scenario_revenue = (
    scenario_spend
    * current_roas
)

revenue_change = (
    scenario_revenue
    - total_revenue
)

revenue_change_percentage = (
    revenue_change
    / total_revenue
    * 100
    if total_revenue > 0
    else 0
)


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "💰 Scenario Spend",
        f"₹{scenario_spend:,.2f}"
    )

with col2:

    st.metric(
        "📈 Estimated Revenue",
        f"₹{scenario_revenue:,.2f}"
    )

with col3:

    st.metric(
        "📊 Revenue Change",
        f"{revenue_change_percentage:+.2f}%"
    )


if budget_change > 0:

    st.success(
        f"""
📈 Increasing the budget by {budget_change}%
could potentially change revenue by
{revenue_change_percentage:.2f}%
if current ROAS remains stable.
"""
    )

elif budget_change < 0:

    st.warning(
        f"""
📉 Reducing the budget by {abs(budget_change)}%
could potentially change revenue by
{revenue_change_percentage:.2f}%
if current ROAS remains stable.
"""
    )

else:

    st.info(
        "ℹ️ No budget change. Current performance "
        "is being used as the baseline."
    )


# ============================================================
# RAG-POWERED PERFORMANCE ANALYSIS
# ============================================================

st.markdown("---")

st.header(
    "🔍 RAG-Powered Performance Analysis"
)

st.write(
    "MarketMind AI retrieves relevant marketing knowledge "
    "from the RAG knowledge base and combines it with "
    "the active campaign KPIs."
)


rag_reasons = []
st.session_state.pop("latest_rag_context", None)


if RAG_AVAILABLE and CHROMA_DIR.exists():

    try:

        @st.cache_resource
        def load_embedding_model():

            return SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

        embedding_model = (
            load_embedding_model()
        )

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        collection = client.get_collection(
            name="marketing_knowledge"
        )

        query = f"""
        Explain marketing performance.

        Best Campaign:
        {best_campaign_name}

        Best Campaign ROAS:
        {best_campaign["ROAS"]:.2f}

        Lowest Campaign:
        {worst_campaign_name}

        Lowest Campaign ROAS:
        {worst_campaign["ROAS"]:.2f}

        Overall ROAS:
        {overall_roas:.2f}

        Health Score:
        {average_health_score:.1f}

        Explain using:
        ROAS, CTR, Conversion Rate, CPA,
        audience targeting, creative quality,
        landing page, offer quality and campaign efficiency.
        """

        query_embedding = (
            embedding_model
            .encode(query)
            .tolist()
        )

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=5
        )

        documents = results.get(
            "documents",
            [[]]
        )

        if (
            documents
            and len(documents[0]) > 0
        ):

            knowledge_context = (
                "\n\n".join(
                    documents[0]
                )
            )
            st.session_state["latest_rag_context"] = knowledge_context[:10000]

            st.success(
                "🧠 RAG Knowledge Retrieved Successfully"
            )

            st.write(
                f"Retrieved "
                f"{len(documents[0])} relevant "
                f"marketing knowledge documents."
            )

            with st.expander(
                "📚 View Retrieved Marketing Knowledge"
            ):

                st.write(
                    knowledge_context
                )

            st.subheader(
                "💡 Performance Explanation"
            )

            st.info(
                f"""
🏆 **{best_campaign_name}** is currently the strongest
campaign based on ROAS.

Its ROAS is **{best_campaign["ROAS"]:.2f}**,
while **{worst_campaign_name}** has a ROAS of
**{worst_campaign["ROAS"]:.2f}**.

The performance difference can be related to
campaign efficiency, audience relevance,
creative effectiveness and conversion performance.

The RAG knowledge base was used to retrieve
marketing concepts relevant to the active KPIs.
"""
            )

            st.subheader(
                "🔎 Possible Reasons"
            )

            # CTR reason
            if (
                pd.notna(best_campaign["CTR"])
                and pd.notna(worst_campaign["CTR"])
            ):

                if (
                    best_campaign["CTR"]
                    > worst_campaign["CTR"]
                ):

                    rag_reasons.append(
                        f"{best_campaign_name} has a higher CTR "
                        "than the lower-ROAS campaign, which may "
                        "indicate stronger ad engagement."
                    )

                else:

                    rag_reasons.append(
                        "CTR does not explain the complete ROAS "
                        "difference, so other efficiency factors "
                        "should also be reviewed."
                    )

            else:

                rag_reasons.append(
                    "CTR comparison is unavailable because "
                    "Clicks data is not available in the active dataset."
                )

            # Conversion reason
            if (
                pd.notna(
                    best_campaign[
                        "Conversion_Rate"
                    ]
                )
                and pd.notna(
                    worst_campaign[
                        "Conversion_Rate"
                    ]
                )
            ):

                if (
                    best_campaign[
                        "Conversion_Rate"
                    ]
                    >
                    worst_campaign[
                        "Conversion_Rate"
                    ]
                ):

                    rag_reasons.append(
                        f"{best_campaign_name} has a higher "
                        "conversion rate, which may indicate "
                        "stronger audience or offer relevance."
                    )

                else:

                    rag_reasons.append(
                        "Conversion rate does not fully explain "
                        "the ROAS difference."
                    )

            else:

                rag_reasons.append(
                    "Conversion-rate comparison is unavailable "
                    "because Clicks data is not available."
                )

            # CPA
            if (
                pd.notna(
                    best_campaign["CPA"]
                )
                and pd.notna(
                    worst_campaign["CPA"]
                )
            ):

                if (
                    best_campaign["CPA"]
                    < worst_campaign["CPA"]
                ):

                    rag_reasons.append(
                        f"{best_campaign_name} has a lower CPA, "
                        "indicating more efficient acquisition."
                    )

                else:

                    rag_reasons.append(
                        "CPA is not lower for the strongest "
                        "ROAS campaign, so revenue efficiency "
                        "may be influenced by other factors."
                    )

            for reason in rag_reasons:

                st.write(
                    "• " + reason
                )

            # ------------------------------------------------
            # RAG DECISION
            # ------------------------------------------------

            st.subheader(
                "🧠 MarketMind AI Decision"
            )

            if (
                best_campaign["Health_Score"]
                >= 75
            ):

                st.success(
                    f"""
**Scale Carefully:** {best_campaign_name}

The campaign is performing strongly relative
to the current dataset. Consider gradual
budget scaling while monitoring ROAS.
"""
                )

            elif (
                best_campaign["Health_Score"]
                >= 50
            ):

                st.warning(
                    f"""
**Optimize Before Scaling:** {best_campaign_name}

The campaign is relatively strong but has
optimization opportunities before significant scaling.
"""
                )

            else:

                st.error(
                    f"""
**Optimize:** {best_campaign_name}

The campaign is currently below the dataset
benchmark. Review performance before scaling.
"""
                )

            if (
                worst_campaign["Health_Score"]
                < 50
            ):

                st.error(
                    f"""
**Immediate Optimization:** {worst_campaign_name}

Review targeting, creatives, landing page,
audience relevance and acquisition efficiency.
"""
                )

            else:

                st.warning(
                    f"""
**Monitor:** {worst_campaign_name}

This campaign has the lowest ROAS in the
active dataset. Review its KPIs before scaling.
"""
                )

        else:

            st.warning(
                "No relevant marketing knowledge was found "
                "in the RAG knowledge base."
            )

    except Exception as e:

        st.warning(
            "RAG analysis could not be loaded. "
            "Core MarketMind analytics are still available."
        )

        st.caption(
            f"RAG details: {e}"
        )

else:

    st.info(
        "RAG knowledge base is not available. "
        "Core marketing analytics are still available."
    )


# ============================================================
# DASHBOARD CONTEXT + FLOATING CHATBOT
# ============================================================

latest_seo = st.session_state.get("latest_seo_audit")
latest_pagespeed = {}
for strategy in ("mobile", "desktop"):
    result = st.session_state.get(f"pagespeed_{strategy}_result")
    tested_url = st.session_state.get(f"pagespeed_{strategy}_url")
    if result and tested_url == st.session_state.get("pagespeed_url", ""):
        latest_pagespeed[strategy] = {
            "scores": result["scores"],
            "metrics": result["metrics"],
            "mobile_checks": result["mobile_checks"],
            "opportunities": result["opportunities"][:5],
        }

def dataframe_records_for_ai(frame):
    """Convert mixed-column frames to JSON-safe records and round numeric fields only."""
    clean = frame.copy()
    numeric_columns = clean.select_dtypes(include=[np.number]).columns
    clean[numeric_columns] = clean[numeric_columns].round(2)
    clean = clean.astype(object).where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


def series_dict_for_ai(series):
    """Convert one mixed campaign row without applying numeric methods to text."""
    output = {}
    for key, value in series.items():
        if pd.isna(value):
            output[str(key)] = None
        elif pd.api.types.is_number(value):
            output[str(key)] = round(float(value), 2)
        else:
            output[str(key)] = str(value)
    return output


daily_trend = daily_revenue.tail(90).copy()
daily_trend["Date"] = daily_trend["Date"].dt.strftime("%Y-%m-%d")
daily_trend_records = dataframe_records_for_ai(daily_trend)
ai_context = {
    "active_date_range": {
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
    },
    "filtered_source_rows": int(len(filtered_df)),
    "overall_kpis": {
        "spend": round(float(total_spend), 2),
        "revenue": round(float(total_revenue), 2),
        "conversions": round(float(total_conversions), 2),
        "impressions": round(float(total_impressions), 2),
        "clicks": round(float(total_clicks), 2) if pd.notna(total_clicks) else None,
        "ctr_percent": round(float(overall_ctr), 2) if pd.notna(overall_ctr) else None,
        "roas": round(float(overall_roas), 2) if pd.notna(overall_roas) else None,
        "average_campaign_health_score": round(float(average_health_score), 1),
    },
    "campaigns": dataframe_records_for_ai(campaign_data),
    "channels": dataframe_records_for_ai(channel_data),
    "daily_revenue_trend_last_90_days": daily_trend_records,
    "selected_campaign": {
        "name": selected_campaign,
        "metrics": series_dict_for_ai(selected_data),
    },
    "best_campaign": best_campaign_name,
    "campaign_needing_attention": worst_campaign_name,
    "ml_revenue_forecast": round(float(prediction), 2) if prediction is not None else None,
    "budget_scenario": {
        "budget_change_percent": int(budget_change),
        "estimated_spend": round(float(scenario_spend), 2),
        "estimated_revenue": round(float(scenario_revenue), 2),
        "estimated_revenue_change_percent": round(float(revenue_change_percentage), 2),
        "assumption": "Current ROAS remains stable; this is a scenario estimate, not a guarantee.",
    },
    "latest_on_page_technical_seo_audit": latest_seo,
    "latest_pagespeed_mobile_desktop_results": latest_pagespeed,
    "retrieved_marketing_knowledge": st.session_state.get("latest_rag_context"),
}
ai_context_json = json.dumps(ai_context, ensure_ascii=False, default=str, separators=(",", ":"))
ai_context_id = hashlib.sha256(ai_context_json.encode("utf-8")).hexdigest()[:16]
if st.session_state.get("marketmind_chat_context") != ai_context_id:
    st.session_state["marketmind_chat_messages"] = []
    st.session_state["marketmind_chat_context"] = ai_context_id
chat_messages = st.session_state.setdefault("marketmind_chat_messages", [])

if st.session_state.get("marketmind_chat_open", False):
    with st.container(key="marketmind_chat_panel"):
        title_col, clear_col, close_col = st.columns([3.8, 2.5, 0.8])
        with title_col:
            st.markdown("### 💬 Ask MarketMind")
        with clear_col:
            if st.button("Clear chat", key="marketmind_clear_chat", use_container_width=True):
                st.session_state["marketmind_chat_messages"] = []
                st.rerun()
        with close_col:
            if st.button("×", key="marketmind_close_chat", use_container_width=True):
                st.session_state["marketmind_chat_open"] = False
                st.rerun()
        st.caption("Your free dashboard helper · Ask about campaigns, KPIs, SEO, or speed.")
        with st.container(key="marketmind_chat_history", height=350, border=False):
            if not chat_messages:
                st.markdown("**Hi! What would you like to understand?**")
                st.caption("Try: Which campaign has the highest ROAS? · What is my CTR? · How is my SEO?")
            for chat_message in chat_messages[-10:]:
                with st.chat_message(chat_message["role"]):
                    st.markdown(chat_message["content"])
        with st.form("marketmind_floating_chat_form", clear_on_submit=True):
            input_col, send_col = st.columns([5, 1])
            with input_col:
                user_question = st.text_input(
                    "Ask about your dashboard",
                    placeholder="Type your question…",
                    label_visibility="collapsed",
                )
            with send_col:
                send_question = st.form_submit_button("➤", use_container_width=True)
        if send_question and user_question.strip():
            chat_messages.append({"role": "user", "content": user_question.strip()})
            with st.spinner("Checking your dashboard…"):
                chat_answer = answer_dashboard_question(user_question, ai_context)
            chat_messages.append({"role": "assistant", "content": chat_answer})
            st.session_state["marketmind_chat_messages"] = chat_messages[-20:]
            st.rerun()
else:
    with st.container(key="marketmind_chat_launcher"):
        if st.button("🤖", key="marketmind_open_chat", help="Ask MarketMind", type="primary"):
            st.session_state["marketmind_chat_open"] = True
            st.rerun()



# ============================================================
# MARKETING REPORT
# ============================================================

st.markdown("---")

st.subheader(
    "📄 Generate Marketing Report"
)


report_campaigns = campaign_data.copy()


report_text = f"""
=================================================
                MARKETMIND AI
        MARKETING PERFORMANCE REPORT
=================================================

ACTIVE DATASET
--------------
Rows        : {len(filtered_df)}
Start Date  : {filtered_df["Date"].min().strftime("%d %b %Y")}
End Date    : {filtered_df["Date"].max().strftime("%d %b %Y")}

MARKETING KPIs
--------------

Total Spend       : ₹{total_spend:,.2f}
Total Revenue     : ₹{total_revenue:,.2f}
Conversions       : {total_conversions:,.0f}
CTR               : {format_percentage(overall_ctr)}
ROAS              : {overall_roas:.2f}

CAMPAIGN PERFORMANCE
--------------------

"""


for _, row in report_campaigns.iterrows():

    report_text += f"""
Campaign          : {row["Campaign"]}
Spend             : ₹{row["Spend"]:,.2f}
Revenue           : ₹{row["Revenue"]:,.2f}
Conversions       : {row["Conversions"]:,.0f}
ROAS              : {row["ROAS"]:.2f}
CTR               : {format_percentage(row["CTR"])}
Conversion Rate   : {format_percentage(row["Conversion_Rate"])}
CPA               : {format_currency(row["CPA"]) if pd.notna(row["CPA"]) else "N/A"}
Health Score      : {row["Health_Score"]:.1f}/100
Health Status     : {row["Health_Status"]}
-------------------------------------------------
"""


report_text += f"""

BUSINESS INSIGHT
----------------

Best Campaign:
{best_campaign_name}

Best Campaign ROAS:
{best_campaign["ROAS"]:.2f}

Campaign Requiring Most Attention:
{worst_campaign_name}

Lowest ROAS:
{worst_campaign["ROAS"]:.2f}

Average Health Score:
{average_health_score:.1f}/100


RECOMMENDED BUSINESS DIRECTION
------------------------------

The strongest campaign can be considered for
controlled budget scaling if its performance
remains stable.

The lower-performing campaign should be reviewed
before significant budget increases.

Recommended review areas:

• Audience targeting
• Ad creatives
• Landing page experience
• Offer quality
• Conversion flow
• Customer acquisition cost


MARKETMIND AI DECISION
----------------------

Analyse → Why → What's Next → What Should I Do?


=================================================
Generated by MarketMind AI
=================================================
"""


st.download_button(
    label="📥 Download Marketing Report",
    data=report_text,
    file_name="MarketMind_Marketing_Report.txt",
    mime="text/plain"
)



# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "📊 MarketMind AI | Marketing Intelligence & "
    "Decision Support System"
)


