import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

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
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROFESSIONAL ANIMATION
# ============================================================

st.markdown(
    """
    <style>

    h1 {
        animation: titleEnter 1.2s ease-out;
    }

    @keyframes titleEnter {
        0% {
            opacity: 0;
            transform: translateY(-25px);
        }

        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    [data-testid="stMetric"] {
        animation: cardEnter 0.8s ease-out;
        transition: transform 0.25s ease;
    }

    @keyframes cardEnter {
        0% {
            opacity: 0;
            transform: translateY(25px);
        }

        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
    }

    .stButton > button {
        transition: transform 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
    }

    .stAlert {
        animation: fadeIn 0.7s ease-out;
    }

    @keyframes fadeIn {
        0% {
            opacity: 0;
        }

        100% {
            opacity: 1;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


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
    Dataset-relative recommendation engine.
    """

    actions = []

    campaign = row["Campaign"]

    # --------------------------------------------------------
    # Dataset benchmarks
    # --------------------------------------------------------

    median_roas = (
        campaign_data["ROAS"]
        .median()
    )

    median_cpa = (
        campaign_data["CPA"]
        .median()
    )

    median_ctr = (
        campaign_data["CTR"]
        .median()
        if campaign_data["CTR"].notna().any()
        else np.nan
    )

    median_conversion = (
        campaign_data["Conversion_Rate"].median()
        if campaign_data["Conversion_Rate"].notna().any()
        else np.nan
    )

    # --------------------------------------------------------
    # ROAS
    # --------------------------------------------------------

    if pd.notna(row["ROAS"]):

        if row["ROAS"] > median_roas * 1.10:

            actions.append(
                "Consider gradually scaling budget "
                "while monitoring efficiency."
            )

        elif row["ROAS"] < median_roas * 0.90:

            actions.append(
                "Review spending efficiency and "
                "optimize the campaign before scaling."
            )

        else:

            actions.append(
                "Maintain the current budget and "
                "continue monitoring performance."
            )

    # --------------------------------------------------------
    # CTR
    # --------------------------------------------------------

    if pd.notna(row["CTR"]) and pd.notna(median_ctr):

        if row["CTR"] < median_ctr * 0.90:

            actions.append(
                "Test new ad creatives, headlines "
                "and calls-to-action."
            )

    # --------------------------------------------------------
    # Conversion Rate
    # --------------------------------------------------------

    if (
        pd.notna(row["Conversion_Rate"])
        and pd.notna(median_conversion)
    ):

        if (
            row["Conversion_Rate"]
            < median_conversion * 0.90
        ):

            actions.append(
                "Review landing page experience, "
                "offer quality and conversion flow."
            )

    # --------------------------------------------------------
    # CPA
    # --------------------------------------------------------

    if pd.notna(row["CPA"]) and pd.notna(median_cpa):

        if row["CPA"] > median_cpa * 1.10:

            actions.append(
                "Optimize audience targeting and "
                "acquisition efficiency."
            )

    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    if not actions:

        actions.append(
            "Performance is stable relative to the "
            "current dataset. Continue monitoring KPIs."
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

st.title("📊 MarketMind AI")

st.subheader(
    "Marketing Intelligence & Performance Analytics"
)

st.write(
    "Analyse your marketing performance, identify trends, "
    "understand why performance happened and decide what to do next."
)


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

st.subheader(
    "📅 Date Filter"
)

min_date = active_df["Date"].min().date()

max_date = active_df["Date"].max().date()

col1, col2 = st.columns(2)

with col1:

    start_date = st.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        key="mm_start_date"
    )

with col2:

    end_date = st.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="mm_end_date"
    )


if start_date > end_date:

    st.error(
        "⚠️ Start Date cannot be after End Date."
    )

    filtered_df = active_df.copy()

else:

    filtered_df = active_df[
        (
            active_df["Date"]
            >= pd.Timestamp(start_date)
        )
        &
        (
            active_df["Date"]
            <= pd.Timestamp(end_date)
        )
    ].copy()


st.caption(
    f"Showing data from "
    f"{start_date.strftime('%d %b %Y')} "
    f"to "
    f"{end_date.strftime('%d %b %Y')}"
)


if filtered_df.empty:

    st.warning(
        "No data available for the selected date range."
    )

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

daily_revenue = (
    filtered_df
    .groupby("Date", as_index=False)
    ["Revenue"]
    .sum()
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
# BUSINESS SUMMARY
# ============================================================

st.markdown("---")

st.header(
    "🧠 MarketMind Business Summary"
)


# Dataset-relative overall interpretation

if len(campaign_data) == 1:

    performance_message = (
        "The dataset contains one campaign. "
        "Performance is evaluated using its own KPI values."
    )

else:

    if average_health_score >= 70:

        performance_message = (
            "Overall campaign performance is relatively strong "
            "within the active dataset."
        )

    elif average_health_score >= 50:

        performance_message = (
            "Overall campaign performance is moderate "
            "with optimization opportunities."
        )

    else:

        performance_message = (
            "Overall campaign performance is below the "
            "dataset benchmark and requires optimization."
        )


st.info(
    f"""
### 📊 Current Business Performance

**{performance_message}**

🏆 **Best Campaign:** {best_campaign_name}

ROAS: **{best_campaign["ROAS"]:.2f}**

⚠️ **Campaign Requiring Most Attention:**
{worst_campaign_name}

ROAS: **{worst_campaign["ROAS"]:.2f}**

❤️ **Average Campaign Health Score:**
{average_health_score:.1f}/100

💰 **Total Marketing Spend:**
₹{total_spend:,.2f}

💵 **Total Revenue:**
₹{total_revenue:,.2f}

🚀 **Overall ROAS:**
{overall_roas:.2f}
"""
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