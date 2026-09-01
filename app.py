import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# MARKETING MIND AI - CLEAN PROFESSIONAL UI
# ==========================================

st.markdown("""
<style>

/* ---------- BACKGROUND ---------- */

.stApp {
    background: linear-gradient(
        135deg,
        #f5f9ff 0%,
        #ffffff 45%,
        #f3f0ff 100%
    );
}


/* ---------- MAIN CONTENT ---------- */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ---------- MAIN TITLE ---------- */

.main-title {
    font-size: 42px;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 8px;

    background: linear-gradient(
        90deg,
        #2563eb,
        #6366f1,
        #9333ea,
        #2563eb
    );

    background-size: 300% 100%;

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    animation: gradientMove 5s ease infinite;
}


/* ---------- TITLE ANIMATION ---------- */

@keyframes gradientMove {

    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }

}


/* ---------- SUBTITLE ---------- */

.subtitle {
    color: #475569;
    font-size: 17px;
    margin-bottom: 25px;
}


/* ---------- KPI CARDS ---------- */

[data-testid="stMetric"] {

    background: rgba(255, 255, 255, 0.95);

    border: 1px solid #e2e8f0;

    border-radius: 16px;

    padding: 18px;

    box-shadow:
        0 5px 20px rgba(30, 64, 175, 0.08);

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;
}


/* KPI HOVER ANIMATION */

[data-testid="stMetric"]:hover {

    transform: translateY(-6px);

    box-shadow:
        0 14px 30px rgba(37, 99, 235, 0.15);
}


/* ---------- KPI VALUE ---------- */

[data-testid="stMetricValue"] {
    color: #172554 !important;
    font-weight: 800 !important;
}


/* ---------- KPI LABEL ---------- */

[data-testid="stMetricLabel"] {
    color: #64748b !important;
}


/* ---------- BUTTON ---------- */

.stButton > button {

    border-radius: 10px;

    border: none;

    background: linear-gradient(
        135deg,
        #2563eb,
        #6366f1
    );

    color: white;

    font-weight: 700;

    transition: all 0.25s ease;

    box-shadow:
        0 5px 15px rgba(37, 99, 235, 0.20);
}


.stButton > button:hover {

    transform: translateY(-3px);

    box-shadow:
        0 10px 25px rgba(99, 102, 241, 0.30);
}


/* ---------- ALERT BOX ---------- */

.stAlert {
    border-radius: 14px !important;
}


/* ---------- DATAFRAME ---------- */

[data-testid="stDataFrame"] {

    border-radius: 14px;

    overflow: hidden;

    box-shadow:
        0 5px 20px rgba(30, 64, 175, 0.07);
}


/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {

    background: #f8fbff;

    border-right: 1px solid #e2e8f0;
}


/* ---------- MOBILE ---------- */

@media (max-width: 768px) {

    .main-title {
        font-size: 30px;
    }

    .subtitle {
        font-size: 15px;
    }

}

</style>
""", unsafe_allow_html=True)


   

# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

DATA_FILE = DATA_DIR / "marketing_kpi_data.csv"

MODEL_FILE = BASE_DIR / "models" / "revenue_model.pkl"

CHROMA_DIR = DATA_DIR / "chroma_db"

AI_INSIGHT_FILE = DATA_DIR / "llm_business_insight.txt"


# ==========================================
# LOAD DATA
# ==========================================

try:

    df = pd.read_csv(DATA_FILE)

except Exception as e:

    st.error("❌ Marketing dataset could not be loaded.")

    st.code(str(e))

    st.stop()


# ==========================================
# DATE CONVERSION
# ==========================================

df["Date"] = pd.to_datetime(df["Date"])


# ==========================================
# TITLE
# ==========================================

st.title("📊 MarketMind AI")

st.subheader(
    "Marketing Intelligence & Performance Analytics"
)

st.write(
    "Analyse your marketing performance, "
    "identify trends, understand why performance "
    "happened and decide what to do next."
)


# ==========================================
# KPI CALCULATIONS
# ==========================================

total_spend = df["Spend"].sum()

total_revenue = df["Revenue"].sum()

total_conversions = df["Conversions"].sum()

total_clicks = df["Clicks"].sum()

total_impressions = df["Impressions"].sum()


if total_impressions > 0:

    overall_ctr = (
        total_clicks / total_impressions
    ) * 100

else:

    overall_ctr = 0


if total_spend > 0:

    overall_roas = (
        total_revenue / total_spend
    )

else:

    overall_roas = 0


# ==========================================
# KPI CARDS
# ==========================================

st.markdown("## 📌 Marketing KPIs")

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "💰 Total Spend",
        f"₹{total_spend:,.0f}"
    )


with col2:

    st.metric(
        "💵 Revenue",
        f"₹{total_revenue:,.0f}"
    )


with col3:

    st.metric(
        "🎯 Conversions",
        f"{total_conversions:,}"
    )


with col4:

    st.metric(
        "📈 CTR",
        f"{overall_ctr:.2f}%"
    )


with col5:

    st.metric(
        "🚀 ROAS",
        f"{overall_roas:.2f}"
    )

# ==========================================
# UPLOAD YOUR OWN MARKETING DATA
# ==========================================

st.markdown("---")
st.subheader("📥 Upload Your Own Marketing Data")

uploaded_file = st.file_uploader(
    "Upload your marketing dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):
            uploaded_df = pd.read_csv(uploaded_file)

        else:
            uploaded_df = pd.read_excel(uploaded_file)

        st.success("✅ Marketing data uploaded successfully!")

        st.write("### 📊 Uploaded Data Preview")

        st.dataframe(
            uploaded_df.head(10),
            use_container_width=True
        )

        st.write("### 📋 Dataset Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Rows",
                uploaded_df.shape[0]
            )

        with col2:
            st.metric(
                "Columns",
                uploaded_df.shape[1]
            )

        with col3:
            st.metric(
                "Missing Values",
                int(uploaded_df.isnull().sum().sum())
            )

    except Exception as e:

        st.error(
            f"❌ Unable to read the uploaded file: {e}"
        )

# ==========================================
## ==========================================
# DATE FILTER
# ==========================================

# ==========================================
# DATE FILTER
# ==========================================
# ==========================================
# DATE FILTER
# ==========================================

st.markdown("---")
st.subheader("📅 Date Filter")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        key="start_date"
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="end_date"
    )

# Validate date range
if start_date > end_date:

    st.error("⚠️ Start Date cannot be after End Date.")

    filtered_df = df.copy()

else:

    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date)

    filtered_df = df[
        (df["Date"] >= start_timestamp)
        &
        (df["Date"] <= end_timestamp)
    ]

st.caption(
    f"Showing data from {start_date.strftime('%d %b %Y')} "
    f"to {end_date.strftime('%d %b %Y')}"
)
# ==========================================
# DAILY REVENUE
# ==========================================

st.markdown("---")

st.subheader("📈 Daily Revenue Trend")


daily_revenue = (
    filtered_df
    .groupby("Date")["Revenue"]
    .sum()
    .reset_index()
)


fig_revenue = px.line(
    daily_revenue,
    x="Date",
    y="Revenue",
    title="Daily Revenue"
)


st.plotly_chart(
    fig_revenue,
    width="stretch"
)


# ==========================================
# CAMPAIGN PERFORMANCE
# ==========================================

st.markdown("---")

st.subheader("🎯 Campaign Performance")


campaign_data = (
    filtered_df
    .groupby("Campaign")
    .agg(
        Spend=("Spend", "sum"),
        Revenue=("Revenue", "sum"),
        Clicks=("Clicks", "sum"),
        Conversions=("Conversions", "sum"),
        Impressions=("Impressions", "sum")
    )
    .reset_index()
)


# ==========================================
# CAMPAIGN KPI CALCULATIONS
# ==========================================

campaign_data["ROAS"] = (
    campaign_data["Revenue"]
    /
    campaign_data["Spend"].replace(0, 1)
)


campaign_data["CTR"] = (
    campaign_data["Clicks"]
    /
    campaign_data["Impressions"].replace(0, 1)
) * 100


campaign_data["Conversion_Rate"] = (
    campaign_data["Conversions"]
    /
    campaign_data["Clicks"].replace(0, 1)
) * 100


campaign_data["CPA"] = (
    campaign_data["Spend"]
    /
    campaign_data["Conversions"].replace(0, 1)
)


campaign_data = campaign_data.round(2)


# ==========================================
# CAMPAIGN CHARTS
# ==========================================

col1, col2 = st.columns(2)


with col1:

    fig_campaign = px.bar(
        campaign_data,
        x="Campaign",
        y="Revenue",
        title="Revenue by Campaign"
    )

    st.plotly_chart(
        fig_campaign,
        width="stretch"
    )


with col2:

    fig_roas = px.bar(
        campaign_data,
        x="Campaign",
        y="ROAS",
        title="ROAS by Campaign"
    )

    st.plotly_chart(
        fig_roas,
        width="stretch"
    )


# ==========================================
# CHANNEL PERFORMANCE
# ==========================================

st.markdown("---")

st.subheader("📣 Channel Performance")


channel_data = (
    filtered_df
    .groupby("Channel")
    .agg(
        Spend=("Spend", "sum"),
        Revenue=("Revenue", "sum"),
        Clicks=("Clicks", "sum"),
        Conversions=("Conversions", "sum"),
        Impressions=("Impressions", "sum")
    )
    .reset_index()
)


channel_data["ROAS"] = (
    channel_data["Revenue"]
    /
    channel_data["Spend"].replace(0, 1)
)


channel_data["CTR"] = (
    channel_data["Clicks"]
    /
    channel_data["Impressions"].replace(0, 1)
) * 100


channel_data = channel_data.round(2)


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
        width="stretch"
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
        width="stretch"
    )


# ==========================================
# CAMPAIGN HEALTH SCORE
# ==========================================

st.markdown("---")

st.subheader("❤️ Campaign Health Score")


def calculate_health_score(row):

    score = 0

    # ROAS - 40 points

    if row["ROAS"] >= 180:

        score += 40

    elif row["ROAS"] >= 170:

        score += 30

    elif row["ROAS"] >= 160:

        score += 20

    else:

        score += 10


    # CTR - 20 points

    if row["CTR"] >= 4.5:

        score += 20

    elif row["CTR"] >= 4:

        score += 15

    elif row["CTR"] >= 3.5:

        score += 10

    else:

        score += 5


    # Conversion Rate - 25 points

    if row["Conversion_Rate"] >= 8.5:

        score += 25

    elif row["Conversion_Rate"] >= 8:

        score += 20

    elif row["Conversion_Rate"] >= 7:

        score += 10

    else:

        score += 5


    # CPA - 15 points

    if row["CPA"] <= 10:

        score += 15

    elif row["CPA"] <= 11:

        score += 10

    elif row["CPA"] <= 12:

        score += 5

    return score


campaign_data["Health_Score"] = campaign_data.apply(
    calculate_health_score,
    axis=1
)


def health_status(score):

    if score >= 85:

        return "Excellent"

    elif score >= 70:

        return "Good"

    elif score >= 50:

        return "Needs Improvement"

    else:

        return "Critical"


campaign_data["Health_Status"] = (
    campaign_data["Health_Score"]
    .apply(health_status)
)


st.dataframe(
    campaign_data[
        [
            "Campaign",
            "Health_Score",
            "Health_Status"
        ]
    ],
    width="stretch"
)


# ==========================================
# BEST & WORST CAMPAIGNS
# ==========================================

best_campaign = campaign_data.loc[
    campaign_data["ROAS"].idxmax()
]

worst_campaign = campaign_data.loc[
    campaign_data["ROAS"].idxmin()
]


best_campaign_name = best_campaign["Campaign"]

worst_campaign_name = worst_campaign["Campaign"]


average_health_score = (
    campaign_data["Health_Score"].mean()
)


# ==========================================
# CAMPAIGN DEEP DIVE
# ==========================================

st.markdown("---")

st.header("🔎 Campaign Deep Dive")


selected_campaign = st.selectbox(
    "Select a campaign",
    campaign_data["Campaign"].tolist()
)


selected_data = campaign_data[
    campaign_data["Campaign"] == selected_campaign
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
        f"{selected_data['CTR']:.2f}%"
    )


with col3:

    st.metric(
        "Conversion Rate",
        f"{selected_data['Conversion_Rate']:.2f}%"
    )


with col4:

    st.metric(
        "CPA",
        f"₹{selected_data['CPA']:.2f}"
    )


# ==========================================
# WHY DID THIS HAPPEN?
# ==========================================

st.subheader("🔍 Campaign Analysis - Why?")


reasons = []


if selected_data["ROAS"] < 165:

    reasons.append(
        "ROAS is relatively low, indicating weaker "
        "revenue efficiency."
    )


if selected_data["CTR"] < 4.5:

    reasons.append(
        "CTR is below the target level, suggesting "
        "the ad creative or messaging may need improvement."
    )


if selected_data["Conversion_Rate"] < 8:

    reasons.append(
        "Conversion rate is relatively low, which may "
        "indicate landing page, offer or audience issues."
    )


if selected_data["CPA"] > 10.5:

    reasons.append(
        "CPA is relatively high, indicating higher "
        "customer acquisition cost."
    )


if not reasons:

    reasons.append(
        "The campaign is performing strongly across "
        "the major KPIs."
    )


for reason in reasons:

    st.info("• " + reason)


# ==========================================
# WHAT'S NEXT
# ==========================================

st.subheader("🔮 Campaign Outlook")


if selected_data["Health_Score"] >= 85:

    st.success(
        "The campaign is currently strong. "
        "If the trend continues, consider gradually "
        "scaling the budget while monitoring ROAS."
    )

elif selected_data["Health_Score"] >= 70:

    st.warning(
        "The campaign has optimization opportunities. "
        "Improve weaker KPIs before significantly "
        "increasing budget."
    )

else:

    st.error(
        "The campaign requires immediate attention. "
        "Optimize targeting, creatives and conversion "
        "flow before scaling."
    )


# ==========================================
# WHAT SHOULD I DO?
# ==========================================

st.subheader("🎯 Recommended Action")


actions = []


if selected_data["ROAS"] < 165:

    actions.append(
        "Reduce inefficient spending and review campaign targeting."
    )

elif selected_data["ROAS"] < 175:

    actions.append(
        "Optimize campaign performance before increasing the budget."
    )

else:

    actions.append(
        "Consider gradually increasing the budget if performance remains stable."
    )


if selected_data["CTR"] < 4.5:

    actions.append(
        "Test new creatives, headlines and calls-to-action."
    )


if selected_data["Conversion_Rate"] < 8:

    actions.append(
        "Improve landing page experience and conversion flow."
    )


if selected_data["CPA"] > 10.5:

    actions.append(
        "Optimize audience targeting to reduce acquisition cost."
    )


for i, action in enumerate(actions, start=1):

    st.write(
        f"**{i}.** {action}"
    )


# ==========================================
# BUSINESS SUMMARY
# ==========================================

st.markdown("---")

st.header("🧠 MarketMind Business Summary")


if overall_roas >= 175:

    performance_message = (
        "Overall marketing performance is strong."
    )

elif overall_roas >= 165:

    performance_message = (
        "Overall marketing performance is moderate "
        "and has optimization opportunities."
    )

else:

    performance_message = (
        "Overall marketing performance needs attention "
        "and optimization."
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


# ==========================================
# CAMPAIGN KPI TABLE
# ==========================================

st.markdown("---")

st.subheader("📋 Campaign KPI Table")


st.dataframe(
    campaign_data,
    width="stretch"
)


# ==========================================
# MARKETMIND INSIGHT
# ==========================================

st.markdown("---")

st.subheader("🤖 MarketMind Insight")


st.success(
    f"""
🏆 Best Campaign: {best_campaign_name}

ROAS: {best_campaign["ROAS"]:.2f}

⚠️ Lowest ROAS Campaign: {worst_campaign_name}

ROAS: {worst_campaign["ROAS"]:.2f}
"""
)


# ==========================================
# CAMPAIGN-WISE RECOMMENDATIONS
# ==========================================

st.markdown("---")

st.header("🎯 Campaign Recommendations")


for _, row in campaign_data.iterrows():

    campaign = row["Campaign"]

    roas = row["ROAS"]

    ctr = row["CTR"]

    conversion_rate = row["Conversion_Rate"]

    cpa = row["CPA"]

    health_score = row["Health_Score"]


    campaign_actions = []


    # ROAS

    if roas < 165:

        campaign_actions.append(
            "Reduce or pause inefficient spending and review campaign efficiency."
        )

    elif roas < 175:

        campaign_actions.append(
            "Optimize the campaign before increasing the budget."
        )

    else:

        campaign_actions.append(
            "Consider gradually increasing budget if performance remains stable."
        )


    # CTR

    if ctr < 4.5:

        campaign_actions.append(
            "Test new ad creatives, headlines and calls-to-action."
        )


    # Conversion Rate

    if conversion_rate < 8:

        campaign_actions.append(
            "Improve landing page experience, offer and conversion flow."
        )


    # CPA

    if cpa > 10.5:

        campaign_actions.append(
            "Optimize audience targeting to reduce acquisition cost."
        )


    final_actions = " ".join(campaign_actions)


    if health_score >= 85:

        st.success(
            f"""
🟢 **{campaign} — Strong Campaign**

{final_actions}
"""
        )

    elif health_score >= 70:

        st.warning(
            f"""
🟡 **{campaign} — Optimization Recommended**

{final_actions}
"""
        )

    else:

        st.error(
            f"""
🔴 **{campaign} — Immediate Attention Required**

{final_actions}
"""
        )


# ==========================================
# ML PREDICTION
# ==========================================

st.markdown("---")

st.header("🔮 AI Revenue Forecast")

st.write(
    "MarketMind AI uses a machine learning model "
    "to estimate future revenue based on recent "
    "marketing performance."
)


try:

    model = joblib.load(
        MODEL_FILE
    )


    latest_data = (
        df
        .sort_values("Date")
        .iloc[-1]
    )


    latest_impressions = latest_data["Impressions"]

    latest_clicks = latest_data["Clicks"]

    latest_conversions = latest_data["Conversions"]

    latest_spend = latest_data["Spend"]

    latest_revenue = latest_data["Revenue"]


    if latest_impressions > 0:

        latest_ctr = (
            latest_clicks /
            latest_impressions
        ) * 100

    else:

        latest_ctr = 0


    if latest_clicks > 0:

        latest_conversion_rate = (
            latest_conversions /
            latest_clicks
        ) * 100

    else:

        latest_conversion_rate = 0


    if latest_spend > 0:

        latest_roas = (
            latest_revenue /
            latest_spend
        )

    else:

        latest_roas = 0


    prediction_input = pd.DataFrame({

        "Impressions": [
            latest_impressions
        ],

        "Clicks": [
            latest_clicks
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

    })


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


except FileNotFoundError:

    st.warning(
        "Revenue prediction model not found. "
        "Run revenue_prediction.py first."
    )


except Exception as e:

    st.error(
        f"Prediction error: {e}"
    )


# ==========================================
# AI BUSINESS INSIGHT
# ==========================================

st.markdown("---")

st.header("🤖 AI Business Insight")


try:

    with open(
        AI_INSIGHT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        ai_insight = file.read()


    st.info(
        ai_insight
    )


except FileNotFoundError:

    st.warning(
        "AI business insight file not found. "
        "Run the AI insight generator first."
    )


# ==========================================
# WHAT-IF SCENARIO
# ==========================================

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
    total_spend *
    (1 + budget_change / 100)
)


scenario_revenue = (
    scenario_spend *
    current_roas
)


revenue_change = (
    scenario_revenue -
    total_revenue
)


if total_revenue > 0:

    revenue_change_percentage = (
        revenue_change /
        total_revenue
    ) * 100

else:

    revenue_change_percentage = 0


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
could potentially increase revenue by
{revenue_change_percentage:.2f}% if current ROAS
remains stable.
"""
    )

elif budget_change < 0:

    st.warning(
        f"""
📉 Reducing the budget by {abs(budget_change)}%
could potentially reduce revenue by
{abs(revenue_change_percentage):.2f}% if current ROAS
remains stable.
"""
    )

else:

    st.info(
        "ℹ️ No budget change. Current performance "
        "is being used as the baseline."
    )


# ==========================================
# DYNAMIC RAG INSIGHT
# ==========================================

st.markdown("---")

st.header("🔍 RAG-Powered Performance Analysis")


st.write(
    "MarketMind AI retrieves relevant marketing "
    "knowledge from the RAG knowledge base and "
    "combines it with real campaign KPIs."
)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


try:

    # ======================================
    # CHECK CHROMA DATABASE
    # ======================================

    if not CHROMA_DIR.exists():

        st.warning(
            "⚠️ ChromaDB folder was not found."
        )

        st.code(
            str(CHROMA_DIR)
        )

    else:

        # ==================================
        # LOAD EMBEDDING MODEL
        # ==================================

        embedding_model = (
            load_embedding_model()
        )


        # ==================================
        # CONNECT TO CHROMADB
        # ==================================

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )


        # ==================================
        # GET COLLECTION
        # ==================================

        collection = client.get_collection(
            name="marketing_knowledge"
        )


        # ==================================
        # CREATE DYNAMIC QUERY
        # ==================================

        query = f"""

Explain why the marketing performance happened.

Best Performing Campaign:
{best_campaign_name}

ROAS:
{best_campaign["ROAS"]:.2f}

CTR:
{best_campaign["CTR"]:.2f}%

Conversion Rate:
{best_campaign["Conversion_Rate"]:.2f}%

CPA:
₹{best_campaign["CPA"]:.2f}

Revenue:
₹{best_campaign["Revenue"]:,.2f}


Lowest Performing Campaign:
{worst_campaign_name}

ROAS:
{worst_campaign["ROAS"]:.2f}

CTR:
{worst_campaign["CTR"]:.2f}%

Conversion Rate:
{worst_campaign["Conversion_Rate"]:.2f}%

CPA:
₹{worst_campaign["CPA"]:.2f}

Revenue:
₹{worst_campaign["Revenue"]:,.2f}


Overall ROAS:
{overall_roas:.2f}

Explain possible reasons using:

CTR
Conversion Rate
ROAS
CPA
Audience Targeting
Creative Quality
Landing Page
Offer Quality
Audience Relevance
Campaign Efficiency

"""


        # ==================================
        # CREATE EMBEDDING
        # ==================================

        query_embedding = (
            embedding_model
            .encode(query)
            .tolist()
        )


        # ==================================
        # SEARCH RAG KNOWLEDGE
        # ==================================

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=5
        )


        # ==================================
        # GET DOCUMENTS
        # ==================================

        documents = results.get(
            "documents",
            [[]]
        )


        if documents and len(documents[0]) > 0:

            knowledge_context = "\n\n".join(
                documents[0]
            )


            # ==================================
            # RAG SUCCESS
            # ==================================

            st.success(
                "🧠 RAG Knowledge Retrieved Successfully"
            )


            st.write(
                f"Retrieved {len(documents[0])} "
                "relevant marketing knowledge documents."
            )


            # ==================================
            # RETRIEVED KNOWLEDGE
            # ==================================

            with st.expander(
                "📚 View Retrieved Marketing Knowledge"
            ):

                st.write(
                    knowledge_context
                )


            # ==================================
            # PERFORMANCE EXPLANATION
            # ==================================

            st.subheader(
                "💡 Performance Explanation"
            )


            st.info(
                f"""
🏆 **{best_campaign_name}** is currently the
strongest campaign based on ROAS.

Its ROAS is **{best_campaign["ROAS"]:.2f}**,
while **{worst_campaign_name}** has a ROAS of
**{worst_campaign["ROAS"]:.2f}**.

The performance difference can be related to
campaign efficiency, audience relevance,
creative effectiveness and conversion performance.

The RAG knowledge base was used to retrieve
marketing concepts relevant to these KPIs.
"""
            )


            # ==================================
            # KPI-BASED REASONS
            # ==================================

            st.subheader(
                "🔎 Possible Reasons"
            )


            rag_reasons = []


            if (
                best_campaign["CTR"]
                >
                worst_campaign["CTR"]
            ):

                rag_reasons.append(
                    f"{best_campaign_name} has a higher CTR "
                    "than the lowest-performing campaign. "
                    "This may indicate stronger ad relevance "
                    "or more effective creatives."
                )

            else:

                rag_reasons.append(
                    "CTR is not significantly stronger in "
                    "the best campaign, so other factors "
                    "may be contributing to the ROAS difference."
                )


            if (
                best_campaign["Conversion_Rate"]
                >
                worst_campaign["Conversion_Rate"]
            ):

                rag_reasons.append(
                    f"{best_campaign_name} has a higher "
                    "conversion rate, suggesting better "
                    "audience quality, offer relevance or "
                    "landing page performance."
                )

            else:

                rag_reasons.append(
                    "Conversion rate does not explain the "
                    "entire performance difference, so "
                    "targeting and revenue efficiency should "
                    "also be reviewed."
                )


            if (
                best_campaign["CPA"]
                <
                worst_campaign["CPA"]
            ):

                rag_reasons.append(
                    f"{best_campaign_name} has a lower CPA, "
                    "indicating more efficient customer acquisition."
                )

            else:

                rag_reasons.append(
                    "CPA is not lower for the best campaign, "
                    "so ROAS may be influenced by revenue value "
                    "or other campaign factors."
                )


            for reason in rag_reasons:

                st.write(
                    "• " + reason
                )


            # ==================================
            # RAG DECISION
            # ==================================

            st.subheader(
                "🧠 MarketMind AI Decision"
            )


            if best_campaign["ROAS"] >= 180:

                st.success(
                    f"""
**Scale Carefully:** {best_campaign_name}

This campaign shows strong ROAS.
Consider gradual budget scaling while monitoring
CTR, conversion rate and CPA.
"""
                )

            else:

                st.warning(
                    f"""
**Optimize Before Scaling:** {best_campaign_name}

The campaign is performing relatively well,
but additional optimization may be useful
before increasing budget significantly.
"""
                )


            if worst_campaign["ROAS"] < 165:

                st.error(
                    f"""
**Immediate Optimization:** {worst_campaign_name}

Review targeting, creatives, landing page,
audience relevance and acquisition cost
before increasing its budget.
"""
                )

            else:

                st.warning(
                    f"""
**Monitor:** {worst_campaign_name}

The campaign is weaker than the best campaign.
Review its KPIs and optimize before scaling.
"""
                )


        else:

            st.warning(
                "⚠️ No relevant marketing knowledge "
                "was found in the RAG knowledge base."
            )


except Exception as e:

    st.error(
        "❌ RAG Insight Error"
    )

    st.caption(
        f"Details: {e}"
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "📊 MarketMind AI | Marketing Intelligence & "
    "Decision Support System"
)

# ==========================================
# GENERATE MARKETING REPORT
# ==========================================

st.markdown("---")
st.subheader("📄 Generate Marketing Report")

# Create campaign summary for report
report_campaigns = (
    filtered_df
    .groupby("Campaign", as_index=False)
    .agg({
        "Spend": "sum",
        "Revenue": "sum",
        "Conversions": "sum"
    })
)

report_campaigns["ROAS"] = (
    report_campaigns["Revenue"]
    / report_campaigns["Spend"]
)

# Best Campaign
best_campaign = report_campaigns.loc[
    report_campaigns["ROAS"].idxmax()
]

# Lowest Campaign
worst_campaign = report_campaigns.loc[
    report_campaigns["ROAS"].idxmin()
]

# Create Report
report_text = f"""
=============================================
        MARKETMIND AI
   MARKETING PERFORMANCE REPORT
=============================================

REPORT PERIOD
-------------
Start Date : {filtered_df["Date"].min().strftime("%d %b %Y")}
End Date   : {filtered_df["Date"].max().strftime("%d %b %Y")}


MARKETING KPIs
--------------

Total Spend    : ₹{total_spend:,.2f}
Total Revenue  : ₹{total_revenue:,.2f}
Conversions    : {total_conversions:,}
CTR            : {overall_ctr:.2f}%
ROAS           : {overall_roas:.2f}


CAMPAIGN PERFORMANCE
--------------------
"""

for _, row in report_campaigns.iterrows():

    report_text += f"""
Campaign      : {row["Campaign"]}
Spend         : ₹{row["Spend"]:,.2f}
Revenue       : ₹{row["Revenue"]:,.2f}
Conversions   : {row["Conversions"]:,}
ROAS          : {row["ROAS"]:.2f}
---------------------------------------------
"""


report_text += f"""

BUSINESS INSIGHT
----------------

🏆 Best Campaign
Campaign : {best_campaign["Campaign"]}
ROAS     : {best_campaign["ROAS"]:.2f}


⚠️ Campaign Requiring Attention
Campaign : {worst_campaign["Campaign"]}
ROAS     : {worst_campaign["ROAS"]:.2f}


RECOMMENDED BUSINESS DIRECTION
------------------------------

Monitor the strongest campaign for potential budget scaling
while maintaining ROAS, CTR and conversion performance.

Optimize the weaker campaign by reviewing:

• Audience targeting
• Ad creatives
• Landing page experience
• Conversion flow
• Customer acquisition cost


MARKETMIND AI DECISION
----------------------

Analyse → Why → What's Next → What Should I Do?


=============================================
Generated by MarketMind AI
=============================================
"""


st.download_button(
    label="📥 Download Marketing Report",
    data=report_text,
    file_name="MarketMind_Marketing_Report.txt",
    mime="text/plain"
)