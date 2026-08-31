# ==========================================
# MARKETMIND AI
# LLM BUSINESS INSIGHT GENERATOR
# ==========================================

import pandas as pd


# ==========================================
# LOAD MARKETING DATA
# ==========================================

df = pd.read_csv(
    "data/marketing_kpi_data.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)


# ==========================================
# CALCULATE CAMPAIGN PERFORMANCE
# ==========================================

campaign_data = (
    df
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
# KPI CALCULATIONS
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


# ==========================================
# FIND BEST & WORST CAMPAIGN
# ==========================================

best_campaign = campaign_data.loc[
    campaign_data["ROAS"].idxmax()
]

worst_campaign = campaign_data.loc[
    campaign_data["ROAS"].idxmin()
]


# ==========================================
# OVERALL PERFORMANCE
# ==========================================

total_spend = df["Spend"].sum()

total_revenue = df["Revenue"].sum()

overall_roas = (
    total_revenue / total_spend
    if total_spend > 0
    else 0
)


# ==========================================
# GENERATE BUSINESS INSIGHT
# ==========================================

if overall_roas >= 175:

    performance = (
        "Overall marketing performance is strong."
    )

elif overall_roas >= 165:

    performance = (
        "Overall marketing performance is moderate "
        "with opportunities for optimization."
    )

else:

    performance = (
        "Overall marketing performance needs "
        "attention and optimization."
    )


# ==========================================
# CREATE AI-STYLE INSIGHT
# ==========================================

insight = f"""
MARKETMIND AI - BUSINESS INSIGHT
=================================

Overall Performance
-------------------

{performance}

Overall ROAS:
{overall_roas:.2f}


Best Performing Campaign
------------------------

Campaign:
{best_campaign["Campaign"]}

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


Lowest Performing Campaign
--------------------------

Campaign:
{worst_campaign["Campaign"]}

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


Why This Happened
-----------------

The difference between the campaigns may be influenced
by CTR, conversion rate, customer acquisition cost,
audience targeting, creative effectiveness and
landing page performance.


Recommended Business Direction
------------------------------

The strongest campaign can be monitored for potential
budget scaling while maintaining ROAS.

The weaker campaign should be optimized by reviewing
targeting, creatives, audience relevance and conversion
flow before increasing the budget.


MarketMind AI Decision
----------------------

Analyse → Why → What's Next → What Should I Do

MarketMind AI combines marketing KPIs and business
rules to provide decision-support insights.
"""


# ==========================================
# SAVE INSIGHT
# ==========================================

with open(
    "data/llm_business_insight.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(insight)


# ==========================================
# SUCCESS MESSAGE
# ==========================================

print("==========================================")
print("MarketMind AI Business Insight Generated!")
print("==========================================")
print("Best Campaign:", best_campaign["Campaign"])
print("Worst Campaign:", worst_campaign["Campaign"])
print("Overall ROAS:", round(overall_roas, 2))
print("Saved: data/llm_business_insight.txt")
print("==========================================")