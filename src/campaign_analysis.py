import pandas as pd

# Load KPI dataset
df = pd.read_csv("data/marketing_kpi_data.csv")

print("\n========================================")
print("   MARKETMIND AI - PERFORMANCE ANALYSIS")
print("========================================")


# -----------------------------------------
# 1. Campaign-wise Performance
# -----------------------------------------

campaign_analysis = df.groupby("Campaign").agg(
    Impressions=("Impressions", "sum"),
    Clicks=("Clicks", "sum"),
    Conversions=("Conversions", "sum"),
    Spend=("Spend", "sum"),
    Revenue=("Revenue", "sum")
).reset_index()


# Calculate campaign KPIs
campaign_analysis["CTR"] = (
    campaign_analysis["Clicks"] /
    campaign_analysis["Impressions"]
) * 100

campaign_analysis["CPC"] = (
    campaign_analysis["Spend"] /
    campaign_analysis["Clicks"]
)

campaign_analysis["Conversion_Rate"] = (
    campaign_analysis["Conversions"] /
    campaign_analysis["Clicks"]
) * 100

campaign_analysis["CPA"] = (
    campaign_analysis["Spend"] /
    campaign_analysis["Conversions"]
)

campaign_analysis["ROAS"] = (
    campaign_analysis["Revenue"] /
    campaign_analysis["Spend"]
)


# Round values
campaign_analysis = campaign_analysis.round(2)


print("\n===== CAMPAIGN PERFORMANCE =====\n")

print(campaign_analysis.to_string(index=False))


# -----------------------------------------
# 2. Best Campaign
# -----------------------------------------

best_campaign = campaign_analysis.loc[
    campaign_analysis["ROAS"].idxmax()
]

print("\n===== BEST CAMPAIGN =====")

print(
    f"Campaign : {best_campaign['Campaign']}"
)

print(
    f"ROAS     : {best_campaign['ROAS']}"
)

print(
    f"Revenue  : ₹{best_campaign['Revenue']:,.2f}"
)


# -----------------------------------------
# 3. Worst Campaign
# -----------------------------------------

worst_campaign = campaign_analysis.loc[
    campaign_analysis["ROAS"].idxmin()
]

print("\n===== WORST CAMPAIGN =====")

print(
    f"Campaign : {worst_campaign['Campaign']}"
)

print(
    f"ROAS     : {worst_campaign['ROAS']}"
)

print(
    f"Revenue  : ₹{worst_campaign['Revenue']:,.2f}"
)


# -----------------------------------------
# 4. Channel-wise Performance
# -----------------------------------------

channel_analysis = df.groupby("Channel").agg(
    Impressions=("Impressions", "sum"),
    Clicks=("Clicks", "sum"),
    Conversions=("Conversions", "sum"),
    Spend=("Spend", "sum"),
    Revenue=("Revenue", "sum")
).reset_index()


# Calculate channel KPIs

channel_analysis["CTR"] = (
    channel_analysis["Clicks"] /
    channel_analysis["Impressions"]
) * 100

channel_analysis["CPC"] = (
    channel_analysis["Spend"] /
    channel_analysis["Clicks"]
)

channel_analysis["Conversion_Rate"] = (
    channel_analysis["Conversions"] /
    channel_analysis["Clicks"]
) * 100

channel_analysis["CPA"] = (
    channel_analysis["Spend"] /
    channel_analysis["Conversions"]
)

channel_analysis["ROAS"] = (
    channel_analysis["Revenue"] /
    channel_analysis["Spend"]
)


channel_analysis = channel_analysis.round(2)


print("\n===== CHANNEL PERFORMANCE =====\n")

print(channel_analysis.to_string(index=False))


# -----------------------------------------
# 5. Best Channel
# -----------------------------------------

best_channel = channel_analysis.loc[
    channel_analysis["ROAS"].idxmax()
]

print("\n===== BEST CHANNEL =====")

print(
    f"Channel : {best_channel['Channel']}"
)

print(
    f"ROAS    : {best_channel['ROAS']}"
)


# -----------------------------------------
# 6. Worst Channel
# -----------------------------------------

worst_channel = channel_analysis.loc[
    channel_analysis["ROAS"].idxmin()
]

print("\n===== WORST CHANNEL =====")

print(
    f"Channel : {worst_channel['Channel']}"
)

print(
    f"ROAS    : {worst_channel['ROAS']}"
)


# -----------------------------------------
# 7. Save Results
# -----------------------------------------

campaign_analysis.to_csv(
    "data/campaign_performance.csv",
    index=False
)

channel_analysis.to_csv(
    "data/channel_performance.csv",
    index=False
)


print("\n========================================")
print("Analysis files created successfully!")
print("========================================")

print("\ndata/campaign_performance.csv")
print("data/channel_performance.csv")