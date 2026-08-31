import pandas as pd

# Load marketing dataset
df = pd.read_csv("data/marketing_data.csv")

# Calculate CTR
df["CTR"] = (df["Clicks"] / df["Impressions"]) * 100

# Calculate CPC
df["CPC"] = df["Spend"] / df["Clicks"]

# Calculate Conversion Rate
df["Conversion_Rate"] = (
    df["Conversions"] / df["Clicks"]
) * 100

# Calculate CPA
df["CPA"] = df["Spend"] / df["Conversions"]

# Calculate ROAS
df["ROAS"] = df["Revenue"] / df["Spend"]

# Round calculated values
df["CTR"] = df["CTR"].round(2)
df["CPC"] = df["CPC"].round(2)
df["Conversion_Rate"] = df["Conversion_Rate"].round(2)
df["CPA"] = df["CPA"].round(2)
df["ROAS"] = df["ROAS"].round(2)

# Display first 10 records
print("\n===== MARKETMIND AI KPI ANALYSIS =====\n")

print(df[
    [
        "Date",
        "Campaign",
        "Channel",
        "Impressions",
        "Clicks",
        "Conversions",
        "Spend",
        "Revenue",
        "CTR",
        "CPC",
        "Conversion_Rate",
        "CPA",
        "ROAS"
    ]
].head(10))

# Overall KPI summary
print("\n===== OVERALL KPI SUMMARY =====\n")

total_impressions = df["Impressions"].sum()
total_clicks = df["Clicks"].sum()
total_conversions = df["Conversions"].sum()
total_spend = df["Spend"].sum()
total_revenue = df["Revenue"].sum()

overall_ctr = (total_clicks / total_impressions) * 100
overall_cpc = total_spend / total_clicks
overall_conversion_rate = (
    total_conversions / total_clicks
) * 100
overall_cpa = total_spend / total_conversions
overall_roas = total_revenue / total_spend

print(f"Total Impressions   : {total_impressions:,}")
print(f"Total Clicks        : {total_clicks:,}")
print(f"Total Conversions   : {total_conversions:,}")
print(f"Total Spend         : ₹{total_spend:,.2f}")
print(f"Total Revenue       : ₹{total_revenue:,.2f}")
print(f"Overall CTR         : {overall_ctr:.2f}%")
print(f"Overall CPC         : ₹{overall_cpc:.2f}")
print(f"Conversion Rate     : {overall_conversion_rate:.2f}%")
print(f"CPA                 : ₹{overall_cpa:.2f}")
print(f"ROAS                : {overall_roas:.2f}")

# Save KPI dataset
df.to_csv(
    "data/marketing_kpi_data.csv",
    index=False
)

print("\nKPI dataset saved successfully!")
print("data/marketing_kpi_data.csv")