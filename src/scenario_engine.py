import pandas as pd


# ==========================================
# MARKETMIND AI - WHAT IF SCENARIO ENGINE
# ==========================================

print("\n========================================")
print("   MARKETMIND AI - SCENARIO ENGINE")
print("========================================")


# ==========================================
# 1. LOAD MARKETING DATA
# ==========================================

df = pd.read_csv(
    "data/marketing_kpi_data.csv"
)


# ==========================================
# 2. CAMPAIGN ANALYSIS
# ==========================================

campaign_data = (
    df.groupby("Campaign")
    .agg(
        Spend=("Spend", "sum"),
        Revenue=("Revenue", "sum"),
        Conversions=("Conversions", "sum"),
        Clicks=("Clicks", "sum"),
        Impressions=("Impressions", "sum")
    )
    .reset_index()
)


# ==========================================
# 3. CALCULATE CURRENT ROAS
# ==========================================

campaign_data["ROAS"] = (
    campaign_data["Revenue"]
    /
    campaign_data["Spend"]
)


# ==========================================
# 4. SCENARIO
# ==========================================

spend_increase = 20


campaign_data["Scenario_Spend"] = (
    campaign_data["Spend"]
    *
    (1 + spend_increase / 100)
)


# ==========================================
# 5. ESTIMATE SCENARIO REVENUE
# ==========================================

campaign_data["Scenario_Revenue"] = (
    campaign_data["Scenario_Spend"]
    *
    campaign_data["ROAS"]
)


# ==========================================
# 6. REVENUE CHANGE
# ==========================================

campaign_data["Revenue_Change"] = (
    campaign_data["Scenario_Revenue"]
    -
    campaign_data["Revenue"]
)


campaign_data["Revenue_Change_Percent"] = (
    campaign_data["Revenue_Change"]
    /
    campaign_data["Revenue"]
) * 100


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n===== WHAT-IF SCENARIO =====")

print(
    f"\nScenario: Increase marketing spend by "
    f"{spend_increase}%"
)


for _, row in campaign_data.iterrows():

    print("\n----------------------------------------")

    print(
        f"Campaign: {row['Campaign']}"
    )

    print(
        f"Current Spend: ₹{row['Spend']:,.2f}"
    )

    print(
        f"Scenario Spend: ₹{row['Scenario_Spend']:,.2f}"
    )

    print(
        f"Current Revenue: ₹{row['Revenue']:,.2f}"
    )

    print(
        f"Estimated Revenue: "
        f"₹{row['Scenario_Revenue']:,.2f}"
    )

    print(
        f"Estimated Revenue Change: "
        f"{row['Revenue_Change_Percent']:.2f}%"
    )


# ==========================================
# 8. SAVE RESULTS
# ==========================================

campaign_data.to_csv(
    "data/scenario_results.csv",
    index=False
)


print("\n========================================")
print("Scenario analysis completed!")
print("========================================")

print("\nSaved:")
print("data/scenario_results.csv")