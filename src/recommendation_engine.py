import pandas as pd


print("\n========================================")
print("   MARKETMIND AI - RECOMMENDATION ENGINE")
print("========================================")


# ==========================================
# 1. LOAD MARKETING DATA
# ==========================================

df = pd.read_csv(
    "data/marketing_kpi_data.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)


# ==========================================
# 2. CAMPAIGN ANALYSIS
# ==========================================

campaign_data = (
    df.groupby("Campaign")
    .agg(
        Impressions=("Impressions", "sum"),
        Clicks=("Clicks", "sum"),
        Conversions=("Conversions", "sum"),
        Spend=("Spend", "sum"),
        Revenue=("Revenue", "sum")
    )
    .reset_index()
)


# ==========================================
# 3. CALCULATE KPIs
# ==========================================

campaign_data["ROAS"] = (
    campaign_data["Revenue"]
    /
    campaign_data["Spend"]
)

campaign_data["CTR"] = (
    campaign_data["Clicks"]
    /
    campaign_data["Impressions"]
) * 100

campaign_data["Conversion_Rate"] = (
    campaign_data["Conversions"]
    /
    campaign_data["Clicks"]
) * 100

campaign_data["CPA"] = (
    campaign_data["Spend"]
    /
    campaign_data["Conversions"]
)

# SHOW ACTUAL KPI VALUES
print("\nCampaign KPI Values:")
print("----------------------------------------")

print(
    campaign_data[
        [
            "Campaign",
            "ROAS",
            "CTR",
            "Conversion_Rate",
            "CPA"
        ]
    ].round(2).to_string(index=False)
)

# ==========================================
# 4. CAMPAIGN HEALTH SCORE
# ==========================================

def calculate_health_score(row):

    score = 0

    # ROAS Score - 40 points
    if row["ROAS"] >= 180:
        score += 40
    elif row["ROAS"] >= 170:
        score += 30
    elif row["ROAS"] >= 160:
        score += 20
    else:
        score += 10

    # CTR Score - 20 points
    if row["CTR"] >= 4.5:
        score += 20
    elif row["CTR"] >= 4.0:
        score += 15
    elif row["CTR"] >= 3.5:
        score += 10
    else:
        score += 5

    # Conversion Rate Score - 25 points
    if row["Conversion_Rate"] >= 8.5:
        score += 25
    elif row["Conversion_Rate"] >= 8:
        score += 20
    elif row["Conversion_Rate"] >= 7:
        score += 10
    else:
        score += 5

    # CPA Score - 15 points
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
print("\nCampaign Health Score:")
print("----------------------------------------")

print(
    campaign_data[
        [
            "Campaign",
            "Health_Score",
            "Health_Status"
        ]
    ].to_string(index=False)
)
# ==========================================
# 4. CREATE RECOMMENDATIONS
# ==========================================

recommendations = []


for _, row in campaign_data.iterrows():

    campaign = row["Campaign"]

    roas = row["ROAS"]
    ctr = row["CTR"]
    conversion_rate = row["Conversion_Rate"]
    cpa = row["CPA"]


    # ======================================
    # HIGH PRIORITY CONDITIONS
    # ======================================
     
     # ======================================
    # HIGH PRIORITY
    # ======================================

    if roas < 165 or conversion_rate < 8:

        recommendations.append({
            "Campaign": campaign,
            "Priority": "High",
            "Recommendation":
                "Campaign performance needs immediate attention. "
                "Review audience targeting, ad creative, landing "
                "page experience and conversion performance before "
                "increasing the budget."
        })

    # ======================================
    # MEDIUM PRIORITY
    # ======================================

    elif roas < 175 or cpa > 10.5:

        recommendations.append({
            "Campaign": campaign,
            "Priority": "Medium",
            "Recommendation":
                "Campaign performance is below the stronger "
                "campaigns. Review targeting and creative "
                "performance and optimize the campaign before "
                "scaling the budget."
        })

    elif ctr < 2:

        recommendations.append({
            "Campaign": campaign,
            "Priority": "Medium",
            "Recommendation":
                "CTR is relatively low. Test new ad creatives, "
                "headlines and stronger calls-to-action to "
                "increase engagement."
        })

    # ======================================
    # LOW PRIORITY
    # ======================================

    else:

        recommendations.append({
            "Campaign": campaign,
            "Priority": "Low",
            "Recommendation":
                "Campaign performance is strong. Continue "
                "monitoring ROAS, CTR and conversion rate while "
                "gradually testing budget optimization."
        })
     
      
# ==========================================
# 5. CREATE DATAFRAME
# ==========================================

recommendations_df = pd.DataFrame(
    recommendations
)


# ==========================================
# 6. SAVE RECOMMENDATIONS
# ==========================================

recommendations_df.to_csv(
    "data/recommendations.csv",
    index=False
)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\nGenerated Recommendations:")
print("----------------------------------------")

print(
    recommendations_df.to_string(
        index=False
    )
)


print("\n========================================")
print("Recommendation Engine completed!")
print("========================================")

print(
    "\nSaved:"
)

print(
    "data/recommendations.csv"
)