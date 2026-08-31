import pandas as pd

# Load KPI data
df = pd.read_csv("data/marketing_kpi_data.csv")

print("\n========================================")
print("   MARKETMIND AI - ROOT CAUSE ANALYSIS")
print("========================================")


# -----------------------------------------
# 1. Create daily performance summary
# -----------------------------------------

daily = df.groupby("Date").agg(
    Impressions=("Impressions", "sum"),
    Clicks=("Clicks", "sum"),
    Conversions=("Conversions", "sum"),
    Spend=("Spend", "sum"),
    Revenue=("Revenue", "sum")
).reset_index()


# -----------------------------------------
# 2. Calculate daily KPIs
# -----------------------------------------

daily["CTR"] = (
    daily["Clicks"] / daily["Impressions"]
) * 100

daily["CPC"] = (
    daily["Spend"] / daily["Clicks"]
)

daily["Conversion_Rate"] = (
    daily["Conversions"] / daily["Clicks"]
) * 100

daily["CPA"] = (
    daily["Spend"] / daily["Conversions"]
)

daily["ROAS"] = (
    daily["Revenue"] / daily["Spend"]
)


# -----------------------------------------
# 3. Compare first half vs second half
# -----------------------------------------

mid_point = len(daily) // 2

first_half = daily.iloc[:mid_point]
second_half = daily.iloc[mid_point:]


# -----------------------------------------
# 4. Calculate average KPIs
# -----------------------------------------

first_ctr = first_half["CTR"].mean()
second_ctr = second_half["CTR"].mean()

first_cpc = first_half["CPC"].mean()
second_cpc = second_half["CPC"].mean()

first_conversion = first_half["Conversion_Rate"].mean()
second_conversion = second_half["Conversion_Rate"].mean()

first_cpa = first_half["CPA"].mean()
second_cpa = second_half["CPA"].mean()

first_roas = first_half["ROAS"].mean()
second_roas = second_half["ROAS"].mean()


# -----------------------------------------
# 5. Calculate percentage changes
# -----------------------------------------

ctr_change = (
    (second_ctr - first_ctr) / first_ctr
) * 100

cpc_change = (
    (second_cpc - first_cpc) / first_cpc
) * 100

conversion_change = (
    (second_conversion - first_conversion)
    / first_conversion
) * 100

cpa_change = (
    (second_cpa - first_cpa) / first_cpa
) * 100

roas_change = (
    (second_roas - first_roas) / first_roas
) * 100


# -----------------------------------------
# 6. Display comparison
# -----------------------------------------

print("\n===== PERFORMANCE COMPARISON =====\n")

print(f"CTR Change              : {ctr_change:.2f}%")
print(f"CPC Change              : {cpc_change:.2f}%")
print(f"Conversion Rate Change : {conversion_change:.2f}%")
print(f"CPA Change              : {cpa_change:.2f}%")
print(f"ROAS Change             : {roas_change:.2f}%")


# -----------------------------------------
# 7. Root Cause Detection
# -----------------------------------------

print("\n===== ROOT CAUSE ANALYSIS =====\n")

reasons = []


# CTR
if ctr_change < -10:
    reasons.append(
        "CTR dropped significantly. Ad engagement may have decreased."
    )

elif ctr_change > 10:
    reasons.append(
        "CTR increased significantly. Ad engagement improved."
    )


# CPC
if cpc_change > 10:
    reasons.append(
        "CPC increased significantly. Advertising became more expensive."
    )

elif cpc_change < -10:
    reasons.append(
        "CPC decreased. Traffic became more cost-efficient."
    )


# Conversion Rate
if conversion_change < -10:
    reasons.append(
        "Conversion rate decreased. Landing page or audience quality may be affecting conversions."
    )

elif conversion_change > 10:
    reasons.append(
        "Conversion rate improved. Campaign traffic is converting better."
    )


# CPA
if cpa_change > 10:
    reasons.append(
        "CPA increased. Acquiring customers became more expensive."
    )

elif cpa_change < -10:
    reasons.append(
        "CPA decreased. Customer acquisition became more efficient."
    )


# ROAS
if roas_change < -10:
    reasons.append(
        "ROAS decreased. Marketing return became weaker."
    )

elif roas_change > 10:
    reasons.append(
        "ROAS improved. Marketing return became stronger."
    )


# -----------------------------------------
# 8. Display detected reasons
# -----------------------------------------

if len(reasons) == 0:

    print(
        "No major performance changes were detected."
    )

else:

    for i, reason in enumerate(reasons, start=1):

        print(f"{i}. {reason}")


# -----------------------------------------
# 9. Save daily analysis
# -----------------------------------------

daily.to_csv(
    "data/daily_performance.csv",
    index=False
)


print("\n========================================")
print("Root Cause Analysis completed!")
print("========================================")

print("\nSaved:")
print("data/daily_performance.csv")