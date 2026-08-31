import os
import pandas as pd
from openai import OpenAI


# ==========================================
# MARKETMIND AI - LLM ENGINE
# ==========================================

print("\n========================================")
print("   MARKETMIND AI - LLM ENGINE")
print("========================================")


# ==========================================
# 1. CHECK API KEY
# ==========================================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("\nERROR: OPENAI_API_KEY not found.")
    print("Please set your API key before running this file.")
    exit()


client = OpenAI(api_key=api_key)


# ==========================================
# 2. LOAD MARKETING DATA
# ==========================================

df = pd.read_csv(
    "data/marketing_kpi_data.csv"
)


# ==========================================
# 3. CAMPAIGN ANALYSIS
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
# 4. CALCULATE KPIs
# ==========================================

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


campaign_data["ROAS"] = (
    campaign_data["Revenue"]
    /
    campaign_data["Spend"]
)


campaign_data["CPA"] = (
    campaign_data["Spend"]
    /
    campaign_data["Conversions"]
)


# ==========================================
# 5. BEST & WORST CAMPAIGN
# ==========================================

best_campaign = campaign_data.loc[
    campaign_data["ROAS"].idxmax()
]


worst_campaign = campaign_data.loc[
    campaign_data["ROAS"].idxmin()
]


# ==========================================
# 6. LOAD RAG KNOWLEDGE
# ==========================================

try:

    with open(
        "data/ai_insight.txt",
        "r",
        encoding="utf-8"
    ) as file:

        rag_context = file.read()

except FileNotFoundError:

    rag_context = (
        "No additional marketing knowledge "
        "was found."
    )


# ==========================================
# 7. CREATE LLM PROMPT
# ==========================================

prompt = f"""
You are MarketMind AI, an intelligent
marketing analytics assistant.

Analyze the following marketing performance
data and provide a professional business insight.

BEST CAMPAIGN
Campaign: {best_campaign["Campaign"]}
ROAS: {best_campaign["ROAS"]:.2f}
CTR: {best_campaign["CTR"]:.2f}%
Conversion Rate: {best_campaign["Conversion_Rate"]:.2f}%
CPA: ₹{best_campaign["CPA"]:.2f}

WORST CAMPAIGN
Campaign: {worst_campaign["Campaign"]}
ROAS: {worst_campaign["ROAS"]:.2f}
CTR: {worst_campaign["CTR"]:.2f}%
Conversion Rate: {worst_campaign["Conversion_Rate"]:.2f}%
CPA: ₹{worst_campaign["CPA"]:.2f}

MARKETING KNOWLEDGE
{rag_context}

Your response must contain:

1. Overall Performance
2. Why did this happen?
3. What is likely to happen next?
4. What should the marketer do?
5. Key risks
6. Short executive summary

Use simple professional business language.

Do not invent data.
Use the provided marketing data as the
main evidence.
"""


# ==========================================
# 8. CALL LLM
# ==========================================

print("\nGenerating AI business analysis...")


response = client.responses.create(
    model="gpt-5-mini",
    input=prompt
)


ai_response = response.output_text


# ==========================================
# 9. DISPLAY RESULT
# ==========================================

print("\n========================================")
print("       MARKETMIND AI INSIGHT")
print("========================================\n")

print(ai_response)


# ==========================================
# 10. SAVE RESULT
# ==========================================

with open(
    "data/llm_insight.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(ai_response)


print("\n========================================")
print("LLM Insight generated successfully!")
print("========================================")

print("\nSaved:")
print("data/llm_insight.txt")