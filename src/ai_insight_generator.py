import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

print("\n========================================")
print("   MARKETMIND AI - AI INSIGHT GENERATOR")
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
# RAG KNOWLEDGE RETRIEVAL
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_collection(
    name="marketing_knowledge"
)


# ==========================================
# SEARCH MARKETING KNOWLEDGE
# ==========================================

query = """
Why is the performance of the marketing campaigns
different and what actions should a marketer take
based on ROAS, CTR, conversion rate and CPA?
"""

query_embedding = embedding_model.encode(
    [query]
).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)

knowledge_context = "\n\n".join(
    results["documents"][0]
)


# ==========================================
# 4. FIND BEST AND WORST CAMPAIGNS
# ==========================================

best_campaign = campaign_data.loc[
    campaign_data["ROAS"].idxmax()
]

worst_campaign = campaign_data.loc[
    campaign_data["ROAS"].idxmin()
]


# ==========================================
# 5. OVERALL METRICS
# ==========================================

total_spend = campaign_data["Spend"].sum()

total_revenue = campaign_data["Revenue"].sum()

total_conversions = campaign_data[
    "Conversions"
].sum()

overall_roas = (
    total_revenue
    /
    total_spend
)


# ==========================================
# 6. GENERATE BUSINESS INSIGHT
# ==========================================

insight = f"""
MARKETMIND AI BUSINESS INSIGHT

Overall Performance
-------------------
Total Marketing Spend: ₹{total_spend:,.2f}

Total Revenue: ₹{total_revenue:,.2f}

Total Conversions: {total_conversions:,.0f}

Overall ROAS: {overall_roas:.2f}


Best Performing Campaign
------------------------
Campaign: {best_campaign["Campaign"]}

ROAS: {best_campaign["ROAS"]:.2f}

CTR: {best_campaign["CTR"]:.2f}%

Conversion Rate:
{best_campaign["Conversion_Rate"]:.2f}%


Lowest Performing Campaign
--------------------------
Campaign: {worst_campaign["Campaign"]}

ROAS: {worst_campaign["ROAS"]:.2f}

CTR: {worst_campaign["CTR"]:.2f}%

Conversion Rate:
{worst_campaign["Conversion_Rate"]:.2f}%


Business Interpretation
-----------------------

The campaign with the highest ROAS is
{best_campaign["Campaign"]}. This campaign
is currently generating the strongest return
relative to marketing spend.

The campaign with the lowest ROAS is
{worst_campaign["Campaign"]}. This campaign
should be reviewed for targeting, creative
quality, audience relevance and conversion
performance.

Recommended Business Direction
------------------------------

Focus more budget on campaigns with
consistently strong ROAS.

Review campaigns with weak ROAS before
increasing their spending.

Monitor CTR and conversion rate together
because strong traffic does not always
result in conversions.

RAG MARKETING KNOWLEDGE
-----------------------

{knowledge_context}
"""


# ==========================================
# 7. DISPLAY
# ==========================================

print(insight)


# ==========================================
# 8. SAVE INSIGHT
# ==========================================

with open(
    "data/ai_insight.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(insight)


print("\n========================================")
print("AI Insight generated successfully!")
print("========================================")

print("\nSaved:")
print("data/ai_insight.txt")