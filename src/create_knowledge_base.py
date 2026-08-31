import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# MARKETING KNOWLEDGE
# ==========================================

documents = [

    """
    CTR (Click Through Rate) measures the percentage
    of people who click an advertisement after seeing it.

    A low CTR may indicate weak ad creatives,
    poor headlines, irrelevant messaging,
    or weak calls-to-action.
    """,

    """
    Conversion Rate measures the percentage of users
    who complete a desired action after clicking an advertisement.

    A low conversion rate may indicate landing page problems,
    poor offers, complicated forms, or mismatch between
    advertisement messaging and landing page content.
    """,

    """
    ROAS (Return on Ad Spend) measures how much revenue
    is generated for every unit of advertising spend.

    A low ROAS indicates that the campaign is generating
    relatively low revenue compared with its advertising cost.
    """,

    """
    CPA (Cost Per Acquisition) measures the average cost
    required to acquire one customer or conversion.

    A high CPA may indicate inefficient targeting,
    expensive advertising traffic, low conversion rates,
    or poor campaign optimization.
    """,

    """
    Audience targeting affects campaign performance.

    If advertisements are shown to an audience that is
    not highly relevant to the product or service,
    CTR and conversion rate may decrease while CPA increases.
    """,

    """
    Ad creatives have a strong influence on CTR.

    Testing different images, videos, headlines,
    messages and calls-to-action can help improve
    user engagement and click-through rate.
    """,

    """
    Landing page experience can strongly affect conversion rate.

    A slow page, unclear message, poor mobile experience,
    complicated navigation or weak call-to-action
    can reduce conversions.
    """,

    """
    Offer quality can influence both CTR and conversion rate.

    A relevant and attractive offer can increase user interest,
    while a weak or unclear offer may reduce conversions.
    """,

    """
    When CTR is high but conversion rate is low,
    the advertisement may be attracting users successfully
    but the landing page, offer or conversion process
    may not meet user expectations.
    """,

    """
    When CTR is low and conversion rate is also low,
    marketers should first review ad creatives,
    audience targeting and message relevance.
    """,

    """
    When ROAS is high and CPA is low,
    the campaign is generally more efficient.

    Such campaigns may have potential for gradual budget scaling,
    while monitoring performance stability.
    """,

    """
    Marketing campaign optimization should consider
    multiple KPIs together rather than relying on
    a single metric.

    CTR, conversion rate, ROAS and CPA should be
    evaluated together to understand campaign health.
    """
]


# ==========================================
# CONNECT TO CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path="data/chroma_db"
)


# ==========================================
# CREATE COLLECTION
# ==========================================

collection = client.get_or_create_collection(
    name="marketing_knowledge"
)


# ==========================================
# CREATE EMBEDDINGS
# ==========================================

embeddings = model.encode(
    documents
).tolist()


# ==========================================
# ADD KNOWLEDGE TO DATABASE
# ==========================================

collection.upsert(
    ids=[
        f"marketing_{i}"
        for i in range(len(documents))
    ],
    documents=documents,
    embeddings=embeddings
)


# ==========================================
# SUCCESS MESSAGE
# ==========================================

print("==========================================")
print("MarketMind AI Knowledge Base Created!")
print("==========================================")
print(f"Total Documents: {len(documents)}")
print("Collection: marketing_knowledge")
print("Database: data/chroma_db")
print("==========================================")