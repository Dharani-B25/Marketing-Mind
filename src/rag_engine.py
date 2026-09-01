import os
import chromadb
from sentence_transformers import SentenceTransformer


print("\n========================================")
print("   MARKETMIND AI - RAG ENGINE")
print("========================================")


# ==========================================
# 1. FIND PROJECT ROOT
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

KNOWLEDGE_PATH = os.path.join(
    BASE_DIR,
    "knowledge",
    "marketing_knowledge.txt"
)


# ==========================================
# 2. LOAD KNOWLEDGE
# ==========================================

with open(
    KNOWLEDGE_PATH,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


# ==========================================
# 3. SPLIT KNOWLEDGE INTO CHUNKS
# ==========================================

documents = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]


print(f"\nKnowledge chunks: {len(documents)}")


# ==========================================
# 4. LOAD EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# 5. CREATE IN-MEMORY VECTOR DATABASE
# ==========================================

client = chromadb.Client()


collection = client.get_or_create_collection(
    name="marketing_knowledge"
)


# ==========================================
# 6. CREATE EMBEDDINGS
# ==========================================

embeddings = embedding_model.encode(
    documents
).tolist()


# ==========================================
# 7. STORE KNOWLEDGE
# ==========================================

ids = [
    f"marketing_{i}"
    for i in range(len(documents))
]


collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)


print("\nKnowledge successfully stored!")


# ==========================================
# 8. TEST SEARCH
# ==========================================

query = "Why can conversion rate decrease?"


query_embedding = embedding_model.encode(
    [query]
).tolist()


results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)


print("\n========================================")
print("RAG SEARCH RESULT")
print("========================================")


for i, document in enumerate(
    results["documents"][0],
    start=1
):

    print(f"\nResult {i}:")
    print(document)


print("\n========================================")
print("RAG ENGINE COMPLETED")
print("========================================")