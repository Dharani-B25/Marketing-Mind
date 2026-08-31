import chromadb
from sentence_transformers import SentenceTransformer


print("\n========================================")
print("   MARKETMIND AI - RAG ENGINE")
print("========================================")


# ==========================================
# 1. LOAD KNOWLEDGE
# ==========================================

with open(
    "knowledge/marketing_knowledge.txt",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


# ==========================================
# 2. SPLIT KNOWLEDGE INTO CHUNKS
# ==========================================

documents = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]


print(f"\nKnowledge chunks: {len(documents)}")


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# 4. CREATE VECTOR DATABASE
# ==========================================

client = chromadb.PersistentClient(
    path="data/chroma_db"
)


collection = client.get_or_create_collection(
    name="marketing_knowledge"
)


# ==========================================
# 5. CREATE EMBEDDINGS
# ==========================================

embeddings = embedding_model.encode(
    documents
).tolist()


# ==========================================
# 6. STORE KNOWLEDGE
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
# 7. TEST SEARCH
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