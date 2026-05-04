import json
from datetime import datetime
import os
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
DOCS_PATH = "docs"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3
SEARCH_K = 10  # 🔥 deeper retrieval
USE_LLM = False
MAX_CONTEXT_CHUNKS = 2
RELEVANCE_THRESHOLD = 1.5

# 🔐 USERS (RBAC)
USERS = {
    "ujjwal": "employee",
    "kamal": "manager",
    "admin": "admin"
}

# -----------------------------
# INIT
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = OpenAI()

def log_query(data):
    with open("logs.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")

# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
def load_documents():
    return [
        {
            "doc_id": "doc1.txt",
            "text": open("docs/doc1.txt", encoding="utf-8").read(),
            "allowed_roles": ["manager", "admin"]  # finance restricted
        },
        {
            "doc_id": "doc2.txt",
            "text": open("docs/doc2.txt", encoding="utf-8").read(),
            "allowed_roles": ["employee", "manager", "admin"]  # open access
        }
    ]

# -----------------------------
# CHUNKING (WITH OVERLAP)
# -----------------------------
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap

    return chunks

# -----------------------------
# BUILD INDEX + METADATA
# -----------------------------
def build_index(docs):
    all_chunks = []
    metadata = []

    for doc in docs:
        chunks = chunk_text(doc["text"])

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)

            metadata.append({
                "doc_id": doc["doc_id"],
                "chunk_id": i,
                "text": chunk,
                "allowed_roles": doc["allowed_roles"]
            })

    embeddings = model.encode(all_chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, metadata

# -----------------------------
# QUERY LLM
# -----------------------------
def query_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Answer ONLY using provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content

# -----------------------------
# MAIN
# -----------------------------
def main():
    docs = load_documents()
    index, metadata = build_index(docs)

    # 🔐 USER LOGIN
    username = input("Enter username: ")
    user_role = USERS.get(username)

    if not user_role:
        print("Invalid user")
        return

    print(f"Logged in as {username} ({user_role})")
    print("\nRAG system ready! Type your question:\n")

    while True:
        query = input(">> ")
        
        raw_texts = []

        # Step 1: Embed query
        query_embedding = model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        # Step 2: Retrieve more candidates
        k = min(SEARCH_K, len(metadata))
        distances, indices = index.search(query_embedding, k)

        # -----------------------------
        # 🔍 RAW RETRIEVAL (DEBUG)
        # -----------------------------
        raw_results = []

        print("\n--- RAW RETRIEVAL (Before RBAC) ---\n")

        for i, idx in enumerate(indices[0]):
            score = distances[0][i]

            if score > 1e10:
                continue

            meta = metadata[idx]
            raw_texts.append(meta["text"])

            print(f"[Rank {i}] (Score: {score:.4f})")
            print(f"Doc: {meta['doc_id']} | Allowed Roles: {meta['allowed_roles']}")
            print(meta["text"])
            print("-" * 40)

            raw_results.append(meta)

        # -----------------------------
        # 🔐 RBAC FILTER
        # -----------------------------
        authorized_chunks = []

        for i, meta in enumerate(raw_results):
            score = distances[0][i]

            if user_role not in meta["allowed_roles"]:
                continue

            authorized_chunks.append((meta["text"], score))

            if len(authorized_chunks) >= TOP_K:
                break
            
        authorized_chunks = [
            text for (text, score) in authorized_chunks
            if score < RELEVANCE_THRESHOLD
        ]

        # -----------------------------
        # 🔍 AFTER RBAC
        # -----------------------------
        print("\n--- AFTER RBAC FILTER ---\n")

        if not authorized_chunks:
            print("No accessible data found for your role.")
            print("\n" + "=" * 50 + "\n")
            continue

        for chunk in authorized_chunks:
            print(chunk)
            print("-" * 40)
            
        log_data = {
            "user": username,
            "role": user_role,
            "query": query,
            "raw_results": [meta["text"] for meta in raw_results], 
            "authorized_results": authorized_chunks,
            "timestamp": datetime.utcnow().isoformat()
        }
        log_query(log_data)

        # -----------------------------
        # CONTEXT BUILD
        # -----------------------------
        context = "\n".join(authorized_chunks[:MAX_CONTEXT_CHUNKS])

        prompt = f"""
Answer ONLY from the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

        # -----------------------------
        # LLM (optional)
        # -----------------------------
        if USE_LLM:
            answer = query_llm(prompt)
            print("\nAnswer:\n", answer)
        else:
            print("\n(Context mode ON — LLM not called)\n")

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
