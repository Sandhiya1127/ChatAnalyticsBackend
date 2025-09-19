import re
from datetime import datetime
import os
import uuid

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Load same embedding model you use in your app
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"}
)

# 🔹 Path to your existing Chroma DB
persist_dir = "chroma/chroma_feedback"

# --- Load Chroma ---
vector_db = Chroma(
    persist_directory=persist_dir,
    embedding_function=embedding_model
)

def normalize_question(q: str) -> str:
    """Lowercase, strip, collapse spaces, remove trailing punctuation."""
    if not q:
        return ""
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)   # collapse multiple spaces
    q = re.sub(r"[?!.;,]+$", "", q)  # remove trailing ? . , etc.
    return q

def migrate_chroma():
    print("🚀 Starting migration...")

    # fetch all docs from chroma
    try:
        all_data = vector_db._collection.get(include=["documents", "metadatas"])
    except Exception as e:
        print(f"❌ Failed to fetch from Chroma: {e}")
        return

    docs = all_data.get("documents", [])
    metas = all_data.get("metadatas", [])
    ids = all_data.get("ids", [])  # always returned

    print(f"📦 Found {len(docs)} docs in collection")

    # dedup map
    new_store = {}

    for doc, meta, doc_id in zip(docs, metas, ids):
        q_norm = normalize_question(doc)
        new_id = f"global:{q_norm}" if str(doc_id).startswith("global:") else f"{meta.get('session_id', 'unknown')}:{q_norm}"

        # If duplicate already exists, decide which one to keep
        if new_id in new_store:
            old_meta = new_store[new_id]["meta"]
            # keep the one with positive feedback OR latest timestamp
            if meta.get("feedback") == "positive" or meta.get("timestamp", "") > old_meta.get("timestamp", ""):
                new_store[new_id] = {"doc": q_norm, "meta": meta}
        else:
            new_store[new_id] = {"doc": q_norm, "meta": meta}

    print(f"✅ Deduplicated down to {len(new_store)} docs")

    # Clear existing collection only if docs exist
    if ids:
        vector_db._collection.delete(ids=ids)
        print("🗑️ Cleared old collection")
    else:
        print("⚠️ No docs to delete in collection")

    # Reinsert normalized & deduped docs
    if new_store:
        vector_db.add_texts(
            texts=[item["doc"] for item in new_store.values()],
            ids=[doc_id for doc_id in new_store.keys()],
            metadatas=[item["meta"] for item in new_store.values()]
        )
        print("🎉 Migration complete — all questions normalized & deduplicated")
    else:
        print("⚠️ No docs to reinsert after deduplication")

if __name__ == "__main__":
    migrate_chroma()
