import os
import uuid
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Load same embedding model you use in your app
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"}
)

# Path to your persisted Chroma DB
PERSIST_DIR = "db/chroma_feedback"

# Connect
vector_db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)

def cleanup_docs():
    print("🔍 Fetching all docs from Chroma...")
    # Get all docs (be careful: Chroma doesn’t support .all() directly, so we trick with empty search)
    docs = vector_db.similarity_search("", k=1000)  # adjust k if you have >1000 docs
    print(f"📦 Found {len(docs)} docs")

    updated, skipped = 0, 0
    for doc in docs:
        meta = doc.metadata
        question = doc.page_content.strip()

        if "doc_id" in meta and meta["doc_id"]:
            skipped += 1
            continue  # already good

        session_id = meta.get("session_id", str(uuid.uuid4()))
        doc_id = f"{session_id}:{question.lower()}"

        print(f"🛠 Fixing missing doc_id for: {question[:50]}... → {doc_id}")

        # Delete old doc (Chroma requires ID to delete, so we use similarity workaround)
        try:
            vector_db.delete([doc_id])  # safe, if already existed
        except Exception:
            pass

        # Re-insert with fixed doc_id
        vector_db.add_texts(
            texts=[question],
            metadatas=[{
                **meta,
                "doc_id": doc_id
            }],
            ids=[doc_id]
        )
        updated += 1

    print(f"✅ Cleanup done: {updated} updated, {skipped} already had doc_id")

if __name__ == "__main__":
    cleanup_docs()
