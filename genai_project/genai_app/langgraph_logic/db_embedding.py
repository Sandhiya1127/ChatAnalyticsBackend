
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

# Use BAAI bge-small or bge-large
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",  # or "bge-large-en-v1.5"
    model_kwargs={"device": "cpu"}  # or "cuda" if you have GPU
)

def embed_schema_for_user(user_id, db_id, schema_text):
    path = f"vectorstores/{user_id}/{db_id}"
    os.makedirs(path, exist_ok=True)
    vectorstore = FAISS.from_documents([Document(page_content=schema_text)], embedding_model)
    vectorstore.save_local(path)

def get_context(question, user_id, db_id):
    
    path = f"vectorstores/{user_id}/{db_id}"
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vectorstore = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(question, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

