# rag_sql_dashboard/views.py

import os
import json
import base64
import io
import pandas as pd
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sqlalchemy import create_engine, text, inspect
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
VECTOR_DIM = 768
OLLAMA_URL = "http://localhost:11500/api/generate"
engine_cache = {}

# Efficient DB connection caching
def get_engine(db_config):
    key = json.dumps(db_config)
    if key in engine_cache:
        return engine_cache[key]

    db_url = f"{db_config['dialect']}://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    engine = create_engine(db_url)
    engine_cache[key] = engine
    return engine

class VectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(VECTOR_DIM)
        self.data = []

    def add(self, texts):
        vectors = embedding_model.encode(texts)
        self.index.add(np.array(vectors).astype('float32'))
        self.data.extend(texts)

    def query(self, text, k=3):
        if not self.data:
            return []
        vector = embedding_model.encode([text])
        D, I = self.index.search(np.array(vector).astype('float32'), k)
        return [self.data[i] for i in I[0] if i < len(self.data)]

store = VectorStore()

def read_file(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    elif name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return None

def generate_chart(df):
    plt.figure()
    df.plot(kind='bar', x=df.columns[0], y=df.columns[1])
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def extract_sql(text):
    if "```sql" in text:
        return text.split("```sql")[1].split("```", 1)[0].strip()
    elif "SELECT" in text.upper():
        return text.split(";")[0] + ";"
    return text

@csrf_exempt
def upload_file(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid method'}, status=405)

    uploaded_file = request.FILES.get('file')
    if uploaded_file:
        allowed = ['.csv', '.xls', '.xlsx', '.pdf']
        if not any(uploaded_file.name.endswith(ext) for ext in allowed):
            return JsonResponse({'error': 'Unsupported file type'}, status=400)

        content = read_file(uploaded_file)
        if isinstance(content, pd.DataFrame):
            for col in content.columns:
                sample = content[[col]].dropna().astype(str).head(2).to_string(index=False)
                store.add([f"Column `{col}` sample:\n{sample}"])
        elif isinstance(content, str):
            store.add([content])
        return JsonResponse({'status': 'File embedded into vector store'})

    db_config = json.loads(request.POST.get('db_config', '{}'))
    if db_config:
        engine = get_engine(db_config)
        insp = inspect(engine)
        table_names = insp.get_table_names()
        for table in table_names:
            df = pd.read_sql(f"SELECT * FROM {table} LIMIT 5", engine)
            store.add([f"Table `{table}` columns: {', '.join(df.columns)}\nSample:\n{df.head(2).to_string(index=False)}"])
        return JsonResponse({'status': 'DB schema embedded'})

    return JsonResponse({'error': 'No file or DB config provided'}, status=400)

@csrf_exempt
def ask(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid method'}, status=405)

    data = json.loads(request.body)
    question = data.get('question', '').strip()
    db_config = data.get('db_config')

    if not question:
        return JsonResponse({'error': 'Missing question'}, status=400)

    # If no vector store data and no db connected, treat as natural chat
    if not store.data and not db_config:
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": "qwen2.5:7b",
                "prompt": question,
                "stream": False
            })
            response.raise_for_status()
            return JsonResponse({'answer': response.json()['response']})
        except Exception as e:
            return JsonResponse({'error': f'LLM response failed: {str(e)}'}, status=500)

    store.add([question])

    # Determine if the user is asking for SQL explicitly
    if any(word in question.lower() for word in ["sql", "query", "select", "table", "columns"]):
        rag_context = "\n\n".join(store.query(question, k=3))
        prompt = f"""
You are a Qwen2.5-7B SQL assistant.
Use the below context to generate a valid SQL query for the user's question.

Context:
{rag_context}

Question:
{question}

SQL:
"""
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False
            })
            response.raise_for_status()
            sql = extract_sql(response.json()['response'])
            answer_text = response.json()['response']
        except Exception as e:
            return JsonResponse({'error': f'LLM request failed: {str(e)}'}, status=500)

        if not db_config:
            return JsonResponse({'answer': answer_text})

        if not sql.strip().lower().startswith("select"):
            return JsonResponse({"sql": sql, "answer": answer_text, "data": []})

        try:
            engine = get_engine(db_config)
            df = pd.read_sql(text(sql), engine)
            chart = generate_chart(df)
            return JsonResponse({
                'sql': sql,
                'answer': answer_text + "\n\n" + df.head().to_markdown(index=False),
                'data': df.to_dict(orient="records"),
                'chart_type': 'column',
                'series_data': df.iloc[:, 1].tolist(),
                'x_categories': df.iloc[:, 0].tolist()
            })
        except Exception as e:
            return JsonResponse({'error': f'SQL execution failed: {str(e)}'}, status=500)

    # If it's a general question, just chat
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "qwen2.5:7b",
            "prompt": question,
            "stream": False
        })
        response.raise_for_status()
        return JsonResponse({'answer': response.json()['response']})
    except Exception as e:
        return JsonResponse({'error': f'LLM response failed: {str(e)}'}, status=500)


