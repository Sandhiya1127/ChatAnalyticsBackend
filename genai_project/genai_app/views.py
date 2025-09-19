
import os
import json
import uuid
import traceback
import re
import base64
from io import BytesIO
from datetime import datetime
import time
import tiktoken
import calendar
import difflib
from difflib import SequenceMatcher

from django.utils.timezone import now

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError, ServiceResponseError
import requests.exceptions
from .narrative import llm_generate_narrative
import json
import uuid
from urllib.parse import quote_plus
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sqlalchemy import create_engine, text
from datetime import datetime
import logging

from genai_app.utils.db_introspect import extract_schema_from_sqlalchemy
from genai_app.langgraph_logic.db_embedding import embed_schema_for_user
from .langgraph_logic.langgraph_runner import run_sql_generation_graph

session_store = {}
conversation_memory_store = {}
logger = logging.getLogger(__name__)

import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY1")


import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import logging
import requests
import traceback
from django.conf import settings

import warnings
from django.http import JsonResponse
import os, json, uuid, logging, traceback, time, requests
from datetime import datetime, timedelta

from typing import Dict, Any, List, Tuple, Optional

from urllib.parse import quote_plus
from django.views.decorators.csrf import csrf_exempt
from sqlalchemy import create_engine


import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain.prompts import PromptTemplate
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import sessionmaker

# from langchain_groq import ChatGroq
# === Config ===

from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.react.base import ReActChain
from langchain.agents.agent import AgentOutputParser
from langchain.schema.agent import AgentFinish
from langchain.schema.output_parser import OutputParserException
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from langchain_community.utilities import SQLDatabase
from langchain.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from sqlalchemy import create_engine, inspect, text
from langchain.agents import Tool, initialize_agent
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from sqlalchemy import create_engine, inspect, MetaData, text
from sqlalchemy.exc import SQLAlchemyError
# from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseRetriever
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# === Config PDF ===

from PyPDF2 import PdfReader
from django.conf import settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
# from langchain_openai import AzureChatOpenAI, ChatOpenAI
# from langchain_azure_ai.chat_models.inference import AzureAIChatCompletionsModel

GROQ_API_KEY = os.getenv("GROQ_API_KEY1")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global stores
session_store = {}
schema_context_store = {}
conversation_memory_store = {}


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Config

# State
dataframe_map = {}
vectorstore_map = {}
conversation_memory = {}
memory_cache = {}


UPLOAD_DIR = os.path.join(settings.BASE_DIR, "media/pdf_files")
VECTORSTORE_DIR = os.path.join(settings.BASE_DIR, "media/vectorstore")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
ADMIN_VECTORSTORE_NAME = "admin_base"
ADMIN_VECTORSTORE_PATH = os.path.join(VECTORSTORE_DIR, ADMIN_VECTORSTORE_NAME)
# === Config ===
logger = logging.getLogger(__name__)
session_store = {}
schema_context_store = {}  # store db schema context per session
query_preview_store = {}



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 100
 # Base chunk size
MAX_CHUNKS_PER_QUERY = 15  # Increased for large files
VECTOR_DIR = getattr(settings, 'VECTOR_DIR', './vectors')
os.makedirs(VECTOR_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY 
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1") 

MAX_CHUNKS = 5
MAX_CONTEXT_CHARS = 20000  # Maximum chunks to process per file

RERANK_TOP_K = 25  # Initial retrieval
FINAL_TOP_K = 10   # After reranking
OPENROUTER_API_KEY = "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" 
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY1")

# Global stores
dataframe_map = {}
vectorstore_map = {}
conversation_memory = {}
memory_cache = {}
metadata_cache = {}
session_store = {}


import os


# Initialize models
# Initialize models
try:
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # embedding_model = OpenAIEmbeddings(
    #     model="text-embedding-3-large",
    #     openai_api_key=OPENROUTER_API_KEY
    # )
    rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
except Exception as e:
    # logger.error(f"Failed to initialize models: {e}")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    rerank_model = None



logger = logging.getLogger(__name__)

# Token counting utility
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback estimation: ~4 chars per token
        return len(text) // 4

def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """Truncate text to fit within token limit"""
    if count_tokens(text, model) <= max_tokens:
        return text
   
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    except:
        # Fallback: character-based truncation
        estimated_chars = max_tokens * 4
        return text[:estimated_chars]

# rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")



def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters for most models)"""
    return len(text) // 4

def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit"""
    if not text:
        return text
    
    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= max_tokens:
        return text
    
    # Calculate approximate character limit
    char_limit = max_tokens * 4
    if len(text) <= char_limit:
        return text
    
    # Truncate and add indicator
    truncated = text[:char_limit - 100]  # Leave room for truncation message
    return truncated + "\n\n[Note: Content truncated due to length limits]"

def smart_context_management(semantic_context: str, memory_context: str, data_overview: str, question: str) -> dict:
    """Intelligently manage context to fit within token limits"""
    
    # Token budget allocation (leaving room for question, instructions, and response)
    MAX_TOTAL_TOKENS = 220000  # Conservative limit
    QUESTION_TOKENS = estimate_tokens(question)
    INSTRUCTIONS_TOKENS = 1000  # Approximate
    BUFFER_TOKENS = 5000  # Safety buffer
    
    AVAILABLE_TOKENS = MAX_TOTAL_TOKENS - QUESTION_TOKENS - INSTRUCTIONS_TOKENS - BUFFER_TOKENS
    
    # Prioritize allocation
    DATA_OVERVIEW_MAX = min(3000, AVAILABLE_TOKENS // 4)  # 25% max
    MEMORY_MAX = min(8000, AVAILABLE_TOKENS // 3)  # 33% max  
    SEMANTIC_MAX = AVAILABLE_TOKENS - DATA_OVERVIEW_MAX - MEMORY_MAX  # Remaining
    
    logger.info(f"Token budget - Available: {AVAILABLE_TOKENS}, Data: {DATA_OVERVIEW_MAX}, Memory: {MEMORY_MAX}, Semantic: {SEMANTIC_MAX}")
    
    # Truncate each component
    truncated_data_overview = truncate_text(data_overview, DATA_OVERVIEW_MAX)
    truncated_memory = truncate_text(memory_context, MEMORY_MAX)
    truncated_semantic = truncate_text(semantic_context, SEMANTIC_MAX)
    
    return {
        'data_overview': truncated_data_overview,
        'memory_context': truncated_memory,
        'semantic_context': truncated_semantic,
        'total_estimated_tokens': (
            estimate_tokens(truncated_data_overview) + 
            estimate_tokens(truncated_memory) + 
            estimate_tokens(truncated_semantic) + 
            QUESTION_TOKENS + INSTRUCTIONS_TOKENS
        )
    }

import re


def extract_sql(query_str: str) -> str:
    """
    Extracts the SQL code block from a mixed LLM response.
    Supports triple backticks and plain SQL in paragraphs.
    """
    # Try code block first (```sql ... ```)
    match = re.search(r"```sql(.*?)```", query_str, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try just finding the first SQL keyword line onward
    sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']
    lines = query_str.strip().splitlines()
    for i, line in enumerate(lines):
        if any(line.strip().upper().startswith(k) for k in sql_keywords):
            return "\n".join(lines[i:]).strip()

    # As fallback, return entire string
    return query_str.strip()

@dataclass
class ChunkMetadata:
    """Metadata for each chunk"""
    chunk_id: str
    source_rows: Tuple[int, int]  # start, end row indices
    columns: List[str]
    data_types: Dict[str, str]
    summary_stats: Dict[str, any]
    semantic_tags: List[str]

class EnhancedDataProcessor:
    """Enhanced data processing with semantic understanding"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def analyze_data_characteristics(self, df: pd.DataFrame) -> Dict:
        """Analyze data to determine optimal chunking strategy"""
        char_stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'data_types': df.dtypes.to_dict(),
            'null_percentages': (df.isnull().sum() / len(df)).to_dict(),
            'unique_ratios': {},
            'text_columns': [],
            'numeric_columns': [],
            'categorical_columns': []
        }
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it's text or categorical
                unique_ratio = df[col].nunique() / len(df)
                char_stats['unique_ratios'][col] = unique_ratio
                
                if unique_ratio > 0.5:  # High uniqueness = text
                    char_stats['text_columns'].append(col)
                else:  # Low uniqueness = categorical
                    char_stats['categorical_columns'].append(col)
            else:
                char_stats['numeric_columns'].append(col)
                
        return char_stats
    
    def create_semantic_summary(self, chunk_df: pd.DataFrame) -> str:
        """Create semantic summary of a chunk"""
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"Data chunk with {len(chunk_df)} rows and {len(chunk_df.columns)} columns")
        
        # Column information
        summary_parts.append(f"Columns: {', '.join(chunk_df.columns.tolist())}")
        
        # Statistical summary for numeric columns
        numeric_cols = chunk_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_summary = []
            for col in numeric_cols:
                if not chunk_df[col].isna().all():
                    mean_val = chunk_df[col].mean()
                    std_val = chunk_df[col].std()
                    min_val = chunk_df[col].min()
                    max_val = chunk_df[col].max()
                    stats_summary.append(f"{col}: mean={mean_val:.2f}, std={std_val:.2f}, range=[{min_val}, {max_val}]")
            if stats_summary:
                summary_parts.append("Numeric statistics: " + "; ".join(stats_summary))
        
        # Categorical summaries
        categorical_cols = chunk_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if not chunk_df[col].isna().all():
                top_values = chunk_df[col].value_counts().head(3)
                if len(top_values) > 0:
                    top_str = ", ".join([f"{k}({v})" for k, v in top_values.items()])
                    summary_parts.append(f"{col} top values: {top_str}")
        
        # Sample rows as context
        sample_rows = chunk_df.head(2).to_dict('records')
        if sample_rows:
            summary_parts.append(f"Sample data: {json.dumps(sample_rows, default=str)}")
            
        return " | ".join(summary_parts)

class AdaptiveChunker:
    """Adaptive chunking based on data characteristics"""
    
    def __init__(self, processor: EnhancedDataProcessor):
        self.processor = processor
        
    def determine_chunk_strategy(self, df: pd.DataFrame) -> Dict:
        """Determine optimal chunking strategy based on data"""
        char = self.processor.analyze_data_characteristics(df)
        
        # Adaptive chunk size based on data complexity
        base_chunk_size = CHUNK_SIZE
        
        # Adjust based on data characteristics
        if char['total_rows'] > 100000:  # Large dataset
            base_chunk_size = min(2000, max(500, base_chunk_size))
        elif char['total_rows'] > 10000:  # Medium dataset
            base_chunk_size = min(1500, max(300, base_chunk_size))
        else:  # Small dataset
            base_chunk_size = min(1000, max(100, base_chunk_size))
            
        # Adjust based on text content
        text_heavy = len(char['text_columns']) > len(char['numeric_columns'])
        if text_heavy:
            base_chunk_size = int(base_chunk_size * 0.7)  # Smaller chunks for text
            
        return {
            'chunk_size': base_chunk_size,
            'overlap_ratio': 0.1,  # 10% overlap
            'use_semantic_splitting': text_heavy,
            'characteristics': char
        }
    
    def create_adaptive_chunks(self, df: pd.DataFrame, session_id: str) -> List[Tuple[str, ChunkMetadata]]:
        """Create adaptive chunks with metadata"""
        strategy = self.determine_chunk_strategy(df)
        chunk_size = strategy['chunk_size']
        overlap_size = int(chunk_size * strategy['overlap_ratio'])
        
        logger.info(f"📦 Adaptive chunking: {len(df)} rows, chunk_size={chunk_size}, overlap={overlap_size}")
        
        chunks = []
        chunk_count = 0
        
        for i in range(0, len(df), chunk_size - overlap_size):
            end_idx = min(i + chunk_size, len(df))
            chunk_df = df.iloc[i:end_idx].dropna(how='all')
            
            if chunk_df.empty:
                continue
                
            # Create chunk content with semantic summary
            semantic_summary = self.processor.create_semantic_summary(chunk_df)
            csv_content = chunk_df.to_csv(index=False)
            
            # Combined content for better embedding
            chunk_content = f"{semantic_summary}\n\nRAW_DATA:\n{csv_content}"
            
            # Create metadata
            metadata = ChunkMetadata(
                chunk_id=f"{session_id}_chunk_{chunk_count}",
                source_rows=(i, end_idx),
                columns=chunk_df.columns.tolist(),
                data_types={col: str(dtype) for col, dtype in chunk_df.dtypes.items()},
                summary_stats=self._calculate_chunk_stats(chunk_df),
                semantic_tags=self._extract_semantic_tags(chunk_df)
            )
            
            chunks.append((chunk_content, metadata))
            chunk_count += 1
            
        logger.info(f"✅ Created {chunk_count} adaptive chunks")
        return chunks
    
    def _calculate_chunk_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate statistical summary for chunk"""
        stats = {}
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                stats[col] = {
                    'mean': df[col].mean() if not df[col].isna().all() else None,
                    'std': df[col].std() if not df[col].isna().all() else None,
                    'min': df[col].min() if not df[col].isna().all() else None,
                    'max': df[col].max() if not df[col].isna().all() else None
                }
            elif df[col].dtype == 'object':
                stats[col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(3).to_dict()
                }
        return stats
    
    def _extract_semantic_tags(self, df: pd.DataFrame) -> List[str]:
        """Extract semantic tags from chunk data"""
        tags = []
        
        # Add column-based tags
        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['name', 'title', 'description']):
                tags.append('text_content')
            elif any(word in col_lower for word in ['date', 'time', 'created', 'updated']):
                tags.append('temporal_data')
            elif any(word in col_lower for word in ['price', 'cost', 'amount', 'value']):
                tags.append('financial_data')
            elif any(word in col_lower for word in ['id', 'key', 'identifier']):
                tags.append('identifier_data')
                
        # Add data type tags
        if len(df.select_dtypes(include=['object']).columns) > 0:
            tags.append('categorical_data')
        if len(df.select_dtypes(include=[np.number]).columns) > 0:
            tags.append('numeric_data')
            
        return tags

class EnhancedRetriever:
    """Enhanced retrieval with multiple strategies"""
    
    def __init__(self):
        self.rerank_model = rerank_model
        
    def hybrid_retrieve(self, vectorstore: FAISS, query: str, k: int = RERANK_TOP_K) -> List[Tuple[Document, float]]:
        """Hybrid retrieval combining semantic and keyword matching"""
        try:
            # Primary semantic search
            semantic_results = vectorstore.similarity_search_with_score(query, k=k)
            
            # Query expansion for better matching
            expanded_queries = self._expand_query(query)
            
            # Retrieve with expanded queries
            all_results = {}
            for expanded_query in expanded_queries:
                results = vectorstore.similarity_search_with_score(expanded_query, k=k//2)
                for doc, score in results:
                    doc_id = hash(doc.page_content)
                    if doc_id not in all_results or all_results[doc_id][1] > score:
                        all_results[doc_id] = (doc, score)
            
            # Combine and deduplicate
            combined_results = list(all_results.values())
            combined_results.extend(semantic_results)
            
            # Remove duplicates and sort
            unique_results = {}
            for doc, score in combined_results:
                doc_id = hash(doc.page_content)
                if doc_id not in unique_results or unique_results[doc_id][1] > score:
                    unique_results[doc_id] = (doc, score)
            
            final_results = sorted(unique_results.values(), key=lambda x: x[1])
            return final_results[:k]
            
        except Exception as e:
            logger.error(f"Hybrid retrieval error: {e}")
            return vectorstore.similarity_search_with_score(query, k=k)
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and variations"""
        query_lower = query.lower()
        expansions = [query]
        
        # Add variations for common data terms
        expansions_map = {
            'total': ['sum', 'aggregate', 'count'],
            'average': ['mean', 'avg'],
            'maximum': ['max', 'highest', 'largest'],
            'minimum': ['min', 'lowest', 'smallest'],
            'count': ['number', 'quantity', 'total'],
            'unique': ['distinct', 'different'],
            'group': ['category', 'segment', 'type']
        }
        
        for original, variants in expansions_map.items():
            if original in query_lower:
                for variant in variants:
                    expanded = query_lower.replace(original, variant)
                    expansions.append(expanded)
        
        return expansions[:5]  # Limit expansions
    
    def rerank_results(self, query: str, documents: List[Document], scores: List[float]) -> List[Tuple[Document, float]]:
        """Rerank results using cross-encoder"""
        if not self.rerank_model or len(documents) <= 1:
            return list(zip(documents, scores))
        
        try:
            # Prepare pairs for reranking
            pairs = [(query, doc.page_content[:512]) for doc in documents]  # Truncate for efficiency
            
            # Get reranking scores
            rerank_scores = self.rerank_model.predict(pairs)
            
            # Combine with original scores
            combined_scores = []
            for i, (doc, orig_score) in enumerate(zip(documents, scores)):
                # Weighted combination of semantic similarity and rerank score
                combined_score = 0.6 * (1 - orig_score) + 0.4 * rerank_scores[i]
                combined_scores.append((doc, combined_score))
            
            # Sort by combined score (higher is better)
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            return combined_scores
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return list(zip(documents, scores))

class EnhancedRAGSystem:
    """Complete enhanced RAG system"""
    
    def __init__(self):
        self.processor = EnhancedDataProcessor()
        self.chunker = AdaptiveChunker(self.processor)
        self.retriever = EnhancedRetriever()
        
    def process_and_store(self, df: pd.DataFrame, session_id: str):
        """Process dataframe and create vectorstore"""
        logger.info(f"🧠 Processing dataframe for session {session_id}")
        
        # Create adaptive chunks
        chunks_with_metadata = self.chunker.create_adaptive_chunks(df, session_id)
        
        # Store metadata
        metadata_cache[session_id] = {
            'chunks': [metadata for _, metadata in chunks_with_metadata],
            'data_characteristics': self.processor.analyze_data_characteristics(df),
            'created_at': datetime.now().isoformat()
        }
        
        # Create documents for vectorstore
        documents = []
        for chunk_content, metadata in chunks_with_metadata:
            doc = Document(
                page_content=chunk_content,
                metadata={
                    'chunk_id': metadata.chunk_id,
                    'source_rows': metadata.source_rows,
                    'columns': metadata.columns,
                    'semantic_tags': metadata.semantic_tags
                }
            )
            documents.append(doc)
        
        # Create and store vectorstore
        logger.info("📌 Creating enhanced FAISS vectorstore")
        vector_db = FAISS.from_documents(documents, embedding_model)
        
        # Save to disk
        vector_path = os.path.join(VECTOR_DIR, f"vector_store_{session_id}")
        os.makedirs(os.path.dirname(vector_path), exist_ok=True)
        vector_db.save_local(vector_path)
        
        # Cache in memory
        vectorstore_map[session_id] = vector_db
        logger.info("✅ Enhanced vectorstore created and saved")
    
  
    def retrieve_context(self, session_id: str, question: str, max_tokens: int = 4000) -> str:
        """Enhanced context retrieval with token management"""
        vectorstore = vectorstore_map.get(session_id)
    
        if not vectorstore:
            try:
                logger.info("🔄 Loading vectorstore from disk")
                vector_path = os.path.join(VECTOR_DIR, f"vector_store_{session_id}")
                vectorstore = FAISS.load_local(vector_path, embedding_model, allow_dangerous_deserialization=True)
                vectorstore_map[session_id] = vectorstore
            except Exception as e:
                logger.error(f"Failed to load vectorstore: {e}")
                return ""
    
        try:
            # Start with more results, then filter by tokens
            initial_k = min(RERANK_TOP_K * 2, 20)  # Get more initially
            results = self.retriever.hybrid_retrieve(vectorstore, question, k=initial_k)
        
            if not results:
                return ""
        
            # Extract documents and scores
            documents = [doc for doc, score in results]
            scores = [score for doc, score in results]
        
            # Rerank results
            reranked_results = self.retriever.rerank_results(question, documents, scores)
        
            # Smart context selection based on tokens
            context_parts = []
            total_tokens = 0
            used_chunks = 0
        
            for i, (doc, score) in enumerate(reranked_results):
                # Skip very low relevance chunks
                if score < 0.3:  # Adjust threshold as needed
                    continue
                
                chunk_info = f"[Chunk {used_chunks+1} - Score: {score:.2f}]"
                chunk_content = f"{chunk_info}\n{doc.page_content}\n"
            
                # Count tokens for this chunk
                chunk_tokens = count_tokens(chunk_content)
            
                # Check if adding this chunk would exceed limit
                if total_tokens + chunk_tokens > max_tokens:
                    # Try to fit a truncated version
                    remaining_tokens = max_tokens - total_tokens - count_tokens(chunk_info + "\n\n")
                    if remaining_tokens > 100:  # Only if we have meaningful space left
                        truncated_content = truncate_to_tokens(doc.page_content, remaining_tokens)
                        chunk_content = f"{chunk_info}\n{truncated_content}\n"
                        context_parts.append(chunk_content)
                        used_chunks += 1
                    break
            
                context_parts.append(chunk_content)
                total_tokens += chunk_tokens
                used_chunks += 1
            
                # Don't exceed reasonable number of chunks
                if used_chunks >= FINAL_TOP_K:
                    break
        
            final_context = "\n".join(context_parts)
            logger.info(f"🔍 Retrieved {used_chunks} chunks using ~{total_tokens} tokens")
            return final_context
        
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
            return ""


# Initialize enhanced RAG system
rag_system = EnhancedRAGSystem()

# Memory management functions
# def add_to_memory(session_id: str, q: str, a: str):
#     """Add Q&A to conversation memory"""
#     conversation_memory.setdefault(session_id, []).append((q, a))
    
#     # Keep only last 10 conversations
#     if len(conversation_memory[session_id]) > 10:
#         conversation_memory[session_id] = conversation_memory[session_id][-10:]
    
#     # Update memory cache
#     memory_cache[session_id] = "\n".join([
#         f"Q: {q}\nA: {a}" for q, a in conversation_memory[session_id][-5:]
#     ])

def add_to_memory(session_id: str, q: str, a: str, max_entries: int = 8):
    """Add Q&A to conversation memory with size management"""
    # Truncate long questions and answers to prevent memory bloat
    q_truncated = q[:500] if len(q) > 500 else q
    a_truncated = a[:800] if len(a) > 800 else a
   
    conversation_memory.setdefault(session_id, []).append((q_truncated, a_truncated))
   
    # Keep only recent conversations
    if len(conversation_memory[session_id]) > max_entries:
        conversation_memory[session_id] = conversation_memory[session_id][-max_entries:]
   
    # Update memory cache with token awareness
    memory_cache[session_id] = get_memory_context(session_id, max_tokens=1000)

# def get_memory_context(session_id: str) -> str:
#     """Get conversation memory context"""
#     return memory_cache.get(session_id, "")

def get_memory_context(session_id: str, max_tokens: int = 1000) -> str:
    """Get conversation memory context with token limit"""
    if session_id not in conversation_memory:
        return ""
   
    # Get recent conversations
    recent_conversations = conversation_memory[session_id][-5:]  # Last 5 conversations
   
    # Build memory context with token awareness
    memory_parts = []
    total_tokens = 0
   
    # Add conversations in reverse order (most recent first)
    for q, a in reversed(recent_conversations):
        # Create a concise memory entry
        memory_entry = f"Q: {q[:200]}{'...' if len(q) > 200 else ''}\nA: {a[:300]}{'...' if len(a) > 300 else ''}\n"
        entry_tokens = count_tokens(memory_entry)
       
        if total_tokens + entry_tokens > max_tokens:
            break
           
        memory_parts.insert(0, memory_entry)  # Insert at beginning to maintain order
        total_tokens += entry_tokens
   
    result = "\n".join(memory_parts) if memory_parts else ""
    logger.debug(f"Memory context: {len(memory_parts)} entries, ~{total_tokens} tokens")
    return result





def get_data_overview(df: pd.DataFrame, max_tokens: int = 800) -> str:
    """Get comprehensive but concise data overview"""
    overview_parts = []
   
    # Essential info (always include)
    overview_parts.append(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
    overview_parts.append(f"Columns: {', '.join(df.columns.tolist()[:20])}{'...' if len(df.columns) > 20 else ''}")
   
    # Track tokens
    current_tokens = count_tokens("\n".join(overview_parts))
   
    # Data types (compressed)
    if current_tokens < max_tokens - 100:
        type_summary = df.dtypes.value_counts().to_dict()
        type_str = ", ".join([f"{k}: {v}" for k, v in list(type_summary.items())[:5]])
        overview_parts.append(f"Types: {type_str}")
        current_tokens = count_tokens("\n".join(overview_parts))
   
    # Missing data (only if significant)
    if current_tokens < max_tokens - 150:
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            missing_cols = missing_data[missing_data > 0]
            if len(missing_cols) <= 5:
                missing_dict = missing_cols.to_dict()
                overview_parts.append(f"Missing: {missing_dict}")
            else:
                overview_parts.append(f"Missing data in {len(missing_cols)} columns")
            current_tokens = count_tokens("\n".join(overview_parts))
   
    # Key statistics (selective)
    if current_tokens < max_tokens - 200:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:  # Limit to top 3
            if current_tokens >= max_tokens - 100:
                break
            stats = df[col].describe()
            stats_str = f"{col}: μ={stats['mean']:.1f}, σ={stats['std']:.1f}, range=[{stats['min']:.1f}, {stats['max']:.1f}]"
            if count_tokens(stats_str) < 50:  # Only if concise
                overview_parts.append(stats_str)
                current_tokens = count_tokens("\n".join(overview_parts))
   
    # Top categorical values (very selective)
    if current_tokens < max_tokens - 100:
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:2]:  # Only top 2
            if current_tokens >= max_tokens - 80:
                break
            top_values = df[col].value_counts().head(2).to_dict()
            cat_str = f"{col} top: {top_values}"
            if count_tokens(cat_str) < 60:
                overview_parts.append(cat_str)
                current_tokens = count_tokens("\n".join(overview_parts))
   
    final_overview = "\n".join(overview_parts)
    final_tokens = count_tokens(final_overview)
    logger.debug(f"Data overview: ~{final_tokens} tokens")
    return final_overview

@csrf_exempt
def upload_file(request):
    """Enhanced file upload with better processing"""
    if request.method != 'POST' or not request.FILES.get('file'):
        return JsonResponse({'error': 'No file received.'}, status=400)

    try:
        file = request.FILES['file']
        session_id = str(uuid.uuid4())
        filename = f"{session_id}_{file.name}"
        filepath = os.path.join(settings.MEDIA_ROOT, filename)

        # Ensure media directory exists
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

        # Save file
        with open(filepath, 'wb+') as f:
            for chunk in file.chunks():
                f.write(chunk)

        logger.info(f"📁 Uploaded: {filename}")

        # Read file with better error handling
        try:
            if filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(filepath, engine='openpyxl')
            elif filename.endswith('.csv'):
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    return JsonResponse({'error': 'Could not decode CSV file'}, status=400)
            else:
                return JsonResponse({'error': 'Unsupported file format'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'File parsing error: {str(e)}'}, status=400)

        # Clean column names
        df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
        
        # Remove empty rows
        df = df.dropna(how='all')
        
        if df.empty:
            return JsonResponse({'error': 'File contains no data'}, status=400)

        # Store dataframe
        dataframe_map[session_id] = df

        logger.info(f"✅ File processed: {df.shape} shape")
        
        # Prepare vectorstore asynchronously for large files
        if len(df) > 1000:
            # For large files, process in background
            import threading
            thread = threading.Thread(
                target=rag_system.process_and_store,
                args=(df, session_id)
            )
            thread.start()
        else:
            # For small files, process immediately
            rag_system.process_and_store(df, session_id)

        return JsonResponse({
            'message': 'File uploaded and processed successfully.',
            'session_id': session_id,
            'filename': filename,
            'row_count': len(df),
            'columns': df.columns.tolist(),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        })

    except Exception as e:
        logger.error(f"Upload error: {traceback.format_exc()}")
        return JsonResponse({'error': f'Upload error: {str(e)}'}, status=500)
    

# === Extract text from PDF ===
def extract_text_from_pdf(file_path):
    print(f"Extracting text from PDF: {file_path}")
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    print("Text extraction complete.")
    return text


# === Ingest and merge PDFs into session-based vectorstore ===
def ingest_pdf_to_vectorstore(file_path, vectorstore_path):
    print(f"Ingesting PDF to vectorstore: {file_path} -> {vectorstore_path}")
    
    # Step 1: Extract text and split into chunks
    text = extract_text_from_pdf(file_path)
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = [Document(page_content=c) for c in splitter.split_text(text)]

    # Step 2: Merge with existing vectorstore if it exists
    if os.path.exists(os.path.join(vectorstore_path, "index.faiss")):
        try:
            print("Merging into existing vectorstore...")
            existing_vectordb = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
            new_vectordb = FAISS.from_documents(chunks, embedding_model)
            existing_vectordb.merge_from(new_vectordb)
            existing_vectordb.save_local(vectorstore_path)
            print("Vectorstore updated with new document.")
            return
        except Exception as e:
            print(f"Error merging vectorstore, rebuilding from scratch: {e}")

    # Step 3: Create new vectorstore
    print("Creating new vectorstore...")
    vectordb = FAISS.from_documents(chunks, embedding_model)
    vectordb.save_local(vectorstore_path)
    print("Vectorstore ingestion complete.")


# === Upload PDF endpoint (per-user session) ===
@csrf_exempt
def upload_pdf(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    pdf_file = request.FILES.get('file')
    session_id = request.POST.get('session_id')

    if not pdf_file or not session_id:
        return JsonResponse({'error': 'Missing file or session_id'}, status=400)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{pdf_file.name}")
    with open(file_path, 'wb+') as destination:
        for chunk in pdf_file.chunks():
            destination.write(chunk)

    vectorstore_path = os.path.join(VECTORSTORE_DIR, f"{session_id}_store")
    ingest_pdf_to_vectorstore(file_path, vectorstore_path)

    return JsonResponse({'message': f'PDF uploaded and indexed for session {session_id}.'})


# === Ask question using session-specific vectorstore ===
@csrf_exempt
def ask_questionpdf(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    try:
        data = json.loads(request.body)
    except Exception as e:
        return JsonResponse({'error': 'Invalid JSON.'}, status=400)

    question = data.get("query")
    session_id = data.get("session_id")

    if not question or not session_id:
        return JsonResponse({'error': 'Missing query or session_id'}, status=400)

    print(f"Received question: {question} | session_id: {session_id}")

    vectorstore_path = os.path.join(VECTORSTORE_DIR, f"{session_id}_store")
    if not os.path.exists(os.path.join(vectorstore_path, "index.faiss")):
        return JsonResponse({'answer': "No documents found for this session. Please upload at least one PDF."})

    try:
        vectordb = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
        print("Vectorstore loaded.")
    except Exception as e:
        print(f"Vectorstore load error: {e}")
        return JsonResponse({'answer': "Failed to load session knowledge base. Please try again."})
    

    retriever = vectordb.as_retriever(search_type="similarity", k=5)

    llm = ChatOpenAI(
        model="google/gemma-3-27b-it:free",  
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",  
        temperature=0,
        max_tokens=1024
    )

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    You are an expert AI assistant. Use the below context to answer the user's question.

    IMPORTANT:
    - Do NOT start your answer with phrases like 'Based on the provided text' or 'According to the text'.
    - Answer directly in a clear, confident, and natural tone.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt}
    )

    print("Generating answer using LLM...")
    result = qa_chain.invoke({"query": question})
    answer = result if isinstance(result, str) else result.get("result", "")

    print(f"Answer generated: {answer}")
    return JsonResponse({'answer': answer})

  


def get_schema_info(engine):
    """Get comprehensive schema information with exact table naming"""
    logger.info("=== SCHEMA INFO RETRIEVAL START ===")
    
    try:
        inspector = inspect(engine)
        schema_info = {'tables': {}}

        for schema_name in inspector.get_schema_names():
            if schema_name not in ['information_schema', 'pg_catalog', 'pg_toast']:
                for table_name in inspector.get_table_names(schema=schema_name):
                    columns = inspector.get_columns(table_name, schema=schema_name)
                    
                    # Use double quotes around schema and table for exact referencing
                    qualified_table_name = f'"{schema_name}"."{table_name}"'

                    schema_info['tables'][qualified_table_name] = {
                        'columns': {
                            col['name']: {
                                'type': str(col['type']),
                                'nullable': col['nullable'],
                                'default': col.get('default')
                            }
                            for col in columns
                        }
                    }
        
        logger.info(f"Schema info retrieval completed. Total tables processed: {len(schema_info['tables'])}")
        return schema_info

    except Exception as e:
        logger.error(f"Critical error in schema inspection: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {'tables': {}}


# views.py
import json
import uuid
import logging
from urllib.parse import quote_plus

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError, SQLAlchemyError

logger = logging.getLogger(__name__)

# If you have these globals defined elsewhere, import them.
# Otherwise, define them once at module scope.
try:
    session_store
    conversation_memory_store
except NameError:
    session_store = {}
    conversation_memory_store = {}

# ---- Optional: real embed helper if available ----
try:
    # from genai_app.somewhere import embed_schema_for_user as _real_embed_schema
    _real_embed_schema = None  # replace with your real import
except Exception:
    _real_embed_schema = None

def _embed_schema_for_user(user_id: str, db_id: str, schema_text: str) -> None:
    """Call your real embed function if present; otherwise no-op."""
    try:
        if _real_embed_schema:
            _real_embed_schema(user_id=user_id, db_id=db_id, schema_text=schema_text)
        else:
            logger.info("Skipping schema embedding (helper not wired).")
    except Exception as e:
        logger.warning("Schema embedding failed: %s", e)

def _extract_schema_from_sqlalchemy(engine) -> str:
    """Lightweight schema introspection—works on most Postgres setups."""
    try:
        insp = inspect(engine)
        schema_names = insp.get_schema_names()
    except Exception as e:
        logger.warning("Could not list schemas: %s", e)
        schema_names = ["public"]

    lines = []
    for schema in schema_names:
        try:
            tables = insp.get_table_names(schema=schema)
        except Exception as e:
            logger.warning("Could not list tables for schema %s: %s", schema, e)
            continue

        for t in tables:
            try:
                cols = insp.get_columns(t, schema=schema)
                col_list = ", ".join(f'"{c.get("name")}" {c.get("type")}' for c in cols)
                lines.append(f'{schema}."{t}" ({col_list})')
            except Exception as e:
                logger.warning("Could not inspect %s.%s: %s", schema, t, e)

    return "\n".join(lines) if lines else ""


import uuid, json
import uuid, json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from urllib.parse import quote_plus

@csrf_exempt
def connect_database(request):
    print("➡️ connect_database called")

    try:
        # Always use static DB from settings.py
        data = settings.STATIC_DB
        user = "static_user"
        print(f"ℹ️ Using static credentials for user={user}")

        pg_user = data["postgres_user"]
        pg_pass = data["postgres_password"]
        pg_host = data["postgres_host"]
        pg_port = data["postgres_port"]
        pg_db   = data["postgres_db"]

        print(f"🔑 DB creds -> user={pg_user}, host={pg_host}, port={pg_port}, db={pg_db}")

        # Build connection string
        encoded_pwd = quote_plus(str(pg_pass))
        conn_str = f"postgresql://{pg_user}:{encoded_pwd}@{pg_host}:{pg_port}/{pg_db}"
        print(f"🔗 Connection string built: {conn_str.replace(encoded_pwd, '***')}")

        # Create engine
        engine = create_engine(
            conn_str,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={"connect_timeout": 8},
        )
        print("✅ SQLAlchemy engine created")

        # Test connection
        with engine.connect() as conn:
            print("⏳ Testing connection with SELECT 1…")
            conn.execute(text("SELECT 1"))
            print("✅ Connection test passed")

        # Extract schema if helper exists
        schema_text = ""
        try:
            print("⏳ Extracting schema…")
            schema_text = _extract_schema_from_sqlalchemy(engine)
            print(f"✅ Schema extracted (length={len(schema_text)})")
        except Exception as ex:
            print(f"⚠️ Schema extraction failed: {ex}")
            schema_text = ""

        # Store session
        session_id = str(uuid.uuid4())
        session_store[session_id] = {"engine": engine, "user_id": user}
        conversation_memory_store[session_id] = []
        print(f"💾 Session stored with id={session_id}")

        try:
            print("⏳ Running embed schema helper…")
            _embed_schema_for_user(user_id=user, db_id=session_id, schema_text=schema_text)
            print("✅ Embed schema done")
        except Exception as ex:
            print(f"⚠️ Embed schema skipped: {ex}")

        print("🎉 Database connection successful")
        return JsonResponse({
            "message": "Connected successfully",
            "session_id": session_id,
            "has_schema": bool(schema_text),
        })

    except OperationalError as e:
        print(f"❌ OperationalError: {e}")
        return JsonResponse({
            "error": "Database connection failed",
            "detail": str(e)
        }, status=502)

    except SQLAlchemyError as e:
        print(f"❌ SQLAlchemyError: {e}")
        return JsonResponse({
            "error": "SQLAlchemy error",
            "detail": str(e)
        }, status=500)

    except Exception as e:
        print(f"❌ Unhandled error: {e}")
        return JsonResponse({
            "error": "Unhandled error",
            "detail": str(e)
        }, status=500)



@csrf_exempt
def connect_databaseworkingwithoutstatic(request):
    if request.method != "POST":
        return JsonResponse({"error": "Use POST"}, status=405)

    try:
        data = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    user = data.get("user_id", "test_user_001")
    pg_user = data.get("postgres_user")
    pg_pass = data.get("postgres_password")
    pg_host = data.get("postgres_host")
    pg_port = data.get("postgres_port", 5432)
    pg_db   = data.get("postgres_db")

    missing = [k for k, v in {
        "postgres_user": pg_user,
        "postgres_password": pg_pass,
        "postgres_host": pg_host,
        "postgres_db": pg_db,
    }.items() if not v]
    if missing:
        return JsonResponse(
            {"error": "Missing fields", "missing": missing},
            status=400
        )

    try:
        pg_port = int(pg_port)
    except (TypeError, ValueError):
        return JsonResponse({"error": "postgres_port must be an integer"}, status=400)

    # Build engine
    encoded_pwd = quote_plus(str(pg_pass))
    conn_str = f"postgresql://{pg_user}:{encoded_pwd}@{pg_host}:{pg_port}/{pg_db}"

    engine = None
    try:
        engine = create_engine(
            conn_str,
            pool_pre_ping=True,
            pool_recycle=1800,  # avoid stale connections
            connect_args={"connect_timeout": 8},
        )

        # Basic connectivity check
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        # Extract schema (best-effort)
        schema_text = _extract_schema_from_sqlalchemy(engine)

        # Create a session id and store engine
        session_id = str(uuid.uuid4())
        session_store[session_id] = {"engine": engine, "user_id": user}
        conversation_memory_store[session_id] = []

        # Try embedding (no-op if helper isn’t wired)
        _embed_schema_for_user(user_id=user, db_id=session_id, schema_text=schema_text)

        return JsonResponse({
            "message": "Connected successfully",
            "session_id": session_id,
            "has_schema": bool(schema_text),
        })

    except OperationalError as e:
        if engine:
            engine.dispose()
        logger.exception("DB connection failed")
        return JsonResponse({
            "error": "Database connection failed",
            "detail": str(e.__cause__ or e),
            "hints": [
                "Verify host/port/db/user/password",
                "Ensure the database accepts connections from this machine",
                "If password has special characters, it is already URL-encoded via quote_plus"
            ]
        }, status=502)

    except SQLAlchemyError as e:
        if engine:
            engine.dispose()
        logger.exception("SQLAlchemy error")
        return JsonResponse({
            "error": "SQLAlchemy error",
            "detail": str(e)
        }, status=500)

    except Exception as e:
        if engine:
            engine.dispose()
        logger.exception("Unhandled error in connect_database")
        return JsonResponse({
            "error": "Unhandled error",
            "detail": str(e)
        }, status=500)


import traceback
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from datetime import datetime
from sqlalchemy import text
import traceback
import json

### ✅ Final Working Version with `extract_sql_block`

import re

import re
# ✅ Hardcoded schema (as provided)
FULL_SCHEMA = '''
Table: "bi_dwh"."main_cai_lib"
Columns:
- own_damage_premium
- vehicle_age
- third_party_premium
- total_premium_payable
- vehicle_idv
- total_revenue
- policy_tenure
- number_of_claims
- claims_approved
- claim_approval_rate
- customer_tenure
- customer_life_time_value
- customerid
- chassis_number
- engine_number
- vehicle_register_number
- state
- zone
- business_type
- car_manufacturer
- vehicle_model
- product_name
- policy_no
- tie_up
- vehicle_model_variant
- policy_start_date_year
- policy_end_date_year
- policy_start_date_month
- policy_end_date_month
- is_churn
- customer_segment
- branch_name
- main_churn_reason
- primary_recommendation
- insured_client_name
'''

import sqlparse

def validate_sql_columns(sql: str, valid_columns: set) -> set:
    tokens = sqlparse.parse(sql)[0].tokens
    words = set()
    
    for token in tokens:
        for t in token.flatten():
            if t.ttype is None and t.value not in ('SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'COUNT', '(', ')', '=', 'IS', 'NOT', 'NULL'):
                words.add(t.value)
    
    return words - valid_columns



import re

_SQL_FENCE = re.compile(
    r"```(?:sql|postgresql|postgres|pgsql)?\s*([\s\S]*?)\s*```",
    re.IGNORECASE
)
_ANY_FENCE = re.compile(r"```([\s\S]*?)```", re.DOTALL)
_SELECT_WITH = re.compile(r"(?is)\b(SELECT|WITH)\b.*?(?=(?:```|$))")

def _cleanup_sql(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    # drop leading "SQL:" labels if present
    s = re.sub(r"(?i)^\s*sql\s*:\s*", "", s)
    # normalize weird quotes and fix LIMIT spacing
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"\bLIMIT(\d+)", r"LIMIT \1", s, flags=re.IGNORECASE)
    return s.strip()

def extract_sql_block(text) -> str:
    # 1) normalize input
    if isinstance(text, tuple):
        text = text[0]
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""

    s = text.strip()
    if not s:
        return ""

    # 2) prefer explicit SQL fences
    m = _SQL_FENCE.search(s)
    if m:
        return _cleanup_sql(m.group(1))

    # 3) generic fenced code block fallback
    m = _ANY_FENCE.search(s)
    if m:
        body = m.group(1)
        # try to isolate a SELECT/WITH inside the fence
        sw = _SELECT_WITH.search(body)
        return _cleanup_sql(sw.group(0) if sw else body)

    # 4) no fences: grab first SELECT/WITH chunk from the whole text
    sw = _SELECT_WITH.search(s)
    if sw:
        return _cleanup_sql(sw.group(0))

    # 5) last resort: return cleaned full text
    return _cleanup_sql(s)


import re

def extract_columns_from_schema(schema: str) -> set:
    return set(re.findall(r'- ([a-zA-Z0-9_]+)', schema))


# Assuming FULL_SCHEMA is already defined
def extract_column_names(schema: str) -> set:
    lines = schema.splitlines()
    columns = set()

    for line in lines:
        match = re.match(r'^\s*-\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
        if match:
            columns.add(match.group(1))
    
    return columns

# 👇 Call this once to generate the column set
VALID_COLUMNS = extract_columns_from_schema(FULL_SCHEMA)

# ✅ Print or use the result
print(VALID_COLUMNS)


# Return CSV response
import requests
import os

import re
import time
import requests
from loguru import logger  # Make sure logger is configured
import os
from genai_app.utils.llm_utils import llm_generate_chart_config
from genai_app.langgraph_logic.langgraph_runner import generate_summary_from_rows


AZURE_ENDPOINT = os.getenv("AZURE_INFERENCE_ENDPOINT")  # https://genaiprochurn.services.ai.azure.com/models
AZURE_API_KEY = os.getenv("AZURE_INFERENCE_API_KEY")
AZURE_MODEL = os.getenv("AZURE_INFERENCE_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8-prochurn-demo")
AZURE_API_VERSION = "2024-05-01-preview"
MAX_PROMPT_TOKENS = int(os.getenv("AZURE_MAX_PROMPT_TOKENS", "120000"))

_SYSTEM_PROMPT = os.getenv("AZURE_SYSTEM_PROMPT", "You are a helpful SQL assistant.")

_client: Optional[ChatCompletionsClient] = None

def _get_client() -> ChatCompletionsClient:
    global _client
    if _client is None:
        print(f"DEBUG - AZURE_ENDPOINT from Django: {os.getenv('AZURE_INFERENCE_ENDPOINT')}")
        print(f"DEBUG - AZURE_API_KEY from Django: {os.getenv('AZURE_INFERENCE_API_KEY', 'NOT SET')}")
        if not AZURE_ENDPOINT or not AZURE_API_KEY:
            logger.error("Azure credentials not configured")
            raise RuntimeError("Set AZURE_INFERENCE_ENDPOINT and AZURE_INFERENCE_API_KEY.")
        
        # Ensure endpoint has the correct format
        endpoint = AZURE_ENDPOINT.rstrip('/')
        if not endpoint.endswith('/models'):
            endpoint = f"{endpoint}/models"
        
        logger.info(f"Initializing Azure client with endpoint: {endpoint}")
        
        _client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(AZURE_API_KEY),
            api_version=AZURE_API_VERSION,
        )
    return _client



def _clean_model_text(text: str) -> str:
    if not text:
        return ""
    # Strip SQL/code blocks and inline code
    text = re.sub(r"```sql.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def llm_generate_recommendation(question: str, rows: List[Dict[str, Any]]) -> str:
    preview_rows = rows[:10] if rows else []
    if isinstance(preview_rows, list) and preview_rows and isinstance(preview_rows[0], dict):
        table_str = "\n".join(", ".join(f"{k}: {v}" for k, v in row.items()) for row in preview_rows)
    else:
        table_str = "No rows available."

    system_prompt = (
        "You are a data-driven business analyst assistant. Based on the user's question and the query output data, "
        "generate a concise, actionable, and professional business recommendation. "
        "Focus on explaining the key trends, risks, or opportunities in simple business terms, "
        "helping the user understand what actions or decisions they can take next. "
        "Strictly avoid including any SQL code, technical jargon, or step-by-step query explanations. "
        "Only provide a high-level insight that would help a manager or decision-maker."
        "You are a business analyst who writes like a helpful colleague. "
        "Based on the user's question and the query output data, "
        "write a concise, actionable recommendation in plain business language. "
        "Be concrete and tie suggestions to the data shown. "
        # "Do not include SQL, technical jargon, or step-by-step query explanations. "
        # "One short paragraph only. No emojis."
    )

    user_prompt = f"""User Question: {question}

Data Preview (first 10 rows):
{table_str}

Please provide only a one-paragraph business recommendation. Do not include SQL queries or explanations.
"""

    # Size guard (rough)
    full_prompt_words = len((system_prompt + user_prompt).split())
    if full_prompt_words > 5000:
        logger.warning("Prompt too large, truncating...")
        # Hard truncation safeguards
        user_prompt = user_prompt[:8000]
        system_prompt = system_prompt[:2000]

    messages = [
        SystemMessage(content=system_prompt),
        UserMessage(content=user_prompt),
    ]

    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            resp = _get_client().complete(
                messages=messages,
                model=AZURE_MODEL,
                temperature=0.1,
                top_p=0.9,
                max_tokens=1000,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )

            if not resp.choices:
                logger.warning("Empty choices from Azure Inference.")
                raise RuntimeError("Empty response")

            raw_text = (resp.choices[0].message.content or "").strip()
            cleaned = _clean_model_text(raw_text)
            return cleaned or "No clear recommendation could be generated from the data provided."

        except HttpResponseError as e:
            status = getattr(e, "status_code", None)
            logger.warning(f"Azure HttpResponseError (status={status}): {e}")
            if status in (429, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            break
        except (ServiceRequestError, ServiceResponseError, TimeoutError, ConnectionError) as e:
            logger.warning(f"Transient Azure error: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            break

    return "I apologize, but I'm having trouble processing your request right now."

import csv, re, json, traceback
from django.http import HttpResponse, StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.timezone import now
from sqlalchemy import text
from .utils.stream import jsonl_line
from .feedback.feedback_view import vector_db,general_db
from .feedback.feedback_view import store_general_feedback


# def normalize_question(q: str) -> str:
#     """Normalize question text for exact-match checks (case + whitespace insensitive)."""
#     return " ".join((q or "").split()).lower().strip()


# ✅ Normalizer shared across all feedback/storage
def normalize_question(q: str) -> str:
    """Lowercase, strip, collapse spaces, remove trailing punctuation."""
    if not q:
        return ""
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)         # collapse multiple spaces
    q = re.sub(r"[?!.;,]+$", "", q)    # remove trailing ? . , etc.
    return q


#workingcode bfr general db
def retrieve_context11(question, top_k=5):
    """Retrieve most recent positive feedback doc for reuse."""
    try:
        results = vector_db.similarity_search(question, k=top_k)
        positives = [
            doc for doc in results if doc.metadata.get("feedback") == "positive"
        ]
        if not positives:
            return ""  # no positive feedback yet

        # ✅ Pick the most recent by timestamp
        positives.sort(key=lambda d: d.metadata.get("timestamp", ""), reverse=True)
        best_doc = positives[0]
        print(f"♻️ Reusing SQL from {best_doc.metadata.get('session_id')} (doc_id={best_doc.metadata.get('doc_id')})")
        return best_doc.page_content
    except Exception as e:
        print(f"⚠️ RAG retrieval failed: {e}")
        return ""


from django.utils.timezone import now
#workingcode bfr general db
def store_interaction_in_chroma11(question, answer, sql, summary, recommendation, session_id, feedback="auto"):
    """Store or update Q/A/SQL into Chroma for future retrieval (with deterministic doc_id + timestamp)."""
    try:
        from langchain.schema import Document
        doc_id = f"{session_id}:{question.lower().strip()}"
        metadata = {
            "answer": answer,
            "sql": sql,
            "summary": summary,
            "recommendation": recommendation,
            "session_id": session_id,
            "feedback": feedback,
            "doc_id": doc_id,
            "timestamp": now().isoformat()  # ✅ store save time
        }
        vector_db.add_texts(
            texts=[question],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(f"💾 Stored/updated doc_id={doc_id} (feedback={feedback})")
    except Exception as e:
        print(f"⚠️ Could not store in Chroma: {e}")



def store_interaction_in_chroma_session(question, answer, sql, summary, recommendation, session_id, feedback="auto"):
    """Store/update Q/A/SQL into Chroma with deduplication scoped to a single session."""
    try:
        q_norm = normalize_question(question)

        results = vector_db.similarity_search_with_score(q_norm, k=5)

        existing_doc_id = None
        for doc, score in results:
            if normalize_question(doc.page_content) == q_norm and doc.metadata.get("session_id") == session_id:
                existing_doc_id = doc.metadata.get("doc_id")
                break

        if existing_doc_id:
            print(f"♻️ [SESSION] Updating existing doc_id={existing_doc_id} for question='{question}' with feedback={feedback}")
            vector_db._collection.update(
                ids=[existing_doc_id],
                metadatas=[{
                    "question": question,
                    "sql": sql,
                    "answer": answer,
                    "summary": summary,
                    "recommendation": recommendation,
                    "session_id": session_id,
                    "feedback": feedback,
                    "doc_id": existing_doc_id,
                    "timestamp": now().isoformat()
                }]
            )
        else:
            doc_id = f"{session_id}:{q_norm}"
            print(f"💾 [SESSION] Inserting new doc_id={doc_id} for question='{question}' (feedback={feedback})")
            vector_db.add_texts(
                texts=[q_norm],
                ids=[doc_id],
                metadatas=[{
                    "question": question,
                    "sql": sql,
                    "answer": answer,
                    "summary": summary,
                    "recommendation": recommendation,
                    "session_id": session_id,
                    "feedback": feedback,
                    "doc_id": doc_id,
                    "timestamp": now().isoformat()
                }]
            )
    except Exception as e:
        print(f"❌ [SESSION] Failed to store interaction in Chroma: {e}")




# =====================================
# 🔹 Store Interaction in Chroma (Global + Session)
# =====================================
def store_interaction_in_chroma(question, answer, sql, summary, recommendation, session_id, feedback="auto"):
    """Store/update Q/A/SQL into Chroma with deduplication across ALL sessions (global memory)."""
    try:
        # ✅ Normalize question
        q_norm = normalize_question(question)

        # 🔎 Search globally for same normalized question
        results = vector_db.similarity_search_with_score(q_norm, k=5)

        existing_doc_id = None
        for doc, score in results:
            if normalize_question(doc.page_content) == q_norm:
                existing_doc_id = doc.metadata.get("doc_id")
                break

        if existing_doc_id:
            print(f"♻️ [GLOBAL] Updating existing doc_id={existing_doc_id} for question='{question}' with feedback={feedback}")
            vector_db._collection.update(
                ids=[existing_doc_id],
                metadatas=[{
                    "question": question,
                    "sql": sql,
                    "answer": answer,
                    "summary": summary,
                    "recommendation": recommendation,
                    "session_id": session_id,
                    "feedback": feedback,
                    "doc_id": existing_doc_id,
                    "timestamp": now().isoformat()
                }]
            )
        else:
            doc_id = f"global:{q_norm}"
            print(f"💾 [GLOBAL] Inserting new doc_id={doc_id} for question='{question}' (feedback={feedback})")
            vector_db.add_texts(
                texts=[q_norm],
                ids=[doc_id],
                metadatas=[{
                    "question": question,
                    "sql": sql,
                    "answer": answer,
                    "summary": summary,
                    "recommendation": recommendation,
                    "session_id": session_id,
                    "feedback": feedback,
                    "doc_id": doc_id,
                    "timestamp": now().isoformat()
                }]
            )

        # ✅ Also store session-specific version
        session_doc_id = f"{session_id}:{q_norm}"
        vector_db.add_texts(
            texts=[q_norm],
            ids=[session_doc_id],
            metadatas=[{
                "question": question,
                "sql": sql,
                "answer": answer,
                "summary": summary,
                "recommendation": recommendation,
                "session_id": session_id,
                "feedback": feedback,
                "doc_id": session_doc_id,
                "timestamp": now().isoformat()
            }]
        )
        print(f"💾 [SESSION] Stored doc_id={session_doc_id} for session={session_id}")

    except Exception as e:
        print(f"❌ [GLOBAL] Failed to store interaction in Chroma: {e}")



# def store_interaction_in_chroma(question, answer, sql, summary, recommendation, session_id, feedback="auto"):
#     """Store/update Q/A/SQL into Chroma with deduplication across ALL sessions (global memory)."""
#     try:
#         q_norm = normalize_question(question)

#         results = vector_db.similarity_search_with_score(q_norm, k=5)
#         existing_doc_id = None
#         for doc, score in results:
#             if normalize_question(doc.page_content) == q_norm:
#                 existing_doc_id = doc.metadata.get("doc_id")
#                 break

#         if existing_doc_id:
#             print(f"♻️ [GLOBAL] Updating existing doc_id={existing_doc_id} for question='{question}' with feedback={feedback}")
#             vector_db._collection.update(
#                 ids=[existing_doc_id],
#                 metadatas=[{
#                     "question": question,
#                     "sql": sql,
#                     "answer": answer,
#                     "summary": summary,
#                     "recommendation": recommendation,
#                     "session_id": session_id,
#                     "feedback": feedback,
#                     "doc_id": existing_doc_id,
#                     "timestamp": now().isoformat()
#                 }]
#             )
#         else:
#             doc_id = f"global:{q_norm}"
#             print(f"💾 [GLOBAL] Inserting new doc_id={doc_id} for question='{question}' (feedback={feedback})")
#             vector_db.add_texts(
#                 texts=[q_norm],
#                 ids=[doc_id],
#                 metadatas=[{
#                     "question": question,
#                     "sql": sql,
#                     "answer": answer,
#                     "summary": summary,
#                     "recommendation": recommendation,
#                     "session_id": session_id,
#                     "feedback": feedback,
#                     "doc_id": doc_id,
#                     "timestamp": now().isoformat()
#                 }]
#             )
#     except Exception as e:
#         print(f"❌ [GLOBAL] Failed to store interaction in Chroma: {e}")



# =====================================
# 🔹 Store Interaction (SQL side)
# =====================================
# def store_interaction_in_chroma159(question, answer, sql, summary, recommendation, session_id, feedback="auto"):
#     """Store or update Q/A/SQL into Chroma for future retrieval (with deterministic doc_id + timestamp)."""
#     try:
#         normalized_q = " ".join(question.split()).lower().strip()
#         doc_id = f"{session_id}:{normalized_q}"
#         # doc_id = f"{session_id}:{question.lower().strip()}"
#         metadata = {
#             "answer": answer,
#             "sql": sql,
#             "summary": summary,
#             "recommendation": recommendation,
#             "session_id": session_id,
#             "feedback": feedback,
#             "doc_id": doc_id,
#             "timestamp": now().isoformat()  # ✅ store save time
#         }
#         vector_db.add_texts(
#             texts=[question],
#             metadatas=[metadata],
#             ids=[doc_id]
#         )
#         print(f"💾 Stored/updated SQL doc_id={doc_id} (feedback={feedback})")
#     except Exception as e:
#         print(f"⚠️ Could not store in Chroma: {e}")

# =====================================
# 🔹 Store General Interaction (ask_qwen)
# =====================================


# =====================================
# 🔹 Retrieve Context without similarity score (reuse only positive, enrich with auto)
# =====================================
def retrieve_context159(question, top_k=5):
    """Retrieve most recent positive feedback doc for reuse. Use 'auto' only as enrichment."""
    try:
        results = vector_db.similarity_search(question, k=top_k)
        positives = [doc for doc in results if doc.metadata.get("feedback") == "positive"]

        if positives:
            positives.sort(key=lambda d: d.metadata.get("timestamp", ""), reverse=True)
            best_doc = positives[0]
            print(f"♻️ Reusing SQL from {best_doc.metadata.get('session_id')} (doc_id={best_doc.metadata.get('doc_id')})")
            return best_doc.metadata.get("sql", "")

        # fallback: enrich with auto docs (but do not reuse directly)
        autos = [doc for doc in results if doc.metadata.get("feedback") == "auto"]
        if autos:
            return "\n\n".join([
                f"-- Past SQL (unrated): {doc.metadata.get('sql')}"
                for doc in autos if doc.metadata.get("sql")
            ])

        return ""
    except Exception as e:
        print(f"⚠️ RAG retrieval failed: {e}")
        return ""


# === Retrieval Function ===
def retrieve_context(question, top_k=5, threshold=0.8):
    """Retrieve most recent positive doc (if above threshold) or enrich with autos."""
    try:
        results = vector_db.similarity_search_with_score(question, k=top_k)

        debug_matches = []
        for doc, score in results:
            debug_matches.append({
                "question": doc.page_content,
                "sql": doc.metadata.get("sql"),
                "feedback": doc.metadata.get("feedback"),
                "score": float(score),
                "session_id": doc.metadata.get("session_id"),
                "doc_id": doc.metadata.get("doc_id")
            })

        # ✅ Positive matches first
        positives = [
            (doc, score) for doc, score in results
            if doc.metadata.get("feedback") == "positive" and score >= threshold
        ]
        if positives:
            positives.sort(key=lambda d: d[0].metadata.get("timestamp", ""), reverse=True)
            best_doc, best_score = positives[0]
            print(f"♻️ Reusing SQL (score={best_score:.2f}) from {best_doc.metadata.get('doc_id')}")
            return {
                "sql": best_doc.metadata.get("sql", ""),
                "matched_question": best_doc.page_content,
                "score": float(best_score),
                "top_k_debug": debug_matches
            }

        # ✅ Auto matches (lower bar)
        autos = [
            (doc, score) for doc, score in results
            if doc.metadata.get("feedback") == "auto" and score >= threshold * 0.7
        ]
        if autos:
            enrichment = "\n\n".join([
                f"-- Past SQL (unrated, score={score:.2f}): {doc.metadata.get('sql')}"
                for doc, score in autos if doc.metadata.get("sql")
            ])
            return {
                "sql": enrichment,
                "matched_question": autos[0][0].page_content,
                "score": float(autos[0][1]),
                "top_k_debug": debug_matches
            }

        return {"sql": "", "matched_question": None, "score": 0.0, "top_k_debug": debug_matches}

    except Exception as e:
        print(f"⚠️ RAG retrieval failed: {e}")
        return {"sql": "", "matched_question": None, "score": 0.0, "top_k_debug": []}



@csrf_exempt
def ask_question(request):
    try:
        start_time = now()

        # ✅ Handle GET request for CSV export
        if request.method == "GET" and request.GET.get("export") == "true":
            session_id = request.GET.get("session_id")
            question = request.GET.get("question")

            print("🔍 Export request received")
            print(f"🔎 Session ID requested: {session_id}")
            print(f"🔎 Question: {question}")
            print("🔍 Current session_store keys:", list(session_store.keys()))

            if not session_id:
                return JsonResponse({"error": "Missing session_id parameter"}, status=400)
            if not question:
                return JsonResponse({"error": "Missing question parameter"}, status=400)

            # 🔧 FIX 1: Better session lookup with fallback
            session_data = None
            actual_session_id = None
            
            if session_id in session_store:
                session_data = session_store[session_id]
                actual_session_id = session_id
                print(f"✅ Found exact session match: {session_id}")
            else:
                print(f"❌ Session {session_id} not found, searching in conversation history...")
                for stored_session_id, stored_data in session_store.items():
                    history = conversation_memory_store.get(stored_session_id, [])
                    for entry in history:
                        if entry.get("question", "").strip().lower() == question.strip().lower():
                            session_data = stored_data
                            actual_session_id = stored_session_id
                            print(f"✅ Found session via question match: {stored_session_id}")
                            break
                    if session_data:
                        break

            if not session_data:
                print(f"❌ No session found for question: {question}")
                return JsonResponse({
                    "error": f"Session ID {session_id} not found. Please run the query first via chat.",
                    "available_sessions": list(session_store.keys()),
                    "debug_info": f"Searched for question: '{question}'"
                }, status=404)

            try:
                user_id = session_data.get("user_id", "export_user")
                engine = session_data.get("engine")

                if not engine:
                    return JsonResponse({"error": "Database engine not found in session"}, status=500)

                print(f"✅ Session found. User ID: {user_id}, Actual Session: {actual_session_id}")

                history = conversation_memory_store.get(actual_session_id, [])
                sql = None

                for entry in reversed(history):
                    stored_question = entry.get("question", "").strip().lower()
                    search_question = question.strip().lower()
                    if stored_question == search_question or search_question in stored_question:
                        sql = entry.get("sql")
                        print(f"📋 Found SQL in history: {sql}")
                        break

                if not sql:
                    print("🔄 Generating new SQL for export...")
                    raw_response = run_sql_generation_graph(question, user_id=user_id, db_id=actual_session_id, history=history)
                    sql, _ = raw_response if isinstance(raw_response, tuple) else (extract_sql_block(raw_response), None)
                    print(f"🆕 Generated SQL: {sql}")

                if not sql or not sql.strip():
                    return JsonResponse({"error": "No SQL query generated"}, status=400)

                sql_lower = sql.strip().lower()
                if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
                    return JsonResponse({"error": "Invalid SQL query type"}, status=400)

                print(f"🚀 Executing SQL query...")
                with engine.connect() as conn:
                    result = conn.execute(text(sql))
                    rows = [dict(row._mapping) for row in result]

                print(f"✅ Query executed successfully. Found {len(rows)} rows")

                if not rows:
                    response = HttpResponse(content_type='text/csv')
                    response['Content-Disposition'] = 'attachment; filename="export_no_data.csv"'
                    response['Access-Control-Allow-Origin'] = '*'
                    response.write("No data found for this query")
                    return response

                from urllib.parse import quote
                timestamp = now().strftime("%Y%m%d_%H%M%S")
                filename = f"export_{timestamp}.csv"
                
                response = HttpResponse(content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename="{filename}"'
                response['Access-Control-Allow-Origin'] = '*'
                response['Access-Control-Allow-Headers'] = 'Content-Type'
                response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'

                writer = csv.DictWriter(response, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

                print(f"📁 CSV file created successfully with {len(rows)} rows")
                return response

            except Exception as e:
                print(f"❌ Export execution error: {str(e)}")
                traceback.print_exc()
                return JsonResponse({
                    "error": f"Export failed: {str(e)}",
                    "details": "Check server logs for more information",
                    "session_used": actual_session_id
                }, status=500)

        elif request.method == "POST":
            data = json.loads(request.body)
            session_id = data.get("session_id")
            question = data.get("question")
            user_id = data.get("user_id", "test_user_001")

            print(f"📨 POST request received")
            print(f"🔎 Session ID: {session_id}")
            print(f"🔎 Question: {question}")
            print(f"🔎 User ID: {user_id}")

            if not all([session_id, question]):
                return JsonResponse({"error": "Missing session_id or question"}, status=400)

            if session_id not in session_store:
                print(f"⚠️ Session {session_id} not found")

            history = conversation_memory_store.get(session_id, [])
            print(f"🧠 Running SQL generation with session_id={session_id}, user_id={user_id}")

            try:
                try:
                    VALID_COLUMNS = extract_columns_from_schema(FULL_SCHEMA)
                    validation_enabled = True
                except NameError:
                    validation_enabled = False

                # 🔹 RAG injection
                # context = retrieve_context(question)
                ctx = retrieve_context(question)
                context = ctx["sql"]
                augmented_question = f"""
                You are a SQL assistant. Use the following context (schemas, past queries, feedback):

                {context}

                User Question: {question}
                """

                raw_response = run_sql_generation_graph(augmented_question, user_id=user_id, db_id=session_id, history=history)

                if isinstance(raw_response, tuple) and len(raw_response) == 3:
                    sql_raw, recommendation, summary = raw_response
                else:
                    sql_raw, recommendation = raw_response if isinstance(raw_response, tuple) else (extract_sql_block(raw_response), None)
                    summary = ""

                sql = extract_sql_block(sql_raw)
                print("📝 Extracted SQL:", sql)
                sql = re.sub(r'\bLIMIT(\d+)', r'LIMIT \1', sql, flags=re.IGNORECASE)

                if validation_enabled:
                    invalid_cols = validate_sql_columns(sql, VALID_COLUMNS)
                    if invalid_cols:
                        return JsonResponse({
                            "answer": f"Invalid columns in SQL: {', '.join(invalid_cols)}",
                            "success": False,
                            "query_used": sql,
                            "rows": [],
                            "row_count": 0,
                            "session_id": session_id,
                            "response_time": "0.00s"
                        }, status=400)

            except Exception as sql_gen_error:
                print(f"❌ SQL generation error: {str(sql_gen_error)}")
                return JsonResponse({"answer": "Failed to generate SQL query","success": False,"error": str(sql_gen_error),
                    "rows": [],"row_count": 0,"session_id": session_id,"response_time": "0.00s"}, status=500)

            if not sql or not sql.strip().lower().startswith(("select", "with")):
                return JsonResponse({"answer": "Invalid or failed SQL generation","success": False,"query_used": sql or "No SQL generated",
                    "rows": [],"row_count": 0,"session_id": session_id,"response_time": "0.00s"}, status=500)

            if session_id not in session_store:
                return JsonResponse({"answer": "Session not found","success": False,
                    "error": f"Session ID {session_id} not found in session_store","rows": [],
                    "row_count": 0,"session_id": session_id,"response_time": "0.00s"}, status=404)

            engine = session_store[session_id]["engine"]

            try:
                print(f"✅ Executing SQL on session: {session_id}")
                with engine.connect() as conn:
                    result = conn.execute(text(sql))
                    rows = [dict(row._mapping) for row in result]
                print(f"✅ SQL executed successfully. Row count: {len(rows)}")

                summary = generate_summary_from_rows(question, sql, rows)
                narrative = llm_generate_narrative(question, rows)

            except Exception as e:
                print("❌ SQL Execution Error:", str(e))
                traceback.print_exc()
                return JsonResponse({"answer": "SQL execution failed.","success": False,"query_used": sql,"error": str(e),
                    "rows": [],"row_count": 0,"response_time": "0.00s","session_id": session_id}, status=500)

            answer = ""
            if rows:
                if len(rows) > 50:
                    answer = f"Found {len(rows)} results. Too many to display here - please download the full results using the download button."
                else:
                    formatted_rows = [", ".join(str(v) for v in row.values()) for row in rows[:3]]
                    answer = "\n".join(formatted_rows)
                    if len(rows) > 3:
                        answer += f"\n...and {len(rows) - 3} more rows."
            else:
                answer = "No data found."

            chart_config = None
            if rows and isinstance(rows[0], dict):
                try:
                    chart_config = llm_generate_chart_config(question, rows)
                    if chart_config and isinstance(chart_config, dict):
                        if 'series' not in chart_config or not chart_config['series']:
                            chart_config = None
                        else:
                            valid_series = []
                            for series in chart_config['series']:
                                if 'data' in series and series['data']:
                                    valid_series.append(series)
                            chart_config['series'] = valid_series if valid_series else None
                except Exception as chart_err:
                    print("⚠️ Chart generation failed:", chart_err)
                    chart_config = None

            try:
                if not recommendation:
                    recommendation = llm_generate_recommendation(question, rows)
            except Exception as rec_err:
                recommendation = "Could not generate recommendation at this time."

            conversation_memory_store.setdefault(session_id, []).append({
                "question": question,"sql": sql,"row_count": len(rows),"timestamp": now().isoformat()
            })

            if session_id in session_store:
                session_store[session_id]["user_id"] = user_id

            # 🔹 Store in Chroma
            # store_interaction_in_chroma(question, answer, sql, summary, recommendation, session_id, feedback="auto")
            store_interaction_in_chroma(question, answer, sql, summary, recommendation, session_id, feedback="auto")

            total_time = (now() - start_time).total_seconds()

            return JsonResponse({"answer": answer,"success": True,"query_used": sql,"rows": rows,
                "narrative": narrative,"chart_config": chart_config,"summary": summary,
                "row_count": len(rows),"recommendation": recommendation,
                "response_time": f"{total_time:.2f}s","session_id": session_id,
                "history": conversation_memory_store[session_id],"rag_debug": {   # ✅ Debug info
        "matched_question": ctx["matched_question"],
        "similarity_score": ctx["score"],
        "top_k_matches": ctx["top_k_debug"]
    }})

        elif request.method == "OPTIONS":
            response = HttpResponse()
            response['Access-Control-Allow-Origin'] = '*'
            response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
            return response

        else:
            return JsonResponse({"error": "Method not allowed. Use GET for export or POST for queries."}, status=405)

    except Exception as e:
        print("💥 Unexpected error in ask_question:")
        traceback.print_exc()
        return JsonResponse({"answer": "Something went wrong.","success": False,"error": str(e),
            "rows": [],"row_count": 0,"response_time": "0.00s",
            "session_id": request.GET.get("session_id") if request.method == "GET" else json.loads(request.body).get("session_id", "unknown") if request.method == "POST" else "unknown"}, status=500)





import os
import json
import re
import traceback
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from sqlalchemy import text
from django.utils.timezone import now
from .humanizer import humanize_narrative,safe_generate_narrative 

import json, time, traceback, re, datetime, uuid
from decimal import Decimal
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.serializers.json import DjangoJSONEncoder
from sqlalchemy import text

# ---------- Enhanced SQL Processing Helpers ----------
def safe_extract_sql_text(maybe_sql: object) -> str:
    text_ = str(maybe_sql or "").lstrip("\ufeff")
    if "```" in text_:
        try:
            unwrapped = extract_sql_block(text_)
            if isinstance(unwrapped, str) and unwrapped.strip():
                return unwrapped.strip()
        except Exception:
            pass
    m = re.search(r'(?is)\b(with|select)\b', text_)
    return text_[m.start():].strip() if m else text_.strip()

def extract_time_context_from_question(question: str) -> dict:
    """Extract time-related context from user questions to ensure proper SQL generation"""
    context = {}
    question_lower = question.lower()
    
    # Extract year ranges
    year_patterns = [
        r'in (\d{4})',
        r'during (\d{4})',
        r'for (\d{4})',
        r'from (\d{4}) to (\d{4})',
        r'between (\d{4}) and (\d{4})',
        r'(\d{4})-(\d{4})',
    ]
    
    for pattern in year_patterns:
        match = re.search(pattern, question_lower)
        if match:
            groups = match.groups()
            if len(groups) == 1:
                context['single_year'] = groups[0]
            elif len(groups) == 2:
                context['year_range'] = {'start': groups[0], 'end': groups[1]}
            break
    
    # Extract month/quarter context
    if 'quarter' in question_lower or 'q1' in question_lower or 'q2' in question_lower:
        context['time_granularity'] = 'quarter'
    elif 'month' in question_lower or 'monthly' in question_lower:
        context['time_granularity'] = 'month'
    elif 'year' in question_lower or 'annual' in question_lower:
        context['time_granularity'] = 'year'
    
    return context

def fix_cte_structure(sql: str) -> str:
    """Fix Common Table Expression (CTE) structure issues"""
    if not sql or not sql.strip():
        return sql
    
    # Check if this is a WITH query
    if not re.match(r'\s*WITH\s+', sql, re.IGNORECASE):
        return sql
    
    try:
        # Pattern to find nested WITH clauses that should be at top level
        # Look for cases where a CTE is defined inside another CTE
        nested_with_pattern = r'(\s*WITH\s+\w+\s+AS\s*\([^)]*?WITH\s+(\w+)\s+AS\s*\([^)]+\)[^)]*?\))'
        
        if re.search(nested_with_pattern, sql, re.IGNORECASE | re.DOTALL):
            print("🔧 Fixing nested CTE structure...")
            
            # Extract all CTE definitions and reorganize them
            cte_definitions = []
            main_query = sql
            
            # Find all CTE patterns
            cte_pattern = r'WITH\s+(\w+)\s+AS\s*\(([^)]+(?:\([^)]*\)[^)]*)*)\)'
            matches = list(re.finditer(cte_pattern, sql, re.IGNORECASE | re.DOTALL))
            
            if len(matches) > 1:
                # Multiple CTEs found - need to restructure
                all_ctes = []
                for match in matches:
                    cte_name = match.group(1)
                    cte_body = match.group(2).strip()
                    all_ctes.append(f"{cte_name} AS (\n{cte_body}\n)")
                
                # Find the main SELECT query (after all CTEs)
                last_match = matches[-1]
                remaining_sql = sql[last_match.end():].strip()
                
                # Look for the main SELECT
                main_select_match = re.search(r'SELECT\s+.*', remaining_sql, re.IGNORECASE | re.DOTALL)
                if main_select_match:
                    main_query = main_select_match.group(0)
                    
                    # Reconstruct the query with proper CTE structure
                    fixed_sql = "WITH " + ",\n".join(all_ctes) + "\n" + main_query
                    return fixed_sql
        
        return sql
        
    except Exception as e:
        print(f"⚠️ CTE structure fix failed: {e}")
        return sql

def repair_split_time_context(sql: str) -> str:
    """Enhanced to better handle time-based CTEs and fix CTE structure"""
    s = str(sql or "")
    low = s.lower()
    
    # First fix any CTE structure issues
    s = fix_cte_structure(s)
    
    # If already has WITH clause, check if time context is properly formed
    if low.startswith("with") or "time_context" not in low:
        return s
    
    # Look for broken CTE patterns
    m = re.search(r'\)\s*select\b', s, flags=re.IGNORECASE | re.DOTALL)
    if not (s.strip().lower().startswith("select") and m):
        return s
    
    paren_end = m.start()
    sel_after = re.search(r'\bselect\b', s[paren_end:], flags=re.IGNORECASE)
    if not sel_after:
        return s
    
    sel_idx = paren_end + sel_after.start()
    prefix = s[:paren_end].rstrip()
    rest = s[sel_idx:].lstrip()
    
    # Enhanced: ensure time filtering is preserved
    time_filters = re.findall(r'policy_end_date_year\s*[>=<]+\s*\d{4}', prefix, flags=re.IGNORECASE)
    if time_filters:
        return f"WITH time_context AS ({prefix})\n{rest}"
    
    return s
def repair_time_context_alias(sql: str) -> str:
    """Enhanced alias repair with better time context handling"""
    pat = re.compile(r'WITH\s+time_context\s+AS\s*\(\s*(.*?)\s*\)',
                     flags=re.IGNORECASE | re.DOTALL)
    
    def _fix(m):
        body = m.group(1)
        
        # Check if there's a table alias being used but not defined
        # Look for patterns like "mcl.column_name" in WHERE clause
        alias_usage = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*[a-zA-Z_]', body)
        
        if alias_usage:
            # Find the most common alias being used (usually 'mcl')
            from collections import Counter
            alias_counts = Counter(alias_usage)
            most_common_alias = alias_counts.most_common(1)[0][0] if alias_counts else 'mcl'
            
            # Check if the FROM clause already has an alias
            from_match = re.search(r'FROM\s+([^\s\)]+)(?:\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*))?\s*(?:WHERE|GROUP|ORDER|LIMIT|\)|$)', 
                                 body, flags=re.IGNORECASE)
            
            if from_match:
                table_name = from_match.group(1)
                existing_alias = from_match.group(2)
                
                if not existing_alias:
                    # No alias defined, but alias is being used - add the alias
                    body = re.sub(
                        r'(FROM\s+' + re.escape(table_name) + r')(\s+)',
                        r'\1 ' + most_common_alias + r'\2',
                        body,
                        flags=re.IGNORECASE
                    )
                    print(f"🔧 Added missing alias '{most_common_alias}' to table '{table_name}' in time_context CTE")
                elif existing_alias != most_common_alias:
                    # Different alias defined than what's being used - replace all usages
                    body = re.sub(
                        r'\b' + re.escape(most_common_alias) + r'\s*\.',
                        existing_alias + '.',
                        body
                    )
                    print(f"🔧 Replaced alias '{most_common_alias}' with '{existing_alias}' in time_context CTE")
        
        return f'WITH time_context AS (\n{body}\n)'
    
    return pat.sub(_fix, sql)

def fix_all_cte_aliases(sql: str) -> str:
    """Fix aliases in all CTEs, not just time_context"""
    # Pattern to match any CTE
    cte_pattern = re.compile(r'WITH\s+(\w+)\s+AS\s*\(\s*(.*?)\s*\)(?=\s*,|\s+SELECT|\s*$)',
                           flags=re.IGNORECASE | re.DOTALL)
    
    def fix_cte_alias(match):
        cte_name = match.group(1)
        cte_body = match.group(2)
        
        # Find alias usage patterns
        alias_usage = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*[a-zA-Z_]', cte_body)
        
        if alias_usage:
            from collections import Counter
            alias_counts = Counter(alias_usage)
            most_common_alias = alias_counts.most_common(1)[0][0] if alias_counts else None
            
            if most_common_alias:
                # Check FROM clause
                from_match = re.search(r'FROM\s+([^\s\)]+)(?:\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*))?\s*(?:WHERE|GROUP|ORDER|LIMIT|\)|$)', 
                                     cte_body, flags=re.IGNORECASE)
                
                if from_match:
                    table_name = from_match.group(1)
                    existing_alias = from_match.group(2)
                    
                    if not existing_alias:
                        # Add the missing alias
                        cte_body = re.sub(
                            r'(FROM\s+' + re.escape(table_name) + r')(\s+)',
                            r'\1 ' + most_common_alias + r'\2',
                            cte_body,
                            flags=re.IGNORECASE
                        )
                        print(f"🔧 Fixed missing alias '{most_common_alias}' in CTE '{cte_name}'")
        
        return f'WITH {cte_name} AS (\n{cte_body}\n)'
    
    return cte_pattern.sub(fix_cte_alias, sql)

def sanitize_sql_before_exec(sql: str, question: str = "") -> str:
    """Enhanced sanitization with comprehensive CTE alias fixing"""
    s = safe_extract_sql_text(sql)
    
    # Fix CTE structure issues first
    s = fix_cte_structure(s)
    
    # Fix all CTE aliases (enhanced version)
    s = fix_all_cte_aliases(s)
    
    # Then apply other fixes
    s = repair_split_time_context(s)
    
    # Add time context if missing
    if question:
        s = enhance_sql_with_time_context(s, question)
    
    # Fix LIMIT clause formatting
    s = re.sub(r'\blimit\s*(\d+)\b', r'LIMIT \1', s, flags=re.IGNORECASE)
    
    # Final validation - check for remaining alias issues
    if re.search(r'\bmcl\s*\.', s) and not re.search(r'FROM\s+[^\s]+\s+(?:AS\s+)?mcl\b', s, re.IGNORECASE):
        print("⚠️ Still found mcl alias usage without proper FROM clause definition")
        # Emergency fix: replace all mcl. with nothing if no proper alias found
        table_match = re.search(r'FROM\s+([^\s\)]+)', s, re.IGNORECASE)
        if table_match:
            table_name = table_match.group(1)
            # Add mcl alias to the main table reference
            s = re.sub(r'(FROM\s+' + re.escape(table_name) + r')(\s+)', r'\1 mcl\2', s, flags=re.IGNORECASE)
            print(f"🚨 Emergency fix: Added mcl alias to {table_name}")
    
    print(f"🔧 Final sanitized SQL:\n{s}")
    return s

def enhance_sql_with_time_context(sql: str, question: str) -> str:
    """Add proper time context to SQL based on user question"""
    time_context = extract_time_context_from_question(question)
    
    if not time_context:
        return sql
    
    # If SQL already has time filters, don't modify
    if re.search(r'policy_end_date_year\s*[>=<]', sql, flags=re.IGNORECASE):
        return sql
    
    # Add time filtering based on extracted context
    if 'single_year' in time_context:
        year = time_context['single_year']
        if 'WHERE' in sql.upper():
            sql = re.sub(r'WHERE', f'WHERE policy_end_date_year = {year} AND', sql, count=1, flags=re.IGNORECASE)
        else:
            # Find a good place to insert WHERE clause
            from_match = re.search(r'FROM\s+[^\s]+(?:\s+[a-zA-Z_][a-zA-Z0-9_]*)?', sql, flags=re.IGNORECASE)
            if from_match:
                insert_pos = from_match.end()
                sql = sql[:insert_pos] + f' WHERE policy_end_date_year = {year}' + sql[insert_pos:]
    
    elif 'year_range' in time_context:
        start_year = time_context['year_range']['start']
        end_year = time_context['year_range']['end']
        if 'WHERE' in sql.upper():
            sql = re.sub(r'WHERE', f'WHERE policy_end_date_year BETWEEN {start_year} AND {end_year} AND', 
                        sql, count=1, flags=re.IGNORECASE)
        else:
            from_match = re.search(r'FROM\s+[^\s]+(?:\s+[a-zA-Z_][a-zA-Z0-9_]*)?', sql, flags=re.IGNORECASE)
            if from_match:
                insert_pos = from_match.end()
                sql = sql[:insert_pos] + f' WHERE policy_end_date_year BETWEEN {start_year} AND {end_year}' + sql[insert_pos:]
    
    return sql

# def sanitize_sql_before_exec(sql: str, question: str = "") -> str:
#     """Enhanced sanitization with time context awareness and CTE fixing"""
#     s = safe_extract_sql_text(sql)
    
#     # Fix CTE structure issues first
#     s = fix_cte_structure(s)
    
#     # Then apply other fixes
#     s = repair_split_time_context(s)
#     s = repair_time_context_alias(s)
    
#     # Add time context if missing
#     if question:
#         s = enhance_sql_with_time_context(s, question)
    
#     # Fix LIMIT clause formatting
#     s = re.sub(r'\blimit\s*(\d+)\b', r'LIMIT \1', s, flags=re.IGNORECASE)
    
#     print(f"🔧 Sanitized SQL:\n{s}")
#     return s


def _jsonable(x):
    """Recursively coerce DB types into JSON-safe values."""
    try:
        import numpy as _np
        _np_types = ( _np.integer, _np.floating, _np.bool_, _np.ndarray )
    except Exception:
        _np_types = tuple()

    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Decimal):
        return int(x) if x == int(x) else float(x)
    if isinstance(x, (datetime.date, datetime.datetime, datetime.time)):
        return x.isoformat()
    if isinstance(x, (uuid.UUID, bytes)):
        return str(x)
    if _np_types and isinstance(x, _np_types):
        try:
            return _jsonable(x.item())
        except Exception:
            try:
                return [_jsonable(v) for v in x.tolist()]
            except Exception:
                return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    return str(x)

def _line(event: str, **payload) -> str:
    obj = {"event": event}
    obj.update(payload)
    return json.dumps(_jsonable(obj), cls=DjangoJSONEncoder, ensure_ascii=False) + "\n"

def _ev(event: str, **payload):
    return _line(event, **payload)

# ---------- Enhanced Humanizer for Conversational Tone ----------
def generate_conversational_opener(question: str, context: dict = None) -> str:
    """Generate a completely dynamic, LLM-powered conversational opener"""
    from .humanizer import generate_dynamic_conversational_opener, _derive_contextual_metrics
    
    # Create rich metrics for dynamic generation
    metrics = {
        "row_count": 0,  # Will be updated when we have results
        "data_size": "unknown",
        "has_data": True,
        "time_context": context.get('time_context') if context else {}
    }
    
    return generate_dynamic_conversational_opener(question, metrics)


# ✅ Load threshold from .env or fallback to 0.85
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", 0.85))
print(f"threshold={RAG_SIMILARITY_THRESHOLD:.1f}")
# === Streaming Ask Function ===


@csrf_exempt
def ask_question_stream(request):
    if request.method == "OPTIONS":
        r = HttpResponse()
        r["Access-Control-Allow-Origin"] = "*"
        r["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        r["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
        return r

    if request.method != "POST":
        return JsonResponse({"error": "Use POST"}, status=405)

    try:
        payload = json.loads(request.body or "{}")
        session_id = payload.get("session_id")
        question = payload.get("question")
        user_id = payload.get("user_id", "stream_user")

        if not session_id or not question:
            return JsonResponse({"error": "Missing session_id or question"}, status=400)
        if session_id not in session_store:
            return JsonResponse({"error": f"Session {session_id} not found"}, status=404)

        engine = session_store[session_id]["engine"]
        hist = conversation_memory_store.get(session_id, [])
        t0 = now()

        def gen():
            # ✅ Send a human-like opener immediately
            try:
                preview_narrative = safe_generate_narrative(question, sql="", rows=[])
                opener = preview_narrative.get("opener", "")
                if not opener or len(opener) < 10:
                    opener = "Let me analyze your question and gather insights for you."
                else:
                    opener_lower = opener.lower()
                    bad_phrases = [
                        "no data", "no records", "nothing found",
                        "unfortunately", "however", "but when i", "aren't any"
                    ]
                    if any(bp in opener_lower for bp in bad_phrases):
                        opener = "Let me analyze your question and gather insights for you."

                yield jsonl_line({"event": "narrative_opener", "text": opener})
            except Exception as e:
                print(f"⚠️ Early opener generation failed: {e}")
                opener = "Let me analyze your question and gather insights for you."
                yield jsonl_line({"event": "narrative_opener", "text": opener})

            yield jsonl_line({"event": "phase", "message": "Understanding your question…"})

            try:
                ctx = retrieve_context(question)
                score = ctx.get("score", 0.0) or 0.0
                matched_q = ctx.get("matched_question", "")

                # 🔎 Debug log
                print("🔎 [RAG DEBUG] Top-k matches:", json.dumps({
                    "matched_question": matched_q,
                    "similarity_score": score,
                    "top_k_matches": ctx.get("top_k_debug"),
                }, indent=2), flush=True)

                # Emit debug
                yield jsonl_line({
                    "event": "rag_debug",
                    "matches": {
                        "matched_question": matched_q,
                        "similarity_score": score,
                        "top_k_matches": ctx.get("top_k_debug"),
                    }
                })

                sql = None
                if ctx.get("sql"):
                    q_norm = normalize_question(question)
                    matched_norm = normalize_question(matched_q)
                    print(f"🔎 [RAG DEBUG] Normalized question: {q_norm}, matched question: {matched_norm}")

                    if q_norm == matched_norm:
                        sql = ctx["sql"]
                        print(f"♻️ Reusing SQL (EXACT match) for: {matched_q}")
                    elif score >= RAG_SIMILARITY_THRESHOLD:
                        sql = ctx["sql"]
                        print(f"♻️ Reusing SQL (score={score:.2f}) for: {matched_q}")
                    else:
                        print(f"🆕 [NEW SQL GEN] No strong match (score={score:.2f}, threshold={RAG_SIMILARITY_THRESHOLD}). Generating fresh SQL…")
                else:
                    print("🆕 [NEW SQL GEN] No SQL found in memory, generating fresh SQL…")

                # Generate if no reuse
                if not sql:
                    context = ctx.get("sql", "")
                    augmented_question = f"""
                    You are a SQL assistant. Use the following context (schemas, past queries, feedback):

                    {context}

                    User Question: {question}
                    """
                    raw = run_sql_generation_graph(
                        augmented_question, user_id=user_id, db_id=session_id, history=hist
                    )
                    sql_raw = raw[0] if isinstance(raw, tuple) else raw
                    sql = extract_sql_block(sql_raw or "")

                sql = re.sub(r"\bLIMIT(\d+)", r"LIMIT \1", sql, flags=re.IGNORECASE)

            except Exception as e:
                traceback.print_exc()
                yield jsonl_line({"event": "error", "message": f"SQL generation failed: {e}"})
                return

            if not sql or not sql.strip().lower().startswith(("select", "with")):
                yield jsonl_line({"event": "error", "message": "Invalid or empty SQL generated"})
                return

            yield jsonl_line({"event": "sql", "sql": sql})

            # Run SQL
            yield jsonl_line({"event": "phase", "message": "Querying the database…"})
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(sql))
                    rows = [dict(row._mapping) for row in result]
                print(f"✅ SQL executed. Row count: {len(rows)}")
            except Exception as e:
                traceback.print_exc()
                yield jsonl_line({"event": "error", "message": f"SQL execution failed: {e}"})
                return

            row_count = len(rows)
            yield jsonl_line({"event": "rows_preview", "rows": rows[:8], "row_count": row_count})

            yield jsonl_line({"event": "phase", "message": "Analyzing results…"})
            try:
                # ✅ Human-like narrative instead of static summary
                narrative_obj = safe_generate_narrative(question, sql, rows)
                if not opener:
                    opener = narrative_obj.get("opener", opener)
                insights = narrative_obj.get("insights", [])
                recs = narrative_obj.get("recommendations", [])
                next_step = narrative_obj.get("next_step", "")

                # Emit each component separately if frontend supports it
                if insights:
                    yield jsonl_line({"event": "insights", "list": insights})
                if recs:
                    yield jsonl_line({"event": "recommendations", "list": recs})
                if next_step:
                    yield jsonl_line({"event": "next_step", "text": next_step})

                # Also keep a flat summary (for backward compatibility)
                summary = " ".join(insights) if insights else opener

            except Exception as e:
                print(f"⚠️ Narrative generation failed: {e}")
                summary = ""

            try:
                narr = llm_generate_narrative(question, rows)
                yield jsonl_line({"event": "narrative", "obj": narr})
            except Exception:
                narr = None

            try:
                rec = llm_generate_recommendation(question, rows)
                if rec:
                    yield jsonl_line({"event": "recommendation", "text": rec})
            except Exception:
                rec = None

            try:
                cfg = llm_generate_chart_config(question, rows)
                if cfg:
                    yield jsonl_line({"event": "chart", "config": cfg})
            except Exception:
                cfg = None

            # Format final answer
            if row_count == 0:
                answer = "No data found."
            elif row_count > 50:
                answer = f"Found {row_count} results. Too many to display here — use the CSV download."
            else:
                first = [", ".join(str(v) for v in r.values()) for r in rows[:3]]
                answer = "\n".join(first)
                if row_count > 3:
                    answer += f"\n...and {row_count - 3} more rows."

            # Update session + store
            conversation_memory_store.setdefault(session_id, []).append({
                "question": question, "sql": sql, "row_count": row_count, "timestamp": now().isoformat()
            })
            session_store[session_id]["user_id"] = user_id

            store_interaction_in_chroma(question=question,
                answer=answer,
                sql=sql,
                summary=summary,
                recommendation=rec,
                session_id=session_id,
                feedback="auto")

            total = (now() - t0).total_seconds()
            yield jsonl_line({
                "event": "final",
                "payload": {
                    "answer": answer,
                    "summary": summary,
                    "rows": rows[:50],
                    "row_count": row_count,
                    "chart_config": cfg,
                    "recommendation": rec,
                    "narrative": narr,
                    "query_used": sql,
                    "response_time": f"{total:.2f}s",
                    "session_id": session_id,
                    "rag_debug": {
                        "matched_question": ctx.get("matched_question"),
                        "similarity_score": score,
                        "top_k_matches": ctx.get("top_k_debug")
                    },
                    "history": conversation_memory_store.get(session_id, []),
                    "conversational_opener": opener  # ✅ store human opener
                },
            })

        resp = StreamingHttpResponse(gen(), content_type="application/x-ndjson")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        resp["Access-Control-Allow-Origin"] = "*"
        return resp

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def ask_question_stream169(request):
    if request.method == "OPTIONS":
        r = HttpResponse()
        r["Access-Control-Allow-Origin"] = "*"
        r["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        r["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
        return r

    if request.method != "POST":
        return JsonResponse({"error": "Use POST"}, status=405)

    try:
        payload = json.loads(request.body or "{}")
        session_id = payload.get("session_id")
        question = payload.get("question")
        user_id = payload.get("user_id", "stream_user")

        if not session_id or not question:
            return JsonResponse({"error": "Missing session_id or question"}, status=400)
        if session_id not in session_store:
            return JsonResponse({"error": f"Session {session_id} not found"}, status=404)

        engine = session_store[session_id]["engine"]
        hist = conversation_memory_store.get(session_id, [])
        t0 = now()

        def gen():
            yield jsonl_line({"event": "phase", "message": "Understanding your question…"})

            try:
                ctx = retrieve_context(question)
                score = ctx.get("score", 0.0) or 0.0
                matched_q = ctx.get("matched_question", "")

                # 🔎 Debug log
                print("🔎 [RAG DEBUG] Top-k matches:", json.dumps({
                    "matched_question": matched_q,
                    "similarity_score": score,
                    "top_k_matches": ctx.get("top_k_debug"),
                }, indent=2), flush=True)

                # Emit debug
                yield jsonl_line({
                    "event": "rag_debug",
                    "matches": {
                        "matched_question": matched_q,
                        "similarity_score": score,
                        "top_k_matches": ctx.get("top_k_debug"),
                    }
                })

                sql = None
                if ctx.get("sql"):
                    q_norm = normalize_question(question)
                    matched_norm = normalize_question(matched_q)
                    print(f"🔎 [RAG DEBUG] Normalized question: {q_norm}, matched question: {matched_norm}")

                    if q_norm == matched_norm:
                        sql = ctx["sql"]
                        print(f"♻️ Reusing SQL (EXACT match) for: {matched_q}")
                    elif score >= RAG_SIMILARITY_THRESHOLD:
                        sql = ctx["sql"]
                        print(f"♻️ Reusing SQL (score={score:.2f}) for: {matched_q}")
                    else:
                        print(f"🆕 [NEW SQL GEN] No strong match (score={score:.2f}, threshold={RAG_SIMILARITY_THRESHOLD}). Generating fresh SQL…")
                else:
                    print("🆕 [NEW SQL GEN] No SQL found in memory, generating fresh SQL…")

                # Generate if no reuse
                if not sql:
                    context = ctx.get("sql", "")
                    augmented_question = f"""
                    You are a SQL assistant. Use the following context (schemas, past queries, feedback):

                    {context}

                    User Question: {question}
                    """
                    raw = run_sql_generation_graph(
                        augmented_question, user_id=user_id, db_id=session_id, history=hist
                    )
                    sql_raw = raw[0] if isinstance(raw, tuple) else raw
                    sql = extract_sql_block(sql_raw or "")

                sql = re.sub(r"\bLIMIT(\d+)", r"LIMIT \1", sql, flags=re.IGNORECASE)

            except Exception as e:
                traceback.print_exc()
                yield jsonl_line({"event": "error", "message": f"SQL generation failed: {e}"})
                return

            if not sql or not sql.strip().lower().startswith(("select", "with")):
                yield jsonl_line({"event": "error", "message": "Invalid or empty SQL generated"})
                return

            yield jsonl_line({"event": "sql", "sql": sql})

            # Run SQL
            yield jsonl_line({"event": "phase", "message": "Querying the database…"})
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(sql))
                    rows = [dict(row._mapping) for row in result]
                print(f"✅ SQL executed. Row count: {len(rows)}")
            except Exception as e:
                traceback.print_exc()
                yield jsonl_line({"event": "error", "message": f"SQL execution failed: {e}"})
                return

            row_count = len(rows)
            yield jsonl_line({"event": "rows_preview", "rows": rows[:8], "row_count": row_count})

            # yield jsonl_line({"event": "phase", "message": "Analyzing results…"})
            # try:
            #     summary = generate_summary_from_rows(question, sql, rows)
            #     if summary:
            #         yield jsonl_line({"event": "summary", "text": summary})
            # except Exception:
            #     summary = ""

            yield jsonl_line({"event": "phase", "message": "Analyzing results…"})
            try:
                # ✅ Human-like narrative instead of static summary
                narrative_obj = safe_generate_narrative(question, sql, rows)
                opener = narrative_obj.get("opener", "")
                insights = narrative_obj.get("insights", [])
                recs = narrative_obj.get("recommendations", [])
                next_step = narrative_obj.get("next_step", "")

                # Emit each component separately if frontend supports it
                if opener:
                    yield jsonl_line({"event": "narrative_opener", "text": opener})
                if insights:
                    yield jsonl_line({"event": "insights", "list": insights})
                if recs:
                    yield jsonl_line({"event": "recommendations", "list": recs})
                if next_step:
                    yield jsonl_line({"event": "next_step", "text": next_step})

                # Also keep a flat summary (for backward compatibility)
                summary = " ".join(insights) if insights else opener

            except Exception as e:
                print(f"⚠️ Narrative generation failed: {e}")
                summary = ""


            try:
                narr = llm_generate_narrative(question, rows)
                yield jsonl_line({"event": "narrative", "obj": narr})
            except Exception:
                narr = None

            try:
                rec = llm_generate_recommendation(question, rows)
                if rec:
                    yield jsonl_line({"event": "recommendation", "text": rec})
            except Exception:
                rec = None

            try:
                cfg = llm_generate_chart_config(question, rows)
                if cfg:
                    yield jsonl_line({"event": "chart", "config": cfg})
            except Exception:
                cfg = None

            # Format final answer
            if row_count == 0:
                answer = "No data found."
            elif row_count > 50:
                answer = f"Found {row_count} results. Too many to display here — use the CSV download."
            else:
                first = [", ".join(str(v) for v in r.values()) for r in rows[:3]]
                answer = "\n".join(first)
                if row_count > 3:
                    answer += f"\n...and {row_count - 3} more rows."

            # Update session + store
            conversation_memory_store.setdefault(session_id, []).append({
                "question": question, "sql": sql, "row_count": row_count, "timestamp": now().isoformat()
            })
            session_store[session_id]["user_id"] = user_id

            store_interaction_in_chroma(question=question,
                answer=answer,
                sql=sql,
                summary=summary,
                recommendation=rec,
                session_id=session_id,
                feedback="auto")

            total = (now() - t0).total_seconds()
            yield jsonl_line({
                "event": "final",
                "payload": {
                    "answer": answer,
                    "summary": summary,
                    "rows": rows[:50],
                    "row_count": row_count,
                    "chart_config": cfg,
                    "recommendation": rec,
                    "narrative": narr,
                    "query_used": sql,
                    "response_time": f"{total:.2f}s",
                    "session_id": session_id,
                    "rag_debug": {
                        "matched_question": ctx.get("matched_question"),
                        "similarity_score": score,
                        "top_k_matches": ctx.get("top_k_debug")
                    }
                },
            })

        resp = StreamingHttpResponse(gen(), content_type="application/x-ndjson")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        resp["Access-Control-Allow-Origin"] = "*"
        return resp

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


# 159 working code bfr threshhold
@csrf_exempt
def ask_question_stream159(request):
    if request.method == "OPTIONS":
        r = HttpResponse()
        r["Access-Control-Allow-Origin"] = "*"
        r["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        r["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
        return r

    if request.method != "POST":
        return JsonResponse({"error": "Use POST"}, status=405)

    try:
        payload = json.loads(request.body or "{}")
        session_id = payload.get("session_id")
        question = payload.get("question")
        user_id = payload.get("user_id", "stream_user")

        if not session_id or not question:
            return JsonResponse({"error": "Missing session_id or question"}, status=400)
        if session_id not in session_store:
            return JsonResponse({"error": f"Session {session_id} not found"}, status=404)

        engine = session_store[session_id]["engine"]
        hist = conversation_memory_store.get(session_id, [])
        t0 = now()

        def gen():
            yield jsonl_line({"event": "phase", "message": "Understanding your question…"})

            try:
                # context = retrieve_context(question)
                ctx = retrieve_context(question)

                # --- 🔎 Print in backend ---
                print("🔎 [RAG DEBUG] Top-k matches:", json.dumps({
                    "matched_question": ctx.get("matched_question"),
                    "similarity_score": ctx.get("score"),
                    "top_k_matches": ctx.get("top_k_debug"),
                }, indent=2), flush=True)

                # --- 🔎 Emit to frontend as first event ---
                yield jsonl_line({
                    "event": "rag_debug",
                    "matches": {
                        "matched_question": ctx.get("matched_question"),
                        "similarity_score": ctx.get("score"),
                        "top_k_matches": ctx.get("top_k_debug"),
                    }
                })

                # ctx = retrieve_context(question)
                context = ctx["sql"]
                augmented_question = f"""
                You are a SQL assistant. Use the following context (schemas, past queries, feedback):

                {context}

                User Question: {question}
                """
                raw = run_sql_generation_graph(augmented_question, user_id=user_id, db_id=session_id, history=hist)
                sql_raw = raw[0] if isinstance(raw, tuple) else raw
                sql = extract_sql_block(sql_raw or "")
                sql = re.sub(r"\bLIMIT(\d+)", r"LIMIT \1", sql, flags=re.IGNORECASE)
            except Exception as e:
                yield jsonl_line({"event": "error", "message": f"SQL generation failed: {e}"})
                return

            if not sql or not sql.strip().lower().startswith(("select", "with")):
                yield jsonl_line({"event": "error", "message": "Invalid or empty SQL generated"})
                return
            yield jsonl_line({"event": "sql", "sql": sql})

            yield jsonl_line({"event": "phase", "message": "Querying the database…"})
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(sql))
                    rows = [dict(row._mapping) for row in result]
            except Exception as e:
                yield jsonl_line({"event": "error", "message": f"SQL execution failed: {e}"})
                return

            row_count = len(rows)
            yield jsonl_line({"event": "rows_preview", "rows": rows[:8], "row_count": row_count})

            yield jsonl_line({"event": "phase", "message": "Analyzing results…"})
            try:
                summary = generate_summary_from_rows(question, sql, rows)
                if summary:
                    yield jsonl_line({"event": "summary", "text": summary})
            except Exception:
                summary = ""

            try:
                narr = llm_generate_narrative(question, rows)
                yield jsonl_line({"event": "narrative", "obj": narr})
            except Exception:
                narr = None

            try:
                rec = llm_generate_recommendation(question, rows)
                if rec:
                    yield jsonl_line({"event": "recommendation", "text": rec})
            except Exception:
                rec = None

            try:
                cfg = llm_generate_chart_config(question, rows)
                if cfg:
                    yield jsonl_line({"event": "chart", "config": cfg})
            except Exception:
                cfg = None

            if row_count == 0:
                answer = "No data found."
            elif row_count > 50:
                answer = f"Found {row_count} results. Too many to display here — use the CSV download."
            else:
                first = [", ".join(str(v) for v in r.values()) for r in rows[:3]]
                answer = "\n".join(first)
                if row_count > 3:
                    answer += f"\n...and {row_count - 3} more rows."

            conversation_memory_store.setdefault(session_id, []).append({
                "question": question, "sql": sql, "row_count": row_count, "timestamp": now().isoformat()
            })
            session_store[session_id]["user_id"] = user_id

            # 🔹 Store interaction in Chroma
            # store_interaction_in_chroma(question, answer, sql, summary, rec, session_id, feedback="auto")
            # 🔹 Store interaction in Chroma
            store_interaction_in_chroma(question, answer, sql, summary, rec, session_id, feedback="auto")


            total = (now() - t0).total_seconds()
            yield jsonl_line({
                "event": "final",
                "payload": {
                    "answer": answer,
                    "summary": summary,
                    "rows": rows[:50],
                    "row_count": row_count,
                    "chart_config": cfg,
                    "recommendation": rec,
                    "narrative": narr,
                    "query_used": sql,
                    "response_time": f"{total:.2f}s",
                    "session_id": session_id,
                    "rag_debug": {   # ✅ include debug info here
            "matched_question": ctx.get("matched_question"),
            "similarity_score": ctx.get("score"),
            "top_k_matches": ctx.get("top_k_debug")
        }
                },
            })

        resp = StreamingHttpResponse(gen(), content_type="application/x-ndjson")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        resp["Access-Control-Allow-Origin"] = "*"
        return resp

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)



from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def inspect_chroma(request):
    """
    Inspect all documents stored in Chroma DB.
    Returns question, sql, feedback, summary, recommendation, etc.
    """
    try:
        # Fetch all docs from Chroma
        docs = vector_db.get()

        results = []
        for doc, meta in zip(docs["documents"], docs["metadatas"]):
            results.append({
                "question": doc,
                "sql": meta.get("sql"),
                "answer": meta.get("answer"),
                "summary": meta.get("summary"),
                "recommendation": meta.get("recommendation"),
                "session_id": meta.get("session_id"),
                "feedback": meta.get("feedback", "unknown")
            })

        return JsonResponse({
            "count": len(results),
            "results": results
        }, safe=False)

    except Exception as e:
        print("❌ Error in inspect_chroma:", str(e))
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def inspect_general_chroma(request):
    try:
        docs = general_db.get()
        out = []
        for doc_id, meta, text in zip(docs["ids"], docs["metadatas"], docs["documents"]):
            out.append({
                "doc_id": doc_id,
                "question": text,
                "answer": meta.get("answer"),
                "feedback": meta.get("feedback"),
                "session_id": meta.get("session_id"),
                "timestamp": meta.get("timestamp")
            })
        return JsonResponse(out, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)



import csv
from django.http import HttpResponse
@csrf_exempt
def ask_questionworkingbftrain(request):
    try:
        start_time = now()

        # ✅ Handle GET request for CSV export
        if request.method == "GET" and request.GET.get("export") == "true":
            session_id = request.GET.get("session_id")
            question = request.GET.get("question")

            print("🔍 Export request received")
            print(f"🔎 Session ID requested: {session_id}")
            print(f"🔎 Question: {question}")
            print("🔍 Current session_store keys:", list(session_store.keys()))

            if not session_id:
                return JsonResponse({"error": "Missing session_id parameter"}, status=400)
            if not question:
                return JsonResponse({"error": "Missing question parameter"}, status=400)

            # 🔧 FIX 1: Better session lookup with fallback
            session_data = None
            actual_session_id = None
            
            # Try exact match first
            if session_id in session_store:
                session_data = session_store[session_id]
                actual_session_id = session_id
                print(f"✅ Found exact session match: {session_id}")
            else:
                # 🔧 FIX 2: Fallback - look for any session with the same user query in history
                print(f"❌ Session {session_id} not found, searching in conversation history...")
                for stored_session_id, stored_data in session_store.items():
                    history = conversation_memory_store.get(stored_session_id, [])
                    for entry in history:
                        if entry.get("question", "").strip().lower() == question.strip().lower():
                            session_data = stored_data
                            actual_session_id = stored_session_id
                            print(f"✅ Found session via question match: {stored_session_id}")
                            break
                    if session_data:
                        break

            if not session_data:
                print(f"❌ No session found for question: {question}")
                print(f"🔍 Available sessions: {list(session_store.keys())}")
                return JsonResponse({
                    "error": f"Session ID {session_id} not found. Please run the query first via chat.",
                    "available_sessions": list(session_store.keys()),
                    "debug_info": f"Searched for question: '{question}'"
                }, status=404)

            try:
                user_id = session_data.get("user_id", "export_user")
                engine = session_data.get("engine")

                if not engine:
                    return JsonResponse({"error": "Database engine not found in session"}, status=500)

                print(f"✅ Session found. User ID: {user_id}, Actual Session: {actual_session_id}")

                history = conversation_memory_store.get(actual_session_id, [])
                sql = None

                # 🔧 FIX 3: More flexible question matching
                for entry in reversed(history):
                    stored_question = entry.get("question", "").strip().lower()
                    search_question = question.strip().lower()
                    
                    # Try exact match first, then partial match
                    if stored_question == search_question or search_question in stored_question:
                        sql = entry.get("sql")
                        print(f"📋 Found SQL in history: {sql}")
                        break

                if not sql:
                    print("🔄 Generating new SQL for export...")
                    try:
                        raw_response = run_sql_generation_graph(question, user_id=user_id, db_id=actual_session_id, history=history)
                        sql, _ = raw_response if isinstance(raw_response, tuple) else (extract_sql_block(raw_response), None)
                        print(f"🆕 Generated SQL: {sql}")
                    except Exception as gen_error:
                        print(f"❌ SQL generation failed: {str(gen_error)}")
                        return JsonResponse({"error": f"Failed to generate SQL: {str(gen_error)}"}, status=500)

                if not sql or not sql.strip():
                    return JsonResponse({"error": "No SQL query generated"}, status=400)

                # 🔧 FIX 4: More flexible SQL validation
                sql_lower = sql.strip().lower()
                if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
                    return JsonResponse({"error": "Invalid SQL query type"}, status=400)

                print(f"🚀 Executing SQL query...")
                with engine.connect() as conn:
                    result = conn.execute(text(sql))
                    rows = [dict(row._mapping) for row in result]

                print(f"✅ Query executed successfully. Found {len(rows)} rows")

                if not rows:
                    # 🔧 FIX 5: Return empty CSV instead of plain text
                    response = HttpResponse(content_type='text/csv')
                    response['Content-Disposition'] = 'attachment; filename="export_no_data.csv"'
                    response['Access-Control-Allow-Origin'] = '*'
                    response.write("No data found for this query")
                    return response

                # 🔧 FIX 6: Better filename with timestamp
                from urllib.parse import quote
                timestamp = now().strftime("%Y%m%d_%H%M%S")
                filename = f"export_{timestamp}.csv"
                
                response = HttpResponse(content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename="{filename}"'
                response['Access-Control-Allow-Origin'] = '*'
                response['Access-Control-Allow-Headers'] = 'Content-Type'
                response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'

                writer = csv.DictWriter(response, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

                print(f"📁 CSV file created successfully with {len(rows)} rows")
                return response

            except Exception as e:
                print(f"❌ Export execution error: {str(e)}")
                traceback.print_exc()
                return JsonResponse({
                    "error": f"Export failed: {str(e)}",
                    "details": "Check server logs for more information",
                    "session_used": actual_session_id
                }, status=500)

        elif request.method == "POST":
            data = json.loads(request.body)
            session_id = data.get("session_id")
            question = data.get("question")
            user_id = data.get("user_id", "test_user_001")

            print(f"📨 POST request received")
            print(f"🔎 Session ID: {session_id}")
            print(f"🔎 Question: {question}")
            print(f"🔎 User ID: {user_id}")

            if not all([session_id, question]):
                return JsonResponse({"error": "Missing session_id or question"}, status=400)

            if session_id not in session_store:
                print(f"⚠️ Session {session_id} not found")

            history = conversation_memory_store.get(session_id, [])
            print(f"🧠 Running SQL generation with session_id={session_id}, user_id={user_id}")

            try:
                # 🔧 FIX 7: Handle column validation gracefully
                try:
                    VALID_COLUMNS = extract_columns_from_schema(FULL_SCHEMA)
                    validation_enabled = True
                except NameError:
                    print("⚠️ Column validation not available")
                    validation_enabled = False

                # raw_response = run_sql_generation_graph(question, user_id=user_id, db_id=session_id, history=history)

                # if isinstance(raw_response, tuple):
                #     sql_raw, recommendation = raw_response
                # else:
                #     sql_raw, recommendation = raw_response, None

                raw_response = run_sql_generation_graph(question, user_id=user_id, db_id=session_id, history=history)

                if isinstance(raw_response, tuple) and len(raw_response) == 3:
                    sql_raw, recommendation, summary = raw_response
                else:
                    sql_raw, recommendation = raw_response if isinstance(raw_response, tuple) else (extract_sql_block(raw_response), None)
                    summary = ""

                sql = extract_sql_block(sql_raw)
                print("📝 Extracted SQL:", sql)
                # ✅ Fix spacing issues in LIMIT clauses (e.g., LIMIT1 → LIMIT 1)
                sql = re.sub(r'\bLIMIT(\d+)', r'LIMIT \1', sql, flags=re.IGNORECASE)


                if validation_enabled:
                    invalid_cols = validate_sql_columns(sql, VALID_COLUMNS)
                    if invalid_cols:
                        return JsonResponse({
                            "answer": f"Invalid columns in SQL: {', '.join(invalid_cols)}",
                            "success": False,
                            "query_used": sql,
                            "rows": [],
                            "row_count": 0,
                            "session_id": session_id,
                            "response_time": "0.00s"
                        }, status=400)

            except Exception as sql_gen_error:
                print(f"❌ SQL generation error: {str(sql_gen_error)}")
                return JsonResponse({
                    "answer": "Failed to generate SQL query",
                    "success": False,
                    "error": str(sql_gen_error),
                    "rows": [],
                    "row_count": 0,
                    "session_id": session_id,
                    "response_time": "0.00s"
                }, status=500)

            if not sql or not sql.strip().lower().startswith(("select", "with")):
                return JsonResponse({
                    "answer": "Invalid or failed SQL generation",
                    "success": False,
                    "query_used": sql or "No SQL generated",
                    "rows": [],
                    "row_count": 0,
                    "session_id": session_id,
                    "response_time": "0.00s"
                }, status=500)

            if session_id not in session_store:
                return JsonResponse({
                    "answer": "Session not found",
                    "success": False,
                    "error": f"Session ID {session_id} not found in session_store",
                    "rows": [],
                    "row_count": 0,
                    "session_id": session_id,
                    "response_time": "0.00s"
                }, status=404)

            engine = session_store[session_id]["engine"]

            try:
                print(f"✅ Executing SQL on session: {session_id}")
                with engine.connect() as conn:
                    result = conn.execute(text(sql))
                    rows = [dict(row._mapping) for row in result]
                print(f"✅ SQL executed successfully. Row count: {len(rows)}")

                # ✅ Generate business-friendly summary from real SQL output
                summary = generate_summary_from_rows(question, sql, rows)
                narrative = llm_generate_narrative(question, rows)

            except Exception as e:
                print("❌ SQL Execution Error:", str(e))
                traceback.print_exc()
                return JsonResponse({
                    "answer": "SQL execution failed.",
                    "success": False,
                    "query_used": sql,
                    "error": str(e),
                    "rows": [],
                    "row_count": 0,
                    "response_time": "0.00s",
                    "session_id": session_id
                }, status=500)

            # 🔧 FIX 8: Better answer formatting
            # if len(rows) == 1 and len(rows[0]) == 1:
            #     answer = f"The result is {list(rows[0].values())[0]}."
            # elif rows:
            #     if len(rows) > 50:
            #         answer = f"Found {len(rows)} results. Too many to display here - please download the full results using the download button."
            #     else:
            #         # Show first 3 rows in a more readable format
            #         answer_parts = []
            #         for i, row in enumerate(rows[:3]):
            #             row_str = ", ".join(f"{k}: {v}" for k, v in row.items())
            #             answer_parts.append(f"Row {i+1}: {row_str}")
                    
            #         answer = "; ".join(answer_parts)
            #         if len(rows) > 3:
            #             answer += f" ...and {len(rows) - 3} more rows."
            # else:
            #     answer = "No data found."

            answer = ""
            if rows:
                if len(rows) > 50:
                    answer = f"Found {len(rows)} results. Too many to display here - please download the full results using the download button."
                else:
                    formatted_rows = [", ".join(str(v) for v in row.values()) for row in rows[:3]]
                    answer = "\n".join(formatted_rows)
                    if len(rows) > 3:
                        answer += f"\n...and {len(rows) - 3} more rows."
            else:
                answer = "No data found."


            # try:
            #     chart_config = llm_generate_chart_config(question, rows)
            #     print("📊 Chart config:", chart_config)

            # except Exception as chart_err:
            #     print("⚠️ Chart generation failed:", chart_err)
            #     chart_config = None

            chart_config = None
            # if rows and len(rows) > 1: 
            # if len(rows) == 1 and isinstance(rows[0], dict):
            if rows and isinstance(rows[0], dict):
            #  # Only generate chart if we have multiple data points
                try:
                    chart_config = llm_generate_chart_config(question, rows)
                    print("📊 Generated chart config:", json.dumps(chart_config, indent=2))
                    
                    # Validate chart config before sending
                    if chart_config and isinstance(chart_config, dict):
                        # Ensure required fields exist
                        if 'series' not in chart_config or not chart_config['series']:
                            print("⚠️ Invalid chart config - missing or empty series")
                            chart_config = None
                        else:
                            # Validate each series has data
                            valid_series = []
                            for series in chart_config['series']:
                                if 'data' in series and series['data']:
                                    valid_series.append(series)
                            
                            if valid_series:
                                chart_config['series'] = valid_series
                            else:
                                chart_config = None
                                
                except Exception as chart_err:
                    print("⚠️ Chart generation failed:", chart_err)
                    chart_config = None


            try:
                if not recommendation:
                    recommendation = llm_generate_recommendation(question, rows)
            except Exception as rec_err:
                print("⚠️ Recommendation generation failed:", rec_err)
                recommendation = "Could not generate recommendation at this time."

            # 🔧 FIX 9: Store more context in conversation memory
            conversation_memory_store.setdefault(session_id, []).append({
                "question": question,
                "sql": sql,
                "row_count": len(rows),
                "timestamp": now().isoformat()
            })

            if session_id in session_store:
                session_store[session_id]["user_id"] = user_id

            total_time = (now() - start_time).total_seconds()

            return JsonResponse({
                "answer": answer,
                "success": True,
                "query_used": sql,
                "rows": rows,
                "narrative": narrative, 
                "chart_config": chart_config,
                "summary": summary,
                "row_count": len(rows),
                "recommendation": recommendation,
                "response_time": f"{total_time:.2f}s",
                "session_id": session_id,
                "history": conversation_memory_store[session_id]
            })

        # 🔧 FIX 10: Handle OPTIONS request for CORS
        elif request.method == "OPTIONS":
            response = HttpResponse()
            response['Access-Control-Allow-Origin'] = '*'
            response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
            return response

        else:
            return JsonResponse({"error": "Method not allowed. Use GET for export or POST for queries."}, status=405)

    except Exception as e:
        print("💥 Unexpected error in ask_question:")
        traceback.print_exc()
        return JsonResponse({
            "answer": "Something went wrong.",
            "success": False,
            "error": str(e),
            "rows": [],
            "row_count": 0,
            "response_time": "0.00s",
            "session_id": request.GET.get("session_id") if request.method == "GET" else json.loads(request.body).get("session_id", "unknown") if request.method == "POST" else "unknown"
        }, status=500)


from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.timezone import now
from sqlalchemy import text
from .utils.stream import jsonl_line 



@csrf_exempt
def ask_question_streamworkingbftrain(request):
    if request.method == "OPTIONS":
        r = HttpResponse()
        r["Access-Control-Allow-Origin"] = "*"
        r["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        r["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
        return r

    if request.method != "POST":
        return JsonResponse({"error": "Use POST"}, status=405)

    try:
        payload = json.loads(request.body or "{}")
        session_id = payload.get("session_id")
        question = payload.get("question")
        user_id = payload.get("user_id", "stream_user")

        if not session_id or not question:
            return JsonResponse({"error": "Missing session_id or question"}, status=400)
        if session_id not in session_store:
            return JsonResponse({"error": f"Session {session_id} not found"}, status=404)

        engine = session_store[session_id]["engine"]
        hist = conversation_memory_store.get(session_id, [])
        t0 = now()

        def gen():
            # Phase
            yield jsonl_line({"event": "phase", "message": "Understanding your question…"})

            # SQL generation
            try:
                raw = run_sql_generation_graph(question, user_id=user_id, db_id=session_id, history=hist)
                sql_raw = raw[0] if isinstance(raw, tuple) else raw
                sql = extract_sql_block(sql_raw or "")
                sql = re.sub(r"\bLIMIT(\d+)", r"LIMIT \1", sql, flags=re.IGNORECASE)
            except Exception as e:
                yield jsonl_line({"event": "error", "message": f"SQL generation failed: {e}"})
                return

            if not sql or not sql.strip().lower().startswith(("select", "with")):
                yield jsonl_line({"event": "error", "message": "Invalid or empty SQL generated"})
                return
            yield jsonl_line({"event": "sql", "sql": sql})

            # Execute
            yield jsonl_line({"event": "phase", "message": "Querying the database…"})
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(sql))
                    rows = [dict(row._mapping) for row in result]
            except Exception as e:
                yield jsonl_line({"event": "error", "message": f"SQL execution failed: {e}"})
                return

            row_count = len(rows)
            yield jsonl_line({"event": "rows_preview", "rows": rows[:8], "row_count": row_count})

            # Analyze
            yield jsonl_line({"event": "phase", "message": "Analyzing results…"})
            try:
                summary = generate_summary_from_rows(question, sql, rows)
                if summary:
                    yield jsonl_line({"event": "summary", "text": summary})
            except Exception:
                summary = ""

            try:
                narr = llm_generate_narrative(question, rows)
                yield jsonl_line({"event": "narrative", "obj": narr})
            except Exception:
                narr = None

            try:
                rec = llm_generate_recommendation(question, rows)
                if rec:
                    yield jsonl_line({"event": "recommendation", "text": rec})
            except Exception:
                rec = None

            try:
                cfg = llm_generate_chart_config(question, rows)
                if cfg:
                    yield jsonl_line({"event": "chart", "config": cfg})
            except Exception:
                cfg = None

            # Short headline “answer”
            if row_count == 0:
                answer = "No data found."
            elif row_count > 50:
                answer = f"Found {row_count} results. Too many to display here — use the CSV download."
            else:
                first = [", ".join(str(v) for v in r.values()) for r in rows[:3]]
                answer = "\n".join(first)
                if row_count > 3:
                    answer += f"\n...and {row_count - 3} more rows."

            # Memory
            conversation_memory_store.setdefault(session_id, []).append({
                "question": question, "sql": sql, "row_count": row_count, "timestamp": now().isoformat()
            })
            session_store[session_id]["user_id"] = user_id

            # Final
            total = (now() - t0).total_seconds()
            yield jsonl_line({
                "event": "final",
                "payload": {
                    "answer": answer,
                    "summary": summary,
                    "rows": rows[:50],
                    "row_count": row_count,
                    "chart_config": cfg,
                    "recommendation": rec,
                    "narrative": narr,
                    "query_used": sql,
                    "response_time": f"{total:.2f}s",
                    "session_id": session_id,
                },
            })

        resp = StreamingHttpResponse(gen(), content_type="application/x-ndjson")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        resp["Access-Control-Allow-Origin"] = "*"
        return resp

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

        

@csrf_exempt
def ask_questioned247(request):
    try:
        start_time = now()

        # ✅ Handle GET request for file export
        if request.method == "GET" and request.GET.get("export") == "true":
            session_id = request.GET.get("session_id")
            question = request.GET.get("question")
            print("🔍 Current session_store keys:", list(session_store.keys()))
            print(f"🔎 session_id requested: {session_id}")

            if not session_id or not question:
                return JsonResponse({"error": "Missing session_id or question in GET request"}, status=400)

            if session_id not in session_store:
                return JsonResponse({"error": f"Session ID {session_id} not found. Please run the query first via chat."}, status=404)

            # ✅ ADD THIS LINE HERE
            user_id = session_store[session_id].get("user_id", "export_user")
            engine = session_store[session_id]["engine"]

            # Run SQL generation again based on question + session_id
            history = conversation_memory_store.get(session_id, [])
            raw_response = run_sql_generation_graph(question, user_id= user_id, db_id=session_id, history=history)
            sql = extract_sql_block(raw_response)

            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = [dict(row._mapping) for row in result]

        

            if not rows:
                return HttpResponse("No data to export", content_type="text/plain")

            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename=churn_output.csv'
            writer = csv.DictWriter(response, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
            return response

        # ✅ Handle POST request
        data = json.loads(request.body)
        session_id = data.get("session_id")
        question = data.get("question")
        user_id = data.get("user_id", "test_user_001")

        if not all([session_id, question]):
            return JsonResponse({"error": "Missing session_id or question"}, status=400)

        history = conversation_memory_store.get(session_id, [])
        print(f"🧠 Running SQL generation with session_id={session_id}, user_id={user_id}")

        VALID_COLUMNS = extract_columns_from_schema(FULL_SCHEMA)
        raw_response = run_sql_generation_graph(question, user_id=user_id, db_id=session_id, history=history)
        sql = extract_sql_block(raw_response)
        print("📝 Extracted SQL:", sql)

        invalid_cols = validate_sql_columns(sql, VALID_COLUMNS)
        if invalid_cols:
            return JsonResponse({
                "answer": f"Invalid columns in SQL: {', '.join(invalid_cols)}",
                "success": False,
                "query_used": sql,
                "rows": [],
                "row_count": 0,
                "session_id": session_id,
                "response_time": "0.00s"
            }, status=400)

        if not sql.strip().lower().startswith(("select", "with")):
            return JsonResponse({
                "answer": "Invalid or failed SQL generation",
                "success": False,
                "query_used": sql,
                "rows": [],
                "row_count": 0,
                "session_id": session_id,
                "response_time": "0.00s"
            }, status=500)

        if session_id not in session_store:
            raise Exception(f"Session ID {session_id} not found in session_store")

        engine = session_store[session_id]["engine"]
        # with engine.connect() as conn:
        #     result = conn.execute(text(sql))
        #     rows = [dict(row._mapping) for row in result]

        try:
            print(f"✅ Executing SQL on session: {session_id}")
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = [dict(row._mapping) for row in result]
            print(f"✅ SQL executed successfully. Row count: {len(rows)}")
        except Exception as e:
            print("❌ SQL Execution Error:", str(e))
            traceback.print_exc()
            return JsonResponse({
                "answer": "SQL execution failed.",
                "success": False,
                "query_used": sql,
                "error": str(e),
                "rows": [],
                "row_count": 0,
                "response_time": "0.00s",
                "session_id": session_id
            }, status=500)


        # Generate human answer
        if len(rows) == 1 and len(rows[0]) == 1:
            answer = f"The result is {list(rows[0].values())[0]}."
        elif rows:
            answer = "; ".join(", ".join(f"{k}: {v}" for k, v in row.items()) for row in rows[:5])
            if len(rows) > 5:
                answer += f" ...and {len(rows) - 5} more rows."
        elif rows:
            if len(rows) > 50:
                answer = f"Too many results to display ({len(rows)} rows). [Please download the full results as CSV]"
            else:
                answer = "; ".join(", ".join(f"{k}: {v}" for k, v in row.items()) for row in rows)
        else:
            answer = "No data found."

        # Save conversation
        conversation_memory_store.setdefault(session_id, []).append({
            "question": question,
            "sql": sql,
            "timestamp": now().isoformat()
        })

        total_time = (now() - start_time).total_seconds()

        return JsonResponse({
            "answer": answer,
            "success": True,
            "query_used": sql,
            "rows": rows,
            "row_count": len(rows),
            "response_time": f"{total_time:.2f}s",
            "session_id": session_id,
            "history": conversation_memory_store[session_id]
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({
            "answer": "Something went wrong.",
            "success": False,
            "error": str(e),
            "rows": [],
            "row_count": 0,
            "response_time": "0.00s",
            "session_id": request.GET.get("session_id", "unknown")
        }, status=500)

@csrf_exempt

def disconnect_database(request):
    """Disconnect database and clear session"""
    try:
        data = json.loads(request.body)
        session_id = data.get("session_id")
        
        if session_id in session_store:
            del session_store[session_id]
            del schema_context_store[session_id]
            del conversation_memory_store[session_id]
            
            return JsonResponse({"message": "Database disconnected successfully"})
        else:
            return JsonResponse({"error": "Invalid session"}, status=404)
            
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)



@csrf_exempt
def ask_question_auto(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    try:
        data = json.loads(request.body)
        question = data.get("question", "").strip().lower()

        pdf_keywords = ["summarize", "pdf", "insights", "churn causes", "explain", "reason"]
        db_keywords = [
            "idv", "policy", "branch", "zone", "segment", "customer", "premium", "vehicle", "claim", "renewal",
            "recommendation", "churn probability", "retention"
        ]

        if any(kw in question for kw in pdf_keywords):
            return ask_questionbot(request)
        elif any(kw in question for kw in db_keywords):
            return ask_question(request)
        else:
            return JsonResponse({
                "popup_required": True,
                "message": "Should I check the uploaded PDF or business database?"
            })

    except Exception as e:
        logger.error(f"Error in ask_question_auto: {e}")
        return JsonResponse({'error': str(e)}, status=500)



import requests
import json

question = "How would you build the tallest building ever?"

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
  "Authorization": f"Bearer <OPENROUTER_API_KEY>",
  "Content-Type": "application/json"
}

payload = {
  "model": "openai/gpt-4o",
  "messages": [{"role": "user", "content": question}],
  "stream": True
}

buffer = ""
with requests.post(url, headers=headers, json=payload, stream=True) as r:
  for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
    buffer += chunk
    while True:
      try:
        # Find the next complete SSE line
        line_end = buffer.find('\n')
        if line_end == -1:
          break

        line = buffer[:line_end].strip()
        buffer = buffer[line_end + 1:]

        if line.startswith('data: '):
          data = line[6:]
          if data == '[DONE]':
            break

          try:
            data_obj = json.loads(data)
            content = data_obj["choices"][0]["delta"].get("content")
            if content:
              print(content, end="", flush=True)
          except json.JSONDecodeError:
            pass
      except Exception:
        break


# churn given266

# === Extract text from PDF ===
def extract_text_from_pdf(file_path):
    print(f"Extracting text from PDF: {file_path}")
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    print("Text extraction complete.")
    return text


# === Ingest extracted text to FAISS vectorstore ===
def ingest_pdf_to_vectorstore(file_path, vectorstore_path):
    print(f"Ingesting PDF to vectorstore: {file_path} -> {vectorstore_path}")
    text = extract_text_from_pdf(file_path)
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = [Document(page_content=c) for c in splitter.split_text(text)]
    vectordb = FAISS.from_documents(chunks, embedding_model)
    vectordb.save_local(vectorstore_path)
    print("Vectorstore ingestion complete.")


# === Always load the admin's stored vectorstore ===
def get_best_vectorstore(query):
    print(f"Loading permanent admin vectorstore for query: {query}")

    try:
        vectordb = FAISS.load_local(
            ADMIN_VECTORSTORE_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print("Permanent vectorstore loaded.")
        return vectordb
    except Exception as e:
        print(f"Failed to load admin vectorstore: {e}")
        return None


# === Endpoint for Admin to Upload PDF ===
@csrf_exempt
def upload_pdfbot(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    pdf_file = request.FILES.get('file')
    if not pdf_file:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, pdf_file.name)
    with open(file_path, 'wb+') as destination:
        for chunk in pdf_file.chunks():
            destination.write(chunk)

    print(f"PDF uploaded: {pdf_file.name}")

    # Always store to the admin_base vectorstore
    ingest_pdf_to_vectorstore(file_path, ADMIN_VECTORSTORE_PATH)

    return JsonResponse({'message': 'PDF uploaded and permanently indexed for all users.'})



# churn given266
# views.py
import json, requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA  # NEW
from langchain.retrievers import ContextualCompressionRetriever  # NEW
from langchain.retrievers.document_compressors import CrossEncoderReranker  # NEW
import json, requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA  # NEW
from langchain.retrievers import ContextualCompressionRetriever  # NEW
from langchain.retrievers.document_compressors import CrossEncoderReranker  # NEW



# === Endpoint for any user to ask questions ===
@csrf_exempt
def ask_questionbot(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    # robust JSON parsing
    try:
        data = json.loads((request.body or b"{}").decode("utf-8"))
    except Exception:
        return JsonResponse({'answer': "Invalid JSON body."}, status=400)

    question = str(data.get("query") or "").strip()
    print(f"Received question: {question}")
    if not question:
        return JsonResponse({'answer': "Please type a question."})

    vectordb = get_best_vectorstore(question)
    if not vectordb:
        return JsonResponse({'answer': "Sorry, the knowledge base is not available. Please contact admin."})

    # ---------- NEW: probe KB first with a threshold ----------
    try:
        probe = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.30}  # soft threshold
        )
        try:
            probe_docs = probe.invoke(question)  # new LC API
        except Exception:
            probe_docs = probe.get_relevant_documents(question)  # backwards compat
    except Exception:
        probe_docs = []

    # ---------- NEW: if nothing relevant in KB, answer generally ----------
    if not probe_docs:
        try:
            general_llm = ChatOpenAI(
                model="google/gemma-3-27b-it:free",
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0,
                max_tokens=256
            )
            general_prompt = (
                "You are a helpful assistant. Answer the user's question clearly in 2–4 sentences. "
                "Provide a complete, standalone answer (not a fragment). "
                "Do NOT mention Prochurn/ProSync or documentation unless the user asked about it.\n\n"
                f"Question:\n{question}\n\nAnswer:"
            )
            general_answer = general_llm.invoke(general_prompt).content
            return JsonResponse({'answer': general_answer})
        except Exception:
            # if general generation fails, fall through to RAG as a safe default
            pass

    # ---------- YOUR ORIGINAL RAG CODE (UNCHANGED) ----------
    # retriever = vectordb.as_retriever(search_type="similarity", k=5)

    # llm = ChatOpenAI(
    #     # model="meta-llama/llama-4-maverick:free",  
    #     model="google/gemma-3-27b-it:free",  
    #     openai_api_key=OPENROUTER_API_KEY,
    #     openai_api_base="https://openrouter.ai/api/v1",  
    #     temperature=0,
    #     max_tokens=1024
    # )

    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=retriever,
    # )

    # print("Generating answer using LLM...")
    # result = qa_chain.invoke({"query": question})
    # answer = result if isinstance(result, str) else result.get("result", "")
    # print(f"Answer generated: {answer}")
    # return JsonResponse({'answer': answer})

    retriever = vectordb.as_retriever(search_type="similarity", k=5)

    # ===== NEW: Fix reranker wiring (no pydantic error) with safe fallbacks =====
    try:
        # HuggingFaceCrossEncoder instance is required (not a string)
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        cross_encoder = HuggingFaceCrossEncoder("BAAI/bge-reranker-large")
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
        retriever = ContextualCompressionRetriever(
            base_retriever=retriever,
            compressor=reranker
        )
    except Exception as e:
        print("Reranker unavailable, switching to embeddings filter:", e)
        try:
            # Lightweight fallback that removes off-topic text using your vectorstore's embedder
            from langchain_community.document_transformers import EmbeddingsFilter
            embedder = getattr(vectordb, "_embedding_function", None)
            if embedder is not None:
                ef = EmbeddingsFilter(embeddings=embedder, similarity_threshold=0.60)
                retriever = ContextualCompressionRetriever(
                    base_retriever=retriever,
                    compressor=ef
                )
            else:
                print("No embedder found on vectorstore; using base retriever.")
        except Exception as e2:
            print("EmbeddingsFilter unavailable; using base retriever:", e2)

    llm = ChatOpenAI(
        model="google/gemma-3-27b-it:free",  
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",  
        temperature=0,
        max_tokens=1024
    )

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    You are an expert AI assistant. Use the below context to answer the user's question.

    IMPORTANT:
    - Answer ONLY what was asked; ignore unrelated sections from the context.
    - If the question asks "what is/define/explain X", reply with a concise 1–3 sentence definition that starts with "X is ...".
    - Do NOT include headings or long lists unless explicitly requested.
    - If context is insufficient, say so briefly and ask for specifics.
    - Do NOT start with phrases like 'Based on the provided text' or 'According to the text'.

    Context:
    {context}

    Question:
    {question}

    Answer (max 80 words):
    """
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt}
    )

    print("Generating answer using LLM...")
    result = qa_chain.invoke({"query": question})
    answer = result if isinstance(result, str) else result.get("result", "")

    return JsonResponse({'answer': answer})




class ChartGenerator:
    def __init__(self):
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Chart type keywords mapping
        self.chart_keywords = {
            'bar': ['bar'],
            'histogram': ['histogram', 'distribution'],
            'column': ['column', 'frequency'],
            'line': ['line', 'trend', 'time series', 'over time', 'timeline'],
            'pie': ['pie', 'donut', 'proportion', 'percentage', 'distribution'],
            'scatter': ['scatter', 'correlation', 'relationship', 'vs', 'against'],
            'box': ['box', 'boxplot', 'quartile', 'outlier', 'distribution'],
            'violin': ['violin', 'density', 'distribution'],
            'heatmap': ['heatmap', 'correlation matrix', 'heat map'],
            'area': ['area', 'filled', 'stacked area'],
            'bubble': ['bubble', 'size', 'three dimensional'],
            'radar': ['radar', 'spider', 'polar'],
            'funnel': ['funnel', 'conversion', 'stages'],
            'waterfall': ['waterfall', 'cumulative', 'breakdown'],
            'treemap': ['treemap', 'hierarchy', 'nested'],
            'sunburst': ['sunburst', 'hierarchical', 'nested pie'],
            'gauge': ['gauge', 'speedometer', 'meter'],
            'candlestick': ['candlestick', 'ohlc', 'stock'],
            'sankey': ['sankey', 'flow', 'alluvial']
        }
        
        # Comparison keywords
        self.comparison_keywords = [
            'compare', 'comparison', 'vs', 'versus', 'against', 'difference',
            'between', 'contrast', 'relative', 'side by side'
        ]

    def detect_chart_type(question, df):
        """Detect the appropriate chart type based on the question"""
        question_lower = question.lower()
        
        # Check for specific chart type mentions
        if any(word in question_lower for word in ['pie', 'distribution', 'percentage', 'proportion']):
            return 'pie'
        elif any(word in question_lower for word in ['column', 'stacked']):
            return 'column'
        elif any(word in question_lower for word in ['bar', 'top', 'bottom', 'ranking', 'compare']):
            return 'bar'
        elif any(word in question_lower for word in ['line', 'trend', 'over time', 'timeline']):
            return 'line'
        elif any(word in question_lower for word in ['scatter', 'correlation', 'relationship']):
            return 'scatter'
        elif any(word in question_lower for word in ['histogram', 'frequency']):
            return 'histogram'
        else:
            # Default logic based on data types
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                return 'bar'  # Default to bar chart for multiple numeric columns
            else:
                return 'bar'

    def extract_columns_from_question(question, df):
        """Extract relevant columns from the question and dataframe"""
        question_lower = question.lower()
        columns = []
        
        # Look for column names in the question
        for col in df.columns:
            if col.lower() in question_lower:
                columns.append(col)
        
        # If no specific columns found, use heuristics
        if not columns:
            # For "top X" questions, look for quantity/sales columns
            if 'top' in question_lower or 'sales' in question_lower:
                quantity_cols = [col for col in df.columns if any(word in col.lower() for word in ['sales', 'amount', 'quantity', 'total', 'revenue'])]
                name_cols = [col for col in df.columns if any(word in col.lower() for word in ['name', 'product', 'category', 'item'])]
                columns = name_cols + quantity_cols
            else:
                # Default to first text column and first numeric column
                text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                columns = (text_cols[:1] + numeric_cols[:1])
        
        return columns[:2]  # Limit to 2 columns for simplicity

    def prepare_data_for_chart(df, columns, chart_type, question):
        """Prepare data for chart generation"""
        if len(columns) < 2:
            # If only one column, add a count column
            if len(columns) == 1:
                col = columns[0]
                prepared_df = df[col].value_counts().reset_index()
                prepared_df.columns = [col, 'count']
                return prepared_df
            else:
                return df.head(10)  # Default fallback
        
        # Handle "top N" requests
        if 'top' in question.lower():
            # Extract number from question
            import re
            numbers = re.findall(r'\d+', question)
            n = int(numbers[0]) if numbers else 5
            
            # Assuming first column is category, second is value
            category_col, value_col = columns[0], columns[1]
            if pd.api.types.is_numeric_dtype(df[value_col]):
                prepared_df = df.nlargest(n, value_col)[[category_col, value_col]]
            else:
                prepared_df = df[[category_col, value_col]].head(n)
        else:
            prepared_df = df[columns].head(20)  # Limit to 20 rows for performance
        
        return prepared_df
    
    def create_matplotlib_chart(self, df: pd.DataFrame, columns: dict, chart_type: str, question: str) -> str:
        """Create chart using matplotlib/seaborn"""
        plt.figure(figsize=(12, 8))
        
        try:
            if chart_type == 'bar':
                if columns['color']:
                    sns.barplot(data=df, x=columns['x'], y=columns['y'], hue=columns['color'])
                else:
                    sns.barplot(data=df, x=columns['x'], y=columns['y'])
                plt.xticks(rotation=45)
                
            elif chart_type == 'line':
                if columns['color']:
                    sns.lineplot(data=df, x=columns['x'], y=columns['y'], hue=columns['color'])
                else:
                    sns.lineplot(data=df, x=columns['x'], y=columns['y'])
                
            elif chart_type == 'scatter':
                if columns['size']:
                    plt.scatter(df[columns['x']], df[columns['y']], 
                              s=df[columns['size']], alpha=0.6, 
                              c=df[columns['color']].astype('category').cat.codes if columns['color'] else 'blue')
                else:
                    sns.scatterplot(data=df, x=columns['x'], y=columns['y'], hue=columns['color'])
                
            elif chart_type == 'box':
                if columns['x'] and columns['y']:
                    sns.boxplot(data=df, x=columns['x'], y=columns['y'])
                else:
                    numeric_col = columns['y'] or df.select_dtypes(include=[np.number]).columns[0]
                    sns.boxplot(y=df[numeric_col])
                plt.xticks(rotation=45)
                
            elif chart_type == 'violin':
                if columns['x'] and columns['y']:
                    sns.violinplot(data=df, x=columns['x'], y=columns['y'])
                else:
                    numeric_col = columns['y'] or df.select_dtypes(include=[np.number]).columns[0]
                    sns.violinplot(y=df[numeric_col])
                plt.xticks(rotation=45)
                
            elif chart_type == 'heatmap':
                # Correlation heatmap
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) >= 2:
                    corr_matrix = numeric_df.corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                
            elif chart_type == 'area':
                if columns['color']:
                    # Stacked area chart
                    pivot_df = df.pivot_table(values=columns['y'], index=columns['x'], columns=columns['color'], fill_value=0)
                    pivot_df.plot.area(stacked=True, alpha=0.7)
                else:
                    plt.fill_between(df[columns['x']], df[columns['y']], alpha=0.7)
                    
            elif chart_type == 'pie':
                if columns['x'] and columns['y']:
                    plt.pie(df[columns['y']], labels=df[columns['x']], autopct='%1.1f%%')
                else:
                    # Use value counts of categorical column
                    cat_col = columns['x'] or df.select_dtypes(include=['object']).columns[0]
                    value_counts = df[cat_col].value_counts()
                    plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            
            # Set title and labels
            plt.title(f"Chart: {question[:50]}{'...' if len(question) > 50 else ''}", fontsize=14, pad=20)
            if columns['x']:
                plt.xlabel(columns['x'].replace('_', ' ').title())
            if columns['y']:
                plt.ylabel(columns['y'].replace('_', ' ').title())
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error creating matplotlib chart: {e}")
            plt.close()
            return None

    def create_plotly_chart(df, columns, chart_type, question):
        """Create a Plotly chart based on the data and parameters"""
        try:
            if len(columns) < 2:
                return None
                
            x_col, y_col = columns[0], columns[1]
            
            # Create the appropriate chart
            if chart_type == 'bar':
                fig = go.Figure(data=[
                    go.Bar(
                        x=df[x_col],
                        y=df[y_col],
                        marker_color='rgb(55, 83, 109)'
                    )
                ])
                fig.update_layout(
                    title=f'{y_col} by {x_col}',
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    template='plotly_white'
                )
                
            elif chart_type == 'line':
                fig = go.Figure(data=[
                    go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        mode='lines+markers',
                        line=dict(color='rgb(55, 83, 109)')
                    )
                ])
                fig.update_layout(
                    title=f'{y_col} over {x_col}',
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    template='plotly_white'
                )
                
            elif chart_type == 'pie':
                fig = go.Figure(data=[
                    go.Pie(
                        labels=df[x_col],
                        values=df[y_col],
                        hole=0.3
                    )
                ])
                fig.update_layout(
                    title=f'Distribution of {y_col} by {x_col}',
                    template='plotly_white'
                )
                
            else:  # Default to bar
                fig = go.Figure(data=[
                    go.Bar(
                        x=df[x_col],
                        y=df[y_col],
                        marker_color='rgb(55, 83, 109)'
                    )
                ])
                fig.update_layout(
                    title=f'{y_col} by {x_col}',
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    template='plotly_white'
                )
            
            # Convert to JSON-serializable format
            chart_json = fig.to_dict()
            
            return chart_json
            
        except Exception as e:
            logger.error(f"Error creating Plotly chart: {e}")
            return None


    def generate_comparison_chart(self, df: pd.DataFrame, columns: dict, question: str) -> dict:
        """Generate comparison charts (side-by-side or overlay)"""
        try:
            # Create subplot with multiple charts
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Chart 1', 'Chart 2'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            if columns['color']:
                # Split data by color column for comparison
                unique_values = df[columns['color']].unique()[:2]  # Take first 2 for comparison
                
                for i, value in enumerate(unique_values):
                    subset = df[df[columns['color']] == value]
                    
                    fig.add_trace(
                        go.Bar(x=subset[columns['x']], y=subset[columns['y']], 
                              name=f"{columns['color']}: {value}"),
                        row=1, col=i+1
                    )
            
            fig.update_layout(
                title=f"Comparison Chart: {question[:50]}{'...' if len(question) > 50 else ''}",
                template="plotly_white",
                height=600
            )
            
            return fig.to_dict()
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return None

    def generate_chart(self, df: pd.DataFrame, question: str, chart_format: str = 'plotly') -> dict:
        """Main method to generate chart based on question and data"""
        try:
            logger.info(f"Generating chart for question: {question[:100]}...")
            
            # Detect chart type
            chart_type = self.detect_chart_type(question, df)
            logger.info(f"Detected chart type: {chart_type}")
            
            # Extract relevant columns
            columns = self.extract_columns_from_question(question, df)
            logger.info(f"Extracted columns: {columns}")
            
            # Prepare data
            df_prepared = self.prepare_data(df, columns, chart_type, question)
            logger.info(f"Prepared data shape: {df_prepared.shape}")
            
            # Check if it's a comparison request
            is_comparison = any(word in question.lower() for word in self.comparison_keywords)
            
            result = {
                'success': True,
                'chart_type': chart_type,
                'columns_used': columns,
                'data_points': len(df_prepared),
                'is_comparison': is_comparison
            }
            
            if is_comparison and chart_format == 'plotly':
                chart_data = self.generate_comparison_chart(df_prepared, columns, question)
                result['chart_data'] = chart_data
                result['format'] = 'plotly_comparison'
            elif chart_format == 'plotly':
                chart_data = self.create_plotly_chart(df_prepared, columns, chart_type, question)
                result['chart_data'] = chart_data
                result['format'] = 'plotly'
            else:
                chart_data = self.create_matplotlib_chart(df_prepared, columns, chart_type, question)
                result['chart_data'] = chart_data
                result['format'] = 'matplotlib'
            
            if not result['chart_data']:
                result['success'] = False
                result['error'] = 'Failed to generate chart'
            
            return result
            
        except Exception as e:
            logger.error(f"Error in generate_chart: {e}")
            return {
                'success': False,
                'error': str(e),
                'chart_type': 'unknown'
            }
        

def detect_chart_request(question):
    """Detect if the question is asking for a chart/visualization"""
    chart_keywords = [
        'chart', 'graph', 'plot', 'visualize', 'visualization', 'show me', 
        'display', 'bar chart', 'line chart', 'pie chart', 'histogram',
        'scatter plot','bottom', 'compare', 'trend', 'distribution'
    ]
    return any(keyword in question.lower() for keyword in chart_keywords)


def extract_math_instruction(question):
    """Extract math-related logic from the question"""
    q = question.lower()
    instructions = {}

    if 'top' in q:
        for word in q.split():
            if word.isdigit():
                instructions['top_n'] = int(word)
                break
    if 'bottom' in q:
        for word in q.split():
            if word.isdigit():
                instructions['bottom_n'] = int(word)
                break
    if 'average' in q or 'mean' in q:
        instructions['agg'] = 'mean'
    if 'count' in q or 'how many' in q:
        instructions['agg'] = 'count'
    if 'sum' in q or 'total' in q:
        instructions['agg'] = 'sum'
    if 'compare' in q or 'vs' in q or 'versus' in q:
        instructions['compare'] = True
    if 'difference' in q:
        instructions['difference'] = True

    return instructions

def get_numeric_columns(df):
    """Get numeric columns from dataframe"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df):
    """Get categorical columns from dataframe"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def prepare_chart_data(df, x_col, y_col, chart_type, instructions=None, limit=20):
    """Prepare data for different chart types with support for math and comparison"""
    try:
        chart_df = df[[x_col, y_col]].dropna()

        agg_func = instructions.get('agg', 'sum') if instructions else 'sum'

        if chart_type == 'pie':
            if chart_df[x_col].dtype == 'object':
                grouped = chart_df.groupby(x_col)[y_col].agg(agg_func).sort_values(ascending=False)

                if instructions:
                    if 'top_n' in instructions:
                        grouped = grouped.head(instructions['top_n'])
                    elif 'bottom_n' in instructions:
                        grouped = grouped.tail(instructions['bottom_n'])

                if len(grouped) > limit:
                    top_data = grouped.head(limit - 1)
                    others_sum = grouped.tail(len(grouped) - (limit - 1)).sum()
                    if others_sum > 0:
                        top_data['Others'] = others_sum
                    grouped = top_data

                return {
                    'categories': grouped.index.tolist(),
                    'data': [{'name': name, 'y': float(value)} for name, value in grouped.items()]
                }

        else:
            if chart_df[x_col].dtype == 'object':
                grouped = chart_df.groupby(x_col)[y_col].agg(['sum', 'mean', 'count']).sort_values(agg_func, ascending=False)

                if instructions:
                    if 'top_n' in instructions:
                        grouped = grouped.head(instructions['top_n'])
                    elif 'bottom_n' in instructions:
                        grouped = grouped.tail(instructions['bottom_n'])

                return {
                    'categories': grouped.index.tolist(),
                    'series': [{
                        'name': y_col,
                        'data': grouped[agg_func].tolist()
                    }]
                }
            else:
                if len(chart_df) > limit:
                    chart_df = chart_df.sample(n=limit).sort_values(x_col)

                return {
                    'categories': chart_df[x_col].tolist(),
                    'series': [{
                        'name': y_col,
                        'data': chart_df[y_col].tolist()
                    }]
                }

    except Exception as e:
        logger.error(f"Error preparing chart data: {e}")
        return None


def determine_chart_type(question, df, x_col, y_col):
    """Determine appropriate chart type based on question and data"""
    question_lower = question.lower()
   
    # Explicit chart type requests
    if 'pie' in question_lower:
        return 'pie'
    elif 'line' in question_lower:
        return 'line'
    elif 'bar' in question_lower:
        return 'bar'
    elif 'column' in question_lower:
        return 'column'
   
    # Auto-determine based on data types
    x_is_categorical = df[x_col].dtype == 'object'
    x_is_date = pd.api.types.is_datetime64_any_dtype(df[x_col])
   
    # Time series data -> line chart
    if x_is_date:
        return 'line'
   
    # Categorical data with few categories -> pie chart
    if x_is_categorical and df[x_col].nunique() <= 8:
        if 'distribution' in question_lower or 'breakdown' in question_lower:
            return 'pie'
   
    # Categorical data -> column chart
    if x_is_categorical:
        return 'column'
   
    # Numeric data -> line chart for trends, bar for comparisons
    if 'trend' in question_lower or 'over time' in question_lower:
        return 'line'
   
    return 'column'  # Default

def select_columns_for_chart(df, question):
    """Smart column selection based on question"""
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
   
    question_lower = question.lower()
   
    # Try to find columns mentioned in the question
    mentioned_cols = []
    for col in df.columns:
        if col.lower() in question_lower:
            mentioned_cols.append(col)
   
    if len(mentioned_cols) >= 2:
        # Use mentioned columns
        x_col = mentioned_cols[0]
        y_col = mentioned_cols[1]
        if x_col in numeric_cols and y_col in categorical_cols:
            x_col, y_col = y_col, x_col  # Swap if needed
    else:
        # Auto-select based on data types
        if categorical_cols and numeric_cols:
            x_col = categorical_cols[0]  # First categorical column
            y_col = numeric_cols[0]      # First numeric column
        elif len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
        else:
            return None, None
   
    return x_col, y_col

def generate_chart_config(chart_type, title, categories, series_data):
    """Generate Highcharts configuration"""
    base_config = {
        'chart': {'type': chart_type},
        'title': {'text': title},
        'credits': {'enabled': False},
        'exporting': {'enabled': True}
    }
   
    if chart_type == 'pie':
        base_config.update({
            'series': [{
                'name': 'Value',
                'data': series_data,
                'dataLabels': {
                    'enabled': True,
                    'format': '{point.name}: {point.percentage:.1f}%'
                }
            }]
        })
    else:
        base_config.update({
            'xAxis': {
                'categories': categories,
                'title': {'text': 'Categories'}
            },
            'yAxis': {
                'title': {'text': 'Values'}
            },
            'series': series_data
        })
   
    return base_config

def extract_top_bottom_limit(question: str) -> dict:
    """Extract top/bottom and the number from the question like 'top 5', 'bottom 3'"""
    question = question.lower()
    result = {'type': None, 'limit': None}
    match = re.search(r'\b(top|bottom)\s*(\d+)', question)
    if match:
        result['type'] = match.group(1)
        result['limit'] = int(match.group(2))
    return result


def generate_chart_for_question(df, question):
    """Main chart generation function with enhanced logic for math, top/bottom, comparison"""
    try:
        instructions = extract_math_instruction(question)

        # Detect column usage
        x_col, y_col = select_columns_for_chart(df, question)
        if not x_col or not y_col:
            return {
                'success': False,
                'error': 'Could not determine appropriate columns for chart'
            }

        # Compute difference if required
        if instructions.get('difference'):
            numeric_cols = get_numeric_columns(df)
            if len(numeric_cols) >= 2:
                df['Difference'] = df[numeric_cols[0]] - df[numeric_cols[1]]
                y_col = 'Difference'

        # Aggregate if needed
        if 'agg' in instructions and instructions['agg'] != 'count':
            df = df.groupby(x_col, as_index=False).agg({y_col: instructions['agg']})
        elif instructions.get('agg') == 'count':
            df = df.groupby(x_col, as_index=False).agg({y_col: 'count'})

        # Determine chart type
        chart_type = determine_chart_type(question, df, x_col, y_col)

        # Limit data based on top_n or bottom_n
        limit = instructions.get('top_n') or instructions.get('bottom_n') or 20
        chart_data = prepare_chart_data(df, x_col, y_col, chart_type, limit=limit)

        if not chart_data:
            return {
                'success': False,
                'error': 'Could not prepare chart data'
            }

        # Manual top/bottom sorting
        if 'series' in chart_data and 'data' in chart_data['series'][0]:
            data_points = chart_data['series'][0]['data']
            categories = chart_data['categories']
            combined = list(zip(categories, data_points))

            if 'top_n' in instructions:
                combined = sorted(combined, key=lambda x: x[1], reverse=True)[:instructions['top_n']]
            elif 'bottom_n' in instructions:
                combined = sorted(combined, key=lambda x: x[1])[:instructions['bottom_n']]

            if combined:
                categories, data_points = zip(*combined)
                chart_data['categories'] = list(categories)
                chart_data['series'][0]['data'] = list(data_points)

        # Set chart title
        title = f"{y_col} by {x_col}"

        # Generate Highcharts config
        if chart_type == 'pie':
            highcharts_config = generate_chart_config(chart_type, title, [], chart_data['data'])
        else:
            highcharts_config = generate_chart_config(chart_type, title, chart_data['categories'], chart_data['series'])

        return {
            'success': True,
            'chart_type': chart_type,
            'chart_config': highcharts_config,
            'columns_used': [x_col, y_col],
            'title': title
        }

    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

# def generate_chart_for_question(self, df, question, chart_format='highcharts'):
#     chart_type = self.detect_chart_type(question, df)
#     columns = self.extract_columns_from_question(question, df)

#     if len(columns) < 2:
#         return {"success": False, "error": "Not enough columns", "chart_type": chart_type}

#     df_prepared = self.prepare_data_for_chart(df, columns, chart_type, question)

#     if chart_format == 'plotly':
#         chart_data = self.convert_to_plotly_format(df_prepared, columns[0], columns[1], chart_type)
#     else:
#         chart_data = self.convert_to_highcharts_format(df_prepared, columns[0], columns[1], chart_type)

#     return {
#         "success": True,
#         "chart_type": chart_type,
#         "columns_used": columns,
#         "chart_data": chart_data
#     }




import re
import pandas as pd
import numpy as np

# === Basic Lemmatization fallback (no NLTK needed) ===
def simple_lemmatize(word):
    if word.endswith('ies'):
        return word[:-3] + 'y'
    elif word.endswith('es'):
        return word[:-2]
    elif word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word

def lemmatize_question_words(question):
    words = re.findall(r'\w+', question.lower())
    return [simple_lemmatize(w) for w in words]

def extract_top_n(question):
    match = re.search(r'top\s+(\d+)', question.lower())
    return int(match.group(1)) if match else 10

def extract_bottom_n(question):
    match = re.search(r'(bottom|lowest|least|below)\s+(\d+)', question.lower())
    return int(match.group(2)) if match else None

def extract_comparison_items(question):
    match = re.search(r'compare\s+(.+?)\s+(vs|and)\s+(.+)', question.lower())
    if match:
        return [match.group(1).strip(), match.group(3).strip()]
    return None

def match_column(columns, question):
    lemmatized_words = lemmatize_question_words(question)
    for word in lemmatized_words:
        for col in columns:
            col_clean = col.lower().replace("_", "").replace(" ", "")
            if word in col_clean or word == col_clean:
                return col
    return None

def find_best_y_col(numeric_cols):
    priority_keywords = ["sales", "amount", "revenue", "total", "value", "income", "price", "cost", "quantity"]
    for keyword in priority_keywords:
        for col in numeric_cols:
            if keyword in col.lower():
                return col
    return numeric_cols[0] if numeric_cols else None

def detect_chart_type11(question, df):
    q = question.lower()
    if 'scatter' in q:
        return 'scatter'
    elif 'histogram' in q:
        return 'histogram'
    elif 'area' in q:
        return 'area'
    elif 'bubble' in q:
        return 'bubble'
    elif 'gantt' in q:
        return 'gantt'
    elif 'treemap' in q:
        return 'treemap'
    elif 'box' in q or 'whisker' in q:
        return 'boxplot'
    elif 'pie' in q:
        return 'pie'
    elif 'bar' in q:
        return 'bar'
    elif 'column' in q:
        return 'column'
    elif 'line' in q:
        return 'line'
    elif 'chart' in q or 'graph' in q or 'plot' in q:
        if 'date' in df.columns or 'time' in df.columns or any('date' in c for c in df.columns):
            return 'line'
        elif len(df.select_dtypes(include=['object'])) >= 1:
            return 'bar'
        else:
            return 'histogram'
    else:
        if any('date' in c or 'time' in c for c in df.columns):
            return 'line'
        elif len(df.select_dtypes(include=['object'])) >= 1 and len(df.select_dtypes(include=['number'])) >= 1:
            return 'bar'
        elif len(df.select_dtypes(include=['number'])) == 1:
            return 'histogram'
    return 'bar'

def generate_chart_data(self,df, question):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

    chart_type = self.detect_chart_type(question, df)
    x_col = match_column(text_cols + datetime_cols, question) or (datetime_cols[0] if datetime_cols else (text_cols[0] if text_cols else None))
    y_col = match_column(numeric_cols, question) or find_best_y_col(numeric_cols)

    if not x_col or not y_col:
        return None

    if x_col in datetime_cols:
        df[x_col] = pd.to_datetime(df[x_col])
        df[x_col] = df[x_col].dt.date

    top_n = extract_top_n(question)
    bottom_n = extract_bottom_n(question)
    compare_items = extract_comparison_items(question)

    df_grouped = df.groupby(x_col)[y_col]

    if 'count' in question.lower() or y_col not in question.lower():
        agg = df_grouped.count()
    else:
        agg = df_grouped.sum()

    if compare_items:
        agg = agg[agg.index.astype(str).str.lower().isin([item.lower() for item in compare_items])]
    elif bottom_n:
        agg = agg.nsmallest(bottom_n)
    else:
        agg = agg.nlargest(top_n)

    categories = agg.index.tolist()
    values = agg.values.tolist()

    series_data = (
        [{"name": name, "y": val} for name, val in zip(categories, values)]
        if chart_type == 'pie' else values
    )

    title = f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}"

    return {
        "chart_type": chart_type,
        "x_categories": [] if chart_type == 'pie' else categories,
        "series_data": series_data,
        "chart_title": title
    }

def generate_text_answer(df, question):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    lemmatized_words = lemmatize_question_words(question)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()

    if "average" in lemmatized_words:
        col = match_column(numeric_cols, question)
        if col:
            return f"The average of {col} is {df[col].mean():.2f}"
    elif "sum" in lemmatized_words or "total" in lemmatized_words:
        col = match_column(numeric_cols, question)
        if col:
            return f"The total of {col} is {df[col].sum():.2f}"
    elif "count" in lemmatized_words:
        col = match_column(text_cols + numeric_cols, question)
        if col:
            return f"The count of {col} is {df[col].count()}"
    elif "maximum" in lemmatized_words or "highest" in lemmatized_words:
        col = match_column(numeric_cols, question)
        if col:
            return f"The maximum of {col} is {df[col].max()}"
    elif "minimum" in lemmatized_words or "lowest" in lemmatized_words:
        col = match_column(numeric_cols, question)
        if col:
            return f"The minimum of {col} is {df[col].min()}"

    return "Sorry, I could not find a relevant answer in the data."

def answer_from_file(df, question):
    chart = generate_chart_data(df, question)
    text = generate_text_answer(df, question)
    return {
        "answer": text,
        **(chart if chart else {})
    }
# backend/views.py


@csrf_exempt
def connect_databaseold(request):
    try:
        data = json.loads(request.body)
        db_uri = data.get("db_uri")
        session_id = str(uuid.uuid4())

        # Setup DB engine
        if db_uri == "USE_POSTGRESQL":
            engine = create_engine(
                f"postgresql://{data['postgres_user']}:{data['postgres_password']}@{data['postgres_host']}:{data.get('postgres_port', 5432)}/{data['postgres_db']}"
            )
        elif db_uri == "USE_MYSQL":
            engine = create_engine(
                f"mysql+mysqlconnector://{data['mysql_user']}:{data['mysql_password']}@{data['mysql_host']}/{data['mysql_db']}"
            )
        elif db_uri == "USE_LOCALDB":
            engine = create_engine("sqlite:///your/local/path/Student.db")
        else:
            return JsonResponse({"error": "Unsupported DB type"}, status=400)

        db = SQLDatabase(engine)
        llm = OllamaLLM(model="llama3", temperature=0.1)

        # === Accurate Schema Introspection ===
        try:
            inspector = inspect(engine)
            schema_info = ""
            for schema_name in inspector.get_schema_names():
                if schema_name.startswith("pg_") or schema_name in ("information_schema",):
                    continue
                for table in inspector.get_table_names(schema=schema_name):
                    cols = inspector.get_columns(table, schema=schema_name)
                    col_names = [col['name'] for col in cols]
                    schema_info += f"\nSchema: {schema_name}, Table: {table}, Columns: {col_names}"
        except Exception as e:
            schema_info = ""
            logger.warning(f"Could not extract schema info: {e}")

        schema_context_store[session_id] = schema_info

        def run_query_tool(query):
            cleaned_query = query.strip().split("\n")[0]
            cleaned_query = re.sub(r"[^\x20-\x7E]+$", "", cleaned_query)

            logger.info(f"[Query Preview] {cleaned_query}")
            query_preview_store[session_id] = cleaned_query

            try:
                with engine.connect() as conn:
                    result = conn.execute(text(cleaned_query))
                    rows = result.fetchall()
                    logger.info(f"[Query Result] {rows}")
                    if not rows:
                        return "No results found."
                    if len(rows[0]) == 1:
                        return ", ".join(str(row[0]) for row in rows)
                    else:
                        return "\n".join(", ".join(str(cell) for cell in row) for row in rows)
            except Exception as e:
                logger.error(f"[Query Error] {e}")
                return f"Query failed: {str(e)}"

        tools = [
            Tool(
                name="SQLExecutor",
                func=run_query_tool,
                description=f"""
                    Use this tool to run SQL queries against the connected database.
                    Only use schema/tables/columns defined below:
                    {schema_info}
                """
            )
        ]

        # Use prompt that includes all required fields
        prompt_template = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            template="""
You are a SQL expert AI assistant. You can use the following tools:
{tools}

Only refer to the schema structure below to answer:
{input}

Begin reasoning step-by-step.

{agent_scratchpad}
"""
        )

        tool_names = ", ".join([tool.name for tool in tools])

        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt_template.partial(tool_names=tool_names)
        )

        # agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        session_store[session_id] = agent_executor
        return JsonResponse({"message": "Connected", "session_id": session_id})

    except Exception as e:
        logger.error(f"DB connect error: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def ask_questionfordb(request):
    try:
        data = json.loads(request.body)
        session_id = data.get("session_id")
        question = data.get("question", "").strip()

        if not session_id or not question:
            return JsonResponse({"error": "Missing session_id or question"}, status=400)

        agent_executor = session_store.get(session_id)
        if not agent_executor:
            return JsonResponse({"error": "Session expired or invalid"}, status=404)

        schema_context = schema_context_store.get(session_id, "")
        prompt = f"""
You are a helpful AI assistant working on a connected SQL database.
Only answer based on the schema structure below:
{schema_context}

Answer the user's question accurately and with reasoning.
User question: {question}
        """

        try:
            result = agent_executor.invoke({"input": prompt})
            answer = result.get("output") if isinstance(result, dict) else str(result)
            preview = query_preview_store.get(session_id)

        except OutputParserException as oe:
            logger.error(f"Output parsing failed: {oe}")
            answer = "I had trouble understanding the response. Please try rephrasing."
            preview = None

        except Exception as e:
            logger.error(f"Agent error: {e}")
            answer = "I'm sorry, I couldn't understand that. Please try rephrasing."
            preview = None

        return JsonResponse({"answer": answer, "query_preview": preview, "success": True})

    except Exception as e:
        logger.error(f"Ask question error: {e}\n{traceback.format_exc()}")
        return JsonResponse({"error": str(e)}, status=500)

    
@csrf_exempt
def connect_database11(request):
    try:
        data = json.loads(request.body)
        db_uri = data.get("db_uri")
        session_id = str(uuid.uuid4())

        # Setup DB engine
        if db_uri == "USE_POSTGRESQL":
            engine = create_engine(
                f"postgresql://{data['postgres_user']}:{data['postgres_password']}@{data['postgres_host']}:{data.get('postgres_port', 5432)}/{data['postgres_db']}"
            )
        elif db_uri == "USE_MYSQL":
            engine = create_engine(
                f"mysql+mysqlconnector://{data['mysql_user']}:{data['mysql_password']}@{data['mysql_host']}/{data['mysql_db']}"
            )
        elif db_uri == "USE_LOCALDB":
            engine = create_engine("sqlite:///your/local/path/Student.db")
        else:
            return JsonResponse({"error": "Unsupported DB type"}, status=400)

        db = SQLDatabase(engine)
        llm = Ollama(model="llama3", temperature=0.1)

        # Extract schema for prompt context
        try:
            schema_info = db.get_table_info()
        except Exception as e:
            schema_info = ""
            logger.warning(f"Could not extract schema info: {e}")

        schema_context_store[session_id] = schema_info

        # SQL tool to run queries
        def run_query_tool(query):
            try:
                return db.run(query)
            except Exception as e:
                return f"Query failed: {str(e)}"

        tools = [
            Tool(
                name="SQLExecutor",
                func=run_query_tool,
                description=f"""
                    Use this tool to run SQL queries against the connected database.
                    Only use schema/tables/columns defined below:

                    {schema_info}
                """
            )
        ]

        agent = initialize_agent(
            tools,
            llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate"
        )

        session_store[session_id] = agent
        return JsonResponse({"message": "Connected", "session_id": session_id})

    except Exception as e:
        logger.error(f"DB connect error: {e}")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def ask_question11(request):
    try:
        data = json.loads(request.body)
        session_id = data.get("session_id")
        question = data.get("question", "").strip()

        if not session_id or not question:
            return JsonResponse({"error": "Missing session_id or question"}, status=400)

        agent = session_store.get(session_id)
        if not agent:
            return JsonResponse({"error": "Session expired or invalid"}, status=404)

        try:
            result = agent.invoke({"input": question})
            answer = result if isinstance(result, str) else str(result)

        except OutputParserException as oe:
            logger.error(f"Output parsing failed: {oe}")
            answer = "I had trouble understanding the response. Please try rephrasing."

        except Exception as e:
            logger.error(f"Agent error: {e}")
            answer = "I'm sorry, I couldn't understand that. Please try rephrasing."

        return JsonResponse({"answer": answer, "success": True})

    except Exception as e:
        logger.error(f"Ask question error: {e}\n{traceback.format_exc()}")
        return JsonResponse({"error": str(e)}, status=500)



def call_llm_with_retry(prompt: str, max_retries: int = 3, base_delay: float = 1.0) -> str:
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not defined")
        return "API key configuration error. Please check your settings."

    estimated_tokens = len(prompt.split()) * 1.5
    logger.info(f"Estimated prompt tokens: {estimated_tokens}")

    if estimated_tokens > 220000:
        logger.error(f"Prompt too long: {estimated_tokens} tokens (max: 220000)")
        return "The prompt is too long for the current model. Please try with a shorter question."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    model = "meta-llama/llama-4-maverick-17b-128e-instruct"
    # model = "meta-llama/llama-4-maverick-17b-128e-instruct"

    logger.info(f"Starting Groq LLM call with {max_retries} max retries")

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}")
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "max_tokens": 4000
            }
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=60)
            logger.info(f"Response status code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and result['choices']:
                    answer = result['choices'][0].get('message', {}).get('content', '').strip()
                    if answer:
                        return answer
            elif response.status_code in [429, 502, 503, 504]:
                time.sleep(base_delay * (2 ** attempt))
                continue

        except Exception as e:
            logger.warning(f"Retry {attempt+1} failed: {e}")
            time.sleep(base_delay)

    logger.error("All retry attempts failed")
    return "I'm currently experiencing issues reaching the AI service. Please try again later."

from datetime import datetime



@csrf_exempt
def ask_qwen(request):
    try:
        data = json.loads(request.body)
        session_id = data.get("session_id", "").strip()
        question = data.get("question", "").strip()

        if not question:
            return JsonResponse({'error': 'Question is required'}, status=400)
        if not session_id:
            return JsonResponse({'error': 'Session ID is required'}, status=400)

        logger.info(f"Processing question for session {session_id}: {question[:100]}")

        df = dataframe_map.get(session_id)
        has_data = df is not None

        if has_data:
            TOKEN_BUDGETS = {
                'question': len(question.split()) * 1.5,
                'data_overview': 800,
                'memory_context': 1000,
                'semantic_context': 4000,
                'prompt_template': 1000,
                'response_buffer': 8000,
                'safety_margin': 2000
            }

            semantic_context = ""
            memory_context = ""
            data_overview = ""

            prompt_parts = [
                "You are an expert data analyst AI that provides accurate answers based on uploaded datasets.",
                "",
                "IMPORTANT: Use ONLY the provided data. Never make assumptions or use external knowledge.",
                "",
                "### Dataset Overview:",
                data_overview,
                "",
                "### Previous Conversation Context:",
                memory_context or "No previous context.",
                "",
                "### Relevant Data Chunks:",
                semantic_context or "No relevant chunks found.",
                "",
                f"### User Question:",
                question,
                "",
                "### Instructions:",
                "1. Analyze the question carefully",
                "2. Use only the provided data chunks and dataset information",
                "3. If you need to perform calculations, show your work",
                "4. If the data doesn't contain enough information to answer, say so clearly",
                "5. Provide specific numbers, values, and examples from the actual data",
                "6. Be precise and factual - no guessing or assumptions",
                "7. Keep your answer concise but comprehensive",
                "",
                "Answer:"
            ]

            prompt = "\n".join(prompt_parts)
            prompt_tokens = len(prompt.split()) * 1.5
            logger.info(f"Final prompt: {prompt_tokens} tokens")

            answer = call_llm_with_retry(prompt)
            if not answer:
                answer = "I couldn't find an answer based on the uploaded data."

            return JsonResponse({
                "question": question,
                "answer": answer,
                "session_id": session_id,
                "chunks_used": 0,
                "prompt_tokens": prompt_tokens,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "has_chart": False
            })

        fallback_prompt = f"""You are an intelligent assistant. Answer the following question as accurately and helpfully as possible.

Question: {question}

Answer:"""

        answer = call_llm_with_retry(fallback_prompt)
        if not answer:
            answer = "Sorry, I couldn't generate a helpful answer. Please try rephrasing."

        return JsonResponse({
            "question": question,
            "answer": answer,
            "session_id": session_id,
            "chunks_used": 0,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "has_chart": False
        })

    except Exception as e:
        logger.error(f"Exception in ask_qwen: {traceback.format_exc()}")
        return JsonResponse({
            'error': f'Processing error: {str(e)}',
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'has_chart': False
        }, status=500)


@csrf_exempt
def ask_qwen19(request):
    """Enhanced question answering endpoint with optional chart generation and data handling"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid method'}, status=405)

    try:
        # Parse JSON
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)

        question = data.get("question", "").strip()
        session_id = data.get("session_id", "").strip()

        if not question:
            return JsonResponse({'error': 'Question is required'}, status=400)
        if not session_id:
            return JsonResponse({'error': 'Session ID is required'}, status=400)

        logger.info(f"Processing question for session {session_id}: {question[:100]}")

        df = None
        has_data = False

        # Try getting the dataframe
        if 'dataframe_map' in globals():
            df = dataframe_map.get(session_id)
            if df is not None:
                has_data = True
                logger.info(f"Found dataframe with shape: {df.shape}")
            else:
                logger.warning(f"No dataframe found for session {session_id}")
        else:
            logger.error("dataframe_map is not defined")

        # === CHART GENERATION ===
        if detect_chart_request(question) and has_data:
            try:
                logger.info("Chart-related request detected. Attempting chart generation...")
                chart_result = generate_chart_for_question(df, question)

                if chart_result.get('success'):
                    logger.info(f"Chart generated: {chart_result.get('chart_type')}")
                    return JsonResponse({
                        "question": question,
                        "answer": f"I've generated a {chart_result.get('chart_type')} chart showing {chart_result.get('title')}.",
                        "chart": chart_result.get('chart_config'),
                        "chart_type": chart_result.get('chart_type'),
                        "columns_used": chart_result.get('columns_used', []),
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat(),
                        "success": True,
                        "has_chart": True
                    })
                else:
                    logger.warning(f"Chart generation failed: {chart_result.get('error')}")
                    # Fallback to text-only response if chart fails

            except Exception as chart_error:
                logger.error(f"Chart generation error: {str(chart_error)}")
                logger.error(traceback.format_exc())

        # === TEXT ANSWER GENERATION ===

        # If we have data, generate context-aware response
        if has_data:
            semantic_context = ""
            memory_context = ""
            data_overview = ""

            try:
                if 'rag_system' in globals():
                    semantic_context = rag_system.retrieve_context(session_id, question)
                    logger.debug(f"Retrieved semantic context (length: {len(semantic_context)})")
            except Exception as e:
                logger.warning(f"Error retrieving semantic context: {e}")

            try:
                if 'get_memory_context' in globals():
                    memory_context = get_memory_context(session_id)
                    logger.debug(f"Retrieved memory context (length: {len(memory_context)})")
            except Exception as e:
                logger.warning(f"Error retrieving memory context: {e}")

            try:
                if 'get_data_overview' in globals():
                    data_overview = get_data_overview(df)
                    logger.debug(f"Retrieved data overview (length: {len(data_overview)})")
            except Exception as e:
                logger.warning(f"Error retrieving data overview: {e}")

            prompt = f"""You are an expert data analyst AI that provides accurate answers based on uploaded datasets.

IMPORTANT: Use ONLY the provided data. Never make assumptions or use external knowledge.

### Dataset Overview:
{data_overview}

### Previous Conversation Context:
{memory_context}

### Relevant Data Chunks:
{semantic_context}

### User Question:
{question}

### Instructions:
1. Analyze the question carefully
2. Use only the provided data chunks and dataset information
3. If you need to perform calculations, show your work
4. If the data doesn't contain enough information to answer, say so clearly
5. Provide specific numbers, values, and examples from the actual data
6. Be precise and factual - no guessing or assumptions

Answer:"""

            logger.info("Calling LLM for data-aware answer...")
            answer = call_llm_with_retry(prompt)
            if not answer:
                answer = "I couldn't find an answer based on the uploaded data."

            # Save to memory
            try:
                if 'add_to_memory' in globals():
                    add_to_memory(session_id, question, answer)
            except Exception as e:
                logger.warning(f"Error saving to memory: {e}")

            chunks_used = semantic_context.count("[Chunk") if semantic_context else 0

            return JsonResponse({
                "question": question,
                "answer": answer,
                "session_id": session_id,
                "chunks_used": chunks_used,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "has_chart": False
            })

        # === GENERAL MODEL FALLBACK === (no data uploaded)
        else:
            logger.info("No dataset found, using general-purpose LLM mode.")
            fallback_prompt = f"""You are an intelligent assistant. Answer the following question as accurately and helpfully as possible.

Question: {question}

Answer:"""

            answer = call_llm_with_retry(fallback_prompt)
            if not answer:
                answer = "Sorry, I couldn't generate a helpful answer. Please try rephrasing."

            return JsonResponse({
                "question": question,
                "answer": answer,
                "session_id": session_id,
                "chunks_used": 0,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "has_chart": False
            })

    except Exception as e:
        logger.error(f"Exception in ask_qwen: {traceback.format_exc()}")
        return JsonResponse({
            'error': f'Processing error: {str(e)}',
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'has_chart': False
        }, status=500)

# last running with chart but not general question
@csrf_exempt
def ask_qwenlast(request):
    """Enhanced question answering endpoint with chart generation"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid method'}, status=405)

    try:
        # Parse request data
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)

        question = data.get("question", "").strip()
        session_id = data.get("session_id", "").strip()

        # Validate inputs
        if not question:
            return JsonResponse({'error': 'Question is required'}, status=400)
       
        if not session_id:
            return JsonResponse({'error': 'Session ID is required'}, status=400)

        logger.info(f"Processing question for session {session_id}: {question[:100]}...")

        # Get dataframe
        if 'dataframe_map' not in globals():
            logger.error("dataframe_map is not defined")
            return JsonResponse({'error': 'Server configuration error'}, status=500)
           
        df = dataframe_map.get(session_id)
        if df is None:
            logger.warning(f"Session {session_id} not found in dataframe_map")
            return JsonResponse({'error': 'Session not found or file not processed'}, status=404)

        logger.info(f"Found dataframe with shape: {df.shape}")

        # Check if user wants a chart
        if detect_chart_request(question):
            logger.info("Chart request detected, generating chart...")
            try:
                chart_result = generate_chart_for_question(df, question)
               
                if chart_result.get('success'):
                    logger.info(f"Chart generated successfully: {chart_result.get('chart_type')}")
                   
                    return JsonResponse({
                        "question": question,
                        "answer": f"I've generated a {chart_result.get('chart_type')} chart showing {chart_result.get('title')}.",
                        "chart": chart_result.get('chart_config'),
                        "chart_type": chart_result.get('chart_type'),
                        "columns_used": chart_result.get('columns_used', []),
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat(),
                        "success": True,
                        "has_chart": True
                    })
                else:
                    logger.warning(f"Chart generation failed: {chart_result.get('error')}")
                    # Continue to text-only response
                   
            except Exception as chart_error:
                logger.error(f"Chart generation error: {str(chart_error)}")
                logger.error(traceback.format_exc())

        # Get enhanced context (your existing code)
        semantic_context = ""
        memory_context = ""
        data_overview = ""
       
        try:
            if 'rag_system' in globals():
                semantic_context = rag_system.retrieve_context(session_id, question)
                logger.debug(f"Retrieved semantic context (length: {len(semantic_context)})")
        except Exception as e:
            logger.warning(f"Error retrieving semantic context: {e}")

        try:
            if 'get_memory_context' in globals():
                memory_context = get_memory_context(session_id)
                logger.debug(f"Retrieved memory context (length: {len(memory_context)})")
        except Exception as e:
            logger.warning(f"Error retrieving memory context: {e}")

        try:
            if 'get_data_overview' in globals():
                data_overview = get_data_overview(df)
                logger.debug(f"Retrieved data overview (length: {len(data_overview)})")
        except Exception as e:
            logger.warning(f"Error retrieving data overview: {e}")

        # Build enhanced prompt
        prompt = f"""You are an expert data analyst AI that provides accurate answers based on uploaded datasets.

IMPORTANT: Use ONLY the provided data. Never make assumptions or use external knowledge.

### Dataset Overview:
{data_overview}

### Previous Conversation Context:
{memory_context}

### Relevant Data Chunks:
{semantic_context}

### User Question:
{question}

### Instructions:
1. Analyze the question carefully
2. Use only the provided data chunks and dataset information
3. If you need to perform calculations, show your work
4. If the data doesn't contain enough information to answer, say so clearly
5. Provide specific numbers, values, and examples from the actual data
6. Be precise and factual - no guessing or assumptions

Answer:"""

        logger.info("Calling LLM with retry mechanism")
       
        # Call LLM (your existing call_llm_with_retry function)
        answer = call_llm_with_retry(prompt)

        if not answer or answer.startswith("I'm currently experiencing issues"):
            logger.warning("LLM call failed or returned error message")
            if not answer:
                answer = "I couldn't generate an answer. Please try rephrasing your question."

        logger.info(f"Generated answer (length: {len(answer)})")

        # Save to memory
        try:
            if 'add_to_memory' in globals():
                add_to_memory(session_id, question, answer)
                logger.debug("Successfully saved to memory")
        except Exception as mem_err:
            logger.warning(f"Memory save error: {mem_err}")

        # Calculate chunks used
        chunks_used = 0
        if semantic_context:
            chunks_used = len(semantic_context.split("[Chunk")) - 1

        # Return text response
        response_data = {
            "question": question,
            "answer": answer,
            "session_id": session_id,
            "chunks_used": chunks_used,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "has_chart": False
        }
       
        logger.info(f"Returning successful response for session {session_id}")
        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Question answering error: {traceback.format_exc()}")
        return JsonResponse({
            'error': f'Processing error: {str(e)}',
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'has_chart': False
        }, status=500)

@csrf_exempt
def ask_qwenrr(request):
    """Enhanced question answering endpoint with chart generation and better error handling"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid method'}, status=405)

    try:
        # Parse request data
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)

        question = data.get("question", "").strip()
        session_id = data.get("session_id", "").strip()

        # Validate inputs
        if not question:
            return JsonResponse({'error': 'Question is required'}, status=400)
        
        if not session_id:
            return JsonResponse({'error': 'Session ID is required'}, status=400)

        logger.info(f"Processing question for session {session_id}: {question[:100]}...")

        # Get dataframe - make sure dataframe_map is accessible
        if 'dataframe_map' not in globals():
            logger.error("dataframe_map is not defined")
            return JsonResponse({'error': 'Server configuration error'}, status=500)
            
        df = dataframe_map.get(session_id)
        if df is None:
            logger.warning(f"Session {session_id} not found in dataframe_map")
            return JsonResponse({'error': 'Session not found or file not processed'}, status=404)

        logger.info(f"Found dataframe with shape: {df.shape}")

      
        print("[DEBUG] Raw question:", question)
        
        if detect_chart_request(question):
            logger.info("Chart request detected, generating chart...")
            try:
                chart_generator = ChartGenerator()
                chart_result = chart_generator.generate_chart_for_question(df, question,chart_format='highcharts')
                logger.info("Chart generation complete")

                logger.debug(f"Raw chart_result: {chart_result}")

                if chart_result and chart_result.get('success'):
                    logger.info(f"Chart generated successfully: {chart_result.get('chart_type', 'unknown')}")
                    
                    # Format chart data properly for frontend
                    chart_data = chart_result.get('chart_data', {})
                    
                    # Create proper response with chart
                    response_data = {
                        "question": question,
                        "answer": f"I've generated a {chart_result.get('chart_type', 'chart')} visualization for your question.",
                        "chart": {
                            "type": chart_result.get('chart_type', 'bar'),
                            "data": chart_data.get('data', []),
                            "layout": chart_data.get('layout', {}),
                            "config": {"displayModeBar": True, "responsive": True}
                        },
                        "chart_type": chart_result.get('chart_type'),
                        "columns_used": chart_result.get('columns_used', []),
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat(),
                        "success": True,
                        "has_chart": True
                    }
                    
                    logger.info("Returning chart response")
                    return JsonResponse(response_data)
                else:
                    logger.warning(f"Chart generation failed: {chart_result.get('error', 'Unknown error') if chart_result else 'No result returned'}")
                    # Continue to text-only response
                    
            except Exception as chart_error:
                logger.error(f"Chart generation error: {str(chart_error)}")
                logger.error(traceback.format_exc())

        # Get enhanced context - with error handling for each component
        semantic_context = ""
        memory_context = ""
        data_overview = ""
        
        try:
            if 'rag_system' in globals():
                semantic_context = rag_system.retrieve_context(session_id, question)
                logger.debug(f"Retrieved semantic context (length: {len(semantic_context)})")
            else:
                logger.warning("rag_system not available")
        except Exception as e:
            logger.warning(f"Error retrieving semantic context: {e}")

        try:
            if 'get_memory_context' in globals():
                memory_context = get_memory_context(session_id)
                logger.debug(f"Retrieved memory context (length: {len(memory_context)})")
            else:
                logger.warning("get_memory_context function not available")
        except Exception as e:
            logger.warning(f"Error retrieving memory context: {e}")

        try:
            if 'get_data_overview' in globals():
                data_overview = get_data_overview(df)
                logger.debug(f"Retrieved data overview (length: {len(data_overview)})")
            else:
                logger.warning("get_data_overview function not available")
        except Exception as e:
            logger.warning(f"Error retrieving data overview: {e}")

        # Build enhanced prompt
        prompt = f"""You are an expert data analyst AI that provides accurate answers based on uploaded datasets.

IMPORTANT: Use ONLY the provided data. Never make assumptions or use external knowledge.

### Dataset Overview:
{data_overview}

### Previous Conversation Context:
{memory_context}

### Relevant Data Chunks:
{semantic_context}

### User Question:
{question}

### Instructions:
1. Analyze the question carefully
2. Use only the provided data chunks and dataset information
3. If you need to perform calculations, show your work
4. If the data doesn't contain enough information to answer, say so clearly
5. Provide specific numbers, values, and examples from the actual data
6. Be precise and factual - no guessing or assumptions

Answer:"""

        logger.info("Calling LLM with retry mechanism")
        
        # Call LLM with retry mechanism
        answer = call_llm_with_retry(prompt)

        if not answer or answer.startswith("I'm currently experiencing issues"):
            logger.warning("LLM call failed or returned error message")
            if not answer:
                answer = "I couldn't generate an answer. Please try rephrasing your question."

        logger.info(f"Generated answer (length: {len(answer)})")

        # Save to memory with error handling
        try:
            if 'add_to_memory' in globals():
                add_to_memory(session_id, question, answer)
                logger.debug("Successfully saved to memory")
            else:
                logger.warning("add_to_memory function not available")
        except Exception as mem_err:
            logger.warning(f"Memory save error: {mem_err}")

        # Calculate chunks used
        chunks_used = 0
        if semantic_context:
            chunks_used = len(semantic_context.split("[Chunk")) - 1

        # Return response with additional metadata
        response_data = {
            "question": question,
            "answer": answer,
            "session_id": session_id,
            "chunks_used": chunks_used,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "has_chart": False
        }
        
        logger.info(f"Returning successful response for session {session_id}")
        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Question answering error: {traceback.format_exc()}")
        return JsonResponse({
            'error': f'Processing error: {str(e)}',
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'has_chart': False
        }, status=500)


#accurate running without chart  
@csrf_exempt
def ask_qwenee(request):
    """Enhanced question answering endpoint with better error handling"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid method'}, status=405)

    try:
        # Parse request data
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)

        question = data.get("question", "").strip()
        session_id = data.get("session_id", "").strip()

        # Validate inputs
        if not question:
            return JsonResponse({'error': 'Question is required'}, status=400)
        
        if not session_id:
            return JsonResponse({'error': 'Session ID is required'}, status=400)

        logger.info(f"Processing question for session {session_id}: {question[:100]}...")

        # Get dataframe - make sure dataframe_map is accessible
        if 'dataframe_map' not in globals():
            logger.error("dataframe_map is not defined")
            return JsonResponse({'error': 'Server configuration error'}, status=500)
            
        df = dataframe_map.get(session_id)
        if df is None:
            logger.warning(f"Session {session_id} not found in dataframe_map")
            return JsonResponse({'error': 'Session not found or file not processed'}, status=404)

        logger.info(f"Found dataframe with shape: {df.shape}")

        # Get enhanced context - with error handling for each component
        semantic_context = ""
        memory_context = ""
        data_overview = ""
        
        try:
            if 'rag_system' in globals():
                semantic_context = rag_system.retrieve_context(session_id, question)
                logger.debug(f"Retrieved semantic context (length: {len(semantic_context)})")
            else:
                logger.warning("rag_system not available")
        except Exception as e:
            logger.warning(f"Error retrieving semantic context: {e}")

        try:
            if 'get_memory_context' in globals():
                memory_context = get_memory_context(session_id)
                logger.debug(f"Retrieved memory context (length: {len(memory_context)})")
            else:
                logger.warning("get_memory_context function not available")
        except Exception as e:
            logger.warning(f"Error retrieving memory context: {e}")

        try:
            if 'get_data_overview' in globals():
                data_overview = get_data_overview(df)
                logger.debug(f"Retrieved data overview (length: {len(data_overview)})")
            else:
                logger.warning("get_data_overview function not available")
        except Exception as e:
            logger.warning(f"Error retrieving data overview: {e}")

        # Build enhanced prompt
        prompt = f"""You are an expert data analyst AI that provides accurate answers based on uploaded datasets.

IMPORTANT: Use ONLY the provided data. Never make assumptions or use external knowledge.

### Dataset Overview:
{data_overview}

### Previous Conversation Context:
{memory_context}

### Relevant Data Chunks:
{semantic_context}

### User Question:
{question}

### Instructions:
1. Analyze the question carefully
2. Use only the provided data chunks and dataset information
3. If you need to perform calculations, show your work
4. If the data doesn't contain enough information to answer, say so clearly
5. Provide specific numbers, values, and examples from the actual data
6. Be precise and factual - no guessing or assumptions

Answer:"""

        logger.info("Calling LLM with retry mechanism")
        
        # Call LLM with retry mechanism
        answer = call_llm_with_retry(prompt)

        if not answer or answer.startswith("I'm currently experiencing issues"):
            logger.warning("LLM call failed or returned error message")
            if not answer:
                answer = "I couldn't generate an answer. Please try rephrasing your question."

        logger.info(f"Generated answer (length: {len(answer)})")

        # Save to memory with error handling
        try:
            if 'add_to_memory' in globals():
                add_to_memory(session_id, question, answer)
                logger.debug("Successfully saved to memory")
            else:
                logger.warning("add_to_memory function not available")
        except Exception as mem_err:
            logger.warning(f"Memory save error: {mem_err}")

        # Calculate chunks used
        chunks_used = 0
        if semantic_context:
            chunks_used = len(semantic_context.split("[Chunk")) - 1

        # Return response with additional metadata
        response_data = {
            "question": question,
            "answer": answer,
            "session_id": session_id,
            "chunks_used": chunks_used,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        
        logger.info(f"Returning successful response for session {session_id}")
        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Question answering error: {traceback.format_exc()}")
        return JsonResponse({
            'error': f'Processing error: {str(e)}',
            'success': False,
            'timestamp': datetime.now().isoformat()
        }, status=500)

@csrf_exempt
def get_session_info(request):
    """Get information about a session"""
    if request.method != 'GET':
        return JsonResponse({'error': 'Invalid method'}, status=405)
    
    session_id = request.GET.get('session_id')
    if not session_id:
        return JsonResponse({'error': 'Session ID required'}, status=400)
    
    try:
        df = dataframe_map.get(session_id)
        if df is None:
            return JsonResponse({'error': 'Session not found'}, status=404)
        
        metadata = metadata_cache.get(session_id, {})
        memory = conversation_memory.get(session_id, [])
        
        return JsonResponse({
            'session_id': session_id,
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
            },
            'processing_info': {
                'chunks_created': len(metadata.get('chunks', [])),
                'created_at': metadata.get('created_at'),
                'data_characteristics': metadata.get('data_characteristics', {})
            },
            'conversation_history': len(memory),
            'last_questions': [q for q, a in memory[-3:]]  # Last 3 questions
        })
    
    except Exception as e:
        logger.error(f"Session info error: {e}")
        return JsonResponse({'error': str(e)}, status=500)
    

# import requests

@csrf_exempt

def check_intent(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get("question", "")

        prompt = f"""
Classify the user's intent strictly as YES or NO.

If the question is a general greeting, chit-chat, or general knowledge (e.g. "hi", "hello", "how are you", "who is the PM of India"), respond with NO.

If it is about querying the connected database schema or fetching data from tables, respond with YES.
If the question is trying to analyze, query, or summarize tabular or structured data (e.g. Excel, PDF tables, database), respond with YES.

Question: "{question}"
Only respond YES or NO.
"""

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",  # ✅ Groq Cloud endpoint
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",  # ✅ Use your Groq API key
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/llama-4-maverick-17b-128e-instruct",  # ✅ Groq model
                    "messages": [
                        {"role": "system", "content": "You are an intent classifier."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 10
                },
                timeout=30
            )
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            return JsonResponse({"answer": answer})

        except Exception as e:
            return JsonResponse({"answer": "No", "error": str(e)}, status=500)
