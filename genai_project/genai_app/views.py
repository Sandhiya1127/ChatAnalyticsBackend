from __future__ import annotations

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
from django.core.cache import cache


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
from .schema_utils import validate_sql_against_catalog,auto_fix_schema_mismatches,build_dynamic_fallback_map,build_type_map_from_catalog,check_question_vs_schema,embed_schema_catalog
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

# def _embed_schema_for_user(user_id: str, db_id: str, schema_text: str) -> None:
#     """Call your real embed function if present; otherwise no-op."""
#     try:
#         if _real_embed_schema:
#             _real_embed_schema(user_id=user_id, db_id=db_id, schema_text=schema_text)
#         else:
#             logger.info("Skipping schema embedding (helper not wired).")
#     except Exception as e:
#         logger.warning("Schema embedding failed: %s", e)



# from sentence_transformers import SentenceTransformer
# import chromadb

# # Initialize once
# chroma_client = chromadb.PersistentClient(path="./chroma_store")
# collection = chroma_client.get_or_create_collection("schemas")
# embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast model

# def _embed_schema_for_user(user_id: str, db_id: str, schema_text: str):
#     if not schema_text.strip():
#         logger.info("No schema text to embed, skipping.")
#         return
    
#     # Chunk schema (avoid feeding huge text blocks)
#     chunks = [schema_text[i:i+800] for i in range(0, len(schema_text), 800)]
#     embeddings = embedder.encode(chunks).tolist()

#     # Store in Chroma (or your vector DB)
#     for idx, chunk in enumerate(chunks):
#         doc_id = f"{db_id}_{idx}"
#         collection.upsert(
#             ids=[doc_id],
#             embeddings=[embeddings[idx]],
#             metadatas=[{"user_id": user_id, "db_id": db_id}],
#             documents=[chunk],
#         )
#     logger.info(f"✅ Embedded schema for db_id={db_id}, chunks={len(chunks)}")

# def _extract_schema_from_sqlalchemy(engine) -> str:
#     """Lightweight schema introspection—works on most Postgres setups."""
#     try:
#         insp = inspect(engine)
#         schema_names = insp.get_schema_names()
#     except Exception as e:
#         logger.warning("Could not list schemas: %s", e)
#         schema_names = ["public"]

#     lines = []
#     for schema in schema_names:
#         try:
#             tables = insp.get_table_names(schema=schema)
#         except Exception as e:
#             logger.warning("Could not list tables for schema %s: %s", schema, e)
#             continue

#         for t in tables:
#             try:
#                 cols = insp.get_columns(t, schema=schema)
#                 col_list = ", ".join(f'"{c.get("name")}" {c.get("type")}' for c in cols)
#                 lines.append(f'{schema}."{t}" ({col_list})')
#             except Exception as e:
#                 logger.warning("Could not inspect %s.%s: %s", schema, t, e)

#     return "\n".join(lines) if lines else ""



from sentence_transformers import SentenceTransformer
import chromadb
from sqlalchemy import inspect
from .schema_utils import find_relevant_table

# Initialize once
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection("schemas")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast model


def _embed_schema_for_user(user_id: str, db_id: str, schema_text: str):
    if not schema_text.strip():
        print("⚠️ [Embed] No schema text to embed, skipping.")
        return
    
    print(f"🔎 [Embed] Preparing to embed schema for user={user_id}, db_id={db_id}")
    # Chunk schema (avoid feeding huge text blocks)
    chunks = [schema_text[i:i+800] for i in range(0, len(schema_text), 800)]
    print(f"✂️ [Embed] Split schema into {len(chunks)} chunks (800 chars each).")

    embeddings = embedder.encode(chunks).tolist()
    print(f"🧩 [Embed] Generated {len(embeddings)} embeddings.")

    # Store in Chroma (or your vector DB)
    for idx, chunk in enumerate(chunks):
        doc_id = f"{db_id}_{idx}"
        print(f"➕ [Embed] Upserting doc_id={doc_id} with length={len(chunk)} chars")
        collection.upsert(
            ids=[doc_id],
            embeddings=[embeddings[idx]],
            metadatas=[{"user_id": user_id, "db_id": db_id}],
            documents=[chunk],
        )
    print(f"✅ [Embed] Finished embedding schema for db_id={db_id}, total chunks={len(chunks)}")


def _extract_schema_from_sqlalchemy(engine) -> str:
    """Lightweight schema introspection—works on most Postgres setups."""
    print("🔎 [Schema Extract] Starting SQLAlchemy introspection...")
    try:
        insp = inspect(engine)
        schema_names = insp.get_schema_names()
        print(f"📂 [Schema Extract] Found schemas: {schema_names}")
    except Exception as e:
        print(f"⚠️ [Schema Extract] Could not list schemas: {e}")
        schema_names = ["public"]

    lines = []
    for schema in schema_names:
        print(f"➡️ [Schema Extract] Inspecting schema: {schema}")
        try:
            tables = insp.get_table_names(schema=schema)
            print(f"   🗄 Tables in {schema}: {tables}")
        except Exception as e:
            print(f"⚠️ [Schema Extract] Could not list tables for schema {schema}: {e}")
            continue

        for t in tables:
            try:
                cols = insp.get_columns(t, schema=schema)
                col_list = ", ".join(f'"{c.get("name")}" {c.get("type")}' for c in cols)
                line = f'{schema}."{t}" ({col_list})'
                lines.append(line)
                print(f"     📑 Table: {schema}.{t} → {col_list}")
            except Exception as e:
                print(f"⚠️ [Schema Extract] Could not inspect {schema}.{t}: {e}")

    result = "\n".join(lines) if lines else ""
    print(f"✅ [Schema Extract] Finished. Total tables extracted: {len(lines)}")
    return result


# ----------------------------------------
# schema_utils.py (new helpers)
# ----------------------------------------
# from sqlalchemy import inspect

# from sqlalchemy import inspect

# def build_schema_catalog(engine):
#     """
#     Introspect all schemas and tables, return { "schema.table": [col1, col2, ...] }
#     """
#     insp = inspect(engine)
#     catalog = {}
#     print("🔎 Starting schema introspection...")

#     for schema in insp.get_schema_names():
#         if schema.startswith("pg_") or schema in {"information_schema"}:
#             print(f"  ⏭ Skipping system schema: {schema}")
#             continue

#         print(f"📂 Schema: {schema}")
#         for table in insp.get_table_names(schema=schema):
#             cols = [c["name"] for c in insp.get_columns(table, schema=schema)]
#             fq_name = f'"{schema}"."{table}"'
#             catalog[fq_name] = cols
#             print(f"    🗄 Table: {fq_name} → Columns: {cols}")

#     print("✅ Final schema_catalog keys:", list(catalog.keys()))
#     return catalog

# def embed_schema_catalog(user_id, db_id, schema_catalog, embedder, collection):
#     """
#     Store schema metadata into Chroma so LLM can retrieve relevant tables.
#     """
#     print(f"🔎 Embedding schema catalog for user={user_id}, db_id={db_id}")
#     docs, metas, ids = [], [], []
#     for i, (fq_table, cols) in enumerate(schema_catalog.items()):
#         doc = f"Table {fq_table} has columns: {', '.join(cols)}"
#         docs.append(doc)
#         metas.append({
#             "table": fq_table,      # ✅ ensure table is included
#             "user_id": user_id,
#             "db_id": db_id
#         })
#         ids.append(f"{db_id}_{i}")
#         print(f"  ➕ Prepared doc for {fq_table}: {doc}")

#     embeddings = embedder.encode(docs).tolist()
#     print(f"🧩 Generated {len(embeddings)} embeddings, uploading to Chroma...")

#     collection.upsert(
#         ids=ids,
#         embeddings=embeddings,
#         metadatas=metas,
#         documents=docs
#     )
#     print("✅ Schema catalog embedded into Chroma.")








# views.py
import json
import uuid
import logging
from urllib.parse import quote_plus
 
import uuid, json
import uuid, json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from urllib.parse import quote_plus



import uuid
import traceback
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
from django.conf import settings

"""
production_views.py

Drop-in replacement for your views with production-ready code.
Simply import these views in your urls.py instead of the old ones.

USAGE:
1. Save this file as: genai_app/production_views.py
2. In urls.py, change:
   from genai_app import views
   TO:
   from genai_app import production_views as views

That's it! No other changes needed.
"""

import json
import uuid
import traceback
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
from django.conf import settings
from genai_app.langgraph_logic.langgraph_runner import SessionManager,build_schema_catalog



# ============================================================================
# TABLE SELECTION - Smart Selection
# ============================================================================

from sqlalchemy import create_engine, inspect, text
from typing import Dict, List, Any, Optional

def build_schema_catalogbfryaml(
    engine,
    prefer_schema: str = "stage",
    allowed_tables: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Build schema catalog with optional table filtering.
    
    Args:
        engine: SQLAlchemy engine
        prefer_schema: Preferred schema to prioritize
        allowed_tables: Dict mapping schema names to lists of allowed table names
                       Example: {"stage": ["app_main_2024", "loan_main_2024"]}
                       If None, includes all tables
    
    Returns:
        Dict mapping full table names to their metadata
    """
    
    # Default filtering: Only app_main and loan_main from stage schema
    if allowed_tables is None:
        allowed_tables = {
            "stage": ["app_main_2024", "loan_main_2024"]
        }
    
    print(f"🔎 Building schema catalog (prefer schema: {prefer_schema})...")
    print(f"📋 Table filter: {allowed_tables}")
    
    inspector = inspect(engine)
    schemas = inspector.get_schema_names()
    
    print(f"📊 Found {len(schemas)} schemas: {schemas}")
    
    schema_catalog = {}
    
    # Helper function to check if table is allowed
    def is_table_allowed(schema: str, table: str) -> bool:
        if not allowed_tables:
            return True  # No filter, allow all
        
        if schema not in allowed_tables:
            return False  # Schema not in allowed list
        
        # Check if table matches any pattern in allowed list
        allowed = allowed_tables[schema]
        for pattern in allowed:
            # Exact match or prefix match (for tables like app_main_2024, app_main_2025, etc.)
            if table == pattern or table.startswith(pattern.replace("_2024", "")):
                return True
        
        return False
    
    # Process schemas in priority order (prefer_schema first)
    schema_order = [prefer_schema] + [s for s in schemas if s != prefer_schema]
    
    for schema in schema_order:
        # Skip system schemas
        if schema in ['information_schema', 'pg_catalog']:
            continue
        
        try:
            table_names = inspector.get_table_names(schema=schema)
            
            for table in table_names:
                # Check if table is allowed
                if not is_table_allowed(schema, table):
                    print(f"    ⏭️ Skipped \"{schema}\".\"{table}\" (filtered out)")
                    continue
                
                full_table_name = f"{schema}.{table}"
                
                # Skip if already processed from preferred schema
                table_base = table.split('.')[0]
                if any(k.endswith(f".{table_base}") for k in schema_catalog.keys()):
                    print(f"    ⏭️ Skipped \"{schema}\".\"{table}\" (already have from {prefer_schema})")
                    continue
                
                try:
                    columns = inspector.get_columns(table, schema=schema)
                    pk_constraint = inspector.get_pk_constraint(table, schema=schema)
                    foreign_keys = inspector.get_foreign_keys(table, schema=schema)
                    
                    pk_columns = pk_constraint.get('constrained_columns', []) if pk_constraint else []
                    
                    column_info = []
                    for col in columns:
                        col_dict = {
                            'name': col['name'],
                            'type': str(col['type']),
                            'nullable': col.get('nullable', True),
                            'default': str(col.get('default')) if col.get('default') else None,
                            'is_primary_key': col['name'] in pk_columns
                        }
                        column_info.append(col_dict)
                    
                    # Get sample values
                    sample_values = {}
                    try:
                        with engine.connect() as conn:
                            sample_query = text(
                                f'SELECT * FROM "{schema}"."{table}" LIMIT 3'
                            )
                            result = conn.execute(sample_query)
                            rows = result.fetchall()
                            
                            if rows:
                                for col_name in result.keys():
                                    values = [row[result.keys().index(col_name)] for row in rows if row[result.keys().index(col_name)] is not None]
                                    if values:
                                        sample_values[col_name] = values[:3]
                    except Exception as e:
                        print(f"    ⚠️ Could not get samples from {full_table_name}: {e}")
                    
                    schema_catalog[full_table_name] = {
                        'schema': schema,
                        'table': table,
                        'columns': column_info,
                        'primary_keys': pk_columns,
                        'foreign_keys': foreign_keys,
                        'sample_values': sample_values
                    }
                    
                    print(f"    ✅ Included \"{schema}\".\"{table}\" → {len(columns)} cols")
                    
                except Exception as e:
                    print(f"    ❌ Error processing {schema}.{table}: {e}")
                    continue
        
        except Exception as e:
            print(f"  ❌ Error accessing schema {schema}: {e}")
            continue
    
    print(f"✅ Schema catalog built with {len(schema_catalog)} tables")
    
    return schema_catalog


def get_table_schema_text(schema_catalog: Dict[str, Any], table_name: str) -> str:
    """
    Generate a human-readable schema description for a specific table.
    """
    if table_name not in schema_catalog:
        return f"Table {table_name} not found in catalog."
    
    table_info = schema_catalog[table_name]
    schema = table_info['schema']
    table = table_info['table']
    columns = table_info['columns']
    
    lines = [f"Table: {schema}.{table}"]
    lines.append("-" * 50)
    
    for col in columns:
        pk_marker = " (PK)" if col['is_primary_key'] else ""
        nullable = "NULL" if col['nullable'] else "NOT NULL"
        lines.append(f"  {col['name']}: {col['type']} {nullable}{pk_marker}")
    
    # Add sample values if available
    if table_info.get('sample_values'):
        lines.append("\nSample values:")
        for col_name, values in list(table_info['sample_values'].items())[:5]:
            lines.append(f"  {col_name}: {values}")
    
    return "\n".join(lines)


import json
import traceback
from urllib.parse import quote_plus
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from sqlalchemy import create_engine, text
import yaml
from sqlalchemy import inspect, text

# def build_schema_catalog(engine, allowed_tables=None, prefer_schema=None):
#     print("Building schema catalog views...")
#     """
#     Automatically builds a schema catalog from ALL schemas in a PostgreSQL database.

#     Returns:
#         dict: {table_name: {columns: [...], indexes: [...], joins: [...], unique_constraints: [...], enums: {...}}}
#     """
#     inspector = inspect(engine)
#     schema_catalog = {}

#     # Fetch all schemas in the database
#     schemas = inspector.get_schema_names()

#     for schema in schemas:
#         tables = inspector.get_table_names(schema=schema)
#         for table_name in tables:
#             full_table_name = f"{schema}.{table_name}"  # fully qualified name
#             if allowed_tables and full_table_name not in allowed_tables and table_name not in allowed_tables:
#                 continue

#             # Columns
#             columns_info = []
#             for col in inspector.get_columns(table_name, schema=schema):
#                 col_info = {
#                     "name": col["name"],
#                     "type": str(col["type"]),
#                     "nullable": col["nullable"],
#                     "default": col.get("default"),
#                     "is_primary": col["name"] in [pk["column_name"] for pk in inspector.get_pk_constraint(table_name, schema=schema).get("constrained_columns", [])]
#                 }
#                 columns_info.append(col_info)

#             # Primary keys
#             pk_info = inspector.get_pk_constraint(table_name, schema=schema)

#             # Indexes
#             indexes = [idx["name"] for idx in inspector.get_indexes(table_name, schema=schema)]

#             # Unique constraints
#             uniques = [uc["name"] for uc in inspector.get_unique_constraints(table_name, schema=schema)]

#             # Foreign keys / joins
#             joins = []
#             for fk in inspector.get_foreign_keys(table_name, schema=schema):
#                 join_info = {
#                     "table": f"{fk['referred_schema']}.{fk['referred_table']}" if fk.get('referred_schema') else fk["referred_table"],
#                     "left_columns": fk["constrained_columns"],
#                     "right_columns": fk["referred_columns"],
#                     "type": "INNER"  # default
#                 }
#                 joins.append(join_info)

#             # Enum types (basic)
#             enums = {}
#             for col in columns_info:
#                 if "ENUM" in str(col["type"]).upper():
#                     enums[col["name"]] = get_enum_values(engine, str(col["type"]))

#             # Assemble table metadata
#             table_metadata = {
#                 "schema": schema,
#                 "table": table_name,
#                 "columns": columns_info,
#                 "indexes": indexes,
#                 "unique_constraints": uniques,
#                 "joins": joins,
#                 "enums": enums,
#                 "keywords": [table_name]  # optional: can add custom keywords
#             }

#             schema_catalog[full_table_name] = table_metadata

#     return schema_catalog


from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy import inspect, text
from pathlib import Path
import yaml

def build_schema_catalog(engine, allowed_tables=None, prefer_schema=None):
    print("Building schema catalog views...")
    inspector = inspect(engine)
    schema_catalog = {}

    # Fetch all schemas
    schemas = inspector.get_schema_names()
    for schema in schemas:
        if prefer_schema and schema != prefer_schema:
            continue
        tables = inspector.get_table_names(schema=schema)
        for table_name in tables:
            full_table_name = f"{schema}.{table_name}"
            if allowed_tables and full_table_name not in allowed_tables and table_name not in allowed_tables:
                continue

            # Columns
            columns_info = []
            # Use information_schema to avoid SQLAlchemy caching issues
            sql = text(f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table
                ORDER BY ordinal_position
            """)
            with engine.connect() as conn:
                result = conn.execute(sql, {"schema": schema, "table": table_name})
                for row in result.mappings():  # ✅ This makes rows dict-like
                    columns_info.append({
                        "name": row["column_name"],
                        "type": row["data_type"],
                        "nullable": row["is_nullable"] == "YES",
                        "default": row["column_default"],
                        "is_primary": row["column_name"] in inspector.get_pk_constraint(table_name, schema=schema).get("constrained_columns", [])
                    })

            # Indexes
            indexes = [idx["name"] for idx in inspector.get_indexes(table_name, schema=schema)]

            # Unique constraints
            uniques = [uc["name"] for uc in inspector.get_unique_constraints(table_name, schema=schema)]

            # Foreign keys / joins
            joins = []
            for fk in inspector.get_foreign_keys(table_name, schema=schema):
                join_info = {
                    "table": f"{fk['referred_schema']}.{fk['referred_table']}" if fk.get('referred_schema') else fk["referred_table"],
                    "left_columns": fk["constrained_columns"],
                    "right_columns": fk["referred_columns"],
                    "type": "INNER"
                }
                joins.append(join_info)

            # Assemble table metadata
            table_metadata = {
                "schema": schema,
                "table": table_name,
                "columns": columns_info,
                "indexes": indexes,
                "unique_constraints": uniques,
                "joins": joins,
                "enums": {},  # optional
                "keywords": [table_name]
            }

            schema_catalog[full_table_name] = table_metadata

    return schema_catalog


# ✅ Optimized YAML save
def save_schema_to_yaml(schema_catalog, output_dir='yaml_schema'):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    updated_count = 0

    for table_name, table_data in schema_catalog.items():
        file_path = output_path / f"{table_name}.yaml"
        new_yaml_str = yaml.safe_dump(table_data, sort_keys=False)

        if file_path.exists():
            with open(file_path, 'r') as f:
                existing_yaml_str = f.read()
            if existing_yaml_str == new_yaml_str:
                continue  # Skip unchanged files

        with open(file_path, 'w') as f:
            f.write(new_yaml_str)
        updated_count += 1
        print(f"✅ YAML updated for table: {table_name}")

    print(f"✅ YAML save complete. {updated_count} table(s) updated in {output_dir}")



def get_enum_values(engine, enum_type_name):
    """
    Fetches enum values for a given enum type from PostgreSQL
    """
    sql = text(f"SELECT unnest(enum_range(NULL::{enum_type_name})) AS value")
    with engine.connect() as conn:
        result = conn.execute(sql)
        return [row["value"] for row in result.fetchall()]

# -----------------------------
# Helper to save YAML files
# -----------------------------
# def save_schema_to_yaml(schema_catalog, output_dir='yaml_schema'):
#     """Save the schema catalog as YAML files for each table."""
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True)
#     for table_name, table_data in schema_catalog.items():
#         file_path = output_path / f"{table_name}.yaml"
#         with open(file_path, 'w') as f:
#             yaml.safe_dump(table_data, f, sort_keys=False)
#     print(f"✅ YAML files saved in {output_dir}")


import yaml
from pathlib import Path

def save_schema_to_yaml(schema_catalog, output_dir='yaml_schema'):
    """Save the schema catalog as YAML files for each table, only updating changed files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    updated_count = 0

    for table_name, table_data in schema_catalog.items():
        file_path = output_path / f"{table_name}.yaml"
        
        # Convert table data to YAML string
        new_yaml_str = yaml.safe_dump(table_data, sort_keys=False)
        
        # Check if file exists and content is the same
        if file_path.exists():
            with open(file_path, 'r') as f:
                existing_yaml_str = f.read()
            if existing_yaml_str == new_yaml_str:
                # No changes, skip writing
                continue
        
        # Write the new/updated YAML
        with open(file_path, 'w') as f:
            f.write(new_yaml_str)
        updated_count += 1
        print(f"✅ YAML updated for table: {table_name}")

    print(f"✅ YAML save complete. {updated_count} table(s) updated in {output_dir}")





# 🆕 UPDATE connect_database to force refresh
@csrf_exempt
def connect_database(request):
    print("➡️ connect_database called")

    if request.method == "OPTIONS":
        resp = HttpResponse()
        resp["Access-Control-Allow-Origin"] = "*"
        resp["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    try:
        data = json.loads(request.body or "{}")
        user_id = getattr(request.user, 'id', 'anonymous')
        
        # 🆕 CHECK FOR FORCE REFRESH FLAG
        force_refresh = data.get("force_refresh", True)  # Default to True for always fresh schema

        # Use static or dynamic DB config
        if data.get("use_static", True) and hasattr(settings, 'STATIC_DB'):
            db_config = settings.STATIC_DB
            print("📊 Using static database config")
        else:
            db_config = {
                "postgres_user": data.get("postgres_user"),
                "postgres_password": data.get("postgres_password"),
                "postgres_host": data.get("postgres_host"),
                "postgres_port": data.get("postgres_port", 5432),
                "postgres_db": data.get("postgres_db")
            }
            print("📊 Using dynamic credentials")

        # Build connection string
        pg_user = db_config["postgres_user"]
        pg_pass = db_config["postgres_password"]
        pg_host = db_config["postgres_host"]
        pg_port = db_config["postgres_port"]
        pg_db = db_config["postgres_db"]

        encoded_pwd = quote_plus(str(pg_pass))
        conn_str = f"postgresql://{pg_user}:{encoded_pwd}@{pg_host}:{pg_port}/{pg_db}"

        engine = create_engine(
            conn_str,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={"connect_timeout": 8}
        )

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Connection successful")

        # 🆕 FORCE REFRESH: Delete old YAML files before building new schema
        if force_refresh:
            yaml_path = Path('yaml_schema')
            if yaml_path.exists():
                import shutil
                shutil.rmtree(yaml_path)
                print("🔄 Old YAML schema cleared - will regenerate fresh")
        
        # Build schema catalog with table filtering
        allowed_tables = data.get("allowed_tables", None)  # optional filter
        try:
            schema_catalog = build_schema_catalog(
                engine,
                prefer_schema="stage",
                allowed_tables=allowed_tables
            )
            print(f"✅ Schema catalog built: {len(schema_catalog)} tables found")

            # --- Auto-generate YAML files ---
            save_schema_to_yaml(schema_catalog, output_dir='yaml_schema')
        except Exception as e:
            print(f"❌ Failed to build schema catalog: {e}")
            traceback.print_exc()
            return JsonResponse({"error": f"Failed to build schema: {str(e)}"}, status=500)

        if not schema_catalog:
            return JsonResponse(
                {"error": "No allowed tables found in database. Check your table filter configuration."},
                status=400
            )

        # Create session (memory or Redis)
        try:
            session_id = SessionManager.create_session(
                engine=engine,
                user_id=user_id,
                schema_catalog=schema_catalog
            )
            print(f"✅ Session created: {session_id}")
        except Exception as e:
            print(f"❌ Session creation failed: {e}")
            traceback.print_exc()
            return JsonResponse({"error": f"Failed to create session: {str(e)}"}, status=500)

        # Optional schema embedding
        if embed_schema_catalog and embedder and collection:
            try:
                embed_schema_catalog(
                    user_id=user_id,
                    db_id=session_id,
                    schema_catalog=schema_catalog,
                    embedder=embedder,
                    collection=collection
                )
                print("✅ Schema embedded")
            except Exception as e:
                print(f"⚠️ Embedding failed (non-critical): {e}")

        return JsonResponse({
            "message": "Connected successfully",
            "session_id": session_id,
            "tables": list(schema_catalog.keys()),
            "table_count": len(schema_catalog),
            "storage": "redis" if SessionManager._use_redis else "memory",
            "schema_refreshed": force_refresh  # 🆕 INDICATE IF SCHEMA WAS REFRESHED
        })

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

# -----------------------------
# Main DB connect view
# -----------------------------
@csrf_exempt
def connect_database1010(request):
    print("➡️ connect_database called")

    if request.method == "OPTIONS":
        resp = HttpResponse()
        resp["Access-Control-Allow-Origin"] = "*"
        resp["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    try:
        data = json.loads(request.body or "{}")
        user_id = getattr(request.user, 'id', 'anonymous')

        # Use static or dynamic DB config
        if data.get("use_static", True) and hasattr(settings, 'STATIC_DB'):
            db_config = settings.STATIC_DB
            print("📊 Using static database config")
        else:
            db_config = {
                "postgres_user": data.get("postgres_user"),
                "postgres_password": data.get("postgres_password"),
                "postgres_host": data.get("postgres_host"),
                "postgres_port": data.get("postgres_port", 5432),
                "postgres_db": data.get("postgres_db")
            }
            print("📊 Using dynamic credentials")

        # Build connection string
        pg_user = db_config["postgres_user"]
        pg_pass = db_config["postgres_password"]
        pg_host = db_config["postgres_host"]
        pg_port = db_config["postgres_port"]
        pg_db = db_config["postgres_db"]

        encoded_pwd = quote_plus(str(pg_pass))
        conn_str = f"postgresql://{pg_user}:{encoded_pwd}@{pg_host}:{pg_port}/{pg_db}"

        engine = create_engine(
            conn_str,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={"connect_timeout": 8}
        )

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Connection successful")

        # Build schema catalog with table filtering
        allowed_tables = data.get("allowed_tables", None)  # optional filter
        try:
            schema_catalog = build_schema_catalog(
                engine,
                prefer_schema="stage",
                allowed_tables=allowed_tables
            )
            print(f"✅ Schema catalog built: {len(schema_catalog)} tables found")

            # --- Auto-generate YAML files ---
            save_schema_to_yaml(schema_catalog, output_dir='yaml_schema')
        except Exception as e:
            print(f"❌ Failed to build schema catalog: {e}")
            traceback.print_exc()
            return JsonResponse({"error": f"Failed to build schema: {str(e)}"}, status=500)

        if not schema_catalog:
            return JsonResponse(
                {"error": "No allowed tables found in database. Check your table filter configuration."},
                status=400
            )

        # Create session (memory or Redis)
        try:
            session_id = SessionManager.create_session(
                engine=engine,
                user_id=user_id,
                schema_catalog=schema_catalog
            )
            print(f"✅ Session created: {session_id}")
        except Exception as e:
            print(f"❌ Session creation failed: {e}")
            traceback.print_exc()
            return JsonResponse({"error": f"Failed to create session: {str(e)}"}, status=500)

        # Optional schema embedding
        if embed_schema_catalog and embedder and collection:
            try:
                embed_schema_catalog(
                    user_id=user_id,
                    db_id=session_id,
                    schema_catalog=schema_catalog,
                    embedder=embedder,
                    collection=collection
                )
                print("✅ Schema embedded")
            except Exception as e:
                print(f"⚠️ Embedding failed (non-critical): {e}")

        return JsonResponse({
            "message": "Connected successfully",
            "session_id": session_id,
            "tables": list(schema_catalog.keys()),
            "table_count": len(schema_catalog),
            "storage": "redis" if SessionManager._use_redis else "memory"
        })

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


# # ============================================================================
# # VIEWS bfr yaml file
# # ============================================================================
# @csrf_exempt
# def connect_database(request):
#     """Connect to database with filtered table selection."""
#     print("➡️ connect_database called")
    
#     if request.method == "OPTIONS":
#         resp = HttpResponse()
#         resp["Access-Control-Allow-Origin"] = "*"
#         resp["Access-Control-Allow-Methods"] = "POST, OPTIONS"
#         resp["Access-Control-Allow-Headers"] = "Content-Type"
#         return resp
    
#     try:
#         data = json.loads(request.body or "{}")
#         user_id = request.user.id

#         # user_id = data.get("user_id", "static_user")
        
#         # Support both static and dynamic config
#         if data.get("use_static", True) and hasattr(settings, 'STATIC_DB'):
#             db_config = settings.STATIC_DB
#             print("📊 Using static database config")
#         else:
#             db_config = {
#                 "postgres_user": data.get("postgres_user"),
#                 "postgres_password": data.get("postgres_password"),
#                 "postgres_host": data.get("postgres_host"),
#                 "postgres_port": data.get("postgres_port", 5432),
#                 "postgres_db": data.get("postgres_db")
#             }
#             print("📊 Using dynamic credentials")
        
#         # Build connection
#         pg_user = db_config["postgres_user"]
#         pg_pass = db_config["postgres_password"]
#         pg_host = db_config["postgres_host"]
#         pg_port = db_config["postgres_port"]
#         pg_db = db_config["postgres_db"]
        
#         encoded_pwd = quote_plus(str(pg_pass))
#         conn_str = f"postgresql://{pg_user}:{encoded_pwd}@{pg_host}:{pg_port}/{pg_db}"
        
#         engine = create_engine(
#             conn_str,
#             pool_pre_ping=True,
#             pool_recycle=1800,
#             connect_args={"connect_timeout": 8}
#         )
        
#         # Test connection
#         with engine.connect() as conn:
#             conn.execute(text("SELECT 1"))
#         print("✅ Connection successful")
        
#         # Build schema catalog with filtering
#         allowed_tables = {
#             "stage": ["app_main_2024", "loan_main_2024"]
#         }
        
#         try:
#             schema_catalog = build_schema_catalog(
#                 engine,
#                 prefer_schema="stage",
#                 allowed_tables=allowed_tables
#             )
#             print(f"✅ Schema catalog built: {len(schema_catalog)} tables found")
#         except Exception as e:
#             print(f"❌ Failed to build schema catalog: {e}")
#             traceback.print_exc()
#             return JsonResponse({"error": f"Failed to build schema: {str(e)}"}, status=500)
        
#         if not schema_catalog:
#             return JsonResponse(
#                 {"error": "No allowed tables found in database. Check your table filter configuration."},
#                 status=400
#             )
        
#         # Create session (now uses memory fallback if Redis is unavailable)
#         try:
#             session_id = SessionManager.create_session(
#                 engine=engine,
#                 user_id=user_id,
#                 schema_catalog=schema_catalog
#             )
#             print(f"✅ Session created: {session_id}")
#         except Exception as e:
#             print(f"❌ Session creation failed: {e}")
#             traceback.print_exc()
#             return JsonResponse({"error": f"Failed to create session: {str(e)}"}, status=500)
        
#         # Embed schema if available (optional, won't crash if not available)
#         if embed_schema_catalog and embedder and collection:
#             try:
#                 embed_schema_catalog(
#                     user_id=user_id,
#                     db_id=session_id,
#                     schema_catalog=schema_catalog,
#                     embedder=embedder,
#                     collection=collection
#                 )
#                 print("✅ Schema embedded")
#             except Exception as e:
#                 print(f"⚠️ Embedding failed (non-critical): {e}")
        
#         return JsonResponse({
#             "message": "Connected successfully",
#             "session_id": session_id,
#             "tables": list(schema_catalog.keys()),
#             "table_count": len(schema_catalog),
#             "storage": "redis" if SessionManager._use_redis else "memory"
#         })
    
#     except Exception as e:
#         print(f"❌ Connection failed: {e}")
#         traceback.print_exc()
#         return JsonResponse({"error": str(e)}, status=500)


def jsonl_line(obj: dict) -> str:
    """Format JSONL."""
    return json.dumps(obj) + '\n'


# @csrf_exempt
# def ask_question_stream(request):
#     """
#     Two-stage streaming Q&A:
#     Stage 1: Generate SQL and wait for user confirmation
#     Stage 2: Execute SQL and generate insights after confirmation
#     """
#     if request.method == "OPTIONS":
#         resp = HttpResponse()
#         resp["Access-Control-Allow-Origin"] = "*"
#         resp["Access-Control-Allow-Methods"] = "POST, OPTIONS"
#         resp["Access-Control-Allow-Headers"] = "Content-Type"
#         return resp

#     if request.method != "POST":
#         return JsonResponse({"error": "Use POST"}, status=405)

#     try:
#         payload = json.loads(request.body or "{}")
#         session_id = payload.get("session_id")
#         question = payload.get("question")
#         confirm_run = payload.get("confirm_run", False)  # ✅ Check if user confirmed
#         user_id = payload.get("user_id") or (request.user.id if request.user.is_authenticated else None)

#         if not session_id or not question:
#             return JsonResponse({"error": "Missing session_id or question"}, status=400)

#         session_data = SessionManager.get_session(session_id)
#         if not session_data:
#             return JsonResponse({"error": "Session not found"}, status=404)

#         engine = SessionManager.get_engine(session_id)
#         if not engine:
#             return JsonResponse({"error": "Could not create engine"}, status=500)

#         def gen():
#             # ═══════════════════════════════════════════════════════════
#             # STAGE 1: GENERATE SQL (Always execute this stage first)
#             # ═══════════════════════════════════════════════════════════
#             if not confirm_run:
#                 yield jsonl_line({"event": "phase", "message": "Generating SQL query..."})

#                 try:
#                     sql, explanation, _ = run_sql_generation_graph(
#                         question=question,
#                         user_id=user_id,
#                         session_id=session_id,
#                         history=SessionManager.get_conversation_history(session_id)
#                     )

#                     if sql.startswith("-- ERROR"):
#                         yield jsonl_line({
#                             "event": "error", 
#                             "message": f"Failed to generate SQL: {sql}"
#                         })
#                         return

#                     # ✅ Send SQL to frontend for confirmation
#                     yield jsonl_line({
#                         "event": "sql_generated",
#                         "sql": sql,
#                         "explanation": explanation or "SQL query generated successfully",
#                         "message": "Please review the SQL and confirm execution"
#                     })

#                     # ✅ STOP HERE - Wait for user confirmation
#                     # Frontend will make another request with confirm_run=True
#                     return

#                 except Exception as e:
#                     traceback.print_exc()
#                     yield jsonl_line({
#                         "event": "error",
#                         "message": f"SQL generation failed: {str(e)}"
#                     })
#                     return

#             # ═══════════════════════════════════════════════════════════
#             # STAGE 2: EXECUTE SQL (Only if confirm_run=True)
#             # ═══════════════════════════════════════════════════════════
#             else:
#                 # First, regenerate the SQL (user might have triggered regeneration)
#                 yield jsonl_line({"event": "phase", "message": "Generating SQL query..."})
                
#                 try:
#                     sql, explanation, _ = run_sql_generation_graph(
#                         question=question,
#                         user_id=user_id,
#                         session_id=session_id,
#                         history=SessionManager.get_conversation_history(session_id)
#                     )

#                     if sql.startswith("-- ERROR"):
#                         yield jsonl_line({
#                             "event": "error",
#                             "message": f"SQL generation failed: {sql}"
#                         })
#                         return

#                 except Exception as e:
#                     traceback.print_exc()
#                     yield jsonl_line({
#                         "event": "error",
#                         "message": f"SQL generation error: {str(e)}"
#                     })
#                     return

#                 # Now execute the confirmed SQL
#                 yield jsonl_line({"event": "phase", "message": "Executing SQL query..."})
                
#                 try:
#                     start_time = time.time()
#                     with engine.connect() as conn:
#                         result = conn.execute(text(sql))
#                         rows = [dict(row._mapping) for row in result]

#                     execution_time = time.time() - start_time
#                     df = pd.DataFrame(rows).fillna('')
#                     rows = df.to_dict(orient="records")
#                     row_count = len(rows)

#                     # ✅ Send preview of results
#                     yield jsonl_line({
#                         "event": "rows_preview",
#                         "rows": rows[:10],
#                         "row_count": row_count
#                     })

#                 except Exception as e:
#                     traceback.print_exc()
#                     yield jsonl_line({
#                         "event": "error",
#                         "message": f"Query execution failed: {str(e)}"
#                     })
#                     return

#                 # ───────────────────────────────────────────────────────
#                 # Generate Insights, Recommendations, Charts
#                 # ───────────────────────────────────────────────────────
#                 yield jsonl_line({"event": "phase", "message": "Analyzing results..."})
                
#                 narrative_obj = {}
#                 opener = ""
#                 insights = []
#                 recs = []
#                 next_step = ""

#                 # Try structured narrative generation
#                 try:
#                     narrative_obj = safe_generate_narrative(question, sql, rows)
#                     opener = narrative_obj.get("opener", "")
#                     insights = narrative_obj.get("insights", [])
#                     recs = narrative_obj.get("recommendations", [])
#                     next_step = narrative_obj.get("next_step", "")

#                     if opener:
#                         yield jsonl_line({"event": "narrative_opener", "text": opener})
#                     if insights:
#                         yield jsonl_line({"event": "insights", "list": insights})
#                     if recs:
#                         yield jsonl_line({"event": "recommendations", "list": recs})
#                     if next_step:
#                         yield jsonl_line({"event": "next_step", "text": next_step})

#                 except Exception as e:
#                     print(f"⚠️ Structured narrative failed: {e}")

#                 # Fallback LLM narrative
#                 narr = None
#                 try:
#                     narr = llm_generate_narrative(question, rows)
#                     if narr:
#                         yield jsonl_line({"event": "narrative", "obj": narr})
#                 except Exception as e:
#                     print(f"⚠️ LLM narrative failed: {e}")

#                 # Generate recommendation
#                 rec = None
#                 try:
#                     rec = llm_generate_recommendation(question, rows)
#                     if rec:
#                         yield jsonl_line({"event": "recommendation", "text": rec})
#                 except Exception as e:
#                     print(f"⚠️ Recommendation generation failed: {e}")

#                 # Generate chart config
#                 cfg = None
#                 try:
#                     cfg = llm_generate_chart_config(question, rows)
#                     if cfg:
#                         yield jsonl_line({"event": "chart_config", "config": cfg})
#                 except Exception as e:
#                     print(f"⚠️ Chart generation failed: {e}")

#                 # ───────────────────────────────────────────────────────
#                 # Save to conversation history
#                 # ───────────────────────────────────────────────────────
#                 SessionManager.add_to_conversation(session_id, {
#                     "question": question,
#                     "sql": sql,
#                     "row_count": row_count,
#                     "timestamp": datetime.now().isoformat()
#                 })

#                 # ───────────────────────────────────────────────────────
#                 # Send final response
#                 # ───────────────────────────────────────────────────────
#                 answer_text = opener or f"Query executed successfully. Retrieved {row_count} rows."
                
#                 yield jsonl_line({
#                     "event": "final",
#                     "payload": {
#                         "answer": answer_text,
#                         "rows": rows[:50],  # Limit rows sent to frontend
#                         "row_count": row_count,
#                         "chart_config": cfg,
#                         "recommendation": rec,
#                         "narrative": narr or narrative_obj,
#                         "query_used": sql,
#                         "session_id": session_id,
#                         "conversational_opener": opener,
#                         "response_time": f"{execution_time:.2f}s",
#                         "history": SessionManager.get_conversation_history(session_id)
#                     }
#                 })

#         # Return streaming response
#         resp = StreamingHttpResponse(gen(), content_type="application/x-ndjson")
#         resp["Cache-Control"] = "no-cache"
#         resp["X-Accel-Buffering"] = "no"
#         resp["Access-Control-Allow-Origin"] = "*"
#         return resp

#     except Exception as e:
#         traceback.print_exc()
#         return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def ask_question_stream(request):
    """
    Two-stage streaming Q&A:
    Stage 1: Generate SQL and wait for user confirmation
    Stage 2: Execute SQL and generate insights after confirmation
    """
    if request.method == "OPTIONS":
        resp = HttpResponse()
        resp["Access-Control-Allow-Origin"] = "*"
        resp["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    if request.method != "POST":
        return JsonResponse({"error": "Use POST"}, status=405)

    try:
        payload = json.loads(request.body or "{}")
        session_id = payload.get("session_id")
        question = payload.get("question")
        confirm_run = payload.get("confirm_run", False)  # ✅ Check if user confirmed
        regenerate = payload.get("regenerate", False)  # ✅ Check if regeneration requested
        user_id = payload.get("user_id") or (request.user.id if request.user.is_authenticated else None)

        if not session_id or not question:
            return JsonResponse({"error": "Missing session_id or question"}, status=400)

        session_data = SessionManager.get_session(session_id)
        if not session_data:
            return JsonResponse({"error": "Session not found"}, status=404)

        engine = SessionManager.get_engine(session_id)
        if not engine:
            return JsonResponse({"error": "Could not create engine"}, status=500)

        def gen():
            # STAGE 1: GENERATE SQL (Always execute this stage first)
            if not confirm_run:
                yield jsonl_line({"event": "phase", "message": "Generating SQL query..."})

                try:
                    # ✅ Add variation for regeneration to get different SQL
                    history = SessionManager.get_conversation_history(session_id)
                    if regenerate:
                        # Add a note to the history to encourage different approach
                        history.append({
                            "role": "system",
                            "content": "Previous SQL was rejected. Generate a different approach."
                        })
                    
                    sql, explanation, _ = run_sql_generation_graph(
                        question=question,
                        user_id=user_id,
                        session_id=session_id,
                        history=history
                    )

                    if sql.startswith("-- ERROR"):
                        yield jsonl_line({
                            "event": "error", 
                            "message": f"Failed to generate SQL: {sql}"
                        })
                        return

                    # ✅ Send SQL to frontend for confirmation
                    yield jsonl_line({
                        "event": "sql_generated",
                        "sql": sql,
                        "explanation": explanation or "SQL query generated successfully",
                        "message": "Please review the SQL and confirm execution"
                    })

                    # ✅ STOP HERE - Wait for user confirmation
                    # Frontend will make another request with confirm_run=True
                    return

                except Exception as e:
                    traceback.print_exc()
                    yield jsonl_line({
                        "event": "error",
                        "message": f"SQL generation failed: {str(e)}"
                    })
                    return

            # STAGE 2: EXECUTE SQL (Only if confirm_run=True)
            else:
                # First, regenerate the SQL (user might have triggered regeneration)
                yield jsonl_line({"event": "phase", "message": "Generating SQL query..."})
                
                try:
                    sql, explanation, _ = run_sql_generation_graph(
                        question=question,
                        user_id=user_id,
                        session_id=session_id,
                        history=SessionManager.get_conversation_history(session_id)
                    )

                    if sql.startswith("-- ERROR"):
                        yield jsonl_line({
                            "event": "error",
                            "message": f"SQL generation failed: {sql}"
                        })
                        return

                except Exception as e:
                    traceback.print_exc()
                    yield jsonl_line({
                        "event": "error",
                        "message": f"SQL generation error: {str(e)}"
                    })
                    return

                # Now execute the confirmed SQL
                yield jsonl_line({"event": "phase", "message": "Executing SQL query..."})
                
                try:
                    start_time = time.time()
                    with engine.connect() as conn:
                        result = conn.execute(text(sql))
                        rows = [dict(row._mapping) for row in result]

                    execution_time = time.time() - start_time
                    df = pd.DataFrame(rows).fillna('')
                    rows = df.to_dict(orient="records")
                    row_count = len(rows)

                    # ✅ Send preview of results
                    yield jsonl_line({
                        "event": "rows_preview",
                        "rows": rows[:10],
                        "row_count": row_count
                    })

                except Exception as e:
                    traceback.print_exc()
                    error_message = str(e)
                    
                    # ✅ Extract meaningful error from SQL exception
                    if "does not exist" in error_message:
                        error_message = f"Column/Table error: {error_message.split('LINE')[0].strip()}"
                    elif "syntax error" in error_message.lower():
                        error_message = f"SQL syntax error: {error_message}"
                    
                    yield jsonl_line({
                        "event": "error",
                        "message": f"Query execution failed: {error_message}",
                        "sql": sql,
                        "suggestion": "Please click 'Regenerate' to try a different query."
                    })
                    return

                # ───────────────────────────────────────────────────────
                # Generate Insights, Recommendations, Charts
                # ───────────────────────────────────────────────────────
                yield jsonl_line({"event": "phase", "message": "Analyzing results..."})
                
                narrative_obj = {}
                opener = ""
                insights = []
                recs = []
                next_step = ""

                # Try structured narrative generation
                try:
                    narrative_obj = safe_generate_narrative(question, sql, rows)
                    opener = narrative_obj.get("opener", "")
                    insights = narrative_obj.get("insights", [])
                    recs = narrative_obj.get("recommendations", [])
                    next_step = narrative_obj.get("next_step", "")

                    if opener:
                        yield jsonl_line({"event": "narrative_opener", "text": opener})
                    if insights:
                        yield jsonl_line({"event": "insights", "list": insights})
                    if recs:
                        yield jsonl_line({"event": "recommendations", "list": recs})
                    if next_step:
                        yield jsonl_line({"event": "next_step", "text": next_step})

                except Exception as e:
                    print(f"⚠️ Structured narrative failed: {e}")

                # Fallback LLM narrative
                narr = None
                try:
                    narr = llm_generate_narrative(question, rows)
                    if narr:
                        yield jsonl_line({"event": "narrative", "obj": narr})
                except Exception as e:
                    print(f"⚠️ LLM narrative failed: {e}")

                # Generate recommendation
                rec = None
                try:
                    rec = llm_generate_recommendation(question, rows)
                    if rec:
                        yield jsonl_line({"event": "recommendation", "text": rec})
                except Exception as e:
                    print(f"⚠️ Recommendation generation failed: {e}")

                # Generate chart config
                cfg = None
                try:
                    cfg = llm_generate_chart_config(question, rows)
                    if cfg:
                        yield jsonl_line({"event": "chart_config", "config": cfg})
                except Exception as e:
                    print(f"⚠️ Chart generation failed: {e}")

                # ───────────────────────────────────────────────────────
                # Save to conversation history
                # ───────────────────────────────────────────────────────
                SessionManager.add_to_conversation(session_id, {
                    "question": question,
                    "sql": sql,
                    "row_count": row_count,
                    "timestamp": datetime.now().isoformat()
                })

                # ───────────────────────────────────────────────────────
                # Send final response
                # ───────────────────────────────────────────────────────
                answer_text = opener or f"Query executed successfully. Retrieved {row_count} rows."
                
                yield jsonl_line({
                    "event": "final",
                    "payload": {
                        "answer": answer_text,
                        "rows": rows[:50],  # Limit rows sent to frontend
                        "row_count": row_count,
                        "chart_config": cfg,
                        "recommendation": rec,
                        "narrative": narr or narrative_obj,
                        "query_used": sql,
                        "session_id": session_id,
                        "conversational_opener": opener,
                        "response_time": f"{execution_time:.2f}s",
                        "history": SessionManager.get_conversation_history(session_id)
                    }
                })

        # Return streaming response
        resp = StreamingHttpResponse(gen(), content_type="application/x-ndjson")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        resp["Access-Control-Allow-Origin"] = "*"
        return resp

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)



@csrf_exempt
def ask_question_stream710(request):
    """Streaming question answering with enriched insights, charts, and narrative."""

    if request.method == "OPTIONS":
        resp = HttpResponse()
        resp["Access-Control-Allow-Origin"] = "*"
        resp["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    if request.method != "POST":
        return JsonResponse({"error": "Use POST"}, status=405)

    try:
        payload = json.loads(request.body or "{}")
        session_id = payload.get("session_id")
        question = payload.get("question")
        user_id = request.user.id

        # user_id = payload.get("user_id", "default_user")

        if not session_id or not question:
            return JsonResponse({"error": "Missing session_id or question"}, status=400)

        session_data = SessionManager.get_session(session_id)
        if not session_data:
            return JsonResponse({"error": "Session not found"}, status=404)

        engine = SessionManager.get_engine(session_id)
        if not engine:
            return JsonResponse({"error": "Could not create engine"}, status=500)

        schema_catalog = session_data.get("schema_catalog", {})

        print(f"📊 Session has {len(schema_catalog)} tables")

        def gen():
            yield jsonl_line({"event": "narrative_opener", "text": "Analyzing..."})
            yield jsonl_line({"event": "phase", "message": "Understanding question..."})

            history = SessionManager.get_conversation_history(session_id)

            try:
                sql, explanation, _ = run_sql_generation_graph(
                    question=question,
                    user_id=user_id,
                    session_id=session_id,
                    history=history
                )

                if sql.startswith("-- ERROR"):
                    yield jsonl_line({"event": "error", "message": sql})
                    return

                yield jsonl_line({"event": "sql", "sql": sql})

            except Exception as e:
                traceback.print_exc()
                yield jsonl_line({"event": "error", "message": f"SQL generation failed: {e}"})
                return

            yield jsonl_line({"event": "phase", "message": "Querying database..."})

            try:
                with engine.connect() as conn:
                    result = conn.execute(text(sql))
                    rows = [dict(row._mapping) for row in result]

                df = pd.DataFrame(rows)
                df = df.fillna('')
                rows = df.to_dict(orient="records")

                print(f"✅ Query executed: {len(rows)} rows")

            except Exception as e:
                traceback.print_exc()
                yield jsonl_line({"event": "error", "message": f"Query failed: {e}"})
                return

            row_count = len(rows)

            yield jsonl_line({
                "event": "rows_preview",
                "rows": rows[:10],
                "row_count": row_count
            })

            yield jsonl_line({"event": "phase", "message": "Analyzing results..."})

            summary = ""
            opener = ""
            rec = None
            cfg = None
            narr = None

            try:
                # 🔍 Rich insights from narrative generator
                narrative_obj = safe_generate_narrative(question, sql, rows)
                opener = narrative_obj.get("opener", "")
                insights = narrative_obj.get("insights", [])
                recs = narrative_obj.get("recommendations", [])
                next_step = narrative_obj.get("next_step", "")

                if insights:
                    yield jsonl_line({"event": "insights", "list": insights})
                if recs:
                    yield jsonl_line({"event": "recommendations", "list": recs})
                if next_step:
                    yield jsonl_line({"event": "next_step", "text": next_step})

                summary = " ".join(insights) if insights else opener

            except Exception as e:
                print(f"⚠️ Narrative generation failed: {e}")
                summary = ""

            try:
                narr = llm_generate_narrative(question, rows)
                if narr:
                    yield jsonl_line({"event": "narrative", "obj": narr})
            except Exception:
                pass

            try:
                rec = llm_generate_recommendation(question, rows)
                if rec:
                    yield jsonl_line({"event": "recommendation", "text": rec})
            except Exception:
                pass

            try:
                cfg = llm_generate_chart_config(question, rows)
                if cfg:
                    yield jsonl_line({"event": "chart", "config": cfg})
            except Exception:
                pass

            # Format final fallback answer
            if row_count == 0:
                answer = "No records found."
            elif row_count == 1:
                answer = ", ".join([f"{k}: {v}" for k, v in rows[0].items()])
            elif row_count > 50:
                answer = f"Found {row_count} results. Too many to display here — use the CSV download."
            else:
                preview = "\n".join([", ".join([str(v) for v in row.values()]) for row in rows[:3]])
                answer = preview
                if row_count > 3:
                    answer += f"\n...and {row_count - 3} more rows."

            SessionManager.add_to_conversation(session_id, {
                "question": question,
                "sql": sql,
                "row_count": row_count,
                "timestamp": datetime.now().isoformat()
            })

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
                    "session_id": session_id,
                    "conversational_opener": opener,
                    "history": SessionManager.get_conversation_history(session_id)
                }
            })

        resp = StreamingHttpResponse(gen(), content_type="application/x-ndjson")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        resp["Access-Control-Allow-Origin"] = "*"
        return resp

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


# @csrf_exempt
# def ask_question_stream(request):
#     """Streaming question answering."""
    
#     if request.method == "OPTIONS":
#         resp = HttpResponse()
#         resp["Access-Control-Allow-Origin"] = "*"
#         resp["Access-Control-Allow-Methods"] = "POST, OPTIONS"
#         resp["Access-Control-Allow-Headers"] = "Content-Type"
#         return resp
    
#     if request.method != "POST":
#         return JsonResponse({"error": "Use POST"}, status=405)
    
#     try:
#         payload = json.loads(request.body or "{}")
#         session_id = payload.get("session_id")
#         question = payload.get("question")
#         user_id = payload.get("user_id", "default_user")
        
#         if not session_id or not question:
#             return JsonResponse({"error": "Missing session_id or question"}, status=400)
        
#         session_data = SessionManager.get_session(session_id)
#         if not session_data:
#             return JsonResponse({"error": "Session not found"}, status=404)
        
#         engine = SessionManager.get_engine(session_id)
#         if not engine:
#             return JsonResponse({"error": "Could not create engine"}, status=500)
        
#         schema_catalog = session_data.get("schema_catalog", {})
        
#         print(f"📊 Session has {len(schema_catalog)} tables")
        
#         def gen():
#             yield jsonl_line({"event": "narrative_opener", "text": "Analyzing..."})
#             yield jsonl_line({"event": "phase", "message": "Understanding question..."})
            
#             history = SessionManager.get_conversation_history(session_id)
            
#             try:
#                 sql, explanation, _ = run_sql_generation_graph(
#                     question=question,
#                     user_id=user_id,
#                     session_id=session_id,
#                     history=history
#                 )
                
#                 if sql.startswith("-- ERROR"):
#                     yield jsonl_line({"event": "error", "message": sql})
#                     return
                
#                 yield jsonl_line({"event": "sql", "sql": sql})
                
#             except Exception as e:
#                 traceback.print_exc()
#                 yield jsonl_line({"event": "error", "message": f"SQL generation failed: {e}"})
#                 return
            
#             yield jsonl_line({"event": "phase", "message": "Querying database..."})
            
#             try:
#                 with engine.connect() as conn:
#                     result = conn.execute(text(sql))
#                     rows = [dict(row._mapping) for row in result]
                
#                 df = pd.DataFrame(rows)
#                 df = df.fillna('')
#                 rows = df.to_dict(orient="records")
                
#                 print(f"✅ Query executed: {len(rows)} rows")
                
#             except Exception as e:
#                 traceback.print_exc()
#                 yield jsonl_line({"event": "error", "message": f"Query failed: {e}"})
#                 return
            
#             row_count = len(rows)
            
#             yield jsonl_line({
#                 "event": "rows_preview",
#                 "rows": rows[:10],
#                 "row_count": row_count
#             })
            
#             yield jsonl_line({"event": "phase", "message": "Analyzing results..."})
            
#             if row_count == 0:
#                 answer = "No records found."
#             elif row_count == 1:
#                 answer = ", ".join([f"{k}: {v}" for k, v in rows[0].items()])
#             else:
#                 preview = "\n".join([", ".join([str(v) for v in row.values()]) for row in rows[:3]])
#                 answer = preview
#                 if row_count > 3:
#                     answer += f"\n...and {row_count - 3} more"
            
#             SessionManager.add_to_conversation(session_id, {
#                 "question": question,
#                 "sql": sql,
#                 "row_count": row_count,
#                 "timestamp": datetime.now().isoformat()
#             })
            
#             yield jsonl_line({
#                 "event": "final",
#                 "payload": {
#                     "answer": answer,
#                     "rows": rows[:50],
#                     "row_count": row_count,
#                     "query_used": sql,
#                     "session_id": session_id,
#                     "history": SessionManager.get_conversation_history(session_id)
#                 }
#             })
        
#         resp = StreamingHttpResponse(gen(), content_type="application/x-ndjson")
#         resp["Cache-Control"] = "no-cache"
#         resp["X-Accel-Buffering"] = "no"
#         resp["Access-Control-Allow-Origin"] = "*"
#         return resp
        
    # except Exception as e:
    #     traceback.print_exc()
    #     return JsonResponse({"error": str(e)}, status=500)







@csrf_exempt
def get_session_info(request):
    """Get information about a session."""
    try:
        data = json.loads(request.body or "{}")
        session_id = data.get("session_id")
        
        if not session_id:
            return JsonResponse({"error": "Missing session_id"}, status=400)
        
        session_data = SessionManager.get_session(session_id)
        if not session_data:
            return JsonResponse({"error": "Session not found"}, status=404)
        
        history = SessionManager.get_conversation_history(session_id)
        
        return JsonResponse({
            "session_id": session_id,
            "user_id": session_data.get("user_id"),
            "created_at": session_data.get("created_at"),
            "table_count": len(session_data.get("schema_catalog", {})),
            "tables": list(session_data.get("schema_catalog", {}).keys()),
            "conversation_count": len(history)
        })
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def disconnect_database(request):
    """Disconnect and cleanup a session."""
    try:
        data = json.loads(request.body or "{}")
        session_id = data.get("session_id")
        
        if not session_id:
            return JsonResponse({"error": "Missing session_id"}, status=400)
        
        SessionManager.delete_session(session_id)
        
        return JsonResponse({"message": "Session disconnected successfully"})
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


# ============================================================================
# URL CONFIGURATION
# Add these to your urls.py:
# ============================================================================

# def get_session_info(request):
#     """Get information about a session"""
#     if request.method != 'GET':
#         return JsonResponse({'error': 'Invalid method'}, status=405)
    
#     session_id = request.GET.get('session_id')
#     if not session_id:
#         return JsonResponse({'error': 'Session ID required'}, status=400)
    
#     try:
#         df = dataframe_map.get(session_id)
#         if df is None:
#             return JsonResponse({'error': 'Session not found'}, status=404)
        
#         metadata = metadata_cache.get(session_id, {})
#         memory = conversation_memory.get(session_id, [])
        
#         return JsonResponse({
#             'session_id': session_id,
#             'dataset_info': {
#                 'rows': len(df),
#                 'columns': len(df.columns),
#                 'column_names': df.columns.tolist(),
#                 'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
#             },
#             'processing_info': {
#                 'chunks_created': len(metadata.get('chunks', [])),
#                 'created_at': metadata.get('created_at'),
#                 'data_characteristics': metadata.get('data_characteristics', {})
#             },
#             'conversation_history': len(memory),
#             'last_questions': [q for q, a in memory[-3:]]  # Last 3 questions
#         })
    
#     except Exception as e:
#         logger.error(f"Session info error: {e}")
#         return JsonResponse({'error': str(e)}, status=500)
    
# @csrf_exempt
# def connect_database(request):
#     """Enhanced database connection with better session management"""
#     print("➡️ connect_database called")
#     print(f"📊 Current session_store before connection: {list(session_store.keys())}")
    
#     try:
#         data = settings.STATIC_DB
#         user = "static_user"

#         pg_user = data["postgres_user"]
#         pg_pass = data["postgres_password"]
#         pg_host = data["postgres_host"]
#         pg_port = data["postgres_port"]
#         pg_db = data["postgres_db"]

#         encoded_pwd = quote_plus(str(pg_pass))
#         conn_str = f"postgresql://{pg_user}:{encoded_pwd}@{pg_host}:{pg_port}/{pg_db}"

#         engine = create_engine(
#             conn_str,
#             pool_pre_ping=True,
#             pool_recycle=1800,
#             connect_args={"connect_timeout": 8}
#         )
        
#         # Test connection
#         with engine.connect() as conn:
#             conn.execute(text("SELECT 1"))
#         print("✅ Database connection successful")

#         # 🔑 Build full schema catalog
#         schema_catalog = build_schema_catalog(engine)
#         print(f"✅ Schema catalog built with {len(schema_catalog)} tables")

#         session_id = str(uuid.uuid4())
#         session_data = {
#             "engine": engine,
#             "user_id": user,
#             "schema_catalog": schema_catalog
#         }
        
#         # Store session data
#         session_store[session_id] = session_data
#         conversation_memory_store[session_id] = []
        
#         print(f"✅ Session {session_id} created and stored")
#         print(f"📊 Session_store now has: {list(session_store.keys())}")

#         # 🔑 Embed schema catalog into Chroma
#         try:
#             embed_schema_catalog(user_id=user, db_id=session_id,
#                                schema_catalog=schema_catalog,
#                                embedder=embedder, collection=collection)
#             print("✅ Schema catalog embedded successfully")
#         except Exception as embed_error:
#             print(f"⚠️ Schema embedding failed: {embed_error}")
#             # Continue anyway - embedding failure shouldn't break the connection

#         return JsonResponse({
#             "message": "Connected successfully",
#             "session_id": session_id,
#             "has_schema": bool(schema_catalog),
#             "tables": list(schema_catalog.keys()),
#             "session_count": len(session_store)
#         })

#     except Exception as e:
#         print(f"❌ Database connection failed: {e}")
#         traceback.print_exc()
#         return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def debug_sessions(request):
    """Debug endpoint to check session health"""
    check_session_health()
    
    session_details = {}
    for session_id, session_data in session_store.items():
        session_details[session_id] = {
            "has_engine": "engine" in session_data,
            "has_schema": "schema_catalog" in session_data,
            "table_count": len(session_data.get("schema_catalog", {})),
            "user_id": session_data.get("user_id", "unknown")
        }
    
    return JsonResponse({
        "session_count": len(session_store),
        "sessions": list(session_store.keys()),
        "conversation_count": len(conversation_memory_store),
        "session_details": session_details
    })

@csrf_exempt
def connect_databasebfrmultitable(request):
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
            connect_args={"connect_timeout": 8,
                           "options": "-csearch_path=final"},
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
# liberty working (as provided)
# FULL_SCHEMA = '''
# Table: "bi_dwh"."main_cai_lib"
# Columns:
# - own_damage_premium
# - vehicle_age
# - third_party_premium
# - total_premium_payable
# - vehicle_idv
# - total_revenue
# - policy_tenure
# - number_of_claims
# - claims_approved
# - claim_approval_rate
# - customer_tenure
# - customer_life_time_value
# - customerid
# - chassis_number
# - engine_number
# - vehicle_register_number
# - state
# - zone
# - business_type
# - car_manufacturer
# - vehicle_model
# - product_name
# - policy_no
# - tie_up
# - vehicle_model_variant
# - policy_start_date_year
# - policy_end_date_year
# - policy_start_date_month
# - policy_end_date_month
# - is_churn
# - customer_segment
# - branch_name
# - main_churn_reason
# - primary_recommendation
# - insured_client_name
# '''


#lusa
FULL_SCHEMA = '''
Table: "final_lusa"."apploan_metrics_2024"
Columns:
- application_id
- channel_code
- channel_group
- merchant_id
- merchant_vertical
- client_id
- customer_id
- loan_application_purpose
- loan_application_city
- loan_application_state
- requested_amount
- approval_amount
- loan_application_status
- funded_date
- loan_servicer_boarding
- loan_group_boarding
- issuing_bank
- decision_source
- note_principal
- annual_percentage_rate
- loan_term
- loan_amount_funded
- fico_score
- vantage_score
- is_fraud_suspected
- is_fraud_confirmed
- merchant_discount
- loan_application_date
- interest_rate
- annual_income
'''

import sqlparse

def validate_sql_columns(sql: str, valid_columns: set) -> set:
    tokens = sqlparse.parse(sql)[0].tokens
    words = set()
    
    for token in tokens:
        for t in token.flatten():
            if t.ttype is None and t.value not in ('SELECT', 'FROM', 'WHERE', 'AND', 'OR',
                'COUNT', 'SUM', 'AVG', 'ORDER', 'BY',
                'GROUP', 'LIMIT', '(', ')', '=', 'IS',
                'NOT', 'NULL'):
            # if t.ttype is None and t.value not in ('SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'COUNT', '(', ')', '=', 'IS', 'NOT', 'NULL'):
                words.add(t.value)
    
    return words - valid_columns



import re

# _SQL_FENCE = re.compile(
#     r"```(?:sql|postgresql|postgres|pgsql)?\s*([\s\S]*?)\s*```",
#     re.IGNORECASE
# )
# _ANY_FENCE = re.compile(r"```([\s\S]*?)```", re.DOTALL)
# _SELECT_WITH = re.compile(r"(?is)\b(SELECT|WITH)\b.*?(?=(?:```|$))")

# def _cleanup_sql(s: str) -> str:
#     if not s:
#         return ""
#     s = s.strip()
#     # drop leading "SQL:" labels if present
#     s = re.sub(r"(?i)^\s*sql\s*:\s*", "", s)
#     # normalize weird quotes and fix LIMIT spacing
#     s = s.replace("’", "'").replace("‘", "'")
#     s = re.sub(r"\bLIMIT(\d+)", r"LIMIT \1", s, flags=re.IGNORECASE)
#     return s.strip()

# def extract_sql_block(text) -> str:
#     # 1) normalize input
#     if isinstance(text, tuple):
#         text = text[0]
#     if text is None:
#         return ""
#     if not isinstance(text, str):
#         try:
#             text = str(text)
#         except Exception:
#             return ""

#     s = text.strip()
#     if not s:
#         return ""

#     # 2) prefer explicit SQL fences
#     m = _SQL_FENCE.search(s)
#     if m:
#         return _cleanup_sql(m.group(1))

#     # 3) generic fenced code block fallback
#     m = _ANY_FENCE.search(s)
#     if m:
#         body = m.group(1)
#         # try to isolate a SELECT/WITH inside the fence
#         sw = _SELECT_WITH.search(body)
#         return _cleanup_sql(sw.group(0) if sw else body)

#     # 4) no fences: grab first SELECT/WITH chunk from the whole text
#     sw = _SELECT_WITH.search(s)
#     if sw:
#         return _cleanup_sql(sw.group(0))

#     # 5) last resort: return cleaned full text
#     return _cleanup_sql(s)



# def _cleanup_sql(s: str) -> str:
#     if not s:
#         return ""
#     s = s.strip()
#     # drop leading "SQL:" labels if present
#     s = re.sub(r"(?i)^\s*sql\s*:\s*", "", s)
#     # normalize weird quotes and fix LIMIT spacing
#     s = s.replace("’", "'").replace("‘", "'")
#     s = re.sub(r"\bLIMIT(\d+)", r"LIMIT \1", s, flags=re.IGNORECASE)

#     # ✅ NEW: handle multiple SQL statements (keep only first)
#     if ";" in s:
#         parts = [p.strip() for p in s.split(";") if p.strip()]
#         if parts:
#             s = parts[0]

#     return s.strip()

import re

# # --- Regexes ---
# _SQL_FENCE = re.compile(
#     r"```(?:sql|postgresql)?\s*(?P<body>.*?)(?:```|$)",
#     flags=re.IGNORECASE | re.DOTALL,
# )

# _ANY_FENCE = re.compile(
#     r"```\w*\s*(?P<body>.*?)(?:```|$)",
#     flags=re.DOTALL,
# )

# _SELECT_WITH = re.compile(
#     r"\b(SELECT|WITH)\b[\s\S]+?",
#     flags=re.IGNORECASE
# )

# def _cleanup_sql(s: str) -> str:
#     if not s:
#         return ""
#     # remove leading markdown comments and stray backticks
#     s = re.sub(r"^-- .*?$", "", s, flags=re.MULTILINE).strip()
#     s = s.strip("`").strip()
#     return s

import re

_SQL_FENCE = re.compile(r"```(?:sql|postgresql)?\s*(?P<body>.*?)(?:```|$)", re.IGNORECASE|re.DOTALL)
_ANY_FENCE  = re.compile(r"```\w*\s*(?P<body>.*?)(?:```|$)", re.DOTALL)
_SELECT_WITH = re.compile(r"\b(SELECT|WITH)\b[\s\S]+?", re.IGNORECASE)

def _cleanup_sql(s: str) -> str:
    if not s: return ""
    s = re.sub(r"^-- .*?$", "", s, flags=re.MULTILINE).strip()
    return s.strip("`").strip()

def coerce_sql(raw) -> str:
    """Pull a SQL-ish string out of whatever the LLM returned."""
    if raw is None: return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        # common keys the graph might use
        for k in ("sql", "query", "generated_sql", "code", "text"):
            if k in raw and isinstance(raw[k], str):
                return raw[k]
        # fall back to first str-like value
        for v in raw.values():
            if isinstance(v, str) and ("select" in v.lower() or "with" in v.lower() or "```" in v):
                return v
        return str(raw)
    if isinstance(raw, (list, tuple)):
        for item in raw:
            s = coerce_sql(item)
            if s: return s
        return ""
    return str(raw)
def extract_sql_block(text) -> str:
    # --- Guard against None or non-string inputs ---
    if not text or not isinstance(text, str):
        return ""

    s = coerce_sql(text).strip()
    if not s:
        return ""

    # --- Try fenced SQL first ---
    m = _SQL_FENCE.search(s)
    if m:
        body = m.group("body").strip()
        if _SELECT_WITH.search(body):
            return _cleanup_sql(body)

    # --- Try any fence fallback ---
    m = _ANY_FENCE.search(s)
    if m:
        body = m.group("body")
        sw = _SELECT_WITH.search(body)
        return _cleanup_sql(sw.group(0) if sw else body)

    # --- Try bare SELECT extraction ---
    sw = _SELECT_WITH.search(s)
    if sw:
        candidate = s[sw.start():].strip()
        fence_pos = candidate.find("```")
        if fence_pos != -1:
            candidate = candidate[:fence_pos].strip()
        return _cleanup_sql(candidate)

    # --- If nothing looks like SQL, return empty string ---
    if not _SELECT_WITH.search(s):
        return ""

    return _cleanup_sql(s)

# def extract_sql_block(text) -> str:
#     # 1) normalize input
#     if isinstance(text, tuple):
#         text = text[0]
#     if text is None:
#         return ""
#     if not isinstance(text, str):
#         try:
#             text = str(text)
#         except Exception:
#             return ""

#     s = text.strip()
#     if not s:
#         return ""

#     # 2) prefer explicit SQL fences
#     m = _SQL_FENCE.search(s)
#     if m:
#         return _cleanup_sql(m.group(1))

#     # 3) generic fenced code block fallback
#     m = _ANY_FENCE.search(s)
#     if m:
#         body = m.group(1)
#         # try to isolate a SELECT/WITH inside the fence
#         sw = _SELECT_WITH.search(body)
#         return _cleanup_sql(sw.group(0) if sw else body)

#     # 4) no fences: grab first SELECT/WITH chunk from the whole text
#     sw = _SELECT_WITH.search(s)
#     if sw:
#         return _cleanup_sql(sw.group(0))

#     # 5) last resort: return cleaned full text
#     return _cleanup_sql(s)


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

# Groq working code final6-10

# def llm_generate_recommendation(question, rows):
#     print("llm_generate_recommendation",llm_generate_recommendation)
#     # Prepare data preview
#     preview_rows = rows[:10] if rows else []
#     if isinstance(preview_rows, list) and preview_rows and isinstance(preview_rows[0], dict):
#         table_str = "\n".join(", ".join(f"{k}: {v}" for k, v in row.items()) for row in preview_rows)
#     else:
#         table_str = "No rows available."

#     system_prompt = (
#         "You are a data-driven business analyst assistant. Based on the user's question and the query output data, "
#         "generate a concise, actionable, and professional business recommendation. "
#         "Focus on explaining the key trends, risks, or opportunities in simple business terms, "
#         "helping the user understand what actions or decisions they can take next. "
#         "Strictly avoid including any SQL code, technical jargon, or step-by-step query explanations. "
#         "Only provide a high-level insight that would help a manager or decision-maker."
#         "You are a business analyst who writes like a helpful colleague. "
#         "Based on the user's question and the query output data, "
#         "write a concise, actionable recommendation in plain business language. "
#         "Be concrete and tie suggestions to the data shown. "
#         # "Do not include SQL, technical jargon, or step-by-step query explanations. "
#         # "One short paragraph only. No emojis."
#     )

#     user_prompt = f"""User Question: {question}

# Data Preview (first 10 rows):
# {table_str}

# Please provide only a one-paragraph business recommendation. Do not include SQL queries or explanations.
# """

#     # API headers
#     # GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Ensure this is set in your environment

#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     # Truncation safeguard
#     full_prompt = system_prompt + "\n\n" + user_prompt
#     word_limit = 5000
#     if len(full_prompt.split()) > word_limit:
#         logger.warning("Prompt too large, truncating...")
#         user_prompt = user_prompt[:8000]
#         system_prompt = system_prompt[:2000]

#     # Format messages for chat API
#     messages = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     messages.append({"role": "user", "content": user_prompt})

#     payload = {
#         "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
#         "messages": messages,
#         "temperature": 0.1,
#         "max_tokens": 1000
#     }

#     GROQ_BASE_URL = "https://api.groq.com/openai/v1"

#     max_retries = 3
#     for attempt in range(max_retries):
#         try:
#             response = requests.post(
#                 f"{GROQ_BASE_URL}/chat/completions",
#                 headers=headers,
#                 json=payload,
#                 timeout=30
#             )

#             if response.status_code == 200:
#                 raw_text = response.json()["choices"][0]["message"]["content"].strip()

#                 # ✅ Strip SQL code blocks and inline code from the recommendation
#                 cleaned = re.sub(r"```sql.*?```", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
#                 cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
#                 cleaned = re.sub(r"`[^`]+`", "", cleaned)  # Optional: inline `code` like `column_name`
                
#                 return cleaned.strip()

#                 # return response.json()["choices"][0]["message"]["content"].strip()

#             elif response.status_code == 429:
#                 try:
#                     error_data = response.json()
#                     message = error_data.get("error", {}).get("message", "")
#                     logger.warning(f"Rate limit hit: {message}")
#                     match = re.search(r'try again in ([\\d\\.]+)s', message)
#                     if match:
#                         wait_time = float(match.group(1))
#                         logger.info(f"Sleeping for {wait_time} seconds due to rate limit...")
#                         time.sleep(wait_time)
#                     else:
#                         time.sleep(2 ** attempt)
#                     continue
#                 except Exception as parse_error:
#                     logger.error(f"Error parsing rate limit retry time: {parse_error}")
#                     time.sleep(2 ** attempt)
#                     continue

#             else:
#                 logger.error(f"Groq Cloud API error: {response.status_code} - {response.text}")
#                 break

#         except Exception as e:
#             logger.error(f"Groq Cloud API call failed: {e}")
#             time.sleep(2 ** attempt)

#     return "I apologize, but I'm having trouble processing your request right now."



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



# def llm_generate_recommendation(question: str, rows: List[Dict[str, Any]]) -> str:
#     preview_rows = rows[:10] if rows else []
#     if isinstance(preview_rows, list) and preview_rows and isinstance(preview_rows[0], dict):
#         table_str = "\n".join(", ".join(f"{k}: {v}" for k, v in row.items()) for row in preview_rows)
#     else:
#         table_str = "No rows available."

#     system_prompt = (
#         "You are a data-driven business analyst assistant. Based on the user's question and the query output data, "
#         "generate a concise, actionable, and professional business recommendation. "
#         "Focus on explaining the key trends, risks, or opportunities in simple business terms, "
#         "helping the user understand what actions or decisions they can take next. "
#         "Strictly avoid including any SQL code, technical jargon, or step-by-step query explanations. "
#         "Only provide a high-level insight that would help a manager or decision-maker."
#         "You are a business analyst who writes like a helpful colleague. "
#         "Based on the user's question and the query output data, "
#         "write a concise, actionable recommendation in plain business language. "
#         "Be concrete and tie suggestions to the data shown. "
#         # "Do not include SQL, technical jargon, or step-by-step query explanations. "
#         # "One short paragraph only. No emojis."
#     )

#     user_prompt = f"""User Question: {question}

# Data Preview (first 10 rows):
# {table_str}

# Please provide only a one-paragraph business recommendation. Do not include SQL queries or explanations.
# """

#     # Size guard (rough)
#     full_prompt_words = len((system_prompt + user_prompt).split())
#     if full_prompt_words > 5000:
#         logger.warning("Prompt too large, truncating...")
#         # Hard truncation safeguards
#         user_prompt = user_prompt[:8000]
#         system_prompt = system_prompt[:2000]

#     messages = [
#         SystemMessage(content=system_prompt),
#         UserMessage(content=user_prompt),
#     ]

#     max_retries = 3
#     base_delay = 1.0

#     for attempt in range(max_retries):
#         try:
#             resp = _get_client().complete(
#                 messages=messages,
#                 model=AZURE_MODEL,
#                 temperature=0.1,
#                 top_p=0.9,
#                 max_tokens=1000,
#                 presence_penalty=0.0,
#                 frequency_penalty=0.0,
#             )

#             if not resp.choices:
#                 logger.warning("Empty choices from Azure Inference.")
#                 raise RuntimeError("Empty response")

#             raw_text = (resp.choices[0].message.content or "").strip()
#             cleaned = _clean_model_text(raw_text)
#             return cleaned or "No clear recommendation could be generated from the data provided."

#         except HttpResponseError as e:
#             status = getattr(e, "status_code", None)
#             logger.warning(f"Azure HttpResponseError (status={status}): {e}")
#             if status in (429, 502, 503, 504) and attempt < max_retries - 1:
#                 time.sleep(base_delay * (2 ** attempt))
#                 continue
#             break
#         except (ServiceRequestError, ServiceResponseError, TimeoutError, ConnectionError) as e:
#             logger.warning(f"Transient Azure error: {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(base_delay * (2 ** attempt))
#                 continue
#             break
#         except Exception as e:
#             logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(base_delay * (2 ** attempt))
#                 continue
#             break

#     return "I apologize, but I'm having trouble processing your request right now."

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
# 🔹 Store Interaction in Chroma (Global + Session) bfr vertical and channel error
# =====================================
def store_interaction_in_chroma229(question, answer, sql, summary, recommendation, session_id, feedback="auto"):
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
    """Retrieve most relevant SQL context, with focus-term filtering, reranking, and similarity scoring."""
    try:
        results = vector_db.similarity_search_with_score(question, k=top_k)

        # ✅ Extract focus terms from current question
        focus_terms_list = [
            "vertical", "channel", "month", "quarter", "year",
            "trend", "least", "highest", "average", "minimum",
            "maximum", "count", "volume", "distribution"
        ]
        q_norm = normalize_question(question)
        q_focus = [t for t in focus_terms_list if t in q_norm.lower()]

        debug_matches = []
        for doc, score in results:
            debug_matches.append({
                "question": doc.page_content,
                "sql": doc.metadata.get("sql"),
                "feedback": doc.metadata.get("feedback"),
                "score": float(score),
                "session_id": doc.metadata.get("session_id"),
                "doc_id": doc.metadata.get("doc_id"),
                "focus_terms": doc.metadata.get("focus_terms", []),
                "timestamp": doc.metadata.get("timestamp", "")
            })

        # ✅ Step 1: Exact focus overlap + rerank
        overlap_matches = [
            (doc, score) for doc, score in results
            if set(q_focus) & set(doc.metadata.get("focus_terms", []))
        ]
        if overlap_matches:
            # Sort by (score DESC, timestamp DESC)
            overlap_matches.sort(
                key=lambda d: (d[1], d[0].metadata.get("timestamp", "")),
                reverse=True
            )
            best_doc, best_score = overlap_matches[0]
            print(f"♻️ Focus-aware reuse (score={best_score:.2f}, focus={q_focus}) from {best_doc.metadata.get('doc_id')}")
            return {
                "sql": best_doc.metadata.get("sql", ""),
                "matched_question": best_doc.page_content,
                "score": float(best_score),
                "top_k_debug": debug_matches
            }

        # ✅ Step 2: Positive matches (if no focus overlap) → also rerank
        positives = [
            (doc, score) for doc, score in results
            if doc.metadata.get("feedback") == "positive" and score >= threshold
        ]
        if positives:
            positives.sort(
                key=lambda d: (d[1], d[0].metadata.get("timestamp", "")),
                reverse=True
            )
            best_doc, best_score = positives[0]
            print(f"♻️ Reusing SQL (positive, score={best_score:.2f}) from {best_doc.metadata.get('doc_id')}")
            return {
                "sql": best_doc.metadata.get("sql", ""),
                "matched_question": best_doc.page_content,
                "score": float(best_score),
                "top_k_debug": debug_matches
            }

        # ✅ Step 3: Auto matches (lower bar) → enrichment
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


# === Retrieval Function ===without second rerank 
def retrieve_contextwtorerank(question, top_k=5, threshold=0.8):
    """Retrieve most relevant SQL context, with focus-term filtering + similarity score."""
    try:
        results = vector_db.similarity_search_with_score(question, k=top_k)

        # ✅ Extract focus terms from current question
        focus_terms_list = [
            "vertical", "channel", "month", "quarter", "year",
            "trend", "least", "highest", "average", "minimum",
            "maximum", "count", "volume", "distribution"
        ]
        q_norm = normalize_question(question)
        q_focus = [t for t in focus_terms_list if t in q_norm.lower()]

        debug_matches = []
        for doc, score in results:
            debug_matches.append({
                "question": doc.page_content,
                "sql": doc.metadata.get("sql"),
                "feedback": doc.metadata.get("feedback"),
                "score": float(score),
                "session_id": doc.metadata.get("session_id"),
                "doc_id": doc.metadata.get("doc_id"),
                "focus_terms": doc.metadata.get("focus_terms", [])
            })

        # ✅ Step 1: Exact focus overlap (best-case match)
        overlap_matches = [
            (doc, score) for doc, score in results
            if set(q_focus) & set(doc.metadata.get("focus_terms", []))
        ]
        if overlap_matches:
            # Pick best by score
            best_doc, best_score = max(overlap_matches, key=lambda d: d[1])
            print(f"♻️ Focus-aware reuse (score={best_score:.2f}, focus={q_focus}) from {best_doc.metadata.get('doc_id')}")
            return {
                "sql": best_doc.metadata.get("sql", ""),
                "matched_question": best_doc.page_content,
                "score": float(best_score),
                "top_k_debug": debug_matches
            }

        # ✅ Step 2: Positive matches (if no focus overlap)
        positives = [
            (doc, score) for doc, score in results
            if doc.metadata.get("feedback") == "positive" and score >= threshold
        ]
        if positives:
            positives.sort(key=lambda d: d[0].metadata.get("timestamp", ""), reverse=True)
            best_doc, best_score = positives[0]
            print(f"♻️ Reusing SQL (positive, score={best_score:.2f}) from {best_doc.metadata.get('doc_id')}")
            return {
                "sql": best_doc.metadata.get("sql", ""),
                "matched_question": best_doc.page_content,
                "score": float(best_score),
                "top_k_debug": debug_matches
            }

        # ✅ Step 3: Auto matches (lower bar)
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



# === Retrieval Function === brf channel and vertical
def retrieve_context229(question, top_k=5, threshold=0.8):
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

# =====================================
# 🔹 Store Interaction in Chroma (with sanitization + safe focus_terms)
# =====================================
def store_interaction_in_chroma(
    question, answer, sql, summary, recommendation, session_id, feedback="auto"
):
    """Store/update Q/A/SQL into Chroma with deduplication across ALL sessions (global memory)."""
    try:
        # ✅ Normalize question
        q_norm = normalize_question(question)

        # ✅ Sanitize SQL before saving
        sql = sanitize_sql_for_nulls(sql)

        # ✅ Extract focus terms dynamically
        focus_terms_list = [
            "vertical",
            "channel",
            "month",
            "quarter",
            "year",
            "trend",
            "least",
            "highest",
            "average",
            "minimum",
            "maximum",
            "count",
            "volume",
            "distribution",
        ]
        q_focus = [t for t in focus_terms_list if t in q_norm.lower()]

        # ✅ Flatten list → string (Chroma requires primitive values only)
        safe_focus = ",".join(q_focus) if q_focus else None

        # 🔎 Search globally for same normalized question
        results = vector_db.similarity_search_with_score(q_norm, k=5)

        existing_doc_id = None
        for doc, score in results:
            if normalize_question(doc.page_content) == q_norm:
                existing_doc_id = doc.metadata.get("doc_id")
                break

        if existing_doc_id:
            print(
                f"♻️ [GLOBAL] Updating existing doc_id={existing_doc_id} "
                f"for question='{question}' with feedback={feedback}"
            )
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
                    "focus_terms": safe_focus,  # ✅ string not list
                    "timestamp": now().isoformat(),
                }],
            )
        else:
            doc_id = f"global:{q_norm}"
            print(
                f"💾 [GLOBAL] Inserting new doc_id={doc_id} "
                f"for question='{question}' (feedback={feedback})"
            )
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
                    "focus_terms": safe_focus,  # ✅ string not list
                    "timestamp": now().isoformat(),
                }],
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
                "focus_terms": safe_focus,  # ✅ string not list
                "timestamp": now().isoformat(),
            }],
        )
        print(
            f"💾 [SESSION] Stored doc_id={session_doc_id} "
            f"with focus={safe_focus} for session={session_id}"
        )

    except Exception as e:
        print(f"❌ [GLOBAL] Failed to store interaction in Chroma: {e}")


# =====================================
# 🔹 Store Interaction in Chroma (Global + Session) with Focus Tagging
# =====================================
def store_interaction_in_chromabfrnull(question, answer, sql, summary, recommendation, session_id, feedback="auto"):
    """Store/update Q/A/SQL into Chroma with deduplication across ALL sessions (global memory)."""
    try:
        # ✅ Normalize question
        q_norm = normalize_question(question)

        # ✅ Extract focus terms dynamically
        focus_terms_list = [
            "vertical", "channel", "month", "quarter", "year",
            "trend", "least", "highest", "average", "minimum",
            "maximum", "count", "volume", "distribution"
        ]
        q_focus = [t for t in focus_terms_list if t in q_norm.lower()]

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
                    "focus_terms": q_focus,  # ✅ new tag
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
                    "focus_terms": q_focus,  # ✅ new tag
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
                "focus_terms": q_focus,  # ✅ new tag
                "timestamp": now().isoformat()
            }]
        )
        print(f"💾 [SESSION] Stored doc_id={session_doc_id} with focus={q_focus} for session={session_id}")

    except Exception as e:
        print(f"❌ [GLOBAL] Failed to store interaction in Chroma: {e}")

# === Streaming Ask Function ===



# =====================================
# 🔹 DataFrame Cleaner (Post-fetch safeguard)
# =====================================
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:  # only string-like
            df = df[df[col].notna()]
            df = df[df[col].astype(str).str.strip() != ""]
            df = df[~df[col].astype(str).str.lower().eq("nan")]
    return df

import re

# =====================================
# 🔹 SQL Sanitizer (Dynamic for NULL/Empty/NAN) with Logging
# =====================================
# =====================================
# 🔹 SQL Sanitizer (Fix for ORDER/LIMIT leakage)
# =====================================
# def sanitize_sql_for_nulls(sql: str) -> str:
#     """
#     Post-process generated SQL to ignore NULL, empty, or 'nan' values dynamically.
#     Extracts clean column names and appends IS NOT NULL filters safely.
#     """
#     if not sql or not sql.strip().lower().startswith(("select", "with")):
#         return sql

#     # Extract GROUP BY columns cleanly
#     group_cols = []
#     m = re.search(r"GROUP BY\s+([\w\s,._-]+)", sql, flags=re.IGNORECASE)
#     if m:
#         group_cols = [c.strip() for c in m.group(1).split(",")]

#     # Fallback → SELECT clause (exclude functions, numbers, keywords)
#     if not group_cols:
#         m = re.search(r"SELECT\s+(.*?)\s+FROM", sql, flags=re.IGNORECASE | re.DOTALL)
#         if m:
#             raw = m.group(1)
#             candidates = [c.strip().split(" ")[0] for c in raw.split(",")]
#             group_cols = [
#                 c for c in candidates
#                 if "." in c or c.isidentifier()
#             ]

#     # ✅ Strip trailing ORDER/LIMIT etc.
#     clean_cols = []
#     for col in group_cols:
#         col = re.split(r"\b(ORDER|LIMIT|ASC|DESC)\b", col, 1, flags=re.IGNORECASE)[0]
#         col = col.strip().strip(",")
#         if col and not col.lower().startswith(("avg", "sum", "count", "extract")):
#             clean_cols.append(col)

#     if not clean_cols:
#         return sql  # nothing to sanitize

#     # Build filter clause
#     conditions = [
#         f"{col} IS NOT NULL AND {col} <> '' AND LOWER({col}) <> 'nan'"
#         for col in clean_cols
#     ]
#     filter_snippet = " AND ".join(conditions)

#     print(f"🧹 [SQL SANITIZER] Applied NULL/empty filter on columns: {clean_cols}")

#     # Case 1: already has WHERE → append
#     if re.search(r"\bWHERE\b", sql, re.IGNORECASE):
#         sql = re.sub(
#             r"\bWHERE\b",
#             f"WHERE {filter_snippet} AND (",
#             sql,
#             count=1,
#             flags=re.IGNORECASE,
#         )
#         if "GROUP BY" in sql.upper():
#             sql = sql.replace("GROUP BY", ") GROUP BY")
#         elif "ORDER BY" in sql.upper():
#             sql = sql.replace("ORDER BY", ") ORDER BY")
#         else:
#             sql = sql + ")"

#     # Case 2: no WHERE → inject before GROUP BY / ORDER BY
#     else:
#         if "GROUP BY" in sql.upper():
#             sql = sql.replace("GROUP BY", f"WHERE {filter_snippet} GROUP BY")
#         elif "ORDER BY" in sql.upper():
#             sql = sql.replace("ORDER BY", f"WHERE {filter_snippet} ORDER BY")
#         else:
#             sql = sql + f" WHERE {filter_snippet}"

#     return sql
import re
import re
import sqlparse
import re
import sqlparse

def sanitize_sql_for_nulls239(sql: str) -> str:
    """
    Dynamically sanitize generated SQL:
    - Add IS NOT NULL filters for GROUP BY / SELECT columns.
    - Skip CASE, functions, aggregates, aliases.
    - Auto-cast SUM/AVG/MIN/MAX args to NUMERIC safely.
    - Auto-cast CASE expressions mixing text + numbers.
    - Works across SELECT / WITH / subqueries.
    """
    if not sql or not sql.strip().lower().startswith(("select", "with")):
        return sql

    # --- 1) Collect aliases from SELECT clause ---
    aliases = set()
    m_sel = re.search(r"SELECT\s+(.*?)\s+FROM", sql, flags=re.IGNORECASE | re.DOTALL)
    if m_sel:
        raw = m_sel.group(1)
        for part in raw.split(","):
            part = part.strip()
            alias_match = re.search(r"\s+AS\s+(\w+)$", part, flags=re.IGNORECASE)
            if alias_match:
                aliases.add(alias_match.group(1).lower())
            else:
                tokens = part.split()
                if len(tokens) > 1 and tokens[-1].isidentifier():
                    aliases.add(tokens[-1].lower())

    # --- 2) Extract GROUP BY columns ---
    group_cols = []
    m_grp = re.search(r"GROUP BY\s+([\w\s,._-]+)", sql, flags=re.IGNORECASE)
    if m_grp:
        group_cols = [c.strip() for c in m_grp.group(1).split(",")]

    # --- 2b) Fallback → try SELECT columns ---
    if not group_cols and m_sel:
        raw = m_sel.group(1)
        candidates = [c.strip().split()[0] for c in raw.split(",")]
        group_cols = [c for c in candidates if c]

    # --- 3) Clean candidate columns ---
    clean_cols = []
    for col in group_cols:
        col = col.strip().strip(",")
        if (
            not col
            or "(" in col or "case" in col.lower()
            or col.lower() in aliases
        ):
            continue
        clean_cols.append(col)

    # --- 4) Build filter clause ---
    if clean_cols:
        conditions = []
        for col in clean_cols:
            col_lower = col.lower()
            if any(x in col_lower for x in ["score", "amount", "id", "num", "count", "age", "year", "month", "date"]):
                conditions.append(f"{col} IS NOT NULL")
            else:
                conditions.append(f"{col} IS NOT NULL AND {col} <> '' AND LOWER({col}) <> 'nan'")
        filter_snippet = " AND ".join(conditions)

        print(f"🧹 [SQL SANITIZER] Applied NULL/empty filter on columns: {clean_cols}")

        if re.search(r"\bWHERE\b", sql, re.IGNORECASE):
            sql = re.sub(r"\bWHERE\b", f"WHERE {filter_snippet} AND", sql, count=1, flags=re.IGNORECASE)
        else:
            if "GROUP BY" in sql.upper():
                sql = sql.replace("GROUP BY", f"WHERE {filter_snippet} GROUP BY")
            elif "ORDER BY" in sql.upper():
                sql = sql.replace("ORDER BY", f"WHERE {filter_snippet} ORDER BY")
            else:
                sql = sql + f" WHERE {filter_snippet}"

    # --- 5) Auto-cast aggregates (SUM/AVG/MIN/MAX) ---
    def _cast_numeric(match):
        func, col = match.groups()
        col = col.strip()
        # ✅ Safe casting: handle '', 'nan'
        wrapped = f"NULLIF(NULLIF(LOWER({col}), 'nan'), '')"
        return f"{func}(CAST({wrapped} AS NUMERIC))"

    sql = re.sub(
        r"\b(SUM|AVG|MIN|MAX)\s*\(\s*([a-zA-Z0-9_\".]+)\s*\)",
        _cast_numeric,
        sql,
        flags=re.IGNORECASE,
    )

    # --- 6) Fix CASE expressions mixing text + numbers ---
    sql = re.sub(
        r"CASE\s+(.*?)\s+THEN\s+([a-zA-Z0-9_\".]+)\s+ELSE\s+0\s+END",
        r"CASE \1 THEN CAST(NULLIF(NULLIF(LOWER(\2), 'nan'), '') AS NUMERIC) ELSE 0::NUMERIC END",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return sql


import logging
logger = logging.getLogger(__name__)
import re

import re
import sqlparse
import re
import sqlparse

import re
import sqlparse
import re
import sqlparse

import re
import sqlparse
import re
import re
import re
import re
import re
import re

# def sanitize_sql_for_nulls(sql: str) -> str:
#     """
#     BULLETPROOF SQL Sanitizer (string-literal safe)
#     - Skips touching anything inside '...' (regex/text literals).
#     - Handles NULL/NaN/empty values across GROUP BY, SELECT, aggregates.
#     - Smart type casting for text vs numeric columns.
#     - Recursive handling of CTEs and subqueries.
#     - Safe numeric casting with guards for literals/functions.
#     - Validation fallback to original query if malformed.
#     """

#     if not sql or not sql.strip().lower().startswith(("select", "with")):
#         return sql

#     original_sql = sql

#     try:
#         # ----------------------------------------
#         # Step 1. Extract string literals
#         # ----------------------------------------
#         str_map = {}
#         def extract_literals(match):
#             key = f"__STR_LITERAL_{len(str_map)}__"
#             str_map[key] = match.group(0)   # store full '...'
#             return key

#         sql = re.sub(r"'([^']|'')*'", extract_literals, sql)

#         # ----------------------------------------
#         # Step 2. Strip comments + dangling commas
#         # ----------------------------------------
#         sql = re.sub(r'--[^\n]*', '', sql)  # remove inline comments
#         sql = re.sub(                       # remove dangling commas after comments
#             r",\s*(\)|FROM|WHERE|GROUP|ORDER|HAVING|LIMIT)",
#             r" \1", sql, flags=re.IGNORECASE
#         )

#         # ----------------------------------------
#         # Helpers
#         # ----------------------------------------
#         def is_numeric_column(col_name: str) -> bool:
#             col_lower = col_name.lower()
#             numeric_indicators = [
#                 "amount","price","cost","fee","rate","score","value","total","sum","avg","count",
#                 "num","number","quantity","qty","balance","income","salary","wage","revenue",
#                 "profit","loss","interest","percentage","percent","ratio","funded","requested",
#                 "approved","disbursed","outstanding","principal","tenure","term_months",
#                 "_id","id","year","month","day","age","duration","length"
#             ]
#             return any(ind in col_lower for ind in numeric_indicators)

#         def safe_numeric_cast(expr: str, in_where: bool = False) -> str:
#             expr = expr.strip()

#             # 🚫 NEVER cast inside WHERE/HAVING filters
#             if in_where:
#                 return expr

#             # 🚫 Skip if already inside DISTINCT/COUNT
#             if re.search(r"\bCOUNT\s*\(\s*DISTINCT", expr, flags=re.IGNORECASE):
#                 return expr

#             # 🚫 Skip if column is clearly an identifier (id-like)
#             if expr.lower().endswith("_id") or expr.lower() == "application_id":
#                 return expr

#             # 🚫 Skip if expression already contains IN() clause
#             if re.search(r"\bIN\s*\(", expr, flags=re.IGNORECASE):
#                 return expr
            
#              # 🚫 Skip if expression already numeric or a CASE
#             if re.search(r"\bCASE\b|\bWHEN\b|\bEND\b", expr, re.IGNORECASE):
#                 return expr
#             if "::NUMERIC" in expr.upper() or "~" in expr:
#                 return expr

#             # # 🚫 Skip if it's a CASE expression
#             if expr.upper().startswith("CASE") or expr.upper().endswith("END"):
#                 return expr

#             # 🚫 Skip literals, functions, placeholders
#             if (
#                 expr.isdigit() or re.match(r"^\d+(\.\d+)?$", expr) or
#                 expr.upper().startswith(("'", '"')) or
#                 expr.upper().startswith(("COALESCE","ROUND","POWER","MOD")) or
#                 "::NUMERIC" in expr.upper() or
#                 expr.startswith("__STR_LITERAL_") or
#                 expr.startswith("(")
#             ):
#                 return expr

#             # 🚫 Skip NOT IN guards
#             if re.search(r"NOT\s+IN\s*\(", expr, flags=re.IGNORECASE):
#                 return expr

#             # ✅ Default numeric cast
#             return f"""(
#                 CASE 
#                     WHEN {expr} IS NULL THEN NULL
#                     WHEN TRIM({expr}::text) = '' THEN NULL
#                     WHEN UPPER(TRIM({expr}::text)) IN ('NAN','NULL','N/A','NA') THEN NULL
#                     WHEN TRIM({expr}::text) ~ '^-?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$'
#                         THEN TRIM({expr}::text)::NUMERIC
#                     ELSE NULL
#                 END
#             )"""

#         # ----------------------------------------
#         # Recursive processors
#         # ----------------------------------------
#         def process_query_recursively(query_sql: str, level: int = 0) -> str:
#             if query_sql.strip().lower().startswith("with"):
#                 cte_pattern = r"(WITH\s+.*?)(\s+SELECT\s+.*)"
#                 cte_match = re.search(cte_pattern, query_sql, flags=re.IGNORECASE | re.DOTALL)
#                 if cte_match:
#                     cte_part = process_cte_definitions(cte_match.group(1))
#                     main_query = sanitize_single_query(cte_match.group(2), level)
#                     return cte_part + main_query
#             return sanitize_single_query(query_sql, level)

#         def process_cte_definitions(cte_sql: str) -> str:
#             cte_pattern = r"(\w+)\s+AS\s*\((.*?)\)(?=\s*,\s*\w+\s+AS\s*\(|\s*SELECT|\s*$)"
#             def repl(match):
#                 return f"{match.group(1)} AS ({sanitize_single_query(match.group(2).strip(), level=1)})"
#             return re.sub(cte_pattern, repl, cte_sql, flags=re.IGNORECASE | re.DOTALL)

#         def sanitize_single_query(single_sql: str, level: int = 0) -> str:
#             subq_pattern = r"\(\s*(SELECT\s+.*?)\s*\)(?=\s*(?:AS\s+\w+)?\s*(?:WHERE|GROUP|ORDER|HAVING|LIMIT|UNION|\)|,|$))"
#             single_sql = re.sub(
#                 subq_pattern,
#                 lambda m: f"({sanitize_single_query(m.group(1), level+1)})",
#                 single_sql,
#                 flags=re.IGNORECASE | re.DOTALL
#             )
#             return apply_main_sanitization(single_sql, level)

#         # ----------------------------------------
#         # Main sanitization
#         # ----------------------------------------
#         def apply_main_sanitization(sql: str, level: int = 0) -> str:
#             # --- WHERE filters for GROUP BY cols ---
#             def build_where_filters(sql_text: str) -> str:
#                 m_grp = re.search(
#                     r"GROUP\s+BY\s+([\w\s,._()-]+?)(?:\s+(?:HAVING|ORDER|LIMIT|$))",
#                     sql_text, flags=re.IGNORECASE | re.DOTALL
#                 )
#                 if not m_grp:
#                     return sql_text
#                 group_cols = [c.strip() for c in m_grp.group(1).split(",")]
#                 filter_conditions = []
#                 for col in group_cols:
#                     if not col or "(" in col or col.isdigit():
#                         continue
#                     if is_numeric_column(col):
#                         filter_conditions.append(f"{col} IS NOT NULL")
#                     else:
#                         filter_conditions.append(
#                             f"{col} IS NOT NULL AND TRIM({col}::text) != '' "
#                             f"AND UPPER(TRIM({col}::text)) NOT IN ('NAN','NULL','N/A','NA')"
#                         )
#                 if not filter_conditions:
#                     return sql_text
#                 filter_snippet = " AND ".join(filter_conditions)
#                 if re.search(r'\bWHERE\b', sql_text, flags=re.IGNORECASE):
#                     return re.sub(
#                         r'(\bWHERE\b\s+)', f'\\1{filter_snippet} AND ',
#                         sql_text, count=1, flags=re.IGNORECASE
#                     )
#                 return re.sub(
#                     r'(\s+(?:GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b)',
#                     f' WHERE {filter_snippet}\\1', sql_text, 1, flags=re.IGNORECASE
#                 ) if re.search(r'(GROUP|ORDER|HAVING|LIMIT)\b', sql_text, flags=re.IGNORECASE) \
#                   else sql_text.strip() + f" WHERE {filter_snippet}"

#             sql = build_where_filters(sql)

#             # --- Aggregates with safe casting (dynamic) ---
#             sql = re.sub(
#                 r"\b(SUM|AVG|MIN|MAX)\s*\((.*?)\)",
#                 lambda m: f"{m.group(1).upper()}({safe_numeric_cast(m.group(2), in_where=False)})",
#                 sql, flags=re.IGNORECASE | re.DOTALL
#             )
#             sql = re.sub(
#                 r"\b(SUM|AVG|MIN|MAX|COUNT)\s*\(\s*DISTINCT\s+(.*?)\)",
#                 lambda m: f"{m.group(1).upper()}(DISTINCT {safe_numeric_cast(m.group(2), in_where=False)})",
#                 sql, flags=re.IGNORECASE | re.DOTALL
#             )

#             # --- Arithmetic ops (only outside WHERE) ---
#             sql = re.sub(
#                 r"([a-zA-Z0-9_\".]+)\s*\+\s*([a-zA-Z0-9_\".]+)",
#                 lambda m: f"{safe_numeric_cast(m.group(1), False)} + {safe_numeric_cast(m.group(2), False)}",
#                 sql, flags=re.IGNORECASE
#             )

#              # --- Division sanitization (avoid /0) ---
#             sql = re.sub(
#                 r"([a-zA-Z0-9_\".]+)\s*/\s*([a-zA-Z0-9_\".]+)",
#                 lambda m: f"{safe_numeric_cast(m.group(1), False)} / NULLIF({safe_numeric_cast(m.group(2), False)},0)",
#                 sql, flags=re.IGNORECASE
#             )

#             # (similar patch for -, *, /, POWER, MOD etc – always call safe_numeric_cast with in_where=False)

#             # --- Cleanup ---
#             sql = re.sub(r'\s+', ' ', sql).strip()
#             return sql

#         # ----------------------------------------
#         # Run + Validate
#         # ----------------------------------------
#         result = process_query_recursively(sql)

#         # Put string literals back
#         for key, val in str_map.items():
#             result = result.replace(key, val)

#         # Validation
#         if not re.search(r'\bSELECT\b.*\bFROM\b', result, flags=re.IGNORECASE | re.DOTALL):
#             return original_sql
#         if result.count("(") != result.count(")"):
#             return original_sql

#         return result

#     except Exception as e:
#         print(f"⚠️ [SQL SANITIZER] Error: {e}")
#         return original_sql

import re
import re
import re
import re

# def sanitize_sql_for_nulls(sql: str) -> str:
#     """
#     BULLETPROOF SQL Sanitizer (string-literal safe, WHERE/HAVING shielded, CASE shielded)
#     - Skips touching anything inside '...' (regex/text literals).
#     - Skips touching CASE ... END blocks (nested-safe).
#     - Handles NULL/NaN/empty values across GROUP BY, SELECT, aggregates.
#     - Smart type casting for text vs numeric columns.
#     - Recursive handling of CTEs and subqueries.
#     - Safe numeric casting with guards for literals/functions.
#     - WHERE/HAVING sections are shielded (only light filters allowed).
#     - Validation fallback to original query if malformed.
#     """

#     if not sql or not sql.strip().lower().startswith(("select", "with")):
#         return sql

#     original_sql = sql

#     try:
#         # ----------------------------------------
#         # Step 1. Extract string literals
#         # ----------------------------------------
#         str_map = {}
#         def extract_literals(match):
#             key = f"__STR_LITERAL_{len(str_map)}__"
#             str_map[key] = match.group(0)
#             return key
#         sql = re.sub(r"'([^']|'')*'", extract_literals, sql)

#         # ----------------------------------------
#         # Step 2. Strip comments + dangling commas
#         # ----------------------------------------
#         sql = re.sub(r'--[^\n]*', '', sql)  
#         sql = re.sub(
#             r",\s*(\)|FROM|WHERE|GROUP|ORDER|HAVING|LIMIT)",
#             r" \1", sql, flags=re.IGNORECASE
#         )

        # ----------------------------------------
        # Step 2b. Auto-patch "this vs last month" CTEs dynamically
        # ----------------------------------------
        # def patch_this_vs_last_month(sql_text: str) -> str:
        #     cte_matches = re.findall(r"\b(this_month|last_month)_(\w+)\b", sql_text, re.IGNORECASE)
        #     if not cte_matches:
        #         return sql_text

        #     seen = {m[0].lower(): m[1] for m in cte_matches}
        #     alias_suffix = list(seen.values())[0]

        #     if "this_month" in seen and "last_month" in seen:
        #         return sql_text

        #     with_block_match = re.search(
        #         rf"(\b(?:this_month|last_month)_{alias_suffix}\b\s+AS\s*\(.*?\))",
        #         sql_text, flags=re.IGNORECASE | re.DOTALL
        #     )
        #     if not with_block_match:
        #         return sql_text

        #     existing_block = with_block_match.group(1)

        #     def rewrite_dates(block: str, target: str) -> str:
        #         if target == "this_month":
        #             block = re.sub(r"CURRENT_DATE\s*-\s*INTERVAL\s*'1 month'",
        #                            "CURRENT_DATE", block, flags=re.IGNORECASE)
        #         else:  # last_month
        #             block = re.sub(r"CURRENT_DATE(\s*\+?\s*INTERVAL\s*'1 month')?",
        #                            "CURRENT_DATE - INTERVAL '1 month'", block, flags=re.IGNORECASE)
        #         return re.sub(r"\b(this_month|last_month)_"+alias_suffix,
        #                       f"{target}_month_{alias_suffix}", block, flags=re.IGNORECASE)

        #     if "this_month" not in seen:
        #         new_block = rewrite_dates(existing_block, "this_month")
        #         sql_text = sql_text.replace(existing_block, existing_block + ",\n" + new_block)
        #     elif "last_month" not in seen:
        #         new_block = rewrite_dates(existing_block, "last_month")
        #         sql_text = sql_text.replace(existing_block, existing_block + ",\n" + new_block)

        #     return sql_text

        # sql = patch_this_vs_last_month(sql)

        # ----------------------------------------
        # Helpers
        # ----------------------------------------
        # def safe_numeric_cast(expr: str, in_where: bool = False) -> str:
        #     expr = expr.strip()

        #     if in_where:
        #         return expr

        #     skip_keywords = [
        #         "CURRENT_DATE", "CURRENT_TIMESTAMP", "NOW()", "INTERVAL",
        #         "DATE_TRUNC", "EXTRACT", "AGE(", "TO_DATE", "TO_CHAR"
        #     ]
        #     if any(expr.upper().startswith(kw) for kw in skip_keywords):
        #         return expr

        #     if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", expr, flags=re.IGNORECASE):
        #         return expr
        #     if expr.lower() in {"approved_count", "total_count"}:
        #         return expr
        #     if expr.lower().endswith("_id") or expr.lower() == "application_id":
        #         return expr
        #     if re.search(r"\b(IN|NOT IN)\s*\(", expr, flags=re.IGNORECASE):
        #         return expr
        #     if re.search(r"\bCASE\b|\bWHEN\b|\bEND\b", expr, re.IGNORECASE):
        #         return expr
        #     if "::NUMERIC" in expr.upper() or "~" in expr:
        #         return expr
        #     if expr.upper().startswith("CASE") or expr.upper().endswith("END"):
        #         return expr
        #     if (expr.isdigit() or re.match(r"^\d+(\.\d+)?$", expr)
        #         or expr.upper().startswith(("'", '"'))
        #         or expr.upper().startswith(("COALESCE", "ROUND", "POWER", "MOD"))
        #         or expr.startswith("__STR_LITERAL_")
        #         or expr.startswith("(")):
        #         return expr

        #     return f"""(
        #         CASE 
        #             WHEN {expr} IS NULL THEN NULL
        #             WHEN TRIM({expr}::text) = '' THEN NULL
        #             WHEN UPPER(TRIM({expr}::text)) IN ('NAN','NULL','N/A','NA') THEN NULL
        #             WHEN TRIM({expr}::text) ~ '^-?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$'
        #                 THEN TRIM({expr}::text)::NUMERIC
        #             ELSE NULL
        #         END
        #     )"""


        # def extract_column_types(schema_text: str) -> dict:
        #     """
        #     Parse FULL_SCHEMA and return {col_name: col_type}.
        #     Example line: "- application_id: bigint"
        #     """
        #     col_types = {}
        #     for line in schema_text.splitlines():
        #         line = line.strip()
        #         if not line.startswith("- "):
        #             continue
        #         parts = line.replace("-", "").split(":", 1)
        #         if len(parts) == 2:
        #             col = parts[0].strip()
        #             ctype = parts[1].strip().lower()
        #             col_types[col.lower()] = ctype
        #     return col_types


        # def extract_primary_ids_from_schema(schema_text: str):
        #     """
        #     Parse FULL_SCHEMA and return list of primary identifier columns.
        #     """
        #     id_cols = []
        #     for line in schema_text.splitlines():
        #         line = line.strip()
        #         if not line.startswith("- "):
        #             continue
        #         parts = line.replace("-", "").split(":", 1)
        #         if not parts:
        #             continue
        #         col = parts[0].strip()
        #         if col.lower().endswith("_id") or col.lower() == "id":
        #             id_cols.append(col)
        #     return id_cols

        #         # Build dynamically from FULL_SCHEMA
        # COL_TYPES = extract_column_types(FULL_SCHEMA) 
        # PRIMARY_ID_COLS = extract_primary_ids_from_schema(FULL_SCHEMA)
        # print("🔑 Dynamic ID columns:", PRIMARY_ID_COLS)

        #   # ----------------------------------------
        # # Step 1. Dynamic schema validation
        # # ----------------------------------------
        # schema_cols = list(COL_TYPES.keys()) if isinstance(COL_TYPES, dict) else []
        # if not isinstance(schema_cols, list):
        #     logging.error(f"[SQL SANITIZER] schema_cols is not list but {type(schema_cols)}")
        #     schema_cols = []
        # analysis = check_question_vs_schema(question, schema_cols)

        # if analysis["status"] == "error":
        #     logging.warning(f"[SQL SANITIZER] {analysis['message']}")
        #     # 🚫 Return structured JSON error instead of SQL
        #     return json.dumps(analysis)

        # logging.debug(f"[SQL SANITIZER] Required cols = {analysis['required']}")


        # def safe_numeric_cast(expr: str, COL_TYPES: dict, in_where: bool = False) -> str:
        #     expr = expr.strip()
        #     if in_where:
        #         return expr

        #     # 🔑 Schema-aware: if column exists and type is numeric, don’t cast
        #     if expr.lower() in COL_TYPES:
        #         ctype = COL_TYPES[expr.lower()]
        #         if any(t in ctype for t in ["int", "decimal", "numeric", "double", "real", "float"]):
        #             return expr  # already numeric

        #     skip_keywords = [
        #         "CURRENT_DATE", "CURRENT_TIMESTAMP", "NOW()", "INTERVAL",
        #         "DATE_TRUNC", "EXTRACT", "AGE(", "TO_DATE", "TO_CHAR"
        #     ]
        #     if any(expr.upper().startswith(kw) for kw in skip_keywords):
        #         return expr

        #     # Skip aggregates
        #     if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", expr, flags=re.IGNORECASE):
        #         return expr

        #     # Skip IDs
        #     if expr.lower().endswith("_id") or expr.lower() == "application_id":
        #         return expr

        #     # Skip CASE/WHEN
        #     if re.search(r"\bCASE\b|\bWHEN\b|\bEND\b", expr, re.IGNORECASE):
        #         return expr

        #     # Skip literals
        #     if (expr.isdigit() or re.match(r"^\d+(\.\d+)?$", expr)
        #         or expr.upper().startswith(("'", '"'))
        #         or "::NUMERIC" in expr.upper()
        #         or expr.startswith("__STR_LITERAL_")
        #         or expr.startswith("(")):
        #         return expr

        #     # Otherwise → safe cast (for varchar/text columns)
        #     return f"""(
        #         CASE 
        #             WHEN {expr} IS NULL THEN NULL
        #             WHEN TRIM({expr}::text) = '' THEN NULL
        #             WHEN UPPER(TRIM({expr}::text)) IN ('NAN','NULL','N/A','NA') THEN NULL
        #             WHEN TRIM({expr}::text) ~ '^-?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$'
        #                 THEN TRIM({expr}::text)::NUMERIC
        #             ELSE NULL
        #         END
        #     )"""



def sanitize_sql_for_nulls239(sql: str) -> str:
    """
    Legacy function - kept for backward compatibility
    """
    return sanitize_sql_for_nulls(sql)

def sanitize_sql_for_nulls151now(sql: str) -> str:
    """
    Dynamically sanitize generated SQL:
    - Add IS NOT NULL filters for GROUP BY / SELECT columns.
    - Skip CASE, functions, aggregates, aliases.
    - Auto-cast SUM/AVG/MIN/MAX args to NUMERIC safely.
    - Auto-cast CASE expressions mixing text + numbers.
    - Auto-cast comparisons like col > 0, col < 0, col >= 0, col <= 0 when col may be TEXT.
    """
    if not sql or not sql.strip().lower().startswith(("select", "with")):
        return sql

    # --- 1) Collect aliases from SELECT clause ---
    aliases = set()
    m_sel = re.search(r"SELECT\s+(.*?)\s+FROM", sql, flags=re.IGNORECASE | re.DOTALL)
    if m_sel:
        raw = m_sel.group(1)
        for part in raw.split(","):
            part = part.strip()
            alias_match = re.search(r"\s+AS\s+(\w+)$", part, flags=re.IGNORECASE)
            if alias_match:
                aliases.add(alias_match.group(1).lower())
            else:
                tokens = part.split()
                if len(tokens) > 1 and tokens[-1].isidentifier():
                    aliases.add(tokens[-1].lower())

    # --- 2) Extract GROUP BY columns ---
    group_cols = []
    m_grp = re.search(r"GROUP BY\s+([\w\s,._-]+)", sql, flags=re.IGNORECASE)
    if m_grp:
        group_cols = [c.strip() for c in m_grp.group(1).split(",")]

    if not group_cols and m_sel:
        raw = m_sel.group(1)
        candidates = [c.strip().split()[0] for c in raw.split(",")]
        group_cols = [c for c in candidates if c]

    # --- 3) Clean candidate columns ---
        # --- 3) Clean candidate columns ---
    clean_cols = []
    for col in group_cols:
        col = col.strip().strip(",")
        if (
            not col
            or "(" in col or "case" in col.lower()
            or col.lower() in aliases
            or col.isdigit()   # 🚫 NEW: skip pure numbers like "4"
        ):
            continue
        clean_cols.append(col)


    # --- 4) Build filter clause ---
    if clean_cols:
        conditions = []
        for col in clean_cols:
            col_lower = col.lower()
            if any(x in col_lower for x in ["score", "amount", "id", "num", "count", "age", "year", "month", "date"]):
                conditions.append(f"{col} IS NOT NULL")
            else:
                conditions.append(f"{col} IS NOT NULL AND {col} <> '' AND LOWER({col}) <> 'nan'")
        filter_snippet = " AND ".join(conditions)

        print(f"🧹 [SQL SANITIZER] Applied NULL/empty filter on columns: {clean_cols}")

        if re.search(r"\bWHERE\b", sql, re.IGNORECASE):
            sql = re.sub(r"\bWHERE\b", f"WHERE {filter_snippet} AND", sql, count=1, flags=re.IGNORECASE)
        else:
            if "GROUP BY" in sql.upper():
                sql = sql.replace("GROUP BY", f"WHERE {filter_snippet} GROUP BY")
            elif "ORDER BY" in sql.upper():
                sql = sql.replace("ORDER BY", f"WHERE {filter_snippet} ORDER BY")
            else:
                sql = sql + f" WHERE {filter_snippet}"

    # --- 5) Auto-cast aggregates (SUM/AVG/MIN/MAX) ---
    def _cast_numeric(match):
        func, col = match.groups()
        col = col.strip()
        wrapped = f"NULLIF(NULLIF(LOWER({col}), 'nan'), '')"
        return f"{func}(CAST({wrapped} AS NUMERIC))"

    sql = re.sub(
        r"\b(SUM|AVG|MIN|MAX)\s*\(\s*([a-zA-Z0-9_\".]+)\s*\)",
        _cast_numeric,
        sql,
        flags=re.IGNORECASE,
    )

    # --- 6) Fix CASE expressions mixing text + numbers ---
    sql = re.sub(
        r"CASE\s+(.*?)\s+THEN\s+([a-zA-Z0-9_\".]+)\s+ELSE\s+0\s+END",
        r"CASE \1 THEN CAST(NULLIF(NULLIF(LOWER(\2), 'nan'), '') AS NUMERIC) ELSE 0::NUMERIC END",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # --- 7) Fix comparisons like col > 0, col < 0, col >= 0, col <= 0 for TEXT columns ---
    def _cast_comparison(match):
        col = match.group(1).strip()
        op = match.group(2).strip()
        val = match.group(3).strip()
        return f"CAST(NULLIF({col}, '') AS NUMERIC) {op} {val}"

    sql = re.sub(
        r"([a-zA-Z0-9_\".]+)\s*(>=|<=|>|<)\s*([0-9]+)",
        _cast_comparison,
        sql,
        flags=re.IGNORECASE,
    )

    return sql

# ----------------------------------------
# NEW: Schema-aware type map builder
# ----------------------------------------
def build_type_map_from_schemabrfgen(schema_text: str):
    """
    Parse FULL_SCHEMA docstring and infer column types (numeric vs text).
    """
    type_map = {}
    for line in schema_text.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        # Extract column name
        try:
            col = line.split()[1]
        except IndexError:
            continue
        col = col.replace(":", "").strip()

        desc = line.lower()
        if any(t in desc for t in ["text", "string", "char", "purpose", "city", "state", "vertical", "bank", "group"]):
            type_map[col] = "text"
        elif any(t in desc for t in ["int", "numeric", "bigint", "amount", "score", "rate", "term", "income", "id"]):
            type_map[col] = "numeric"
        else:
            type_map[col] = "unknown"
    return type_map


# ----------------------------------------
# NEW: Auto-fix mismatched WHERE clauses
# ----------------------------------------
def auto_fix_schema_mismatchesbfrgen(sql: str, schema_text: str) -> str:
    type_map = build_type_map_from_schema(schema_text)

    # Dynamic mapping for common *_id → *_vertical/_state/_city fallbacks
    fallback_map = {
        "merchant_id": "merchant_vertical",
        "customer_id": "loan_application_state",
        "client_id": "merchant_vertical",
    }

    def repl(match):
        col, val = match.group(1), match.group(2)
        col = col.strip('"')
        if col in type_map:
            ctype = type_map[col]
            # If column is numeric but compared against string literal → rewrite
            if ctype == "numeric" and re.match(r"'[A-Za-z ]+'", val):
                fallback = fallback_map.get(col)
                if fallback and fallback in type_map and type_map[fallback] == "text":
                    return f"{fallback} ILIKE '%' || {val.strip()} || '%'"
                else:
                    return f"-- ERROR: {col} (numeric) compared to {val} (string)"
        return match.group(0)

    # Pattern: col = 'string'
    sql = re.sub(r'(\w+)\s*=\s*(\'[^\']+\')', repl, sql, flags=re.IGNORECASE)
    return sql


# @csrf_exempt
# def ask_question_stream(request):
#     if request.method == "OPTIONS":
#         r = HttpResponse()
#         r["Access-Control-Allow-Origin"] = "*"
#         r["Access-Control-Allow-Methods"] = "POST, OPTIONS"
#         r["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
#         return r

#     if request.method != "POST":
#         return JsonResponse({"error": "Use POST"}, status=405)

#     try:
#         payload = json.loads(request.body or "{}")
#         session_id = payload.get("session_id")
#         question = payload.get("question")
#         user_id = payload.get("user_id", "stream_user")

#         if not session_id or not question:
#             return JsonResponse({"error": "Missing session_id or question"}, status=400)
#         if session_id not in session_store:
#             return JsonResponse({"error": f"Session {session_id} not found"}, status=404)

#         engine = session_store[session_id]["engine"]
#         hist = conversation_memory_store.get(session_id, [])
#         t0 = now()

#         def gen():
#             # ✅ Send a human-like opener immediately
#             try:
#                 preview_narrative = safe_generate_narrative(question, sql="", rows=[])
#                 opener = preview_narrative.get("opener", "")
#                 if not opener or len(opener) < 10:
#                     opener = "Let me analyze your question and gather insights for you."
#                 else:
#                     opener_lower = opener.lower()
#                     bad_phrases = [
#                         "no data", "no records", "nothing found",
#                         "unfortunately", "however", "but when i", "aren't any"
#                     ]
#                     if any(bp in opener_lower for bp in bad_phrases):
#                         opener = "Let me analyze your question and gather insights for you."

#                 yield jsonl_line({"event": "narrative_opener", "text": opener})
#             except Exception as e:
#                 print(f"⚠️ Early opener generation failed: {e}")
#                 opener = "Let me analyze your question and gather insights for you."
#                 yield jsonl_line({"event": "narrative_opener", "text": opener})

#             yield jsonl_line({"event": "phase", "message": "Understanding your question…"})

#             try:
#                 ctx = retrieve_context(question)
#                 score = ctx.get("score", 0.0) or 0.0
#                 matched_q = ctx.get("matched_question", "")

#                 # 🔎 Debug log
#                 print("🔎 [RAG DEBUG] Top-k matches:", json.dumps({
#                     "matched_question": matched_q,
#                     "similarity_score": score,
#                     "top_k_matches": ctx.get("top_k_debug"),
#                 }, indent=2), flush=True)

#                 # Emit debug
#                 yield jsonl_line({
#                     "event": "rag_debug",
#                     "matches": {
#                         "matched_question": matched_q,
#                         "similarity_score": score,
#                         "top_k_matches": ctx.get("top_k_debug"),
#                     }
#                 })

#                 sql = None
#                 # if ctx.get("sql"):
#                 #     q_norm = normalize_question(question)
#                 #     matched_norm = normalize_question(matched_q)
#                 #     print(f"🔎 [RAG DEBUG] Normalized question: {q_norm}, matched question: {matched_norm}")

#                 #     if q_norm == matched_norm:
#                 #         sql = ctx["sql"]
#                 #         print(f"♻️ Reusing SQL (EXACT match) for: {matched_q}")
#                 #     elif score >= RAG_SIMILARITY_THRESHOLD:
#                 #         sql = ctx["sql"]
#                 #         print(f"♻️ Reusing SQL (score={score:.2f}) for: {matched_q}")
#                 #     else:
#                 #         print(f"🆕 [NEW SQL GEN] No strong match (score={score:.2f}, threshold={RAG_SIMILARITY_THRESHOLD}). Generating fresh SQL…")
#                 if ctx.get("sql"):
#                         q_norm = normalize_question(question)
#                         matched_norm = normalize_question(matched_q)
#                         print(f"🔎 [RAG DEBUG] Normalized question: {q_norm}, matched question: {matched_norm}")

#                         candidate_sql = ctx["sql"]

#                         # 🚫 Prevent reusing churn SQL
#                         if "main_cai_lib" in candidate_sql or "policy_no" in candidate_sql:
#                             print("⚠️ Skipping reuse: matched SQL is from churn schema, not loan schema.")
#                             sql = None

#                         elif q_norm == matched_norm:
#                             sql = candidate_sql
#                             print(f"♻️ Reusing SQL (EXACT match) for: {matched_q}")

#                         elif score >= RAG_SIMILARITY_THRESHOLD:
#                             ### 🔑 PATCH START: Intent guard ###
#                             # Define focus keywords for dynamic reuse
#                             focus_terms = [
#                                 "vertical", "channel", "month", "quarter", "year", "trend",
#                                 "least", "highest", "average", "minimum", "maximum", "count", "volume"
#                             ]
#                             q_focus = [t for t in focus_terms if t in q_norm.lower()]
#                             m_focus = [t for t in focus_terms if t in matched_norm.lower()]

#                             if set(q_focus) == set(m_focus):
#                                 sql = candidate_sql
#                                 print(f"♻️ Reusing SQL (score={score:.2f}, focus={q_focus}) for: {matched_q}")
#                             else:
#                                 sql = None
#                                 print(f"🆕 [NEW SQL GEN] Similarity high but focus mismatch "
#                                     f"(Q={q_focus}, M={m_focus}). Forcing new SQL…")
#                             ### 🔑 PATCH END ###

#                         else:
#                             print(f"🆕 [NEW SQL GEN] No strong match (score={score:.2f}, threshold={RAG_SIMILARITY_THRESHOLD}). Generating fresh SQL…")

#                 else:
#                     print("🆕 [NEW SQL GEN] No SQL found in memory, generating fresh SQL…")

#                 # Generate if no reuse
#                 if not sql:
#                     context = ctx.get("sql", "")
#                     augmented_question = f"""
#                     You are a SQL assistant. Use the following context (schemas, past queries, feedback):

#                     {context}

#                     User Question: {question}
#                     """
#                     raw = run_sql_generation_graph(
#                         augmented_question, user_id=user_id, db_id=session_id, history=hist
#                     )
#                     sql_raw = raw[0] if isinstance(raw, tuple) else raw
#                     sql = extract_sql_block(sql_raw or "")

#                 # sql = re.sub(r"\bLIMIT(\d+)", r"LIMIT \1", sql, flags=re.IGNORECASE)
#                 # print(f"📝 Generated SQL:\n{sql}\n", flush=True)
#                 # ✅ Apply sanitizer
#                 sql = re.sub(r"\bLIMIT(\d+)", r"LIMIT \1", sql, flags=re.IGNORECASE)
#                 # ✅ Apply schema-aware fix
#                 sql = sanitize_sql_for_nulls(sql)
#                 sql = auto_fix_schema_mismatches(sql, FULL_SCHEMA)


#                 print(f"📝 Generated SQL:\n{sql}\n", flush=True)


#             except Exception as e:
#                 traceback.print_exc()
#                 yield jsonl_line({"event": "error", "message": f"SQL generation failed: {e}"})
#                 return

#             if not sql or not sql.strip().lower().startswith(("select", "with")):
#                 yield jsonl_line({"event": "error", "message": "Invalid or empty SQL generated"})
#                 return

#             yield jsonl_line({"event": "sql", "sql": sql})
#             print(f"📝 Generated SQL refer:\n{sql}\n", flush=True)

#             # Run SQL
#             yield jsonl_line({"event": "phase", "message": "Querying the database…"})
#             try:
#                 with engine.connect() as conn:
#                     result = conn.execute(text(sql))
#                     rows = [dict(row._mapping) for row in result]

#                      # ✅ Clean at DataFrame level also
#                 df = pd.DataFrame(rows)
#                 df = clean_dataframe(df)
#                 rows = df.to_dict(orient="records")

#                 print(f"✅ SQL executed. Row count: {len(rows)}")
#             except Exception as e:
#                 traceback.print_exc()
#                 yield jsonl_line({"event": "error", "message": f"SQL execution failed: {e}"})
#                 return

#             row_count = len(rows)
#             yield jsonl_line({"event": "rows_preview", "rows": rows[:8], "row_count": row_count})

#             yield jsonl_line({"event": "phase", "message": "Analyzing results…"})
#             try:
#                 # ✅ Human-like narrative instead of static summary
#                 narrative_obj = safe_generate_narrative(question, sql, rows)
#                 if not opener:
#                     opener = narrative_obj.get("opener", opener)
#                 insights = narrative_obj.get("insights", [])
#                 recs = narrative_obj.get("recommendations", [])
#                 next_step = narrative_obj.get("next_step", "")

#                 # Emit each component separately if frontend supports it
#                 if insights:
#                     yield jsonl_line({"event": "insights", "list": insights})
#                 if recs:
#                     yield jsonl_line({"event": "recommendations", "list": recs})
#                 if next_step:
#                     yield jsonl_line({"event": "next_step", "text": next_step})

#                 # Also keep a flat summary (for backward compatibility)
#                 summary = " ".join(insights) if insights else opener

#             except Exception as e:
#                 print(f"⚠️ Narrative generation failed: {e}")
#                 summary = ""

#             try:
#                 narr = llm_generate_narrative(question, rows)
#                 yield jsonl_line({"event": "narrative", "obj": narr})
#             except Exception:
#                 narr = None

#             try:
#                 rec = llm_generate_recommendation(question, rows)
#                 if rec:
#                     yield jsonl_line({"event": "recommendation", "text": rec})
#             except Exception:
#                 rec = None

#             try:
#                 cfg = llm_generate_chart_config(question, rows)
#                 if cfg:
#                     yield jsonl_line({"event": "chart", "config": cfg})
#             except Exception:
#                 cfg = None

#             # Format final answer
#             if row_count == 0:
#                 answer = "No data found."
#             elif row_count > 50:
#                 answer = f"Found {row_count} results. Too many to display here — use the CSV download."
#             else:
#                 first = [", ".join(str(v) for v in r.values()) for r in rows[:3]]
#                 answer = "\n".join(first)
#                 if row_count > 3:
#                     answer += f"\n...and {row_count - 3} more rows."

#             # Update session + store
#             conversation_memory_store.setdefault(session_id, []).append({
#                 "question": question, "sql": sql, "row_count": row_count, "timestamp": now().isoformat()
#             })
#             session_store[session_id]["user_id"] = user_id

#             store_interaction_in_chroma(question=question,
#                 answer=answer,
#                 sql=sql,
#                 summary=summary,
#                 recommendation=rec,
#                 session_id=session_id,
#                 feedback="auto")

#             total = (now() - t0).total_seconds()
#             yield jsonl_line({
#                 "event": "final",
#                 "payload": {
#                     "answer": answer,
#                     "summary": summary,
#                     "rows": rows[:50],
#                     "row_count": row_count,
#                     "chart_config": cfg,
#                     "recommendation": rec,
#                     "narrative": narr,
#                     "query_used": sql,
#                     "response_time": f"{total:.2f}s",
#                     "session_id": session_id,
#                     "rag_debug": {
#                         "matched_question": ctx.get("matched_question"),
#                         "similarity_score": score,
#                         "top_k_matches": ctx.get("top_k_debug")
#                     },
#                     "history": conversation_memory_store.get(session_id, []),
#                     "conversational_opener": opener  # ✅ store human opener
#                 },
#             })

#         resp = StreamingHttpResponse(gen(), content_type="application/x-ndjson")
#         resp["Cache-Control"] = "no-cache"
#         resp["X-Accel-Buffering"] = "no"
#         resp["Access-Control-Allow-Origin"] = "*"
#         return resp

#     except Exception as e:
#         traceback.print_exc()
#         return JsonResponse({"error": str(e)}, status=500)
import re
import time
import requests
from loguru import logger
import logging
logger = logging.getLogger(__name__)
from .schema_utils import pick_target_table_from_question,sanitize_sql_for_nulls



# Groq working code final6-10
# def call_llm_with_retry(prompt: str, max_retries: int = 3, base_delay: float = 1.0) -> str:
#     if not GROQ_API_KEY:
#         logger.error("GROQ_API_KEY is not defined")
#         return "API key configuration error. Please check your settings."

#     estimated_tokens = len(prompt.split()) * 1.5
#     logger.info(f"Estimated prompt tokens: {estimated_tokens}")

#     if estimated_tokens > 220000:
#         logger.error(f"Prompt too long: {estimated_tokens} tokens (max: 220000)")
#         return "The prompt is too long for the current model. Please try with a shorter question."

#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json",
#     }

#     model = "meta-llama/llama-4-maverick-17b-128e-instruct"
#     # model = "meta-llama/llama-4-maverick-17b-128e-instruct"

#     logger.info(f"Starting Groq LLM call with {max_retries} max retries")

#     for attempt in range(max_retries):
#         try:
#             logger.info(f"Attempt {attempt + 1}/{max_retries}")
#             payload = {
#                 "model": model,
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": 0.1,
#                 "top_p": 0.9,
#                 "frequency_penalty": 0.0,
#                 "presence_penalty": 0.0,
#                 "max_tokens": 4000
#             }
#             response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=60)
#             logger.info(f"Response status code: {response.status_code}")

#             if response.status_code == 200:
#                 result = response.json()
#                 if 'choices' in result and result['choices']:
#                     answer = result['choices'][0].get('message', {}).get('content', '').strip()
#                     if answer:
#                         return answer
#             elif response.status_code in [429, 502, 503, 504]:
#                 time.sleep(base_delay * (2 ** attempt))
#                 continue

#         except Exception as e:
#             logger.warning(f"Retry {attempt+1} failed: {e}")
#             time.sleep(base_delay)

#     logger.error("All retry attempts failed")
#     return "I'm currently experiencing issues reaching the AI service. Please try again later."




def call_llm_with_retry(prompt: str, max_retries: int = 3, base_delay: float = 1.0) -> str:
    # Estimate prompt tokens (rough)
    estimated_tokens = int(len(prompt.split()) * 1.5)
    logger.info(f"Estimated prompt tokens: {estimated_tokens}")
    MAX_PROMPT_TOKENS = 120_000  # adjust if your deployed model supports a different context length
    if estimated_tokens > MAX_PROMPT_TOKENS:
        logger.error(f"Prompt too long: {estimated_tokens} tokens (max: {MAX_PROMPT_TOKENS})")
        return "The prompt is too long for the current model. Please try with a shorter question."

    logger.info(f"Starting Azure Inference call with {max_retries} max retries")

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}")
            resp = _get_client().complete(
                messages=[UserMessage(content=prompt)],
                model=AZURE_MODEL,
                temperature=0.1,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_tokens=4000,
            )
            if resp.choices:
                content = getattr(resp.choices[0].message, "content", "") or ""
                content = content.strip()
                if content:
                    return content
            logger.warning("Empty response from Azure Inference; retrying if attempts remain.")

        except HttpResponseError as e:
            status = getattr(e, "status_code", None)
            logger.warning(f"Azure HttpResponseError (status={status}): {e}")
            if status in (429, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            break  # non-retryable or retries exhausted

        except (ServiceRequestError, ServiceResponseError, TimeoutError, ConnectionError) as e:
            logger.warning(f"Transient Azure error: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            break

        except Exception as e:
            logger.warning(f"Retry {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            break

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

import re
import time
import requests
from loguru import logger 
import math
# top of views.py (once)
import re

def is_metric_column(col_name: str, value) -> bool:
    """
    Detect dynamically if a column should preserve decimals (rate, percentage, ratio, APR, etc.).
    Works based on:
      - Column name heuristics
      - Value range & type heuristics
    """
    lname = str(col_name).lower()

    # 1. Column name contains metric keywords
    metric_keywords = ["rate", "percent", "percentage", "ratio", "apr", "margin", "yield", "growth"]
    if any(k in lname for k in metric_keywords):
        return True

    # 2. Value heuristics (numeric)
    if isinstance(value, (float, int)):
        # Exclude exact integers
        if isinstance(value, float) and not value.is_integer():
            # If between 0–1 → ratio
            if 0 < value < 1:
                return True
            # If between 0–100 with decimals → likely percentage
            if 0 < value < 100:
                return True

    return False


def format_result_rows(rows: list[dict]) -> list[dict]:
    """
    Format SQL result rows dynamically:
      - Keep precision for metrics (rates, percentages, ratios, etc.)
      - Round non-metrics safely
      - Leave integers untouched
    """
    formatted = []
    for row in rows:
        new_row = {}
        for col, val in row.items():
            if isinstance(val, float):
                if is_metric_column(col, val):
                    # Metrics: keep 4 decimal places max
                    new_row[col] = float(f"{val:.4f}")
                else:
                    # Non-metrics: round to 2 decimals (money, amounts)
                    new_row[col] = float(f"{val:.2f}")
            elif isinstance(val, (int,)):
                new_row[col] = val  # keep integers as-is
            else:
                new_row[col] = val
        formatted.append(new_row)
    return formatted



# ---------- ADD: Dynamic SQL post-linter & identifier healer ----------

import re
from difflib import SequenceMatcher

# ---------- ADD: LLM SQL pre-parser & scrubber ----------
# ===================== NEW HELPERS (place near your other helpers) =====================
import re

_SQL_CODE_FENCE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_MAYBE_FQ_RE = re.compile(
    r'"?([A-Za-z_][\w$]*)"?\s*\.\s*"?([A-Za-z_][\w$]*)"?',
    re.IGNORECASE,
)

def _normalize_identifier(fq: str) -> str:
    return fq.replace('"','').replace(' ', '').lower()

def _requote_fq(norm_fq: str) -> str:
    # norm_fq like: schema.table
    parts = norm_fq.split(".")
    if len(parts) != 2:
        return norm_fq
    return f'"{parts[0]}"."{parts[1]}"'
# --- dynamic repair & guard helpers (schema-agnostic) ---
import re
from difflib import get_close_matches

def collapse_stacked_quotes(sql: str) -> str:
    sql = re.sub(r'""+', '"', sql)
    sql = re.sub(r'"\s*\.\s*""', '"."', sql)
    sql = re.sub(r'""\s*\.\s*"', '"."', sql)
    sql = re.sub(r'("{2,})', '"', sql)
    sql = re.sub(r'("([A-Za-z_]\w*)")"', r'\1', sql)
    return sql

def fix_common_keyword_typos(sql: str) -> str:
    repls = {
        r'\bFROM\s+FROM\b': 'FROM',
        r'\bGROUP\s+BY\s+BY\b': 'GROUP BY',
        r'\bORDER\s+BY\s+BY\b': 'ORDER BY',
        r'\bEXTRACTRACT\b': 'EXTRACT',
        r'\bYYEAR\b': 'YEAR',
        r'\bMMONTH\b': 'MONTH',
        r',\s*(GROUP BY|ORDER BY|WHERE|HAVING)\b': r' \1',
    }
    for pat, rep in repls.items():
        sql = re.sub(pat, rep, sql, flags=re.IGNORECASE)
    return sql
# --- LLM SQL pre-parse: robust to tuple/list/dict/backticks/header tags ---
import re
from typing import Any

def _coerce_to_text(raw: Any) -> str:
    """
    Accept tuple/list/dict/str and return the most likely SQL-bearing text.
    - tuple/list: pick first non-empty str
    - dict: look in common keys
    - str: return as-is
    """
    if raw is None:
        return ""

    if isinstance(raw, str):
        return raw

    if isinstance(raw, (tuple, list)):
        for item in raw:
            if isinstance(item, str) and item.strip():
                return item
        # last resort
        return " ".join(str(x) for x in raw if x is not None)

    if isinstance(raw, dict):
        for k in ("sql", "text", "content", "message", "output"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return str(raw)

    # anything else
    return str(raw)

def _strip_llm_wrappers(txt: str) -> str:
    """
    Remove common wrappers the model adds:
    - fenced ```sql ... ```
    - unfenced ```
    - <|header_start|>sql, <|header_end|>, etc.
    - accidental 'sql\n' prefixes
    """
    if not txt:
        return ""

    # header tokens
    txt = re.sub(r'<\|header_start\|>\s*sql', '', txt, flags=re.IGNORECASE)
    txt = re.sub(r'<\|header_end\|>', '', txt, flags=re.IGNORECASE)

    # fenced block with language
    m = re.search(r"```sql\s*(.*?)```", txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # any fenced block
    m = re.search(r"```\s*(.*?)```", txt, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    # sometimes models prefix with "sql\n"
    txt = re.sub(r'^\s*sql\s+', '', txt, flags=re.IGNORECASE)

    return txt.strip()

def _first_sql_statement(txt: str) -> str:
    """
    Keep from the first WITH/SELECT onward; drop commentary before.
    """
    m = re.search(r"\b(WITH|SELECT)\b.*", txt, flags=re.IGNORECASE | re.DOTALL)
    return m.group(0).strip() if m else txt

def _quick_typos_and_quotes(sql: str) -> str:
    """
    Quick fix for common tokenization glitches (seen in your logs).
    """
    if not sql:
        return sql

    # collapse stacked/dangled quotes & dotted identifiers
    sql = re.sub(r'""+', '"', sql)
    sql = re.sub(r'"\s*\.\s*""', '"."', sql)
    sql = re.sub(r'""\s*\.\s*"', '"."', sql)
    sql = re.sub(r'\.\s*\.', '.', sql)

    # fix duplicate keywords
    fixes = {
        r'\bFROM\s+FROM\b': 'FROM',
        r'\bGROUP\s+BY\s+BY\b': 'GROUP BY',
        r'\bORDER\s+BY\s+BY\b': 'ORDER BY',
        r'\bEXTRACTRACT\b': 'EXTRACT',
        r'\bYYEAR\b': 'YEAR',
        r'\bMMONTH\b': 'MONTH',
        r'\bIS\s+IS\b': 'IS',
        r',\s*(GROUP BY|ORDER BY|WHERE|HAVING)\b': r' \1',
    }
    for pat, rep in fixes.items():
        sql = re.sub(pat, rep, sql, flags=re.IGNORECASE)

    # normalize whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    return sql

def preparse_llm_sql(raw: Any) -> str:
    """
    Robust pre-parser:
    1) coerce raw to text (handles tuple/list/dict/str)
    2) strip model wrappers
    3) keep first SQL statement
    4) quick typo/quote cleanup
    """
    txt = _coerce_to_text(raw)
    if not txt:
        return ""
    txt = _strip_llm_wrappers(txt)
    if not txt:
        return ""
    sql = _first_sql_statement(txt)
    sql = _quick_typos_and_quotes(sql)
    return sql


_QUOTED_FQ = re.compile(r'"([A-Za-z_]\w*)"\s*\.\s*"([A-Za-z_]\w*)"')
_BARE_FQ   = re.compile(r'\b([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\b')

def build_catalog_index(schema_catalog: dict):
    fq_to_canonical, schemas, tables_by_schema, cols_by_fq = {}, set(), {}, {}
    for fq, cols in (schema_catalog or {}).items():
        s, t = fq.replace('"','').split('.')
        key = f"{s.lower()}.{t.lower()}"
        fq_to_canonical[key] = fq
        schemas.add(s)
        tables_by_schema.setdefault(s, set()).add(t)
        cols_by_fq[key] = {c.lower() for c in cols}
    return fq_to_canonical, sorted(schemas), {k: sorted(v) for k,v in tables_by_schema.items()}, cols_by_fq

def snap_fq_tables_to_catalog(sql: str, schema_catalog: dict) -> str:
    if not schema_catalog: 
        return sql
    fq_to_canonical, schemas, tables_by_schema, _ = build_catalog_index(schema_catalog)

    def replace_fq(schema_raw, table_raw):
        s = schema_raw.replace('"','')
        t = table_raw.replace('"','')
        key = f"{s.lower()}.{t.lower()}"
        if key in fq_to_canonical:
            sch, tab = fq_to_canonical[key].replace('"','').split('.')
            return f'"{sch}"."{tab}"'
        s_guess = get_close_matches(s, [x.lower() for x in schemas], n=1, cutoff=0.8)
        if not s_guess:
            return f'"{s}"."{t}"'
        s_best = next(x for x in schemas if x.lower()==s_guess[0])
        t_guess = get_close_matches(t, [x.lower() for x in tables_by_schema[s_best]], n=1, cutoff=0.75)
        if not t_guess:
            return f'"{s}"."{t}"'
        t_best = next(x for x in tables_by_schema[s_best] if x.lower()==t_guess[0])
        return f'"{s_best}"."{t_best}"'

    sql = _QUOTED_FQ.sub(lambda m: replace_fq(m.group(1), m.group(2)), sql)
    sql = _BARE_FQ.sub(lambda m: replace_fq(m.group(1), m.group(2)), sql)
    return sql

def extract_used_tables_and_aliases(sql: str):
    used_fq, alias_to_fq = set(), {}
    it = re.finditer(
        r'\b(FROM|JOIN)\s+("[A-Za-z_]\w*"\s*\.\s*"[A-Za-z_]\w*"|[A-Za-z_]\w*\s*\.\s*[A-Za-z_]\w*)(?:\s+AS)?\s+([A-Za-z_]\w+)?',
        sql, flags=re.IGNORECASE
    )
    for m in it:
        fq = m.group(2).replace('"','').lower().replace(' ','')
        used_fq.add(fq)
        if m.group(3):
            alias_to_fq[m.group(3).lower()] = fq
    return used_fq, alias_to_fq

def snap_columns_to_catalog(sql: str, schema_catalog: dict) -> str:
    if not schema_catalog:
        return sql
    fq_to_canonical, _, _, cols_by_fq = build_catalog_index(schema_catalog)
    used_fq, alias_to_fq = extract_used_tables_and_aliases(sql)

    qualifier_to_fq = {**alias_to_fq}
    for fq in used_fq:
        _, tab = fq.split('.')
        qualifier_to_fq[tab] = fq

    qual = '|'.join(map(re.escape, sorted(qualifier_to_fq.keys(), key=len, reverse=True))) or r'[A-Za-z_]\w*'
    col_pat = re.compile(rf'\b({qual})\s*\.\s*"?(?P<col>[A-Za-z_]\w*)"?', flags=re.IGNORECASE)

    def repl(m):
        q = m.group(1).lower()
        col_raw = m.group('col')
        fq = qualifier_to_fq.get(q)
        if not fq:
            return m.group(0)
        cols = list(cols_by_fq.get(fq, []))
        if not cols:
            return m.group(0)
        if col_raw.lower() in cols:
            col_ok = col_raw
        else:
            guess = get_close_matches(col_raw.lower(), cols, n=1, cutoff=0.8)
            if not guess:
                return m.group(0)
            col_ok = next(c for c in cols if c==guess[0])
        return re.sub(r'("?)[A-Za-z_]\w*\1\s*\.\s*"?[A-Za-z_]\w*"?', f'{m.group(1)}.{col_ok}', m.group(0))

    return col_pat.sub(repl, sql)

def enforce_allowed_tables(sql: str, allowed_tables: list[str]) -> str:
    if not allowed_tables:
        return sql
    allowed_norm = { fq.replace('"','').lower() for fq in allowed_tables }
    def killer(match):
        clause = match.group(0)
        fq = clause.replace('"','')
        m = re.search(r'([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)', fq)
        if not m:
            return clause
        key = f"{m.group(1).lower()}.{m.group(2).lower()}"
        if key in allowed_norm:
            return clause
        return f"-- REMOVED (not allowed) {clause}"
    return re.sub(r'\b(FROM|JOIN)\b[^;]+', killer, sql, flags=re.IGNORECASE)

def dedupe_identifiers(sql: str) -> str:
    sql = re.sub(r'\.\s*\.', '.', sql)
    sql = re.sub(r'"\s*"', '"', sql)
    return sql
# --- Minimal final patcher for common LLM glitches ---
_ID = r'"?[A-Za-z_][\w$]*"?'
def fix_common_llm_glitches(sql: str) -> str:
    if not sql:
        return sql

    # 1) collapse repeated schema.table.table
    #    e.g. "stage"."loan_main_2024"."loan_main_2024".col -> "stage"."loan_main_2024".col
    sql = re.sub(rf'(({_ID})\.({_ID}))\.\3(\.)', r'\1\4', sql)

    # 2) kill doubled keywords
    replacements = {
        r'\bFROM\s+FROM\b': 'FROM',
        r'\bGROUP\s+BY\s+BY\b': 'GROUP BY',
        r'\bORDER\s+BY\s+BY\b': 'ORDER BY',
        r'\bIS\s+IS\b': 'IS',
        r'\bEXTRACTRACT\b': 'EXTRACT',
        r'\bYYEAR\b': 'YEAR',
        r'\bMMONTH\b': 'MONTH',
        r',\s*(WHERE|GROUP BY|ORDER BY|HAVING)\b': r' \1',
    }
    for pat, rep in replacements.items():
        sql = re.sub(pat, rep, sql, flags=re.IGNORECASE)

    # 3) collapse crazy quotes and dot spacing
    sql = re.sub(r'""+', '"', sql)
    sql = re.sub(r'"\s*\.\s*"', '"."', sql)
    sql = re.sub(r'\.\s*\.', '.', sql)

    # 4) normalize EXTRACT/AGE month diff if model invented EPOCH/math
    #    Replace any "EXTRACT(EPOCH FROM age(a,b))/30.44" with canonical MOB in months
    def _mob_expr(tbl: str, report_col: str, funded_col: str) -> str:
        return (f" (DATE_PART('year', AGE({tbl}.{report_col}, {tbl}.{funded_col})) * 12"
                f" + DATE_PART('month', AGE({tbl}.{report_col}, {tbl}.{funded_col}))) ")

    # Try to detect the table qualifier used most in the query
    m = re.search(rf'({_ID}\.{_ID})', sql)
    tbl = m.group(1) if m else None
    if tbl:
        # heuristics: prefer these column names if present
        rcol = 'reportdate'
        fcol = 'fundeddate'
        # only patch if the ugly EPOCH pattern is present
        if re.search(r'EXTRACT\s*\(\s*EPOCH\s+FROM\s+AGE\(', sql, flags=re.IGNORECASE):
            mob = _mob_expr(tbl, rcol, fcol)
            sql = re.sub(r'EXTRACT\s*\(\s*EPOCH\s+FROM\s+AGE\([^)]+\)\)\s*/\s*\d+(\.\d+)?',
                         mob, sql, flags=re.IGNORECASE)

    # 5) fix month filter: prefer date_trunc('month', reportdate)=DATE 'YYYY-MM-01'
    #    Repair obvious "'2025-" duplication artifacts
    sql = re.sub(r"'2025-\s*'2025-", "'2025-", sql)

    return sql


def _dedupe_tokens(s: str) -> str:
    import re
    for pat, repl in [
        (r'\bWHEN\s+WHEN\b', 'WHEN'),
        (r'\bEND\s+END\b', 'END'),
        (r'\bFROM\s+FROM\b', 'FROM'),
        (r'\bWHERE\s+WHERE\b', 'WHERE'),
        (r'\bGROUP\s+BY\s+BY\b', 'GROUP BY'),
        (r'\bORDER\s+BY\s+BY\b', 'ORDER BY'),
        (r'EXTRACT\s*\(\s*MONTH\s+EXTRACT\s*\(\s*MONTH', 'EXTRACT(MONTH'),
        (r'EXTRACT\s*\(\s*YEAR\s+EXTRACT\s*\(\s*YEAR', 'EXTRACT(YEAR'),
    ]:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    return s


def strip_duplicate_keywords(sql: str) -> str:
    if not sql:
        return sql
    fixes = {
        r'\bWHERE\s+WHERE\b': 'WHERE',
        r'\bSELECT\s+SELECT\b': 'SELECT',
        r'\bAND\s+AND\b': 'AND',
        r'\bOR\s+OR\b': 'OR',
    }
    for pat, rep in fixes.items():
        sql = re.sub(pat, rep, sql, flags=re.IGNORECASE)
    return sql


def choose_allowed_tables(question: str, schema_catalog: dict, embedder, collection, max_tables: int = 3):
    if not schema_catalog:
        return []
    q_emb = embedder.encode([question]).tolist()[0]
    res = collection.query(query_embeddings=[q_emb], n_results=max_tables)
    picked, metas = [], (res.get("metadatas") or [[]])[0]
    for m in metas:
        t = (m or {}).get("table")
        if t and t in schema_catalog:
            picked.append(t)
    if not picked:
        picked = list(schema_catalog.keys())[:max_tables]
    return picked


# =================== END NEW HELPERS ===================

# ---------- END ADD ----------






# find anything that *looks* like schema.table even if quotes are messy
# e.g. "stage".loan_main_2024, stage"."loan_main4", "stage"."loan_main_2024"
_MAYBE_FQ_RE = re.compile(
    r'("?[A-Za-z_][\w$]*"?)[\s]*\.[\s]*("?[A-Za-z_][\w$]*"?)'
)

# simple aggregate names
_AGG = r'(SUM|AVG|COUNT|MIN|MAX)'
# `) SUM(` -> `), SUM(` ; also fixes after alias: `AS x) SUM(` -> `AS x, SUM(`
_MISSING_COMMA_AGG_RE = re.compile(r'\)\s+' + _AGG + r'\s*\(', re.IGNORECASE)
_ALIAS_EXTRA_PAREN_RE = re.compile(r'(AS\s+[A-Za-z_]\w*)\)\s+' + _AGG + r'\s*\(', re.IGNORECASE)

def _normalize_identifier(token: str) -> str:
    """Remove quotes/spaces → lower, ' "stage" . "loan" ' → 'stage.loan'."""
    t = token.replace('"', '').replace(' ', '').strip()
    return t.lower()

def _requote_fq(fq: str) -> str:
    """stage.loan_main_2024 → "stage"."loan_main_2024"."""
    if '.' not in fq:
        return fq
    s, t = fq.split('.', 1)
    return f'"{s}"."{t}"'

def _known_tables_set(schema_catalog: dict) -> set[str]:
    return { fq.replace('"','').lower() for fq in schema_catalog.keys() }

def _closest(s: str, pool: set[str], min_score: float = 0.72) -> str | None:
    """Fuzzy map to the closest valid table."""
    if not pool:
        return None
    best = max(pool, key=lambda k: SequenceMatcher(None, s, k).ratio())
    score = SequenceMatcher(None, s, best).ratio()
    return best if score >= min_score else None

def _heal_schema_tables(sql: str, schema_catalog: dict) -> str:
    """
    For each token matching schema.table (even if quotes are broken),
    map to closest known table from schema_catalog.
    """
    known = _known_tables_set(schema_catalog)
    if not known:
        return sql

    def repl(m):
        left, right = m.group(1), m.group(2)
        raw = f"{left}.{right}"
        norm = _normalize_identifier(raw)
        # already exact?
        if norm in known:
            return f'{_requote_fq(norm)}'
        # try best match
        best = _closest(norm, known)
        return _requote_fq(best) if best else raw

    # First pass: heal likely tokens
    s = _MAYBE_FQ_RE.sub(repl, sql)

    # Second pass: if something like stage"."loan_main4" survived, run again
    s = _MAYBE_FQ_RE.sub(repl, s)
    return s

def _fix_aggregate_commas_and_alias(sql: str) -> str:
    s = sql
    # missing comma between aggregates
    s = _MISSING_COMMA_AGG_RE.sub(lambda m: f'), {m.group(1)}(', s)
    # extra ')' after alias before next aggregate
    s = _ALIAS_EXTRA_PAREN_RE.sub(lambda m: f'{m.group(1)}, {m.group(2)}(', s)
    return s

def _trim_excess_paren(sql: str) -> str:
    """
    If we have exactly one more ')' than '(' and it appears right after an alias,
    drop that one. Non-destructive otherwise.
    """
    if sql.count(')') == sql.count('(') + 1:
        # remove first extra ) that comes right after "AS alias)"
        s2, n = re.subn(r'(AS\s+[A-Za-z_]\w*)\)\s', r'\1 ', sql, count=1, flags=re.IGNORECASE)
        if n == 1:
            return s2
    return sql

def post_sanitize_lint_dynamic(sql: str, schema_catalog: dict) -> str:
    """
    Final dynamic cleanup:
      1) heal schema.table tokens to nearest valid
      2) add missing commas between aggregates
      3) remove redundant ')' after aliases (only when safe)
      4) try trim a single excess ')' if parens off by one
    """
    if not isinstance(sql, str) or not sql.strip():
        return sql

    s = sql
    s = _heal_schema_tables(s, schema_catalog)
    s = _fix_aggregate_commas_and_alias(s)
    s = _trim_excess_paren(s)

    # If still unbalanced, leave as-is (don’t over-correct)
    return s
# ---------- END ADD ----------

# ---------- ADD: last-mile dynamic SQL repair ----------
import re

# ---------- REPLACE: last-mile dynamic SQL repair (idempotent & safe) ----------
import re

_ID = r'"?[A-Za-z_][\w$]*"?'
_FQ = rf'{_ID}\s*\.\s*{_ID}'

def _requote_fq_norm(fq: str) -> str:
    """Ensure fq is quoted: schema.table or "schema"."table" -> "schema"."table"."""
    if not fq:
        return fq
    parts = fq.replace('"','').split('.')
    if len(parts) != 2:
        return fq
    return f'"{parts[0]}"."{parts[1]}"'

def _find_table_with_column(schema_catalog: dict, col_name: str) -> str | None:
    """Return fully-quoted schema.table that contains the given column (case-insensitive)."""
    want = (col_name or '').lower()
    for fq, cols in (schema_catalog or {}).items():
        try:
            if any(c.lower() == want for c in cols):
                return fq  # already quoted like "schema"."table"
        except Exception:
            continue
    return None

def _cast_chargeoffs_expr(fq_charge_table: str) -> str:
    """
    Parenthesized CAST expression for chargeoffs to NUMERIC, stripping non-digits.
    NOTE: returns an *expression*, never a quoted identifier.
    """
    fq = _requote_fq_norm(fq_charge_table)
    return f"(NULLIF(REGEXP_REPLACE({fq}.chargeoffs, '[^0-9\\.-]', '', 'g'), '')::numeric)"

def _basic_token_cleanup(s: str) -> str:
    # de-dupe keywords & operators
    s = re.sub(r'\bWHEN\s+WHEN\b', 'WHEN', s, flags=re.IGNORECASE)
    s = re.sub(r'\bEND\s+END\b', 'END', s, flags=re.IGNORECASE)
    s = re.sub(r'\bFROM\s+FROM\b', 'FROM', s, flags=re.IGNORECASE)
    s = re.sub(r'\bWHERE\s+WHERE\b', 'WHERE', s, flags=re.IGNORECASE)
    s = re.sub(r'\bGROUP\s+BY\s+BY\b', 'GROUP BY', s, flags=re.IGNORECASE)
    s = re.sub(r'\bORDER\s+BY\s+BY\b', 'ORDER BY', s, flags=re.IGNORECASE)
    s = re.sub(r'>\s*>', '>', s)
    s = re.sub(r'<\s*<', '<', s)
    s = re.sub(r'=\s*=', '=', s)
    # quotes & dots
    s = re.sub(r'""+', '"', s)
    s = re.sub(r'"\s*\.\s*"', '"."', s)
    s = re.sub(r'\.\s*\.', '.', s)
    # duplicated schema qualifier like "stage"."stage"."tbl" -> "stage"."tbl"
    s = re.sub(rf'(({_ID})\s*\.\s*({_ID}))\s*\.\s*\2(\s*\.)', r'\1\4', s)
    # EXTRACT glitches
    s = re.sub(r'(?i)\bEXTRACT\s*\(\s*MON\b', 'EXTRACT(MONTH', s)
    s = re.sub(r'(?i)\bEXTRACT\s*\(\s*YEAR\s+EXTRACT\s*\(\s*YEAR', 'EXTRACT(YEAR', s)
    s = re.sub(r'(?i)\bEXTRACT\s*\(\s*MONTH\s+EXTRACT\s*\(\s*MONTH', 'EXTRACT(MONTH', s)
    return s


def _strip_illegal_quotes(s: str) -> str:
    import re
    # remove quotes immediately before a keyword
    kw = r'(SELECT|FROM|WHERE|GROUP|ORDER|BY|HAVING|CASE|WHEN|THEN|ELSE|END|SUM|COUNT|AVG|MIN|MAX|DATE_TRUNC|EXTRACT|AGE|CAST|NULLIF|REGEXP_REPLACE)\b'
    s = re.sub(rf'"\s+(?={kw})', '', s, flags=re.IGNORECASE)
    # remove quotes directly before a parenthesized expression after any qualifier:  ..."tbl".(  →  (
    s = re.sub(r'"?\s*\.\s*"\s*\(', '(', s)
    # collapse any leftover quote-space-quote
    s = re.sub(r'"\s+"', '"', s)
    return s


def last_mile_sql_guard(sql: str, schema_catalog: dict, question: str = "") -> str:
    """
    Final defensive pass (idempotent):
      - light cleanup for duplicate tokens and broken quotes/dots
      - replace any use of chargeoffs with a *parenthesized expression* (no quoting as identifier)
      - skip if already casted (idempotency)
    """
    if not sql or not isinstance(sql, str):
        return sql

    s = _basic_token_cleanup(sql)

    # If no 'chargeoffs' present, we're done with lightweight fixes
    if not re.search(r'(?i)\bchargeoffs\b', s):
        return re.sub(r'\s+', ' ', s).strip()

    # Find the table that actually has 'chargeoffs'
    charge_fq = _find_table_with_column(schema_catalog, 'chargeoffs')
    if not charge_fq:
        # Leave query as-is (other passes may still succeed)
        return re.sub(r'\s+', ' ', s).strip()

    cast_expr = _cast_chargeoffs_expr(charge_fq)

    # Idempotency: if expression already present, don't re-wrap
    already_casted = re.search(r'(?i)REGEXP_REPLACE\([^)]*chargeoffs', s) is not None

    # Use a placeholder while rewriting to avoid double-substitution
    PH = "__CO_CAST_EXPR__"

    if not already_casted:
        # 1) Replace fully-qualified occurrences first:  schema.table.chargeoffs  →  (CAST_EXPR)
        s = re.sub(
            rf'(?i){_FQ}\s*\.\s*chargeoffs\b',
            PH,
            s
        )

        # 2) Replace unqualified 'chargeoffs' that are not part of an identifier, function, or already replaced
        #    Use word boundaries and ensure it's not like foochargeoffs or chargeoffsv2
        s = re.sub(r'(?i)\bchargeoffs\b', PH, s)

        # 3) Collapse accidental double placeholders, just in case
        s = re.sub(rf'{PH}{PH}+', PH, s)

        # 4) Finally materialize the placeholder as the *expression* (no extra quoting)
        s = s.replace(PH, cast_expr)

    # One more light pass to ensure we never get: "schema"."table".(expr)
    # i.e., remove any qualifier directly followed by a parenthetical expression
    s = re.sub(
        rf'(?i){_FQ}\s*\.\s*\(',
        '(',
        s
    )

    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    s = _strip_illegal_quotes(s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ---------- END REPLACE ----------
# ===================== ULTRA-FIX LAYER (ADD; DO NOT DELETE ANYTHING) =====================
import re
from difflib import get_close_matches

_ID = r'"?[A-Za-z_][\w$]*"?'
_FQ = rf'{_ID}\s*\.\s*{_ID}'

def _catalog_indices(schema_catalog: dict):
    """Build quick lookups."""
    fq_canon = {}             # norm fq -> quoted canonical fq
    cols_by_fq = {}           # norm fq -> set of lower-case column names
    all_cols  = set()
    table_norms = set()
    for fq, cols in (schema_catalog or {}).items():
        s, t = fq.replace('"','').split('.')
        key = f"{s.lower()}.{t.lower()}"
        table_norms.add(key)
        fq_canon[key] = f'"{s}"."{t}"'
        cols_by_fq[key] = {c.lower() for c in cols}
        all_cols.update(c.lower() for c in cols)
    return fq_canon, cols_by_fq, table_norms, all_cols

def _insert_missing_dots_between_quotes(sql: str) -> str:
    # … "schema""table" → "schema"."table"
    sql = re.sub(r'"([A-Za-z_]\w*)"\s*"\s*([A-Za-z_]\w*)"', r'"\1"." \2"', sql)  # temp space to keep from re-collapsing
    sql = sql.replace('".\" ', '"."')  # clean the temp space pattern if formed
    sql = re.sub(r'"\s*"\.', '".', sql) # "".
    return sql

def _collapse_duplicated_table_qual(sql: str) -> str:
    # "s"."t"."t".col  -> "s"."t".col
    return re.sub(rf'(({_ID})\s*\.\s*({_ID}))\s*\.\s*\3(\s*\.)', r'\1\4', sql)

def _normalize_interval_literals(sql: str) -> str:
    # Fix "INTERVAL6 month" → INTERVAL '6 month(s)'
    def repl(m):
        n, unit = m.group(1), m.group(2)
        unit = unit.lower()
        if not unit.endswith('s'):
            unit += 's'
        return f"INTERVAL '{n} {unit}'"
    sql = re.sub(r'\bINTERVAL\s*([0-9]+)\s*(year|years|month|months|day|days|hour|hours|minute|minutes)\b', repl, sql, flags=re.IGNORECASE)

    # Also fix "+ INTERVAL6 month" missing space
    sql = re.sub(r'\+\s*INTERVAL\s*([0-9]+)\s*([A-Za-z]+)', lambda m: f"+ INTERVAL '{m.group(1)} {m.group(2)}'", sql, flags=re.IGNORECASE)
    return sql

def _repair_date_literals(sql: str) -> str:
    # Keep only valid YYYY-MM-DD inside single-quoted literals; drop garbage tails like 20242024
    def clean_date_literal(m):
        body = m.group(1)
        # Extract first YYYY-MM-DD
        m2 = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', body)
        return f"'{m2.group(1)}'" if m2 else f"'{body}'"
    sql = re.sub(r"'([^']+)'", clean_date_literal, sql)

    # Fix duplicated prefix "'2025-' '2025-…"
    sql = re.sub(r"'\s*2025-\s*'\s*2025-", "'2025-", sql)
    return sql

def _token_cleanup_hard(sql: str) -> str:
    # Build on your own cleanups; add stronger guards for stray concatenations.
    for pat, rep in [
        (r'\.\s*\.', '.'), (r'""+', '"'), (r'"\s*\.\s*"', '"."'),
        (r'\bFROM\s+FROM\b', 'FROM'), (r'\bGROUP\s+BY\s+BY\b', 'GROUP BY'),
        (r'\bORDER\s+BY\s+BY\b', 'ORDER BY'), (r'\bWHERE\s+WHERE\b', 'WHERE'),
        (r'\bSELECT\s+SELECT\b', 'SELECT'), (r'\bIS\s+IS\b', 'IS'),
        (r',\s*(WHERE|GROUP BY|ORDER BY|HAVING)\b', r' \1'),
    ]:
        sql = re.sub(pat, rep, sql, flags=re.IGNORECASE)
    sql = _insert_missing_dots_between_quotes(sql)
    sql = _collapse_duplicated_table_qual(sql)
    return sql

def _used_tables_and_aliases(sql: str):
    used_fq, alias_to_fq = set(), {}
    it = re.finditer(
        rf'\b(FROM|JOIN)\s+(({_ID})\s*\.\s*({_ID}))(?:\s+AS)?\s+([A-Za-z_]\w+)?',
        sql, flags=re.IGNORECASE
    )
    for m in it:
        fq = m.group(2).replace('"','').lower().replace(' ','')
        used_fq.add(fq)
        if m.group(5):
            alias_to_fq[m.group(5).lower()] = fq
    return used_fq, alias_to_fq

def _closest_name(name: str, pool: set[str], cutoff=0.78):
    if not pool:
        return None
    cand = get_close_matches(name.lower(), list(pool), n=1, cutoff=cutoff)
    return cand[0] if cand else None

def _repair_columns(sql: str, schema_catalog: dict) -> str:
    """
    Fix misspelled columns both qualified and unqualified.
    """
    if not schema_catalog:
        return sql

    fq_canon, cols_by_fq, table_norms, all_cols = _catalog_indices(schema_catalog)
    used_fq, alias_to_fq = _used_tables_and_aliases(sql)

    # Map qualifier -> fq
    qualifier_to_fq = {**alias_to_fq}
    for fq in used_fq:
        s, t = fq.split('.')
        qualifier_to_fq[t] = fq  # allow table-qualified refs without alias

    # 1) Qualified references: qualifier.col  → snap 'col'
    qual = '|'.join(map(re.escape, sorted(qualifier_to_fq.keys(), key=len, reverse=True))) or r'[A-Za-z_]\w*'
    qcol_pat = re.compile(rf'\b({qual})\s*\.\s*"?(?P<col>[A-Za-z_]\w*)"?', flags=re.IGNORECASE)

    def qrepl(m):
        q   = m.group(1).lower()
        col = m.group('col')
        fq  = qualifier_to_fq.get(q)
        if not fq:
            return m.group(0)
        cols = cols_by_fq.get(fq, set())
        if col.lower() in cols:
            return m.group(0)  # ok
        guess = _closest_name(col, cols)
        if not guess:
            return m.group(0)
        # keep the original qualifier verbatim; replace only the column token
        return re.sub(r'"?[A-Za-z_]\w*"?\s*$', guess, m.group(0))

    sql = qcol_pat.sub(qrepl, sql)

    # 2) Unqualified bare columns in SELECT/WHERE/ON: try to qualify if unique across used tables
    #    We only fix if name clearly matches exactly one table's column.
    bare_col_pat = re.compile(r'\b([A-Za-z_]\w{3,})\b')  # avoid tiny tokens
    def bare_repl(m):
        tok = m.group(1)
        if tok.lower() in {'select','from','where','join','on','and','or','not','sum','avg','min','max','count','as','case','when','then','else','end','group','by','order','limit','offset','with','distinct','interval','date_trunc','extract','age','cast','nullif','regexp_replace'}:
            return tok
        # If token already appears as part of a qualifier.col pattern, skip (handled above)
        # Try to resolve against used tables
        candidates = []
        for fq in used_fq:
            if tok.lower() in cols_by_fq.get(fq, set()):
                candidates.append((fq, tok.lower()))
        if len(candidates) == 1:
            fq = candidates[0][0]
            s, t = fq.split('.')
            return f'"{t}".{tok}'  # minimally qualify with table
        # Fuzzy single hit across used tables
        fuzzy = []
        for fq in used_fq:
            guess = _closest_name(tok, cols_by_fq.get(fq, set()))
            if guess:
                fuzzy.append((fq, guess))
        if len(fuzzy) == 1:
            fq, guess = fuzzy[0]
            s, t = fq.split('.')
            return f'"{t}".{guess}'
        return tok

    # apply bare col repair only inside SELECT ... FROM ... / WHERE / GROUP BY / ORDER BY contexts to avoid string/date literals
    sql = bare_col_pat.sub(bare_repl, sql)

    return sql

def _snap_tables_again(sql: str, schema_catalog: dict) -> str:
    """
    After heavy cleanup, run one more strict snap of schema.table tokens:
    - heal "schema.table" to canonical quoting
    - kill unknown tables (leave as-is but quoted) to avoid mis-snapping to trash
    """
    if not schema_catalog:
        return sql
    fq_canon, _, table_norms, _ = _catalog_indices(schema_catalog)

    def repl(m):
        left, right = m.group(1), m.group(2)
        raw = f"{left}.{right}"
        key = raw.replace('"','').replace(' ','').lower()
        if key in fq_canon:
            return fq_canon[key]
        # fuzzy table fix
        guess = _closest_name(key, table_norms, cutoff=0.80)
        return fq_canon.get(guess, f'"{left.replace("\"","").strip()}"."{right.replace("\"","").strip()}"')

    return re.sub(r'("?[A-Za-z_][\w$]*"?)[\s]*\.[\s]*("?[A-Za-z_][\w$]*"?)', lambda m: repl(m), sql)

def sql_ultrafix(sql: str, schema_catalog: dict) -> str:
    """
    Strong, idempotent finalizer that you can run AFTER your existing passes.
    """
    if not sql or not isinstance(sql, str):
        return sql

    s = sql

    # 1) normalize gross tokenization issues (quotes/dots/duplicates)
    s = _token_cleanup_hard(s)

    # 2) fix intervals and dates
    s = _normalize_interval_literals(s)
    s = _repair_date_literals(s)

    # 3) fuzzy-repair columns (qualified + unqualified) using live catalog + used tables
    s = _repair_columns(s, schema_catalog)

    # 4) snap tables again to canonical quoting after repairs
    s = _snap_tables_again(s, schema_catalog)

    # 5) final whitespace + trivial tidying
    s = re.sub(r'\s+', ' ', s).strip()
    return s
# ===================== END ULTRA-FIX LAYER =====================


# ===================== ULTRA-FIX LAYER v2 (ADD ONLY) =====================
import re
from difflib import get_close_matches

_ID = r'"?[A-Za-z_][\w$]*"?'
_FQ = rf'{_ID}\s*\.\s*{_ID}'

def _catalog_indices_v2(schema_catalog: dict):
    schemas, tables_by_schema, cols_by_fq = set(), {}, {}
    fq_canon, all_cols = {}, set()
    for fq, cols in (schema_catalog or {}).items():
        s, t = fq.replace('"','').split('.')
        sL, tL = s.lower(), t.lower()
        key = f'{sL}.{tL}'
        fq_canon[key] = f'"{s}"."{t}"'
        schemas.add(s)
        tables_by_schema.setdefault(sL, set()).add(t)
        cols_by_fq[key] = {c.lower() for c in cols}
        all_cols.update(c.lower() for c in cols)
    return fq_canon, schemas, tables_by_schema, cols_by_fq, all_cols

def _closest_v2(name: str, pool: set[str], cutoff=0.78):
    if not pool:
        return None
    cand = get_close_matches(name.lower(), list(pool), n=1, cutoff=cutoff)
    return cand[0] if cand else None

def _strip_line_comments(sql: str) -> str:
    # remove `-- ...` comments outside of string literals
    out, i, in_s = [], 0, False
    while i < len(sql):
        c = sql[i]
        if c == "'" and (i == 0 or sql[i-1] != "\\"):
            in_s = not in_s
            out.append(c)
            i += 1
            continue
        if not in_s and c == '-' and i+1 < len(sql) and sql[i+1] == '-':
            # skip until line end
            while i < len(sql) and sql[i] not in '\r\n':
                i += 1
            continue
        out.append(c)
        i += 1
    return ''.join(out)

def _dedupe_adjacent_quoted_identifiers(sql: str) -> str:
    # ..."foo""foo" -> ..."foo"
    return re.sub(r'("([A-Za-z_][\w$]*)")\s*\1', r'\1', sql)

def _dedupe_triple_qual(sql: str) -> str:
    # "sch"."tab"."tab".col → "sch"."tab".col
    return re.sub(rf'(({_ID})\s*\.\s*({_ID}))\s*\.\s*\3(\s*\.)', r'\1\4', sql)

def _dedupe_glued_year_suffix(token: str) -> str:
    # loan_main_20242024 -> loan_main_2024 ; 2025073120250731 -> 20250731
    m = re.search(r'(.*?)(\d{4})(\2)+$', token)
    if m:
        return f'{m.group(1)}{m.group(2)}'
    return token

def _dedupe_glued_repeats(token: str) -> str:
    # chargeoffschargeoffs -> chargeoffs
    half = token[:len(token)//2]
    if token == half + half and len(half) >= 5:
        return half
    return token

def _dedupe_glued_identifiers(sql: str, schema_catalog: dict) -> str:
    """
    Inside double-quotes, if an identifier looks like it contains duplicated pieces,
    try to collapse them while snapping back to the nearest catalog name.
    """
    fq_canon, schemas, tables_by_schema, cols_by_fq, all_cols = _catalog_indices_v2(schema_catalog)

    def fix_inside_quotes(m):
        raw = m.group(1)
        tok = _dedupe_glued_year_suffix(raw)
        tok = _dedupe_glued_repeats(tok)
        # try snap to a known column/table if close
        guess = _closest_v2(tok, all_cols, cutoff=0.85)
        if guess:
            return f'"{guess}"'
        # otherwise just return cleaned token
        return f'"{tok}"'

    return re.sub(r'"([A-Za-z_][\w$]*)"', fix_inside_quotes, sql)

def _kill_qualifier_before_paren_expr(sql: str) -> str:
    # "sch"."tab"(  ...  -> ( ...
    sql = re.sub(rf'(?i){_FQ}\s*\(', '(', sql)
    # "alias"(  ...  -> ( ...
    sql = re.sub(rf'(?i)"[A-Za-z_][\w$]*"\s*\(', '(', sql)
    # remove accidental quotes around expressions: (expr)" -> (expr)
    sql = re.sub(r'\)\s*"', ')', sql)
    sql = re.sub(r'"\s*\(', '(', sql)
    return sql

def _limit_qual_depth(sql: str) -> str:
    # prevent db.schema.table misread: reduce any a.b.c to a.b (drop .c)
    return re.sub(rf'(({_ID})\s*\.\s*({_ID})\s*\.\s*({_ID}))', r'\2.\3', sql)

def _normalize_interval_literals_v2(sql: str) -> str:
    def repl(m):
        n, unit = m.group(1), m.group(2).lower()
        if not unit.endswith('s'):
            unit += 's'
        return f"INTERVAL '{n} {unit}'"
    sql = re.sub(r'\bINTERVAL\s*([0-9]+)\s*([A-Za-z]+)\b', repl, sql, flags=re.IGNORECASE)
    sql = re.sub(r'\+\s*INTERVAL\s*([0-9]+)\s*([A-Za-z]+)', lambda m: f"+ INTERVAL '{m.group(1)} {m.group(2)}'", sql, flags=re.IGNORECASE)
    return sql

def _repair_date_literals_v2(sql: str) -> str:
    def clean(m):
        body = m.group(1)
        m2 = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', body)
        return f"'{m2.group(1)}'" if m2 else f"'{body}'"
    sql = re.sub(r"'([^']+)'", clean, sql)
    # "'2025-" "'2025-xx" → keep first
    sql = re.sub(r"'\s*([0-9]{4})-\s*'\s*\1-", r"'\1-", sql)
    return sql


def _snap_tables_again_v2(sql: str, schema_catalog: dict) -> str:
    fq_canon, schemas, tables_by_schema, cols_by_fq, _ = _catalog_indices_v2(schema_catalog)
    table_norms = set(fq_canon.keys())
    def repl(m):
        left, right = m.group(1), m.group(2)
        key = f'{left.replace("\"","").strip().lower()}.{right.replace("\"","").strip().lower()}'
        if key in fq_canon:
            return fq_canon[key]
        guess = _closest_v2(key, table_norms, cutoff=0.80)
        return fq_canon.get(guess, f'"{left.replace("\"","").strip()}"."{right.replace("\"","").strip()}"')
    return re.sub(r'("?[A-Za-z_][\w$]*"?)[\s]*\.[\s]*("?[A-Za-z_][\w$]*"?)', repl, sql)

def _dedupe_chargeoffs_again(sql: str, schema_catalog: dict) -> str:
    # If model produced chargeoffschargeoffs (quoted or not), collapse to chargeoffs where valid.
    valid = set()
    for fq, cols in (schema_catalog or {}).items():
        if any(c.lower() == 'chargeoffs' for c in cols):
            valid.add(fq)
    if not valid:
        return sql
    sql = re.sub(r'(?i)\bchargeoffschargeoffs\b', 'chargeoffs', sql)
    sql = re.sub(r'(?i)"chargeoffschargeoffs"', '"chargeoffs"', sql)
    return sql
def _rebuild_from_if_missing(
    sql: str,
    schema_catalog: dict,
    fallback_from_fq: str | None = None,
    **kw,
) -> str:
    """
    If FROM is missing (e.g. allowlist removed it), add FROM <fallback_from_fq>.

    Backward compatible:
      - accepts fallback_from_fq=...
      - accepts fallback_fq=... (older name)
    """
    # support old param name too
    if fallback_from_fq is None:
        fallback_from_fq = kw.get("fallback_fq")

    # already has FROM? leave it
    if re.search(r'\bFROM\b', sql, flags=re.IGNORECASE):
        return sql

    # choose a fallback table
    fq = fallback_from_fq or (next(iter(schema_catalog)) if schema_catalog else None)
    if not fq:
        return sql

    # inject FROM right before WHERE/GROUP BY/ORDER BY/LIMIT or end of string
    return re.sub(
        r'(?is)^(SELECT\b.*?)\s*(WHERE|GROUP BY|ORDER BY|LIMIT|$)',
        lambda m: f'{m.group(1)} FROM {fq} {m.group(2)}',
        sql,
    )



def sql_ultrafix_v2(sql: str, schema_catalog: dict, fallback_from_fq: str|None=None) -> str:
    """
    Stronger, idempotent end-fixer. Run AFTER your existing guards (and ultrafix v1).
    """
    if not sql or not isinstance(sql, str):
        return sql

    s = sql
    s = _strip_line_comments(s)
    s = _dedupe_adjacent_quoted_identifiers(s)
    s = _dedupe_triple_qual(s)
    s = _limit_qual_depth(s)
    s = _dedupe_glued_identifiers(s, schema_catalog)
    s = _dedupe_chargeoffs_again(s, schema_catalog)
    s = _kill_qualifier_before_paren_expr(s)
    s = _normalize_interval_literals_v2(s)
    s = _repair_date_literals_v2(s)
    s = _snap_tables_again_v2(s, schema_catalog)
    # If FROM vanished due to allow-list removal, re-add using best guess.
    s = _rebuild_from_if_missing(s, schema_catalog, fallback_from_fq=fallback_from_fq)
    s = re.sub(r'\s+', ' ', s).strip()
    return s
# ===================== END ULTRA-FIX LAYER v2 =====================


def sanitize_sql_for_nulls_simple(sql: str, schema_catalog: dict) -> str:
    """
    Lightweight version that only adds NULL guards to aggregates.
    Use AFTER unified_sql_repair if you need extra safety.
    """
    
    # Wrap aggregates with NULLIF for division safety
    sql = re.sub(
        r'(SUM|AVG|MIN|MAX|COUNT)\s*\(([^()]+)\)',
        lambda m: f'{m.group(1)}(COALESCE({m.group(2)}, 0))',
        sql,
        flags=re.I
    )
    
    # Guard division
    sql = re.sub(
        r'/\s*(COUNT|SUM|AVG)\s*\(',
        r'/ NULLIF(\1(',
        sql,
        flags=re.I
    )
    
    return sql
# top-level with your other imports
import re


"""
Complete Unified SQL Repair System
Handles all SQL sanitization, repair, and validation in a single coordinated pipeline.
Drop-in replacement for multiple scattered repair functions.
"""

import re
from difflib import get_close_matches
from typing import Any
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

# def unified_sql_repair(
#     raw_llm_output: Any,
#     schema_catalog: dict,
#     question: str = "",
#     fallback_table: str | None = None
# ) -> str:
#     """
#     Single entry point for all SQL repair operations.
    
#     Args:
#         raw_llm_output: Raw output from LLM (str, tuple, list, or dict)
#         schema_catalog: Dict of {"schema"."table": [col1, col2, ...]}
#         question: Original user question for context
#         fallback_table: Table to use if FROM clause is missing
    
#     Returns:
#         Repaired, valid SQL string
#     """
    
#     original_input = str(raw_llm_output)[:200]  # For logging
    
#     try:
#         # Stage 1: Extract SQL from LLM wrapper formats
#         sql = _extract_sql_from_llm_output(raw_llm_output)
#         logger.debug(f"[Stage 1] Extracted SQL length: {len(sql)}")
        
#         if not sql or len(sql.strip()) < 10:
#             logger.warning("[Stage 1] Empty or too short SQL, generating fallback")
#             return _generate_safe_fallback_sql(schema_catalog, fallback_table)
        
#         # Stage 2: Basic syntax cleanup (before any semantic repairs)
#         sql = _cleanup_basic_syntax(sql)
#         logger.debug(f"[Stage 2] After basic cleanup: {sql[:150]}")
        
#         # Stage 3: Build catalog indices once
#         catalog_idx = _build_catalog_indices(schema_catalog)
#         logger.debug(f"[Stage 3] Catalog has {len(catalog_idx['fq_canon'])} tables")
        
#         # Stage 4: Fix table references
#         sql = _repair_table_references(sql, catalog_idx)
#         logger.debug(f"[Stage 4] After table repair: {sql[:150]}")
        
#         # Stage 5: Fix column references
#         sql = _repair_column_references(sql, catalog_idx)
#         logger.debug(f"[Stage 5] After column repair: {sql[:150]}")

#         # Stage 1-5: Same as before
#         sql = _extract_sql_from_llm_output(raw_llm_output)
#         logger.debug(f"[Stage 1] Extracted SQL length: {len(sql)}")
        
#         if not sql or len(sql.strip()) < 10:
#             logger.warning("[Stage 1] Empty or too short SQL, generating fallback")
#             return _generate_safe_fallback_sql(schema_catalog, fallback_table)
        
#         sql = _cleanup_basic_syntax(sql)
#         logger.debug(f"[Stage 2] After basic cleanup: {sql[:150]}")
        
#         catalog_idx = _build_catalog_indices(schema_catalog)
#         logger.debug(f"[Stage 3] Catalog has {len(catalog_idx['fq_canon'])} tables")
        
#         sql = _repair_table_references(sql, catalog_idx)
#         logger.debug(f"[Stage 4] After table repair: {sql[:150]}")
        
#         sql = _repair_column_references(sql, catalog_idx)
#         logger.debug(f"[Stage 5] After column repair: {sql[:150]}")
        
#         # Stage 6: Fix date/interval literals AND date column types
#         sql = _repair_date_and_interval_literals(sql)
#         sql = _repair_date_columns(sql, schema_catalog)
#         logger.debug(f"[Stage 6] After date/interval repair: {sql[:150]}")
        
#         # Stage 7: Add NULL/zero guards AND dynamic type casting
#         sql = _add_safety_guards(sql, schema_catalog, catalog_idx)
#         logger.debug(f"[Stage 7] After safety guards: {sql[:150]}")
        
#         # Stage 8-9: Same as before
#         sql = _ensure_from_clause(sql, schema_catalog, fallback_table)
#         logger.debug(f"[Stage 8] After FROM check: {sql[:150]}")
        
#         sql = _final_validation(sql, schema_catalog)
#         logger.debug(f"[Stage 9] Final SQL: {sql[:200]}")
        
#         return sql
        
#     except Exception as e:
#         logger.error(f"SQL repair failed: {e}. Input was: {original_input}")
#         return _generate_safe_fallback_sql(schema_catalog, fallback_table)


import re
import logging
from difflib import get_close_matches
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
import re
import logging
from typing import Any
from difflib import get_close_matches  # you already use this in table/col repair
logger = logging.getLogger(__name__)

# -------------------------------
# Hardened FINAL helper passes
# -------------------------------

def _cleanup_basic_syntax_FINAL11(sql: str) -> str:
    # remove comments
    sql = re.sub(r'--[^\n]*', '', sql)

    # de-dupe keywords
    for kw in ['SELECT','FROM','WHERE','GROUP BY','ORDER BY','HAVING','WHEN','END','CASE']:
        sql = re.sub(rf'\b({kw})\s+\1\b', r'\1', sql, flags=re.I)

    # duplicated DATE and silly year self-comparisons
    sql = re.sub(r"\bDATE\s+DATE\s+(')", r"DATE \1", sql, flags=re.I)
    sql = re.sub(r'\s+(AND|OR)\s+\(?\s*(20\d{2})\s*=\s*\2\s*\)?(?=\s+(AND|OR|$))', ' ', sql, flags=re.I)
    sql = re.sub(r'\bWHERE\s+\(?\s*(20\d{2})\s*=\s*\1\s*\)?\s+(AND|OR)\s+', 'WHERE ', sql, flags=re.I)

    # ROUND( ... ) AS alias → ensure closing paren
    sql = re.sub(r'\bROUND\s*\(([^)]*)\s+AS\b', r'ROUND(\1) AS', sql, flags=re.I)

    # 👇 close NULLIF aggregate before AS alias; also ensure ", 0)" for division guards
    sql = re.sub(
        r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s+AS\b',
        r'NULLIF(\1(\2)\3, 0) AS', sql, flags=re.I
    )
    sql = re.sub(
        r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s*\)\s*/',
        r'NULLIF(\1(\2)\3, 0) /', sql, flags=re.I
    )

    # strip bare years from GROUP BY / ORDER BY lists
    sql = re.sub(r'\bGROUP\s+BY\s*(.*?)(?=\bORDER\b|\bLIMIT\b|$)',
                 lambda m: 'GROUP BY ' + re.sub(r'(?:^|,)\s*20\d{2}\s*(?=,|$)', ',', m.group(1)).strip(' ,'),
                 sql, flags=re.I)
    sql = re.sub(r'\bORDER\s+BY\s*(.*?)(?=\bLIMIT\b|$)',
                 lambda m: 'ORDER BY ' + re.sub(r'(?:^|,)\s*20\d{2}\s*(?=,|$)', ',', m.group(1)).strip(' ,'),
                 sql, flags=re.I)

    # misc normalizations you already had
    sql = re.sub(r'\bGROUP\s+BY\s+BY\b','GROUP BY',sql,flags=re.I)
    sql = re.sub(r'\bORDER\s+BY\s+BY\b','ORDER BY',sql,flags=re.I)
    sql = re.sub(r'\bEXTRACTRACT\b','EXTRACT',sql,flags=re.I)
    sql = re.sub(r'\bYYEAR\b','YEAR',sql,flags=re.I)
    sql = re.sub(r'\bMMONTH\b','MONTH',sql,flags=re.I)
    sql = re.sub(r'""+','"',sql)
    sql = re.sub(r'"\s*\.\s*"', '"."', sql)
    sql = re.sub(r'\.\s*\.', '.', sql)
    sql = re.sub(r'"([A-Za-z_]\w*)"\s*"\s*([A-Za-z_]\w*)"', r'"\1"."\2"', sql)
    sql = re.sub(r',\s*(WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|FROM)\b', r' \1', sql, flags=re.I)
    sql = re.sub(r'>\s*>', '>', sql); sql = re.sub(r'<\s*<', '<', sql); sql = re.sub(r'=\s*=', '=', sql)
    sql = re.sub(r"(INTERVAL\s*'\s*\d+\s*)month\b", r"\1months", sql, flags=re.I)

    return re.sub(r'\s+', ' ', sql).strip()


def _repair_date_and_interval_literals_FINAL11(sql: str) -> str:
    sql = re.sub(r'\bINTERVAL\s*([0-9]+)\s*(year|month|day|hour|minute|second)s?\b',
                 lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}s'", sql, flags=re.I)
    sql = re.sub(r'\b([0-9]+)(days?|months?|years?)\b',
                 lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}'", sql, flags=re.I)
    sql = re.sub(r"'(\d{4}-\d{2}-\d{2})'\s*'(\d{4}-\d{2}-\d{2})'", r"'\1'", sql)
    sql = re.sub(r"'(\d{4}-\d{2}-\d{2})'\s*([+-])\s*INTERVAL", r"DATE '\1' \2 INTERVAL", sql, flags=re.I)
    return sql


def _repair_date_columns_FINAL(sql: str, schema_catalog: dict) -> str:
    # gather date-ish columns by name
    date_cols = set()
    for _, cols in (schema_catalog or {}).items():
        for c in cols:
            lc = c.lower()
            if any(x in lc for x in ['date','time','created','updated','funded','purchase','report','maturity','expiration']):
                date_cols.add(lc)
    if not date_cols:
        return sql

    # EXTRACT(unit FROM col) → CAST(col AS DATE)
    def fix_extract(m):
        unit, col = m.group(1), m.group(2).strip()
        base = col.split('.')[-1].replace('"','').lower()
        if base in date_cols:
            return f"EXTRACT({unit} FROM CAST({col} AS DATE))"
        return m.group(0)

    sql = re.sub(r'EXTRACT\s*\(\s*(\w+)\s+FROM\s+(["\w.]+)\s*\)', fix_extract, sql, flags=re.I)

    # date arithmetic col ± INTERVAL/DATE → CAST(col AS DATE) ± ...
    for base in date_cols:
        sql = re.sub(
            rf'([\w."]+\.{base})(?!\s*(?:::DATE|AS\s+DATE))\s*([+-])\s*(INTERVAL|DATE)',
            lambda m: f"CAST({m.group(1)} AS DATE) {m.group(2)} {m.group(3)}",
            sql, flags=re.I
        )
    return sql

import re
# logger = logging.getLogger(__name__)  # assumed available

# def _add_safety_guards_FINAL(sql: str, schema_catalog: dict, catalog_idx: dict) -> str:
#     """
#     1) Guard divisions by aggregates by inserting NULLIF(AGG(...), 0)
#     2) Correct the 'open-NULLIF right before AS' case that caused your syntax error
#     3) Keep your dynamic numeric casting afterward
#     """
#     # (A) If we divide by an aggregate, open a NULLIF(
#     #     e.g.  "/ COUNT("  ->  "/ NULLIF(COUNT("
#     sql = re.sub(
#         r'/\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(',
#         r'/ NULLIF(\1(',
#         sql,
#         flags=re.I
#     )

#     # (B) If we now have a *closed* NULLIF(...), make sure it has ", 0)"
#     #     Works with and without an immediate ::cast after the aggregate.
#     sql = re.sub(
#         r'NULLIF\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s*\)',
#         r'NULLIF(\1(\2)\3, 0)',
#         sql,
#         flags=re.I
#     )

#     # (C) 🚑 Critical fix: if the NULLIF was injected and the very next token is AS,
#     #     close it before the alias. This is the exact failure you’re seeing.
#     #     Example broken:  "... / NULLIF(COUNT(col) AS approval_rate"
#     #     Becomes:         "... / NULLIF(COUNT(col), 0) AS approval_rate"
#     sql = re.sub(
#         r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s+(AS\b)',
#         r'NULLIF(\1(\2)\3, 0) \4',
#         sql,
#         flags=re.I
#     )

#     # (D) One more belt-and-suspenders: if somehow we still have an unclosed NULLIF before
#     #     a clause/comma/operator, close it.
#     #     This catches oddities like "... / NULLIF(COUNT(col)) , ..." or before ) or /
#     sql = re.sub(
#         r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s*(?=[,/)\+\-\*]|FROM\b|WHERE\b|GROUP\b|ORDER\b|LIMIT\b)',
#         r'NULLIF(\1(\2)\3, 0)',
#         sql,
#         flags=re.I
#     )

#     # (E) Your existing dynamic numeric casting heuristic
#     sql = _cast_text_columns_to_numeric(sql, schema_catalog)

#     return sql


# def _final_validation_FINAL(sql: str, schema_catalog: dict) -> str:
#     """
#     Final sanity: structure, parens, & last-mile token cleanup.
#     Also collapses 'DATE DATE' duplicates that can reappear late.
#     """
#     # whitespace normalize early so regex anchors behave
#     sql = re.sub(r'\s+', ' ', sql).strip()

#     # Late sweep: collapse accidental double DATE tokens (e.g. "DATE DATE '2024-09-01'")
#     sql = re.sub(r"\bDATE\s+DATE\s+(')", r"DATE \1", sql, flags=re.I)

#     # Also normalize INTERVAL singular → plural (optional but helpful)
#     sql = re.sub(r"(INTERVAL\s*'\s*\d+\s*)month\b", r"\1months", sql, flags=re.I)

#     # Ensure there’s a basic SELECT ... FROM shape
#     if not re.search(r'\bSELECT\b.*\bFROM\b', sql, flags=re.I | re.S):
#         logger.warning("[Final] SQL missing SELECT...FROM structure")
#         return _generate_safe_fallback_sql(schema_catalog)

#     # Balance parentheses (tolerate small off-by-N; otherwise fallback)
#     opens, closes = sql.count('('), sql.count(')')
#     if opens != closes:
#         diff = opens - closes
#         if 0 < diff <= 3:
#             sql += ')' * diff
#         elif diff < 0 and -diff <= 3:
#             sql = '(' * (-diff) + sql
#         else:
#             logger.error("[Final] Too many unbalanced parens, generating fallback")
#             return _generate_safe_fallback_sql(schema_catalog)

#     # Trim funky trailing ');)' patterns
#     sql = re.sub(r';\s*\)+\s*$', ';', sql)

#     # One last guard: if any lingering "NULLIF(... AS" slipped through, fix it.
#     if re.search(r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\([^)]*\)\s*(::\w+)?\s+AS\b', sql, flags=re.I):
#         sql = re.sub(
#             r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s+AS\b',
#             r'NULLIF(\1(\2)\3, 0) AS',
#             sql,
#             flags=re.I
#         )

#     # Normalize again at the end
#     return re.sub(r'\s+', ' ', sql).strip()


# ------------------------------------------------------------
# Your two functions (drop-in replacements)
# ------------------------------------------------------------
# sql_repair.py

import logging
import re
from difflib import get_close_matches
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# =========================
# Public entry point
# =========================

def unified_sql_repair11(
    raw_llm_output: Any,
    schema_catalog: Dict[str, List[str]],
    question: str = "",
    fallback_table: str | None = None
) -> str:
    """
    Single entry point for all SQL repair operations.

    Args:
        raw_llm_output: Raw output from LLM (str, tuple, list, or dict)
        schema_catalog: Dict of {"schema"."table": [col1, col2, ...]}
        question: Original user question for context (optional)
        fallback_table: Fully-qualified table to use if FROM clause is missing

    Returns:
        Repaired, valid SQL string (PostgreSQL dialect)
    """
    original_input = str(raw_llm_output)[:200]

    try:
        # Stage 1: Extract SQL from LLM wrapper formats
        sql = _extract_sql_from_llm_output(raw_llm_output)
        logger.debug(f"[Stage 1] Extracted SQL length: {len(sql)}")

        if not sql or len(sql.strip()) < 10:
            logger.warning("[Stage 1] Empty or too short SQL, generating fallback")
            return _generate_safe_fallback_sql(schema_catalog, fallback_table)

        # Stage 2: Basic syntax cleanup (hardened)
        sql = _cleanup_basic_syntax_FINAL(sql)
        logger.debug(f"[Stage 2] After basic cleanup: {sql[:200]}")

        # Stage 3: Build catalog indices
        catalog_idx = _build_catalog_indices(schema_catalog)
        logger.debug(f"[Stage 3] Catalog has {len(catalog_idx['fq_canon'])} tables")

        # Stage 4: Fix table references
        sql = _repair_table_references(sql, catalog_idx)
        logger.debug(f"[Stage 4] After table repair: {sql[:200]}")

        # Stage 5: Fix column references
        sql = _repair_column_references(sql, catalog_idx)
        logger.debug(f"[Stage 5] After column repair: {sql[:200]}")

        # Stage 6: Fix date/interval literals AND date column types
        sql = _repair_date_and_interval_literals_FINAL(sql)
        sql = _repair_date_columns_FINAL(sql, schema_catalog)
        logger.debug(f"[Stage 6] After date/interval repair: {sql[:200]}")

        # Stage 7: Add safety guards & dynamic numeric casting
        sql = _add_safety_guards_FINAL(sql, schema_catalog, catalog_idx)
        logger.debug(f"[Stage 7] After safety guards: {sql[:200]}")

        # Stage 8: Ensure FROM clause exists
        sql = _ensure_from_clause(sql, schema_catalog, fallback_table)
        logger.debug(f"[Stage 8] After FROM check: {sql[:200]}")

        # Stage 9: Final validation & balancing
        sql = _final_validation_FINAL(sql, schema_catalog)
        logger.debug(f"[Stage 9] Final SQL: {sql[:300]}")

        return sql

    except Exception as e:
        logger.error(f"SQL repair failed: {e}. Input was: {original_input}")
        return _generate_safe_fallback_sql(schema_catalog, fallback_table)


# =========================
# Stage 1: Extract
# =========================


import re
import logging
logger = logging.getLogger(__name__)
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# --------- helpers for sanitizing ORDER/GROUP ---------

_AGG_RE = re.compile(r'\b(SUM|AVG|COUNT|MIN|MAX)\s*\(', re.I)

def _mask_commas_outside_parens_and_quotes(text: str) -> str:
    """Return text with commas inside (...) or quoted strings masked as \x00."""
    buf = []
    depth = 0
    in_s = False
    in_d = False
    i = 0
    while i < len(text):
        ch = text[i]

        # toggle quotes (naive but robust enough for SQL lists)
        if ch == "'" and not in_d:
            in_s = not in_s
            buf.append(ch)
            i += 1
            continue
        if ch == '"' and not in_s:
            in_d = not in_d
            buf.append(ch)
            i += 1
            continue

        if not in_s and not in_d:
            if ch == '(':
                depth += 1
                buf.append(ch)
                i += 1
                continue
            if ch == ')':
                depth = max(0, depth - 1)
                buf.append(ch)
                i += 1
                continue
            if ch == ',' and depth > 0:
                buf.append('\x00')  # mask comma inside parens
                i += 1
                continue

        buf.append(ch)
        i += 1

    return ''.join(buf)

def _count_select_items(sql: str) -> int:
    """
    Very tolerant counter for SELECT items before the first FROM.
    Handles commas inside parentheses and quoted strings.
    """
    m = re.search(r'^\s*SELECT\s+(.*?)\s+FROM\b', sql, flags=re.I | re.S)
    if not m:
        return 0
    segment = m.group(1)
    cleaned = _mask_commas_outside_parens_and_quotes(segment)
    items = [p.strip() for p in cleaned.split(',') if p.strip()]
    return len(items)

def _has_aggregates(sql: str) -> bool:
    return bool(_AGG_RE.search(sql))

def _has_alias_in_select(sql: str, alias: str) -> bool:
    m = re.search(r'^\s*SELECT\s+(.*?)\s+FROM\b', sql, flags=re.I | re.S)
    if not m:
        return False
    sel = m.group(1)
    # look for "AS alias" or trailing alias on expression
    return bool(re.search(rf'\bAS\s+{re.escape(alias)}\b|\b{re.escape(alias)}\b', sel, flags=re.I))

def _sanitize_group_order_clauses(sql: str) -> str:
    """
    Clean GROUP BY / ORDER BY:
      - Remove stray 20xx literals
      - Keep positional ORDER BY n only if n <= select_items
      - Never glue tokens (always preserve spaces)
      - If GROUP BY becomes empty but aggregates exist, restore a safe group:
          prefer GROUP BY month (if alias exists) else GROUP BY 1
      - If ORDER BY becomes empty and month alias exists, ORDER BY month
    """
    select_count = _count_select_items(sql)
    has_aggs = _has_aggregates(sql)
    has_month_alias = _has_alias_in_select(sql, 'month')

    def _split_list(lst: str) -> list[str]:
        masked = _mask_commas_outside_parens_and_quotes(lst)
        parts = [p.strip() for p in masked.split(',')]
        return [p.replace('\x00', ',').strip() for p in parts if p.strip()]

    def _scrub_list(lst: str, *, allow_positions: bool) -> str:
        parts = _split_list(lst)
        cleaned = []
        for p in parts:
            # drop bare 4-digit years
            if re.fullmatch(r'20\d{2}', p):
                continue
            if allow_positions and re.fullmatch(r'\d+', p):
                pos = int(p)
                # keep only valid positional indexes
                if 1 <= pos <= max(1, select_count):
                    cleaned.append(p)
                continue
            cleaned.append(p)
        return ', '.join(cleaned)

    # --- GROUP BY ---
    def _fix_group(m):
        inner = m.group(1)
        fixed = _scrub_list(inner, allow_positions=False)
        if fixed:
            return ' GROUP BY ' + fixed + ' '
        # empty after scrub:
        if has_aggs:
            if has_month_alias:
                return ' GROUP BY month '
            return ' GROUP BY 1 ' if select_count >= 1 else ' '
        return ' '

    # --- ORDER BY ---
    def _fix_order(m):
        inner = m.group(1)
        fixed = _scrub_list(inner, allow_positions=True)
        if fixed:
            return ' ORDER BY ' + fixed + ' '
        if has_month_alias:
            return ' ORDER BY month '
        return ' '

    # apply with safe spacing so we never glue tokens
    sql = re.sub(r'\bGROUP\s+BY\s+(.*?)(?=\bORDER\b|\bLIMIT\b|$)', _fix_group, sql, flags=re.I | re.S)
    sql = re.sub(r'\bORDER\s+BY\s+(.*?)(?=\bLIMIT\b|$)', _fix_order, sql, flags=re.I | re.S)

    # compact whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    return sql

# --------- robust extractor (drop-in replacement) ---------

def _extract_sql_from_llm_output11(raw: Any) -> str:
    """
    Extract SQL from common LLM wrappers (strings, tuples, dicts, fenced blocks).
    Keeps content starting from first WITH/SELECT and strips model header tokens.
    """
    if raw is None:
        return ""

    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, (tuple, list)):
        text = ""
        for item in raw:
            if isinstance(item, str) and item.strip():
                text = item
                break
        if not text:
            text = " ".join(str(x) for x in raw if x is not None)
    elif isinstance(raw, dict):
        text = ""
        for key in ("sql", "text", "content", "message", "output"):
            v = raw.get(key)
            if isinstance(v, str) and v.strip():
                text = v
                break
        if not text:
            text = str(raw)
    else:
        text = str(raw)

    if not text or not text.strip():
        return ""

    # remove any header tokens certain models add
    text = re.sub(r'<\|header_start\|>\s*sql', '', text, flags=re.I)
    text = re.sub(r'<\|header_end\|>', '', text, flags=re.I)

    # prefer fenced SQL block if present
    m = re.search(r'```(?:sql)?\s*(.*?)\s*```', text, flags=re.I | re.DOTALL)
    if m:
        text = m.group(1)

    # drop accidental "sql\n" prefix
    text = re.sub(r'^\s*sql\s+', '', text, flags=re.I)

    # keep from first WITH/SELECT onward
    m = re.search(r'\b(WITH|SELECT)\b.*', text, flags=re.I | re.DOTALL)
    if m:
        text = m.group(0)

    # normalize simple duplicates that can break clause parsing
    text = re.sub(r"\bDATE\s+DATE\s+(')", r"DATE \1", text, flags=re.I)
    text = re.sub(r"(INTERVAL\s*'\s*\d+\s*)month\b", r"\1months", text, flags=re.I)

    # final trim
    return re.sub(r'\s+', ' ', text).strip()



# =========================
# Stage 2: Cleanup (FINAL)
# =========================

def _cleanup_basic_syntax_FINAL11(sql: str) -> str:
    # remove comments
    sql = re.sub(r'--[^\n]*', '', sql)

    # de-dupe keywords
    for kw in ['SELECT','FROM','WHERE','GROUP BY','ORDER BY','HAVING','WHEN','END','CASE']:
        sql = re.sub(rf'\b({kw})\s+\1\b', r'\1', sql, flags=re.I)

    # duplicated DATE and silly year self-comparisons
    sql = re.sub(r"\bDATE\s+DATE\s+(')", r"DATE \1", sql, flags=re.I)
    sql = re.sub(r'\s+(AND|OR)\s+\(?\s*(20\d{2})\s*=\s*\2\s*\)?(?=\s+(AND|OR|$))', ' ', sql, flags=re.I)
    sql = re.sub(r'\bWHERE\s+\(?\s*(20\d{2})\s*=\s*\1\s*\)?\s+(AND|OR)\s+', 'WHERE ', sql, flags=re.I)

    # ROUND( ... ) AS alias → ensure closing paren
    sql = re.sub(r'\bROUND\s*\(([^)]*)\s+AS\b', r'ROUND(\1) AS', sql, flags=re.I)

    # close NULLIF aggregate before AS; ensure ", 0)" for guards
    sql = re.sub(
        r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s+AS\b',
        r'NULLIF(\1(\2)\3, 0) AS', sql, flags=re.I
    )
    sql = re.sub(
        r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s*\)\s*/',
        r'NULLIF(\1(\2)\3, 0) /', sql, flags=re.I
    )

    # strip bare years from GROUP BY / ORDER BY lists
    sql = re.sub(r'\bGROUP\s+BY\s*(.*?)(?=\bORDER\b|\bLIMIT\b|$)',
                 lambda m: 'GROUP BY ' + re.sub(r'(?:^|,)\s*20\d{2}\s*(?=,|$)', ',', m.group(1)).strip(' ,'),
                 sql, flags=re.I | re.S)
    sql = re.sub(r'\bORDER\s+BY\s*(.*?)(?=\bLIMIT\b|$)',
                 lambda m: 'ORDER BY ' + re.sub(r'(?:^|,)\s*20\d{2}\s*(?=,|$)', ',', m.group(1)).strip(' ,'),
                 sql, flags=re.I | re.S)

    # misc normalizations
    sql = re.sub(r'\bGROUP\s+BY\s+BY\b','GROUP BY',sql,flags=re.I)
    sql = re.sub(r'\bORDER\s+BY\s+BY\b','ORDER BY',sql,flags=re.I)
    sql = re.sub(r'\bEXTRACTRACT\b','EXTRACT',sql,flags=re.I)
    sql = re.sub(r'\bYYEAR\b','YEAR',sql,flags=re.I)
    sql = re.sub(r'\bMMONTH\b','MONTH',sql,flags=re.I)
    sql = re.sub(r'""+','"',sql)
    sql = re.sub(r'"\s*\.\s*"', '"."', sql)
    sql = re.sub(r'\.\s*\.', '.', sql)
    sql = re.sub(r'"([A-Za-z_]\w*)"\s*"\s*([A-Za-z_]\w*)"', r'"\1"."\2"', sql)
    sql = re.sub(r',\s*(WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|FROM)\b', r' \1', sql, flags=re.I)
    sql = re.sub(r'>\s*>', '>', sql); sql = re.sub(r'<\s*<', '<', sql); sql = re.sub(r'=\s*=', '=', sql)
    sql = re.sub(r"(INTERVAL\s*'\s*\d+\s*)month\b", r"\1months", sql, flags=re.I)

    return re.sub(r'\s+', ' ', sql).strip()


# =========================
# Stage 3: Catalog indices
# =========================

def _build_catalog_indices1(schema_catalog: Dict[str, List[str]]) -> Dict[str, Any]:
    fq_canon: Dict[str, str] = {}
    schemas: Set[str] = set()
    tables_by_schema: Dict[str, Set[str]] = {}
    cols_by_fq: Dict[str, Set[str]] = {}
    all_cols: Set[str] = set()
    col_to_tables: Dict[str, List[str]] = {}

    for fq, cols in (schema_catalog or {}).items():
        parts = fq.replace('"', '').split('.')
        if len(parts) != 2:
            continue
        s, t = parts
        key = f"{s.lower()}.{t.lower()}"
        fq_canon[key] = f'"{s}"."{t}"'
        schemas.add(s)
        tables_by_schema.setdefault(s.lower(), set()).add(t)
        cols_by_fq[key] = {c.lower() for c in cols}
        for c in cols:
            c_lower = c.lower()
            all_cols.add(c_lower)
            col_to_tables.setdefault(c_lower, []).append(key)

    return {
        'fq_canon': fq_canon,
        'schemas': sorted(schemas),
        'tables_by_schema': {k: sorted(v) for k, v in tables_by_schema.items()},
        'cols_by_fq': cols_by_fq,
        'all_cols': all_cols,
        'col_to_tables': col_to_tables
    }


# =========================
# Stage 4: Tables
# =========================

def _repair_table_references(sql: str, catalog_idx: Dict[str, Any]) -> str:
    fq_canon = catalog_idx['fq_canon']
    pattern = r'("?[A-Za-z_][\w$]*"?)\s*\.\s*("?[A-Za-z_][\w$]*"?)'

    def replace(m):
        s_raw, t_raw = m.group(1), m.group(2)
        s = s_raw.replace('"', '').strip().lower()
        t = t_raw.replace('"', '').strip().lower()
        key = f"{s}.{t}"
        if key in fq_canon:
            return fq_canon[key]
        matches = get_close_matches(key, list(fq_canon.keys()), n=1, cutoff=0.75)
        if matches:
            return fq_canon[matches[0]]
        return f'"{s}"."{t}"'

    return re.sub(pattern, replace, sql)


# =========================
# Stage 5: Columns
# =========================

def _repair_column_references(sql: str, catalog_idx: Dict[str, Any]) -> str:
    cols_by_fq = catalog_idx['cols_by_fq']
    used_tables = _extract_used_tables(sql)
    if not used_tables:
        return sql
    alias_map = _extract_alias_map(sql, used_tables)
    if not alias_map:
        return sql

    qualifiers = '|'.join(re.escape(q) for q in sorted(alias_map.keys(), key=len, reverse=True))
    pattern = rf'\b({qualifiers})\s*\.\s*"?([A-Za-z_][\w$]*)"?'

    def replace(m):
        qual, col = m.group(1).lower(), m.group(2)
        fq = alias_map.get(qual)
        if not fq:
            return m.group(0)
        cols = cols_by_fq.get(fq, set())
        if not cols:
            return m.group(0)
        if col.lower() in cols:
            return f'{qual}.{col}'
        matches = get_close_matches(col.lower(), list(cols), n=1, cutoff=0.75)
        if matches:
            return f'{qual}.{matches[0]}'
        return m.group(0)

    return re.sub(pattern, replace, sql, flags=re.I)


def _extract_used_tables(sql: str) -> Set[str]:
    pattern = r'\b(FROM|JOIN)\s+("?[A-Za-z_][\w$]*"?\s*\.\s*"?[A-Za-z_][\w$]*"?)'
    matches = re.findall(pattern, sql, flags=re.I)
    tables: Set[str] = set()
    for _, fq in matches:
        fq_norm = fq.replace('"', '').replace(' ', '').lower()
        tables.add(fq_norm)
    return tables


def _extract_alias_map(sql: str, used_tables: Set[str]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    pattern = r'\b(FROM|JOIN)\s+("?[A-Za-z_][\w$]*"?\s*\.\s*"?[A-Za-z_][\w$]*"?)(?:\s+AS)?\s+([A-Za-z_][\w$]+)?'
    for m in re.finditer(pattern, sql, flags=re.I):
        fq = m.group(2).replace('"', '').replace(' ', '').lower()
        alias = m.group(3)
        if alias:
            alias_map[alias.lower()] = fq
    for fq in used_tables:
        if '.' in fq:
            _, tbl = fq.split('.', 1)
            alias_map[tbl] = fq
    return alias_map


# =========================
# Stage 6: Date & Interval (FINAL)
# =========================

def _repair_date_and_interval_literals_FINA11L(sql: str) -> str:
    sql = re.sub(
        r'\bINTERVAL\s*([0-9]+)\s*(year|month|day|hour|minute|second)s?\b',
        lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}s'", sql, flags=re.I
    )
    sql = re.sub(
        r'\b([0-9]+)(days?|months?|years?)\b',
        lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}'", sql, flags=re.I
    )
    sql = re.sub(r"'(\d{4}-\d{2}-\d{2})'\s*'(\d{4}-\d{2}-\d{2})'", r"'\1'", sql)
    sql = re.sub(r"'(\d{4}-\d{2}-\d{2})'\s*([+-])\s*INTERVAL", r"DATE '\1' \2 INTERVAL", sql, flags=re.I)
    return sql


def _repair_date_columns_FINAL(sql: str, schema_catalog: Dict[str, List[str]]) -> str:
    # gather date-ish columns by name
    date_cols: Set[str] = set()
    for _, cols in (schema_catalog or {}).items():
        for c in cols:
            lc = c.lower()
            if any(x in lc for x in ['date','time','created','updated','funded','purchase','report','maturity','expiration']):
                date_cols.add(lc)
    if not date_cols:
        return sql

    # EXTRACT(unit FROM col) → CAST(col AS DATE)
    def fix_extract(m):
        unit, col = m.group(1), m.group(2).strip()
        base = col.split('.')[-1].replace('"','').lower()
        if base in date_cols:
            return f"EXTRACT({unit} FROM CAST({col} AS DATE))"
        return m.group(0)

    sql = re.sub(r'EXTRACT\s*\(\s*(\w+)\s+FROM\s+(["\w.]+)\s*\)', fix_extract, sql, flags=re.I)

    # date arithmetic col ± INTERVAL/DATE → CAST(col AS DATE) ± ...
    for base in date_cols:
        sql = re.sub(
            rf'([\w."]+\.{base})(?!\s*(?:::DATE|AS\s+DATE|\)))\s*([+-])\s*(INTERVAL|DATE)',
            lambda m: f"CAST({m.group(1)} AS DATE) {m.group(2)} {m.group(3)}",
            sql, flags=re.I
        )
    return sql


# =========================
# Stage 7: Safety Guards (FINAL)
# =========================

# def _add_safety_guards_FINAL(sql: str, schema_catalog: Dict[str, List[str]], catalog_idx: Dict[str, Any]) -> str:
#     # open NULLIF before aggregates in divisions
#     sql = re.sub(r'/\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(', r'/ NULLIF(\1(', sql, flags=re.I)
#     # ensure ', 0)' gets injected (covers casts too)
#     sql = re.sub(r'NULLIF\((COUNT|SUM|AVG|MIN|MAX)\(([^)]*)\)\s*(::\w+)?\)',
#                  r'NULLIF(\1(\2)\3, 0)', sql, flags=re.I)

#     # dynamic numeric casting for likely TEXT numerics used in numeric context
#     sql = _cast_text_columns_to_numeric(sql, schema_catalog)
#     return sql


def _cast_text_columns_to_numeric(sql: str, schema_catalog: Dict[str, List[str]]) -> str:
    # Skip if already processed
    if 'REGEXP_REPLACE' in sql:
        return sql

    # Heuristic: names suggesting numeric content
    patterns = [
        'amount','price','cost','fee','rate','discount','charge','payment','balance','total','value',
        'income','salary','revenue','profit','loss','score','count','quantity','volume','sum',
        'offs','principal','apr','interest','finance','premium','deductible','commission'
    ]

    candidates: List[Tuple[str, str]] = []
    for fq_table, columns in (schema_catalog or {}).items():
        for col in columns:
            if any(p in col.lower() for p in patterns):
                candidates.append((fq_table, col))

    if not candidates:
        return sql

    for fq_table, col_name in candidates:
        if not _is_column_in_numeric_context(sql, col_name):
            continue
        safe_cast_template = "NULLIF(REGEXP_REPLACE({col}, '[^0-9\\.-]', '', 'g'), '')::NUMERIC"
        sql = _replace_column_in_aggregates(sql, fq_table, col_name, safe_cast_template)

    return sql


def _is_column_in_numeric_context(sql: str, col_name: str) -> bool:
    col_escaped = re.escape(col_name)
    if re.search(rf'\b(SUM|AVG|COUNT|MIN|MAX)\s*\([^)]*\b{col_escaped}\b', sql, flags=re.I):
        return True
    if re.search(rf'\b{col_escaped}\b\s*[+\-*/]', sql, flags=re.I):
        return True
    if re.search(rf'\b{col_escaped}\b\s*[<>=].*?\d', sql, flags=re.I):
        return True
    return False


def _replace_column_in_aggregates(sql: str, fq_table: str, col_name: str, cast_template: str) -> str:
    table_parts = fq_table.replace('"', '').split('.')
    schema_name = table_parts[0] if len(table_parts) > 1 else ''
    table_name = table_parts[-1] if table_parts else ''
    col_escaped = re.escape(col_name)

    patterns = [
        (rf'({re.escape(fq_table)}\."{col_escaped}")\b', 'fq_quoted_col'),
        (rf'({re.escape(fq_table)}\.{col_escaped})\b', 'fq_unquoted_col'),
        (rf'({re.escape(schema_name)}\.{re.escape(table_name)}\.{col_escaped})\b', 'schema_table_col'),
        (rf'({re.escape(table_name)}\.{col_escaped})\b', 'table_col'),
        (rf'\b(SUM|AVG|COUNT|MIN|MAX)\s*\(\s*("?{col_escaped}"?)\s*\)', 'bare_in_agg'),
    ]

    for pattern, kind in patterns:
        def repl(m):
            if kind == 'bare_in_agg':
                func = m.group(1)
                col_ref = f"{fq_table}.{col_name}"
                casted = cast_template.replace('{col}', col_ref)
                return f"{func}({casted})"
            col_ref = m.group(1)
            return cast_template.replace('{col}', col_ref)

        sql = re.sub(pattern, repl, sql, flags=re.I)

    return sql


# =========================
# Stage 8: Ensure FROM
# =========================

def _ensure_from_clause(sql: str, schema_catalog: Dict[str, List[str]], fallback_table: str | None) -> str:
    if re.search(r'\bFROM\b', sql, flags=re.I):
        return sql
    fq = fallback_table
    if not fq and schema_catalog:
        fq = next(iter(schema_catalog))
    if not fq:
        return sql
    return re.sub(
        r'(?i)^(SELECT\b.*?)\s*(WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|$)',
        lambda m: f'{m.group(1)} FROM {fq} {m.group(2)}' if m.group(2) else f'{m.group(1)} FROM {fq}',
        sql,
        count=1
    )


# =========================
# Stage 9: Final validation (FINAL)
# =========================

def _final_validation_FINAL1(sql: str, schema_catalog: Dict[str, List[str]]) -> str:
    sql = re.sub(r'\s+', ' ', sql).strip()

    if not re.search(r'\bSELECT\b.*\bFROM\b', sql, flags=re.I | re.S):
        logger.warning("[Final] SQL missing SELECT...FROM structure")
        return _generate_safe_fallback_sql(schema_catalog)

    # balance parentheses
    opens, closes = sql.count('('), sql.count(')')
    if opens != closes:
        diff = opens - closes
        if 0 < diff <= 3:
            sql += ')' * diff
        elif diff < 0 and -diff <= 3:
            sql = '(' * (-diff) + sql
        else:
            logger.error("[Final] Too many unbalanced parens, generating fallback")
            return _generate_safe_fallback_sql(schema_catalog)

    # trim funky trailing ');)'
    sql = re.sub(r';\s*\)+\s*$', ';', sql)
    return sql


# =========================
# Fallback
# =========================

def _generate_safe_fallback_sql(schema_catalog: Dict[str, List[str]], fallback_table: str | None = None) -> str:
    fq = fallback_table
    if not fq and schema_catalog:
        fq = next(iter(schema_catalog))
    if not fq:
        return "SELECT 1 AS status, 'No tables available' AS message;"
    return f"SELECT COUNT(*) AS row_count FROM {fq} LIMIT 1;"


def _extract_sql_from_llm_output1(raw: Any) -> str:
    """Handle tuple/list/dict/str formats and remove markdown fences."""
    if raw is None:
        return ""
    
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, (tuple, list)):
        # first non-empty string item else join printable parts
        text = ""
        for item in raw:
            if isinstance(item, str) and item.strip():
                text = item
                break
        if not text:
            text = " ".join(str(x) for x in raw if x is not None)
    elif isinstance(raw, dict):
        for key in ("sql", "text", "content", "message", "output"):
            if key in raw and isinstance(raw[key], str) and raw[key].strip():
                text = raw[key]
                break
        else:
            text = str(raw)
    else:
        text = str(raw)
    
    if not text or not text.strip():
        return ""
    
    # strip model header tokens
    text = re.sub(r'<\|header_start\|>\s*sql', '', text, flags=re.I)
    text = re.sub(r'<\|header_end\|>', '', text, flags=re.I)
    
    # prefer fenced SQL block if present
    m = re.search(r'```(?:sql)?\s*(.*?)\s*```', text, flags=re.I | re.DOTALL)
    if m:
        text = m.group(1)
    
    # drop accidental "sql\n" prefix
    text = re.sub(r'^\s*sql\s+', '', text, flags=re.I)
    
    # keep from first WITH/SELECT onward
    m = re.search(r'\b(WITH|SELECT)\b.*', text, flags=re.I | re.DOTALL)
    if m:
        text = m.group(0)
    
    return text.strip()

# ------------------------------------------------------------
# Notes:
# - Keep your existing implementations of:
#   _build_catalog_indices, _repair_table_references,
#   _repair_column_references, _ensure_from_clause,
#   _cast_text_columns_to_numeric, _generate_safe_fallback_sql
# - We only *call* the new *_FINAL helpers to avoid shadowing bugs.
# ------------------------------------------------------------


# ============================================================================
# STAGE 2: Basic syntax cleanup
# ============================================================================
import re
import re
# Optional: enable logging in your module
# import logging
# logger = logging.getLogger(__name__)
import re
# Optional:
# import logging
# logger = logging.getLogger(__name__)
import re
# Optional:
# import logging
# logger = logging.getLogger(__name__)

def _cleanup_basic_syntax(sql: str) -> str:
    """Fix common tokenization glitches before semantic repairs."""

    # Remove SQL comments
    sql = re.sub(r'--[^\n]*', '', sql)

    # Fix duplicated keywords (SELECT SELECT -> SELECT)
    for kw in ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'WHEN', 'END', 'CASE']:
        pattern = rf'\b({kw})\s+\1\b'
        sql = re.sub(pattern, r'\1', sql, flags=re.I)

    # 1) Fix duplicated DATE before a literal
    sql = re.sub(r"\bDATE\s+DATE\s+(')", r"DATE \1", sql, flags=re.I)

    # 2) Remove trivial constant year equalities (AND/OR forms)
    sql = re.sub(r'\s+(AND|OR)\s+\(?\s*(20\d{2})\s*=\s*\2\s*\)?(?=\s+(AND|OR|$))', ' ', sql, flags=re.I)
    sql = re.sub(r'\bWHERE\s+\(?\s*(20\d{2})\s*=\s*\1\s*\)?\s+(AND|OR)\s+', 'WHERE ', sql, flags=re.I)

    # 3) Ensure NULLIF(COUNT(...)) has a zero divisor (common shapes)
    sql = re.sub(r'NULLIF\s*\(\s*COUNT\s*\(([^)]*)\)\s*\)\s*/',  r'NULLIF(COUNT(\1), 0) /', sql, flags=re.I)
    sql = re.sub(r'NULLIF\s*\(\s*COUNT\s*\(([^)]*)\)\s*\)\s+AS\b', r'NULLIF(COUNT(\1), 0) AS', sql, flags=re.I)

    # 3a) Cast-aware + open-NULLIF cases
    sql = re.sub(r'NULLIF\s*\(\s*COUNT\s*\(([^)]*)\)\s*(::\w+)?\s+AS\b',
                 r'NULLIF(COUNT(\1)\2, 0) AS', sql, flags=re.I)
    sql = re.sub(r'NULLIF\s*\(\s*COUNT\s*\(([^)]*)\)\s*(::\w+)?\s*\)\s*/',
                 r'NULLIF(COUNT(\1)\2, 0) /', sql, flags=re.I)
    sql = re.sub(r'NULLIF\s*\(\s*COUNT\s*\(([^)]*)\)\s*(::\w+)?\s*\)\s+(?=AS\b)',
                 r'NULLIF(COUNT(\1)\2, 0) ', sql, flags=re.I)

    # 3b) Hallucinated non-zero denominators in NULLIF(COUNT(...), N) → 0  (cast-aware)
    sql = re.sub(r'NULLIF\s*\(\s*COUNT\s*\(([^)]*)\)\s*(::\w+)?\s*,\s*(?!0\b)(\d+)\s*\)',
                 r'NULLIF(COUNT(\1)\2, 0)', sql, flags=re.I)

    # 3c) ROUND missing closing paren before AS (your current error)
    #     Example: ROUND( expr ... AS alias  →  ROUND( expr ... ) AS alias
    sql = re.sub(r'\bROUND\s*\(([^)]*)\s+AS\b', r'ROUND(\1) AS', sql, flags=re.I)

    # 4) Drop stray bare year literals in GROUP BY / ORDER BY (not positions)
    sql = re.sub(r'\bGROUP\s+BY\s*(?:,?\s*)*(?:20\d{2})(?=(?:\s*,|\s*$))', 'GROUP BY ', sql, flags=re.I)
    sql = re.sub(r'\bORDER\s+BY\s*(?:,?\s*)*(?:20\d{2})(?=(?:\s*,|\s*$))', 'ORDER BY ', sql, flags=re.I)

    # 5) Trim accidental extra closing paren after semicolon: "...;)"
    sql = re.sub(r';\s*\)+\s*$', ';', sql)

    # Fix compound keyword duplications
    sql = re.sub(r'\bGROUP\s+BY\s+BY\b', 'GROUP BY', sql, flags=re.I)
    sql = re.sub(r'\bORDER\s+BY\s+BY\b', 'ORDER BY', sql, flags=re.I)

    # Fix function name glitches
    sql = re.sub(r'\bEXTRACTRACT\b', 'EXTRACT', sql, flags=re.I)
    sql = re.sub(r'\bYYEAR\b', 'YEAR', sql, flags=re.I)
    sql = re.sub(r'\bMMONTH\b', 'MONTH', sql, flags=re.I)

    # Collapse stacked/broken quotes
    sql = re.sub(r'""+', '"', sql)
    sql = re.sub(r'"\s*\.\s*"', '"."', sql)
    sql = re.sub(r'\.\s*\.', '.', sql)

    # Fix missing dots between quoted identifiers: "schema""table" -> "schema"."table"
    sql = re.sub(r'"([A-Za-z_]\w*)"\s*"\s*([A-Za-z_]\w*)"', r'"\1"."\2"', sql)

    # Remove dangling commas before clauses
    sql = re.sub(r',\s*(WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|FROM)\b', r' \1', sql, flags=re.I)

    # Fix duplicate comparison operators
    sql = re.sub(r'>\s*>', '>', sql)
    sql = re.sub(r'<\s*<', '<', sql)
    sql = re.sub(r'=\s*=', '=', sql)

    # Remove illogical year comparisons entirely
    sql = re.sub(r'\bAND\s+\d{4}\s*=\s*\d{4}\b', '', sql, flags=re.I)
    sql = re.sub(r'\bWHERE\s+\d{4}\s*=\s*\d{4}\s+AND\b', 'WHERE', sql, flags=re.I)

    # Optional: normalize INTERVAL pluralization (INTERVAL '12 month' -> '12 months')
    sql = re.sub(r"(INTERVAL\s*'\s*\d+\s*)month\b", r"\1months", sql, flags=re.I)

    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()

    return sql


# ============================================================================
# STAGE 3: Build catalog indices
# ============================================================================

def _build_catalog_indice11s(schema_catalog: dict) -> dict:
    """Build lookup structures for fast column/table resolution."""
    
    fq_canon = {}        # normalized fq -> canonical quoted fq
    schemas = set()
    tables_by_schema = {}
    cols_by_fq = {}      # normalized fq -> set of lowercase column names
    all_cols = set()
    col_to_tables = {}   # column name -> list of tables that have it
    
    for fq, cols in (schema_catalog or {}).items():
        # Parse "schema"."table"
        parts = fq.replace('"', '').split('.')
        if len(parts) != 2:
            continue
        
        s, t = parts
        key = f"{s.lower()}.{t.lower()}"
        
        fq_canon[key] = f'"{s}"."{t}"'
        schemas.add(s)
        tables_by_schema.setdefault(s.lower(), set()).add(t)
        cols_by_fq[key] = {c.lower() for c in cols}
        
        for c in cols:
            c_lower = c.lower()
            all_cols.add(c_lower)
            col_to_tables.setdefault(c_lower, []).append(key)
    
    return {
        'fq_canon': fq_canon,
        'schemas': sorted(schemas),
        'tables_by_schema': {k: sorted(v) for k, v in tables_by_schema.items()},
        'cols_by_fq': cols_by_fq,
        'all_cols': all_cols,
        'col_to_tables': col_to_tables
    }


# ============================================================================
# STAGE 4: Repair table references
# ============================================================================

def _repair_table_references11(sql: str, catalog_idx: dict) -> str:
    """Snap schema.table references to canonical forms from catalog."""
    
    fq_canon = catalog_idx['fq_canon']
    
    # Pattern: "schema"."table" or schema.table (possibly with broken quotes/spaces)
    pattern = r'("?[A-Za-z_][\w$]*"?)\s*\.\s*("?[A-Za-z_][\w$]*"?)'
    
    def replace(m):
        s_raw, t_raw = m.group(1), m.group(2)
        s = s_raw.replace('"', '').strip().lower()
        t = t_raw.replace('"', '').strip().lower()
        key = f"{s}.{t}"
        
        # Exact match
        if key in fq_canon:
            return fq_canon[key]
        
        # Fuzzy match
        candidates = list(fq_canon.keys())
        matches = get_close_matches(key, candidates, n=1, cutoff=0.75)
        if matches:
            return fq_canon[matches[0]]
        
        # Return quoted form of what we have
        return f'"{s}"."{t}"'
    
    return re.sub(pattern, replace, sql)


# ============================================================================
# STAGE 5: Repair column references
# ============================================================================

def _repair_column_references11(sql: str, catalog_idx: dict) -> str:
    """Fix misspelled/missing columns based on catalog."""
    
    cols_by_fq = catalog_idx['cols_by_fq']
    
    # Extract tables used in FROM/JOIN
    used_tables = _extract_used_tables(sql)
    if not used_tables:
        return sql
    
    # Map aliases to tables
    alias_map = _extract_alias_map(sql, used_tables)
    if not alias_map:
        return sql
    
    # Pattern: qualifier.column
    qualifiers = '|'.join(re.escape(q) for q in sorted(alias_map.keys(), key=len, reverse=True))
    pattern = rf'\b({qualifiers})\s*\.\s*"?([A-Za-z_][\w$]*)"?'
    
    def replace(m):
        qual, col = m.group(1).lower(), m.group(2)
        fq = alias_map.get(qual)
        if not fq:
            return m.group(0)
        
        cols = cols_by_fq.get(fq, set())
        if not cols:
            return m.group(0)
        
        # Exact match
        if col.lower() in cols:
            return f'{qual}.{col}'
        
        # Fuzzy match
        matches = get_close_matches(col.lower(), list(cols), n=1, cutoff=0.75)
        if matches:
            return f'{qual}.{matches[0]}'
        
        # No match - return original
        return m.group(0)
    
    return re.sub(pattern, replace, sql, flags=re.I)


def _extract_used_tables(sql: str) -> set:
    """Extract fully-qualified table names from FROM/JOIN clauses."""
    pattern = r'\b(FROM|JOIN)\s+("?[A-Za-z_][\w$]*"?\s*\.\s*"?[A-Za-z_][\w$]*"?)'
    matches = re.findall(pattern, sql, flags=re.I)
    tables = set()
    for _, fq in matches:
        fq_norm = fq.replace('"', '').replace(' ', '').lower()
        tables.add(fq_norm)
    return tables


def _extract_alias_map(sql: str, used_tables: set) -> dict:
    """Build {alias: normalized_fq} mapping."""
    alias_map = {}
    
    # FROM/JOIN table AS alias
    pattern = r'\b(FROM|JOIN)\s+("?[A-Za-z_][\w$]*"?\s*\.\s*"?[A-Za-z_][\w$]*"?)(?:\s+AS)?\s+([A-Za-z_][\w$]+)?'
    for m in re.finditer(pattern, sql, flags=re.I):
        fq = m.group(2).replace('"', '').replace(' ', '').lower()
        alias = m.group(3)
        if alias:
            alias_map[alias.lower()] = fq
    
    # Also allow unaliased table name as qualifier
    for fq in used_tables:
        if '.' in fq:
            _, tbl = fq.split('.', 1)
            alias_map[tbl] = fq
    
    return alias_map


# ============================================================================
# STAGE 6: Fix date/interval literals
# ============================================================================

def _repair_date_and_interval_literals(sql: str) -> str:
    """Fix common date/interval syntax errors."""
    
    # Fix INTERVAL syntax: "INTERVAL6 months" or "183days" -> INTERVAL '6 months'
    sql = re.sub(
        r'\bINTERVAL\s*([0-9]+)\s*(year|month|day|hour|minute|second)s?\b',
        lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}s'",
        sql,
        flags=re.I
    )
    
    # Fix bare number + unit: "183days" -> INTERVAL '183 days'
    sql = re.sub(
        r'\b([0-9]+)(days?|months?|years?)\b',
        lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}'",
        sql,
        flags=re.I
    )
    
    # Fix duplicated date literals
    sql = re.sub(
        r"'(\d{4}-\d{2}-\d{2})'\s*'(\d{4}-\d{2}-\d{2})'",
        r"'\1'",
        sql
    )
    
    # Fix date arithmetic
    sql = re.sub(
        r"'(\d{4}-\d{2}-\d{2})'\s*([+-])\s*INTERVAL",
        r"DATE '\1' \2 INTERVAL",
        sql,
        flags=re.I
    )
    
    return sql

import re
# NOTE: assumes a logger is available in your module:
# import logging; logger = logging.getLogger(__name__)

def _repair_date_columns(sql: str, schema_catalog: dict) -> str:
    """Cast date columns to proper types before EXTRACT/date arithmetic.
    
    CRITICAL: Uses CAST() instead of TRIM() to handle both TEXT and TIMESTAMP columns.
    """
    
    # Find date-like columns from schema
    date_cols = set()
    for fq, cols in schema_catalog.items():
        for col in cols:
            col_lower = col.lower()
            if any(x in col_lower for x in ['date', 'time', 'created', 'updated', 'funded', 'purchase', 'report', 'maturity', 'expiration']):
                date_cols.add(col_lower)
                logger.debug(f"[Date Detector] Found date column: {fq}.{col}")
    
    if not date_cols:
        return sql
    
    # Pattern: EXTRACT(... FROM column) where column might be text
    # Pattern: EXTRACT(... FROM column)
    def fix_extract(match):
        unit, col = match.group(1), match.group(2).strip()
        col_base = col.split('.')[-1].replace('"', '').lower() if '.' in col else col.replace('"', '').lower()
        
        if col_base in date_cols:
            # ✅ FIX: Use CAST which works for TIMESTAMP, DATE, and TEXT
            # No TRIM or NULLIF needed - PostgreSQL handles conversion
            safe_cast = f"CAST({col} AS DATE)"
            logger.debug(f"[Date Cast] EXTRACT({unit} FROM {col}) -> EXTRACT({unit} FROM {safe_cast})")
            return f"EXTRACT({unit} FROM {safe_cast})"
        return match.group(0)
    
    sql = re.sub(
        r'EXTRACT\s*\(\s*(\w+)\s+FROM\s+(["\w.]+)\s*\)',
        fix_extract,
        sql,
        flags=re.IGNORECASE
    )
    
    # Also fix date arithmetic: column + INTERVAL
    # We keep BOTH pattern styles from your two snippets, without deleting anything.
    for col in date_cols:
        # --- Pattern version 1 (from the first snippet) ---
        # Avoid double-casting by checking for existing CAST or ::DATE
        pattern_v1 = rf'([\w."]+\.{col})(?!\s*(?:AS\s+DATE|::DATE|\)))\s*([+-])\s*(INTERVAL|DATE)'
        
        def fix_date_arith_v1(match):
            col_ref, op, keyword = match.group(1), match.group(2), match.group(3)
            safe_cast = f"CAST({col_ref} AS DATE)"
            logger.debug(f"[Date Arith v1] {col_ref} {op} {keyword} -> {safe_cast} {op} {keyword}")
            return f"{safe_cast} {op} {keyword}"
        
        sql = re.sub(pattern_v1, fix_date_arith_v1, sql, flags=re.IGNORECASE)
        
        # --- Pattern version 2 (from the second snippet) ---
        # Match schema.table.column or table.column patterns (avoid double-casting)
        pattern_v2 = rf'([\w."]+\.{col})(?!::)(?!\s*AS\s+DATE)\s*([+-])\s*(INTERVAL|DATE)'
        
        def fix_date_arith_v2(match):
            col_ref, op, keyword = match.group(1), match.group(2), match.group(3)
            # ✅ Use CAST instead of ::DATE with TRIM
            safe_cast = f"CAST({col_ref} AS DATE)"
            logger.debug(f"[Date Arith v2] {col_ref} {op} {keyword} -> {safe_cast} {op} {keyword}")
            return f"{safe_cast} {op} {keyword}"
        
        sql = re.sub(pattern_v2, fix_date_arith_v2, sql, flags=re.IGNORECASE)
    
    return sql

# ============================================================================
# STAGE 7: Add safety guards
# ============================================================================

def _add_safety_guards(sql: str, schema_catalog: dict, catalog_idx: dict) -> str:
    """Add NULLIF guards for division and handle special columns dynamically."""
    
    # 1. Guard division by zero
    sql = re.sub(r'/\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(', r'/ NULLIF(\1(', sql, flags=re.I)
    sql = re.sub(r'NULLIF\((COUNT|SUM|AVG|MIN|MAX)\(([^)]+)\)\)', r'NULLIF(\1(\2), 0)', sql, flags=re.I)
    
    # 2. Dynamically detect columns that need numeric casting
    sql = _cast_text_columns_to_numeric(sql, schema_catalog)
    
    return sql


def _cast_text_columns_to_numeric(sql: str, schema_catalog: dict) -> str:
    """
    Dynamically cast text columns to numeric when used in aggregate functions.
    Detects columns by pattern matching against schema and SQL usage.
    """
    
    # Skip if already processed
    if 'REGEXP_REPLACE' in sql:
        return sql
    
    # Build list of potential text columns that might contain numeric data
    text_numeric_candidates = _identify_text_numeric_columns(schema_catalog)
    
    if not text_numeric_candidates:
        return sql
    
    # For each candidate column, check if it's used in numeric context
    for fq_table, col_name in text_numeric_candidates:
        # Check if this column appears in SQL with aggregates
        if not _is_column_in_numeric_context(sql, col_name):
            continue
        
        # Build safe casting expression
        safe_cast_template = "NULLIF(REGEXP_REPLACE({col}, '[^0-9\\.-]', '', 'g'), '')::NUMERIC"
        
        # Replace all references to this column in aggregates
        sql = _replace_column_in_aggregates(sql, fq_table, col_name, safe_cast_template)
    
    return sql


def _identify_text_numeric_columns(schema_catalog: dict) -> list[tuple[str, str]]:
    """
    Identify columns that are likely TEXT but contain numeric data.
    Returns list of (fully_qualified_table, column_name) tuples.
    """
    candidates = []
    
    # Heuristics for identifying text columns with numeric content
    numeric_name_patterns = [
        'amount', 'price', 'cost', 'fee', 'rate', 'discount',
        'charge', 'payment', 'balance', 'total', 'value',
        'income', 'salary', 'revenue', 'profit', 'loss',
        'score', 'count', 'quantity', 'volume', 'sum',
        'offs',  # Catches 'chargeoffs'
        'principal', 'apr', 'interest', 'finance',
        'premium', 'deductible', 'commission'
    ]
    
    for fq_table, columns in schema_catalog.items():
        for col in columns:
            col_lower = col.lower()
            
            # Check if column name suggests numeric content
            if any(pattern in col_lower for pattern in numeric_name_patterns):
                candidates.append((fq_table, col))
                logger.debug(f"[Cast Detector] Found numeric text column: {fq_table}.{col}")
    
    return candidates


def _is_column_in_numeric_context(sql: str, col_name: str) -> bool:
    """Check if column is used in numeric operations (SUM, AVG, division, etc.)."""
    
    col_escaped = re.escape(col_name)
    
    # Pattern: aggregate functions
    agg_pattern = rf'\b(SUM|AVG|COUNT|MIN|MAX)\s*\([^)]*\b{col_escaped}\b'
    if re.search(agg_pattern, sql, flags=re.I):
        logger.debug(f"[Cast Detector] {col_name} used in aggregate function")
        return True
    
    # Pattern: arithmetic operations
    arith_pattern = rf'\b{col_escaped}\b\s*[+\-*/]'
    if re.search(arith_pattern, sql, flags=re.I):
        logger.debug(f"[Cast Detector] {col_name} used in arithmetic operation")
        return True
    
    # Pattern: comparison with numbers
    compare_pattern = rf'\b{col_escaped}\b\s*[<>=].*?\d'
    if re.search(compare_pattern, sql, flags=re.I):
        logger.debug(f"[Cast Detector] {col_name} compared with number")
        return True
    
    return False


def _replace_column_in_aggregates(sql: str, fq_table: str, col_name: str, cast_template: str) -> str:
    """Replace column references inside aggregate functions with safe cast."""
    
    # Extract table name parts for flexible matching
    table_parts = fq_table.replace('"', '').split('.')
    schema_name = table_parts[0] if len(table_parts) > 1 else ''
    table_name = table_parts[-1] if table_parts else ''
    
    col_escaped = re.escape(col_name)
    
    # Build patterns in order of specificity (most specific first)
    patterns = [
        # 1. Fully qualified with quotes: "schema"."table".column or "schema"."table"."column"
        (rf'({re.escape(fq_table)}\."{col_escaped}")\b', 'fq_quoted_col'),
        (rf'({re.escape(fq_table)}\.{col_escaped})\b', 'fq_unquoted_col'),
        
        # 2. Schema.table.column (unquoted)
        (rf'({re.escape(schema_name)}\.{re.escape(table_name)}\.{col_escaped})\b', 'schema_table_col'),
        
        # 3. Table.column
        (rf'({re.escape(table_name)}\.{col_escaped})\b', 'table_col'),
        
        # 4. Bare column name within aggregate function only
        (rf'\b(SUM|AVG|COUNT|MIN|MAX)\s*\(\s*("?{col_escaped}"?)\s*\)', 'bare_in_agg'),
    ]
    
    for pattern, pattern_name in patterns:
        def replace_match(match):
            # Special handling for aggregate with bare column
            if pattern_name == 'bare_in_agg':
                func = match.group(1)
                col_ref = f"{fq_table}.{col_name}"
                casted = cast_template.replace('{col}', col_ref)
                logger.debug(f"[Cast Replace] {pattern_name}: {func}({match.group(2)}) -> {func}({casted})")
                return f"{func}({casted})"
            else:
                # For qualified column references
                col_ref = match.group(1)
                casted = cast_template.replace('{col}', col_ref)
                logger.debug(f"[Cast Replace] {pattern_name}: {col_ref} -> {casted}")
                return casted
        
        sql = re.sub(pattern, replace_match, sql, flags=re.I)
    
    return sql


# ============================================================================
# STAGE 8: Ensure FROM clause exists
# ============================================================================

def _ensure_from_clause(sql: str, schema_catalog: dict, fallback_table: str | None) -> str:
    """Add FROM clause if missing."""
    
    # Check if FROM already exists
    if re.search(r'\bFROM\b', sql, flags=re.I):
        return sql
    
    # Choose fallback table
    fq = fallback_table
    if not fq and schema_catalog:
        fq = next(iter(schema_catalog))
    if not fq:
        return sql
    
    # Insert FROM before WHERE/GROUP/ORDER/LIMIT or at end
    return re.sub(
        r'(?i)^(SELECT\b.*?)\s*(WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|$)',
        lambda m: f'{m.group(1)} FROM {fq} {m.group(2)}' if m.group(2) else f'{m.group(1)} FROM {fq}',
        sql,
        count=1
    )


# ============================================================================
# STAGE 9: Final validation
# ============================================================================

def _final_validation(sql: str, schema_catalog: dict) -> str:
    """Final checks and cleanup."""
    
    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # Validate basic structure
    if not re.search(r'\bSELECT\b.*\bFROM\b', sql, flags=re.I | re.DOTALL):
        logger.warning("[Final] SQL missing SELECT...FROM structure")
        return _generate_safe_fallback_sql(schema_catalog)
    
    # Check parentheses balance
    open_count = sql.count('(')
    close_count = sql.count(')')
    if open_count != close_count:
        logger.warning(f"[Final] Unbalanced parens: {open_count} open, {close_count} close")
        # Try to fix simple cases
        diff = open_count - close_count
        if diff > 0 and diff <= 3:
            sql = sql + ')' * diff
        elif diff < 0 and abs(diff) <= 3:
            sql = '(' * abs(diff) + sql
        else:
            logger.error("[Final] Too many unbalanced parens, generating fallback")
            return _generate_safe_fallback_sql(schema_catalog)
    
    return sql


# ============================================================================
# Fallback SQL generator
# ============================================================================

def _generate_safe_fallback_sql(schema_catalog: dict, fallback_table: str | None = None) -> str:
    """Generate a safe fallback query when repair fails."""
    
    fq = fallback_table
    if not fq and schema_catalog:
        fq = next(iter(schema_catalog))
    
    if not fq:
        return "SELECT 1 AS status, 'No tables available' AS message;"
    
    return f"SELECT COUNT(*) AS row_count FROM {fq} LIMIT 1;"


        
        # Stage 6: Fix date/interval literals
    #     sql = _repair_date_and_interval_literals(sql)
    #     logger.debug(f"[Stage 6] After date/interval repair: {sql[:150]}")

    #     # Stage 6: Fix date/interval literals AND date column types
    #     sql = _repair_date_columns(sql, schema_catalog)  # ← ADD THIS LINE
    #     logger.debug(f"[Stage 6] After date/interval repair: {sql[:150]}")

    #     sql = _add_date_filters(sql, schema_catalog)  # ← Add this
    #     logger.debug(f"[Stage 6] After date/interval repair: {sql[:150]}")
        
    #     # Stage 7: Add NULL/zero guards for arithmetic
    #     sql = _add_safety_guards(sql, schema_catalog, catalog_idx)
    #     logger.debug(f"[Stage 7] After safety guards: {sql[:150]}")
        
    #     # Stage 8: Ensure FROM clause exists
    #     sql = _ensure_from_clause(sql, schema_catalog, fallback_table)
    #     logger.debug(f"[Stage 8] After FROM check: {sql[:150]}")
        
    #     # Stage 9: Final validation
    #     sql = _final_validation(sql, schema_catalog)
    #     logger.debug(f"[Stage 9] Final SQL: {sql[:200]}")
        
    #     return sql
        
    # except Exception as e:
    #     logger.error(f"SQL repair failed: {e}. Input was: {original_input}")
    #     return _generate_safe_fallback_sql(schema_catalog, fallback_table)


# ============================================================================
# STAGE 1: Extract SQL from LLM wrapper formats
# ============================================================================


def _add_safety_guards(sql: str, schema_catalog: dict, catalog_idx: dict) -> str:
    """Add NULLIF guards for division and handle special columns dynamically."""
    
    # 1. Guard division by zero
    sql = re.sub(r'/\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(', r'/ NULLIF(\1(', sql, flags=re.I)
    sql = re.sub(r'NULLIF\((COUNT|SUM|AVG|MIN|MAX)\(([^)]+)\)\)', r'NULLIF(\1(\2), 0)', sql, flags=re.I)
    
    # 2. Dynamically detect columns that need numeric casting
    sql = _cast_text_columns_to_numeric(sql, schema_catalog)
    
    return sql


def _cast_text_columns_to_numeric(sql: str, schema_catalog: dict) -> str:
    """
    Dynamically cast text columns to numeric when used in aggregate functions.
    Detects columns by pattern matching against schema and SQL usage.
    """
    
    # Skip if already processed
    if 'REGEXP_REPLACE' in sql:
        return sql
    
    # Build list of potential text columns that might contain numeric data
    text_numeric_candidates = _identify_text_numeric_columns(schema_catalog)
    
    if not text_numeric_candidates:
        return sql
    
    # For each candidate column, check if it's used in numeric context
    for fq_table, col_name in text_numeric_candidates:
        # Check if this column appears in SQL with aggregates
        if not _is_column_in_numeric_context(sql, col_name):
            continue
        
        # Build safe casting expression
        safe_cast_template = "NULLIF(REGEXP_REPLACE({col}, '[^0-9\\.-]', '', 'g'), '')::NUMERIC"
        
        # Replace all references to this column in aggregates
        sql = _replace_column_in_aggregates(sql, fq_table, col_name, safe_cast_template)
    
    return sql


def _identify_text_numeric_columns(schema_catalog: dict) -> list[tuple[str, str]]:
    """
    Identify columns that are likely TEXT but contain numeric data.
    Returns list of (fully_qualified_table, column_name) tuples.
    """
    candidates = []
    
    # Heuristics for identifying text columns with numeric content
    numeric_name_patterns = [
        'amount', 'price', 'cost', 'fee', 'rate', 'discount',
        'charge', 'payment', 'balance', 'total', 'value',
        'income', 'salary', 'revenue', 'profit', 'loss'
    ]
    
    for fq_table, columns in schema_catalog.items():
        for col in columns:
            col_lower = col.lower()
            
            # Check if column name suggests numeric content
            if any(pattern in col_lower for pattern in numeric_name_patterns):
                candidates.append((fq_table, col))
    
    return candidates


def _is_column_in_numeric_context(sql: str, col_name: str) -> bool:
    """Check if column is used in numeric operations (SUM, AVG, division, etc.)."""
    
    # Pattern: aggregate functions
    agg_pattern = rf'\b(SUM|AVG|COUNT|MIN|MAX)\s*\([^)]*\b{re.escape(col_name)}\b'
    if re.search(agg_pattern, sql, flags=re.I):
        return True
    
    # Pattern: arithmetic operations
    arith_pattern = rf'\b{re.escape(col_name)}\b\s*[+\-*/]'
    if re.search(arith_pattern, sql, flags=re.I):
        return True
    
    # Pattern: comparison with numbers
    compare_pattern = rf'\b{re.escape(col_name)}\b\s*[<>=].*?\d'
    if re.search(compare_pattern, sql, flags=re.I):
        return True
    
    return False


def _replace_column_in_aggregates(sql: str, fq_table: str, col_name: str, cast_template: str) -> str:
    """Replace column references inside aggregate functions with safe cast."""
    
    # Extract table name parts for flexible matching
    table_parts = fq_table.replace('"', '').split('.')
    table_name = table_parts[-1] if table_parts else ''
    
    # Pattern matches:
    # - SUM("schema"."table".column)
    # - SUM(table.column)
    # - SUM(column) -- if table name matches context
    patterns = [
        # Fully qualified with quotes
        rf'({re.escape(fq_table)}\.{col_name})\b',
        # Unquoted schema.table.column
        rf'(\w+\.\w+\.{col_name})\b',
        # table.column
        rf'({re.escape(table_name)}\.{col_name})\b',
        # Just column name (within aggregate)
        rf'\b(SUM|AVG|COUNT|MIN|MAX)\s*\(\s*({col_name})\s*\)'
    ]
    
    for pattern in patterns:
        def replace_match(match):
            # For the last pattern (aggregate with bare column)
            if len(match.groups()) == 2 and match.group(1).upper() in ['SUM', 'AVG', 'COUNT', 'MIN', 'MAX']:
                func = match.group(1)
                col_ref = f"{fq_table}.{match.group(2)}"
                return f"{func}({cast_template.replace('{col}', col_ref)})"
            else:
                # For other patterns
                col_ref = match.group(1)
                return cast_template.replace('{col}', col_ref)
        
        sql = re.sub(pattern, replace_match, sql, flags=re.I)
    
    return sql

def _extract_sql_from_llm_output111(raw: Any) -> str:
    """Handle tuple/list/dict/str formats and remove markdown fences."""
    
    # Coerce to string
    if raw is None:
        return ""
    
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, (tuple, list)):
        # Pick first non-empty string item
        text = ""
        for item in raw:
            if isinstance(item, str) and item.strip():
                text = item
                break
        if not text:
            text = " ".join(str(x) for x in raw if x is not None)
    elif isinstance(raw, dict):
        # Try common keys
        for key in ("sql", "text", "content", "message", "output"):
            if key in raw and isinstance(raw[key], str) and raw[key].strip():
                text = raw[key]
                break
        else:
            text = str(raw)
    else:
        text = str(raw)
    
    if not text or not text.strip():
        return ""
    
    # Remove header tokens (some models add these)
    text = re.sub(r'<\|header_start\|>\s*sql', '', text, flags=re.I)
    text = re.sub(r'<\|header_end\|>', '', text, flags=re.I)
    
    # Extract from fenced blocks: ```sql ... ```
    m = re.search(r'```(?:sql)?\s*(.*?)\s*```', text, flags=re.I | re.DOTALL)
    if m:
        text = m.group(1)
    
    # Remove "sql\n" prefix that some models add
    text = re.sub(r'^\s*sql\s+', '', text, flags=re.I)
    
    # Keep only from first SELECT/WITH onward (drop preamble text)
    m = re.search(r'\b(WITH|SELECT)\b.*', text, flags=re.I | re.DOTALL)
    if m:
        text = m.group(0)
    
    return text.strip()


# ============================================================================
# STAGE 2: Basic syntax cleanup
# ============================================================================

def _cleanup_basic_syntax(sql: str) -> str:
    """Fix common tokenization glitches before semantic repairs."""
    
    # Remove SQL comments
    sql = re.sub(r'--[^\n]*', '', sql)
    
    # Fix duplicated keywords (SELECT SELECT -> SELECT)
    for kw in ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'WHEN', 'END', 'CASE']:
        pattern = rf'\b({kw})\s+\1\b'
        sql = re.sub(pattern, r'\1', sql, flags=re.I)
    
    # Fix compound keyword duplications
    sql = re.sub(r'\bGROUP\s+BY\s+BY\b', 'GROUP BY', sql, flags=re.I)
    sql = re.sub(r'\bORDER\s+BY\s+BY\b', 'ORDER BY', sql, flags=re.I)
    
    # Fix function name glitches
    sql = re.sub(r'\bEXTRACTRACT\b', 'EXTRACT', sql, flags=re.I)
    sql = re.sub(r'\bYYEAR\b', 'YEAR', sql, flags=re.I)
    sql = re.sub(r'\bMMONTH\b', 'MONTH', sql, flags=re.I)
    
    # Collapse stacked/broken quotes
    sql = re.sub(r'""+', '"', sql)
    sql = re.sub(r'"\s*\.\s*"', '"."', sql)
    sql = re.sub(r'\.\s*\.', '.', sql)
    
    # Fix missing dots between quoted identifiers: "schema""table" -> "schema"."table"
    sql = re.sub(r'"([A-Za-z_]\w*)"\s*"\s*([A-Za-z_]\w*)"', r'"\1"."\2"', sql)
    
    # Remove dangling commas before clauses
    sql = re.sub(r',\s*(WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|FROM)\b', r' \1', sql, flags=re.I)
    
    # Fix duplicate comparison operators
    sql = re.sub(r'>\s*>', '>', sql)
    sql = re.sub(r'<\s*<', '<', sql)
    sql = re.sub(r'=\s*=', '=', sql)
    
    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    return sql


# ============================================================================
# STAGE 3: Build catalog indices
# ============================================================================

def _build_catalog_indices(schema_catalog: dict) -> dict:
    """Build lookup structures for fast column/table resolution."""
    
    fq_canon = {}        # normalized fq -> canonical quoted fq
    schemas = set()
    tables_by_schema = {}
    cols_by_fq = {}      # normalized fq -> set of lowercase column names
    all_cols = set()
    col_to_tables = {}   # column name -> list of tables that have it
    
    for fq, cols in (schema_catalog or {}).items():
        # Parse "schema"."table"
        parts = fq.replace('"', '').split('.')
        if len(parts) != 2:
            continue
        
        s, t = parts
        key = f"{s.lower()}.{t.lower()}"
        
        fq_canon[key] = f'"{s}"."{t}"'
        schemas.add(s)
        tables_by_schema.setdefault(s.lower(), set()).add(t)
        cols_by_fq[key] = {c.lower() for c in cols}
        
        for c in cols:
            c_lower = c.lower()
            all_cols.add(c_lower)
            col_to_tables.setdefault(c_lower, []).append(key)
    
    return {
        'fq_canon': fq_canon,
        'schemas': sorted(schemas),
        'tables_by_schema': {k: sorted(v) for k, v in tables_by_schema.items()},
        'cols_by_fq': cols_by_fq,
        'all_cols': all_cols,
        'col_to_tables': col_to_tables
    }


# ============================================================================
# STAGE 4: Repair table references
# ============================================================================

def _repair_table_references(sql: str, catalog_idx: dict) -> str:
    """Snap schema.table references to canonical forms from catalog."""
    
    fq_canon = catalog_idx['fq_canon']
    
    # Pattern: "schema"."table" or schema.table (possibly with broken quotes/spaces)
    pattern = r'("?[A-Za-z_][\w$]*"?)\s*\.\s*("?[A-Za-z_][\w$]*"?)'
    
    def replace(m):
        s_raw, t_raw = m.group(1), m.group(2)
        s = s_raw.replace('"', '').strip().lower()
        t = t_raw.replace('"', '').strip().lower()
        key = f"{s}.{t}"
        
        # Exact match
        if key in fq_canon:
            return fq_canon[key]
        
        # Fuzzy match
        candidates = list(fq_canon.keys())
        matches = get_close_matches(key, candidates, n=1, cutoff=0.75)
        if matches:
            return fq_canon[matches[0]]
        
        # Return quoted form of what we have
        return f'"{s}"."{t}"'
    
    return re.sub(pattern, replace, sql)


# ============================================================================
# STAGE 5: Repair column references
# ============================================================================

def _repair_column_references(sql: str, catalog_idx: dict) -> str:
    """Fix misspelled/missing columns based on catalog."""
    
    cols_by_fq = catalog_idx['cols_by_fq']
    
    # Extract tables used in FROM/JOIN
    used_tables = _extract_used_tables(sql)
    if not used_tables:
        return sql
    
    # Map aliases to tables
    alias_map = _extract_alias_map(sql, used_tables)
    if not alias_map:
        return sql
    
    # Pattern: qualifier.column
    qualifiers = '|'.join(re.escape(q) for q in sorted(alias_map.keys(), key=len, reverse=True))
    pattern = rf'\b({qualifiers})\s*\.\s*"?([A-Za-z_][\w$]*)"?'
    
    def replace(m):
        qual, col = m.group(1).lower(), m.group(2)
        fq = alias_map.get(qual)
        if not fq:
            return m.group(0)
        
        cols = cols_by_fq.get(fq, set())
        if not cols:
            return m.group(0)
        
        # Exact match
        if col.lower() in cols:
            return f'{qual}.{col}'
        
        # Fuzzy match
        matches = get_close_matches(col.lower(), list(cols), n=1, cutoff=0.75)
        if matches:
            return f'{qual}.{matches[0]}'
        
        # No match - return original
        return m.group(0)
    
    return re.sub(pattern, replace, sql, flags=re.I)


def _extract_used_tables(sql: str) -> set:
    """Extract fully-qualified table names from FROM/JOIN clauses."""
    pattern = r'\b(FROM|JOIN)\s+("?[A-Za-z_][\w$]*"?\s*\.\s*"?[A-Za-z_][\w$]*"?)'
    matches = re.findall(pattern, sql, flags=re.I)
    tables = set()
    for _, fq in matches:
        fq_norm = fq.replace('"', '').replace(' ', '').lower()
        tables.add(fq_norm)
    return tables


def _extract_alias_map(sql: str, used_tables: set) -> dict:
    """Build {alias: normalized_fq} mapping."""
    alias_map = {}
    
    # FROM/JOIN table AS alias
    pattern = r'\b(FROM|JOIN)\s+("?[A-Za-z_][\w$]*"?\s*\.\s*"?[A-Za-z_][\w$]*"?)(?:\s+AS)?\s+([A-Za-z_][\w$]+)?'
    for m in re.finditer(pattern, sql, flags=re.I):
        fq = m.group(2).replace('"', '').replace(' ', '').lower()
        alias = m.group(3)
        if alias:
            alias_map[alias.lower()] = fq
    
    # Also allow unaliased table name as qualifier
    for fq in used_tables:
        if '.' in fq:
            _, tbl = fq.split('.', 1)
            alias_map[tbl] = fq
    
    return alias_map


# ============================================================================
# STAGE 6: Fix date/interval literals
# ============================================================================

def _repair_date_and_interval_literals(sql: str) -> str:
    """Fix common date/interval syntax errors."""

     # Fix INTERVAL syntax: "INTERVAL6 months" or "183days" -> INTERVAL '6 months'
    sql = re.sub(
        r'\bINTERVAL\s*([0-9]+)\s*(year|month|day|hour|minute|second)s?\b',
        lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}s'",
        sql,
        flags=re.I
    )
    
    # Fix bare number + unit: "183days" -> INTERVAL '183 days'
    sql = re.sub(
        r'\b([0-9]+)(days?|months?|years?)\b',
        lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}'",
        sql,
        flags=re.I
    )
    
    # Fix duplicated date literals
    sql = re.sub(
        r"'(\d{4}-\d{2}-\d{2})'\s*'(\d{4}-\d{2}-\d{2})'",
        r"'\1'",
        sql
    )
    
    # Fix date arithmetic
    sql = re.sub(
        r"'(\d{4}-\d{2}-\d{2})'\s*([+-])\s*INTERVAL",
        r"DATE '\1' \2 INTERVAL",
        sql,
        flags=re.I
    )
    

    
    # Fix INTERVAL syntax: "INTERVAL6 months" or "183days" -> INTERVAL '6 months'
    sql = re.sub(
        r'\bINTERVAL\s*([0-9]+)\s*(year|month|day|hour|minute|second)s?\b',
        lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}s'",
        sql,
        flags=re.I
    )
    
    # Fix bare number + unit: "183days" -> INTERVAL '183 days'
    sql = re.sub(
        r'\b([0-9]+)(days?|months?|years?)\b',
        lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}'",
        sql,
        flags=re.I
    )
    
    # Fix duplicated date literals: '2025-01-01' '2025-01-01' -> '2025-01-01'
    sql = re.sub(
        r"'(\d{4}-\d{2}-\d{2})'\s*'(\d{4}-\d{2}-\d{2})'",
        r"'\1'",
        sql
    )
    
    # Fix date arithmetic: '2025-01-01' + INTERVAL -> DATE '2025-01-01' + INTERVAL
    sql = re.sub(
        r"'(\d{4}-\d{2}-\d{2})'\s*([+-])\s*INTERVAL",
        r"DATE '\1' \2 INTERVAL",
        sql,
        flags=re.I
    )
    
    return sql


# ============================================================================
# STAGE 7: Add safety guards
# ============================================================================

def _add_safety_guards(sql: str, schema_catalog: dict, catalog_idx: dict) -> str:
    """Add NULLIF guards for division and handle special columns."""
    
    # Guard division by aggregates: / COUNT(...) -> / NULLIF(COUNT(...), 0)
    sql = re.sub(
        r'/\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(',
        r'/ NULLIF(\1(',
        sql,
        flags=re.I
    )
    
    # Close the NULLIF that was opened above
    # This is tricky - we need to match the closing paren of the aggregate
    # Simplified approach: look for patterns like "NULLIF(COUNT(...))" without closing
    sql = re.sub(
        r'NULLIF\((COUNT|SUM|AVG|MIN|MAX)\(([^)]+)\)\)',
        r'NULLIF(\1(\2), 0)',
        sql,
        flags=re.I
    )
    
    # Special handling for 'chargeoffs' column (text that needs numeric cast)
    chargeoffs_table = _find_table_with_column(schema_catalog, 'chargeoffs')
    if chargeoffs_table and 'chargeoffs' in sql.lower():
        # Only wrap if not already wrapped
        if 'REGEXP_REPLACE' not in sql or 'chargeoffs' not in sql:
            cast_expr = f"NULLIF(REGEXP_REPLACE({chargeoffs_table}.chargeoffs, '[^0-9\\.-]', '', 'g'), '')::NUMERIC"
            # Replace qualified references
            sql = re.sub(
                rf'({re.escape(chargeoffs_table)}|[A-Za-z_]\w*)\.chargeoffs\b',
                lambda m: cast_expr if chargeoffs_table in m.group(0) else m.group(0),
                sql,
                flags=re.I
            )
    
    return sql


def _find_table_with_column(schema_catalog: dict, col_name: str) -> str | None:
    """Find the fully-qualified table that contains the given column."""
    col_lower = col_name.lower()
    for fq, cols in schema_catalog.items():
        if any(c.lower() == col_lower for c in cols):
            return fq
    return None


# ============================================================================
# STAGE 8: Ensure FROM clause exists
# ============================================================================

def _ensure_from_clause(sql: str, schema_catalog: dict, fallback_table: str | None) -> str:
    """Add FROM clause if missing."""
    
    # Check if FROM already exists
    if re.search(r'\bFROM\b', sql, flags=re.I):
        return sql
    
    # Choose fallback table
    fq = fallback_table
    if not fq and schema_catalog:
        fq = next(iter(schema_catalog))
    if not fq:
        return sql
    
    # Insert FROM before WHERE/GROUP/ORDER/LIMIT or at end
    return re.sub(
        r'(?i)^(SELECT\b.*?)\s*(WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|$)',
        lambda m: f'{m.group(1)} FROM {fq} {m.group(2)}' if m.group(2) else f'{m.group(1)} FROM {fq}',
        sql,
        count=1
    )


# ============================================================================
# STAGE 9: Final validation
# ============================================================================

def _final_validation(sql: str, schema_catalog: dict) -> str:
    """Final checks and cleanup."""
    
    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # Validate basic structure
    if not re.search(r'\bSELECT\b.*\bFROM\b', sql, flags=re.I | re.DOTALL):
        logger.warning("[Final] SQL missing SELECT...FROM structure")
        return _generate_safe_fallback_sql(schema_catalog)
    
    # Check parentheses balance
    open_count = sql.count('(')
    close_count = sql.count(')')
    if open_count != close_count:
        logger.warning(f"[Final] Unbalanced parens: {open_count} open, {close_count} close")
        # Try to fix simple cases
        diff = open_count - close_count
        if diff > 0 and diff <= 3:
            sql = sql + ')' * diff
        elif diff < 0 and abs(diff) <= 3:
            sql = '(' * abs(diff) + sql
        else:
            logger.error("[Final] Too many unbalanced parens, generating fallback")
            return _generate_safe_fallback_sql(schema_catalog)
    
    return sql

# # NEW: Add this function right after _repair_date_and_interval_literals
# def _repair_date_columns(sql: str, schema_catalog: dict) -> str:
#     """Cast date columns to proper types before EXTRACT/date arithmetic."""
    
#     # Find date-like columns from schema
#     date_cols = set()
#     for fq, cols in schema_catalog.items():
#         for col in cols:
#             col_lower = col.lower()
#             if any(x in col_lower for x in ['date', 'time', 'created', 'updated', 'funded', 'purchase', 'report']):
#                 date_cols.add(col_lower)
    
#     if not date_cols:
#         return sql
    
#     # Pattern: EXTRACT(... FROM column) where column might be text
#     def fix_extract(match):
#         unit, col = match.group(1), match.group(2).strip()
#         col_base = col.split('.')[-1].lower() if '.' in col else col.lower()
        
#         if col_base in date_cols:
#             # Cast to DATE if it's a known date column
#             return f"EXTRACT({unit} FROM {col}::DATE)"
#         return match.group(0)
    
#     sql = re.sub(
#         r'EXTRACT\s*\(\s*(\w+)\s+FROM\s+(["\w.]+)\s*\)',
#         fix_extract,
#         sql,
#         flags=re.IGNORECASE
#     )
    
#     # Also fix date arithmetic: column + INTERVAL
#     for col in date_cols:
#         # Match schema.table.column or table.column patterns
#         pattern = rf'([\w."]+\.{col})\s*([+-])\s*(INTERVAL|DATE)'
#         sql = re.sub(
#             pattern,
#             r'\1::DATE \2 \3',
#             sql,
#             flags=re.IGNORECASE
#         )
    
#     return sql
def _repair_date_columns11(sql: str, schema_catalog: dict) -> str:
    """Cast date columns to proper types before EXTRACT/date arithmetic."""
    
    date_cols = set()
    for fq, cols in schema_catalog.items():
        for col in cols:
            col_lower = col.lower()
            if any(x in col_lower for x in ['date', 'time', 'created', 'updated', 'funded', 'purchase', 'report']):
                date_cols.add(col_lower)
    
    if not date_cols:
        return sql
    
    def fix_extract(match):
        unit, col = match.group(1), match.group(2).strip()
        col_base = col.split('.')[-1].lower() if '.' in col else col.lower()
        
        if col_base in date_cols:
            # Safe cast: handle empty strings and NULL
            safe_cast = f"NULLIF(TRIM({col}), '')::DATE"
            return f"EXTRACT({unit} FROM {safe_cast})"
        return match.group(0)
    
    sql = re.sub(
        r'EXTRACT\s*\(\s*(\w+)\s+FROM\s+(["\w.]+)\s*\)',
        fix_extract,
        sql,
        flags=re.IGNORECASE
    )
    
    # Also fix date arithmetic: column + INTERVAL
    for col in date_cols:
        pattern = rf'([\w."]+\.{col})(?!::)\s*([+-])\s*(INTERVAL|DATE)'
        sql = re.sub(
            pattern,
            lambda m: f"NULLIF(TRIM({m.group(1)}), '')::DATE {m.group(2)} {m.group(3)}",
            sql,
            flags=re.IGNORECASE
        )
    
    return sql


# ============================================================================
# STAGE 7: Add safety guards (COMPLETE DYNAMIC VERSION)
# ============================================================================

def _add_safety_guards(sql: str, schema_catalog: dict, catalog_idx: dict) -> str:
    """Add NULLIF guards for division and handle special columns dynamically."""
    
    # 1. Guard division by zero
    sql = re.sub(r'/\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(', r'/ NULLIF(\1(', sql, flags=re.I)
    sql = re.sub(r'NULLIF\((COUNT|SUM|AVG|MIN|MAX)\(([^)]+)\)\)', r'NULLIF(\1(\2), 0)', sql, flags=re.I)
    
    # 2. Dynamically detect columns that need numeric casting
    sql = _cast_text_columns_to_numeric(sql, schema_catalog)
    
    return sql


def _cast_text_columns_to_numeric(sql: str, schema_catalog: dict) -> str:
    """
    Dynamically cast text columns to numeric when used in aggregate functions.
    Detects columns by pattern matching against schema and SQL usage.
    """
    
    # Skip if already processed
    if 'REGEXP_REPLACE' in sql:
        return sql
    
    # Build list of potential text columns that might contain numeric data
    text_numeric_candidates = _identify_text_numeric_columns(schema_catalog)
    
    if not text_numeric_candidates:
        return sql
    
    # For each candidate column, check if it's used in numeric context
    for fq_table, col_name in text_numeric_candidates:
        # Check if this column appears in SQL with aggregates
        if not _is_column_in_numeric_context(sql, col_name):
            continue
        
        # Build safe casting expression
        safe_cast_template = "NULLIF(REGEXP_REPLACE({col}, '[^0-9\\.-]', '', 'g'), '')::NUMERIC"
        
        # Replace all references to this column in aggregates
        sql = _replace_column_in_aggregates(sql, fq_table, col_name, safe_cast_template)
    
    return sql


def _identify_text_numeric_columns(schema_catalog: dict) -> list[tuple[str, str]]:
    """
    Identify columns that are likely TEXT but contain numeric data.
    Returns list of (fully_qualified_table, column_name) tuples.
    """
    candidates = []
    
    # Heuristics for identifying text columns with numeric content
    numeric_name_patterns = [
        'amount', 'price', 'cost', 'fee', 'rate', 'discount',
        'charge', 'payment', 'balance', 'total', 'value',
        'income', 'salary', 'revenue', 'profit', 'loss',
        'score', 'count', 'quantity', 'volume', 'sum',
        'offs',  # Catches 'chargeoffs'
        'principal', 'apr', 'interest', 'finance',
        'premium', 'deductible', 'commission'
    ]
    
    for fq_table, columns in schema_catalog.items():
        for col in columns:
            col_lower = col.lower()
            
            # Check if column name suggests numeric content
            if any(pattern in col_lower for pattern in numeric_name_patterns):
                candidates.append((fq_table, col))
                logger.debug(f"[Cast Detector] Found numeric text column: {fq_table}.{col}")
    
    return candidates


def _is_column_in_numeric_context(sql: str, col_name: str) -> bool:
    """Check if column is used in numeric operations (SUM, AVG, division, etc.)."""
    
    col_escaped = re.escape(col_name)
    
    # Pattern: aggregate functions
    agg_pattern = rf'\b(SUM|AVG|COUNT|MIN|MAX)\s*\([^)]*\b{col_escaped}\b'
    if re.search(agg_pattern, sql, flags=re.I):
        logger.debug(f"[Cast Detector] {col_name} used in aggregate function")
        return True
    
    # Pattern: arithmetic operations
    arith_pattern = rf'\b{col_escaped}\b\s*[+\-*/]'
    if re.search(arith_pattern, sql, flags=re.I):
        logger.debug(f"[Cast Detector] {col_name} used in arithmetic operation")
        return True
    
    # Pattern: comparison with numbers
    compare_pattern = rf'\b{col_escaped}\b\s*[<>=].*?\d'
    if re.search(compare_pattern, sql, flags=re.I):
        logger.debug(f"[Cast Detector] {col_name} compared with number")
        return True
    
    return False


def _replace_column_in_aggregates(sql: str, fq_table: str, col_name: str, cast_template: str) -> str:
    """Replace column references inside aggregate functions with safe cast."""
    
    # Extract table name parts for flexible matching
    table_parts = fq_table.replace('"', '').split('.')
    schema_name = table_parts[0] if len(table_parts) > 1 else ''
    table_name = table_parts[-1] if table_parts else ''
    
    col_escaped = re.escape(col_name)
    
    # Build patterns in order of specificity (most specific first)
    patterns = [
        # 1. Fully qualified with quotes: "schema"."table".column or "schema"."table"."column"
        (rf'({re.escape(fq_table)}\."{col_escaped}")\b', 'fq_quoted_col'),
        (rf'({re.escape(fq_table)}\.{col_escaped})\b', 'fq_unquoted_col'),
        
        # 2. Schema.table.column (unquoted)
        (rf'({re.escape(schema_name)}\.{re.escape(table_name)}\.{col_escaped})\b', 'schema_table_col'),
        
        # 3. Table.column
        (rf'({re.escape(table_name)}\.{col_escaped})\b', 'table_col'),
        
        # 4. Bare column name within aggregate function only
        (rf'\b(SUM|AVG|COUNT|MIN|MAX)\s*\(\s*("?{col_escaped}"?)\s*\)', 'bare_in_agg'),
    ]
    
    for pattern, pattern_name in patterns:
        def replace_match(match):
            # Special handling for aggregate with bare column
            if pattern_name == 'bare_in_agg':
                func = match.group(1)
                col_ref = f"{fq_table}.{col_name}"
                casted = cast_template.replace('{col}', col_ref)
                logger.debug(f"[Cast Replace] {pattern_name}: {func}({match.group(2)}) -> {func}({casted})")
                return f"{func}({casted})"
            else:
                # For qualified column references
                col_ref = match.group(1)
                casted = cast_template.replace('{col}', col_ref)
                logger.debug(f"[Cast Replace] {pattern_name}: {col_ref} -> {casted}")
                return casted
        
        sql = re.sub(pattern, replace_match, sql, flags=re.I)
    
    return sql


# ============================================================================
# STAGE 6: Fix date/interval literals AND date columns (COMPLETE VERSION)
# ============================================================================

def _repair_date_and_interval_literals(sql: str) -> str:
    """Fix common date/interval syntax errors."""
    
    # Fix INTERVAL syntax: "INTERVAL6 months" or "183days" -> INTERVAL '6 months'
    sql = re.sub(
        r'\bINTERVAL\s*([0-9]+)\s*(year|month|day|hour|minute|second)s?\b',
        lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}s'",
        sql,
        flags=re.I
    )
    
    # Fix bare number + unit: "183days" -> INTERVAL '183 days'
    sql = re.sub(
        r'\b([0-9]+)(days?|months?|years?)\b',
        lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}'",
        sql,
        flags=re.I
    )
    
    # Fix duplicated date literals
    sql = re.sub(
        r"'(\d{4}-\d{2}-\d{2})'\s*'(\d{4}-\d{2}-\d{2})'",
        r"'\1'",
        sql
    )
    
    # Fix date arithmetic
    sql = re.sub(
        r"'(\d{4}-\d{2}-\d{2})'\s*([+-])\s*INTERVAL",
        r"DATE '\1' \2 INTERVAL",
        sql,
        flags=re.I
    )
    
    return sql

from difflib import get_close_matches

def _repair_date_columns(sql: str, schema_catalog: dict) -> str:
    """Cast date columns to proper types before EXTRACT/date arithmetic."""
    
    # Find date-like columns from schema
    date_cols = set()
    for fq, cols in schema_catalog.items():
        for col in cols:
            col_lower = col.lower()
            if any(x in col_lower for x in ['date', 'time', 'created', 'updated', 'funded', 'purchase', 'report', 'maturity', 'expiration']):
                date_cols.add(col_lower)
                logger.debug(f"[Date Detector] Found date column: {fq}.{col}")
    
    if not date_cols:
        return sql
    
    # Pattern: EXTRACT(... FROM column) where column might be text
    def fix_extract(match):
        unit, col = match.group(1), match.group(2).strip()
        col_base = col.split('.')[-1].replace('"', '').lower() if '.' in col else col.replace('"', '').lower()
        
        if col_base in date_cols:
            # Safe cast: handle empty strings and NULL
            safe_cast = f"NULLIF(TRIM({col}), '')::DATE"
            logger.debug(f"[Date Cast] EXTRACT({unit} FROM {col}) -> EXTRACT({unit} FROM {safe_cast})")
            return f"EXTRACT({unit} FROM {safe_cast})"
        return match.group(0)
    
    sql = re.sub(
        r'EXTRACT\s*\(\s*(\w+)\s+FROM\s+(["\w.]+)\s*\)',
        fix_extract,
        sql,
        flags=re.IGNORECASE
    )
    
    # Also fix date arithmetic: column + INTERVAL
    for col in date_cols:
        # Match schema.table.column or table.column patterns (avoid double-casting)
        pattern = rf'([\w."]+\.{col})(?!::)\s*([+-])\s*(INTERVAL|DATE)'
        
        def fix_date_arith(match):
            col_ref, op, keyword = match.group(1), match.group(2), match.group(3)
            safe_cast = f"NULLIF(TRIM({col_ref}), '')::DATE"
            logger.debug(f"[Date Arith] {col_ref} {op} {keyword} -> {safe_cast} {op} {keyword}")
            return f"{safe_cast} {op} {keyword}"
        
        sql = re.sub(pattern, fix_date_arith, sql, flags=re.IGNORECASE)
    
    return sql

def _add_date_filters(sql: str, schema_catalog: dict) -> str:
    """Add WHERE filters to exclude invalid date values."""
    
    date_cols = set()
    for fq, cols in schema_catalog.items():
        for col in cols:
            col_lower = col.lower()
            if any(x in col_lower for x in ['date', 'time', 'created', 'updated', 'funded', 'purchase', 'report']):
                date_cols.add(f"{fq}.{col}")
    
    if not date_cols:
        return sql
    
    # Find tables used in query
    used_cols = []
    for fq_col in date_cols:
        if fq_col.lower() in sql.lower():
            used_cols.append(fq_col)
    
    if not used_cols:
        return sql
    
    # Build filter conditions
    filters = [f"{col} IS NOT NULL AND TRIM({col}) != ''" for col in used_cols]
    filter_clause = " AND " + " AND ".join(filters)
    
    # Insert before existing WHERE or add new WHERE
    if re.search(r'\bWHERE\b', sql, re.IGNORECASE):
        sql = re.sub(
            r'(\bWHERE\b)',
            rf'\1 {filter_clause} AND',
            sql,
            count=1,
            flags=re.IGNORECASE
        )
    else:
        # Add WHERE before GROUP BY/ORDER BY/LIMIT
        sql = re.sub(
            r'(\b(?:GROUP BY|ORDER BY|LIMIT)\b)',
            rf'WHERE {filter_clause} \1',
            sql,
            count=1,
            flags=re.IGNORECASE
        )
    
    return sql

# ============================================================================
# # Fallback SQL generator
# # ============================================================================

# def _generate_safe_fallback_sql(schema_catalog: dict, fallback_table: str | None = None) -> str:
#     """Generate a safe fallback query when repair fails."""
    
#     fq = fallback_table
#     if not fq and schema_catalog:
#         fq = next(iter(schema_catalog))
    
#     if not fq:
#         return "SELECT 1 AS status, 'No tables available' AS message;"
    
#     return f"SELECT COUNT(*) AS row_count FROM {fq} LIMIT 1;"





def _cleanup_basic_syntax_FINAL(sql: str) -> str:
    # remove comments
    sql = re.sub(r'--[^\n]*', '', sql)

    # de-dupe keywords
    for kw in ['SELECT','FROM','WHERE','GROUP BY','ORDER BY','HAVING','WHEN','END','CASE']:
        sql = re.sub(rf'\b({kw})\s+\1\b', r'\1', sql, flags=re.I)

    # fix duplicated DATE, duplicated year self-comparisons
    sql = re.sub(r"\bDATE\s+DATE\s+(')", r"DATE \1", sql, flags=re.I)
    sql = re.sub(r'\s+(AND|OR)\s+\(?\s*(20\d{2})\s*=\s*\2\s*\)?(?=\s+(AND|OR|$))', ' ', sql, flags=re.I)
    sql = re.sub(r'\bWHERE\s+\(?\s*(20\d{2})\s*=\s*\1\s*\)?\s+(AND|OR)\s+', 'WHERE ', sql, flags=re.I)

    # ensure ROUND( ... ) AS alias
    sql = re.sub(r'\bROUND\s*\(([^)]*)\s+AS\b', r'ROUND(\1) AS', sql, flags=re.I)

    # 👇 ensure NULLIF aggregate closes before AS alias
    sql = re.sub(
        r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s+AS\b',
        r'NULLIF(\1(\2)\3, 0) AS', sql, flags=re.I
    )
    # also close if division form forgot the ", 0)"
    sql = re.sub(
        r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s*\)\s*/',
        r'NULLIF(\1(\2)\3, 0) /', sql, flags=re.I
    )

    # drop bare year tokens from GROUP BY / ORDER BY lists
    sql = re.sub(r'\bGROUP\s+BY\s*(.*?)(?=\bORDER\b|\bLIMIT\b|$)',
                 lambda m: 'GROUP BY ' + re.sub(r'(?:^|,)\s*20\d{2}\s*(?=,|$)', ',', m.group(1)).strip(' ,'),
                 sql, flags=re.I)
    sql = re.sub(r'\bORDER\s+BY\s*(.*?)(?=\bLIMIT\b|$)',
                 lambda m: 'ORDER BY ' + re.sub(r'(?:^|,)\s*20\d{2}\s*(?=,|$)', ',', m.group(1)).strip(' ,'),
                 sql, flags=re.I)

    # fixes you already had
    sql = re.sub(r'\bGROUP\s+BY\s+BY\b','GROUP BY',sql,flags=re.I)
    sql = re.sub(r'\bORDER\s+BY\s+BY\b','ORDER BY',sql,flags=re.I)
    sql = re.sub(r'\bEXTRACTRACT\b','EXTRACT',sql,flags=re.I)
    sql = re.sub(r'\bYYEAR\b','YEAR',sql,flags=re.I)
    sql = re.sub(r'\bMMONTH\b','MONTH',sql,flags=re.I)
    sql = re.sub(r'""+','"',sql)
    sql = re.sub(r'"\s*\.\s*\'?"', '"."', sql)
    sql = re.sub(r'\.\s*\.', '.', sql)
    sql = re.sub(r'"([A-Za-z_]\w*)"\s*"\s*([A-Za-z_]\w*)"', r'"\1"."\2"', sql)
    sql = re.sub(r',\s*(WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|FROM)\b', r' \1', sql, flags=re.I)
    sql = re.sub(r'>\s*>', '>', sql); sql = re.sub(r'<\s*<', '<', sql); sql = re.sub(r'=\s*=', '=', sql)
    sql = re.sub(r"(INTERVAL\s*'\s*\d+\s*)month\b", r"\1months", sql, flags=re.I)

    # collapse whitespace
    return re.sub(r'\s+', ' ', sql).strip()


def _repair_date_and_interval_literals_FINAL(sql: str, schema_catalog: Dict[str, List[str]]) -> str:
    # unify interval forms
    sql = re.sub(r'\bINTERVAL\s*([0-9]+)\s*(year|month|day|hour|minute|second)s?\b',
                 lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}s'", sql, flags=re.I)
    sql = re.sub(r'\b([0-9]+)(days?|months?|years?)\b',
                 lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}'", sql, flags=re.I)
    # duplicate literals
    sql = re.sub(r"'(\d{4}-\d{2}-\d{2})'\s*'(\d{4}-\d{2}-\d{2})'", r"'\1'", sql)
    # literal ± interval → DATE literal
    sql = re.sub(r"'(\d{4}-\d{2}-\d{2})'\s*([+-])\s*INTERVAL", r"DATE '\1' \2 INTERVAL", sql, flags=re.I)
    return sql


def _repair_date_columns_FINAL(sql: str, schema_catalog: dict) -> str:
    # collect likely date cols
    date_cols = set()
    for _, cols in (schema_catalog or {}).items():
        for c in cols:
            lc = c.lower()
            if any(x in lc for x in ['date','time','created','updated','funded','purchase','report','maturity','expiration']):
                date_cols.add(lc)
    if not date_cols:
        return sql

    # EXTRACT(unit FROM col) → CAST(col AS DATE)
    def fix_extract(m):
        unit, col = m.group(1), m.group(2).strip()
        base = col.split('.')[-1].replace('"','').lower()
        if base in date_cols:
            return f"EXTRACT({unit} FROM CAST({col} AS DATE))"
        return m.group(0)

    sql = re.sub(r'EXTRACT\s*\(\s*(\w+)\s+FROM\s+(["\w.]+)\s*\)', fix_extract, sql, flags=re.I)

    # date arithmetic col ± INTERVAL/DATE → CAST(col AS DATE) ± ...
    for base in date_cols:
        sql = re.sub(
            rf'([\w."]+\.{base})(?!\s*(?:::DATE|AS\s+DATE))\s*([+-])\s*(INTERVAL|DATE)',
            lambda m: f"CAST({m.group(1)} AS DATE) {m.group(2)} {m.group(3)}",
            sql, flags=re.I
        )
    return sql
import re
import logging
from difflib import get_close_matches
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# =========================
# Stage 2: Basic Syntax Cleanup (FIXED)
# =========================
import re
import logging
from difflib import get_close_matches
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# =========================
# Main Entry Point
# =========================
def _fix_aggregate_text_columns(sql: str, schema_catalog: Dict[str, List[str]], 
                                engine=None) -> str:
    """
    Cast TEXT columns to numeric when used in aggregate functions.
    Uses actual column types from database if engine is provided.
    """
    agg_funcs = ['AVG', 'SUM', 'MIN', 'MAX', 'STDDEV', 'VARIANCE']
    pattern = r'\b(' + '|'.join(agg_funcs) + r')\s*\(\s*"?(\w+)"?\s*\.\s*"?(\w+)"?\s*\)'
    
    # Build a map of columns that need casting (if we have engine access)
    text_columns = set()
    if engine:
        try:
            from sqlalchemy import inspect as sqla_inspect
            inspector = sqla_inspect(engine)
            for table_fqn in schema_catalog.keys():
                schema, table = table_fqn.replace('"', '').split('.')
                for col in inspector.get_columns(table, schema=schema):
                    if 'text' in str(col['type']).lower() or 'varchar' in str(col['type']).lower():
                        text_columns.add(col['name'].lower())
        except Exception as e:
            print(f"⚠️ Could not inspect column types: {e}")
    
    def cast_if_needed(match):
        func = match.group(1)
        alias = match.group(2)
        column = match.group(3)
        
        # Check if column is known to be TEXT
        if text_columns and column.lower() in text_columns:
            return f'{func}(CAST("{alias}"."{column}" AS NUMERIC))'
        
        # Fallback to heuristic
        numeric_keywords = ['score', 'vantage', 'fico', 'amount', 'rate', 'income',
                           'discount', 'principal', 'apr', 'dti', 'fee', 'payment',
                           'term', 'balance', 'value', 'price', 'cost']
        
        if any(kw in column.lower() for kw in numeric_keywords):
            return f'{func}(CAST("{alias}"."{column}" AS NUMERIC))'
        
        return match.group(0)
    
    return re.sub(pattern, cast_if_needed, sql, flags=re.I)

def _fix_extract_syntax(sql: str) -> str:
    """
    Fix malformed EXTRACT patterns:
    - EXTRACT(MONTH) FROM col → EXTRACT(MONTH FROM col)
    - (SELECT EXTRACT(...)) AS alias → EXTRACT(...) AS alias
    """
    # Fix: EXTRACT(unit) FROM column → EXTRACT(unit FROM column)
    sql = re.sub(
        r'EXTRACT\s*\(\s*(YEAR|MONTH|DAY|HOUR|MINUTE|SECOND|QUARTER|WEEK|DOW|DOY)\s*\)\s+FROM\s+',
        r'EXTRACT(\1 FROM ',
        sql,
        flags=re.I
    )
    
    # Fix: (SELECT EXTRACT(...) FROM col) AS alias → EXTRACT(...) AS alias
    sql = re.sub(
        r'\(\s*SELECT\s+(EXTRACT\s*\([^)]+\s+FROM\s+[^)]+\))\s*\)\s+AS\s+("?[\w\.]+"?)(?=\s*,|\s+FROM|\s+GROUP|\s+ORDER)',
        r'\1 AS \2',
        sql,
        flags=re.I
    )
    
    return sql


import re

# ------------------------------
# 6️⃣ Auto-Fix GROUP BY / ORDER BY Aliases
# ------------------------------
def _align_group_order_with_select(sql: str) -> str:
    """
    Ensures that GROUP BY and ORDER BY use the exact SELECT aliases.
    1. Parses SELECT expressions and their aliases.
    2. Rewrites GROUP BY / ORDER BY to use aliases where possible.
    """
    # Step 1: Extract SELECT list
    m = re.search(r'SELECT\s+(.*?)\s+FROM\b', sql, flags=re.I | re.S)
    if not m:
        return sql  # Can't parse SELECT

    select_list = m.group(1)
    # Build map of expression -> alias
    expr_alias_map = {}
    for item in select_list.split(','):
        item = item.strip()
        # Pattern: expr AS alias
        mm = re.match(r'(.+?)\s+AS\s+"?([\w]+)"?$', item, flags=re.I)
        if mm:
            expr, alias = mm.group(1).strip(), mm.group(2).strip()
            expr_alias_map[expr] = alias
        else:
            # No alias; just use expression
            expr_alias_map[item] = item

    # Step 2: Fix GROUP BY
    def replace_group_by(match):
        gb_list = match.group(1)
        parts = [p.strip() for p in gb_list.split(',')]
        fixed_parts = []
        for p in parts:
            for expr, alias in expr_alias_map.items():
                # If the GROUP BY expression matches a SELECT expression, use alias
                if p.lower() == expr.lower() or p.replace('CAST(','').replace(')','').lower() == expr.lower():
                    fixed_parts.append(alias)
                    break
            else:
                fixed_parts.append(p)
        return "GROUP BY " + ", ".join(fixed_parts)

    sql = re.sub(r'GROUP\s+BY\s+(.*?)(?=\bORDER\b|\bLIMIT\b|$)', replace_group_by, sql, flags=re.I|re.S)

    # Step 3: Fix ORDER BY similarly
    def replace_order_by(match):
        ob_list = match.group(1)
        parts = [p.strip() for p in ob_list.split(',')]
        fixed_parts = []
        for p in parts:
            for expr, alias in expr_alias_map.items():
                if p.lower() == expr.lower() or p.replace('CAST(','').replace(')','').lower() == expr.lower():
                    fixed_parts.append(alias)
                    break
            else:
                fixed_parts.append(p)
        return "ORDER BY " + ", ".join(fixed_parts)

    sql = re.sub(r'ORDER\s+BY\s+(.*?)(?=\bLIMIT\b|$)', replace_order_by, sql, flags=re.I|re.S)

    # Collapse spaces
    sql = re.sub(r'\s+', ' ', sql).strip()
    return sql


# ✅ ADD THIS NEW FUNCTION HERE:
def _fix_missing_paren_before_alias(sql: str) -> str:
    """Fix: (COUNT(...) * 100 AS x → (COUNT(...) * 100) AS x"""
    pattern = r'\(([^()]*(?:\([^()]*\)[^()]*)*?)\s*\*\s*(\d+(?:\.\d+)?)\s+(AS\s+\w+)'
    def fix_it(match):
        inner = match.group(1).strip()
        num = match.group(2)
        alias = match.group(3)
        return f'({inner} * {num}) {alias}'
    return re.sub(pattern, fix_it, sql, flags=re.I)

import re

import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# --- Minimal helpers to stabilize bad LLM SQL ---

def _safe_extract_sql(raw: Any) -> str:
    """
    Robustly extract SQL text from LLM output that might be a str/tuple/list/dict.
    Never returns None; always a string (possibly empty).
    """
    text = ""

    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, (tuple, list)):
        for item in raw:
            if isinstance(item, str) and item.strip():
                text = item
                break
        if not text:
            text = " ".join(str(x) for x in raw if x is not None)
    elif isinstance(raw, dict):
        for key in ("sql", "text", "content", "message", "output"):
            v = raw.get(key)
            if isinstance(v, str) and v.strip():
                text = v
                break
        if not text:
            text = str(raw)
    else:
        text = str(raw)

    # fenced block preference
    m = re.search(r'```(?:sql)?\s*(.*?)\s*```', text, flags=re.I | re.S)
    if m:
        text = m.group(1)

    # keep from first WITH/SELECT onward
    m = re.search(r'\b(WITH|SELECT)\b.*', text, flags=re.I | re.S)
    if m:
        text = m.group(0)

    # strip weird model headers
    text = re.sub(r'<\|header_start\|>\s*sql\s*<\|header_end\|>\s*', '', text, flags=re.I)

    return (text or "").strip()

def _fix_extract_and_wrapping(sql_text: str) -> str:
    """
    Fix common EXTRACT()/paren issues and remove a stray '(' before SELECT.
    """
    s = sql_text.strip()

    # Remove a single stray '(' before SELECT
    s = re.sub(r'^\(\s*(SELECT\b)', r'\1', s, flags=re.I)

    # 🔥 FIX 1: EXTRACT(YEAR) FROM ... → EXTRACT(YEAR FROM ...
    # This handles: EXTRACT(part) FROM expr
    s = re.sub(
        r'EXTRACT\s*\(\s*([A-Z_]+)\s*\)\s+FROM\s+',
        r'EXTRACT(\1 FROM ',
        s,
        flags=re.I
    )

    # 🔥 FIX 2: EXTRACT(YEAR FROM col) extra paren → EXTRACT(YEAR FROM col)
    # This handles: EXTRACT(YEAR FROM col)) AS ... (extra closing paren)
    s = re.sub(
        r'EXTRACT\s*\(\s*([A-Z_]+)\s+FROM\s+([^)]+)\)\s*\)\s+AS\b',
        r'EXTRACT(\1 FROM \2) AS',
        s,
        flags=re.I
    )

    # 🔥 FIX 3: Missing opening paren in EXTRACT
    # EXTRACT(YEAR) FROM "a"."col") → EXTRACT(YEAR FROM "a"."col")
    s = re.sub(
        r'EXTRACT\s*\(\s*([A-Z_]+)\s*\)\s+FROM\s+([^)]+?)\)\s+AS\b',
        r'EXTRACT(\1 FROM \2) AS',
        s,
        flags=re.I
    )

    # Drop accidental "AS DATE" right after EXTRACT(...)
    s = re.sub(
        r'(EXTRACT\s*\(\s*[A-Za-z_]+\s+FROM\s+[^)]+\))\s+AS\s+DATE\b',
        r'\1',
        s,
        flags=re.I
    )

    return s


def _fix_all_extract_syntax(sql: str) -> str:
    """
    Comprehensively fix all EXTRACT syntax issues.
    Handles all common LLM mistakes with EXTRACT functions.
    """
    
    # Pattern 1: EXTRACT(YEAR) FROM col) → EXTRACT(YEAR FROM col)
    # Remove extra closing paren after column
    sql = re.sub(
        r'EXTRACT\s*\(\s*([A-Z_]+)\s*\)\s+FROM\s+([^\)]+?)\)\s+AS',
        r'EXTRACT(\1 FROM \2) AS',
        sql,
        flags=re.I
    )
    
    # Pattern 2: EXTRACT(YEAR FROM col)) → EXTRACT(YEAR FROM col)
    # Remove duplicate closing paren
    sql = re.sub(
        r'EXTRACT\s*\(\s*([A-Z_]+)\s+FROM\s+([^)]+)\)\s*\)',
        r'EXTRACT(\1 FROM \2)',
        sql,
        flags=re.I
    )
    
    # Pattern 3: EXTRACT(YEAR) FROM → EXTRACT(YEAR FROM
    # Fix missing FROM inside parentheses
    sql = re.sub(
        r'EXTRACT\s*\(\s*([A-Z_]+)\s*\)\s+FROM\s+',
        r'EXTRACT(\1 FROM ',
        sql,
        flags=re.I
    )
    
    return sql


def _fix_groupby_alias_and_parens(sql_text: str) -> str:
    """
    1) Remove '... ) AS DATE' fragments mistakenly placed in GROUP BY lists.
    2) If SELECT list has more '(' than ')', close them.
    """
    s = sql_text

    # Remove "... ) AS DATE" inside GROUP BY lists
    s = re.sub(
        r'(\bGROUP\s+BY\s+.*?\))\s+AS\s+\w+\s*\)?',
        r'\1',
        s, flags=re.I | re.S
    )

    # If SELECT...FROM segment has more '(' than ')', close them
    m = re.search(r'^\s*SELECT\s+(.*?)\s+FROM\b', s, flags=re.I | re.S)
    if m:
        seg = m.group(1)
        diff = seg.count('(') - seg.count(')')
        if diff > 0:
            s = s.replace(seg, seg + (')' * diff), 1)

    return s

import re

def _fix_groupby_orderby_glue(sql_text: str) -> str:
    s = sql_text

    # Case A: the classic glue: ... GROUP BY "a"."channel_grouporder" BY ...
    # Turn it into: ... GROUP BY "a"."channel_group" ORDER BY ...
    s = re.sub(
        r'(\bGROUP\s+BY\s+[^;]*?)("?[A-Za-z_][A-Za-z0-9_]*"?\.)?"?([A-Za-z_][A-Za-z0-9_]*)order"?\s+BY',
        lambda m: f'{m.group(1)}{(m.group(2) or "")}"{m.group(3)}" ORDER BY',
        s,
        flags=re.I
    )

    # Case B: an extra safety for the most common real token seen in your logs
    s = s.replace('"channel_grouporder" BY', '"channel_group" ORDER BY')

    return s

import re

import re

def _normalize_order_by(sql: str) -> str:
    if not sql:
        return sql

    s = sql

    # 1) Unglue common keyword smash-ups
    #    e.g. "ORDERBY" → "ORDER BY", "BYORDER" → "BY ORDER"
    s = re.sub(r'(?i)\borderby\b', 'ORDER BY', s)
    s = re.sub(r'(?i)\bbyorder\b', 'BY ORDER', s)

    # 2) Specific fix: "ORDER BYORDER BY" → "ORDER BY "
    s = re.sub(r'(?i)\border\s*by\s*order\s*by\b', 'ORDER BY ', s)

    # 3) If multiple ORDER BY blocks exist, keep ONLY the last one
    parts = re.split(r'(?i)\bORDER\s+BY\b', s)
    if len(parts) > 2:
        # parts = [before, list1..., listN]
        head = parts[0].rstrip()
        tail = parts[-1].lstrip()
        s = (head + ' ORDER BY ' + tail).strip()

    # 4) Remove empty ORDER BY (no column/expression after it)
    s = re.sub(r'(?i)\bORDER\s+BY\s*(?=;|\Z)', '', s)

    # 5) Normalize extra spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s


import re

def _fix_glued_group_order(sql: str) -> str:
    s = sql

    # A. Exact glue you’re seeing:  "xxxorder" BY  ->  "xxx" ORDER BY
    s = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)order"\s+BY', r'"\1" ORDER BY', s, flags=re.I)
    s = re.sub(r'(\b[A-Za-z_][A-Za-z0-9_]*?)order"\s+BY', r'\1" ORDER BY', s, flags=re.I)  # unquoted alias before quote

    # B. Unglued but missing space:   )order by   ->   ) ORDER BY
    s = re.sub(r'\)(?=order\s+by\b)', r') ', s, flags=re.I)

    # C. Identifier glued to ORDER (unquoted):  colorder by  ->  col ORDER BY
    s = re.sub(r'(\b[A-Za-z_][A-Za-z0-9_]*)(?=order\s+by\b)', r'\1 ', s, flags=re.I)

    # D. Totally smashed "orderby" -> "ORDER BY"
    s = re.sub(r'\borderby\b', ' ORDER BY ', s, flags=re.I)

    # E. Safety: if the token just before ORDER BY ends with ...order, drop the tail "order"
    def _unglue_trailing_order(m):
        inner = m.group(1)
        inner = re.sub(r'("?[\w\.]+"\s*|\b[\w\.]+\s*)order\s*$', lambda mm: mm.group(0)[:-5], inner, flags=re.I)
        inner = re.sub(r'\s+$', '', inner)
        return f'GROUP BY {inner} ORDER BY'
    s = re.sub(r'\bGROUP\s+BY\s+(.*?)(?=\border\s+by\b)', _unglue_trailing_order, s, flags=re.I | re.S)

    # F. Collapse extra spaces introduced above
    s = re.sub(r'\s+', ' ', s).strip()
    return s



import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ------------------------------
# Helper Functions
# ------------------------------

def _safe_extract_sql(raw: Any) -> str:
    """Extract SQL string safely from LLM output or other inputs."""
    if isinstance(raw, str):
        return raw
    return str(raw)

def _generate_safe_fallback_sql(schema_catalog: Dict[str, List[str]], fallback_table: Optional[str] = None) -> str:
    """Return safe fallback SQL if repair fails."""
    table = fallback_table or (list(schema_catalog.keys())[0] if schema_catalog else 'dummy_table')
    return f'SELECT COUNT(*) AS row_count FROM "{table}" LIMIT 1;'

def _cleanup_basic_syntax_FINAL(sql: str) -> str:
    """Basic cleanup: comments, duplicate keywords, spacing, intervals, etc."""
    sql = re.sub(r'--[^\n]*', '', sql)
    sql = re.sub(r'\s+', ' ', sql).strip()
    sql = re.sub(r"(INTERVAL\s*'\s*\d+\s*)month\b", r"\1months", sql, flags=re.I)
    sql = re.sub(r'\.\s*\.', '.', sql)
    sql = re.sub(r'"([A-Za-z_]\w*)"\s*"\s*([A-Za-z_]\w*)"', r'"\1"."\\2"', sql)
    return sql

def _fix_aggregate_text_columns(sql: str, schema_catalog: Dict[str, List[str]]) -> str:
    """Cast text columns to NUMERIC when used in aggregates."""
    agg_funcs = ['AVG', 'SUM', 'MIN', 'MAX', 'STDDEV', 'VARIANCE']
    pattern = r'\b(' + '|'.join(agg_funcs) + r')\s*\(\s*"?(\w+)"?\s*\.\s*"?(\w+)"?\s*\)'

    numeric_keywords = ['score', 'vantage', 'fico', 'amount', 'rate', 'income', 'discount', 'principal', 'apr', 'dti', 'fee', 'payment', 'term', 'balance', 'value', 'price', 'cost']

    def cast_if_needed(match):
        func, alias, column = match.groups()
        if any(kw in column.lower() for kw in numeric_keywords):
            return f'{func}(CAST("{alias}"."{column}" AS NUMERIC))'
        return match.group(0)

    return re.sub(pattern, cast_if_needed, sql, flags=re.I)

def _repair_date_columns_FINAL(sql: str, schema_catalog: dict) -> str:
    """Cast likely date columns to DATE and fix EXTRACT calls."""
    date_cols = {c.lower() for cols in schema_catalog.values() for c in cols if any(x in c.lower() for x in ['date','time','created','updated','funded','purchase','report','maturity','expiration'])}

    def fix_extract(m):
        unit, col = m.group(1), m.group(2).strip()
        base = col.split('.')[-1].replace('"','').lower()
        if base in date_cols:
            return f"EXTRACT({unit} FROM CAST({col} AS DATE))"
        return m.group(0)

    sql = re.sub(r'EXTRACT\s*\(\s*(\w+)\s+FROM\s+(["\w.]+)\s*\)', fix_extract, sql, flags=re.I)
    return sql

def _repair_date_and_interval_literals_FINAL(sql: str, schema_catalog: Dict[str, List[str]]) -> str:
    """Normalize INTERVAL literals and date arithmetic."""
    sql = re.sub(r'\bINTERVAL\s*([0-9]+)\s*(year|month|day|hour|minute|second)s?\b', lambda m: f"INTERVAL '{m.group(1)} {m.group(2)}s'", sql, flags=re.I)
    sql = re.sub(r"'(\d{4}-\d{2}-\d{2})'\s*([+-])\s*INTERVAL", r"DATE '\1' \2 INTERVAL", sql, flags=re.I)
    return sql

def _normalize_order_by(sql: str) -> str:
    sql = re.sub(r'\borderby\b', 'ORDER BY', sql, flags=re.I)
    sql = re.sub(r'\bbyorder\b', 'BY ORDER', sql, flags=re.I)
    sql = re.sub(r'(?i)\bORDER\s+BY\s*(?=;|\Z)', '', sql)
    sql = re.sub(r'\s+', ' ', sql).strip()
    return sql

def _fix_extract_aliases(sql: str) -> str:
    # Fix EXTRACT syntax and remove extra ')'
    def repl(m):
        unit = m.group(1).upper()
        col = m.group(2).strip().rstrip(')')
        return f"EXTRACT({unit} FROM CAST({col} AS DATE))"
    sql = re.sub(
        r'EXTRACT\s*\(\s*(YEAR|MONTH|DAY|QUARTER|WEEK|DOW)\s+FROM\s+([^\)]+)\)?',
        repl, sql, flags=re.I
    )
    return sql

def _align_group_order_with_select(sql: str) -> str:
    """Rewrite GROUP BY and ORDER BY to match SELECT aliases exactly."""
    m = re.search(r'SELECT\s+(.*?)\s+FROM\b', sql, flags=re.I|re.S)
    if not m:
        return sql

    select_list = m.group(1)
    expr_alias_map = {}
    for item in re.split(r',\s*(?![^()]*\))', select_list):
        item = item.strip()
        mm = re.match(r'(.+?)\s+AS\s+"?([\w]+)"?$', item, flags=re.I)
        if mm:
            expr_alias_map[mm.group(1).strip()] = mm.group(2).strip()
        else:
            # fallback: use last identifier as alias
            alias = item.split()[-1].replace('"','')
            expr_alias_map[item] = alias

    def fix_clause(match):
        clause = match.group(1)
        parts = [p.strip() for p in re.split(r',\s*(?![^()]*\))', clause)]
        fixed = []
        for p in parts:
            for expr, alias in expr_alias_map.items():
                # ignore CAST() wrapper
                clean_p = p.replace('CAST(','').replace(')','').strip()
                if clean_p.lower() == expr.lower():
                    fixed.append(alias)
                    break
            else:
                fixed.append(p)
        return f"{match.group(0).split()[0]} " + ', '.join(fixed)

    sql = re.sub(r'GROUP\s+BY\s+(.*?)(?=\bORDER\b|\bLIMIT\b|$)', fix_clause, sql, flags=re.I|re.S)
    sql = re.sub(r'ORDER\s+BY\s+(.*?)(?=\bLIMIT\b|$)', fix_clause, sql, flags=re.I|re.S)
    return sql


def robust_sql_repair(sql: str, schema_catalog: Optional[Dict[str, List[str]]] = None) -> str:
    """
    One-pass repair for LLM-generated SQL:
    - Fix EXTRACT parentheses / alias issues
    - Fix GROUP BY / ORDER BY alias mismatches and glued tokens
    - Replace _id columns with fallback name mapping
    - CAST numeric-like columns in aggregates
    - Cleanup spacing / semicolons
    """
    s = sql.strip()
    
    # 1️⃣ Remove leading stray '(' before SELECT
    s = re.sub(r'^\(\s*(SELECT\b)', r'\1', s, flags=re.I)
    s = re.sub(r'([^\s]+)\)\s+AS\s+([A-Za-z_]\w*)', r'\1 AS \2', s, flags=re.I)
    
    # 2️⃣ Fix EXTRACT syntax
    s = re.sub(
        r'EXTRACT\s*\(\s*(YEAR|MONTH|DAY|QUARTER|WEEK|DOW)\s+FROM\s+([^\)]+)\)?',
        lambda m: f"EXTRACT({m.group(1).upper()} FROM CAST({m.group(2).strip().rstrip(')')} AS DATE))",
        s,
        flags=re.I
    )
    
    # 3️⃣ Fix glued GROUP BY / ORDER BY tokens
    s = re.sub(r'\)(?=order\s+by\b)', r') ', s, flags=re.I)
    s = re.sub(r'(\))(?=ORDER\s+BY\b)', r'\1 ', s, flags=re.I)
    s = re.sub(r'(\b[A-Za-z_][A-Za-z0-9_]*)order(?=\s+BY\b)', r'\1 ORDER', s, flags=re.I)
    s = re.sub(r'\borderby\b', 'ORDER BY', s, flags=re.I)
    s = re.sub(
        r'\bGROUP\s+BY\s+(.*?)(?=\bORDER\s+BY\b|$)',
        lambda m: 'GROUP BY ' + m.group(1).replace(')ORDER', ') ORDER'),
        s,
        flags=re.I | re.S
    )
    
    # 4️⃣ Extract SELECT list and map aliases (MUST come before fix_clause definition!)
    expr_alias_map = {}
    m = re.search(r'SELECT\s+(.*?)\s+FROM\b', s, flags=re.I | re.S)
    if m:
        select_list = m.group(1)
        for item in re.split(r',\s*(?![^()]*\))', select_list):
            item = item.strip()
            mm = re.match(r'(.+?)\s+AS\s+"?([\w]+)"?$', item, flags=re.I)
            if mm:
                expr_alias_map[mm.group(1).strip()] = mm.group(2).strip()
            else:
                alias = item.split()[-1].replace('"', '')
                expr_alias_map[item] = alias
    
    # 5️⃣ Fix GROUP BY / ORDER BY to match SELECT aliases
    def fix_clause(match):
        clause = match.group(1)
        parts = [p.strip() for p in re.split(r',\s*(?![^()]*\))', clause)]
        fixed = []
        for p in parts:
            clean_p = p.replace('CAST(', '').replace(')', '').strip()
            found = False
            for expr, alias in expr_alias_map.items():
                if clean_p.lower() == expr.lower():
                    fixed.append(alias)
                    found = True
                    break
            if not found:
                fixed.append(p)
        return f"{match.group(0).split()[0]} " + ', '.join(fixed)
    
    s = re.sub(r'GROUP\s+BY\s+(.*?)(?=\bORDER\b|\bLIMIT\b|$)', fix_clause, s, flags=re.I | re.S)
    s = re.sub(r'ORDER\s+BY\s+(.*?)(?=\bLIMIT\b|$)', fix_clause, s, flags=re.I | re.S)
    
    # 6️⃣ Apply fallback map for _id → _name columns
    if schema_catalog:
        fallback_map = {}
        for table, cols in schema_catalog.items():
            for col in cols:
                if col.endswith('_id'):
                    name_col = col[:-3] + '_name'
                    if name_col in cols:  # Only map if name column exists
                        fallback_map[f'"{table}"."{col}"'] = f'"{table}"."{name_col}"'
        
        print(f"✅ Final fallback_map: {fallback_map}")
        for k, v in fallback_map.items():
            s = re.sub(re.escape(k), v, s)
    
    # 7️⃣ Cast numeric-like columns in aggregates
    numeric_keywords = ['score', 'vantage', 'fico', 'amount', 'rate', 'income', 
                       'discount', 'principal', 'apr', 'dti', 'fee', 'payment', 
                       'term', 'balance', 'value', 'price', 'cost']
    agg_funcs = ['AVG', 'SUM', 'MIN', 'MAX', 'STDDEV', 'VARIANCE']
    pattern = r'\b(' + '|'.join(agg_funcs) + r')\s*\(\s*("?[\w]+"?)\s*\.\s*("?[\w]+"?)\s*\)'
    
    def cast_if_needed(match):
        func, alias, col = match.groups()
        col_clean = col.replace('"', '').lower()
        if any(kw in col_clean for kw in numeric_keywords):
            return f'{func}(CAST({alias}.{col} AS NUMERIC))'
        return match.group(0)
    
    s = re.sub(pattern, cast_if_needed, s, flags=re.I)
    
    # 8️⃣ Final cleanup
    s = re.sub(r'\s+', ' ', s).strip()
    if not s.endswith(';'):
        s += ';'
    
    print(f"✅ Final auto-fixed SQL: {s}")
    return s
# ------------------------------
# Unified SQL Repair
# ------------------------------
# ------------------------------
# Unified SQL Repair (Updated)
# ------------------------------
import re
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_extract_sql(raw: Any) -> str:
    """Extract SQL string safely from any input format."""
    if raw is None:
        return ""
    
    if isinstance(raw, (tuple, list)):
        if len(raw) > 0 and raw[0]:
            return str(raw[0]).strip()
        return ""
    
    if isinstance(raw, dict):
        return raw.get('sql', raw.get('query', str(raw)))
    
    if hasattr(raw, 'sql'):
        return str(raw.sql)
    
    sql = str(raw).strip()
    if sql.startswith("('") and sql.endswith("')"):
        sql = sql[2:-2]
    
    return sql


def _generate_safe_fallback_sql(schema_catalog: Dict[str, List[str]], fallback_table: Optional[str] = None) -> str:
    """Return safe fallback SQL if repair fails."""
    table = fallback_table or (list(schema_catalog.keys())[0] if schema_catalog else 'dummy_table')
    return f'SELECT COUNT(*) AS row_count FROM {table} LIMIT 1;'


def _balance_parentheses(text: str) -> str:
    """Ensure parentheses are balanced in a text fragment."""
    open_count = text.count('(')
    close_count = text.count(')')
    
    if open_count > close_count:
        # Add missing closing parentheses
        text += ')' * (open_count - close_count)
    elif close_count > open_count:
        # Remove extra closing parentheses from the end
        diff = close_count - open_count
        for _ in range(diff):
            text = text.rstrip()
            if text.endswith(')'):
                text = text[:-1]
    
    return text


def _detect_join_keys(left_table: str, right_table: str, schema_catalog: Dict[str, List[str]]) -> list[tuple[str, str]]:
    """
    Auto-detect join keys between two tables based on _id columns.
    Returns a list of (left_col, right_col) tuples.
    """
    if not schema_catalog or left_table not in schema_catalog or right_table not in schema_catalog:
        return []
    
    left_cols = schema_catalog[left_table]
    right_cols = schema_catalog[right_table]
    
    join_keys = []
    for l_col in left_cols:
        if l_col.endswith('_id'):
            r_col = l_col  # same column name in right table
            if r_col in right_cols:
                join_keys.append((l_col, r_col))
    return join_keys


def _fix_extract_syntax(sql: str) -> str:
    """Fix EXTRACT syntax issues comprehensively - this is the critical function."""
    
    # CRITICAL FIX: Handle the malformed pattern first
    # Pattern: EXTRACT(UNIT) FROM col) with extra closing paren
    # This needs to be fixed BEFORE we add any CAST operations
    sql = re.sub(
        r'EXTRACT\s*\(\s*(YEAR|MONTH|DAY|QUARTER|WEEK|DOW|HOUR|MINUTE|SECOND)\s*\)\s+FROM\s+([a-zA-Z_][\w.]*)\s*\)',
        lambda m: f"EXTRACT({m.group(1)} FROM {m.group(2)})",
        sql,
        flags=re.I
    )
    
    # Now handle proper EXTRACT patterns
    # Pattern: EXTRACT(UNIT FROM col) without CAST - needs CAST added
    # But DON'T match if it's already inside a CAST or if it already has CAST
    def add_cast_to_extract(match):
        full_match = match.group(0)
        unit = match.group(1)
        column = match.group(2)
        
        # Check if column is already CAST
        if 'CAST(' in column.upper():
            return full_match
        
        # Add CAST around the column
        return f"EXTRACT({unit} FROM CAST({column} AS DATE))"
    
    # Match EXTRACT that doesn't already have CAST inside
    sql = re.sub(
        r'EXTRACT\s*\(\s*(YEAR|MONTH|DAY|QUARTER|WEEK|DOW|HOUR|MINUTE|SECOND)\s+FROM\s+([a-zA-Z_][\w.]*)\s*\)',
        add_cast_to_extract,
        sql,
        flags=re.I
    )
    
    return sql


def _extract_alias_map(sql: str) -> Dict[str, str]:
    """Extract alias mappings from SELECT clause."""
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM\b', sql, flags=re.I | re.S)
    alias_map = {}
    
    if not select_match:
        return alias_map
    
    select_clause = select_match.group(1)
    
    # Split by comma outside of parentheses
    items = []
    paren_depth = 0
    current_item = []
    
    for char in select_clause:
        if char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
        elif char == ',' and paren_depth == 0:
            items.append(''.join(current_item).strip())
            current_item = []
            continue
        current_item.append(char)
    
    if current_item:
        items.append(''.join(current_item).strip())
    
    # Extract aliases
    for item in items:
        # Match: expression AS alias
        as_match = re.search(r'\s+AS\s+"?([a-zA-Z_]\w*)"?\s*$', item, flags=re.I)
        if as_match:
            alias = as_match.group(1)
            expression = item[:as_match.start()].strip()
            
            # Balance parentheses in expression
            expression = _balance_parentheses(expression)
            
            # Store mapping from expression to alias
            alias_map[expression.lower()] = alias
            
            # Also store variations without outer CAST
            expr_no_cast = re.sub(r'^CAST\s*\((.+)\s+AS\s+\w+\)$', r'\1', expression, flags=re.I)
            if expr_no_cast != expression:
                alias_map[expr_no_cast.lower().strip()] = alias
            
            # Store EXTRACT expressions without CAST wrapper
            extract_no_cast = re.sub(
                r'EXTRACT\s*\(\s*(\w+)\s+FROM\s+CAST\s*\(([^)]+)\s+AS\s+\w+\)\s*\)',
                r'EXTRACT(\1 FROM \2)',
                expression,
                flags=re.I
            )
            if extract_no_cast != expression:
                alias_map[extract_no_cast.lower().strip()] = alias
    
    return alias_map


def robust_sql_repair(sql: str, schema_catalog: Optional[Dict[str, List[str]]] = None) -> str:
    """
    Clean, simple SQL repair that preserves structure and ensures balanced parentheses.
    """
    if not sql or not isinstance(sql, str):
        return ""
    
    s = sql.strip()
    
    # Step 1: Basic cleanup
    s = re.sub(r'^\(\s*', '', s)  # Remove leading (
    s = re.sub(r'\s*\)$', '', s)  # Remove trailing )
    s = re.sub(r'\s+', ' ', s)    # Collapse whitespace
    
    # Step 2: Fix EXTRACT syntax issues FIRST (before any other modifications)
    s = _fix_extract_syntax(s)
    
    # Step 3: Fix broken CAST statements
    # Remove double CAST: CAST(CAST(col AS type) AS type2) -> CAST(col AS type2)
    max_iterations = 5
    iteration = 0
    while 'CAST(CAST(' in s.upper() and iteration < max_iterations:
        s = re.sub(
            r'CAST\s*\(\s*CAST\s*\(([^)]+)\s+AS\s+\w+\)\s+AS\s+(\w+)\)',
            r'CAST(\1 AS \2)',
            s,
            flags=re.I
        )
        iteration += 1
    
    # Step 4: Fix aggregate functions with missing closing parens
    # AVG(l.vantage4 AS -> AVG(l.vantage4) AS
    s = re.sub(
        r'(AVG|SUM|MIN|MAX|COUNT|STDDEV|VARIANCE)\s*\(\s*([a-zA-Z_][\w.]*)\s+AS\b',
        r'\1(\2) AS',
        s,
        flags=re.I
    )
    
        # Step 5: Fix glued keywords
    s = re.sub(r'\)ORDER\s+BY\b', ') ORDER BY', s, flags=re.I)
    s = re.sub(r'\)GROUP\s+BY\b', ') GROUP BY', s, flags=re.I)
    s = re.sub(r'\bGROUPBY\b', 'GROUP BY', s, flags=re.I)
    s = re.sub(r'\bORDERBY\b', 'ORDER BY', s, flags=re.I)

    # ✅ Step 5b: Ensure space between GROUP BY / ORDER BY and previous token
    s = re.sub(r'(GROUP BY)([a-zA-Z0-9_"])', r'\1 \2', s, flags=re.I)
    s = re.sub(r'(ORDER BY)([a-zA-Z0-9_"])', r'\1 \2', s, flags=re.I)
    
    # Step 6: Fix INTERVAL syntax
    s = re.sub(r"INTERVAL\s+'?(\d+)\s+(year|month|day|hour|minute|second)s?'?", r"INTERVAL '\1 \2s'", s, flags=re.I)
    
    # Step 7: Extract alias map from SELECT clause
    alias_map = _extract_alias_map(s)
    
    # Step 8: Fix GROUP BY to use aliases and ensure balanced parentheses
    def fix_group_by(match):
        clause = match.group(1).strip()
        if not clause:
            return match.group(0)
        
        parts = []
        paren_depth = 0
        current_part = []
        
        for char in clause:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                part = ''.join(current_part).strip()
                parts.append(_balance_parentheses(part))
                current_part = []
                continue
            current_part.append(char)
        
        if current_part:
            part = ''.join(current_part).strip()
            parts.append(_balance_parentheses(part))
        
        fixed_parts = []
        for part in parts:
            part_lower = part.lower().strip()
            
            # Try to match with alias map
            matched = False
            for expr_key, alias in alias_map.items():
                if part_lower == expr_key:
                    fixed_parts.append(alias)
                    matched = True
                    break
            
            if not matched:
                # Ensure this part has balanced parentheses
                fixed_parts.append(_balance_parentheses(part))
        
        return 'GROUP BY ' + ', '.join(fixed_parts)
    
    s = re.sub(r'GROUP\s+BY\s+(.+?)(?=\s+ORDER\s+BY|\s+HAVING|\s+LIMIT|;|$)', fix_group_by, s, flags=re.I | re.S)
    
    # Step 9: Fix ORDER BY to use aliases
    def fix_order_by(match):
        clause = match.group(1).strip()
        if not clause:
            return match.group(0)
        
        parts = [p.strip() for p in clause.split(',')]
        fixed_parts = []
        
        for part in parts:
            # Check for ASC/DESC
            order_match = re.match(r'(.+?)\s+(ASC|DESC)\s*$', part, flags=re.I)
            if order_match:
                expr = order_match.group(1).strip().lower()
                direction = order_match.group(2)
                
                # Try to find alias
                matched = False
                for expr_key, alias in alias_map.items():
                    if expr == expr_key:
                        fixed_parts.append(f"{alias} {direction}")
                        matched = True
                        break
                
                if not matched:
                    fixed_parts.append(part)
            else:
                expr = part.lower().strip()
                
                # Try to find alias
                matched = False
                for expr_key, alias in alias_map.items():
                    if expr == expr_key:
                        fixed_parts.append(alias)
                        matched = True
                        break
                
                if not matched:
                    fixed_parts.append(part)
        
        return 'ORDER BY ' + ', '.join(fixed_parts)
    
    s = re.sub(r'ORDER\s+BY\s+(.+?)(?=\s+LIMIT|;|$)', fix_order_by, s, flags=re.I | re.S)
    
    # Step 10: Apply _id to _name fallback mapping
    if schema_catalog:
        for table, cols in schema_catalog.items():
            cols_lower = {c.lower(): c for c in cols}
            for col in cols:
                if col.endswith('_id'):
                    name_col = col[:-3] + '_name'
                    if name_col.lower() in cols_lower:
                        # Replace table.col_id with table.col_name
                        patterns = [
                            (f'"{table}"."{col}"', f'"{table}"."{name_col}"'),
                            (f'{table}.{col}', f'{table}.{name_col}'),
                            (f'"{table}".{col}', f'"{table}".{name_col}'),
                        ]
                        for old, new in patterns:
                            s = s.replace(old, new)
    
    # Step 11: Cast numeric columns in aggregates
    numeric_keywords = ['score', 'vantage', 'fico', 'amount', 'rate', 'income', 
                       'discount', 'principal', 'apr', 'dti', 'fee', 'payment', 
                       'term', 'balance', 'value', 'price', 'cost']
    
    def add_numeric_cast(match):
        func = match.group(1)
        col_ref = match.group(2)
        
        # Don't cast if already wrapped in CAST or if it's COUNT
        if 'CAST(' in match.group(0) or func.upper() == 'COUNT':
            return match.group(0)
        
        # Check if column name suggests numeric type
        col_name = col_ref.split('.')[-1].replace('"', '').lower()
        if any(kw in col_name for kw in numeric_keywords):
            return f'{func}(CAST({col_ref} AS NUMERIC))'
        
        return match.group(0)
    
    pattern = r'\b(AVG|SUM|MIN|MAX|STDDEV|VARIANCE)\s*\(\s*([a-zA-Z_][\w."]*)\s*\)'
    s = re.sub(pattern, add_numeric_cast, s, flags=re.I)
    
    # Step 11b: Cast date columns in WHERE clause comparisons
    # Pattern: column_with_date_name >= or <= or > or < or = date/timestamp expression
    date_keywords = ['date', 'time', 'created', 'updated', 'modified', 'deleted', 
                     'added', 'timestamp', 'datetime', 'start', 'end', 'expiry', 
                     'expiration', 'due', 'maturity', 'first', 'last', 'funded',
                     'purchased', 'noted', 'boarding', 'confirmed', 'sent']
    
    def add_date_cast_in_where(match):
        col_ref = match.group(1)
        operator = match.group(2)
        rhs = match.group(3)
        
        # Don't cast if already wrapped in CAST or EXTRACT
        if 'CAST(' in col_ref or 'EXTRACT(' in col_ref.upper():
            return match.group(0)
        
        # Check if column name suggests date type
        col_name = col_ref.split('.')[-1].replace('"', '').lower()
        
        # Check if RHS is a date expression (CURRENT_DATE, interval, date literal, etc.)
        rhs_upper = rhs.upper()
        is_date_rhs = any(keyword in rhs_upper for keyword in ['CURRENT_DATE', 'CURRENT_TIMESTAMP', 'NOW()', 'INTERVAL', '::DATE', '::TIMESTAMP'])
        is_date_rhs = is_date_rhs or re.match(r"'\d{4}-\d{2}-\d{2}", rhs)  # Date literal like '2024-01-01'
        
        if any(kw in col_name for kw in date_keywords) and is_date_rhs:
            return f'CAST({col_ref} AS DATE) {operator} {rhs}'
        
        return match.group(0)
    
    # Match: column >= or <= or > or < or = (expression)
    # This handles cases like: a.applicationdate >= (CURRENT_DATE - INTERVAL '12 months')
    where_pattern = r'\b([a-zA-Z_][\w."]*)\s*(>=|<=|>|<|=)\s*(\([^)]+\)|CURRENT_DATE|CURRENT_TIMESTAMP|NOW\(\)|\'[^\']+\')'
    s = re.sub(where_pattern, add_date_cast_in_where, s, flags=re.I)
    
    # Step 12: Remove excessive closing parentheses patterns
    # Fix pattern: expression))) AS alias -> expression) AS alias
    s = re.sub(r'\)\)\)+(\s+AS\s+)', r')\1', s, flags=re.I)
    
    # Fix pattern: expression)))) in various positions (but be careful not to break valid nested functions)
    iteration = 0
    max_iter = 10
    while ')))' in s and iteration < max_iter:
        s = re.sub(r'\)\)\)', '))', s)
        iteration += 1
    
    # Step 13: Final global balance check
    open_parens = s.count('(')
    close_parens = s.count(')')
    
    if open_parens > close_parens:
        # Add missing closing parentheses at the end (before semicolon if exists)
        if s.endswith(';'):
            s = s[:-1] + ')' * (open_parens - close_parens) + ';'
        else:
            s += ')' * (open_parens - close_parens)
    elif close_parens > open_parens:
        # More complex: need to find and remove extra closing parens
        # Try to remove from end first
        diff = close_parens - open_parens
        for _ in range(diff):
            # Look for unnecessary closing parens before semicolon or at end
            if s.endswith(');'):
                # Check if we can remove a paren before semicolon
                before_semi = s[:-2]
                if before_semi.rstrip().endswith(')'):
                    # Count parens in the part before this
                    temp = before_semi.rstrip()[:-1]
                    if temp.count('(') >= temp.count(')'):
                        before_semi = temp
                        s = before_semi.rstrip() + ';'
            elif s.endswith(')'):
                temp = s.rstrip()[:-1]
                if temp.count('(') >= temp.count(')'):
                    s = temp
    
    # Step 14: Final cleanup
    s = re.sub(r'\s+', ' ', s).strip()
    if not s.endswith(';'):
        s += ';'
    
    return s


def unified_sql_repair(
    raw_llm_output: Any,
    schema_catalog: Dict[str, List[str]],
    question: str = "",
    fallback_table: Optional[str] = None
) -> str:
    """
    Main entry point for SQL repair pipeline.
    """
    original_input = str(raw_llm_output)[:300] if raw_llm_output else "None"
    
    try:
        # Extract SQL from any format
        sql = _safe_extract_sql(raw_llm_output)
        
        if not sql or len(sql.strip()) < 10:
            logger.warning(f"Input SQL too short or empty: {original_input}")
            return _generate_safe_fallback_sql(schema_catalog, fallback_table)
        
        # Run repair
        sql = robust_sql_repair(sql, schema_catalog)
        
        # Validate result
        if not sql or len(sql.strip()) < 10 or 'SELECT' not in sql.upper():
            logger.warning(f"Repaired SQL invalid: {sql}")
            return _generate_safe_fallback_sql(schema_catalog, fallback_table)
        
        # Final validation: check parentheses are balanced
        if sql.count('(') != sql.count(')'):
            logger.warning(f"Unbalanced parentheses in repaired SQL: {sql}")
            sql = _balance_parentheses(sql)
        
        print(f"✅ Final repaired SQL: {sql}")
        return sql
        
    except Exception as e:
        logger.error(f"SQL repair failed: {e}. Input was: {original_input}")
        logger.exception(e)
        return _generate_safe_fallback_sql(schema_catalog, fallback_table)


def ensure_sql_string(sql_output: Any) -> str:
    """
    Final safety wrapper before SQL execution.
    """
    sql = _safe_extract_sql(sql_output)
    sql = sql.replace("\\'", "'").replace('\\"', '"')
    return sql.strip()
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import RegisterSerializer, LoginSerializer, UserSerializer
from django.contrib.auth import get_user_model

User = get_user_model()


@api_view(['POST'])
@permission_classes([AllowAny])
def register_view(request):
    """
    Register a new user
    """
    serializer = RegisterSerializer(data=request.data)
    
    if serializer.is_valid():
        user = serializer.save()
        
        return Response({
            'message': 'User registered successfully',
            'user': {
                'username': user.username,
                'email': user.email
            }
        }, status=status.HTTP_201_CREATED)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    """
    Login user and return JWT tokens
    """
    serializer = LoginSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    username = serializer.validated_data['username']
    password = serializer.validated_data['password']
    
    user = authenticate(username=username, password=password)
    
    if user is None:
        return Response({
            'detail': 'Invalid credentials'
        }, status=status.HTTP_401_UNAUTHORIZED)
    
    # Generate JWT tokens
    refresh = RefreshToken.for_user(user)
    
    return Response({
        'message': 'Login successful',
        'access': str(refresh.access_token),
        'refresh': str(refresh),
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email
        }
    }, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    """
    Logout user by blacklisting refresh token
    """
    try:
        refresh_token = request.data.get('refresh_token')
        token = RefreshToken(refresh_token)
        token.blacklist()
        
        return Response({
            'message': 'Logout successful'
        }, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({
            'detail': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_profile_view(request):
    """
    Get current user profile
    """
    serializer = UserSerializer(request.user)
    return Response(serializer.data)


@api_view(['POST'])
@permission_classes([AllowAny])
def refresh_token_view(request):
    """
    Refresh access token using refresh token
    """
    try:
        refresh_token = request.data.get('refresh')
        token = RefreshToken(refresh_token)
        
        return Response({
            'access': str(token.access_token)
        }, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({
            'detail': 'Invalid refresh token'
        }, status=status.HTTP_401_UNAUTHORIZED)

# ============================================
# Additional helper for your specific pipeline
# ============================================

# def ensure_sql_string(sql_output: Any) -> str:
#     """
#     Final safety wrapper to ensure output is always a clean SQL string.
#     Use this right before executing SQL.
#     """
#     sql = _safe_extract_sql(sql_output)
    
#     # Remove any lingering tuple formatting
#     sql = re.sub(r"^\(\s*['\"]", "", sql)
#     sql = re.sub(r"['\"]\s*,?\s*['\"].*['\"]\s*\)$", "", sql)
    
#     # Remove escaped quotes
#     sql = sql.replace("\\'", "'").replace('\\"', '"')
    
#     return sql.strip()

# --- Your unified repair orchestrator (fixed ordering) ---
import re
from typing import Any, Dict, List, Optional

# def unified_sql_repair(
#     raw_llm_output: Any,
#     schema_catalog: Dict[str, List[str]],
#     question: str = "",
#     fallback_table: Optional[str] = None
# ) -> str:
#     """
#     Single entry point for all SQL repair operations.
#     Repairs EXTRACT usage on text columns for YEAR, MONTH, DAY, QUARTER, WEEK, DOW.
#     Also handles table/column references, NULLIF issues, aliases, and basic syntax cleanup.
#     """

#     original_input = str(raw_llm_output)[:200]

#     try:
#         # Stage 1: Extract SQL
#         sql = _safe_extract_sql(raw_llm_output)
#         if not sql or len(sql.strip()) < 10:
#             return _generate_safe_fallback_sql(schema_catalog, fallback_table)

#         # 🔥 Stage 1a: Fix all EXTRACT syntax for date-like columns
#         sql = re.sub(
#             r'EXTRACT\s*\(\s*(YEAR|MONTH|DAY|QUARTER|WEEK|DOW)\s+FROM\s+([a-zA-Z0-9_." ]+)\)',
#             r'EXTRACT(\1 FROM CAST(\2 AS DATE))',
#             sql,
#             flags=re.IGNORECASE
#         )

#         # Early syntax touch-ups
#         sql = _fix_extract_and_wrapping(sql)
#         sql = _fix_groupby_alias_and_parens(sql)
#         sql = _cleanup_basic_syntax_FINAL(sql)
#         sql = _fix_glued_group_order(sql)

#         # Stage 2: Catalog-based repairs
#         catalog_idx = _build_catalog_indices(schema_catalog)
#         sql = _repair_table_references(sql, catalog_idx)
#         sql = _repair_column_references(sql, catalog_idx)
#         sql = _repair_date_and_interval_literals_FINAL(sql, schema_catalog)
#         sql = _repair_date_columns_FINAL(sql, schema_catalog)
#         sql = _fix_aggregate_text_columns(sql, schema_catalog)
#         sql = _add_safety_guards_FINAL(sql, schema_catalog, catalog_idx)
#         sql = _ensure_from_clause(sql, schema_catalog, fallback_table)
#         sql = _final_validation_FINAL(sql, schema_catalog)

#         # Final cleanups
#         sql = _normalize_order_by(sql)
#         sql = _align_group_order_with_select(sql)  # ✅ NEW
#         sql = _sanitize_group_order_clauses(sql)
#         sql = _fix_missing_paren_before_alias(sql)

#         # 🔥 NUCLEAR FIX (prevents double wrapping or malformed AS clauses)
#         sql = re.sub(
#             r'EXTRACT\s*\(\s*(YEAR|MONTH|DAY|QUARTER|WEEK|DOW)\s*\)\s+FROM\s+([^)]+?)\)\s+AS',
#             lambda m: f'EXTRACT({m.group(1)} FROM {m.group(2)}) AS',
#             sql,
#             flags=re.IGNORECASE
#         )

#         return sql

#     except Exception as e:
#         logger.error(f"SQL repair failed: {e}. Input was: {original_input}")
#         return _generate_safe_fallback_sql(schema_catalog, fallback_table)

# # =========================
# # Stage 2: Basic Cleanup (NO NULLIF HERE)
# # =========================

def _cleanup_basic_syntax_FINAL(sql: str) -> str:
    """
    Fix basic syntax issues WITHOUT injecting NULLIF.
    NULLIF injection happens ONLY in Stage 7.
    """
    
    # Remove comments
    sql = re.sub(r'--[^\n]*', '', sql)

    # De-dupe keywords
    for kw in ['SELECT','FROM','WHERE','GROUP BY','ORDER BY','HAVING','WHEN','END','CASE']:
        sql = re.sub(rf'\b({kw})\s+\1\b', r'\1', sql, flags=re.I)

    # Fix duplicated DATE
    sql = re.sub(r"\bDATE\s+DATE\s+(')", r"DATE \1", sql, flags=re.I)
    
    # Remove silly year self-comparisons
    sql = re.sub(r'\s+(AND|OR)\s+\(?\s*(20\d{2})\s*=\s*\2\s*\)?(?=\s+(AND|OR|$))', ' ', sql, flags=re.I)
    sql = re.sub(r'\bWHERE\s+\(?\s*(20\d{2})\s*=\s*\1\s*\)?\s+(AND|OR)\s+', 'WHERE ', sql, flags=re.I)

    # Ensure ROUND has closing paren
    sql = re.sub(r'\bROUND\s*\(([^)]*)\s+AS\b', r'ROUND(\1) AS', sql, flags=re.I)

    # Drop bare years from GROUP BY / ORDER BY
    sql = re.sub(
        r'\bGROUP\s+BY\s*(.*?)(?=\bORDER\b|\bLIMIT\b|$)',
        lambda m: 'GROUP BY ' + re.sub(r'(?:^|,)\s*20\d{2}\s*(?=,|$)', ',', m.group(1)).strip(' ,'),
        sql,
        flags=re.I
    )
    sql = re.sub(
        r'\bORDER\s+BY\s*(.*?)(?=\bLIMIT\b|$)',
        lambda m: 'ORDER BY ' + re.sub(r'(?:^|,)\s*20\d{2}\s*(?=,|$)', ',', m.group(1)).strip(' ,'),
        sql,
        flags=re.I
    )

    # Misc normalizations
    sql = re.sub(r'\bGROUP\s+BY\s+BY\b','GROUP BY',sql,flags=re.I)
    sql = re.sub(r'\bORDER\s+BY\s+BY\b','ORDER BY',sql,flags=re.I)
    sql = re.sub(r'\bEXTRACTRACT\b','EXTRACT',sql,flags=re.I)
    sql = re.sub(r'\bYYEAR\b','YEAR',sql,flags=re.I)
    sql = re.sub(r'\bMMONTH\b','MONTH',sql,flags=re.I)
    sql = re.sub(r'""+','"',sql)
    sql = re.sub(r'"\s*\.\s*"', '"."', sql)
    sql = re.sub(r'\.\s*\.', '.', sql)
    sql = re.sub(r'"([A-Za-z_]\w*)"\s*"\s*([A-Za-z_]\w*)"', r'"\1"."\2"', sql)
    sql = re.sub(r',\s*(WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|FROM)\b', r' \1', sql, flags=re.I)
    sql = re.sub(r'>\s*>', '>', sql)
    sql = re.sub(r'<\s*<', '<', sql)
    sql = re.sub(r'=\s*=', '=', sql)
    sql = re.sub(r"(INTERVAL\s*'\s*\d+\s*)month\b", r"\1months", sql, flags=re.I)

    return re.sub(r'\s+', ' ', sql).strip()


# =========================
# Stage 7: Safety Guards (IDEMPOTENT - SINGLE PASS)
# =========================


# =========================
# Stage 9: Final Validation
# =========================

def _final_validation_FINAL11(sql: str, schema_catalog: Dict[str, List[str]]) -> str:
    """Final sanity checks and cleanup."""
    
    sql = re.sub(r'\s+', ' ', sql).strip()

    # Remove duplicate DATE tokens
    sql = re.sub(r"\bDATE\s+DATE\s+(')", r"DATE \1", sql, flags=re.I)

    # Normalize INTERVAL
    sql = re.sub(r"(INTERVAL\s*'\s*\d+\s*)month\b", r"\1months", sql, flags=re.I)

    # Ensure basic structure
    if not re.search(r'\bSELECT\b.*\bFROM\b', sql, flags=re.I | re.S):
        logger.warning("[Final] SQL missing SELECT...FROM")
        return _generate_safe_fallback_sql(schema_catalog)

    # Balance parentheses
    opens, closes = sql.count('('), sql.count(')')
    if opens != closes:
        diff = opens - closes
        if 0 < diff <= 3:
            sql += ')' * diff
        elif diff < 0 and -diff <= 3:
            sql = '(' * (-diff) + sql
        else:
            logger.error("[Final] Too many unbalanced parens")
            return _generate_safe_fallback_sql(schema_catalog)

    # Clean up trailing characters
    sql = re.sub(r';\s*\)+\s*$', ';', sql)
    
    # 🔥 FINAL SAFETY: Remove any accidental double ", 0), 0)" patterns
    sql = re.sub(r',\s*0\s*\)\s*,\s*0\s*\)', ', 0)', sql)
    
    return re.sub(r'\s+', ' ', sql).strip()


# =========================
# Stage 7: Safety Guards (FIXED)
# =========================

def _add_safety_guards_FINAL(sql: str, schema_catalog: dict, catalog_idx: dict) -> str:
    """
    Add NULLIF guards for division by zero.
    **IDEMPOTENT**: Won't double-wrap existing NULLIF.
    """
    
    # 🔥 CRITICAL: Skip if already has NULLIF (prevent double-wrapping)
    if 'NULLIF' in sql.upper():
        logger.debug("[Safety Guards] NULLIF already present, skipping injection")
        # Just apply numeric casting
        sql = _cast_text_columns_to_numeric(sql, schema_catalog)
        return sql
    
    # Pattern 1: Inject NULLIF only for division by aggregates
    # / COUNT( → / NULLIF(COUNT(
    sql = re.sub(
        r'/\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(',
        r'/ NULLIF(\1(',
        sql,
        flags=re.I
    )
    
    # Pattern 2: Close ALL opened NULLIF with ", 0)"
    # NULLIF(COUNT(col)) → NULLIF(COUNT(col), 0)
    # NULLIF(COUNT(col)::numeric) → NULLIF(COUNT(col)::numeric, 0)
    sql = re.sub(
        r'NULLIF\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s*\)(?!\s*,\s*0)',
        r'NULLIF(\1(\2)\3, 0)',
        sql,
        flags=re.I
    )
    
    # Pattern 3: Handle expressions that end at AS or operators
    # NULLIF(COUNT(...) AS → NULLIF(COUNT(...), 0) AS
    sql = re.sub(
        r'NULLIF\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s*(?=\s*(AS|,|/|\+|\-|\*|FROM|WHERE|GROUP|ORDER)\b)',
        r'NULLIF(\1(\2)\3, 0) ',
        sql,
        flags=re.I
    )
    
    # Apply numeric casting
    sql = _cast_text_columns_to_numeric(sql, schema_catalog)
    
    return sql





def _extract_sql_from_llm_output(raw: Any) -> str:
    """Extract SQL from various wrapper formats."""
    if raw is None:
        return ""
    
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, (tuple, list)):
        text = ""
        for item in raw:
            if isinstance(item, str) and item.strip():
                text = item
                break
        if not text:
            text = " ".join(str(x) for x in raw if x is not None)
    elif isinstance(raw, dict):
        for key in ("sql", "text", "content", "message", "output"):
            if key in raw and isinstance(raw[key], str) and raw[key].strip():
                text = raw[key]
                break
        else:
            text = str(raw)
    else:
        text = str(raw)
    
    if not text or not text.strip():
        return ""
    
    # Strip model tokens
    text = re.sub(r'<\|header_start\|>\s*sql', '', text, flags=re.I)
    text = re.sub(r'<\|header_end\|>', '', text, flags=re.I)
    
    # Prefer fenced block
    m = re.search(r'```(?:sql)?\s*(.*?)\s*```', text, flags=re.I | re.DOTALL)
    if m:
        text = m.group(1)
    
    # Drop "sql\n" prefix
    text = re.sub(r'^\s*sql\s+', '', text, flags=re.I)
    
    # Keep from first WITH/SELECT
    m = re.search(r'\b(WITH|SELECT)\b.*', text, flags=re.I | re.DOTALL)
    if m:
        text = m.group(0)
    
    return text.strip()
# =========================
# Helper: Numeric Casting
# =========================

def _cast_text_columns_to_numeric(sql: str, schema_catalog: Dict[str, List[str]]) -> str:
    """Dynamically cast text columns to numeric when used in numeric context."""
    
    # Skip if already processed
    if 'REGEXP_REPLACE' in sql:
        return sql

    # Heuristic patterns for numeric columns
    patterns = [
        'amount','price','cost','fee','rate','discount','charge','payment','balance','total','value',
        'income','salary','revenue','profit','loss','score','count','quantity','volume','sum',
        'offs','principal','apr','interest','finance','premium','deductible','commission'
    ]

    candidates: List[Tuple[str, str]] = []
    for fq_table, columns in (schema_catalog or {}).items():
        for col in columns:
            if any(p in col.lower() for p in patterns):
                candidates.append((fq_table, col))

    if not candidates:
        return sql

    for fq_table, col_name in candidates:
        if not _is_column_in_numeric_context(sql, col_name):
            continue
        
        safe_cast_template = "NULLIF(REGEXP_REPLACE({col}, '[^0-9\\.-]', '', 'g'), '')::NUMERIC"
        sql = _replace_column_in_aggregates(sql, fq_table, col_name, safe_cast_template)

    return sql


def _is_column_in_numeric_context(sql: str, col_name: str) -> bool:
    """Check if column is used in numeric operations."""
    col_escaped = re.escape(col_name)
    
    # Check for aggregates
    if re.search(rf'\b(SUM|AVG|COUNT|MIN|MAX)\s*\([^)]*\b{col_escaped}\b', sql, flags=re.I):
        return True
    
    # Check for arithmetic
    if re.search(rf'\b{col_escaped}\b\s*[+\-*/]', sql, flags=re.I):
        return True
    
    # Check for numeric comparison
    if re.search(rf'\b{col_escaped}\b\s*[<>=].*?\d', sql, flags=re.I):
        return True
    
    return False


def _replace_column_in_aggregates(sql: str, fq_table: str, col_name: str, cast_template: str) -> str:
    """Replace column references inside aggregate functions with safe cast."""
    
    table_parts = fq_table.replace('"', '').split('.')
    schema_name = table_parts[0] if len(table_parts) > 1 else ''
    table_name = table_parts[-1] if table_parts else ''
    col_escaped = re.escape(col_name)

    patterns = [
        (rf'({re.escape(fq_table)}\."{col_escaped}")\b', 'fq_quoted_col'),
        (rf'({re.escape(fq_table)}\.{col_escaped})\b', 'fq_unquoted_col'),
        (rf'({re.escape(schema_name)}\.{re.escape(table_name)}\.{col_escaped})\b', 'schema_table_col'),
        (rf'({re.escape(table_name)}\.{col_escaped})\b', 'table_col'),
        (rf'\b(SUM|AVG|COUNT|MIN|MAX)\s*\(\s*("?{col_escaped}"?)\s*\)', 'bare_in_agg'),
    ]

    for pattern, kind in patterns:
        def repl(m):
            if kind == 'bare_in_agg':
                func = m.group(1)
                col_ref = f"{fq_table}.{col_name}"
                casted = cast_template.replace('{col}', col_ref)
                return f"{func}({casted})"
            col_ref = m.group(1)
            return cast_template.replace('{col}', col_ref)

        sql = re.sub(pattern, repl, sql, flags=re.I)

    return sql


# =========================
# Stage 9: Final Validation (FIXED)
# =========================

def _final_validation_FINAL11(sql: str, schema_catalog: Dict[str, List[str]]) -> str:
    """
    Final sanity checks with additional NULLIF closure safety.
    """
    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()

    # Late sweep: collapse accidental double DATE tokens
    sql = re.sub(r"\bDATE\s+DATE\s+(')", r"DATE \1", sql, flags=re.I)

    # Normalize INTERVAL singular → plural
    sql = re.sub(r"(INTERVAL\s*'\s*\d+\s*)month\b", r"\1months", sql, flags=re.I)

    # Ensure basic SELECT...FROM structure
    if not re.search(r'\bSELECT\b.*\bFROM\b', sql, flags=re.I | re.S):
        logger.warning("[Final] SQL missing SELECT...FROM structure")
        return _generate_safe_fallback_sql(schema_catalog)

    # Balance parentheses
    opens, closes = sql.count('('), sql.count(')')
    if opens != closes:
        diff = opens - closes
        if 0 < diff <= 3:
            sql += ')' * diff
        elif diff < 0 and -diff <= 3:
            sql = '(' * (-diff) + sql
        else:
            logger.error("[Final] Too many unbalanced parens, generating fallback")
            return _generate_safe_fallback_sql(schema_catalog)

    # Trim funky trailing ');)' patterns
    sql = re.sub(r';\s*\)+\s*$', ';', sql)

    # 🔥 FINAL SAFETY: One last check for any lingering unclosed NULLIF
    if re.search(r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\([^)]*\)\s*(::\w+)?\s+AS\b', sql, flags=re.I):
        sql = re.sub(
            r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s+AS\b',
            r'NULLIF(\1(\2)\3, 0) AS',
            sql,
            flags=re.I
        )

    # Final whitespace normalization
    return re.sub(r'\s+', ' ', sql).strip()

def _generate_safe_fallback_sql(schema_catalog: Dict[str, List[str]], fallback_table: str | None = None) -> str:
    """Generate safe fallback query."""
    fq = fallback_table
    if not fq and schema_catalog:
        fq = next(iter(schema_catalog))
    if not fq:
        return "SELECT 1 AS status, 'No tables available' AS message;"
    return f"SELECT COUNT(*) AS row_count FROM {fq} LIMIT 1;"



def _final_validation_FINAL(sql: str, schema_catalog: dict) -> str:
    sql = re.sub(r'\s+', ' ', sql).strip()

    # Late sweep: collapse accidental double DATE tokens and normalize intervals
    sql = re.sub(r"\bDATE\s+DATE\s+(')", r"DATE \1", sql, flags=re.I)
    sql = re.sub(r"(INTERVAL\s*'\s*\d+\s*)month\b", r"\1months", sql, flags=re.I)

    if not re.search(r'\bSELECT\b.*\bFROM\b', sql, flags=re.I | re.S):
        logger.warning("[Final] SQL missing SELECT...FROM structure")
        return _generate_safe_fallback_sql(schema_catalog)

    opens, closes = sql.count('('), sql.count(')')
    if opens != closes:
        diff = opens - closes
        if 0 < diff <= 3:
            sql += ')' * diff
        elif diff < 0 and -diff <= 3:
            sql = '(' * (-diff) + sql
        else:
            logger.error("[Final] Too many unbalanced parens, generating fallback")
            return _generate_safe_fallback_sql(schema_catalog)

    sql = re.sub(r';\s*\)+\s*$', ';', sql)

    # 🔧 NEW: sanitize GROUP BY / ORDER BY (removes ", 2024" and "ORDER BY 2024")
    sql = _sanitize_group_order_clauses(sql)

    # Guard: any lingering NULLIF(... AS) – close it.
    sql = re.sub(
        r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s+AS\b',
        r'NULLIF(\1(\2)\3, 0) AS',
        sql,
        flags=re.I
    )
      # normalize easy token errors before sanitizing
    sql = re.sub(r"\bDATE\s+DATE\s+(')", r"DATE \1", sql, flags=re.I)
    sql = re.sub(r"(INTERVAL\s*'\s*\d+\s*)month\b", r"\1months", sql, flags=re.I)
    sql = re.sub(r';\s*\)+\s*$', ';', sql)

    # 🚀 fix GROUP BY / ORDER BY safely (prevents ')EXTRACT(' glue)

    # last guard for NULLIF-before-AS shapes
    sql = re.sub(
        r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*(::\w+)?\s+AS\b',
        r'NULLIF(\1(\2)\3, 0) AS',
        sql, flags=re.I
    )

    return re.sub(r'\s+', ' ', sql).strip()


def _derive_join_hints(schema_catalog: dict, tables: list[str]) -> dict:
    """
    Return:
      {
        "aliases": {'"schema"."app_main_2024"': 'a', '"schema"."loan_main_2024"': 'l'},
        "candidate_keys": ['application_id', 'customer_id', 'loan_id', 'policy_no', ...]
      }
    """
    if len(tables) != 2:
        return {"aliases": {}, "candidate_keys": []}

    t1, t2 = tables
    cols1 = {c.lower() for c in schema_catalog.get(t1, [])}
    cols2 = {c.lower() for c in schema_catalog.get(t2, [])}

    # 1) shared columns
    shared = sorted(cols1 & cols2)

    # 2) prioritize likely keys
    preferred = [
        "application_id", "loan_application_id", "loan_id", "customer_id",
        "policy_no", "policy_id", "app_id", "account_id"
    ]
    ordered = [k for k in preferred if k in shared] + [c for c in shared if c not in preferred]

    # stable short aliases
    aliases = {t1: "a", t2: "l"}
    return {"aliases": aliases, "candidate_keys": ordered}
import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)
_SQL_COL_REF_RE = re.compile(r'\b([a-zA-Z_][\w]*)\s*\.\s*("?[A-Za-z_]\w*"?)\b')

def _strip_quotes(name: str) -> str:
    return name[1:-1] if name.startswith('"') and name.endswith('"') else name

def _normalize_col_name(s: str) -> str:
    """Normalization for fuzzy matching: lower, strip underscores/hyphens, remove 'score'/'bucket' suffix noise."""
    s = s.lower()
    s = re.sub(r'[_\-\s]+', '', s)
    s = re.sub(r'(score|bucket|num|id|code)$', '', s)
    return s

def _repair_column_aliases(sql: str, schema_catalog: Dict[str, list], alias_to_table: Dict[str, str]) -> str:
    """
    Ensure alias.column references point to a table that actually contains that column.
    - schema_catalog: { '"schema"."table"': ['col1','col2', ...], ... }
    - alias_to_table: { 'a': '"stage"."app_main_2024"', 'l': '"stage"."loan_main_2024"' }
    """
    if not sql or not schema_catalog or not alias_to_table:
        return sql

    # Lowercase sets for quick lookup & normalized col name mapping
    table_cols_lc = { fq: {c.lower(): c for c in cols} for fq, cols in schema_catalog.items() }

    # alias -> set of lowercased column names
    alias_cols_map = { alias: set(table_cols_lc.get(fq, {}).keys()) for alias, fq in alias_to_table.items() }

    # col -> locations mapping
    col_to_locations = {}
    for fq, cols_map in table_cols_lc.items():
        for c_lc, original_col in cols_map.items():
            col_to_locations.setdefault(c_lc, []).append((fq, original_col))

    # Build normalized index to support fuzzy match
    norm_index = {}  # normalized -> list of (fq, original_col)
    for fq, cols_map in table_cols_lc.items():
        for c_lc, original_col in cols_map.items():
            norm = _normalize_col_name(c_lc)
            norm_index.setdefault(norm, []).append((fq, original_col))

    modified = False
    def _replacement(match):
        nonlocal modified
        alias = match.group(1)
        col_token = match.group(2)
        col_unq = _strip_quotes(col_token).lower()

        # if alias unknown, leave as-is
        if alias not in alias_cols_map:
            return match.group(0)

        # if exists in alias table already - OK
        if col_unq in alias_cols_map.get(alias, set()):
            return match.group(0)

        # direct exact match elsewhere? try that first
        locations = col_to_locations.get(col_unq, [])

        # if no exact match, try normalized fuzzy match
        if not locations:
            candidates = norm_index.get(_normalize_col_name(col_unq), [])
            locations = candidates

        # If still none, try substring heuristics
        if not locations:
            for cand_norm, entries in norm_index.items():
                if cand_norm and (col_unq.replace('_','') in cand_norm or cand_norm in col_unq.replace('_','')):
                    locations = entries
                    break

        # choose a location that matches one of the known alias tables, prefer same alias -> no change
        for loc, orig_col in locations:
            for cand_alias, cand_fq in alias_to_table.items():
                if cand_fq == loc:
                    # if the found location is the same alias, rewrite to that alias
                    logger.info("Repairing column alias: rewriting %s.%s -> %s.%s", alias, col_token, cand_alias, orig_col)
                    modified = True
                    # preserve quoting if original col_token had quotes
                    out_col = f'"{orig_col}"' if col_token.startswith('"') else orig_col
                    return f"{cand_alias}.{out_col}"

        # if we found some location but not matching an alias, pick first and map to whichever alias points to that fq
        if locations:
            loc_fq, orig_col = locations[0]
            # find alias that owns that fq
            target_alias = None
            for cand_alias, cand_fq in alias_to_table.items():
                if cand_fq == loc_fq:
                    target_alias = cand_alias
                    break
            if target_alias:
                logger.info("Repairing column alias (fallback): rewriting %s.%s -> %s.%s", alias, col_token, target_alias, orig_col)
                modified = True
                out_col = f'"{orig_col}"' if col_token.startswith('"') else orig_col
                return f"{target_alias}.{out_col}"

        # nothing to do — leave original
        return match.group(0)

    repaired = _SQL_COL_REF_RE.sub(_replacement, sql)
    if modified:
        logger.debug("SQL after alias repair:\n%s", repaired)
    return repaired


from typing import Any
from genai_app.utils.db_utils import generate_schema_agnostic_sql, execute_dynamic_sql

"""
COMPLETE WORKING SQL GENERATION SYSTEM
Drop-in replacement for your ask_question_stream view
"""

import re
import json
import traceback
from typing import Dict, List, Any, Tuple
from datetime import datetime
from django.http import StreamingHttpResponse, JsonResponse
from sqlalchemy import text
import pandas as pd


# ============================================================================
# UTILITY: JSONL Formatter
# ============================================================================

def jsonl_line(obj: dict) -> str:
    """Format dict as JSONL (one JSON object per line)."""
    return json.dumps(obj) + '\n'


# ============================================================================
# STEP 1: Dynamic Table Selection (Session-Aware)
# ============================================================================

def select_tables_for_question(
    question: str,
    schema_catalog: Dict[str, List[str]],
    session_id: str
) -> List[str]:
    """
    Select relevant tables from current session's schema_catalog only.
    Uses keyword heuristics to avoid wrong table selection.
    """
    
    available_tables = list(schema_catalog.keys())
    
    if not available_tables:
        return []
    
    if len(available_tables) == 1:
        return available_tables
    
    # Score tables based on question keywords
    q_lower = question.lower()
    table_scores = {}
    
    for table in available_tables:
        score = 0
        table_lower = table.lower()
        
        # Check if question needs specific table
        if any(kw in q_lower for kw in ['loan', 'funded', 'vantage', 'fico', 'dti']):
            if 'loan' in table_lower:
                score += 15
        
        if any(kw in q_lower for kw in ['application', 'approval', 'approved', 'status', 'channel']):
            if 'app' in table_lower:
                score += 15
        
        # Income can be in both tables
        if 'income' in q_lower:
            if 'app' in table_lower:
                score += 8  # Slight preference for app table
            elif 'loan' in table_lower:
                score += 5
        
        # Default small score so all tables are considered
        if score == 0:
            score = 1
        
        table_scores[table] = score
    
    # Return tables sorted by score (highest first)
    sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
    
    # For most questions, single table is enough
    # Only use 2 tables if question clearly needs join
    if any(kw in q_lower for kw in ['both', 'combine', 'join', 'merge', 'together']):
        return [t[0] for t in sorted_tables[:2]]
    
    # Otherwise return top table
    return [sorted_tables[0][0]]


# ============================================================================
# STEP 2: Find Join Keys Between Tables
# ============================================================================

def find_join_keys(tables: List[str], schema_catalog: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Find common columns between tables that can be used for joins.
    """
    
    if len(tables) < 2:
        # Single table - no join needed
        return {
            'aliases': {tables[0]: 'a'} if tables else {},
            'join_keys': [],
            'needs_join': False
        }
    
    # Create aliases
    aliases = {table: chr(ord('a') + i) for i, table in enumerate(tables)}
    
    # Find common columns
    cols1 = set(c.lower() for c in schema_catalog.get(tables[0], []))
    cols2 = set(c.lower() for c in schema_catalog.get(tables[1], []))
    
    common = cols1 & cols2
    
    # Prioritize ID columns
    join_keys = []
    for col in common:
        if any(kw in col for kw in ['id', 'number', 'key']):
            # Get original case-sensitive column name from first table
            original = next((c for c in schema_catalog[tables[0]] if c.lower() == col), col)
            join_keys.append(original)
    
    # Add other common columns if no IDs found
    if not join_keys:
        join_keys = [next((c for c in schema_catalog[tables[0]] if c.lower() == col), col) 
                     for col in list(common)[:3]]
    
    return {
        'aliases': aliases,
        'join_keys': join_keys or ['customerid'],  # fallback
        'needs_join': True
    }


# ============================================================================
# STEP 3: Map Question Terms to Actual Column Names
# ============================================================================

def find_relevant_columns(question: str, tables: List[str], schema_catalog: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Map common question terms to actual column names in selected tables.
    """
    
    q_lower = question.lower()
    mappings = {}
    
    for table in tables:
        cols = schema_catalog.get(table, [])
        cols_lower = [c.lower() for c in cols]
        
        # Status mapping
        if 'status' in q_lower or 'approved' in q_lower:
            for i, col in enumerate(cols_lower):
                if 'status' in col and 'application' in col:
                    mappings['status'] = cols[i]
                    break
        
        # Income mapping
        if 'income' in q_lower:
            for i, col in enumerate(cols_lower):
                if 'income' in col:
                    mappings['income'] = cols[i]
                    break
        
        # Date mapping
        if any(kw in q_lower for kw in ['date', 'month', 'year', 'when']):
            for i, col in enumerate(cols_lower):
                if 'date' in col and 'application' in col:
                    mappings['application_date'] = cols[i]
                    break
        
        # Score mappings
        if 'vantage' in q_lower:
            for i, col in enumerate(cols_lower):
                if 'vantage' in col:
                    mappings['vantage'] = cols[i]
                    break
        
        if 'fico' in q_lower:
            for i, col in enumerate(cols_lower):
                if 'fico' in col:
                    mappings['fico'] = cols[i]
                    break
    
    return mappings


# ============================================================================
# STEP 4: Build SQL Generation Prompt
# ============================================================================

def build_sql_prompt(
    question: str,
    tables: List[str],
    schema_catalog: Dict[str, List[str]],
    join_info: Dict[str, Any],
    column_hints: Dict[str, str]
) -> str:
    """
    Create a focused prompt for SQL generation.
    """
    
    aliases = join_info['aliases']
    
    # Build table info
    table_info = []
    for table in tables:
        alias = aliases[table]
        cols = schema_catalog.get(table, [])
        # Show first 40 cols to stay within token limits
        cols_str = ', '.join(cols[:40])
        if len(cols) > 40:
            cols_str += f'... ({len(cols)} total)'
        table_info.append(f"{table} AS {alias}\nColumns: {cols_str}")
    
    tables_doc = "\n\n".join(table_info)
    
    # Join info
    if join_info['needs_join'] and len(tables) > 1:
        a1, a2 = aliases[tables[0]], aliases[tables[1]]
        join_key = join_info['join_keys'][0]
        join_doc = f"To join tables: {a1}.{join_key} = {a2}.{join_key}"
    else:
        join_doc = "Single table query - no join needed"
    
    # Column hints
    hints_doc = ""
    if column_hints:
        hints_doc = "\nRelevant columns:\n" + "\n".join([f"- {k}: {v}" for k, v in column_hints.items()])
    
    prompt = f"""Generate a PostgreSQL query for this question.

AVAILABLE TABLES:
{tables_doc}

{join_doc}
{hints_doc}

RULES:
1. Use ONLY the tables and columns shown above
2. Prefix all columns with table alias (e.g., a.columnname)
3. Use ILIKE for text matching (case-insensitive)
4. Handle NULL values in WHERE clauses
5. Return ONLY the SQL query - no explanations

USER QUESTION: {question}

SQL:"""
    
    return prompt


# ============================================================================
# STEP 5: Extract and Clean SQL from LLM Response
# ============================================================================

def extract_sql(llm_response: Any) -> str:
    """
    Extract SQL from LLM response, handling various formats.
    """
    
    # Get text from response
    if hasattr(llm_response, 'content'):
        text = llm_response.content
    elif isinstance(llm_response, tuple):
        text = llm_response[0] if llm_response else ""
    else:
        text = str(llm_response)
    
    # Remove markdown code fences
    text = re.sub(r'```(?:sql)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text)
    
    # Remove explanation lines
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Skip explanatory text
        if any(line.lower().startswith(p) for p in ['note:', 'explanation:', 'this query', 'here']):
            continue
        lines.append(line)
    
    sql = ' '.join(lines)
    
    # Clean up
    sql = sql.replace('EXTEXTRACT', 'EXTRACT')
    sql = sql.replace('FROM FROM', 'FROM')
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.strip()
    
    # Remove rogue parentheses
    if sql.startswith('(SELECT'):
        sql = sql[1:]
    
    # Ensure semicolon
    if not sql.endswith(';'):
        sql += ';'
    
    return sql


# ============================================================================
# STEP 6: Main SQL Generation Function
# ============================================================================

def generate_sql_for_question(
    question: str,
    session_id: str,
    schema_catalog: Dict[str, List[str]],
    llm
) -> str:
    """
    Complete SQL generation pipeline.
    Returns executable SQL string.
    """
    
    # Step 1: Select tables
    tables = select_tables_for_question(question, schema_catalog, session_id)
    
    if not tables:
        raise ValueError("No suitable tables found in schema")
    
    print(f"📊 Selected tables: {tables}")
    
    # Step 2: Find join keys
    join_info = find_join_keys(tables, schema_catalog)
    print(f"🔗 Join info: {join_info}")
    
    # Step 3: Find relevant columns
    column_hints = find_relevant_columns(question, tables, schema_catalog)
    print(f"🗺️ Column hints: {column_hints}")
    
    # Step 4: Build prompt
    prompt = build_sql_prompt(question, tables, schema_catalog, join_info, column_hints)
    
    # Step 5: Call LLM
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        print(f"⚠️ LLM call failed: {e}")
        # Fallback to simple query
        table = tables[0]
        alias = join_info['aliases'][table]
        return f"SELECT COUNT(*) AS total FROM {table} {alias};"
    
    # Step 6: Extract SQL
    sql = extract_sql(response)
    
    print(f"✅ Generated SQL:\n{sql}\n")
    
    return sql


"""
COMPLETE SQL GENERATION SYSTEM - FULLY INTEGRATED
Combines dynamic table selection with existing infrastructure
"""

import re
import json
import traceback
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from sqlalchemy import text
import pandas as pd


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def jsonl_line(obj: dict) -> str:
    """Format dict as JSONL (one JSON object per line)."""
    return json.dumps(obj) + '\n'


# ============================================================================
# TABLE SELECTION - Session-Aware
# ============================================================================

def select_tables_for_question(
    question: str,
    schema_catalog: Dict[str, List[str]],
    session_id: str
) -> List[str]:
    """
    Select relevant tables from current session's schema_catalog only.
    Uses keyword heuristics to avoid wrong table selection.
    """
    
    print(f"🔍 select_tables_for_question called")
    print(f"   - question: {question}")
    print(f"   - session_id: {session_id}")
    print(f"   - available tables: {list(schema_catalog.keys())}")
    
    available_tables = list(schema_catalog.keys())
    
    if not available_tables:
        print("⚠️ No tables available!")
        return []
    
    if len(available_tables) == 1:
        print(f"✅ Only one table, returning: {available_tables}")
        return available_tables
    
    # Score tables based on question keywords
    q_lower = question.lower()
    table_scores = {}
    
    for table in available_tables:
        score = 0
        table_lower = table.lower()
        
        # Check if question needs specific table
        if any(kw in q_lower for kw in ['loan', 'funded', 'vantage', 'fico', 'dti']):
            if 'loan' in table_lower:
                score += 15
        
        if any(kw in q_lower for kw in ['application', 'approval', 'approved', 'status', 'channel']):
            if 'app' in table_lower:
                score += 15
        
        # Income can be in both tables
        if 'income' in q_lower:
            if 'app' in table_lower:
                score += 8
            elif 'loan' in table_lower:
                score += 5
        
        # Default small score so all tables are considered
        if score == 0:
            score = 1
        
        table_scores[table] = score
        print(f"   - {table}: score={score}")
    
    # Return tables sorted by score (highest first)
    sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
    
    # For most questions, single table is enough
    # Only use 2 tables if question clearly needs join
    if any(kw in q_lower for kw in ['both', 'combine', 'join', 'merge', 'together']):
        result = [t[0] for t in sorted_tables[:2]]
        print(f"✅ Multi-table query, returning: {result}")
        return result
    
    # Otherwise return top table
    result = [sorted_tables[0][0]]
    print(f"✅ Single-table query, returning: {result}")
    return result


# ============================================================================
# JOIN DISCOVERY
# ============================================================================

def discover_join_keys(tables: List[str], schema_catalog: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Find common columns between tables that can be used for joins.
    """
    
    if len(tables) < 2:
        return {
            'aliases': {tables[0]: 'a'} if tables else {},
            'join_keys': [],
            'candidate_keys': [],
            'needs_join': False
        }
    
    # Create aliases
    aliases = {table: chr(ord('a') + i) for i, table in enumerate(tables)}
    
    # Find common columns
    cols1 = set(c.lower() for c in schema_catalog.get(tables[0], []))
    cols2 = set(c.lower() for c in schema_catalog.get(tables[1], []))
    
    common = cols1 & cols2
    
    # Prioritize ID columns
    join_keys = []
    for col in common:
        if any(kw in col for kw in ['id', 'number', 'key']):
            # Get original case-sensitive column name from first table
            original = next((c for c in schema_catalog[tables[0]] if c.lower() == col), col)
            join_keys.append(original)
    
    # Add other common columns if no IDs found
    if not join_keys:
        join_keys = [next((c for c in schema_catalog[tables[0]] if c.lower() == col), col) 
                     for col in list(common)[:3]]
    
    return {
        'aliases': aliases,
        'join_keys': join_keys or ['customerid'],
        'candidate_keys': join_keys or ['customerid'],
        'needs_join': True
    }


# ============================================================================
# COLUMN MAPPING
# ============================================================================

def normalize_column_names(question: str, schema_catalog: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Map common question terms to actual column names in the schema.
    """
    
    q_lower = question.lower()
    mappings = {}
    
    for table, columns in schema_catalog.items():
        for col in columns:
            col_lower = col.lower()
            
            # Status mappings
            if 'status' in q_lower or 'approved' in q_lower:
                if 'status' in col_lower and 'application' in col_lower:
                    mappings['status'] = col
            
            # Income mappings
            if 'income' in q_lower:
                if 'income' in col_lower:
                    mappings['income'] = col
            
            # Date mappings
            if any(word in q_lower for word in ['date', 'month', 'year', 'when']):
                if 'date' in col_lower and 'application' in col_lower:
                    mappings['application_date'] = col
            
            # Score mappings
            if 'vantage' in q_lower and 'vantage' in col_lower:
                mappings['vantage'] = col
            
            if 'fico' in q_lower and 'fico' in col_lower:
                mappings['fico'] = col
    
    return mappings


# ============================================================================
# PROMPT BUILDING
# ============================================================================

def build_dynamic_sql_prompt(
    question: str,
    tables: List[str],
    schema_catalog: Dict[str, List[str]],
    join_info: Dict[str, Any],
    history: List[Dict] = None
) -> str:
    """
    Build a focused prompt for SQL generation.
    """
    
    aliases = join_info['aliases']
    
    # Build table info with actual columns
    table_docs = []
    for table in tables:
        alias = aliases[table]
        cols = schema_catalog.get(table, [])
        
        # Show first 50 cols to avoid token overflow
        cols_display = ', '.join(cols[:50])
        if len(cols) > 50:
            cols_display += f'... ({len(cols)} total)'
        
        table_docs.append(f"Table: {table} AS {alias}")
        table_docs.append(f"Columns: {cols_display}")
    
    tables_section = "\n\n".join(table_docs)
    
    # Join instructions
    if join_info['needs_join'] and len(tables) > 1:
        a1, a2 = aliases[tables[0]], aliases[tables[1]]
        join_keys = join_info.get('candidate_keys', [])
        if join_keys:
            join_key = join_keys[0]
            keys_str = ', '.join(join_keys[:3])
            joins_section = f"To join: {a1}.{join_key} = {a2}.{join_key}\nAlternative keys: {keys_str}"
        else:
            joins_section = "Use customerid or decisionid for joins"
    else:
        joins_section = "Single table query - no join needed"
    
    # Column hints
    col_hints = normalize_column_names(question, schema_catalog)
    hints_section = ""
    if col_hints:
        hints_section = "\nKey columns for this query:\n"
        for term, col in col_hints.items():
            hints_section += f"- {term}: {col}\n"
    
    # History context (optional)
    history_section = ""
    if history:
        recent = history[-2:]
        if recent:
            history_section = "\nRecent queries:\n"
            for h in recent:
                q = h.get('question', '')[:60]
                history_section += f"- {q}\n"
    
    prompt = f"""Generate a PostgreSQL query.

AVAILABLE TABLES:
{tables_section}

JOIN INSTRUCTIONS:
{joins_section}
{hints_section}

RULES:
1. Use ONLY tables and columns listed above
2. Always prefix columns with alias (e.g., a.columnname)
3. Use ILIKE for case-insensitive text matching
4. Handle NULL values properly
5. Return ONLY the SQL - no explanations
{history_section}

USER QUESTION: {question}

SQL:"""
    
    return prompt


# ============================================================================
# SQL EXTRACTION & REPAIR
# ============================================================================

def repair_generated_sql(
    raw_llm_output: Any,
    tables: List[str],
    schema_catalog: Dict[str, List[str]],
    join_info: Dict[str, Any]
) -> str:
    """
    Extract and repair SQL from LLM response.
    """
    
    # Extract text from various formats
    if hasattr(raw_llm_output, 'content'):
        text = raw_llm_output.content
    elif isinstance(raw_llm_output, tuple):
        text = raw_llm_output[0] if raw_llm_output else ""
    else:
        text = str(raw_llm_output)
    
    # Remove markdown fences
    text = re.sub(r'```(?:sql)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text)
    
    # Remove explanation lines
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if any(line.lower().startswith(p) for p in ['note:', 'explanation:', 'this query']):
            continue
        lines.append(line)
    
    sql = ' '.join(lines)
    
    # Fix common issues
    sql = sql.replace('EXTEXTRACT', 'EXTRACT')
    sql = sql.replace('FROM FROM', 'FROM')
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.strip()
    
    # Remove rogue parentheses
    if sql.startswith('(SELECT') and not sql.endswith(')'):
        sql = sql[1:]
    
    # Ensure semicolon
    if not sql.endswith(';'):
        sql += ';'
    
    return sql


# ============================================================================
# MAIN SQL GENERATION FUNCTION (Your existing function signature)
# ============================================================================


# ============================================================================
# DJANGO VIEW - Drop-in Replacement
# ============================================================================

@csrf_exempt
def ask_question_stream1111(request):
    """
    Complete working streaming view.
    Replace your existing ask_question_stream with this.
    """
    
    if request.method == "OPTIONS":
        resp = HttpResponse()
        resp["Access-Control-Allow-Origin"] = "*"
        resp["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp["Access-Control-Allow-Headers"] = "Content-Type"
        return resp
    
    if request.method != "POST":
        return JsonResponse({"error": "Use POST"}, status=405)
    
    try:
        payload = json.loads(request.body or "{}")
        session_id = payload.get("session_id")
        question = payload.get("question")
        user_id = payload.get("user_id", "default_user")
        
        if not session_id or not question:
            return JsonResponse({"error": "Missing session_id or question"}, status=400)
        
        # Get session
        session_data = session_store.get(session_id)
        if not session_data:
            return JsonResponse({"error": "Session not found. Please connect first."}, status=404)
        
        engine = session_data["engine"]
        schema_catalog = session_data.get("schema_catalog", {})
        
        if not schema_catalog:
            return JsonResponse({"error": "Schema catalog missing"}, status=400)
        
        print(f"📊 Session {session_id} has tables: {list(schema_catalog.keys())}")
        
        def gen():
            """Stream generator for JSONL responses."""
            
            # Opening message
            yield jsonl_line({
                "event": "narrative_opener",
                "text": "Analyzing your question..."
            })
            
            yield jsonl_line({
                "event": "phase",
                "message": "Understanding your question…"
            })
            
            # Get conversation history
            history = conversation_memory_store.get(session_id, [])
            
            # Generate SQL using your existing function
            try:
                print(f"🎯 About to call run_sql_generation_graph...")
                print(f"   - question: {question}")
                print(f"   - session_id: {session_id}")
                print(f"   - schema_catalog keys: {list(schema_catalog.keys())}")
                
                sql, explanation, _ = run_sql_generation_graph(
                    question=question,
                    user_id=user_id,
                    session_id=session_id,
                    history=history
                )
                
                print(f"✅ run_sql_generation_graph returned!")
                print(f"   - sql: {sql[:100]}...")
                print(f"   - explanation: {explanation[:100] if explanation else 'None'}...")
                
                if sql.startswith("-- ERROR"):
                    yield jsonl_line({
                        "event": "error",
                        "message": sql
                    })
                    return
                
                if not sql or sql.strip() == "":
                    yield jsonl_line({
                        "event": "error",
                        "message": "SQL generation returned empty result"
                    })
                    return
                
                yield jsonl_line({"event": "sql", "sql": sql})
                
            except Exception as e:
                traceback.print_exc()
                print(f"❌ Exception in SQL generation: {e}")
                yield jsonl_line({
                    "event": "error",
                    "message": f"SQL generation failed: {str(e)}"
                })
                return
            
            # Execute SQL
            yield jsonl_line({
                "event": "phase",
                "message": "Querying the database…"
            })
            
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(sql))
                    rows = [dict(row._mapping) for row in result]
                
                # Clean data
                df = pd.DataFrame(rows)
                df = df.fillna('')
                rows = df.to_dict(orient="records")
                
                print(f"✅ Query executed. Rows: {len(rows)}")
                
            except Exception as e:
                traceback.print_exc()
                yield jsonl_line({
                    "event": "error",
                    "message": f"Query execution failed: {str(e)}"
                })
                return
            
            row_count = len(rows)
            
            # Send preview
            yield jsonl_line({
                "event": "rows_preview",
                "rows": rows[:10],
                "row_count": row_count
            })
            
            yield jsonl_line({
                "event": "phase",
                "message": "Analyzing results…"
            })
            
            # Build answer
            if row_count == 0:
                answer = "No matching records found."
            elif row_count == 1:
                answer = ", ".join([f"{k}: {v}" for k, v in rows[0].items()])
            else:
                preview = "\n".join([
                    ", ".join([str(v) for v in row.values()])
                    for row in rows[:3]
                ])
                answer = preview
                if row_count > 3:
                    answer += f"\n...and {row_count - 3} more rows"
            
            # Store in memory
            conversation_memory_store.setdefault(session_id, []).append({
                "question": question,
                "sql": sql,
                "row_count": row_count,
                "timestamp": datetime.now().isoformat()
            })
            
            # Final response
            yield jsonl_line({
                "event": "final",
                "payload": {
                    "answer": answer,
                    "rows": rows[:50],
                    "row_count": row_count,
                    "query_used": sql,
                    "session_id": session_id,
                    "history": conversation_memory_store.get(session_id, [])
                }
            })
        
        # Return streaming response
        resp = StreamingHttpResponse(gen(), content_type="application/x-ndjson")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        resp["Access-Control-Allow-Origin"] = "*"
        return resp
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

    
def jsonl_line(obj: dict) -> str:
    """Helper to format JSONL."""
    import json
    return json.dumps(obj) + '\n'


#bfr vertical and channel 
@csrf_exempt
def ask_question_stream229(request):
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
                # if ctx.get("sql"):
                #     q_norm = normalize_question(question)
                #     matched_norm = normalize_question(matched_q)
                #     print(f"🔎 [RAG DEBUG] Normalized question: {q_norm}, matched question: {matched_norm}")

                #     if q_norm == matched_norm:
                #         sql = ctx["sql"]
                #         print(f"♻️ Reusing SQL (EXACT match) for: {matched_q}")
                #     elif score >= RAG_SIMILARITY_THRESHOLD:
                #         sql = ctx["sql"]
                #         print(f"♻️ Reusing SQL (score={score:.2f}) for: {matched_q}")
                #     else:
                #         print(f"🆕 [NEW SQL GEN] No strong match (score={score:.2f}, threshold={RAG_SIMILARITY_THRESHOLD}). Generating fresh SQL…")
                if ctx.get("sql"):
                    q_norm = normalize_question(question)
                    matched_norm = normalize_question(matched_q)
                    print(f"🔎 [RAG DEBUG] Normalized question: {q_norm}, matched question: {matched_norm}")

                    candidate_sql = ctx["sql"]

                    # 🚫 Prevent reusing churn SQL
                    if "main_cai_lib" in candidate_sql or "policy_no" in candidate_sql:
                        print("⚠️ Skipping reuse: matched SQL is from churn schema, not loan schema.")
                        sql = None
                    elif q_norm == matched_norm:
                        sql = candidate_sql
                        print(f"♻️ Reusing SQL (EXACT match) for: {matched_q}")
                    elif score >= RAG_SIMILARITY_THRESHOLD:
                        sql = candidate_sql
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
                print(f"📝 Generated SQL:\n{sql}\n", flush=True)

            except Exception as e:
                traceback.print_exc()
                yield jsonl_line({"event": "error", "message": f"SQL generation failed: {e}"})
                return

            if not sql or not sql.strip().lower().startswith(("select", "with")):
                yield jsonl_line({"event": "error", "message": "Invalid or empty SQL generated"})
                return

            yield jsonl_line({"event": "sql", "sql": sql})
            print(f"📝 Generated SQL refer:\n{sql}\n", flush=True)

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

# buffer = ""
# with requests.post(url, headers=headers, json=payload, stream=True) as r:
#   for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
#     buffer += chunk
#     while True:
#       try:
#         # Find the next complete SSE line
#         line_end = buffer.find('\n')
#         if line_end == -1:
#           break

#         line = buffer[:line_end].strip()
#         buffer = buffer[line_end + 1:]

#         if line.startswith('data: '):
#           data = line[6:]
#           if data == '[DONE]':
#             break

#           try:
#             data_obj = json.loads(data)
#             content = data_obj["choices"][0]["delta"].get("content")
#             if content:
#               print(content, end="", flush=True)
#           except json.JSONDecodeError:
#             pass
#       except Exception:
#         break


def stream_openrouter_response(question):
    """Stream response from OpenRouter API"""
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
                                yield content  # Use yield for streaming
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



# def call_llm_with_retry(prompt: str, max_retries: int = 3, base_delay: float = 1.0) -> str:
#     if not AZURE_API_KEY:
#         logger.error("AZURE_API_KEY is not defined")
#         return "API key configuration error. Please check your settings."

#     estimated_tokens = len(prompt.split()) * 1.5
#     logger.info(f"Estimated prompt tokens: {estimated_tokens}")

#     if estimated_tokens > MAX_PROMPT_TOKENS:
#         logger.error(f"Prompt too long: {estimated_tokens} tokens (max: {MAX_PROMPT_TOKENS})")
#         return "The prompt is too long for the current model. Please try with a shorter question."

#     # --- KEEP original Groq placeholders but redirect to Azure ---
#     headers = {
#         "Authorization": f"Bearer {AZURE_API_KEY}",  # not used directly (SDK handles auth)
#         "Content-Type": "application/json",
#     }

#     model = AZURE_MODEL  # replace with Azure model instead of Groq
#     logger.info(f"Starting Azure LLM call with {max_retries} max retries")

#     client = _get_client()

#     for attempt in range(max_retries):
#         try:
#             logger.info(f"Attempt {attempt + 1}/{max_retries}")

#             response = client.complete(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": _SYSTEM_PROMPT},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.1,
#                 top_p=0.9,
#                 frequency_penalty=0.0,
#                 presence_penalty=0.0,
#                 max_tokens=4000,  # Azure field
#             )

#             # Azure SDK response
#             if response and response.choices:
#                 answer = response.choices[0].message.content.strip()
#                 if answer:
#                     return answer

#         except Exception as e:
#             logger.warning(f"Retry {attempt+1} failed: {e}")
#             time.sleep(base_delay * (2 ** attempt))

#     logger.error("All retry attempts failed")
#     return "I'm currently experiencing issues reaching the Azure AI service. Please try again later."


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



# import requests

# @csrf_exempt
# def check_intent(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         question = data.get("question", "")

#         prompt = f"""
# Classify the user's intent strictly as YES or NO.

# If the question is a general greeting, chit-chat, or general knowledge (e.g. "hi", "hello", "how are you", "who is the PM of India"), respond with NO.

# If it is about querying the connected database schema or fetching data from tables, respond with YES.
# If the question is trying to analyze, query, or summarize tabular or structured data (e.g. Excel, PDF tables, database), respond with YES.

# Question: "{question}"
# Only respond YES or NO.
# """

#         try:
#             # --- Azure Config ---
#             endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT", "").rstrip("/")
#             if not endpoint.endswith("/models"):
#                 endpoint = f"{endpoint}/models"

#             api_key = os.getenv("AZURE_INFERENCE_API_KEY", "")
#             model = os.getenv("AZURE_INFERENCE_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8-prochurn-demo")
#             api_version = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")

#             client = ChatCompletionsClient(
#                 endpoint=endpoint,
#                 credential=AzureKeyCredential(api_key),
#                 api_version=api_version,
#             )

#             response = client.complete(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": "You are an intent classifier."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.0,
#                 max_tokens=10
#             )

#             answer = response.choices[0].message.content.strip()
#             return JsonResponse({"answer": answer})

#         except Exception as e:
#             return JsonResponse({"answer": "No", "error": str(e)}, status=500)


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
            # --- Azure Config ---
            endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT", "").rstrip("/")
            if not endpoint.endswith("/models"):
                endpoint = f"{endpoint}/models"

            api_key = os.getenv("AZURE_INFERENCE_API_KEY", "")
            model = os.getenv("AZURE_INFERENCE_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8-prochurn-demo")
            api_version = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")

            client = ChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
                api_version=api_version,
            )

            response = client.complete(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an intent classifier."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )

            answer = response.choices[0].message.content.strip()
            return JsonResponse({"answer": answer})

        except Exception as e:
            return JsonResponse({"answer": "No", "error": str(e)}, status=500)


# Groq working code final6-10
# @csrf_exempt

# def check_intent(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         question = data.get("question", "")

#         prompt = f"""
# Classify the user's intent strictly as YES or NO.

# If the question is a general greeting, chit-chat, or general knowledge (e.g. "hi", "hello", "how are you", "who is the PM of India"), respond with NO.

# If it is about querying the connected database schema or fetching data from tables, respond with YES.
# If the question is trying to analyze, query, or summarize tabular or structured data (e.g. Excel, PDF tables, database), respond with YES.

# Question: "{question}"
# Only respond YES or NO.
# """

#         try:
#             response = requests.post(
#                 "https://api.groq.com/openai/v1/chat/completions",  # ✅ Groq Cloud endpoint
#                 headers={
#                     "Authorization": f"Bearer {GROQ_API_KEY}",  # ✅ Use your Groq API key
#                     "Content-Type": "application/json"
#                 },
#                 json={
#                     "model": "meta-llama/llama-4-maverick-17b-128e-instruct",  # ✅ Groq model
#                     "messages": [
#                         {"role": "system", "content": "You are an intent classifier."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     "temperature": 0.0,
#                     "max_tokens": 10
#                 },
#                 timeout=30
#             )
#             result = response.json()
#             answer = result["choices"][0]["message"]["content"].strip()
#             return JsonResponse({"answer": answer})

#         except Exception as e:
#             return JsonResponse({"answer": "No", "error": str(e)}, status=500)
