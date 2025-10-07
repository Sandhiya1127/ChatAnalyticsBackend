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
from sqlalchemy import inspect

from sentence_transformers import SentenceTransformer
import chromadb
from sqlalchemy import inspect

# Initialize once
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection("schemas")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast model

def pick_target_table_from_question(question: str, schema_catalog: dict, embedder, collection, top_k: int = 3):
    """
    Auto-detect the most relevant schema.table for a question.
    Uses embeddings over schema_catalog already stored in Chroma.
    Falls back gracefully if metadata is missing or only one table exists.
    """
    print(f"🔎 [Schema Picker] Question: {question}")

    # Encode question
    q_emb = embedder.encode([question]).tolist()[0]

    # Query Chroma
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    print(f"🔎 [Schema Picker] Top-k results: {results}")

    best_table = None

    try:
        if results.get("metadatas"):
            best_meta = results["metadatas"][0][0]
            best_table = best_meta.get("table")  # ✅ safe lookup
            print(f"   → Metadata gave table: {best_table}")

        # fallback: try parsing from documents
        if not best_table and results.get("documents"):
            doc_text = results["documents"][0][0]
            if doc_text.startswith("Table "):
                best_table = doc_text.split(" ")[1]
                print(f"   → Parsed table from doc: {best_table}")

    except Exception as e:
        print(f"⚠️ Table resolution failed: {e}")

    # final fallback: if schema has only one table
    if not best_table and len(schema_catalog) == 1:
        best_table = list(schema_catalog.keys())[0]
        print(f"↩️ Falling back to only available table: {best_table}")

    # last fallback: first table in catalog
    if not best_table and schema_catalog:
        best_table = list(schema_catalog.keys())[0]
        print(f"↩️ Fallback to first table in catalog: {best_table}")

    print(f"✅ [Schema Picker] Picked table: {best_table}")
    return best_table



# --- helpers: normalize names & extract tables from SQL ---
import re

_TABLE_RE = re.compile(
    r'\b(from|join)\s+'
    r'("?[a-zA-Z_][\w$]*"?)[\s\.]+("?[a-zA-Z_][\w$]*"?)',
    flags=re.IGNORECASE | re.DOTALL,
)

def norm_fq_table(name: str) -> str:
    """Normalize ' "stage"."loan_main_2024" ' -> 'stage.loan_main_2024' (lowercased)."""
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.replace(' ', '')
    name = name.replace('"', '')
    return name.lower()

def extract_tables(sql: str) -> set[str]:
    """Return set of normalized schema.table references found in SQL."""
    found = set()
    sql = re.sub(r'\s*\.\s*', '.', sql)  # tolerate spaces around dots
    for _, schema, table in _TABLE_RE.findall(sql):
        s = schema.replace('"', '').strip().lower()
        t = table.replace('"', '').strip().lower()
        found.add(f"{s}.{t}")
    return found

def validate_sql_against_catalog(sql, schema_catalog):
    print("🔎 Validating SQL against schema_catalog...")
    for fq in schema_catalog.keys():
        unquoted = fq.replace('"','')
        if fq in sql or unquoted in sql:
            print(f"✅ SQL references known table: {fq}")
            return sql
    print("❌ SQL did not match any known table!")
    return sql  # <-- don't nuke SQL; just warn. Let DB error if truly wrong.


# def validate_sql_against_catalog(sql, schema_catalog):
#     """
#     Ensure generated SQL actually references a known schema.table
#     """
#     print("🔎 Validating SQL against schema_catalog...")
#     tables = list(schema_catalog.keys())
#     for t in tables:
#         if t in sql:
#             print(f"✅ SQL references known table: {t}")
#             return sql

#     print("❌ SQL did not match any known table!")
#     return "-- ERROR: Generated SQL does not match any known table"


def find_relevant_table(question, schema_catalog, embedder, collection, top_k=3):
    """
    Use embeddings to find the best matching schema.table for a question.
    Handles missing metadata gracefully.
    """
    print(f"🔍 Searching for relevant table for question: {question}")
    q_emb = embedder.encode([question]).tolist()[0]
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    print(f"🔎 [Schema Picker] Raw results: {results}")

    try:
        best_meta = results["metadatas"][0][0]
        best_table = best_meta.get("table")  # ✅ safe lookup
        if not best_table:
            # fallback: parse table from documents if metadata missing
            doc_text = results["documents"][0][0]
            if doc_text.startswith("Table "):
                best_table = doc_text.split(" ")[1]
        if not best_table and len(schema_catalog) == 1:
            best_table = list(schema_catalog.keys())[0]

        print(f"✅ Best match: {best_table}")
        return best_table

    except Exception as e:
        print(f"⚠️ Table resolution failed: {e}")
        if len(schema_catalog) == 1:
            fallback = list(schema_catalog.keys())[0]
            print(f"↩️ Falling back to only available table: {fallback}")
            return fallback
        return None


import re
import logging

def build_type_map_from_catalog(schema_catalog: dict) -> dict:
    """
    Build {column_name: inferred_type} from schema_catalog.
    Uses simple heuristics on column names to guess text vs numeric.
    """
    type_map = {}
    print("🔎 Building type map from schema_catalog...")
    for fq_table, cols in schema_catalog.items():
        print(f"  📂 Table: {fq_table}")
        for col in cols:
            col_l = col.lower()
            if any(t in col_l for t in ["name", "text", "string", "char", "purpose", "city", "state", "vertical", "bank", "group"]):
                type_map[col_l] = "text"
                print(f"    ✅ {col} → text")
            elif any(t in col_l for t in ["id", "int", "num", "amount", "score", "rate", "term", "income"]):
                type_map[col_l] = "numeric"
                print(f"    ✅ {col} → numeric")
            else:
                type_map[col_l] = "unknown"
                print(f"    ⚠️ {col} → unknown")
    print("✅ Final type_map:", type_map)
    return type_map


def build_dynamic_fallback_map(schema_catalog: dict) -> dict:
    """
    Dynamically build a fallback map from *_id columns to related text columns.
    Example:
      customer_id → customer_name / customer_state (if present in same table)
      merchant_id → merchant_name / merchant_vertical
    """
    fallback_map = {}
    print("🔎 Building fallback map...")

    for fq_table, cols in schema_catalog.items():
        lower_cols = [c.lower() for c in cols]
        print(f"  📂 Checking table {fq_table}: {lower_cols}")

        for col in lower_cols:
            if col.endswith("_id") or col == "id":
                base = col.replace("_id", "")
                candidates = [c for c in lower_cols if c.startswith(base) and c not in {col}]
                preferred = next((c for c in candidates if any(s in c for s in ["name", "state", "vertical"])), None)

                if preferred:
                    fallback_map[f"{fq_table}.{col}"] = f"{fq_table}.{preferred}"
                    print(f"    🔗 {fq_table}.{col} → {fq_table}.{preferred}")
                else:
                    print(f"    ⚠️ No fallback found for {fq_table}.{col}")

    print("✅ Final fallback_map:", fallback_map)
    return fallback_map


def auto_fix_schema_mismatches(sql: str, schema_catalog: dict) -> str:
    """
    Auto-fix SQL mismatches where numeric ID columns are compared to string literals,
    using dynamically inferred fallback columns (like *_name, *_state).
    """
    print("🔧 Running auto_fix_schema_mismatches...")
    type_map = build_type_map_from_catalog(schema_catalog)
    fallback_map = build_dynamic_fallback_map(schema_catalog)

    def repl(match):
        col, val = match.group(1), match.group(2)
        col_l = col.strip('"').lower()
        print(f"   🔍 Checking condition: {col} = {val}")

        # Find fully qualified fallback if exists
        fq_matches = [k for k in fallback_map.keys() if k.endswith(f".{col_l}")]
        fallback = fallback_map[fq_matches[0]] if fq_matches else None
        print(f"      • Type: {type_map.get(col_l)} | Fallback: {fallback}")

        if col_l in type_map and type_map[col_l] == "numeric" and re.match(r"'[A-Za-z ]+'", val):
            if fallback and fallback.split(".")[-1] in type_map and type_map[fallback.split(".")[-1]] == "text":
                print(f"      ✅ Replacing {col} with fallback {fallback}")
                return f"{fallback} ILIKE '%' || {val.strip()} || '%'"
            else:
                print(f"      ❌ Invalid comparison: {col} (numeric) vs {val} (string)")
                return f"-- ERROR: {col} (numeric) compared to {val} (string)"

        print("      ↔️ No change needed")
        return match.group(0)

    sql_fixed = re.sub(r'(\w+)\s*=\s*(\'[^\']+\')', repl, sql, flags=re.IGNORECASE)
    print("✅ Final auto-fixed SQL:\n", sql_fixed)
    return sql_fixed



import logging
logger = logging.getLogger(__name__)

import re, json, logging
from difflib import SequenceMatcher

from sqlalchemy import inspect

import logging
logger = logging.getLogger(__name__)
from typing import Dict, List
from sqlalchemy import inspect

CORE_TABLES = {"app_main_2024", "loan_main_2024"}
PREFERRED_SCHEMA = "stage"  # we’ll prefer these FQNs if present

def build_schema_catalog111(engine) -> Dict[str, List[str]]:
    """
    Introspect ONLY:
      • "*/app_main_2024"
      • "*/loan_main_2024"
    Prefer the 'stage' schema if those tables exist there.

    Returns: { '"schema"."table"': [col1, col2, ...] }
    """
    insp = inspect(engine)
    catalog: Dict[str, List[str]] = {}

    print("🔎 Starting STRICT catalog for core tables only...")
    # Exclude system schemas
    all_schemas = [
        s for s in insp.get_schema_names()
        if not s.startswith("pg_") and s not in {"information_schema"}
    ]
    if not all_schemas:
        print("⚠️ No non-system schemas found.")
        return catalog

    # Helper: add table if exists in schema
    def _try_add(schema: str, table: str) -> bool:
        if table in insp.get_table_names(schema=schema):
            cols = [c["name"] for c in insp.get_columns(table, schema=schema)]
            fq = f'"{schema}"."{table}"'
            catalog[fq] = cols
            print(f"    ✅ Included {fq} → {len(cols)} cols")
            return True
        return False

    # 1) Prefer PREFERRED_SCHEMA
    if PREFERRED_SCHEMA in all_schemas:
        print(f"📌 Prefer schema: {PREFERRED_SCHEMA}")
        for t in sorted(CORE_TABLES):
            _try_add(PREFERRED_SCHEMA, t)

    # 2) If any core table missing, scan other schemas for just those tables
    missing = {t for t in CORE_TABLES if not any(f'."{t}"' in k for k in catalog.keys())}
    if missing:
        print(f"🔎 Missing in '{PREFERRED_SCHEMA}': {sorted(missing)} → scanning other schemas")
        for schema in all_schemas:
            if schema == PREFERRED_SCHEMA:
                continue
            for t in sorted(list(missing)):
                if _try_add(schema, t):
                    missing.discard(t)
            if not missing:
                break

    if missing:
        print(f"⚠️ Still missing: {sorted(missing)}")

    print("✅ Final schema_catalog keys:", list(catalog.keys()))
    return catalog



from typing import Dict, List

# Only embed these tables (any schema)
_EMBED_ONLY = {"app_main_2024", "loan_main_2024"}

def _is_core_table(fq_table: str) -> bool:
    """
    fq_table is like:  '"schema"."table"'
    """
    try:
        # Strip quotes and split
        schema, table = fq_table.replace('"', '').split('.', 1)
        return table in _EMBED_ONLY
    except Exception:
        return False
    

def pick_fallback_table(schema_catalog: Dict[str, List[str]]) -> str | None:
    """
    Prefer the two core FQNs if present in the catalog, otherwise None.
    """
    # Prefer stage.* names if present
    preferred = [
        f'"{PREFERRED_SCHEMA}"."app_main_2024"',
        f'"{PREFERRED_SCHEMA}"."loan_main_2024"',
    ]
    for fq in preferred:
        if fq in schema_catalog:
            return fq

    # Else, any-core
    for fq in schema_catalog:
        try:
            _, table = fq.replace('"', '').split('.', 1)
            if table in CORE_TABLES:
                return fq
        except Exception:
            continue
    return None

from typing import Dict, List

def embed_schema_catalog(user_id: str, db_id: str, schema_catalog: Dict[str, List[str]], embedder, collection) -> None:
    """
    Store ONLY app_main_2024 and loan_main_2024 (from any schema) into Chroma.
    """
    print(f"🔎 Embedding (filtered) schema for user={user_id}, db_id={db_id}")
    if not schema_catalog:
        print("⚠️ schema_catalog is empty; nothing to embed.")
        return

    def _is_core(fq: str) -> bool:
        # fq like:  '"schema"."table"'
        try:
            _, table = fq.replace('"', '').split('.', 1)
            return table in CORE_TABLES
        except Exception:
            return False

    items = [(fq, cols) for fq, cols in schema_catalog.items() if _is_core(fq)]
    if not items:
        print("⚠️ No core tables found to embed.")
        return

    docs, metas, ids = [], [], []
    for i, (fq, cols) in enumerate(items):
        doc = f"Table {fq} has columns: {', '.join(cols)}"
        docs.append(doc)
        metas.append({"table": fq, "user_id": user_id, "db_id": db_id})
        ids.append(f"{db_id}_{i}")
        print(f"  ➕ Prepared doc for {fq}: {doc}")

    embeddings = embedder.encode(docs).tolist()
    print(f"🧩 Generated {len(embeddings)} embeddings, uploading to Chroma...")

    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metas, documents=docs)
    print("✅ Filtered schema catalog embedded (only core tables).")


# # all svhema and tab embeddings----------------------------------------
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



def infer_required_columns_dynamic(question: str, schema_cols: list[str], threshold: float = 0.7):
    """Dynamically infer which DB columns the question is referring to using fuzzy matching."""
    q_tokens = re.findall(r"\w+", question.lower())
    required = []

    for token in q_tokens:
        best_match, best_score = None, 0
        for col in schema_cols:
            score = SequenceMatcher(None, token, col.lower()).ratio()
            if score > best_score:
                best_match, best_score = col, score
        if best_match and best_score >= threshold:
            required.append(best_match)

    return list(set(required))

def check_question_vs_schema(question: str, schema_cols: list[str]):
    if not isinstance(schema_cols, (list, set, tuple)):
        return {
            "status": "error",
            "message": f"❌ Invalid schema_cols type: {type(schema_cols)}",
            "required": []
        }

    required = infer_required_columns_dynamic(question, schema_cols)
    missing = [col for col in required if col.lower() not in [c.lower() for c in schema_cols]]

    if missing:
        return {
            "status": "error",
            "message": f"❌ Cannot answer with current table. Missing column(s): {missing}",
            "required": required
        }
    return {"status": "ok", "required": required}



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

import re
import logging
import json

def sanitize_sql_for_nulls(sql: str, question: str = "", schema_catalog: dict = None) -> str:
    """
    BULLETPROOF SQL Sanitizer (string-literal safe, WHERE/HAVING shielded, CASE shielded)
    - Handles NULL/NaN/empty values across GROUP BY, SELECT, aggregates.
    - Safe numeric casting, recursion for subqueries and CTEs.
    - Shields CASE, WHERE/HAVING, IN blocks.
    - Preserves ID columns and guards against division by zero.
    """
    if not sql or not sql.strip().lower().startswith(("select", "with")):
        return sql

    original_sql = sql

    try:
        # ------------------- Step 1: Extract string literals -------------------
        str_map = {}
        def extract_literals(match):
            key = f"__STR_LITERAL_{len(str_map)}__"
            str_map[key] = match.group(0)
            return key
        sql = re.sub(r"'([^']|'')*'", extract_literals, sql)

        # ------------------- Step 2: Strip comments and dangling commas -------------------
        sql = re.sub(r'--[^\n]*', '', sql)
        sql = re.sub(
            r",\s*(\)|FROM|WHERE|GROUP|ORDER|HAVING|LIMIT)",
            r" \1", sql, flags=re.IGNORECASE
        )

        # ------------------- Step 3: Fix EXTRACT on date/time columns -------------------
        for fq, cols in (schema_catalog or {}).items():
            for col in cols:
                col_str = str(col)
                if any(x in col_str.lower() for x in ['date', 'time', 'created', 'updated']):
                    sql = re.sub(
                        rf'EXTRACT\s*\(\s*(\w+)\s+FROM\s+(["\w.]*\.)?{re.escape(col_str)}\s*\)',
                        rf'EXTRACT(\1 FROM \2{col_str}::DATE)',
                        sql,
                        flags=re.IGNORECASE
                    )

        # ------------------- Step 4: Auto-patch "this vs last month" CTEs -------------------
        def patch_this_vs_last_month(sql_text: str) -> str:
            cte_matches = re.findall(r"\b(this_month|last_month)_(\w+)\b", sql_text, re.IGNORECASE)
            if not cte_matches:
                return sql_text

            seen = {m[0].lower(): m[1] for m in cte_matches}
            alias_suffix = list(seen.values())[0]

            if "this_month" in seen and "last_month" in seen:
                return sql_text

            with_block_match = re.search(
                rf"(\b(?:this_month|last_month)_{alias_suffix}\b\s+AS\s*\(.*?\))",
                sql_text, flags=re.IGNORECASE | re.DOTALL
            )
            if not with_block_match:
                return sql_text

            existing_block = with_block_match.group(1)

            def rewrite_dates(block: str, target: str) -> str:
                if target == "this_month":
                    block = re.sub(r"CURRENT_DATE\s*-\s*INTERVAL\s*'1 month'",
                                   "CURRENT_DATE", block, flags=re.IGNORECASE)
                else:  # last_month
                    block = re.sub(r"CURRENT_DATE(\s*\+?\s*INTERVAL\s*'1 month')?",
                                   "CURRENT_DATE - INTERVAL '1 month'", block, flags=re.IGNORECASE)
                return re.sub(r"\b(this_month|last_month)_"+alias_suffix,
                              f"{target}_month_{alias_suffix}", block, flags=re.IGNORECASE)

            if "this_month" not in seen:
                new_block = rewrite_dates(existing_block, "this_month")
                sql_text = sql_text.replace(existing_block, existing_block + ",\n" + new_block)
            elif "last_month" not in seen:
                new_block = rewrite_dates(existing_block, "last_month")
                sql_text = sql_text.replace(existing_block, existing_block + ",\n" + new_block)

            return sql_text

        sql = patch_this_vs_last_month(sql)

        # ------------------- Step 5: Schema parsing -------------------
        COL_TYPES = {}
        PRIMARY_ID_COLS = []
        if schema_catalog:
            for fq_table, cols in schema_catalog.items():
                for col in cols:
                    col_str = str(col)
                    COL_TYPES[col_str.lower()] = "text"
                    if col_str.lower().endswith("_id") or col_str.lower() == "id":
                        PRIMARY_ID_COLS.append(col_str)
        else:
            COL_TYPES = {}
            PRIMARY_ID_COLS = []

        print("🔑 Dynamic ID columns:", PRIMARY_ID_COLS)

        # ------------------- Step 6: Safe numeric cast -------------------
        def safe_numeric_cast(expr: str, COL_TYPES: dict = COL_TYPES, in_where: bool = False) -> str:
            expr = str(expr).strip()
            if in_where:
                return expr
            if expr.lower() in COL_TYPES:
                ctype = COL_TYPES[expr.lower()]
                if any(t in ctype for t in ["int", "decimal", "numeric", "double", "real", "float"]):
                    return expr
            if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", expr, flags=re.IGNORECASE):
                return expr
            if expr.lower().endswith("_id") or expr.lower() in [str(x).lower() for x in PRIMARY_ID_COLS]:
                return expr
            if re.search(r"\bCASE\b|\bWHEN\b|\bEND\b", expr, re.IGNORECASE):
                return expr
            if (expr.isdigit() or re.match(r"^\d+(\.\d+)?$", expr)
                or expr.startswith("__STR_LITERAL_")
                or expr.startswith("(")):
                return expr
            return f"""(
                CASE 
                    WHEN {expr} IS NULL THEN NULL
                    WHEN TRIM({expr}::text) = '' THEN NULL
                    WHEN UPPER(TRIM({expr}::text)) IN ('NAN','NULL','N/A','NA') THEN NULL
                    WHEN TRIM({expr}::text) ~ '^-?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$'
                        THEN TRIM({expr}::text)::NUMERIC
                    ELSE NULL
                END
            )"""

        # ------------------- Step 7: Shield WHERE/HAVING -------------------
        def shield_where_blocks(sql_text: str):
            blocks = {}
            def repl(match):
                key = f"__WHERE_BLOCK_{len(blocks)}__"
                blocks[key] = match.group(0)
                return key
            pattern = r"\b(WHERE|HAVING)\b.*?(?=\b(GROUP|ORDER|HAVING|LIMIT|UNION|$))"
            shielded = re.sub(pattern, repl, sql_text, flags=re.IGNORECASE | re.DOTALL)
            return shielded, blocks

        def unshield_where_blocks(sql_text: str, blocks: dict) -> str:
            for key, val in blocks.items():
                sql_text = sql_text.replace(key, val)
            return sql_text

        # ------------------- Step 8: Ensure ID columns -------------------
        def ensure_id_columns(sql_text: str) -> str:
            if any(kw in sql_text.upper() for kw in ["GROUP BY", "COUNT(", "SUM(", "AVG(", "MIN(", "MAX("]):
                return sql_text
            def repl(match):
                select_block = match.group(0)
                lower_block = select_block.lower()
                needed_ids = [str(col) for col in PRIMARY_ID_COLS if str(col).lower() in sql.lower()]
                missing = [col for col in needed_ids if col.lower() not in lower_block]
                if missing:
                    inject = ", ".join(missing)
                    return re.sub(r"(?i)\bSELECT\b", f"SELECT {inject},", select_block, count=1)
                return select_block
            return re.sub(r"SELECT\s+.*?\bFROM\b", repl, sql_text, flags=re.IGNORECASE | re.DOTALL)

        sql = ensure_id_columns(sql)

        # ------------------- Step 9: Recursive sanitization -------------------
        def process_query_recursively(query_sql: str, level: int = 0) -> str:
            if query_sql.strip().lower().startswith("with"):
                cte_pattern = r"(WITH\s+.*?)(\s+SELECT\s+.*)"
                cte_match = re.search(cte_pattern, query_sql, flags=re.IGNORECASE | re.DOTALL)
                if cte_match:
                    cte_part = process_cte_definitions(cte_match.group(1))
                    main_query = sanitize_single_query(cte_match.group(2), level)
                    return cte_part + main_query
            return sanitize_single_query(query_sql, level)

        def process_cte_definitions(cte_sql: str) -> str:
            cte_pattern = r"(\w+)\s+AS\s*\((.*?)\)(?=\s*,\s*\w+\s+AS\s*\(|\s*SELECT|\s*$)"
            def repl(match):
                return f"{match.group(1)} AS ({sanitize_single_query(match.group(2).strip(), level=1)})"
            return re.sub(cte_pattern, repl, cte_sql, flags=re.IGNORECASE | re.DOTALL)

        def sanitize_single_query(single_sql: str, level: int = 0) -> str:
            subq_pattern = r"\(\s*(SELECT\s+.*?)\s*\)(?=\s*(?:AS\s+\w+)?\s*(?:WHERE|GROUP|ORDER|HAVING|LIMIT|UNION|\)|,|$))"
            single_sql = re.sub(
                subq_pattern,
                lambda m: f"({sanitize_single_query(m.group(1), level+1)})",
                single_sql,
                flags=re.IGNORECASE | re.DOTALL
            )
            return apply_main_sanitization(single_sql, level)

        # ------------------- Step 10: Main sanitization -------------------
        def apply_main_sanitization(sql: str, level: int = 0) -> str:
            sql, blocks = shield_where_blocks(sql)

            # Shield CASE blocks
            case_map = {}
            def shield_case_blocks(sql_text: str) -> str:
                out, i, n = [], 0, len(sql_text)
                while i < n:
                    if sql_text[i:i+4].upper() == "CASE" and (i == 0 or not sql_text[i-1].isalnum()):
                        depth, j = 1, i + 4
                        while j < n and depth > 0:
                            if sql_text[j:j+4].upper() == "CASE" and (j == 0 or not sql_text[j-1].isalnum()):
                                depth += 1; j += 4
                            elif sql_text[j:j+3].upper() == "END" and (j == 0 or not sql_text[j-1].isalnum()):
                                depth -= 1; j += 3
                            else:
                                j += 1
                        block = sql_text[i:j]
                        key = f"__CASE_BLOCK_{len(case_map)}__"
                        case_map[key] = block
                        out.append(key)
                        i = j
                    else:
                        out.append(sql_text[i])
                        i += 1
                return "".join(out)

            sql = shield_case_blocks(sql)

            # Arithmetic, aggregates, division guards
            sql = re.sub(
                r"([a-zA-Z0-9_\".]+)\s*/\s*([a-zA-Z0-9_\".]+)(?!\s*\()",
                lambda m: f"{safe_numeric_cast(m.group(1), COL_TYPES)} / NULLIF({safe_numeric_cast(m.group(2), COL_TYPES)},0)",
                sql, flags=re.IGNORECASE
            )

            sql = re.sub(r"\b(SUM|AVG|MIN|MAX|COUNT)\s*\(([^()]+?)\)",
                         lambda m: f"{m.group(1).upper()}({safe_numeric_cast(m.group(2), COL_TYPES, False)})",
                         sql, flags=re.IGNORECASE | re.DOTALL)

            sql = ensure_id_columns(sql)

            # Restore IN blocks and CASE blocks
            sql = unshield_where_blocks(sql, blocks)
            for key, val in case_map.items():
                sql = sql.replace(key, val)

            sql = re.sub(r'\s+', ' ', sql).strip()
            return sql

        result = process_query_recursively(sql)

        # Restore string literals
        for key, val in str_map.items():
            result = result.replace(key, val)

        # Final post-processing: intervals, division by zero, whitespace
        result = re.sub(r"'(\d{4}-\d{2}-\d{2})'\s*([+-]\s*INTERVAL)", r"DATE '\1' \2", result, flags=re.IGNORECASE)
        result = re.sub(r"/\s*(\(?\s*(COUNT|SUM|AVG|MIN|MAX)\s*\([^()]*\)\s*\)?)", r"/ NULLIF(\1,0)", result, flags=re.IGNORECASE)
        result = re.sub(r'\s+', ' ', result).strip()

        # Safety fallback
        if not re.search(r'\bSELECT\b.*\bFROM\b', result, flags=re.IGNORECASE | re.DOTALL):
            return original_sql
        if result.count("(") != result.count(")"):
            return original_sql

        return result

    except Exception as e:
        print(f"⚠️ [SQL SANITIZER] Error: {e}")
        return original_sql



def sanitize_sql_for_nulls310(sql: str, question: str = "", schema_catalog: dict = None) -> str:
    """
    BULLETPROOF SQL Sanitizer (string-literal safe, WHERE/HAVING shielded, CASE shielded)
    - Skips touching anything inside '...' (regex/text literals).
    - Skips touching CASE ... END blocks (nested-safe).
    - Handles NULL/NaN/empty values across GROUP BY, SELECT, aggregates.
    - Smart type casting for text vs numeric columns.
    - Recursive handling of CTEs and subqueries.
    - Safe numeric casting with guards for literals/functions.
    - WHERE/HAVING sections are shielded (only light filters allowed).
    - Validation fallback to original query if malformed.
    """

    if not sql or not sql.strip().lower().startswith(("select", "with")):
        return sql

    original_sql = sql

    try:
        # ----------------------------------------
        # Step 1. Extract string literals
        # ----------------------------------------
        str_map = {}
        def extract_literals(match):
            key = f"__STR_LITERAL_{len(str_map)}__"
            str_map[key] = match.group(0)
            return key
        sql = re.sub(r"'([^']|'')*'", extract_literals, sql)

        # ----------------------------------------
        # Step 2. Strip comments + dangling commas
        # ----------------------------------------
        sql = re.sub(r'--[^\n]*', '', sql)  
        sql = re.sub(
            r",\s*(\)|FROM|WHERE|GROUP|ORDER|HAVING|LIMIT)",
            r" \1", sql, flags=re.IGNORECASE
        )

        # After Step 1 (extract literals), add:
# ----------------------------------------
# Fix EXTRACT on text columns
        # ----------------------------------------
        for fq, cols in (schema_catalog or {}).items():
            for col in cols:
                if any(x in col.lower() for x in ['date', 'time', 'created', 'updated']):
                    # Fix EXTRACT(unit FROM col) → EXTRACT(unit FROM col::DATE)
                    sql = re.sub(
                        rf'EXTRACT\s*\(\s*(\w+)\s+FROM\s+(["\w.]*\.)?{re.escape(col)}\s*\)',
                        rf'EXTRACT(\1 FROM \2{col}::DATE)',
                        sql,
                        flags=re.IGNORECASE
                    )

        # ----------------------------------------
        # Step 2b. Auto-patch "this vs last month" CTEs dynamically
        # ----------------------------------------
        def patch_this_vs_last_month(sql_text: str) -> str:
            cte_matches = re.findall(r"\b(this_month|last_month)_(\w+)\b", sql_text, re.IGNORECASE)
            if not cte_matches:
                return sql_text

            seen = {m[0].lower(): m[1] for m in cte_matches}
            alias_suffix = list(seen.values())[0]

            if "this_month" in seen and "last_month" in seen:
                return sql_text

            with_block_match = re.search(
                rf"(\b(?:this_month|last_month)_{alias_suffix}\b\s+AS\s*\(.*?\))",
                sql_text, flags=re.IGNORECASE | re.DOTALL
            )
            if not with_block_match:
                return sql_text

            existing_block = with_block_match.group(1)

            def rewrite_dates(block: str, target: str) -> str:
                if target == "this_month":
                    block = re.sub(r"CURRENT_DATE\s*-\s*INTERVAL\s*'1 month'",
                                   "CURRENT_DATE", block, flags=re.IGNORECASE)
                else:  # last_month
                    block = re.sub(r"CURRENT_DATE(\s*\+?\s*INTERVAL\s*'1 month')?",
                                   "CURRENT_DATE - INTERVAL '1 month'", block, flags=re.IGNORECASE)
                return re.sub(r"\b(this_month|last_month)_"+alias_suffix,
                              f"{target}_month_{alias_suffix}", block, flags=re.IGNORECASE)

            if "this_month" not in seen:
                new_block = rewrite_dates(existing_block, "this_month")
                sql_text = sql_text.replace(existing_block, existing_block + ",\n" + new_block)
            elif "last_month" not in seen:
                new_block = rewrite_dates(existing_block, "last_month")
                sql_text = sql_text.replace(existing_block, existing_block + ",\n" + new_block)

            return sql_text

        sql = patch_this_vs_last_month(sql)

        def extract_column_types(schema_text: str) -> dict:
            """
            Parse FULL_SCHEMA and return {col_name: col_type}.
            Example line: "- application_id: bigint"
            """
            col_types = {}
            for line in schema_text.splitlines():
                line = line.strip()
                if not line.startswith("- "):
                    continue
                parts = line.replace("-", "").split(":", 1)
                if len(parts) == 2:
                    col = parts[0].strip()
                    ctype = parts[1].strip().lower()
                    col_types[col.lower()] = ctype
            return col_types


        def extract_primary_ids_from_schema(schema_text: str):
            """
            Parse FULL_SCHEMA and return list of primary identifier columns.
            """
            id_cols = []
            for line in schema_text.splitlines():
                line = line.strip()
                if not line.startswith("- "):
                    continue
                parts = line.replace("-", "").split(":", 1)
                if not parts:
                    continue
                col = parts[0].strip()
                if col.lower().endswith("_id") or col.lower() == "id":
                    id_cols.append(col)
            return id_cols

        # ----------------------------------------
        # 🔑 PATCH: Use schema_catalog instead of FULL_SCHEMA
        # ----------------------------------------
        COL_TYPES = {}
        PRIMARY_ID_COLS = []
        if schema_catalog:
            for fq_table, cols in schema_catalog.items():
                for col in cols:
                    COL_TYPES[col.lower()] = "text"  # fallback
                    if col.lower().endswith("_id") or col.lower() == "id":
                        PRIMARY_ID_COLS.append(col)
        else:
            # fallback to FULL_SCHEMA if no catalog passed
            COL_TYPES = extract_column_types(FULL_SCHEMA) 
            PRIMARY_ID_COLS = extract_primary_ids_from_schema(FULL_SCHEMA)

        print("🔑 Dynamic ID columns:", PRIMARY_ID_COLS)

        # ----------------------------------------
        # Step 1. Dynamic schema validation
        # ----------------------------------------
        schema_cols = list(COL_TYPES.keys()) if isinstance(COL_TYPES, dict) else []
        if not isinstance(schema_cols, list):
            logging.error(f"[SQL SANITIZER] schema_cols is not list but {type(schema_cols)}")
            schema_cols = []
        if question:
            analysis = check_question_vs_schema(question, schema_cols)
            if analysis["status"] == "error":
                logging.warning(f"[SQL SANITIZER] {analysis['message']}")
                return json.dumps(analysis)
            logging.debug(f"[SQL SANITIZER] Required cols = {analysis['required']}")

        # ----------------------------------------
        # safe_numeric_cast (schema-aware already defined inside)
        # ----------------------------------------
        def safe_numeric_cast(expr: str, COL_TYPES: dict = COL_TYPES, in_where: bool = False) -> str:
            expr = expr.strip()
            if in_where:
                return expr
            if expr.lower() in COL_TYPES:
                ctype = COL_TYPES[expr.lower()]
                if any(t in ctype for t in ["int", "decimal", "numeric", "double", "real", "float"]):
                    return expr
            if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", expr, flags=re.IGNORECASE):
                return expr
            if expr.lower().endswith("_id") or expr.lower() in [x.lower() for x in PRIMARY_ID_COLS]:
                return expr
            if re.search(r"\bCASE\b|\bWHEN\b|\bEND\b", expr, re.IGNORECASE):
                return expr
            if (expr.isdigit() or re.match(r"^\d+(\.\d+)?$", expr)
                or expr.startswith("__STR_LITERAL_")
                or expr.startswith("(")):
                return expr
            return f"""(
                CASE 
                    WHEN {expr} IS NULL THEN NULL
                    WHEN TRIM({expr}::text) = '' THEN NULL
                    WHEN UPPER(TRIM({expr}::text)) IN ('NAN','NULL','N/A','NA') THEN NULL
                    WHEN TRIM({expr}::text) ~ '^-?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$'
                        THEN TRIM({expr}::text)::NUMERIC
                    ELSE NULL
                END
            )"""


        # ----------------------------------------
        # Shield WHERE/HAVING
        # ----------------------------------------
        def shield_where_blocks(sql_text: str):
            blocks = {}
            def repl(match):
                key = f"__WHERE_BLOCK_{len(blocks)}__"
                blocks[key] = match.group(0)
                return key
            pattern = r"\b(WHERE|HAVING)\b.*?(?=\b(GROUP|ORDER|HAVING|LIMIT|UNION|$))"
            shielded = re.sub(pattern, repl, sql_text, flags=re.IGNORECASE | re.DOTALL)
            return shielded, blocks

        def unshield_where_blocks(sql_text: str, blocks: dict) -> str:
            for key, val in blocks.items():
                sql_text = sql_text.replace(key, val)
            return sql_text

        # import re

        # def extract_primary_ids_from_schema(schema_text: str):
        #     """
        #     Parse FULL_SCHEMA and return list of primary identifier columns.
        #     """
        #     id_cols = []
        #     for line in schema_text.splitlines():
        #         line = line.strip()
        #         if not line.startswith("- "):
        #             continue
        #         col = line.split()[1] if line.startswith("-") else None
        #         if not col:
        #             continue
        #         col = col.replace(":", "").strip()
        #         if col.lower().endswith("_id") or col.lower() == "id":
        #             id_cols.append(col)
        #     return id_cols




        
        # --- Ensure ID columns are preserved in CTEs/subqueries ---
        # def ensure_id_columns(sql_text: str) -> str:
        #     def repl(match):
        #         select_block = match.group(0)
        #         lower_block = select_block.lower()

        #         needed_ids = [col for col in PRIMARY_ID_COLS if col in sql.lower()]
        #         missing = [col for col in needed_ids if col not in lower_block]
        #         if missing:
        #             inject = ", ".join(missing)
        #             return re.sub(r"(?i)\bSELECT\b", f"SELECT {inject},", select_block, count=1)
        #         return select_block

        #     return re.sub(r"SELECT\s+.*?\bFROM\b", repl, sql_text, flags=re.IGNORECASE | re.DOTALL)

        def ensure_id_columns(sql_text: str) -> str:
            # 🚫 Skip ID injection for aggregates / grouped queries
            if any(kw in sql_text.upper() for kw in ["GROUP BY", "COUNT(", "SUM(", "AVG(", "MIN(", "MAX("]):
                return sql_text

            def repl(match):
                select_block = match.group(0)
                lower_block = select_block.lower()

                needed_ids = [col for col in PRIMARY_ID_COLS if col in sql.lower()]
                missing = [col for col in needed_ids if col not in lower_block]

                if missing:
                    inject = ", ".join(missing)
                    return re.sub(r"(?i)\bSELECT\b", f"SELECT {inject},", select_block, count=1)
                return select_block

            return re.sub(r"SELECT\s+.*?\bFROM\b", repl, sql_text, flags=re.IGNORECASE | re.DOTALL)


        # Call after patch_this_vs_last_month
        sql = ensure_id_columns(sql)

                # Step X: Guard against division by zero
        def patch_division_by_zero(sql_text: str) -> str:
            # Replace any `/ COUNT(...` with `/ NULLIF(COUNT(...),0)`
            sql_text = re.sub(
                r"/\s*(COUNT\s*\([^)]*\))",
                r"/ NULLIF(\1,0)",
                sql_text,
                flags=re.IGNORECASE,
            )
            # Replace any `/ SUM(...` with `/ NULLIF(SUM(...),0)`
            sql_text = re.sub(
                r"/\s*(SUM\s*\([^)]*\))",
                r"/ NULLIF(\1,0)",
                sql_text,
                flags=re.IGNORECASE,
            )
            return sql_text

        sql = patch_division_by_zero(sql)



        # ----------------------------------------
        # Recursive processors
        # ----------------------------------------
        def process_query_recursively(query_sql: str, level: int = 0) -> str:
            if query_sql.strip().lower().startswith("with"):
                cte_pattern = r"(WITH\s+.*?)(\s+SELECT\s+.*)"
                cte_match = re.search(cte_pattern, query_sql, flags=re.IGNORECASE | re.DOTALL)
                if cte_match:
                    cte_part = process_cte_definitions(cte_match.group(1))
                    main_query = sanitize_single_query(cte_match.group(2), level)
                    return cte_part + main_query
            return sanitize_single_query(query_sql, level)

        def process_cte_definitions(cte_sql: str) -> str:
            cte_pattern = r"(\w+)\s+AS\s*\((.*?)\)(?=\s*,\s*\w+\s+AS\s*\(|\s*SELECT|\s*$)"
            def repl(match):
                return f"{match.group(1)} AS ({sanitize_single_query(match.group(2).strip(), level=1)})"
            return re.sub(cte_pattern, repl, cte_sql, flags=re.IGNORECASE | re.DOTALL)

        def sanitize_single_query(single_sql: str, level: int = 0) -> str:
            subq_pattern = r"\(\s*(SELECT\s+.*?)\s*\)(?=\s*(?:AS\s+\w+)?\s*(?:WHERE|GROUP|ORDER|HAVING|LIMIT|UNION|\)|,|$))"
            single_sql = re.sub(
                subq_pattern,
                lambda m: f"({sanitize_single_query(m.group(1), level+1)})",
                single_sql,
                flags=re.IGNORECASE | re.DOTALL
            )
            return apply_main_sanitization(single_sql, level)

        # ----------------------------------------
        # Main sanitization
        # ----------------------------------------
        def apply_main_sanitization(sql: str, level: int = 0) -> str:
            sql, blocks = shield_where_blocks(sql)

            case_map = {}
            def shield_case_blocks(sql_text: str) -> str:
                out, i, n = [], 0, len(sql_text)
                while i < n:
                    if sql_text[i:i+4].upper() == "CASE" and (i == 0 or not sql_text[i-1].isalnum()):
                        depth, j = 1, i + 4
                        while j < n and depth > 0:
                            if sql_text[j:j+4].upper() == "CASE" and (j == 0 or not sql_text[j-1].isalnum()):
                                depth += 1
                                j += 4
                            elif sql_text[j:j+3].upper() == "END" and (j == 0 or not sql_text[j-1].isalnum()):
                                depth -= 1
                                j += 3
                            else:
                                j += 1
                        block = sql_text[i:j]
                        key = f"__CASE_BLOCK_{len(case_map)}__"
                        case_map[key] = block
                        out.append(key)
                        i = j
                    else:
                        out.append(sql_text[i])
                        i += 1
                return "".join(out)
            sql = shield_case_blocks(sql)

            # --- Aggregates (non-greedy) ---
            sql = re.sub(
                r"/\s*(SUM\([^)]*\))",
                r"/ NULLIF(\1,0)",
                sql,
                flags=re.IGNORECASE
            )

            # Division guard: wrap denominator with NULLIF(...,0)
            sql = re.sub(
                r"(/\s*)(COUNT\s*\([^)]*\)|SUM\s*\([^)]*\)|AVG\s*\([^)]*\)|\w+)",
                lambda m: f"/ NULLIF({m.group(2)},0)",
                sql,
                flags=re.IGNORECASE
            )

            sql = re.sub(
                r"\b(SUM|AVG|MIN|MAX|COUNT)\s*\(([^()]+?)\)",
                lambda m: f"{m.group(1).upper()}({safe_numeric_cast(m.group(2),COL_TYPES, False)})",
                sql, flags=re.IGNORECASE | re.DOTALL
            )

            # --- Preserve ID columns in inner SELECTs ---
            sql = ensure_id_columns(sql)

            # --- Shield IN (...) lists ---
            in_map = {}
            def extract_in_block(match):
                key = f"__IN_BLOCK_{len(in_map)}__"
                in_map[key] = match.group(0)
                return key
            sql = re.sub(r"\bIN\s*\([^)]*\)", extract_in_block, sql, flags=re.IGNORECASE | re.DOTALL)

            # --- Arithmetic ops ---
            sql = re.sub(
                r"([a-zA-Z0-9_\".]+)\s*\+\s*([a-zA-Z0-9_\".]+)",
                lambda m: f"{safe_numeric_cast(m.group(1), False)} + {safe_numeric_cast(m.group(2),COL_TYPES, False)}",
                sql, flags=re.IGNORECASE
            )

            sql = re.sub(
                r"\b(SUM|AVG|MIN|MAX|COUNT)\s*\(([^()]+?)\)",
                lambda m: f"{m.group(1).upper()}({safe_numeric_cast(m.group(2), COL_TYPES, False)})",
                sql, flags=re.IGNORECASE | re.DOTALL
            )

            sql = re.sub(
                r"([a-zA-Z0-9_\".]+)\s*/\s*([a-zA-Z0-9_\".]+)(?!\s*\()",   
                lambda m: f"{safe_numeric_cast(m.group(1), COL_TYPES)} / NULLIF({safe_numeric_cast(m.group(2), COL_TYPES)},0)",
                sql, flags=re.IGNORECASE
            )

            sql = re.sub(
                r"(/\s*)(COUNT\s*\([^)]*\))",
                lambda m: f"/ NULLIF({m.group(2)},0)",
                sql, flags=re.IGNORECASE
            )

            sql = re.sub(
                r"([a-zA-Z0-9_\".]+)\s*-\s*([a-zA-Z0-9_\".]+)",
                lambda m: f"{safe_numeric_cast(m.group(1), False)} - {safe_numeric_cast(m.group(2),COL_TYPES, False)}",
                sql, flags=re.IGNORECASE
            )
            sql = re.sub(
                r"([a-zA-Z0-9_\".]+)\s*\*\s*([a-zA-Z0-9_\".]+)",
                lambda m: f"{safe_numeric_cast(m.group(1), False)} * {safe_numeric_cast(m.group(2),COL_TYPES, False)}",
                sql, flags=re.IGNORECASE
            )
            sql = re.sub(
                r"([a-zA-Z0-9_\".]+)\s*/\s*([a-zA-Z0-9_\".]+)(?!\s*\()",   
                lambda m: f"{safe_numeric_cast(m.group(1), False)} / NULLIF({safe_numeric_cast(m.group(2),COL_TYPES, False)},0)",
                sql, flags=re.IGNORECASE
            )

            sql = re.sub(r"\)\s*\)\s*\(", ")) * (", sql)

            for key, val in in_map.items():
                sql = sql.replace(key, val)
            for key, val in case_map.items():
                sql = sql.replace(key, val)

            sql = unshield_where_blocks(sql, blocks)
            sql = re.sub(r'\s+', ' ', sql).strip()
            return sql

        # After recursion
        result = process_query_recursively(sql)

        # Restore literals
        for key, val in str_map.items():
            result = result.replace(key, val)

        # ----------------------------------------
        # Final post-processing patches
        # ----------------------------------------

        # 1. Fix invalid interval arithmetic on date strings
        result = re.sub(
            r"'(\d{4}-\d{2}-\d{2})'\s*([+-]\s*INTERVAL)",
            r"DATE '\1' \2",
            result,
            flags=re.IGNORECASE
        )

        # 2. Guard against division by zero
        result = re.sub(
            r"/\s*(\(?\s*(COUNT|SUM|AVG|MIN|MAX)\s*\([^()]*\)\s*\)?)",
            r"/ NULLIF(\1,0)",
            result,
            flags=re.IGNORECASE
        )

        # 3. Skip unknown column garbage
        result = re.sub(
            r"''\)\s*(IS\s+NOT\s+NULL|<>|=)\s*''",
            "1=1",
            result,
            flags=re.IGNORECASE
        )

        # 4. Fix invalid GROUP BY/ORDER BY constants
        result = re.sub(
            r"(GROUP BY[^;]*?)(,\s*20\d{2})(?=\s|$)",
            r"\1",
            result,
            flags=re.IGNORECASE
        )
        result = re.sub(
            r"(ORDER BY[^;]*?)(,\s*20\d{2})(?=\s|$)",
            r"\1",
            result,
            flags=re.IGNORECASE
        )

        if schema_catalog:
            result = validate_sql_against_catalog(result, schema_catalog)

        # Final compact whitespace
        result = re.sub(r'\s+', ' ', result).strip()


        logger.debug("✅ Final sanitized SQL: %s", result[:500])


        # Final compacting of whitespace
        # result = re.sub(r'\s+', ' ', result).strip()

        # logger.debug("✅ Final sanitized SQL: %s", result[:500])  # preview first 500 chars



        if not re.search(r'\bSELECT\b.*\bFROM\b', result, flags=re.IGNORECASE | re.DOTALL):
            return original_sql
        if result.count("(") != result.count(")"):
            return original_sql

        return result

    except Exception as e:
        print(f"⚠️ [SQL SANITIZER] Error: {e}")
        return original_sql

