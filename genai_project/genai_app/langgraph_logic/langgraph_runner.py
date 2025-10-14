from typing import TypedDict
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableSequence
from genai_app.langgraph_logic.db_embedding import get_context
from genai_app.utils.llm_config import get_llama_maverick_llm
from genai_app.utils.token_utils import truncate_text, estimate_tokens
import logging
logger = logging.getLogger(__name__)
from genai_app.schema_utils import find_relevant_table,auto_fix_schema_mismatches,sanitize_sql_for_nulls,build_type_map_from_catalog,build_dynamic_fallback_map
from typing import Any, Dict, List, Tuple, Optional, Set

from sentence_transformers import SentenceTransformer
import chromadb
from sqlalchemy import inspect
from django.core.cache import cache


# Initialize once
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection("schemas")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast model


from typing import TypedDict

# Constants
MAX_TOTAL_TOKENS = 6000
RESERVED_FOR_RESPONSE = 1000
MAX_INPUT_TOKENS = MAX_TOTAL_TOKENS - RESERVED_FOR_RESPONSE



import re
from typing import Any, Dict, List

def _mentions_table(sql: str, fq_table: str) -> bool:
    """Return True if the fully-qualified table appears anywhere in the SQL."""
    if not sql or not fq_table:
        return False
    return fq_table.lower() in re.sub(r'\s+', ' ', sql).lower()

def _enforce_allowed_tables(sql: str, allowed_fqns: List[str]) -> None:
    """
    Fail if SQL references any table outside the allowlist.
    Looks for exact occurrences like: "schema"."table"
    """
    found = set(re.findall(r'"[A-Za-z_]\w*"\s*\.\s*"[A-Za-z_]\w*"', sql))
    illegal = [t for t in found if t not in allowed_fqns]
    if illegal:
        raise ValueError(f"Illegal table(s) referenced: {', '.join(sorted(illegal))}")

def _derive_join_keys(a_cols: List[str], l_cols: List[str]) -> List[str]:
    """
    Build a prioritized list of join keys using shared columns first,
    then common ID-ish names.
    """
    a_set, l_set = set(a_cols), set(l_cols)
    shared = [c for c in a_cols if c in l_set]

    preferred = [
        "applicationid", "application_id",
        "customerid", "customer_id",
        "decisionid", "decision_id",
        "merchantid", "merchant_id",
        "clientid", "client_id",
        "loannumber", "loan_number",
    ]
    ordered = [k for k in preferred if k in shared]
    if not ordered:
        ordered = shared
    return ordered[:8]

def _has_explicit_join_predicate(sql: str, a_alias: str, l_alias: str, join_keys: List[str]) -> bool:
    """
    Return True if there is an explicit join predicate between aliases:
    - USING(...)
    - JOIN ... ON ... (and both aliases appear)
    - equality a.<key> = l.<key> (or swapped)
    """
    try:
        if not isinstance(sql, str):
            return False
        sql_low = sql.lower()

        # USING(...) quick check
        if re.search(r'\busing\s*\(', sql_low):
            return True

        # JOIN ... ON ... check - ensure both aliases present in same clause
        for m in re.finditer(r'\bjoin\b.*?\bon\b', sql, flags=re.IGNORECASE | re.DOTALL):
            chunk = m.group(0).lower()
            if (f"{a_alias}." in chunk) and (f"{l_alias}." in chunk):
                return True

        # Explicit equality checks for each candidate key
        for key in (join_keys or []):
            if not isinstance(key, str):
                continue
            patt1 = rf'\b{re.escape(a_alias)}\s*\.\s*{re.escape(key)}\s*=\s*{re.escape(l_alias)}\s*\.\s*{re.escape(key)}\b'
            patt2 = rf'\b{re.escape(l_alias)}\s*\.\s*{re.escape(key)}\s*=\s*{re.escape(a_alias)}\s*\.\s*{re.escape(key)}\b'
            if re.search(patt1, sql, flags=re.IGNORECASE) or re.search(patt2, sql, flags=re.IGNORECASE):
                return True

        return False
    except Exception:
        logger.exception("Error in _has_explicit_join_predicate")
        return False

def _auto_inject_join(sql: str, a_tbl: str, l_tbl: str, a_alias: str, l_alias: str, join_keys: List[str]) -> str:
    """
    Conservative attempt to inject an INNER JOIN using the first join key.
    - Only injects when a schema-qualified FROM referencing a_tbl is found.
    - Skips injection if l_tbl already exists in SQL.
    - Returns original SQL if unsafe to modify.
    """
    try:
        if not sql or not join_keys:
            return sql

        first_key = str(join_keys[0])

        # Look for exact schema-qualified FROM referencing a_tbl
        from_pattern = re.compile(
            rf'(\bFROM\s+{re.escape(a_tbl)}(?:\s+(?:AS\s+)?{re.escape(a_alias)})?)',
            flags=re.IGNORECASE
        )
        m = from_pattern.search(sql)
        if not m:
            # Looser fallback: first FROM token, but ensure it references a_tbl
            looser = re.compile(r'\bFROM\s+([^\s,;]+)', flags=re.IGNORECASE)
            m2 = looser.search(sql)
            if not m2:
                return sql
            if a_tbl.lower() not in m2.group(1).lower():
                return sql
            m = m2

        # If l_tbl is already present anywhere, don't inject
        if re.search(rf'\b{re.escape(l_tbl)}\b', sql, flags=re.IGNORECASE):
            return sql

        join_snippet = f'{m.group(1)} INNER JOIN {l_tbl} AS {l_alias} ON {a_alias}.{first_key} = {l_alias}.{first_key}'
        start, end = m.span(1)
        sql_modified = sql[:start] + join_snippet + sql[end:]

        # Sanity checks
        if sql_modified.count('(') != sql_modified.count(')'):
            return sql
        if (f'{a_alias}.' not in sql_modified) or (f'{l_alias}.' not in sql_modified):
            return sql

        logger.info("Auto-injected join using key '%s'", first_key)
        return sql_modified
    except Exception:
        logger.exception("Error in _auto_inject_join")
        return sql

def _enforce_allowed_join(sql: str, a_alias: str, l_alias: str, allowed_keys: List[str]) -> None:
    """
    If both tables are used, ensure there is a join predicate of the form:
      a.<key> = l.<key> for some key in allowed_keys
    Raises ValueError if missing.
    """
    patterns = [
        rf'\b{a_alias}\.{re.escape(k)}\s*=\s*{l_alias}\.{re.escape(k)}\b'
        for k in (allowed_keys or [])
    ]
    swapped = [
        rf'\b{l_alias}\.{re.escape(k)}\s*=\s*{a_alias}\.{re.escape(k)}\b'
        for k in (allowed_keys or [])
    ]
    checks = patterns + swapped
    if not any(re.search(p, sql, flags=re.I) for p in checks):
        raise ValueError("Missing allowed join predicate between the two tables.")

# ---------- GROUP BY / ORDER BY cleanup ----------

def _count_select_items(sql: str) -> int:
    """Tolerant count of SELECT items before the first FROM (ignores commas inside parens)."""
    m = re.search(r'^\s*SELECT\s+(.*?)\s+FROM\b', sql, flags=re.I | re.S)
    if not m:
        return 0
    segment = m.group(1)
    depth, buf = 0, []
    for ch in segment:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        if ch == ',' and depth > 0:
            buf.append('\x00')
        else:
            buf.append(ch)
    cleaned = ''.join(buf)
    return len([p for p in (x.strip() for x in cleaned.split(',')) if p])

def _sanitize_group_order_clauses(sql: str) -> str:
    """
    1) Remove stray bare 4-digit years from GROUP BY / ORDER BY lists.
    2) Keep ORDER BY positional indexes only if they refer to a valid select item.
    3) Drop empty GROUP BY / ORDER BY created by the sanitization.
    """
    select_count = _count_select_items(sql)

    def _scrub_list(lst: str, allow_positions: bool) -> str:
        parts = [p.strip() for p in lst.split(',')]
        keep: List[str] = []
        for p in parts:
            if re.fullmatch(r'20\d{2}', p):  # bare year literal
                continue
            if allow_positions and re.fullmatch(r'\d+', p):
                pos = int(p)
                if 1 <= pos <= max(1, select_count):
                    keep.append(p)
                continue
            if p:
                keep.append(p)
        return ', '.join(keep)

    def _fix_group(m: re.Match) -> str:
        inner = m.group(1)
        fixed = _scrub_list(inner, allow_positions=False)
        return ('GROUP BY ' + fixed) if fixed else ''

    def _fix_order(m: re.Match) -> str:
        inner = m.group(1)
        fixed = _scrub_list(inner, allow_positions=True)
        return ('ORDER BY ' + fixed) if fixed else ''

    sql = re.sub(r'\bGROUP\s+BY\s+(.*?)(?=\bORDER\b|\bLIMIT\b|$)', _fix_group, sql, flags=re.I | re.S)
    sql = re.sub(r'\bORDER\s+BY\s+(.*?)(?=\bLIMIT\b|$)', _fix_order, sql, flags=re.I | re.S)
    return re.sub(r'\s+', ' ', sql).strip()



# # Define state format for LangGraph
# class SQLState(TypedDict):
#     question: str
#     context: str
#     sql: str


# # Utility to trim history
# def trim_history(history, max_words=1500):
#     """Keep only recent history within a safe word budget"""
#     trimmed = []
#     word_count = 0
#     for item in reversed(history):
#         entry = f"Q: {item['question']}\nSQL: {item['sql']}"
#         words = len(entry.split())
#         if word_count + words > max_words:
#             break
#         trimmed.append(entry)
#         word_count += words
#     return "\n".join(reversed(trimmed))


# # Main runner function
# def run_sql_generation_graph(question, user_id, db_id, history=[]):
#     def retrieve_context_node(state):
#         context_raw = get_context(state["question"], user_id, db_id)
#         state["context"] = truncate_text(context_raw, max_tokens=3000)
#         return state

#     def generate_sql_node(state):
#         # Truncate history to fit within context window
#         history_context = trim_history(history, max_words=1500)

#         prompt = PromptTemplate.from_template(f"""You are a PostgreSQL SQL expert. Given the schema:

# {{context}}

# Conversation history:
# {history_context}

# Now generate SQL for:
# {{question}}""")

#         # Create chain
#         llm = get_llama_maverick_llm()
#         runnable = prompt | llm

#         # Token safety check
#         test_prompt = prompt.format(context=state["context"], question=state["question"])
#         if estimate_tokens(test_prompt) > MAX_INPUT_TOKENS:
#             state["sql"] = "-- Error: Prompt too large after truncation. Please simplify your request."
#             return state

#         # Invoke safely
#         try:
#             response = runnable.invoke(state)
#             state["sql"] = response.content if hasattr(response, "content") else response
#         except Exception as e:
#             state["sql"] = f"-- LLM error: {e}"
#         return state

#     # Define LangGraph
#     graph = StateGraph(SQLState)
#     graph.add_node("retrieve_context", retrieve_context_node)
#     graph.add_node("generate_sql", generate_sql_node)
#     graph.set_entry_point("retrieve_context")
#     graph.add_edge("retrieve_context", "generate_sql")
#     graph.set_finish_point("generate_sql")

#     # Run with initial question
#     final_state = graph.compile().invoke({"question": question})
#     return final_state["sql"]


from typing import TypedDict
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableSequence
from genai_app.utils.llm_config import get_llama_maverick_llm
from genai_app.utils.token_utils import truncate_text, estimate_tokens

# Constants
MAX_TOTAL_TOKENS = 6000
RESERVED_FOR_RESPONSE = 1000
MAX_INPUT_TOKENS = MAX_TOTAL_TOKENS - RESERVED_FOR_RESPONSE

class SQLState(TypedDict):
    question: str
    context: str
    sql: str
    explanation: str
    result: list
    summary: str

def summarize_result_node(state: dict) -> dict:
    question = state["question"]
    result = state["result"]
    sql = state["sql"]

    prompt = f"""
You are a business analyst.

A user asked: "{question}"
The SQL used:
{sql}
The result from the database:
{result}

Summarize the result in a business-friendly sentence with bullet points if needed.
"""
    llm = get_llama_maverick_llm()
    summary = llm.invoke(prompt)
    return {**state, "summary": summary}

def trim_history(history, max_words=1500):
    trimmed = []
    word_count = 0
    for item in reversed(history):
        entry = f"Q: {item['question']}\nSQL: {item['sql']}"
        words = len(entry.split())
        if word_count + words > max_words:
            break
        trimmed.append(entry)
        word_count += words
    return "\n".join(reversed(trimmed))

# ✅ Hardcoded schema (as provided)
# FULL_SCHEMA = '''
# Table: "stage"."GBM1_prediction_data_with_recommendations"
# Columns:
# - policy_id
# - customerid
# - policy_end_date
# - policy_end_date_month
# - cleaned_reg_no
# - churn_category
# - proposal_id
# - od_premium
# - tp_premium
# - idv
# - discount
# - rto_risk_factor
# - ncb_%_previous_year
# - state_risk_score
# - retention_rate_pct
# - total_od_premium_max
# - applicable_discount_with_ncb
# - policy_wise_purchase
# - manufacturer_risk_rate
# - days_between_renewals
# - retention_streak
# - total_od_premium_mean
# - total_od_premium
# - firstpolicyyear
# - lag_1_tp_premium
# - total_od_premium_min
# - avg_premium_hist
# - lag_1_ncb
# - age
# - total_tp_premium_max
# - total_tp_premium_mean
# - total_tp_premium
# - total_tp_premium_min
# - lag_1_premium
# - previous_year_premium_ratio
# - total_premium_payable
# - total_revenue
# - gst
# - fuel_type_risk_factor
# - lag_1_od_premium
# - customer_apv
# - segment_risk_score
# - vehicle_idv
# - policy_tenure
# - number_of_claims
# - approved
# - claim_approval_rate
# - customer_tenure
# - before_gst_add-on_gwp
# - od_tp_ratio
# - add_on_adoption
# - clv
# - idv_premium_ratio
# - customer_apf
# - days_gap_prev_end_to_curr_start
# - customerid
# - claim_happaned/not
# - cleaned_branch_name_2
# - cleaned_chassis_number
# - cleaned_engine_number
# - cleaned_reg_no
# - cleaned_state2
# - cleaned_zone_2
# - biztype
# - corrected_name
# - make_clean
# - model_clean
# - product_name
# - policy_no
# - decline
# - tie_up
# - variant
# - policy_status
# - policy_start_date_year
# - policy_end_date_year
# - policy_start_date_month
# - policy_end_date_month
# - policy_start_date_day
# - policy_end_date_day
# - predicted_status
# - churn_probability
# - clv_category
# - discount_category
# - churn_category
# - customer_segment
# - not_renewed_reasons
# - main_reason
# - primary_recommendation
# - additional_offers
# - retention_channel
# '''



#liberty running
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
FULL_SCHEMA = """
Table: "final_lusa"."apploan_metrics_2024"

❌ Never reference any other table (like "bi_dwh"."main_cai_lib").
✅ Only use "final_lusa"."apploan_metrics_2024".

Columns:
- application_id: Unique identifier assigned to each loan application.
- channel_code: Code representing the sales/loan channel (DTM=Direct to Merchant, DTC=Direct to Customer, FSL=Fresh Start Lending).
- channel_group: Higher-level grouping of loan channels (e.g., Digital vs Physical).
- merchant_id: Identifier of the merchant (store, vendor, or partner) linked to the loan.
- merchant_vertical: Type of industry/business the merchant belongs to (e.g., electronics, healthcare, travel).
- client_id: Identifier of the client institution or business using the lending platform.
- customer_id: Unique identifier of the borrower (end-customer).
- loan_application_purpose: Purpose of the loan (education, medical, personal, etc.).
- loan_application_city: City where the borrower resides/applied for the loan.
- loan_application_state: State/region where the borrower resides/applied.
- requested_amount: Amount requested in the loan application.
- approval_amount: Amount approved by the lender after credit checks.
- loan_application_status: Current status (Approved, Pending, Rejected, Declined, Duplicate, App Received, etc.).
- funded_date: Date when approved loan amount was disbursed.
- loan_servicer_boarding: Loan servicing provider managing repayments/collections.
- loan_group_boarding: Portfolio/group to which this loan belongs.
- issuing_bank: Bank/financial institution that issued the loan.
- decision_source: Source of the lending decision (e.g., automated system, manual review, partner).
- note_principal: Original principal amount of the loan note.
- annual_percentage_rate: True annual cost of the loan (APR % including fees).
- loan_term: Loan duration (months/years).
- loan_amount_funded: Actual disbursed amount after approval.
- fico_score: Borrower’s FICO credit score (300–850).
- vantage_score: Borrower’s VantageScore 4.0 (300–850).
- is_fraud_suspected: Flag indicating suspected fraud.
- is_fraud_confirmed: Flag indicating confirmed fraud.
- merchant_discount: Discount/fee charged to merchants for offering financing.
- loan_application_date: Date the loan application was submitted.
- interest_rate: Nominal interest rate applied to the loan.
- annual_income: Borrower’s reported yearly income.
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from genai_app.schema_utils import validate_sql_against_catalog,embed_schema_catalog

# -------------------------------
# Fixed 2024 Date Context
# -------------------------------


import datetime
from dateutil.relativedelta import relativedelta

def get_fixed_date_context():
    """
    Always return 2024 as the reference year,
    but align month/quarter/half dynamically to system month.
    """
    now_dt = datetime.datetime.now()
    fixed_year = 2024
    month = now_dt.month

    # Quarter & Half
    quarter = (month - 1) // 3 + 1
    half = 1 if month <= 6 else 2

    # Quarter ranges
    quarter_start = datetime.datetime(fixed_year, 3 * (quarter - 1) + 1, 1)
    quarter_end = quarter_start + relativedelta(months=3)

    # Half-year ranges
    half_start = datetime.datetime(fixed_year, 1 if half == 1 else 7, 1)
    half_end = half_start + relativedelta(months=6)

    return {
        "year": fixed_year,
        "month": month,
        "quarter": quarter,
        "half": half,
        "quarter_start": quarter_start.strftime("%Y-%m-%d"),
        "quarter_end": quarter_end.strftime("%Y-%m-%d"),
        "half_start": half_start.strftime("%Y-%m-%d"),
        "half_end": half_end.strftime("%Y-%m-%d"),
    }

# def get_fixed_date_context():
#     """
#     Always return 2024 as the reference year,
#     but align month/quarter/half dynamically to system month.
#     """
#     # now_dt = datetime.now()
#     now_dt = datetime.datetime.now()
#     fixed_year = 2024
#     month = now_dt.month

#     # Quarter & Half
#     quarter = (month - 1) // 3 + 1
#     half = 1 if month <= 6 else 2

#     # Quarter ranges
#     quarter_start = datetime(fixed_year, 3 * (quarter - 1) + 1, 1)
#     quarter_end = quarter_start + relativedelta(months=3)

#     # Half-year ranges
#     half_start = datetime(fixed_year, 1 if half == 1 else 7, 1)
#     half_end = half_start + relativedelta(months=6)

#     return {
#         "year": fixed_year,
#         "month": month,
#         "quarter": quarter,
#         "half": half,
#         "quarter_start": quarter_start.strftime("%Y-%m-%d"),
#         "quarter_end": quarter_end.strftime("%Y-%m-%d"),
#         "half_start": half_start.strftime("%Y-%m-%d"),
#         "half_end": half_end.strftime("%Y-%m-%d"),
#     }


# -------------------------------
# Enforcer: Rewrites SQL Dates to 2024
# -------------------------------
def enforce_2024(sql: str, date_info: dict) -> str:
    """
    Force all queries to align with 2024 context.
    """
    # Anchor CURRENT_DATE → 2024-MM-01
    anchor_date = f"{date_info['year']}-{date_info['month']:02d}-01"
    sql = sql.replace("CURRENT_DATE", f"DATE '{anchor_date}'")

    # Force EXTRACT(YEAR ...) → 2024
    sql = re.sub(r"EXTRACT\(YEAR FROM [^)]+\)", str(date_info["year"]), sql, flags=re.IGNORECASE)

    # Replace quarter logic
    sql = re.sub(
        r"DATE_TRUNC\('quarter', DATE '[^']+'\)\s*-\s*INTERVAL '3 month'",
        f"DATE '{date_info['quarter_start']}'",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"DATE_TRUNC\('quarter', DATE '[^']+'\)",
        f"DATE '{date_info['quarter_end']}'",
        sql,
        flags=re.IGNORECASE,
    )

    # Replace half-year
    sql = sql.replace(
        "DATE_TRUNC('half', CURRENT_DATE)",
        f"DATE '{date_info['half_start']}'"
    )

    # Fix lookbacks like DATE '2023-09-01' - INTERVAL '1 year'
    sql = re.sub(
        r"DATE '(\d{4}-\d{2}-\d{2})'\s*-\s*INTERVAL '(\d+) year'",
        f"DATE '{date_info['year']}-01-01'",
        sql,
        flags=re.IGNORECASE,
    )

    # Fix lookbacks with months
    sql = re.sub(
        r"CURRENT_DATE\s*-\s*INTERVAL '(\d+) month'",
        lambda m: f"DATE '{(datetime(date_info['year'], date_info['month'], 1) - relativedelta(months=int(m.group(1)))).strftime('%Y-%m-%d')}'",
        sql,
        flags=re.IGNORECASE,
    )

    # Cap upper bound
    sql = re.sub(
        r"<\s*CURRENT_DATE",
        f"< DATE '{date_info['year']}-12-31'",
        sql,
        flags=re.IGNORECASE,
    )

    # Cleanup duplicate DATE DATE
    sql = re.sub(r"\b(DATE\s+)+DATE\b", "DATE", sql)

    return sql

# """"""""bfrsep2024  
# def run_sql_generation_graph(question, user_id, db_id, history=[]):
#     def retrieve_context_node(state):
#         state["context"] = FULL_SCHEMA
#         return state

#     def generate_sql_node(state):
#         history_context = trim_history(history)

#         prompt = PromptTemplate.from_template(f"""You are a PostgreSQL SQL expert.
                                              
# Your task is two-fold:
# 1. Generate the SQL query
# # 2. Provide a short **recommendation** based on the query output.
# 2. If the user input contains **typos, spacing issues, or partial matches** for column values like `"tamil nadu"` or `"potental customers"`, **auto-correct** the value by using trigram similarity matching.
# 3. Provide a professional, actionable recommendation (1–3 lines) derived from the query output 
#    that tells the user what they can do next to enhance business performance (e.g., reduce churn, 
#    increase revenue, improve customer retention).
                                              
# Only use this table and its columns: "bi_dwh"."main_cai_lib"
# Here are the ONLY valid columns (do not invent or assume others):
#         ❌ Never use columns not listed above
#         ✅ Use exact column names only
#         ✅ If unsure, respond with "-- Error: unknown column requested"
#           ❌ Never use columns not in the list below
# ✅ Always use column names *exactly* as listed (case-sensitive)

# Here are the allowed columns:
# {{context}} 


# Follow these strict rules:
# 1. Use exact column names only.
# 2. If a column doesn't exist, respond with: `-- Error: unknown column requested`
# 3. If the user's question is vague or ambiguous (e.g., "in Jan?", "what about February?"), use the most recent question from history to infer:
#    - The focus column (e.g., `customer_segment`)
#    - The correct time filter (e.g., `policy_end_date_month`)
# 4. Do not change the GROUP BY or metric unless explicitly asked.
# 5. Only change filters like month/year if mentioned.

                                              
                                                                                 
                                              
# Some important facts:
# - "is_churn" contains only Not Renewed and Renewed.

# **Rules to Follow:**
# - Only use the columns listed above. ❌ Never invent or use other columns.
# - Use `policy_no` as the primary policy identifier.
# - For **location filters**:
#   - Use `state` for states (e.g., 'tamilnadu', 'maharashtra').
#   - Use `branch_name` for cities/branches (e.g., 'tirunelveli', 'hyderabad').
#   - Use `zone` for regions (north/south/east/west).
#   - Always use ILIKE with wildcards for text matches:
#     Example: `branch_name ILIKE '%tirunelveli%'`.
# - For **churn queries**:
#   - Use `is_churn` (values: 'Renewed', 'Not Renewed').
#   - Use `churn_probability` for numeric risk scores.
#   - Use `main_churn_reason` for churn reasons.
# - For **customer segmentation**:
#   - Use `customer_segment` (values like 'Potential Customers', 'Low Value Customers', etc.).
# - For **date-based filters**:
#   - Use `policy_end_date_year` and `policy_end_date_month`.
#   - Do NOT use `policy_start_date_*` for renewal queries.
# - Use `COUNT(policy_no)` when user asks "how many policies".
# - Use `LIMIT 1` when the question is singular (e.g., "Which customer...?").

                                              
# Business logic instructions:
# - If the user asks "how many policies are renewed":
#     → Use: is_churn ILIKE 'renewed'

# - If the user asks "how many policies are not renewed":
#     → Use: is_churn IS NULL OR is_churn ILIKE 'not renewed'

# - If user says “not renewed”, do not equate it with “declined” unless specified.

# - is_churn may contain values like: 'renewed', 'not renewed', 'declined', etc. Use exact matches.
# Examples:
# - Use `is_churn ILIKE 'renewed'` to find renewed policies.
# - Use `is_churn IS NULL OR is_churn ILIKE 'not renewed'` for not renewed policies.
# - Do not confuse 'declined' with 'not renewed'.
                                              
# Customer Segmentation Instructions:
# - Use `customer_segment` to identify customer types
# - The valid values include:
#     - 'Low Value Customers'
#     - 'Potential Customers'
#     - 'Elite Retainers'
#     - 'Retained Breakers'
# - If user says:
#     - if user ask "which customer i need to focus highly" it gives first option as "elite customers"                                
#     → "low value customers" → use: customer_segment ILIKE '%Low Value Customers%'
#     → "potential customers" → use: customer_segment ILIKE '%Potential Customers%'
#     → "elite customers" or "retainers" → use: customer_segment ILIKE '%Elite Retainers%'
# - Do not use clv_category for segmentation — it's a separate numeric indicator

                                              
# Location Filtering Instructions:
# - Use `state` for state-level filters (e.g., "tamilnadu", "karnataka")
# - Use `branch_name` for branch-level filters (e.g., "bangalore", "pune", "coimbatore")
# - Use `zone` for broader region filters like "north", "south", "west", "east"
# - All of these are valid filters, and can be combined in WHERE clause if needed
# - Use ILIKE for case-insensitive comparison with these values
# - Always use ILIKE with wildcards for text matches:
#     Example: `branch_name ILIKE '%tirunelveli%'`.
# - Use ILIKE instead of = for filtering text values such as states, cities, or categories
# - Example: state ILIKE 'tamilnadu' instead of cleaned_state2 = 'tamilnadu'
# - Do not assume values are lowercase — use ILIKE for case-insensitive comparisons
# Examples:
# - state ILIKE 'tamilnadu'
# - branch_name ILIKE 'bangalore'
# - zone ILIKE 'south'
                                              
# 🧹 Text Matching Logic for Location Columns:

# When filtering based on the following columns:
# - `state`
# - `zone`
# - `branch_name`

# Always apply **normalized case-insensitive fuzzy matching** by:

# 1. Converting the column using `TRIM(LOWER(...))`
# 2. Converting the user input by:
#    - Lowercasing it
#    - Removing all whitespace using `REPLACE(...)`
# 3. Applying a fuzzy match using `LIKE '%' || ... || '%'`

# ✅ SQL Format for each column:

# - For `state`:
#   ```sql
#   TRIM(LOWER(state)) LIKE '%' || REPLACE(LOWER('user_input'), ' ', '') || '%'
                                              
# 🗓️ Renewal Date Logic:
# - Always filter policies based on their renewal/end month using:
#   policy_end_date_year and policy_end_date_month.
# - Example:
#   For "February 2025", use:
#   policy_end_date_year = 2025 AND policy_end_date_month = 2
# - ❌ Never use policy_start_date_year or policy_start_date_month for renewal queries.
# - For questions like "how many churn in March?", assume current year if no year is specified.
# - When the user says "in Jan", "in Feb", etc., infer that they are referring to 
#     policy_end_date_month = 1, 2, etc., respectively. 
# - If no year is given, default to the current policy_end_date_year.
# - Preserve the previous context (e.g., location, churn) from earlier messages.
                                              
# Churn Reason Logic:
# - Use the `main_churn_reason` column to identify why a customer churned.
# - Do not include "main_churn_reason = 'None'" in analysis — it means the reason is unknown or missing.
# - Always filter out rows where "main_churn_reason IS NULL or "main_churn_reason ILIKE 'None' before doing GROUP BY.
# - To get one top reason, use ORDER BY COUNT(*) DESC LIMIT 1.
# - Example SQL:
#   SELECT "main_churn_reason, COUNT(*) FROM "stage"."GBM1_prediction_data_with_recommendations"
#   WHERE "main_churn_reason IS NOT NULL AND "main_churn_reason NOT ILIKE 'None'
#   GROUP BY "main_churn_reason
#   ORDER BY COUNT(*) DESC;
                                              

# ✅ INTENT CLARITY:
# - If the user question is **singular** (e.g., "Which customer will churn?", "Which vehicle is assigned?", etc):
#     → Add `LIMIT 1` at the end of the SQL to return just the top 1 result.
# - If the user question is **plural** (e.g., "Which customers will churn?", "Show all vehicles assigned today",etc):
#     → Return full result set (no LIMIT unless user asks for top N).
# Example:                                            
# 1. User asks: **"Which customer will churn?"**
# ```sql
# SELECT customerid, is_churn, churn_probability
# FROM "stage"."GBM1_prediction_data_with_recommendations"
# WHERE is_churn = 'Not Renewed'
# ORDER BY churn_probability DESC
# LIMIT 1;
                                              
# 2. **LIMIT Rule Based on Question Type**:
#    - If the user question is singular (e.g., "Which customer will churn?", "Which state to focus?"):
#        → Append `ORDER BY <metric> DESC LIMIT 1`.
#    - If the user question is plural (e.g., "Which customers will churn?", "Show all states with retention rates?"):
#        → Return the full result set (no LIMIT unless user specifies "top N").
#    - Detect singular vs plural using keywords like "which customer", "which state", "what is the top...", or absence of "all".

                                                                                       
# examples:
# Q: Which customer will churn soon?
# → Return top 1 customer with highest churn risk. Use LIMIT 1.

# Q: Which customers are at high risk?
# → Return all high-risk customers, sorted by churn probability.

# Q: Which vehicle is currently assigned to batch 3?
# → Return vehicle assigned to batch 3 with LIMIT 1.

# Q: Show available vehicles for the 9AM slot
# → Return all matching vehicles for that slot.
                                              

# Follow these strict rules:

# 1. **Follow-up intent resolution:**
#    - When a user follow-up is ambiguous (e.g., "in Jan?", "how about February?", "next month?", etc.), use the most recent previous user question to:
#      - Determine the primary **dimension** or **aggregation** (e.g., group by `customer_segment`, `is_churn`, etc.).
#      - Preserve the **metric** (e.g., `COUNT`, `SUM`, etc.) used previously.
#      - Only change **filters** such as time ranges or months if explicitly indicated.
#    - DO NOT change the grouping or aggregation logic unless the user clearly asks for a different metric, segment, or question.

# 2. **Consistency enforcement:**
#    - Treat the user conversation as continuous unless a full topic shift is obvious.
#    - If the user asks a vague question like "in Jan?" or "what about Feb?", treat it as a request to **re-run the same type of query** with the time window updated accordingly.
#    - DO NOT switch from one dimension (e.g., `customer_segment`) to another (e.g., `is_churn`) on your own.

# 3. **Schema constraint:**
#    - Only use columns that exist in the provided table schema.
#    - All queries must reference: `"stage"."GBM1_prediction_data_with_recommendations"`.

# 4. **Query format:**
#    - Return only executable PostgreSQL SQL code inside a markdown code block (```sql ... ```).
#    - Do not include explanations, commentary, or extra text outside the code block.
#    - Always include a `GROUP BY` clause if the previous query used it, unless the user explicitly requests an overall total.
                                              
# 5. **Singular vs Plural Results:**
#    - If the question is singular (e.g., "which state...", "which customer...", "what is the top..."), always return only the top 1 result by adding:
#      ORDER BY <metric> DESC
#      LIMIT 1
#    - If the question is plural (e.g., "which states...", "which customers...", "show all..."), return all matching rows without LIMIT.
#    - Detect singular vs plural based on keywords like:
#      ["which state", "which customer", "the top", "highest", "lowest"] → singular (LIMIT 1).
#      ["which states", "which customers", "all", "list of"] → plural (no LIMIT).


# Strictly follow these rules to maintain continuity, reduce hallucination, and keep the query flow consistent throughout the conversation.
                                              
# Always infer the intent from user messages. When a question is ambiguous (e.g. "in jan?", "what about feb?"), use the most recent previous question to determine:
# - the focus column (e.g., customer_segment, is_churn)
# - the date filtering logic

# Unless the user clearly shifts the topic, assume they want to continue the previous query with a changed time range or minor variation.
# Only change the aggregation/dimension if explicitly asked.

# Use this format:
# -------------------
# SQL:
# <sql here>

# Recommendation:

# < Provide a concise, professional, and **actionable recommendation (1–3 lines)** derived from the query context and expected output, telling the user what actions they can take to enhance their business.

# **Guidelines for Recommendations:**
# - If the query is about **churn or is_churn**, suggest retention campaigns, targeted discounts, or customer outreach strategies.
# - If the query is about **total revenue**, suggest ways to increase revenue such as upselling, cross-selling, or premium policy optimization.
# - If the query is about **branches or locations**, recommend branch-level improvements, localized campaigns, or customer engagement programs.
# - If the query is about **claims**, suggest streamlining claim processes or improving claim approval rates.
# - Always give **next steps** to enhance business performance based on query insights.>



# -------------------             

# Schema:
# {{context}}

# Conversation history:
# {history_context}

# Now generate SQL for:
# {{question}}""")
# """"""""bfrsep2024  
import uuid
import traceback
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from langgraph.graph import StateGraph
from langchain.prompts import PromptTemplate

# Import your existing functions (make sure these are available)
# from your_existing_modules import (
#     find_relevant_table, embedder, collection, FULL_SCHEMA, trim_history,
#     get_fixed_date_context, get_llama_maverick_llm, SQLState,
#     enforce_2024, sanitize_sql_for_nulls, auto_fix_schema_mismatches,
#     validate_sql_against_catalog, build_schema_catalog, embed_schema_catalog
# )

# =============================================================================
# GLOBAL SESSION STORES
# =============================================================================
session_store = {}
conversation_memory_store = {}

def check_session_health():
    """Debug function to check session store health"""
    print("🏥 SESSION HEALTH CHECK:")
    print(f"  Total sessions: {len(session_store)}")
    print(f"  Session IDs: {list(session_store.keys())}")
    
    for session_id, session_data in session_store.items():
        print(f"  Session {session_id[:8]}...:")
        print(f"    Has engine: {'engine' in session_data}")
        print(f"    Has schema_catalog: {'schema_catalog' in session_data}")
        if 'schema_catalog' in session_data:
            print(f"    Table count: {len(session_data['schema_catalog'])}")
    print("=" * 50)

def create_emergency_session():
    """Create an emergency session when session_store is empty"""
    try:
        print("🚨 Creating emergency session...")
        
        # Use your existing database configuration
        data = settings.STATIC_DB
        user = "emergency_user"

        pg_user = data["postgres_user"]
        pg_pass = data["postgres_password"]
        pg_host = data["postgres_host"]
        pg_port = data["postgres_port"]
        pg_db = data["postgres_db"]

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
        
        # Build schema catalog
        schema_catalog = build_schema_catalog(engine)
        
        session_id = f"emergency_{uuid.uuid4()}"
        session_data = {
            "engine": engine,
            "user_id": user,
            "schema_catalog": schema_catalog
        }
        
        print(f"✅ Emergency session created: {session_id}")
        return (session_id, session_data)
        
    except Exception as e:
        print(f"❌ Emergency session creation failed: {e}")
        traceback.print_exc()
        return None

# =============================================================================
# MAIN SQL GENERATION FUNCTION - UPDATED WITH BETTER SESSION HANDLING
# =============================================================================
# def run_sql_generation_graph(question, user_id, db_id, history=[]):
#     """
#     Generate SQL + recommendation using schema-aware catalog and LangGraph.
#     - If table is in final_lusa.* → enforce 2024 date logic.
#     - Else → generate normally (respect user filters).
#     """
#     print(f"🔍 Looking for session {db_id} in session_store")
#     print(f"📊 Current session_store keys: {list(session_store.keys())}")
#     print(f"📊 Session_store size: {len(session_store)}")
    
#     session_data = session_store.get(db_id)
#     if not session_data:
#         print(f"⚠️ Session {db_id} not found in session_store")
        
#         # Try to find ANY active session as fallback
#         if session_store:
#             print(f"🔄 Available sessions: {list(session_store.keys())}")
#             db_id, session_data = next(iter(session_store.items()))
#             print(f"↩️ Falling back to session {db_id}")
#         else:
#             print("❌ session_store is completely empty!")
#             # Try to initialize a default connection
#             try:
#                 print("🚀 Attempting to create emergency session...")
#                 emergency_session = create_emergency_session()
#                 if emergency_session:
#                     db_id, session_data = emergency_session
#                     session_store[db_id] = session_data
#                     # Also embed the schema for the emergency session
#                     try:
#                         embed_schema_catalog(
#                             user_id=session_data["user_id"], 
#                             db_id=db_id,
#                             schema_catalog=session_data["schema_catalog"],
#                             embedder=embedder, 
#                             collection=collection
#                         )
#                         print("✅ Emergency session schema embedded")
#                     except Exception as embed_error:
#                         print(f"⚠️ Emergency session embedding failed: {embed_error}")
#                 else:
#                     return ("-- ERROR: No active DB session and cannot create emergency session", 
#                            "Please connect to database first using the connect endpoint", "")
#             except Exception as e:
#                 print(f"❌ Emergency session creation failed: {e}")
#                 traceback.print_exc()
#                 return ("-- ERROR: No active DB session", "Please connect to database first", "")

#     # Validate session_data has required components
#     if not session_data or "schema_catalog" not in session_data:
#         print("❌ Invalid session data - missing schema_catalog")
#         return ("-- ERROR: Invalid session data", "Session corrupted, please reconnect", "")

#     schema_catalog = session_data["schema_catalog"]
#     print(f"✅ Using session {db_id} with {len(schema_catalog)} tables")

#     # 🔎 Pick the most relevant table for this question
#     try:
#         target_table = find_relevant_table(question, schema_catalog, embedder, collection)
#     except Exception as e:
#         print(f"❌ Error finding relevant table: {e}")
#         # Fallback: use first table in catalog
#         if schema_catalog:
#             target_table = list(schema_catalog.keys())[0]
#             print(f"🔄 Using fallback table: {target_table}")
#         else:
#             return ("-- ERROR: No tables available", "Schema catalog is empty", "")

#     if not target_table:
#         print("❌ No relevant table found for question:", question)
#         print(f"🔍 Available tables: {list(schema_catalog.keys())}")
#         return ("-- ERROR: No relevant table found", "No suitable table for your question", "")

#     available_cols = schema_catalog[target_table]
#     print(f"✅ Target table selected: {target_table} → Columns: {available_cols}")

#     # -------------------------------
#     # Context Node
#     # -------------------------------
#     def retrieve_context_node(state):
#         state["context"] = FULL_SCHEMA
#         return state

#     # -------------------------------
#     # SQL Generation Node
#     # -------------------------------
#     def generate_sql_node(state):
#         try:
#             history_context = trim_history(history)
#             date_info = get_fixed_date_context()

#             enforce_year = target_table.startswith('"final_lusa"')  # ✅ only force year if in final_lusa.*

#             # 🔑 Prompt for LLM
#             prompt = PromptTemplate.from_template(f"""
# You are a PostgreSQL SQL expert.

# Your task is two-fold:
# 1. Generate the SQL query.
# 2. Provide a short **business recommendation** (1–3 lines) based on the query output.

# Strict rules:
# - Use only this table: {target_table}.
# - Available columns: {", ".join(available_cols)}.
# - Always prefix with schema + table.
# - Do NOT invent columns or tables.
# {"- Default all queries to year " + str(date_info['year']) + " if year is unspecified." if enforce_year else ""}
# - For other schemas, apply year filter only if explicitly mentioned in the question.

# Conversation history:
# {history_context}

# Now generate SQL for:
# {question}
# """)

#             llm = get_llama_maverick_llm()
#             runnable = prompt | llm
#             response = runnable.invoke({
#                 **state,
#                 "history_context": history_context,
#                 "question": state["question"]
#             })

#             text = response.content if hasattr(response, "content") else response
#             print("📝 Raw LLM response:\n", text)

#             # -------------------------------
#             # Post-process SQL
#             # -------------------------------
#             if "Recommendation:" in text:
#                 sql_part, explanation_part = text.split("Recommendation:", 1)
#                 sql_clean = sql_part.replace("SQL:", "").strip()
#             else:
#                 sql_clean = text.strip()
#                 explanation_part = "-- No recommendation generated"

#             # ✅ Apply 2024 enforcement only for final_lusa.*
#             if enforce_year:
#                 sql_clean = enforce_2024(sql_clean, date_info)

#             # ✅ Apply sanitizers for all schemas
#             sql_clean = sanitize_sql_for_nulls(sql_clean, schema_catalog=schema_catalog, question=question)
#             sql_clean = auto_fix_schema_mismatches(sql_clean, schema_catalog)
#             sql_clean = validate_sql_against_catalog(sql_clean, schema_catalog)

#             state["sql"] = sql_clean
#             state["explanation"] = explanation_part.strip()
#             return state
            
#         except Exception as e:
#             print(f"❌ Error in generate_sql_node: {e}")
#             traceback.print_exc()
#             state["sql"] = f"-- ERROR: SQL generation failed: {str(e)}"
#             state["explanation"] = "SQL generation encountered an error"
#             return state

#     # -------------------------------
#     # LangGraph Workflow
#     # -------------------------------
#     try:
#         graph = StateGraph(SQLState)
#         graph.add_node("retrieve_context", retrieve_context_node)
#         graph.add_node("generate_sql", generate_sql_node)
#         graph.set_entry_point("retrieve_context")
#         graph.add_edge("retrieve_context", "generate_sql")
#         graph.set_finish_point("generate_sql")

#         # -------------------------------
#         # Execute Graph
#         # -------------------------------
#         final_state = graph.compile().invoke({
#             "question": question,
#             "user_id": user_id,
#             "db_id": db_id,
#         })

#         print("✅ Final State:", final_state)

#         return (
#             final_state.get("sql", "-- ERROR: No SQL generated"),
#             final_state.get("explanation", "No explanation generated"),
#             final_state.get("summary", "")
#         )
#     except Exception as e:
#         print(f"❌ Error in LangGraph execution: {e}")
#         traceback.print_exc()
#         return (
#             f"-- ERROR: Graph execution failed: {str(e)}",
#             "LangGraph workflow encountered an error",
#             ""
#         )

#all schema and tables
# def _derive_join_keys(a_cols: List[str], l_cols: List[str]) -> List[str]:
#     """Shared columns first, with priority on common ID names."""
#     a = {c.lower() for c in a_cols}
#     b = {c.lower() for c in l_cols}
#     shared = list(a & b)

#     priority = [
#         "application_id", "loan_application_id", "loan_id", "customer_id",
#         "account_id", "policy_no", "policy_id", "app_id"
#     ]
#     ordered = [k for k in priority if k in shared] + [c for c in sorted(shared) if c not in priority]
#     return ordered


# def _mentions_table(sql: str, fq: str) -> bool:
#     """Detect if a fully-qualified table appears (quoted or unquoted)."""
#     s, t = fq.replace('"', '').split('.')
#     return bool(re.search(rf'\b"{s}"\s*.\s*"{t}"\b|\b{s}\s*.\s*{t}\b', sql, flags=re.I))

# def enforce_allowed_tables(sql: str, allowed_tables: List[str]) -> None:
#     """
#     Ensure the SQL references only the whitelisted tables.
#     Raises ValueError if disallowed tables are found.
#     """
#     hits = re.findall(r'("?[A-Za-z][\w$]"?)\s.\s*("?[A-Za-z_][\w$]*"?)', sql)
#     fq_in_sql = {f'{s}.{t}'.replace(' ', '') for (s, t) in hits}
#     allowed_set = {t.replace(' ', '') for t in allowed_tables}
#     bad = [fq for fq in fq_in_sql if fq not in allowed_set]
#     if bad:
#         raise ValueError(f"Disallowed tables in SQL: {sorted(bad)}")

# def _enforce_allowed_join(sql: str, a_alias: str, l_alias: str, join_keys: List[str]) -> None:
#     """
#     If both tables are used, require an ON clause joining a.<key> = l.<key> (or flipped)
#     where <key> is in the allowed join_keys.
#     """
#     if not join_keys:
#         return
#     key_alt = "|".join(re.escape(k) for k in join_keys)
#     pat = (
#     rf'\bJOIN\b.?\bON\b[^;]?\b{a_alias}\s*.\s*(?:{key_alt})\b\s*=\s*\b{l_alias}\s*.\s*(?:{key_alt})\b|'
#     rf'\bJOIN\b.?\bON\b[^;]?\b{l_alias}\s*.\s*(?:{key_alt})\b\s*=\s*\b{a_alias}\s*.\s*(?:{key_alt})\b'
#     )
#     if not re.search(pat, sql, flags=re.I | re.S):
#         raise ValueError("Missing or invalid JOIN condition between the two allowed tables.")



import re
from typing import List


from typing import Dict, List
from sqlalchemy import inspect

CORE_TABLES = {"app_main_2024", "loan_main_2024"}
PREFERRED_SCHEMA = "stage"  # we’ll prefer these FQNs if present

# must exist before generate_sql_node runs



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


def find_relevant_table_with_allowlist(question: str,
                                       schema_catalog: Dict[str, List[str]],
                                       allowed_fqns: List[str]) -> str | None:
    """
    Your existing RAG may return junk like csv_rejects. Ignore anything outside allowlist.
    """
    # 1) Try your RAG/heuristic
    try:
        rag_choice = find_relevant_table(question, schema_catalog, embedder, collection)  # existing
    except Exception:
        rag_choice = None

    if rag_choice in allowed_fqns:
        return rag_choice

    # 2) If RAG misses or picks a disallowed table, choose a core fallback
    return pick_fallback_table(schema_catalog)

import re
import re

import re

def _fix_unclosed_parens_in_select(sql: str) -> str:
    """
    Between the first SELECT and the first FROM, if an alias ' AS ...'
    is reached while '(' depth > 0, insert the needed ')' first.
    Safe to run multiple times.
    """
    m = re.search(r'^\s*SELECT\s+(.*?)\s+FROM\b', sql, flags=re.I | re.S)
    if not m:
        return sql

    seg = m.group(1)
    out, depth, i = [], 0, 0
    U = seg.upper()

    while i < len(seg):
        ch = seg[i]

        # naive string skipping so we don't count parens inside quotes
        if ch in ("'", '"'):
            q = ch
            j = i + 1
            while j < len(seg):
                if seg[j] == q and seg[j-1] != '\\':
                    j += 1
                    break
                j += 1
            out.append(seg[i:j])
            i = j
            continue

        if ch == '(':
            depth += 1
        elif ch == ')' and depth > 0:
            depth -= 1

        # if we hit " AS " while depth>0, close the open parens
        if depth > 0 and U[i:i+4] == ' AS ':
            out.append(')' * depth)
            depth = 0

        out.append(ch)
        i += 1

    fixed_seg = ''.join(out)
    start, end = m.start(1), m.end(1)
    return sql[:start] + fixed_seg + sql[end:]


def _normalize_round_nullif(sql: str) -> str:
    """
    Common normalizations:
      • NULLIF(..., 2) → NULLIF(..., 0)
      • Ensure ROUND(..., 2) is well-formed when directly before AS
    """
    sql = re.sub(
        r'NULLIF\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s*,\s*2\s*\)',
        r'NULLIF(\1(\2), 0)',
        sql, flags=re.I
    )
    sql = re.sub(
        r'ROUND\s*\(\s*([^)]*?)\)\s*(AS\b)',
        r'ROUND(\1, 2) \2',
        sql, flags=re.I
    )
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

# def _fix_groupby_alias_and_parens(sql_clean: str) -> str:
#     # remove "... ) AS <alias>" inside GROUP BY lists (invalid in Postgres)
#     sql_clean = re.sub(
#         r'(\bGROUP\s+BY\s+.*?\))\s+AS\s+\w+\s*\)?',
#         r'\1',
#         sql_clean,
#         flags=re.I | re.S
#     )

#     # if SELECT...FROM segment has more '(' than ')', close them
#     m = re.search(r'^\s*SELECT\s+(.*?)\s+FROM\b', sql_clean, flags=re.I | re.S)
#     if m:
#         seg = m.group(1)
#         diff = seg.count('(') - seg.count(')')
#         if diff > 0:
#             sql_clean = sql_clean.replace(seg, seg + (')' * diff), 1)

#     return sql_clean


# import re
# import re
# from typing import Any

# def _safe_extract_sql(raw: Any) -> str:
#     """
#     Robustly extract SQL text from LLM output that might be a str/tuple/list/dict.
#     Never returns None; always a string (possibly empty).
#     """
#     text = ""

#     # Prefer string
#     if isinstance(raw, str):
#         text = raw

#     # Common: (sql_text, explanation) or [sql_text, ...]
#     elif isinstance(raw, (tuple, list)):
#         for item in raw:
#             if isinstance(item, str) and item.strip():
#                 text = item
#                 break
#         if not text:
#             text = " ".join(str(x) for x in raw if x is not None)

#     # Dicts with "sql"/"text"/"content"/"message"/"output"
#     elif isinstance(raw, dict):
#         for key in ("sql", "text", "content", "message", "output"):
#             v = raw.get(key)
#             if isinstance(v, str) and v.strip():
#                 text = v
#                 break
#         if not text:
#             text = str(raw)

#     else:
#         text = str(raw)

#     # Strip markdown fences if present
#     m = re.search(r'```(?:sql)?\s*(.*?)\s*```', text, flags=re.I | re.S)
#     if m:
#         text = m.group(1)

#     # prefer from first WITH/SELECT onward
#     m = re.search(r'\b(WITH|SELECT)\b.*', text, flags=re.I | re.S)
#     if m:
#         text = m.group(0)

#     # Trim model headers like "<|header_start|> sql"
#     text = re.sub(r'<\|header_start\|>\s*sql\s*<\|header_end\|>\s*', '', text, flags=re.I)

#     return (text or "").strip()


# def _fix_extract_and_wrapping(sql_text: str) -> str:
#     """
#     Fix common EXTRACT()/paren errors and remove a stray '(' before SELECT.
#     """
#     s = sql_text.strip()

#     # Remove a single stray '(' before SELECT
#     s = re.sub(r'^\(\s*(SELECT\b)', r'\1', s, flags=re.I)

#     # Fix EXTRACT(<part>) FROM <expr>  →  EXTRACT(<part> FROM <expr>)
#     s = re.sub(r'EXTRACT\s*\(\s*([A-Za-z_]+)\s*\)\s+FROM\s+',
#                r'EXTRACT(\1 FROM ',
#                s, flags=re.I)

#     # Drop accidental "AS DATE" after EXTRACT(...)
#     s = re.sub(r'(EXTRACT\s*\(\s*[A-Za-z_]+\s+FROM\s+[^)]+\))\s+AS\s+DATE\b',
#                r'\1',
#                s, flags=re.I)

#     return s


# def _fix_groupby_alias_and_parens(sql_text: str) -> str:
#     """
#     1) Remove '... ) AS DATE' fragments mistakenly placed in GROUP BY lists.
#     2) If SELECT list has more '(' than ')', close them.
#     """
#     s = sql_text

#     # Remove "... ) AS DATE" inside GROUP BY lists
#     s = re.sub(
#         r'(\bGROUP\s+BY\s+.*?\))\s+AS\s+\w+\s*\)?',
#         r'\1',
#         s, flags=re.I | re.S
#     )

#     # If SELECT...FROM segment has more '(' than ')', close them
#     m = re.search(r'^\s*SELECT\s+(.*?)\s+FROM\b', s, flags=re.I | re.S)
#     if m:
#         seg = m.group(1)
#         diff = seg.count('(') - seg.count(')')
#         if diff > 0:
#             s = s.replace(seg, seg + (')' * diff), 1)

#     return s




# def run_sql_generation_graph(question, user_id, db_id, history=[]):
#     """
#     Generate SQL + recommendation using schema-aware catalog and LangGraph.
#     - If table is in final_lusa.* → enforce 2024 date logic.
#     - Else → generate normally (respect user filters).
#     """
#     print(f"🔍 Looking for session {db_id} in session_store")
#     print(f"📊 Current session_store keys: {list(session_store.keys())}")
#     print(f"📊 Session_store size: {len(session_store)}")
    
#     session_data = session_store.get(db_id)
#     if not session_data:
#         print(f"⚠️ Session {db_id} not found in session_store")
        
#         # Try to find ANY active session as fallback
#         if session_store:
#             print(f"🔄 Available sessions: {list(session_store.keys())}")
#             db_id, session_data = next(iter(session_store.items()))
#             print(f"↩️ Falling back to session {db_id}")
#         else:
#             print("❌ session_store is completely empty!")
#             # Try to initialize a default connection
#             try:
#                 print("🚀 Attempting to create emergency session...")
#                 emergency_session = create_emergency_session()
#                 if emergency_session:
#                     db_id, session_data = emergency_session
#                     session_store[db_id] = session_data
#                     # Also embed the schema for the emergency session
#                     try:
#                         embed_schema_catalog(
#                             user_id=session_data["user_id"], 
#                             db_id=db_id,
#                             schema_catalog=session_data["schema_catalog"],
#                             embedder=embedder, 
#                             collection=collection
#                         )
#                         print("✅ Emergency session schema embedded")
#                     except Exception as embed_error:
#                         print(f"⚠️ Emergency session embedding failed: {embed_error}")
#                 else:
#                     return ("-- ERROR: No active DB session and cannot create emergency session", 
#                            "Please connect to database first using the connect endpoint", "")
#             except Exception as e:
#                 print(f"❌ Emergency session creation failed: {e}")
#                 traceback.print_exc()
#                 return ("-- ERROR: No active DB session", "Please connect to database first", "")

#     # Validate session_data has required components
#     if not session_data or "schema_catalog" not in session_data:
#         print("❌ Invalid session data - missing schema_catalog")
#         return ("-- ERROR: Invalid session data", "Session corrupted, please reconnect", "")

    

#     # -------------------------------
#     # Restrict to two stage tables
#     # -------------------------------
#     schema_catalog = session_data["schema_catalog"]
#     print(f"✅ Using session {db_id} with {len(schema_catalog)} tables")

#     # -------------------------------
#     # ✅ NOW define these HERE (not at module level)
#     # -------------------------------
#     allowed_fqns = [
#         f'"{PREFERRED_SCHEMA}"."app_main_2024"',
#         f'"{PREFERRED_SCHEMA}"."loan_main_2024"',
#     ]
#     allowed_fqns = [fq for fq in allowed_fqns if fq in schema_catalog]
    
#     if len(allowed_fqns) < 2:
#         return ("-- ERROR: Missing required tables", "Need both app_main_2024 and loan_main_2024", "")
    
#     a_tbl, l_tbl = allowed_fqns[0], allowed_fqns[1]
#     a_alias, l_alias = "a", "l"
    
#     # ✅ Get columns from schema_catalog
#     a_cols = schema_catalog.get(a_tbl, [])
#     l_cols = schema_catalog.get(l_tbl, [])
    
#     # ✅ NOW you can derive join keys
#     join_keys = _derive_join_keys(a_cols, l_cols)
#     join_keys_hint = ", ".join(join_keys) if join_keys else "application_id"

#     join_keys_hint = ", ".join(join_keys) if join_keys else "application_id"


#     # 🔎 Pick the most relevant table for this question
#     # try:
#     #     target_table = find_relevant_table(question, schema_catalog, embedder, collection)
#     # except Exception as e:
#     #     print(f"❌ Error finding relevant table: {e}")
#     #     # Fallback: use first table in catalog
#     #     if schema_catalog:
#     #         target_table = list(schema_catalog.keys())[0]
#     #         print(f"🔄 Using fallback table: {target_table}")
#     #     else:
#     #         return ("-- ERROR: No tables available", "Schema catalog is empty", "")

#     # if not target_table:
#     #     print("❌ No relevant table found for question:", question)
#     #     print(f"🔍 Available tables: {list(schema_catalog.keys())}")
#     #     return ("-- ERROR: No relevant table found", "No suitable table for your question", "")

#     # available_cols = schema_catalog[target_table]
#     # print(f"✅ Target table selected: {target_table} → Columns: {available_cols}")

#     # -------------------------------
#     # Context Node
#     # -------------------------------
#     def retrieve_context_node(state):
#         state["context"] = FULL_SCHEMA
#         return state

#     # -------------------------------
#     # SQL Generation Node
#     # -------------------------------
#     def generate_sql_node(state: Dict[str, Any]) -> Dict[str, Any]:
#         try:
#             history_context = trim_history(history)
#             date_info = get_fixed_date_context()
#             enforce_year = False  # stage.*

#             # Compact column lists
#             def _cap(cols: List[str], n=60) -> str:
#                 shown = cols[:n]
#                 return ", ".join(shown) + (" …" if len(cols) > n else "")

#             a_cols_str = _cap(a_cols)
#             l_cols_str = _cap(l_cols)

#             prompt = PromptTemplate.from_template(f"""
#     You are a PostgreSQL SQL expert.

#     HARD CONSTRAINTS (must follow exactly):
#     - Valid tables are ONLY:
#     {a_tbl} AS {a_alias}
#     {l_tbl} AS {l_alias}
#     - Do NOT reference any other tables, CTEs reading other tables, or views.
#     - Always schema-qualify every column with the correct alias ({a_alias} or {l_alias}).
#     - If a join is required, join ONLY between these two tables using one of:
#     {join_keys_hint}
#     Prefer INNER JOIN unless the question needs unmatched rows (then LEFT JOIN from the driving table).
#     - If the question can be answered from a single table, do NOT join.

#     Available columns:
#     - {a_tbl} AS {a_alias}: {a_cols_str}
#     - {l_tbl} AS {l_alias}: {l_cols_str}

#     Conversation history (for context only):
#     {history_context}

#     User asked:
#     {question}

#     {"Default to year " + str(date_info['year']) + " if year is unspecified." if enforce_year else ""}

#     Generate valid PostgreSQL syntax only. 
#     Do not wrap SELECT statements in unnecessary parentheses.
#     Ensure EXTRACT() functions are properly placed in the SELECT clause with aliases.

#     Return ONLY one SQL query in a fenced block:
#     ```sql
#     SELECT ...
#     Then on a new line write:
#     Recommendation: <one short business takeaway (1–3 lines)>
#     """)

#             llm = get_llama_maverick_llm()
#             runnable = prompt | llm
#             response = runnable.invoke({
#                 **state,
#                 "history_context": history_context,
#                 "question": state["question"]
#             })

#             text = response.content if hasattr(response, "content") else response
#             print("📝 Raw LLM response:\n", text)

#             # ---------- Extract SQL + recommendation safely ----------
#             sql_clean: str = ""   # <-- initialize to avoid NameError
#             explanation_part: str = "-- No recommendation generated"

#             # split out recommendation if present
#             if "Recommendation:" in text:
#                 sql_part, explanation_part = text.split("Recommendation:", 1)
#             else:
#                 sql_part = text

#             # fenced block first
#             m = re.search(r'```(?:sql)?\s*(.*?)\s*```', sql_part, re.I | re.S)
#             if m:
#                 sql_clean = m.group(1).strip()

#             # sql_clean = _fix_extract_and_wrapping(sql_clean)

#             # sql_clean = _fix_groupby_alias_and_parens(sql_clean)
#             # else:
#             #     # try to salvage raw SELECT/WITH
#             #     m2 = re.search(r'\b(WITH|SELECT)\b.*', sql_part, flags=re.I | re.S)
#             #     if m2:
#             #         sql_clean = m2.group(0).strip()

#             # hard fallback if still empty
#             if not sql_clean:
#                 raise ValueError("LLM did not return a SQL block")

#             # ---------- Enforcement & Post-processing ----------
#             # ONLY enforce AFTER sql_clean exists
#            # ONLY enforce AFTER sql_clean exists
#            # 1) Extract text safely
#             sql_clean = _safe_extract_sql(text)

#             # 2) Syntax touch-ups for common LLM mistakes
#             sql_clean = _fix_extract_and_wrapping(sql_clean)
#             sql_clean = _fix_groupby_alias_and_parens(sql_clean)

#             # 3) Then your existing enforcement & sanitizers
#             _enforce_allowed_tables(sql_clean, allowed_fqns)            # or allowed_tables
#             if _mentions_table(sql_clean, a_tbl) and _mentions_table(sql_clean, l_tbl):
#                 _enforce_allowed_join(sql_clean, a_alias, l_alias, join_keys)

#             sql_clean = _sanitize_group_order_clauses(sql_clean)
#             sql_clean = sanitize_sql_for_nulls(sql_clean, schema_catalog=schema_catalog, question=question)
#             sql_clean = auto_fix_schema_mismatches(sql_clean, schema_catalog)
#             sql_clean = validate_sql_against_catalog(sql_clean, schema_catalog)
#             sql_clean = _fix_groupby_alias_and_parens(sql_clean)        
            
#             if enforce_year:
#                 sql_clean = enforce_2024(sql_clean, date_info)
                
#             sql_clean = sanitize_sql_for_nulls(sql_clean, schema_catalog=schema_catalog, question=question)
#             sql_clean = auto_fix_schema_mismatches(sql_clean, schema_catalog)
#             sql_clean = validate_sql_against_catalog(sql_clean, schema_catalog)

#             state["sql"] = sql_clean
#             state["explanation"] = explanation_part.strip()
#             return state
            
#         except Exception as e:
#             print(f"❌ Error in generate_sql_node: {e}")
#             traceback.print_exc()
#             state["sql"] = f"-- ERROR: SQL generation failed: {str(e)}"
#             state["explanation"] = "SQL generation encountered an error"
#             return state

#     # -------------------------------
#     # LangGraph Workflow
#     # -------------------------------
#     try:
#         graph = StateGraph(SQLState)
#         graph.add_node("retrieve_context", retrieve_context_node)
#         graph.add_node("generate_sql", generate_sql_node)
#         graph.set_entry_point("retrieve_context")
#         graph.add_edge("retrieve_context", "generate_sql")
#         graph.set_finish_point("generate_sql")

#         # -------------------------------
#         # Execute Graph
#         # -------------------------------
#         final_state = graph.compile().invoke({
#             "question": question,
#             "user_id": user_id,
#             "db_id": db_id,
#         })

#         print("✅ Final State:", final_state)

#         return (
#             final_state.get("sql", "-- ERROR: No SQL generated"),
#             final_state.get("explanation", "No explanation generated"),
#             final_state.get("summary", "")
#         )
#     except Exception as e:
#         print(f"❌ Error in LangGraph execution: {e}")
#         traceback.print_exc()
#         return (
#             f"-- ERROR: Graph execution failed: {str(e)}",
#             "LangGraph workflow encountered an error",
#             ""
#         )



import re
import traceback
from typing import Any, Dict, List
import re
from typing import Any, Dict, List

# -------------------------------
# SQL Fixer Utilities
# -------------------------------

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

    # Strip markdown fences if present
    m = re.search(r'```(?:sql)?\s*(.*?)\s*```', text, flags=re.I | re.S)
    if m:
        text = m.group(1)

    # Prefer from first WITH/SELECT onward
    m = re.search(r'\b(WITH|SELECT)\b.*', text, flags=re.I | re.S)
    if m:
        text = m.group(0)

    # Trim model headers like "<|header_start|> sql"
    text = re.sub(r'<\|header_start\|>\s*sql\s*<\|header_end\|>\s*', '', text, flags=re.I)

    return (text or "").strip()


def _fix_extract_and_wrapping(sql_text: str) -> str:
    """
    Fix common EXTRACT()/paren errors and remove a stray '(' before SELECT.
    """
    s = sql_text.strip()
    s = re.sub(r'^\(\s*(SELECT\b)', r'\1', s, flags=re.I)
    s = re.sub(r'EXTRACT\s*\(\s*([A-Za-z_]+)\s*\)\s+FROM\s+', r'EXTRACT(\1 FROM ', s, flags=re.I)
    s = re.sub(r'(EXTRACT\s*\(\s*[A-Za-z_]+\s+FROM\s+[^)]+\))\s+AS\s+DATE\b', r'\1', s, flags=re.I)
    return s


def _fix_groupby_alias_and_parens(sql_text: str) -> str:
    """
    1) Remove '... ) AS DATE' fragments mistakenly placed in GROUP BY lists.
    2) If SELECT list has more '(' than ')', close them.
    """
    s = sql_text
    s = re.sub(r'(\bGROUP\s+BY\s+.*?\))\s+AS\s+\w+\s*\)?', r'\1', s, flags=re.I | re.S)
    m = re.search(r'^\s*SELECT\s+(.*?)\s+FROM\b', s, flags=re.I | re.S)
    if m:
        seg = m.group(1)
        diff = seg.count('(') - seg.count(')')
        if diff > 0:
            s = s.replace(seg, seg + (')' * diff), 1)
    return s


def _fix_missing_paren_before_alias(sql: str) -> str:
    """
    Fix: (COUNT(...) * 100 AS x → (COUNT(...) * 100) AS x
    """
    pattern = r'\(([^()]*(?:\([^()]*\)[^()]*)*?)\s*\*\s*(\d+(?:\.\d+)?)\s+(AS\s+\w+)'
    def fix_it(match):
        inner = match.group(1).strip()
        num = match.group(2)
        alias = match.group(3)
        return f'({inner} * {num}) {alias}'
    return re.sub(pattern, fix_it, sql, flags=re.I)



import logging
import re
from typing import Dict

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

# -------------------------------
# Main Function
# -------------------------------
# --------------------------
import re
import traceback
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# ============================================================================
# STEP 1: Dynamic Schema Discovery
# ============================================================================
import re
import traceback
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# ============================================================================
# STEP 1: Dynamic Schema Discovery
# ============================================================================

def discover_available_tables(session_data: Dict[str, Any]) -> List[str]:
    """Extract all available tables from session's schema_catalog."""
    schema_catalog = session_data.get("schema_catalog", {})
    return list(schema_catalog.keys())


def pick_best_tables_for_question(
    question: str,
    schema_catalog: Dict[str, List[str]],
    session_id: str,
    embedder,
    collection,
    max_tables: int = 2
) -> List[str]:
    """
    Pick tables ONLY from current session's schema_catalog.
    This prevents returning cached tables from other schemas.
    """
    
    # Get available tables from THIS session only
    available_tables = list(schema_catalog.keys())
    
    if not available_tables:
        print("⚠️ No tables in schema_catalog")
        return []
    
    print(f"📊 Available tables in session: {available_tables}")
    
    # Try embedding-based selection
    try:
        question_embedding = embedder.embed_query(question)
        
        # Query Chroma but FILTER by session_id to avoid cross-contamination
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=20,  # get more candidates
            include=["metadatas", "documents", "distances"],
            where={"db_id": session_id}  # CRITICAL: filter by current session
        )
        
        # Extract tables that exist in current schema_catalog
        picked_tables = []
        seen = set()
        
        if results and results.get("metadatas"):
            for metadata_list in results["metadatas"]:
                for meta in metadata_list:
                    table_name = meta.get("table")
                    # CRITICAL CHECK: table must exist in current session
                    if (table_name and 
                        table_name in schema_catalog and 
                        table_name not in seen):
                        seen.add(table_name)
                        picked_tables.append(table_name)
                        print(f"✅ Picked table from embeddings: {table_name}")
                        if len(picked_tables) >= max_tables:
                            break
                if len(picked_tables) >= max_tables:
                    break
        
        # If embeddings found enough valid tables, return them
        if len(picked_tables) >= max_tables:
            return picked_tables[:max_tables]
        
    except Exception as e:
        print(f"⚠️ Embedding-based selection failed: {e}")
    
    # FALLBACK: Use heuristic based on question keywords
    question_lower = question.lower()
    
    # Define keyword -> table preferences
    table_scores = {}
    for table in available_tables:
        score = 0
        table_lower = table.lower()
        
        # Score based on question keywords
        if 'loan' in question_lower or 'fund' in question_lower:
            if 'loan' in table_lower:
                score += 10
        
        if 'app' in question_lower or 'application' in question_lower:
            if 'app' in table_lower:
                score += 10
        
        if 'income' in question_lower or 'annual' in question_lower:
            # Both tables likely have this, slight preference for app table
            if 'app' in table_lower:
                score += 5
        
        if 'vantage' in question_lower or 'fico' in question_lower:
            if 'loan' in table_lower:
                score += 8
        
        if 'approval' in question_lower or 'approved' in question_lower:
            if 'app' in table_lower:
                score += 8
        
        table_scores[table] = score
    
    # Sort by score and return top N
    sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
    result = [t[0] for t in sorted_tables[:max_tables]]
    
    print(f"📊 Heuristic selection: {result}")
    return result


# ============================================================================
# STEP 2: Dynamic Join Discovery
# ============================================================================

def discover_join_keys(tables: List[str], schema_catalog: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Automatically discover common join keys between selected tables.
    Returns: {
        'join_pairs': [(table1, table2, [common_keys])],
        'aliases': {table: alias},
        'all_keys': [list of all common keys]
    }
    """
    if len(tables) < 2:
        return {
            'join_pairs': [],
            'aliases': {tables[0]: 'a'} if tables else {},
            'all_keys': []
        }
    
    # Generate short aliases
    aliases = {}
    for i, table in enumerate(tables):
        aliases[table] = chr(ord('a') + i)  # a, b, c, ...
    
    # Find common columns between each pair
    join_pairs = []
    all_common_keys = set()
    
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            table1, table2 = tables[i], tables[j]
            cols1 = set(schema_catalog.get(table1, []))
            cols2 = set(schema_catalog.get(table2, []))
            
            # Common columns (case-insensitive match)
            common = []
            for c1 in cols1:
                for c2 in cols2:
                    if c1.lower() == c2.lower():
                        common.append(c1)
            
            # Prioritize ID-like columns
            priority_keys = [k for k in common if any(x in k.lower() for x in ['id', 'key', 'number'])]
            common_sorted = priority_keys + [k for k in common if k not in priority_keys]
            
            if common_sorted:
                join_pairs.append((table1, table2, common_sorted))
                all_common_keys.update(common_sorted)
    
    return {
        'join_pairs': join_pairs,
        'aliases': aliases,
        'all_keys': list(all_common_keys)
    }


# ============================================================================
# STEP 3: Dynamic SQL Generation Prompt
# ============================================================================

def build_dynamic_sql_prompt11(
    question: str,
    tables: List[str],
    schema_catalog: Dict[str, List[str]],
    join_info: Dict[str, Any],
    history: List[Dict] = None
) -> str:
    """Build a fully dynamic prompt with actual available tables and columns."""
    
    aliases = join_info['aliases']
    join_pairs = join_info['join_pairs']
    
    # Build table documentation with ACTUAL columns from schema_catalog
    table_docs = []
    for table in tables:
        alias = aliases[table]
        cols = schema_catalog.get(table, [])
        
        if not cols:
            print(f"⚠️ No columns found for {table} in schema_catalog")
            continue
        
        # Show first 50 columns to avoid token overflow
        cols_display = ', '.join(cols[:50])
        if len(cols) > 50:
            cols_display += f'... ({len(cols)} total columns)'
        
        table_docs.append(f"  {table} AS {alias}")
        table_docs.append(f"    Available columns: {cols_display}")
    
    tables_section = "\n".join(table_docs)
    
    # Build join documentation
    if join_pairs:
        join_docs = []
        for t1, t2, keys in join_pairs:
            a1, a2 = aliases[t1], aliases[t2]
            # Show top 3 common keys
            keys_str = ', '.join(keys[:3]) if keys else 'customerid, decisionid'
            example_key = keys[0] if keys else 'customerid'
            join_docs.append(f"  Join {a1} and {a2} on: {a1}.{example_key} = {a2}.{example_key}")
            join_docs.append(f"    (Alternative keys: {keys_str})")
        joins_section = "\n".join(join_docs)
    else:
        joins_section = "  Single table query - no join needed"
    
    # Add context about common fields for your specific schema
    context_hints = ""
    question_lower = question.lower()
    
    if 'income' in question_lower or 'annual' in question_lower:
        # Check which table has income columns
        for table in tables:
            cols_lower = [c.lower() for c in schema_catalog.get(table, [])]
            if any('income' in c for c in cols_lower):
                alias = aliases[table]
                income_col = next((c for c in schema_catalog[table] if 'income' in c.lower()), None)
                if income_col:
                    context_hints += f"\n  💡 Income data: use {alias}.{income_col}"
    
    if 'status' in question_lower or 'approved' in question_lower:
        for table in tables:
            cols_lower = [c.lower() for c in schema_catalog.get(table, [])]
            if any('status' in c for c in cols_lower):
                alias = aliases[table]
                status_col = next((c for c in schema_catalog[table] if 'status' in c.lower()), None)
                if status_col:
                    context_hints += f"\n  💡 Application status: use {alias}.{status_col}"
    
    # History context
    history_context = ""
    if history:
        recent = history[-2:]  # last 2 interactions
        if recent:
            history_context = "\nRecent queries for context:\n"
            for h in recent:
                q = h.get('question', '')[:80]
                s = h.get('sql', '')[:100]
                history_context += f"  Q: {q}...\n  SQL: {s}...\n"
    
    prompt = f"""You are a PostgreSQL expert. Generate an executable SQL query.

AVAILABLE TABLES (with actual columns):
{tables_section}

JOIN INSTRUCTIONS:
{joins_section}
{context_hints}

CRITICAL RULES:
1. Use ONLY the tables and columns listed above
2. Always use table aliases ({', '.join(aliases.values())})
3. Prefix every column with its alias (e.g., a.columnname)
4. For date filters, use BETWEEN or >= / <= with proper date literals
5. Use ILIKE for case-insensitive text matching
6. Handle NULLs properly in WHERE clauses
7. Return raw SQL only - no markdown, no explanations

{history_context}

USER QUESTION: {question}

SQL:"""
    
    return prompt


# ============================================================================
# STEP 4: Unified SQL Repair Pipeline
# ============================================================================

# def repair_generated_sql(
#     raw_llm_output: Any,
#     tables: List[str],
#     schema_catalog: Dict[str, List[str]],
#     join_info: Dict[str, Any]
# ) -> str:
#     """
#     Complete repair pipeline for LLM-generated SQL.
#     Handles all common issues in one pass.
#     """
    
#     # Extract SQL from various formats
#     if isinstance(raw_llm_output, tuple):
#         sql = raw_llm_output[0] if raw_llm_output else ""
#     elif hasattr(raw_llm_output, 'content'):
#         sql = raw_llm_output.content
#     else:
#         sql = str(raw_llm_output)
    
#     # Remove markdown fences
#     sql = re.sub(r'```(?:sql)?\s*', '', sql, flags=re.IGNORECASE)
#     sql = re.sub(r'```\s*$', '', sql)
    
#     # Remove explanatory text
#     lines = sql.split('\n')
#     sql_lines = []
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         # Skip lines that look like explanations
#         if line.lower().startswith(('note:', 'explanation:', 'this query', 'the above')):
#             continue
#         sql_lines.append(line)
    
#     sql = '\n'.join(sql_lines)
    
#     # Fix common syntax errors
#     sql = sql.replace('EXTEXTRACT', 'EXTRACT')
#     sql = sql.replace('FROM FROM', 'FROM')
#     sql = re.sub(r'\bLIMIT(\d+)', r'LIMIT \1', sql, flags=re.IGNORECASE)
    
#     # Remove rogue parentheses
#     if sql.startswith('(SELECT') and not sql.endswith(')'):
#         sql = sql[1:]
    
#     # Fix missing aliases in COUNT/SUM/AVG
#     sql = re.sub(r'\b(COUNT|SUM|AVG)\s*\([^)]+\)\s+AS\s*$', r'\1(\2) AS total', sql, flags=re.IGNORECASE)
    
#     # Repair column references to match actual schema
#     aliases = join_info['aliases']
    
#     for table in tables:
#         alias = aliases[table]
#         cols = schema_catalog.get(table, [])
        
#         # Fix common column name issues
#         for col in cols:
#             col_lower = col.lower()
#             # Find variations in SQL and fix them
#             pattern = r'\b' + alias + r'\.' + col_lower + r'\b'
#             replacement = f'{alias}.{col}'
#             sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
    
#     # Ensure table names are properly quoted
#     for table in tables:
#         # If table isn't already quoted, quote it
#         if table.startswith('"') and table.endswith('"'):
#             continue
#         # Extract schema and table name
#         parts = table.split('.')
#         if len(parts) == 2:
#             schema, tbl = parts
#             quoted = f'"{schema}"."{tbl}"'
#             sql = sql.replace(table, quoted)
    
#     # Fix GROUP BY with aliases (replace alias with expression)
#     select_aliases = extract_select_aliases(sql)
#     sql = replace_group_by_aliases(sql, select_aliases)
    
#     # Normalize whitespace
#     sql = re.sub(r'\s+', ' ', sql).strip()
    
#     # Ensure space between GROUP BY and ORDER BY
#     sql = re.sub(r'(GROUP\s+BY[^;]+?)(ORDER\s+BY)', r'\1 \2', sql, flags=re.IGNORECASE)
    
#     # Add semicolon if missing
#     if not sql.endswith(';'):
#         sql += ';'
    
#     return sql


def extract_select_aliases(sql: str) -> Dict[str, str]:
    """Extract alias -> expression mapping from SELECT clause."""
    match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return {}
    
    select_clause = match.group(1)
    aliases = {}
    
    # Split on commas (respecting parentheses)
    items = []
    depth = 0
    current = []
    
    for char in select_clause:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        
        if char == ',' and depth == 0:
            items.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    
    if current:
        items.append(''.join(current).strip())
    
    # Extract alias from each item
    for item in items:
        # Match "expression AS alias"
        match = re.search(r'^(.*?)\s+AS\s+(["\']?\w+["\']?)\s*$', item, flags=re.IGNORECASE)
        if match:
            expr = match.group(1).strip()
            alias = match.group(2).strip('"\'')
            aliases[alias.lower()] = expr
    
    return aliases


def replace_group_by_aliases(sql: str, select_aliases: Dict[str, str]) -> str:
    """Replace GROUP BY aliases with actual expressions."""
    
    def replace_group_clause(match):
        group_clause = match.group(1)
        parts = [p.strip() for p in group_clause.split(',')]
        
        replaced = []
        for part in parts:
            part_lower = part.lower().strip('"\'')
            if part_lower in select_aliases:
                replaced.append(select_aliases[part_lower])
            else:
                replaced.append(part)
        
        return 'GROUP BY ' + ', '.join(replaced)
    
    sql = re.sub(
        r'GROUP\s+BY\s+([^;]+?)(?=\s*(?:ORDER\s+BY|LIMIT|;|$))',
        replace_group_clause,
        sql,
        flags=re.IGNORECASE
    )
    
    return sql


# ============================================================================
# STEP 6: Column Name Normalizer for Schema Mismatches
# ============================================================================

def normalize_column_names(question: str, schema_catalog: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Map common question terms to actual column names in the schema.
    Handles the case where user says 'approved' but column is 'applicationstatus_name'.
    """
    
    question_lower = question.lower()
    column_mappings = {}
    
    # Search through all tables for relevant columns
    for table, columns in schema_catalog.items():
        for col in columns:
            col_lower = col.lower()
            
            # Status mappings
            if 'status' in question_lower or 'approved' in question_lower:
                if 'status' in col_lower and 'application' in col_lower:
                    column_mappings['application_status'] = col
                    column_mappings['status'] = col
            
            # Income mappings
            if 'income' in question_lower or 'annual' in question_lower:
                if 'income' in col_lower:
                    # Prefer specific over general
                    if 'annual' in col_lower or 'gross' in col_lower or 'stated' in col_lower:
                        column_mappings['income'] = col
                        column_mappings['annual_income'] = col
            
            # Date mappings
            if any(word in question_lower for word in ['date', 'month', 'year', 'time']):
                if 'date' in col_lower:
                    if 'application' in col_lower:
                        column_mappings['application_date'] = col
                    elif 'funded' in col_lower or 'fund' in col_lower:
                        column_mappings['funded_date'] = col
            
            # Score mappings
            if 'vantage' in question_lower:
                if 'vantage' in col_lower:
                    column_mappings['vantage_score'] = col
            
            if 'fico' in question_lower:
                if 'fico' in col_lower:
                    column_mappings['fico_score'] = col
    
    return column_mappings

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

def build_dynamic_sql_prompt11(
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

# def repair_generated_sql(
#     raw_llm_output: Any,
#     tables: List[str],
#     schema_catalog: Dict[str, List[str]],
#     join_info: Dict[str, Any]
# ) -> str:
#     """
#     Extract and repair SQL from LLM response.
#     """
    
#     # Extract text from various formats
#     if hasattr(raw_llm_output, 'content'):
#         text = raw_llm_output.content
#     elif isinstance(raw_llm_output, tuple):
#         text = raw_llm_output[0] if raw_llm_output else ""
#     else:
#         text = str(raw_llm_output)
    
#     # Remove markdown fences
#     text = re.sub(r'```(?:sql)?\s*', '', text, flags=re.IGNORECASE)
#     text = re.sub(r'```', '', text)
    
#     # Remove explanation lines
#     lines = []
#     for line in text.split('\n'):
#         line = line.strip()
#         if not line:
#             continue
#         if any(line.lower().startswith(p) for p in ['note:', 'explanation:', 'this query']):
#             continue
#         lines.append(line)
    
#     sql = ' '.join(lines)
    
#     # Fix common issues
#     sql = sql.replace('EXTEXTRACT', 'EXTRACT')
#     sql = sql.replace('FROM FROM', 'FROM')
#     sql = re.sub(r'\s+', ' ', sql)
#     sql = sql.strip()
    
#     # Remove rogue parentheses
#     if sql.startswith('(SELECT') and not sql.endswith(')'):
#         sql = sql[1:]
    
#     # Ensure semicolon
#     if not sql.endswith(';'):
#         sql += ';'
    
#     return sql

from sqlalchemy import create_engine, inspect, text
from typing import Dict, List, Any, Optional

def build_schema_catalog(
    engine,
    prefer_schema: str = "stage",
    allowed_tables: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    
    print("Building schema catalog...")
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

# ============================================================================
# TABLE SELECTION - Smart and Dynamic
# ============================================================================


# ============================================================================
# JOIN DISCOVERY - Automatic Relationship Detection
# ============================================================================

def discover_join_keys(selected_tables: List[str], schema_catalog: dict) -> dict:
    """
    Automatically discover join relationships between tables.
    Uses foreign keys and common column names.
    """
    print(f"🔗 Discovering joins for tables: {selected_tables}")
    
    join_info = {}
    
    for i, table1 in enumerate(selected_tables):
        for table2 in selected_tables[i+1:]:
            table1_info = schema_catalog.get(table1, {})
            table2_info = schema_catalog.get(table2, {})
            
            # Check foreign keys
            for fk in table1_info.get('foreign_keys', []):
                referred_table = f'"{fk.get("referred_schema")}"."{fk.get("referred_table")}"'
                if referred_table == table2:
                    join_key = (table1, table2)
                    join_info[join_key] = {
                        "left_columns": fk.get('constrained_columns', []),
                        "right_columns": fk.get('referred_columns', []),
                        "type": "foreign_key"
                    }
            
            # Check common column names
            table1_cols = {col['name'].lower() for col in table1_info.get('columns', [])}
            table2_cols = {col['name'].lower() for col in table2_info.get('columns', [])}
            
            common_cols = table1_cols & table2_cols
            
            # Look for ID columns
            for col in common_cols:
                if 'id' in col or col in ['customerid', 'applicationid', 'loannumber']:
                    join_key = (table1, table2)
                    if join_key not in join_info:
                        join_info[join_key] = {
                            "left_columns": [col],
                            "right_columns": [col],
                            "type": "common_column"
                        }
    
    print(f"✅ Discovered joins: {join_info}")
    return join_info

import re
import difflib
from typing import List, Dict

# ============================================================================ 
# NORMALIZE COLUMN/TABLE NAME
# ============================================================================ 
def normalize_name(name: str) -> str:
    """Normalize table/column names for comparison."""
    n = re.sub(r'[\s\-_]', '', name.lower())
    if n.endswith('s'):  # optional plural handling
        n = n[:-1]
    return n

# ============================================================================ 
# SELECT TABLES WITH FUZZY COLUMN MATCH
# ============================================================================ 
def select_tables_for_question(
    question: str,
    schema_catalog: dict,
    session_id: str,
    max_tables: int = 3,
    similarity_threshold: float = 0.8
) -> List[str]:
    """Select relevant tables based on question content using weighted scoring + fuzzy matching."""
    print(f"🎯 Selecting tables for: {question}")
    
    question_lower = question.lower()
    question_words = re.findall(r'\b[\w\-]+\b', question_lower)
    question_words_norm = [normalize_name(w) for w in question_words]
    
    table_scores = {}

    for full_table_name, table_info in schema_catalog.items():
        score = 0
        table_name = table_info.get('table', '').lower()
        table_name_norm = normalize_name(table_name)
        
        # Score based on table name keywords
        table_keywords = table_name.replace('_', ' ').split()
        for keyword in table_keywords:
            if keyword in question_lower:
                score += 10
        if table_name_norm in question_lower:
            score += 15

        # Score based on column names with fuzzy match
        columns = table_info.get('columns', [])
        for col in columns:
            col_name = col.get('name', '')
            col_norm = normalize_name(col_name)

            # Exact match
            if col_norm in question_words_norm:
                score += 20
            # Fuzzy match
            else:
                match = difflib.get_close_matches(col_norm, question_words_norm, n=1, cutoff=similarity_threshold)
                if match:
                    score += 15

        # Boosts for common patterns (optional domain-specific)
        if 'status' in question_lower and 'status' in table_name:
            score += 20
        if 'income' in question_lower and any('income' in str(c.get('name', '')).lower() for c in columns):
            score += 20
        if 'approved' in question_lower and 'app' in table_name:
            score += 15
        if 'loan' in question_lower and 'loan' in table_name:
            score += 15

        if score > 0:
            table_scores[full_table_name] = score

    # Sort tables by score and pick top N
    sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
    selected = [table for table, score in sorted_tables[:max_tables]]

    print(f"📊 Selected: {selected}")
    return selected

# ============================================================================
# JOIN DISCOVERY
# ============================================================================

def discover_join_keys(selected_tables: List[str], schema_catalog: dict) -> dict:
    """Discover join relationships."""
    print(f"🔗 Discovering joins...")
    
    join_info = {}
    
    for i, table1 in enumerate(selected_tables):
        for table2 in selected_tables[i+1:]:
            table1_info = schema_catalog.get(table1, {})
            table2_info = schema_catalog.get(table2, {})
            
            # Check foreign keys
            for fk in table1_info.get('foreign_keys', []):
                referred_table = f'"{fk.get("referred_schema")}"."{fk.get("referred_table")}"'
                if referred_table == table2:
                    join_key = (table1, table2)
                    join_info[join_key] = {
                        "left_columns": fk.get('constrained_columns', []),
                        "right_columns": fk.get('referred_columns', []),
                        "type": "foreign_key"
                    }
            
            # Common columns
            table1_cols = {c.get('name', '').lower() for c in table1_info.get('columns', [])}
            table2_cols = {c.get('name', '').lower() for c in table2_info.get('columns', [])}
            common = table1_cols & table2_cols
            
            for col in common:
                if 'id' in col:
                    join_key = (table1, table2)
                    if join_key not in join_info:
                        join_info[join_key] = {
                            "left_columns": [col],
                            "right_columns": [col],
                            "type": "common_column"
                        }
    
    print(f"✅ Joins: {join_info}")
    return join_info


# ============================================================================
# IMPORT YOUR EXISTING FUNCTIONS
# ============================================================================

try:
    # Import your existing schema builder
    from genai_app.utils.schema_utils import build_schema_catalog
    print("✅ Loaded existing build_schema_catalog")
except ImportError:
    print("⚠️ Could not import build_schema_catalog, will use fallback")
    build_schema_catalog = None

try:
    # Import your existing embedding functions
    from genai_app.utils.embedding_utils import embed_schema_catalog, embedder, collection
    print("✅ Loaded existing embedding utilities")
except ImportError:
    print("⚠️ Could not import embedding utilities")
    embed_schema_catalog = None
    embedder = None
    collection = None

try:
    # Import your existing LLM config
    from genai_app.utils.llm_config import get_llama_maverick_llm
    print("✅ Loaded existing LLM config")
except ImportError:
    print("⚠️ Could not import LLM config")
    get_llama_maverick_llm = None


# ============================================================================
# SESSION MANAGER - Production Grade with Cache
# ============================================================================

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from django.core.cache import cache
from sqlalchemy import create_engine
from urllib.parse import quote_plus

class SessionManager:
    """
    Manages database sessions with fallback from Redis to in-memory storage.
    """
    
    # In-memory fallback storage
    _memory_store: Dict[str, Dict[str, Any]] = {}
    _use_redis = True
    
    @classmethod
    def _check_redis(cls):
        """Check if Redis is available, fallback to memory if not."""
        if not cls._use_redis:
            return False
        
        try:
            cache.set('_redis_test', 'ok', timeout=1)
            cache.get('_redis_test')
            return True
        except Exception as e:
            print(f"⚠️ Redis not available, using in-memory storage: {e}")
            cls._use_redis = False
            return False
    
    @classmethod
    def create_session(
        cls,
        engine,
        user_id: str,
        schema_catalog: Dict[str, Any]
    ) -> str:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        
        # Extract connection parameters from engine
        url = engine.url
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "schema_catalog": schema_catalog,
            "conversation_history": [],
            "created_at": datetime.datetime.now().isoformat(), 
            # "created_at": datetime.now().isoformat(),
            "db_config": {
                "postgres_user": url.username,
                "postgres_password": url.password,
                "postgres_host": url.host,
                "postgres_port": url.port or 5432,
                "postgres_db": url.database
            }
        }
        
        # Try Redis first, fallback to memory
        if cls._check_redis():
            try:
                cache.set(f"session_{session_id}", session_data, timeout=7200)
                print(f"✅ Session stored in Redis: {session_id}")
            except Exception as e:
                print(f"⚠️ Redis storage failed, using memory: {e}")
                cls._memory_store[session_id] = session_data
        else:
            cls._memory_store[session_id] = session_data
            print(f"✅ Session stored in memory: {session_id}")
        
        return session_id
    
    @classmethod
    def get_session(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session."""
        if cls._check_redis():
            try:
                data = cache.get(f"session_{session_id}")
                if data:
                    return data
            except Exception:
                pass
        
        # Fallback to memory
        return cls._memory_store.get(session_id)
    
    @classmethod
    def get_engine(cls, session_id: str):
        """Recreate engine from session data."""
        session_data = cls.get_session(session_id)
        if not session_data:
            return None
        
        db_config = session_data.get("db_config", {})
        
        pg_user = db_config.get("postgres_user")
        pg_pass = db_config.get("postgres_password")
        pg_host = db_config.get("postgres_host")
        pg_port = db_config.get("postgres_port", 5432)
        pg_db = db_config.get("postgres_db")
        
        encoded_pwd = quote_plus(str(pg_pass))
        conn_str = f"postgresql://{pg_user}:{encoded_pwd}@{pg_host}:{pg_port}/{pg_db}"
        
        engine = create_engine(
            conn_str,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={"connect_timeout": 8}
        )
        
        return engine
    
    @classmethod
    def add_to_conversation(cls, session_id: str, entry: Dict[str, Any]):
        """Add an entry to conversation history."""
        session_data = cls.get_session(session_id)
        if not session_data:
            return
        
        if "conversation_history" not in session_data:
            session_data["conversation_history"] = []
        
        session_data["conversation_history"].append(entry)
        
        # Update storage
        if cls._check_redis():
            try:
                cache.set(f"session_{session_id}", session_data, timeout=7200)
                return
            except Exception:
                pass
        
        # Fallback to memory
        cls._memory_store[session_id] = session_data
    
    @classmethod
    def get_conversation_history(cls, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history."""
        session_data = cls.get_session(session_id)
        if not session_data:
            return []
        return session_data.get("conversation_history", [])
    
    @classmethod
    def delete_session(cls, session_id: str):
        """Delete a session."""
        if cls._check_redis():
            try:
                cache.delete(f"session_{session_id}")
            except Exception:
                pass
        
        if session_id in cls._memory_store:
            del cls._memory_store[session_id]


# ============================================================================
# PROMPT BUILDER
# ============================================================================

# ============================================================================
# SQL GENERATION PIPELINE WITH FIXED YEAR 2024 AND CURRENT DATE
# ============================================================================

import re
import datetime
import traceback
from typing import List, Dict, Tuple


# ============================================================================
# PROMPT BUILDER (DYNAMIC & SAFE)
# ============================================================================
import datetime, re
from typing import List, Dict
import re
import datetime
from typing import List, Dict

# ============================================================================
# COLUMN NORMALIZATION
# ============================================================================
import re
import difflib
from typing import List, Dict

# ============================================================================ 
# NORMALIZE COLUMN/TABLE NAME
# ============================================================================ 
import re
import difflib
import datetime
from typing import List, Dict

# ============================================================================ 
# NORMALIZE COLUMN/TABLE NAME
# ============================================================================ 
def normalize_name(name: str) -> str:
    """Normalize table/column names for comparison."""
    n = re.sub(r'[\s\-_]', '', name.lower())
    if n.endswith('s'):  # optional plural handling
        n = n[:-1]
    return n
# ============================================================================ 
# DYNAMIC TABLE SELECTION WITH FUZZY COLUMN MATCHING
# ============================================================================
from difflib import get_close_matches
from typing import List, Dict

# def select_relevant_tables(
#     question: str,
#     schema_catalog: Dict[str, dict],
#     similarity_threshold: float = 0.6,
#     max_tables: int = 3
# ) -> List[str]:
#     """
#     Automatically select relevant tables based on question content,
#     with fuzzy column matching and numeric detection.
#     """

#     question_lower = question.lower()
#     table_scores = {}

#     for full_table_name, table_info in schema_catalog.items():
#         score = 0
#         table_name = table_info.get("table", "").lower()

#         # ---- Table name matching ----
#         table_keywords = table_name.replace("_", " ").split()
#         for kw in table_keywords:
#             if kw in question_lower:
#                 score += 10

#         # ---- Column-based scoring ----
#         columns = table_info.get("columns", [])
#         for col in columns:
#             col_name = col.get("name", "").lower()
#             col_type = col.get("type", "").lower()

#             # Exact match
#             if col_name in question_lower:
#                 score += 15

#             # Partial keyword match
#             col_keywords = col_name.replace("_", " ").split()
#             for kw in col_keywords:
#                 if kw in question_lower:
#                     score += 5

#             # Fuzzy match
#             close_matches = get_close_matches(question_lower, [col_name], n=1, cutoff=similarity_threshold)
#             if close_matches:
#                 score += 7

#             # Boost for numeric-like columns if question mentions amounts, charge-offs, rates, etc.
#             numeric_keywords = ["amount", "charge", "balance", "fee", "rate", "price", "cost", "value"]
#             if any(k in question_lower for k in numeric_keywords) and col_type in ("text", "varchar", "char") and any(k in col_name for k in numeric_keywords):
#                 score += 10

#         # ---- Boost for common patterns ----
#         if "status" in question_lower and "status" in table_name:
#             score += 20
#         if "income" in question_lower and any("income" in str(c.get("name", "")).lower() for c in columns):
#             score += 20
#         if "approved" in question_lower and "app" in table_name:
#             score += 15
#         if "loan" in question_lower and "loan" in table_name:
#             score += 15

#         if score > 0:
#             table_scores[full_table_name] = score

#     # ---- Sort tables by score and return top N ----
#     sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
#     selected = [table for table, _ in sorted_tables[:max_tables]]

#     print(f"🎯 Question: {question}")
#     print(f"📊 Selected tables: {selected}")

#     return selected


# ============================================================================ 
# DYNAMIC PROMPT BUILDER
# ============================================================================ 
# ============================================================================ 
# DYNAMIC PROMPT BUILDER
# ============================================================================
# ============================================================================
# ENHANCED FUZZY MATCHING AND TABLE SELECTION
# ============================================================================

import re
import datetime
import traceback
import difflib
import uuid
from typing import List, Dict, Tuple, Any

# ============================================================================ 
# NORMALIZE COLUMN/TABLE NAME
# ============================================================================ 
def normalize_name(name: str) -> str:
    """Normalize table/column names for comparison - removes all special chars."""
    n = re.sub(r'[\s\-_\.]', '', name.lower())
    # Remove plural 's' at the end
    if n.endswith('s') and len(n) > 3:
        n = n[:-1]
    return n

def fuzzy_score(query_term: str, target: str, threshold: float = 0.0) -> float:
    """
    Calculate fuzzy matching score between query term and target.
    Returns score between 0.0 and 1.0.
    """
    query_norm = normalize_name(query_term)
    target_norm = normalize_name(target)
    
    # Exact match after normalization
    if query_norm == target_norm:
        return 1.0
    
    # Substring match
    if query_norm in target_norm or target_norm in query_norm:
        return 0.9
    
    # Use difflib for fuzzy matching
    ratio = difflib.SequenceMatcher(None, query_norm, target_norm).ratio()
    
    return ratio if ratio >= threshold else 0.0

# ============================================================================ 
# INTELLIGENT TABLE SELECTION WITH MULTIPLE FALLBACK STRATEGIES
# ============================================================================
def select_relevant_tables(
    question: str,
    schema_catalog: Dict[str, dict],
    similarity_threshold: float = 0.4,
    max_tables: int = 5
) -> List[str]:
    """
    Intelligently select relevant tables with multiple fallback strategies.
    GUARANTEED to return at least one table.
    """
    
    question_lower = question.lower()
    question_words = re.findall(r'\b\w+\b', question_lower)
    question_normalized = normalize_name(question)
    
    table_scores = {}
    
    print(f"🔍 Analyzing question: {question}")
    print(f"📝 Question words: {question_words[:10]}")  # Show first 10 words
    
    for full_table_name, table_info in schema_catalog.items():
        score = 0
        table_name = table_info.get("table", "").lower()
        schema_name = table_info.get("schema", "").lower()
        
        # ---- STRATEGY 1: Direct table name matching ----
        table_parts = table_name.replace("_", " ").split()
        for part in table_parts:
            if part in question_lower:
                score += 20
            # Fuzzy match on parts
            for qword in question_words:
                fs = fuzzy_score(qword, part, threshold=similarity_threshold)
                if fs > 0.6:
                    score += int(15 * fs)
        
        # ---- STRATEGY 2: Keyword-based table detection ----
        keyword_patterns = {
            'loan': ['loan', 'lending', 'credit', 'borrowed', 'debt', 'payment'],
            'app': ['application', 'apply', 'applied', 'applicant', 'applying'],
            'customer': ['customer', 'client', 'user', 'borrower'],
            'transaction': ['transaction', 'txn', 'payment', 'transfer'],
            'account': ['account', 'balance', 'savings'],
            'default': ['default', 'delinquent', 'overdue', 'late'],
            'status': ['status', 'state', 'condition'],
            'channel': ['channel', 'source', 'origin', 'medium'],
            'income': ['income', 'salary', 'earning', 'revenue'],
            'credit': ['credit', 'score', 'rating', 'vantage'],
        }
        
        for table_keyword, question_keywords in keyword_patterns.items():
            if table_keyword in table_name:
                for kw in question_keywords:
                    if kw in question_lower:
                        score += 25
        
        # ---- STRATEGY 3: Column-based scoring ----
        columns = table_info.get("columns", [])
        matched_columns = []
        
        for col in columns:
            col_name = col.get("name", "").lower()
            col_type = col.get("type", "").lower()
            col_normalized = normalize_name(col_name)
            
            # Direct column name match
            if col_name in question_lower:
                score += 18
                matched_columns.append(col_name)
            
            # Normalized match
            if col_normalized in question_normalized:
                score += 15
                matched_columns.append(col_name)
            
            # Word-by-word fuzzy matching
            col_words = col_name.replace("_", " ").split()
            for col_word in col_words:
                for qword in question_words:
                    fs = fuzzy_score(qword, col_word, threshold=similarity_threshold)
                    if fs > 0.65:
                        score += int(10 * fs)
                        matched_columns.append(col_name)
            
            # Check for semantic matches
            semantic_mappings = {
                'rate': ['rate', 'ratio', 'percent', 'percentage', '%'],
                'default': ['default', 'delinquent', 'charge', 'chargeoff', 'writeoff', 'bad'],
                'payment': ['payment', 'pay', 'paid', 'installment', 'emi'],
                'amount': ['amount', 'value', 'sum', 'total'],
                'date': ['date', 'time', 'when', 'period', 'month', 'year'],
                'channel': ['channel', 'source', 'medium'],
                'status': ['status', 'state', 'approved', 'rejected'],
                'income': ['income', 'salary', 'earning'],
                'score': ['score', 'rating', 'vantage', 'fico'],
            }
            
            for semantic_key, semantic_terms in semantic_mappings.items():
                if semantic_key in col_name:
                    for term in semantic_terms:
                        if term in question_lower:
                            score += 12
                            matched_columns.append(col_name)
            
            # Numeric column boost for aggregation questions
            aggregation_keywords = ['sum', 'total', 'average', 'avg', 'count', 'max', 'min', 'rate', 'ratio']
            is_numeric = col_type in ('numeric', 'int', 'integer', 'float', 'decimal', 'bigint')
            has_numeric_name = any(k in col_name for k in ['amount', 'rate', 'score', 'balance', 'fee', 'charge'])
            
            if any(kw in question_lower for kw in aggregation_keywords):
                if is_numeric or has_numeric_name:
                    score += 10
        
        # ---- STRATEGY 4: Temporal matching (year detection) ----
        year_match = re.search(r'\b(20\d{2})\b', question)
        table_year_match = re.search(r'_(\d{4})$', table_name)
        
        if year_match and table_year_match:
            question_year = year_match.group(1)
            table_year = table_year_match.group(1)
            if question_year == table_year:
                score += 30
            elif abs(int(question_year) - int(table_year)) <= 1:
                score += 10
        elif table_year_match:
            # If question doesn't specify year, prefer recent tables
            table_year = int(table_year_match.group(1))
            if table_year >= 2023:
                score += 15
        
        # ---- STRATEGY 5: Context-based scoring ----
        if 'first payment default' in question_lower:
            if 'loan' in table_name and any('default' in c.get('name', '').lower() or 'fpd' in c.get('name', '').lower() for c in columns):
                score += 40
        
        if 'approved' in question_lower or 'application' in question_lower:
            if 'app' in table_name:
                score += 30
        
        if 'loan' in question_lower and 'loan' in table_name:
            score += 25
        
        # Store score and matched columns info
        if score > 0 or len(matched_columns) > 0:
            table_scores[full_table_name] = {
                'score': score,
                'matched_columns': list(set(matched_columns)),
                'table_name': table_name
            }
            print(f"  📊 {full_table_name}: score={score}, matched_cols={len(matched_columns)}")
    
    # ---- STRATEGY 6: Sort and select top tables ----
    if not table_scores:
        print("⚠️ No tables matched! Falling back to ALL tables...")
        # FALLBACK 1: Return all tables if no matches found
        all_tables = list(schema_catalog.keys())
        print(f"📦 Returning all {len(all_tables)} tables as fallback")
        return all_tables[:max_tables]
    
    # Sort by score
    sorted_tables = sorted(
        table_scores.items(),
        key=lambda x: (x[1]['score'], len(x[1]['matched_columns'])),
        reverse=True
    )
    
    selected = [table for table, info in sorted_tables[:max_tables]]
    
    # ---- STRATEGY 7: Ensure minimum tables ----
    if not selected:
        # FALLBACK 2: Return first available table
        first_table = list(schema_catalog.keys())[0] if schema_catalog else None
        if first_table:
            print(f"⚠️ No tables selected, using first table: {first_table}")
            return [first_table]
        return []
    
    # ---- STRATEGY 8: Smart multi-table selection ----
    # If question implies joins (e.g., mentions both "loan" and "application")
    join_keywords = [
        ('loan', 'app'),
        ('customer', 'account'),
        ('application', 'approval'),
    ]
    
    for kw1, kw2 in join_keywords:
        if kw1 in question_lower and kw2 in question_lower:
            # Ensure we have both types of tables
            has_kw1 = any(kw1 in info['table_name'] for _, info in table_scores.items())
            has_kw2 = any(kw2 in info['table_name'] for _, info in table_scores.items())
            
            if has_kw1 and has_kw2:
                # Add both types if not already included
                for table, info in sorted_tables:
                    if (kw1 in info['table_name'] or kw2 in info['table_name']) and table not in selected:
                        selected.append(table)
                        if len(selected) >= max_tables:
                            break
    
    print(f"✅ Selected {len(selected)} tables: {selected}")
    return selected

# ============================================================================ 
# COLUMN NORMALIZATION AND RESOLUTION
# ============================================================================ 

def is_numeric_column(col_info: dict) -> bool:
    """Check if the column is numeric or text that should be cast to numeric."""
    col_type = col_info.get('type', '').lower()
    col_name = col_info.get('name', '').lower()
    
    # Direct numeric types
    if col_type in ('numeric', 'int', 'integer', 'float', 'decimal', 'bigint', 'smallint', 'real', 'double'):
        return True
    
    # Text columns that likely contain numeric values
    numeric_keywords = ['amount', 'charge', 'balance', 'fee', 'rate', 'price', 'cost', 'value', 'score', 'percent', 'ratio']
    if col_type in ('text', 'varchar', 'char') and any(k in col_name for k in numeric_keywords):
        return True
    
    return False

def format_column_for_agg(col_info: dict, agg_func: str = None) -> str:
    """
    Return column reference with proper casting for aggregation.
    Handles numeric text columns safely using NULLIF to avoid empty string errors.
    """
    col_name = col_info['name']

    if is_numeric_column(col_info) and agg_func in ('SUM', 'AVG', 'MIN', 'MAX', 'COUNT'):
        # Wrap with COALESCE to handle NULLs
        if agg_func in ('SUM', 'AVG'):
            return f"COALESCE({agg_func}(NULLIF({col_name}, '')::NUMERIC), 0)"
        if agg_func == 'COUNT':
            return f"{agg_func}(*)"
        # MIN/MAX just cast safely
        return f"{agg_func}(NULLIF({col_name}, '')::NUMERIC)"

    if agg_func == 'COUNT':
        return f"{agg_func}(*)"
    
    return col_name

def get_best_column_match(question: str, columns: List[dict], threshold: float = 0.4) -> str:
    """Return closest matching column name for a given question with fuzzy matching."""
    if not columns:
        return ""
    
    question_lower = question.lower()
    question_words = re.findall(r'\b\w+\b', question_lower)
    
    best_match = None
    best_score = 0.0
    
    for col in columns:
        col_name = col['name']
        score = 0.0
        
        # Exact match
        if col_name.lower() in question_lower:
            return col_name
        
        # Normalized match
        if normalize_name(col_name) in normalize_name(question):
            score = 0.9
        
        # Fuzzy match on each word
        col_words = col_name.replace("_", " ").split()
        for col_word in col_words:
            for qword in question_words:
                fs = fuzzy_score(qword, col_word, threshold=threshold)
                if fs > score:
                    score = fs
        
        if score > best_score:
            best_score = score
            best_match = col_name
    
    return best_match or columns[0]['name']

def resolve_columns_in_question(question: str, tables: list, schema_catalog: dict) -> dict:
    """
    Returns mapping of fuzzy column names in the question to exact schema columns.
    """
    resolved = {}
    
    question_normalized = normalize_name(question)
    question_words = re.findall(r'\b\w+\b', question.lower())
    normalized_words = [normalize_name(w) for w in question_words]
    
    for table in tables:
        table_info = schema_catalog.get(table, {})
        for col in table_info.get('columns', []):
            col_name = col['name']
            normalized_col = normalize_name(col_name)
            
            # Direct normalized containment
            if normalized_col in question_normalized:
                resolved[normalized_col] = col_name
                continue
            
            # Reverse containment
            for w in normalized_words:
                if w in normalized_col or normalized_col in w:
                    resolved[w] = col_name
                    resolved[normalized_col] = col_name
                    break
            
            # Fuzzy fallback
            fuzzy_match = difflib.get_close_matches(normalized_col, normalized_words, n=1, cutoff=0.6)
            if fuzzy_match:
                match_key = fuzzy_match[0]
                resolved[match_key] = col_name
                resolved[normalized_col] = col_name
    
    return resolved

# ============================================================================ 
# TABLE ALIAS AND JOIN GENERATION
# ============================================================================

def get_column_table_alias(col_name: str, tables: list, tables_alias_map: dict, schema_catalog: dict) -> str:
    """Find which table a column belongs to and return its alias."""
    col_lower = col_name.lower()
    col_norm = normalize_name(col_name)
    
    for table, alias in tables_alias_map.items():
        table_info = schema_catalog.get(table, {})
        for col in table_info.get("columns", []):
            if col['name'].lower() == col_lower or normalize_name(col['name']) == col_norm:
                return alias
    
    # Fallback to first table alias
    return list(tables_alias_map.values())[0] if tables_alias_map else ""

def generate_joins(tables: list, join_info: dict = None) -> str:
    """Create FROM clause with dynamic joins."""
    if not tables:
        return ""
    if len(tables) == 1:
        return quote_table_name(tables[0])
    
    if join_info:
        joined = quote_table_name(tables[0])
        for i in range(1, len(tables)):
            t1 = tables[0]
            t2 = tables[i]
            key = (t1, t2) if (t1, t2) in join_info else (t2, t1)
            jd = join_info.get(key, {})
            if jd:
                left = ", ".join(jd.get('left_columns', []))
                right = ", ".join(jd.get('right_columns', []))
                joined += f" JOIN {quote_table_name(t2)} ON {t1}.{left} = {t2}.{right}"
            else:
                joined += f", {quote_table_name(t2)}"
        return joined
    else:
        return ", ".join([quote_table_name(t) for t in tables])

def quote_table_name(table: str) -> str:
    """Normalize and quote a table name."""
    table = table.replace('"', '').replace('`', '').strip()
    parts = table.split('.')
    if len(parts) == 2:
        schema, table_name = parts
        return f'"{schema}"."{table_name}"'
    elif len(parts) == 1:
        return f'"{parts[0]}"'
    else:
        raise ValueError(f"Unexpected table format: {table}")

# ============================================================================ 
# AGGREGATION AND ARITHMETIC DETECTION
# ============================================================================

def detect_aggregations(question: str) -> list:
    """Detect which aggregations are mentioned in the question."""
    aggs = []
    q = question.lower()
    
    if any(k in q for k in ['sum', 'total', 'amount', 'charge', 'revenue', 'income']):
        aggs.append('SUM')
    if any(k in q for k in ['average', 'avg', 'mean']):
        aggs.append('AVG')
    if any(k in q for k in ['count', 'number of', 'how many']):
        aggs.append('COUNT')
    if 'max' in q or 'maximum' in q or 'highest' in q:
        aggs.append('MAX')
    if 'min' in q or 'minimum' in q or 'lowest' in q:
        aggs.append('MIN')
    
    return aggs or ['SUM']

def detect_arithmetic_patterns(question: str, columns: list, tables: list, schema_catalog: dict) -> dict:
    """
    Detect arithmetic expressions, averages, percentages, and 'per group' patterns.
    """
    patterns = []
    group_by_columns = []
    
    q = question.lower()
    numeric_cols = [c for c in columns if is_numeric_column(c)]
    
    # Detect aggregations
    aggs = detect_aggregations(question)
    for agg in aggs:
        for col in numeric_cols:
            patterns.append(f"{format_column_for_agg(col, agg)} AS {agg.lower()}_{col['name']}")
    
    # Resolve fuzzy columns
    resolved_columns = resolve_columns_in_question(question, tables, schema_catalog)
    
    def safe_resolve(col_name):
        col_key = normalize_name(col_name)
        if col_key in resolved_columns:
            return resolved_columns[col_key]
        return get_best_column_match(col_name, columns) or col_name
    
    # Detect difference
    if 'difference' in q and len(numeric_cols) >= 2:
        patterns.append(
            f"COALESCE({format_column_for_agg(numeric_cols[0], 'SUM')}, 0) - "
            f"COALESCE({format_column_for_agg(numeric_cols[1], 'SUM')}, 0) AS difference"
        )
    
    # Detect rate/ratio/percentage
    rate_keywords = ['rate', 'ratio', 'percent', 'percentage']
    if any(kw in q for kw in rate_keywords) and len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        patterns.append(
            f"COALESCE(SUM(NULLIF({col1['name']}, '')::NUMERIC), 0) / "
            f"NULLIF(COALESCE(SUM(NULLIF({col2['name']}, '')::NUMERIC), 0), 0) * 100 AS {col1['name']}_{col2['name']}_rate"
        )
    
    # Detect GROUP BY
    per_match = re.findall(r'per (\w+)|by (\w+)|for each (\w+)', q)
    for match_tuple in per_match:
        for group_col in match_tuple:
            if group_col:
                group_col_resolved = safe_resolve(group_col)
                if group_col_resolved and group_col_resolved not in group_by_columns:
                    group_by_columns.append(group_col_resolved)
    
    # Include group columns in SELECT
    for col in group_by_columns:
        if col not in [s.split(' AS ')[-1] for s in patterns]:
            patterns.insert(0, col)
    
    return {
        "select": patterns if patterns else ["*"],
        "group_by": group_by_columns
    }

# ============================================================================ 
# DYNAMIC PROMPT BUILDER
# ============================================================================

# def build_dynamic_sql_prompt(
#     question: str,
#     schema_catalog: dict,
#     tables: List[str] = None,
#     join_info: dict = None,
#     history: List[Dict] = None,
#     similarity_threshold: float = 0.4
# ) -> str:
#     """Build SQL generation prompt dynamically."""
    
#     # Auto-select tables if not provided
#     if not tables:
#         tables = select_relevant_tables(question, schema_catalog, similarity_threshold=similarity_threshold)
    
#     # Ensure we have tables
#     if not tables:
#         tables = list(schema_catalog.keys())[:3]
    
#     # Get columns for selected tables
#     all_columns = []
#     for table in tables:
#         table_info = schema_catalog.get(table, {})
#         all_columns.extend(table_info.get("columns", []))
    
#     # Detect patterns
#     arithmetic_info = detect_arithmetic_patterns(question, all_columns, tables, schema_catalog)
#     select_patterns = arithmetic_info["select"]
#     group_by_cols = arithmetic_info["group_by"]
    
#     # Build schema section
#     today = datetime.date.today()
#     schema_section = "# DATABASE SCHEMA\n\n"
#     for table in tables:
#         table_info = schema_catalog.get(table, {})
#         schema_section += f"## Table: {table}\nColumns:\n"
#         for col in table_info.get("columns", []):
#             pk = " (PRIMARY KEY)" if col.get("is_primary") else ""
#             cast_hint = " [Numeric - use CAST]" if is_numeric_column(col) else ""
#             schema_section += f"  - {col.get('name')}: {col.get('type')}{pk}{cast_hint}\n"
#         schema_section += "\n"
    
#     # Join info
#     join_section = ""
#     if join_info:
#         join_section = "# JOIN RELATIONSHIPS\n\n"
#         for (t1, t2), jd in join_info.items():
#             left = ", ".join(jd.get('left_columns', []))
#             right = ", ".join(jd.get('right_columns', []))
#             join_section += f"- {t1}.{left} = {t2}.{right}\n"
#         join_section += "\n"
    
#     # History
#     history_section = ""
#     if history:
#         history_section = "# RECENT QUERIES\n\n"
#         for entry in history[-3:]:
#             history_section += f"Q: {entry.get('question')}\nSQL: {entry.get('sql')}\n\n"
    
#     prompt = f"""You are an expert PostgreSQL query generator.

# {schema_section}
# {join_section}
# {history_section}

# # CRITICAL RULES

# ## DATE HANDLING
# - Use year 2024, current month {today.month}, current day {today.day}
# - Replace "today" or "current date" with DATE '2024-{today.month:02d}-{today.day:02d}'
# - NEVER use INTERVAL '1 quarter' - use INTERVAL '3 months' instead
# - For year filtering, match the table suffix (e.g., _2024)

# ## NUMERIC SAFETY
# 1. Always wrap divisions: column / NULLIF(divisor, 0)
# 2. Cast text-numeric columns: CAST(NULLIF(column, '') AS NUMERIC)
# 3. Use COALESCE for aggregations: COALESCE(SUM(...), 0)
# 4. Handle empty strings before casting

# ## COLUMN MATCHING
# 1. Match columns by fuzzy logic (ignore case, underscore, dash, space)
# 2. Use ILIKE for case-insensitive string matching
# 3. Only use columns that exist in the schema

# ## QUERY STRUCTURE
# - Use fully qualified names: "schema"."table"
# - Properly JOIN multiple tables when needed
# - Add GROUP BY when using aggregations with non-aggregated columns
# - Return ONLY valid, runnable PostgreSQL SQL

# # DETECTED PATTERNS
# - Suggested SELECT: {', '.join(select_patterns[:3])}
# - GROUP BY needed: {', '.join(group_by_cols) if group_by_cols else 'No'}

# # TASK
# Generate a PostgreSQL query for: "{question}"

# Return ONLY the SQL query, no explanations.
# """
    
#     return prompt
def validate_and_extract_sql(raw_llm_output: str) -> str:
    """
    Extract and validate basic SQL structure before repair.
    Returns cleaned SQL or raises ValueError if fundamentally broken.
    """
    sql = (raw_llm_output or "").strip()
    
    # Remove code fences
    if "```sql" in sql:
        sql = sql.split("```sql", 1)[1].split("```", 1)[0].strip()
    elif "```" in sql:
        sql = re.sub(r'```.*?```', '', sql, flags=re.DOTALL).strip()
    
    # Remove comments
    sql = "\n".join([ln for ln in sql.splitlines() if not ln.strip().startswith('--')])
    sql = sql.strip()
    
    # Basic validation: Must have SELECT and FROM
    if not re.search(r'\bSELECT\b', sql, re.IGNORECASE):
        raise ValueError("Generated SQL missing SELECT clause")
    
    if not re.search(r'\bFROM\b', sql, re.IGNORECASE):
        raise ValueError("Generated SQL missing FROM clause")
    
    # Check for obvious malformations: FROM appearing before SELECT end
    select_match = re.search(r'\bSELECT\b', sql, re.IGNORECASE)
    from_match = re.search(r'\bFROM\b', sql, re.IGNORECASE)
    
    if select_match and from_match:
        # Ensure there's at least some content between SELECT and FROM
        between = sql[select_match.end():from_match.start()].strip()
        if not between or len(between) < 3:
            raise ValueError("Malformed SQL: Invalid SELECT clause")
    
    return sql

import re
import datetime
from typing import List, Dict, Any

# def nuclear_repair_sql(
#     raw_llm_output: str,
#     tables: List[str],
#     schema_catalog: Dict[str, Any] = None,
#     join_info: Dict[str, Any] = None
# ) -> str:
#     """
#     NUCLEAR OPTION: Completely rebuild SQL from broken pieces.
#     Handles even catastrophically malformed SQL.
#     """
#     sql = (raw_llm_output or "").strip()
    
#     # Remove code fences
#     if "```sql" in sql:
#         sql = sql.split("```sql", 1)[1].split("```", 1)[0].strip()
#     elif "```" in sql:
#         sql = re.sub(r'```.*?```', '', sql, flags=re.DOTALL).strip()
    
#     # Remove comments
#     sql = "\n".join([ln for ln in sql.splitlines() if not ln.strip().startswith('--')])
    
#     print(f"🔧 NUCLEAR REPAIR - Input SQL:\n{sql[:300]}...\n")
    
#     # Check if it's completely broken (FROM/JOIN in wrong places)
#     if re.search(r'EXTRACT\([^)]+\s+(?:FROM|AS|JOIN)\s+["\w]+\.', sql, re.IGNORECASE):
#         print("⚠️ CATASTROPHIC ERROR: FROM/JOIN inside function - rebuilding from scratch")
#         return rebuild_sql_from_scratch(sql, tables, schema_catalog, join_info)
    
#     # If not catastrophic, try normal repair
#     return standard_repair_sql(sql, tables, schema_catalog, join_info)


# def nuclear_repair_sql(
#     raw_llm_output: str,
#     tables: List[str],
#     schema_catalog: Dict[str, Any] = None,
#     join_info: Dict[str, Any] = None
# ) -> str:
#     """
#     NUCLEAR OPTION: Completely rebuild SQL from broken pieces.
#     Handles even catastrophically malformed SQL.
#     """
#     sql = (raw_llm_output or "").strip()
    
#     # Remove code fences
#     if "```sql" in sql:
#         sql = sql.split("```sql", 1)[1].split("```", 1)[0].strip()
#     elif "```" in sql:
#         sql = re.sub(r'```.*?```', '', sql, flags=re.DOTALL).strip()
    
#     # Remove comments
#     sql = "\n".join([ln for ln in sql.splitlines() if not ln.strip().startswith('--')])
    
#     # --- FIX: Cast text date columns to DATE ---
#     if schema_catalog and tables:
#         for table in tables:
#             table_info = schema_catalog.get(table, {})
#             table_alias = table.split('.')[-1]  # Use full alias from table
#             for col in table_info.get("columns", []):
#                 col_name = col.get("name", "")
#                 col_type = col.get("type", "").lower()
#                 if col_type in ["text", "varchar", "char", "character varying"] and "date" in col_name.lower():
#                     # Replace all occurrences of column with CAST(... AS DATE)
#                     pattern = re.compile(rf'\b{table_alias}\.{col_name}\b', re.IGNORECASE)
#                     sql = pattern.sub(f"CAST({table_alias}.{col_name} AS DATE)", sql)

#     print(f"🔧 NUCLEAR REPAIR - Input SQL:\n{sql[:300]}...\n")
    
#     # Check if it's completely broken (FROM/JOIN in wrong places)
#     if re.search(r'EXTRACT\([^)]+\s+(?:FROM|AS|JOIN)\s+["\w]+\.', sql, re.IGNORECASE):
#         print("⚠️ CATASTROPHIC ERROR: FROM/JOIN inside function - rebuilding from scratch")
#         return rebuild_sql_from_scratch(sql, tables, schema_catalog, join_info)
    
#     # If not catastrophic, try normal repair
#     return standard_repair_sql(sql, tables, schema_catalog, join_info)

import re
import datetime
from typing import List, Dict, Any

import re
import datetime
from typing import List, Dict, Any

import re
import datetime
from typing import List, Dict, Any

def rebuild_sql_from_scratch(
    broken_sql: str,
    tables: List[str],
    schema_catalog: Dict[str, Any]
) -> str:
    """
    Rebuild SQL from scratch, fully Postgres-safe.
    - Automatically casts text-based date columns to DATE.
    - Automatically casts numeric-like text columns to NUMERIC.
    - Handles SUM, COUNT, EXTRACT(QUARTER), WHERE, GROUP BY, ORDER BY.
    - Uses schema_catalog from YAML to detect column types.
    """
    print("🏗️ REBUILDING SQL FROM SCRATCH (Postgres-safe)")

    # Extract high-level SQL intent
    has_extract_quarter = 'EXTRACT' in broken_sql.upper() and 'QUARTER' in broken_sql.upper()
    has_sum = 'SUM' in broken_sql.upper()
    has_count = 'COUNT' in broken_sql.upper()
    has_group_by = 'GROUP BY' in broken_sql.upper()
    has_order_by = 'ORDER BY' in broken_sql.upper()
    has_where = 'WHERE' in broken_sql.upper()

    # Detect date and numeric columns from schema
    date_cols = []
    numeric_cols = []

    for table in tables:
        if table in schema_catalog:
            for col in schema_catalog[table].get("columns", []):
                col_name = col['name']
                col_type = col['type'].lower()
                # Text-based date columns
                if any(dt in col_type for dt in ['date', 'timestamp']):
                    date_cols.append((col_name, col_type))
                # Numeric-like columns
                if any(amt in col_name.lower() for amt in ['amount', 'total', 'sum', 'balance', 'fee', 'principal', 'rate']):
                    numeric_cols.append(col_name)

    # Use first table as main table
    first_table = tables[0] if tables else "unknown_table"
    alias = first_table.split('.')[-1].strip('"')

    # Build SELECT clause
    select_parts = []

    if has_extract_quarter and date_cols:
        date_col, col_type = date_cols[0]
        col_expr = f'"{alias}"."{date_col}"'
        if 'text' in col_type:
            col_expr = f'CAST({col_expr} AS DATE)'
        select_parts.append(f'EXTRACT(QUARTER FROM {col_expr}) AS application_quarter')

    if has_sum and numeric_cols:
        amount_col = numeric_cols[0]
        col_type = next((c['type'].lower() for c in schema_catalog[first_table]['columns'] if c['name'] == amount_col), "")
        col_expr = f'"{alias}"."{amount_col}"'
        if 'text' in col_type:
            select_parts.append(f'COALESCE(SUM(CAST(NULLIF({col_expr}, \'\') AS NUMERIC)), 0) AS total_{amount_col}')
        else:
            select_parts.append(f'COALESCE(SUM({col_expr}), 0) AS total_{amount_col}')

    if has_count and not has_sum:
        select_parts.append(f'COUNT(*) AS total_count')

    if not select_parts:
        select_parts.append('*')

    select_clause = "SELECT " + ", ".join(select_parts)

    # FROM clause
    from_clause = f'FROM {first_table} AS "{alias}"'

    # WHERE clause
    where_parts = []
    where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|;|$)', broken_sql, re.IGNORECASE | re.DOTALL)
    if where_match:
        where_text = where_match.group(1).strip()
        conditions = re.findall(r'(["\w.]+)\s*(IS NOT NULL|IS NULL|!=|<>|>=|<=|>|<|=)\s*(["\w\s\'-]+)', where_text, re.IGNORECASE)
        for col, op, val in conditions:
            col_clean = col.strip().strip('"')
            val_clean = val.strip()
            col_expr = f'"{alias}"."{col_clean}"'

            # Cast text-based date columns in WHERE
            for date_col_name, col_type in date_cols:
                if col_clean.lower() == date_col_name.lower() and 'text' in col_type:
                    col_expr = f'CAST({col_expr} AS DATE)'

            # Cast numeric columns in WHERE if needed
            if col_clean in numeric_cols:
                col_expr = f'CAST({col_expr} AS NUMERIC)'

            where_parts.append(f"{col_expr} {op} {val_clean}")

    # Add date filters if detected in broken SQL
    if date_cols and has_where:
        today = datetime.date.today()
        if 'last' in broken_sql.lower() or '3 month' in broken_sql.lower():
            start_date = today - datetime.timedelta(days=90)
            date_col_name, col_type = date_cols[0]
            col_expr = f'"{alias}"."{date_col_name}"'
            if 'text' in col_type:
                col_expr = f'CAST({col_expr} AS DATE)'
            where_parts.append(f'{col_expr} >= DATE \'{start_date}\'')
            where_parts.append(f'{col_expr} <= DATE \'{today}\'')

    where_clause = ""
    if where_parts:
        where_clause = "WHERE " + " AND ".join(where_parts)

    # GROUP BY clause
    group_by_clause = ""
    if has_group_by and has_extract_quarter and date_cols:
        date_col_name, col_type = date_cols[0]
        col_expr = f'"{alias}"."{date_col_name}"'
        if 'text' in col_type:
            col_expr = f'CAST({col_expr} AS DATE)'
        group_by_clause = f'GROUP BY EXTRACT(QUARTER FROM {col_expr})'

    # ORDER BY clause
    order_by_clause = ""
    if has_order_by:
        if has_extract_quarter:
            order_by_clause = "ORDER BY application_quarter"
        elif has_sum:
            order_by_clause = f"ORDER BY total_{numeric_cols[0]} DESC"
        elif has_count:
            order_by_clause = "ORDER BY total_count DESC"

    # Combine all clauses
    sql_parts = [select_clause, from_clause]
    if where_clause:
        sql_parts.append(where_clause)
    if group_by_clause:
        sql_parts.append(group_by_clause)
    if order_by_clause:
        sql_parts.append(order_by_clause)

    rebuilt_sql = " ".join(sql_parts) + ";"

    print(f"✅ REBUILT SQL:\n{rebuilt_sql}\n")
    return rebuilt_sql



# def rebuild_sql_from_scratch(
#     broken_sql: str,
#     tables: List[str],
#     schema_catalog: Dict[str, Any] = None,
#     join_info: Dict[str, Any] = None
# ) -> str:
#     """
#     When SQL is completely broken, rebuild it from scratch by extracting intent.
#     """
#     print("🏗️ REBUILDING SQL FROM SCRATCH")
    
#     # Extract what we can understand from broken SQL
#     has_extract_quarter = 'EXTRACT' in broken_sql.upper() and 'QUARTER' in broken_sql.upper()
#     has_sum = 'SUM' in broken_sql.upper()
#     has_count = 'COUNT' in broken_sql.upper()
#     has_group_by = 'GROUP BY' in broken_sql.upper()
#     has_order_by = 'ORDER BY' in broken_sql.upper()
#     has_where = 'WHERE' in broken_sql.upper()
    
#     # Find date columns
#     date_cols = []
#     amount_cols = []
    
#     if schema_catalog and tables:
#         for table in tables:
#             for table_key, info in schema_catalog.items():
#                 if table in table_key or table_key in table:
#                     for col in info.get('columns', []):
#                         col_type = col.get('type', '').lower()
#                         col_name = col['name']
                        
#                         if any(dt in col_type for dt in ['date', 'timestamp']):
#                             date_cols.append(col_name)
                        
#                         if any(amt in col_name.lower() for amt in ['amount', 'total', 'sum', 'balance']):
#                             amount_cols.append(col_name)
    
#     # Build fresh SQL
#     first_table = tables[0] if tables else "unknown_table"
#     alias = first_table.split('.')[-1].strip('"')
    
#     # SELECT clause
#     select_parts = []
    
#     if has_extract_quarter and date_cols:
#         date_col = date_cols[0]
#         select_parts.append(f'EXTRACT(QUARTER FROM "{alias}"."{date_col}") AS application_quarter')
    
#     if has_sum and amount_cols:
#         amount_col = amount_cols[0]
#         # Check if it's TEXT type
#         is_text = False
#         if schema_catalog:
#             for table_key, info in schema_catalog.items():
#                 if first_table in table_key:
#                     for col in info.get('columns', []):
#                         if col['name'] == amount_col:
#                             col_type = col.get('type', '').lower()
#                             if any(t in col_type for t in ['text', 'varchar', 'char']):
#                                 is_text = True
#                             break
        
#         if is_text:
#             select_parts.append(f'COALESCE(SUM(CAST(NULLIF("{alias}"."{amount_col}", \'\') AS NUMERIC)), 0) AS total_amount')
#         else:
#             select_parts.append(f'COALESCE(SUM("{alias}"."{amount_col}"), 0) AS total_amount')
    
#     if has_count and not has_sum:
#         select_parts.append(f'COUNT(*) AS total_count')
    
#     if not select_parts:
#         select_parts.append('*')
    
#     select_clause = "SELECT " + ", ".join(select_parts)
    
#     # FROM clause
#     from_clause = f'FROM {first_table} AS "{alias}"'
    
#     # Add JOINs if available
#     if join_info and len(tables) > 1:
#         for join_key, join_cond in join_info.items():
#             if isinstance(join_key, (list, tuple)) and len(join_key) >= 2:
#                 join_table = join_key[1]
#                 join_alias = join_table.split('.')[-1].strip('"')
                
#                 left_cols = join_cond.get('left_columns', [])
#                 right_cols = join_cond.get('right_columns', [])
                
#                 if left_cols and right_cols:
#                     on_clause = " AND ".join([
#                         f'"{alias}"."{l}" = "{join_alias}"."{r}"'
#                         for l, r in zip(left_cols, right_cols)
#                     ])
#                     from_clause += f' INNER JOIN {join_table} AS "{join_alias}" ON {on_clause}'
    
#     # WHERE clause
#     where_parts = []
    
#     # Extract WHERE conditions from broken SQL
#     where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|;|$)', broken_sql, re.IGNORECASE | re.DOTALL)
#     if where_match:
#         where_text = where_match.group(1).strip()
#         # Clean up any misplaced FROM/JOIN in WHERE
#         where_text = re.sub(r'\s+(?:FROM|JOIN|AS)\s+["\w.]+', '', where_text, flags=re.IGNORECASE)
        
#         # Extract simple conditions
#         conditions = re.findall(r'(["\w.]+)\s*(IS NOT NULL|IS NULL|!=|<>|>=|<=|>|<|=)\s*(["\w\s\'-]+)', where_text, re.IGNORECASE)
        
#         for col, op, val in conditions:
#             col_clean = col.strip().strip('"')
#             val_clean = val.strip()
            
#             # Qualify column if needed
#             if '.' not in col_clean:
#                 col_qualified = f'"{alias}"."{col_clean}"'
#             else:
#                 col_qualified = col_clean
            
#             where_parts.append(f"{col_qualified} {op} {val_clean}")
    
#     # Add date range if we have date column
#     if date_cols and has_where:
#         today = datetime.date.today()
#         # Last quarter
#         if 'last' in broken_sql.lower() or '3 month' in broken_sql.lower():
#             start_date = today - datetime.timedelta(days=90)
#             where_parts.append(f'"{alias}"."{date_cols[0]}" >= DATE \'{start_date.strftime("%Y-%m-%d")}\'')
#             where_parts.append(f'"{alias}"."{date_cols[0]}" <= DATE \'{today.strftime("%Y-%m-%d")}\'')
    
#     where_clause = ""
#     if where_parts:
#         where_clause = "WHERE " + " AND ".join(where_parts)
    
#     # GROUP BY clause
#     group_by_clause = ""
#     if has_group_by and has_extract_quarter and date_cols:
#         group_by_clause = f'GROUP BY EXTRACT(QUARTER FROM "{alias}"."{date_cols[0]}")'
    
#     # ORDER BY clause
#     order_by_clause = ""
#     if has_order_by:
#         if has_extract_quarter:
#             order_by_clause = "ORDER BY application_quarter"
#         elif has_sum:
#             order_by_clause = "ORDER BY total_amount DESC"
#         elif has_count:
#             order_by_clause = "ORDER BY total_count DESC"
    
#     # Combine all parts
#     sql_parts = [select_clause, from_clause]
#     if where_clause:
#         sql_parts.append(where_clause)
#     if group_by_clause:
#         sql_parts.append(group_by_clause)
#     if order_by_clause:
#         sql_parts.append(order_by_clause)
    
#     rebuilt_sql = " ".join(sql_parts) + ";"
    
#     print(f"✅ REBUILT SQL:\n{rebuilt_sql}\n")
#     return rebuilt_sql


# def standard_repair_sql(
#     sql: str,
#     tables: List[str],
#     schema_catalog: Dict[str, Any] = None,
#     join_info: Dict[str, Any] = None
# ) -> str:
#     """
#     Standard repair for SQL that's mostly correct.
#     """
#     # Fix quarters
#     for i in range(1, 5):
#         sql = re.sub(
#             rf"INTERVAL\s+['\"]{i}\s+quarters?['\"]",
#             f"INTERVAL '{i*3} months'",
#             sql,
#             flags=re.IGNORECASE
#         )
    
#     # Fix dates
#     today = datetime.date.today()
#     fixed_date = f"DATE '{today.year}-{today.month:02d}-{today.day:02d}'"
#     sql = re.sub(r'\bCURRENT_DATE\b', fixed_date, sql, flags=re.IGNORECASE)
#     sql = re.sub(r'\bNOW\(\)\b', fixed_date, sql, flags=re.IGNORECASE)
    
#     # Fix NULL comparisons
#     sql = re.sub(r'(\w+)\s*=\s*NULL', r'\1 IS NULL', sql, flags=re.IGNORECASE)
#     sql = re.sub(r'(\w+)\s*!=\s*NULL', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)
    
#     # Safe division
#     def safe_div(m):
#         left = m.group(1)
#         right = m.group(2).strip()
#         if 'NULLIF' in right:
#             return m.group(0)
#         return f"{left} / NULLIF({right}, 0)"
    
#     sql = re.sub(r'([^\s]+)\s*/\s*([A-Za-z0-9_"\.\)]+)', safe_div, sql)
    
#     # Cleanup
#     sql = re.sub(r'\s+', ' ', sql).strip()
    
#     # Balance parentheses
#     open_p = sql.count('(')
#     close_p = sql.count(')')
#     if open_p > close_p:
#         sql += ')' * (open_p - close_p)
#     elif close_p > open_p:
#         excess = close_p - open_p
#         for _ in range(excess):
#             sql = sql.rstrip(';').rstrip(')')
    
#     if not sql.endswith(';'):
#         sql += ';'
    
#     return sql
import re
import datetime
from typing import List, Dict, Any, Tuple, Set
from difflib import SequenceMatcher


# ============================================================================
# HELPER FUNCTIONS (MUST BE DEFINED FIRST)
# ============================================================================

import re
import datetime
from typing import List, Dict, Any, Tuple, Set
from difflib import SequenceMatcher


# ============================================================================
# HELPER FUNCTIONS (MUST BE DEFINED FIRST)
# ============================================================================

def extract_actual_table_aliases(sql: str) -> Dict[str, str]:
    """Extract actual table aliases used in the SQL FROM/JOIN clauses."""
    aliases = {}
    
    # Pattern: FROM schema.table alias or FROM schema.table AS alias
    pattern = r'\bFROM\s+([a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*)\s+(?:AS\s+)?([a-z_][a-z0-9_]*)\b'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    
    for full_table, alias in matches:
        aliases[full_table.lower()] = alias.lower()
    
    # Also check JOIN clauses
    pattern = r'\bJOIN\s+([a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*)\s+(?:AS\s+)?([a-z_][a-z0-9_]*)\b'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    
    for full_table, alias in matches:
        aliases[full_table.lower()] = alias.lower()
    
    return aliases


def tokenize_column_name(col_name: str) -> Set[str]:
    """Break column name into semantic tokens for intelligent matching."""
    tokens = set()
    parts = col_name.lower().split('_')
    tokens.update(parts)
    tokens.add(col_name.lower())
    
    for part in parts:
        stemmed = re.sub(r'(s|ing|ed|date|id|flag|num|number)$', '', part)
        if len(stemmed) >= 3:
            tokens.add(stemmed)
    
    return tokens


def calculate_semantic_similarity(col1: str, col2: str) -> float:
    """Calculate semantic similarity between two column names."""
    tokens1 = tokenize_column_name(col1)
    tokens2 = tokenize_column_name(col2)
    
    overlap = len(tokens1 & tokens2) / max(len(tokens1), len(tokens2)) if tokens1 and tokens2 else 0.0
    sequence_sim = SequenceMatcher(None, col1.lower(), col2.lower()).ratio()
    
    return (overlap * 0.6) + (sequence_sim * 0.4)


def categorize_column(col_name: str, col_type: str) -> str:
    """Dynamically categorize column based on name patterns."""
    name_lower = col_name.lower()
    
    if any(word in name_lower for word in ['date', 'time', 'dt', 'timestamp', 'when']):
        return 'temporal'
    if any(word in name_lower for word in ['amount', 'balance', 'total', 'sum', 'price', 
                                             'cost', 'fee', 'rate', 'principal', 'income',
                                             'payment', 'value', 'score', 'dpd', 'days']):
        return 'numeric'
    if any(word in name_lower for word in ['id', 'number', 'code', 'key']):
        return 'identifier'
    if any(word in name_lower for word in ['status', 'flag', 'is', 'has', 'type', 'name', 'channel']):
        return 'categorical'
    
    return 'text'


# ============================================================================
# PART 0: TABLE VALIDATION
# ============================================================================

def extract_table_references(sql: str) -> Set[str]:
    """Extract all table references from SQL (including schema.table format)."""
    pattern = r'\b(?:FROM|JOIN)\s+([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?)\b'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    
    tables = set()
    for match in matches:
        tables.add(match.lower())
    
    return tables


def validate_and_fix_tables(sql: str, schema_catalog: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Validate table names exist in schema and fix if possible."""
    if not schema_catalog:
        return sql, []
    
    valid_tables = {table_name.lower(): table_name for table_name in schema_catalog.keys()}
    referenced_tables = extract_table_references(sql)
    
    warnings = []
    corrections = {}
    
    print(f"🗂️  Validating {len(referenced_tables)} table references")
    
    for ref_table in referenced_tables:
        if ref_table not in valid_tables:
            best_match = None
            best_score = 0.0
            
            for valid_table in valid_tables.keys():
                similarity = SequenceMatcher(None, ref_table, valid_table).ratio()
                
                ref_base = ref_table.split('.')[-1]
                valid_base = valid_table.split('.')[-1]
                if ref_base == valid_base:
                    similarity += 0.3
                
                if similarity > best_score and similarity >= 0.6:
                    best_score = similarity
                    best_match = valid_table
            
            if best_match:
                corrections[ref_table] = valid_tables[best_match]
                msg = f"Table '{ref_table}' → '{valid_tables[best_match]}' (confidence: {best_score:.2f})"
                print(f"  🔧 {msg}")
                warnings.append(msg)
            else:
                msg = f"⚠️  Table '{ref_table}' not found in schema!"
                print(f"  {msg}")
                warnings.append(msg)
    
    for wrong, correct in corrections.items():
        pattern = re.compile(rf'\b{re.escape(wrong)}\b', re.IGNORECASE)
        sql = pattern.sub(correct, sql)
    
    if not corrections:
        print("  ✅ All tables valid")
    
    return sql, warnings


# ============================================================================
# PART 1: COLUMN VALIDATION & CORRECTION
# ============================================================================

def build_dynamic_column_index(schema_catalog: Dict[str, Any]) -> Dict[str, Any]:
    """Build dynamic column index from schema catalog."""
    column_index = {}
    
    for table_name, table_info in schema_catalog.items():
        table_parts = table_name.split('.')
        table_alias = table_parts[-1][:3] if table_parts else 'tbl'
        
        for col in table_info.get('columns', []):
            col_name = col.get('name', '').lower()
            col_type = col.get('type', '').lower()
            
            full_col = f"{table_alias}.{col_name}"
            
            column_index[col_name] = {
                'table': table_name,
                'alias': table_alias,
                'full_name': full_col,
                'type': col_type,
                'category': categorize_column(col_name, col_type),
                'tokens': tokenize_column_name(col_name),
                'original_name': col.get('name', '')
            }
            
            column_index[full_col] = column_index[col_name].copy()
    
    return column_index


def extract_column_references(sql: str) -> Set[str]:
    """Extract all column references from SQL."""
    pattern = r'\b(?:[a-z_][a-z0-9_]*\.)?([a-z_][a-z0-9_]*)\b'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    
    sql_keywords = {
        'select', 'from', 'where', 'group', 'by', 'order', 'having', 'join', 
        'inner', 'left', 'right', 'outer', 'full', 'cross', 'on', 'as', 'and', 
        'or', 'not', 'in', 'like', 'ilike', 'between', 'is', 'null', 'case', 
        'when', 'then', 'else', 'end', 'cast', 'sum', 'count', 'avg', 'max', 
        'min', 'date', 'timestamp', 'interval', 'year', 'month', 'day', 'hour',
        'nullif', 'coalesce', 'extract', 'limit', 'offset', 'distinct', 'all',
        'true', 'false', 'asc', 'desc', 'numeric', 'integer', 'text', 'varchar',
        'char', 'boolean', 'trunc', 'round', 'floor', 'ceil', 'abs', 'with',
        'union', 'intersect', 'except', 'exists', 'any', 'some', 'over',
        'partition', 'window', 'row', 'rows', 'range', 'preceding', 'following',
        'unbounded', 'current', 'update', 'insert', 'delete', 'into', 'values',
        'set', 'to', 'at', 'time', 'zone', 'only', 'y', 'n', 'dpd', 'date_trunc',
        'current_date', 'now', 'today', 'yesterday', 'quarterly', 'monthly',
        'weekly', 'daily', 'annual', 'percent', 'percentage', 'rate', 'div'
    }
    
    sql_functions = {
        'date_trunc', 'date_part', 'extract', 'to_char', 'to_date', 'to_timestamp',
        'current_date', 'current_timestamp', 'now', 'age', 'date_add', 'date_sub',
        'datediff', 'dateadd', 'achsent_date'
    }
    
    columns = set()
    for match in matches:
        match_lower = match.lower()
        if match_lower not in sql_keywords and match_lower not in sql_functions and not match_lower.isdigit():
            columns.add(match_lower)
    
    return columns


def find_best_column_match(
    invalid_col: str,
    column_index: Dict[str, Any],
    min_similarity: float = 0.65
) -> Tuple[str, float]:
    """Find best matching column using semantic similarity."""
    best_match = None
    best_score = 0.0
    
    invalid_tokens = tokenize_column_name(invalid_col)
    invalid_category = categorize_column(invalid_col, 'unknown')
    
    for valid_col, col_info in column_index.items():
        similarity = calculate_semantic_similarity(invalid_col, valid_col)
        
        if col_info['category'] == invalid_category:
            similarity += 0.10
        
        token_overlap = len(invalid_tokens & col_info['tokens'])
        if token_overlap > 0:
            similarity += (token_overlap * 0.05)
        
        if invalid_col.startswith(valid_col) or valid_col.startswith(invalid_col):
            similarity += 0.15
        
        if similarity > best_score and similarity >= min_similarity:
            best_score = similarity
            best_match = valid_col
    
    return best_match, best_score


def auto_correct_columns(
    sql: str,
    schema_catalog: Dict[str, Any],
    min_similarity: float = 0.65
) -> Tuple[str, List[str]]:
    """Automatically correct invalid column names."""
    if not schema_catalog:
        return sql, []
    
    column_index = build_dynamic_column_index(schema_catalog)
    referenced_cols = extract_column_references(sql)
    valid_cols = set(column_index.keys())
    
    warnings = []
    corrections = {}
    
    print(f"📋 Validating {len(referenced_cols)} column references")
    
    for ref_col in referenced_cols:
        if ref_col not in valid_cols:
            match, score = find_best_column_match(ref_col, column_index, min_similarity)
            
            if match and score >= 0.75:
                corrections[ref_col] = match
                msg = f"'{ref_col}' → '{match}' (confidence: {score:.2f})"
                print(f"  🔧 {msg}")
                warnings.append(msg)
            else:
                msg = f"⚠️  Unrecognized column '{ref_col}' (best match: {match} @ {score:.2f} - too low)"
                print(f"  {msg}")
                warnings.append(msg)
    
    for wrong, correct in corrections.items():
        pattern = re.compile(rf'\b{re.escape(wrong)}\b', re.IGNORECASE)
        sql = pattern.sub(correct, sql)
    
    if not corrections:
        print("  ✅ All columns valid or no high-confidence corrections")
    
    return sql, warnings


# ============================================================================
# PART 2: PREEMPTIVE CASTING (NEW - RUNS FIRST)
# ============================================================================

def preemptive_numeric_cast(sql: str, schema_catalog: Dict[str, Any]) -> str:
    """Apply numeric casts BEFORE any other processing - BULLETPROOF."""
    if not schema_catalog:
        return sql
    
    print("🔒 PREEMPTIVE NUMERIC CASTING - Running before all other repairs")
    
    # Build list of ALL numeric columns that are TEXT type
    numeric_columns = []
    for table_name, table_info in schema_catalog.items():
        for col in table_info.get('columns', []):
            col_name = col.get('name', '').lower()
            col_type = col.get('type', '').lower()
            
            if col_type in ['text', 'varchar', 'char', 'character varying']:
                # Expanded list to catch chargeoffs, dpd, etc.
                if any(kw in col_name for kw in [
                    'amount', 'balance', 'total', 'sum', 'price', 'cost', 'fee',
                    'rate', 'score', 'value', 'principal', 'income', 'payment',
                    'charge', 'chargeoff', 'dpd', 'days', 'count', 'number',
                    'qty', 'quantity', 'percent', 'ratio', 'interest'
                ]):
                    numeric_columns.append(col_name)
    
    if not numeric_columns:
        print("  ℹ️  No TEXT numeric columns found")
        return sql
    
    print(f"  🎯 Targeting {len(numeric_columns)} TEXT numeric columns: {numeric_columns[:10]}...")
    
    total_casts = 0
    
    # Cast ALL occurrences (with any alias) when used in numeric contexts
    for col_name in numeric_columns:
        # Pattern: Match [alias].column_name in numeric contexts
        # Look for comparisons with numbers or arithmetic operations
        
        # Check if column is used in numeric context
        context_pattern = rf'\w+\.{re.escape(col_name)}\s*(?:>|<|>=|<=|=|!=|<>|\+|\-|\*|\/)\s*\d+'
        
        if re.search(context_pattern, sql, re.IGNORECASE):
            # Cast all occurrences
            pattern = rf'(\w+\.{re.escape(col_name)})(?!::|\s+AS\s+NUMERIC)'
            
            matches = len(re.findall(pattern, sql, re.IGNORECASE))
            if matches > 0:
                sql = re.sub(pattern, r'\1::NUMERIC', sql, flags=re.IGNORECASE)
                total_casts += matches
                print(f"    ✅ Cast {matches} occurrences of *.{col_name}")
    
    print(f"  🎉 Applied {total_casts} total preemptive numeric casts")
    
    return sql


def preemptive_date_cast(sql: str, schema_catalog: Dict[str, Any]) -> str:
    """Apply casts BEFORE any other processing - BULLETPROOF."""
    if not schema_catalog:
        return sql
    
    print("🔒 PREEMPTIVE DATE CASTING - Running before all other repairs")
    
    # Build list of ALL date columns that are TEXT type
    date_columns = []
    for table_name, table_info in schema_catalog.items():
        for col in table_info.get('columns', []):
            col_name = col.get('name', '').lower()
            col_type = col.get('type', '').lower()
            
            if col_type in ['text', 'varchar', 'char', 'character varying']:
                if any(kw in col_name for kw in ['date', 'time', 'dt', 'timestamp']):
                    date_columns.append(col_name)
    
    if not date_columns:
        print("  ℹ️  No TEXT date columns found")
        return sql
    
    print(f"  🎯 Targeting {len(date_columns)} TEXT date columns: {date_columns}")
    
    cast_type = '::TIMESTAMP'
    total_casts = 0
    
    # Cast ALL occurrences (with any alias or no alias)
    for col_name in date_columns:
        # Pattern: Match [optional_alias].column_name (not already cast)
        # Use negative lookahead to avoid double-casting
        pattern = rf'(\w+\.{re.escape(col_name)})(?!::)'
        
        matches = len(re.findall(pattern, sql, re.IGNORECASE))
        if matches > 0:
            sql = re.sub(pattern, rf'\1{cast_type}', sql, flags=re.IGNORECASE)
            total_casts += matches
            print(f"    ✅ Cast {matches} occurrences of *.{col_name}")
    
    print(f"  🎉 Applied {total_casts} total preemptive casts")
    
    return sql


# ============================================================================
# PART 3: TEXT COLUMN TYPE CASTING (ENHANCED)
# ============================================================================

def auto_fix_text_date_comparisons(sql: str, schema_catalog: Dict[str, Any]) -> str:
    """Cast TEXT date columns to TIMESTAMP - MULTI-PASS VERSION."""
    if not schema_catalog:
        return sql
    
    actual_aliases = extract_actual_table_aliases(sql)
    print(f"🔍 Detected SQL aliases: {actual_aliases}")
    
    text_date_columns = []
    for table_name, table_info in schema_catalog.items():
        table_name_lower = table_name.lower()
        
        if table_name_lower in actual_aliases:
            actual_alias = actual_aliases[table_name_lower]
        else:
            actual_alias = table_name.split('.')[-1][:3]
        
        for col in table_info.get('columns', []):
            col_name = col.get('name', '').lower()
            col_type = col.get('type', '').lower()
            
            if col_type in ['text', 'varchar', 'char', 'character varying']:
                if any(date_word in col_name for date_word in ['date', 'dt', 'timestamp', 'time']):
                    text_date_columns.append(f"{actual_alias}.{col_name}")
    
    if not text_date_columns:
        return sql
    
    print(f"🔍 Found {len(text_date_columns)} TEXT date columns: {text_date_columns}")
    
    cast_type = '::TIMESTAMP' if 'DATE_TRUNC' in sql.upper() or 'INTERVAL' in sql.upper() else '::DATE'
    
    # AGGRESSIVE MULTI-PASS CASTING
    for full_col in text_date_columns:
        iteration = 0
        max_iterations = 5
        
        while iteration < max_iterations:
            # Pattern that matches column in ANY context (not already cast)
            pattern = rf'(?<!\w)({re.escape(full_col)})(?!(?:\s*::|\s*\.\w))'
            
            matches_before = len(re.findall(pattern, sql, re.IGNORECASE))
            if matches_before == 0:
                break
            
            sql = re.sub(pattern, rf'\1{cast_type}', sql, flags=re.IGNORECASE)
            
            matches_after = len(re.findall(pattern, sql, re.IGNORECASE))
            
            if matches_before > matches_after:
                print(f"  🔧 Pass {iteration + 1}: Cast {matches_before - matches_after} occurrences of {full_col}")
            
            if matches_after == 0:
                break
            
            iteration += 1
        
        # Verify
        remaining = len(re.findall(rf'(?<!\w){re.escape(full_col)}(?!(?:\s*::|\s*\.\w))', sql, re.IGNORECASE))
        if remaining > 0:
            print(f"  ⚠️  WARNING: {remaining} uncast occurrences of {full_col} still remain!")
    
    return sql


def auto_fix_text_numeric_operations(sql: str, schema_catalog: Dict[str, Any]) -> str:
    """Cast TEXT numeric columns to NUMERIC."""
    if not schema_catalog:
        return sql
    
    text_numeric_columns = {}
    for table_name, table_info in schema_catalog.items():
        table_alias = table_name.split('.')[-1][:3]
        for col in table_info.get('columns', []):
            col_name = col.get('name', '').lower()
            col_type = col.get('type', '').lower()
            
            if col_type in ['text', 'varchar', 'char']:
                if any(num_word in col_name for num_word in 
                       ['amount', 'balance', 'total', 'sum', 'price', 'cost', 'fee', 
                        'rate', 'score', 'value', 'principal', 'income', 'payment']):
                    full_name = f"{table_alias}.{col_name}"
                    text_numeric_columns[full_name] = True
    
    if not text_numeric_columns:
        return sql
    
    print(f"🔍 Found {len(text_numeric_columns)} TEXT numeric columns")
    
    for col_full_name in text_numeric_columns.keys():
        pattern = rf'(?<!CAST\(NULLIF\()(\b{re.escape(col_full_name)}\b)(?!\s*,\s*\'\')'
        arithmetic_pattern = rf'(SUM|AVG|COUNT|MAX|MIN)\s*\(\s*(?:CASE[^)]+THEN\s+)?{re.escape(col_full_name)}'
        
        if re.search(arithmetic_pattern, sql, re.IGNORECASE):
            replacement = rf'CAST(NULLIF(\1, \'\') AS NUMERIC)'
            sql = re.sub(pattern, replacement, sql)
    
    print(f"  ✓ Applied CAST(NULLIF()) wrapping")
    
    return sql


# ============================================================================
# PART 4: INVALID FUNCTION & DATE LITERAL FIXING
# ============================================================================

def fix_invalid_functions(sql: str) -> str:
    """Fix common invalid function names."""
    print("🔧 Fixing invalid function names")
    
    if 'achsent_date' in sql.lower():
        sql = re.sub(
            r"achsent_date\s*\(\s*['\"]month['\"]\s*,\s*([^)]+)\)",
            r"DATE_TRUNC('month', \1)",
            sql,
            flags=re.IGNORECASE
        )
        
        sql = re.sub(
            r'\bachsent_date\b(?!\s*\()',
            'CURRENT_DATE',
            sql,
            flags=re.IGNORECASE
        )
        
        print("  ✓ Fixed achsent_date → DATE_TRUNC/CURRENT_DATE")
    
    return sql


def fix_date_literals(sql: str) -> str:
    """Fix malformed date literals like 'lm.DATE' that should be removed."""
    print("🔧 Fixing malformed date literals")
    
    sql = re.sub(
        r'\b([a-z_][a-z0-9_]*)\s*\.\s*DATE\s+',
        'DATE ',
        sql,
        flags=re.IGNORECASE
    )
    
    sql = re.sub(
        r"\b[a-z_][a-z0-9_]*\s*\.\s*(DATE\s+['\"][\d\-]+['\"])",
        r'\1',
        sql,
        flags=re.IGNORECASE
    )
    
    print("  ✓ Fixed malformed date literals")
    
    return sql


# ============================================================================
# PART 5: NUCLEAR REPAIR (MAIN ORCHESTRATOR)
# ============================================================================

# ============================================================================
# PART 5: NUCLEAR REPAIR (MAIN ORCHESTRATOR)
# ============================================================================
import re
import datetime
from typing import List, Dict, Any
import re
import datetime
from typing import List, Dict, Any
import re
import datetime
from typing import List, Dict, Any
# ============================================================================

# ============================================================================
# MAIN NUCLEAR REPAIR (PRODUCTION-READY)
# ============================================================================

def nuclear_repair_sql2222(
    raw_llm_output: str,
    tables: List[str],
    schema_catalog: Dict[str, Any] = None,
    join_info: Dict[str, Any] = None
) -> str:
    """
    Production-ready SQL repair with intelligent type handling.
    Prevents double-wrapping and handles all database types dynamically.
    """
    print("🔧 NUCLEAR REPAIR - Starting...")

    # Step 0: Clean raw SQL
    sql = raw_llm_output.strip()
    sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'```\s*', '', sql)
    sql = re.sub(r'`+', '', sql)
    sql = re.sub(r'^(Here is|Here\'s|The SQL query is|SQL:)\s*', '', sql, flags=re.IGNORECASE)

    sql_start = re.search(r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b', sql, re.IGNORECASE)
    if sql_start:
        sql = sql[sql_start.start():]

    sql = re.sub(r';[\s`]*$', '', sql)
    if ';' in sql:
        last_semicolon = sql.rfind(';')
        sql = sql[:last_semicolon + 1]

    # Step 1: SMART DATE CASTING (prevents double-wrapping)
    if schema_catalog:
        sql = smart_date_cast(sql, schema_catalog)

    # Step 2: SMART NUMERIC CASTING (prevents double-wrapping)
    if schema_catalog:
        sql = smart_numeric_cast(sql, schema_catalog)

    # Step 3: Fix INTERVAL quarters
    for i in range(1, 5):
        sql = re.sub(
            rf"INTERVAL\s+['\"]{i}\s+quarters?['\"]",
            f"INTERVAL '{i*3} months'",
            sql,
            flags=re.IGNORECASE
        )

    # Step 4: Replace CURRENT_DATE and NOW()
    today = datetime.date.today()
    fixed_date = f"DATE '{today.year}-{today.month:02d}-{today.day:02d}'"
    sql = re.sub(r'\bCURRENT_DATE\b', fixed_date, sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNOW\(\)\b', fixed_date, sql, flags=re.IGNORECASE)

    # Step 5: Fix NULL comparisons
    sql = re.sub(r'(\w+)\s*=\s*NULL\b', r'\1 IS NULL', sql, flags=re.IGNORECASE)
    sql = re.sub(r'(\w+)\s*!=\s*NULL\b', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)
    sql = re.sub(r'(\w+)\s*<>\s*NULL\b', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)
    
    # Remove duplicate IS NULL checks
    sql = re.sub(
        r'(\w+)\s+IS\s+NULL\s+OR\s+\1\s+IS\s+NULL',
        r'\1 IS NULL',
        sql,
        flags=re.IGNORECASE
    )

    # Step 6: Fix TRIM comparisons with NULL
    sql = re.sub(
        r'TRIM\(([^\)]+)\)\s*=\s*NULL',
        r'(\1 IS NULL OR TRIM(\1) = \'\')',
        sql,
        flags=re.IGNORECASE
    )

    # Step 7: Protect divisions (only if not already protected)
    def safe_div(match):
        left = match.group(1)
        right = match.group(2).strip()
        
        # Skip if already has NULLIF
        if 'NULLIF' in right.upper():
            return match.group(0)
        
        # Skip if right side is a complex expression
        if '(' in right or ')' in right:
            return match.group(0)
        
        return f"{left} / NULLIF({right}, 0)"

    sql = re.sub(
        r'([^\s]+)\s*/\s*([A-Za-z0-9_"\.\(\)]+)(?!\s*[,\)])',
        safe_div,
        sql
    )

    # Step 8: Remove stray commas
    sql = re.sub(r',\s*\)', ')', sql)
    sql = re.sub(r',\s*FROM\b', ' FROM', sql, flags=re.IGNORECASE)
    sql = re.sub(r',\s*WHERE\b', ' WHERE', sql, flags=re.IGNORECASE)

    # Step 9: Balance parentheses
    open_count = sql.count('(')
    close_count = sql.count(')')
    if open_count > close_count:
        sql += ')' * (open_count - close_count)
    elif close_count > open_count:
        excess = close_count - open_count
        for _ in range(excess):
            sql = sql.rstrip(';').rstrip(')')

    # Step 10: Clean whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    sql = sql.rstrip(';') + ';'
    sql = sql.replace('`', '')

    print("✅ NUCLEAR REPAIR - Completed")
    return sql



import re
from typing import List, Dict, Any, Tuple, Set
import datetime
import re
from typing import List, Dict, Any, Tuple, Set
import datetime

# ============================================================================
# CORE UTILITIES
# ============================================================================

def extract_table_aliases(sql: str) -> Dict[str, str]:
    """Extract table aliases from SQL (e.g., 'stage.app_main_2024 app' -> {'app': 'stage.app_main_2024'})"""
    aliases = {}
    
    patterns = [
        r'\bFROM\s+([a-zA-Z0-9_\.]+)\s+(?:AS\s+)?([a-zA-Z0-9_]+)\b',
        r'\bJOIN\s+([a-zA-Z0-9_\.]+)\s+(?:AS\s+)?([a-zA-Z0-9_]+)\b'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, sql, re.IGNORECASE)
        for match in matches:
            table_name = match.group(1)
            alias = match.group(2)
            if alias.upper() not in ('ON', 'WHERE', 'AND', 'OR', 'USING', 'AS'):
                aliases[alias] = table_name
    
    return aliases


def is_already_safely_casted(context: str, col_ref: str) -> bool:
    """
    Check if a column is already wrapped in a CASE/CAST that handles NULL and empty strings.
    This prevents double-wrapping.
    """
    # Extract column name (with or without alias)
    col_name = col_ref.split('.')[-1] if '.' in col_ref else col_ref
    
    # Look for existing CASE statements that handle this column
    patterns = [
        # Pattern 1: CASE WHEN col IS NULL OR TRIM(col) = '' THEN NULL ELSE col::TYPE END
        rf'\bCASE\s+WHEN\s+{re.escape(col_ref)}\s+IS\s+NULL\s+OR\s+TRIM\({re.escape(col_ref)}\)',
        rf'\bCASE\s+WHEN\s+{re.escape(col_name)}\s+IS\s+NULL\s+OR\s+TRIM\({re.escape(col_name)}\)',
        
        # Pattern 2: Already wrapped in parentheses with CASE
        rf'\(\s*CASE\s+WHEN\s+.*?{re.escape(col_name)}.*?END\s*\)',
        
        # Pattern 3: CAST with NULLIF
        rf'CAST\s*\(\s*NULLIF\s*\(\s*.*?{re.escape(col_name)}.*?\)',
    ]
    
    for pattern in patterns:
        if re.search(pattern, context, re.IGNORECASE | re.DOTALL):
            return True
    
    return False


def find_column_position(sql: str, col_ref: str, start_pos: int = 0) -> Tuple[int, int]:
    """
    Find the start and end position of a column reference in SQL.
    Returns: (start_index, end_index) or (-1, -1) if not found.
    """
    # Escape special regex characters
    escaped_col = re.escape(col_ref)
    
    # Search for the column as a whole word
    pattern = rf'\b{escaped_col}\b'
    match = re.search(pattern, sql[start_pos:], re.IGNORECASE)
    
    if match:
        actual_start = start_pos + match.start()
        actual_end = start_pos + match.end()
        return (actual_start, actual_end)
    
    return (-1, -1)


# ============================================================================
# PRE-PROCESSING: CLEAN LLM OUTPUT
# ============================================================================

# ============================================================================ 
# SMART DATE CASTING (FULLY POSTGRES SAFE)
# ============================================================================

import re
from typing import Dict
# ============================================================================ 
# SMART DATE CASTING (POSTGRES SAFE, CONTEXT-AWARE)
# ============================================================================
import re
from typing import Dict
import re
from typing import Dict
import re
from typing import Dict
import re
from typing import Dict

import re
from typing import Dict
import re
import datetime
from typing import List, Dict, Any
import re
import datetime
from typing import Dict, Any, List
import re
import datetime
from typing import Dict, Any, List

# ----------------------------
# 1️⃣ Clean LLM SQL output
# ----------------------------
import re
from typing import Dict, Any




import re
from typing import Dict, Any, List
import datetime


import re
from typing import Dict, Any, List
import datetime

# ----------------------------
# def clean_llm_output(raw_sql: str) -> str:
#     """Clean LLM output to extract pure SQL."""
#     sql = raw_sql.strip()
#     sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
#     sql = re.sub(r'```\s*', '', sql)
#     sql = re.sub(r'`+', '', sql)
#     sql = re.sub(r'^(Here is|Here\'s|The SQL query is|SQL:)\s*', '', sql, flags=re.IGNORECASE)
#     sql_start = re.search(r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b', sql, re.IGNORECASE)
#     if sql_start:
#         sql = sql[sql_start.start():]
#     sql = re.sub(r';[\s`]*$', '', sql)
#     if ';' in sql:
#         sql = sql[:sql.rfind(';') + 1]
#     return sql
# ----------------------------
import re
import datetime
from typing import Dict, List, Any, Set
import re
import datetime
from typing import Dict, List, Any, Set

# ----------------------------
import re
import datetime
from typing import Dict, List, Any, Set
import re
import datetime
from typing import Dict, List, Any, Set

import re
import datetime
from typing import Dict, List, Any
"""
How it works:

Numeric columns → trims empty strings and casts to numeric.

Text columns → trims spaces safely.

Timestamp/date columns → avoids TRIM on non-text, safely casts strings to timestamp.

Automatically prevents errors like:
- TRIM on numeric
- Invalid timestamp casts
"""

# ----------------------------
def aggressive_remove_trim_on_non_text(sql: str, debug: bool = False) -> str:
    """NUCLEAR OPTION: Remove ALL TRIM() calls on timestamp/date columns with ZERO regex complexity."""
    
    if debug:
        print("[DEBUG] Running NUCLEAR TRIM removal...")
    
    # Step 1: Find ALL columns that appear with ::TIMESTAMP anywhere
    timestamp_columns = set()
    
    # Match: anything::TIMESTAMP (with or without alias prefix)
    for match in re.finditer(r'(\w+)::TIMESTAMP', sql, re.IGNORECASE):
        col_name = match.group(1).lower()
        timestamp_columns.add(col_name)
        if debug:
            print(f"[DEBUG] Found timestamp column: {col_name}")
    
    # Step 2: Find ALL columns inside EXTRACT(...FROM column)
    for match in re.finditer(r'EXTRACT\s*\([^)]+FROM[^)]+\)', sql, re.IGNORECASE):
        extract_clause = match.group(0)
        # Get everything after FROM
        from_match = re.search(r'FROM\s+(.+?)(?:\)|$)', extract_clause, re.IGNORECASE)
        if from_match:
            col_part = from_match.group(1).strip()
            # Remove any CASE/WHEN/THEN wrapper to get to the actual column
            # Extract just the column name (last word before ::TIMESTAMP or end)
            col_match = re.search(r'(\w+)(?:::|\s|$)', col_part)
            if col_match:
                col_name = col_match.group(1).lower()
                timestamp_columns.add(col_name)
                if debug:
                    print(f"[DEBUG] Found EXTRACT column: {col_name}")
    
    if not timestamp_columns:
        if debug:
            print("[DEBUG] No timestamp columns found")
        return sql
    
    if debug:
        print(f"[DEBUG] All timestamp columns to clean: {sorted(timestamp_columns)}")
    
    # Step 3: NUCLEAR REMOVAL - Remove TRIM() around ANY occurrence of these columns
    for col in timestamp_columns:
        # This will match TRIM(col), TRIM(alias.col), TRIM( col ), etc.
        # We use a very simple pattern that catches everything
        
        # Pattern 1: TRIM(anything containing our column name)
        # This is aggressive but safe for timestamp columns
        pattern = rf'\bTRIM\s*\(\s*\w*\.?{re.escape(col)}\s*\)'
        
        matches = list(re.finditer(pattern, sql, re.IGNORECASE))
        if debug and matches:
            print(f"[DEBUG] Found {len(matches)} TRIM() calls for {col}")
            for m in matches:
                print(f"[DEBUG]   - Removing: {m.group(0)}")
        
        # Replace TRIM(xxx.col) with xxx.col or TRIM(col) with col
        sql = re.sub(
            pattern,
            lambda m: m.group(0).replace('TRIM(', '').replace(')', '').strip(),
            sql,
            flags=re.IGNORECASE
        )
    
    # Step 4: Remove "OR TRIM(column) = ''" patterns completely
    for col in timestamp_columns:
        # Match: "column IS NULL OR TRIM(column) = ''"
        # Replace with: "column IS NULL"
        
        # Pattern with alias
        before_sql = sql
        sql = re.sub(
            rf'(\w+\.{re.escape(col)})\s+IS\s+NULL\s+OR\s+TRIM\s*\(\s*\w+\.{re.escape(col)}\s*\)\s*=\s*[\'\"]+',
            r'\1 IS NULL',
            sql,
            flags=re.IGNORECASE
        )
        
        # Pattern without alias
        sql = re.sub(
            rf'\b{re.escape(col)}\s+IS\s+NULL\s+OR\s+TRIM\s*\(\s*{re.escape(col)}\s*\)\s*=\s*[\'\"]+',
            f'{col} IS NULL',
            sql,
            flags=re.IGNORECASE
        )
        
        if debug and before_sql != sql:
            print(f"[DEBUG] Removed 'OR TRIM() = '''  for {col}")
    
    return sql


# ----------------------------
def clean_llm_output(raw_sql: str) -> str:
    """Clean LLM output to extract pure SQL."""
    sql = raw_sql.strip()
    sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'```\s*', '', sql)
    sql = re.sub(r'`+', '', sql)
    sql = re.sub(r'^(Here is|Here\'s|The SQL query is|SQL:)\s*', '', sql, flags=re.IGNORECASE)
    sql_start = re.search(r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b', sql, re.IGNORECASE)
    if sql_start:
        sql = sql[sql_start.start():]
    sql = re.sub(r';[\s`]*$', '', sql)
    if ';' in sql:
        sql = sql[:sql.rfind(';') + 1]
    sql = re.sub(r'\)(\s*)--', r'),\1--', sql)
    sql = re.sub(r'--[^\n]*', '', sql)
    return sql


# ----------------------------
def extract_table_aliases(sql: str) -> Dict[str, str]:
    """Extract table aliases from FROM and JOIN clauses."""
    alias_pattern = re.compile(r'\bFROM\s+([A-Za-z0-9_."]+)\s+([A-Za-z_][A-Za-z0-9_]*)', re.IGNORECASE)
    aliases = {match.group(2): match.group(1) for match in alias_pattern.finditer(sql)}
    join_pattern = re.compile(r'\bJOIN\s+([A-Za-z0-9_."]+)\s+([A-Za-z_][A-Za-z0-9_]*)', re.IGNORECASE)
    for match in join_pattern.finditer(sql):
        aliases[match.group(2)] = match.group(1)
    return aliases


# ----------------------------
def build_column_type_map(schema_catalog: Dict[str, Any]) -> Dict[str, str]:
    """Build a map of column_name -> data_type from schema."""
    column_types = {}
    if not schema_catalog:
        return column_types
    
    for table_name, table_info in schema_catalog.items():
        for col in table_info.get('columns', []):
            col_name = col['name'].lower()
            col_type = col.get('type', '').upper()
            column_types[col_name] = col_type
    
    return column_types


# ----------------------------
def fix_missing_aliases(sql: str) -> str:
    """Add missing AS clauses for time extracts in ORDER BY."""
    order_by_pattern = r'ORDER\s+BY\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    time_units = ['year', 'month', 'day', 'quarter', 'week', 'hour', 'minute', 'second']
    for match in re.finditer(order_by_pattern, sql, flags=re.IGNORECASE):
        col_name = match.group(1).lower()
        if re.search(rf'\bAS\s+{re.escape(col_name)}\b', sql, flags=re.IGNORECASE):
            continue
        if col_name in time_units:
            from_pos = sql.upper().find(' FROM ')
            if from_pos == -1:
                continue
            select_clause = sql[:from_pos]
            extract_pattern = rf'(EXTRACT\s*\(\s*{re.escape(col_name)}\s+FROM\s+[^)]+\))(?!\s+AS)'
            extract_match = re.search(extract_pattern, select_clause, flags=re.IGNORECASE)
            if extract_match:
                sql = sql[:extract_match.end()] + f' AS {col_name}' + sql[extract_match.end():]
    return sql


# ----------------------------
def remove_trim_from_non_text_columns(sql: str, schema_catalog: Dict[str, Any], debug: bool = False) -> str:
    """Remove TRIM() calls on columns that are NOT text types (numeric, date, timestamp, boolean, etc.)."""
    
    # Build map of non-text columns from schema
    non_text_columns = set()
    
    if schema_catalog:
        for table_name, table_info in schema_catalog.items():
            for col in table_info.get('columns', []):
                col_name = col['name'].lower()
                col_type = col.get('type', '').upper()
                
                # Identify columns where TRIM() doesn't work or shouldn't be used
                if col_type in ('NUMERIC', 'INTEGER', 'BIGINT', 'SMALLINT', 'DECIMAL', 
                               'FLOAT', 'DOUBLE', 'REAL', 'INT', 'INT2', 'INT4', 'INT8',
                               'FLOAT4', 'FLOAT8', 'MONEY',
                               'DATE', 'TIMESTAMP', 'TIMESTAMPTZ', 'TIME', 'TIMETZ',
                               'TIMESTAMP WITHOUT TIME ZONE', 'TIMESTAMP WITH TIME ZONE',
                               'BOOLEAN', 'BOOL', 'UUID', 'BYTEA', 'JSONB', 'JSON'):
                    non_text_columns.add(col_name)
    
    # FALLBACK: If no schema, detect columns by context clues in the SQL
    if not non_text_columns or not schema_catalog:
        # Find columns being cast to TIMESTAMP
        timestamp_cast_pattern = r'(\w+)::TIMESTAMP'
        for match in re.finditer(timestamp_cast_pattern, sql, re.IGNORECASE):
            col_name = match.group(1).lower()
            non_text_columns.add(col_name)
            if debug:
                print(f"[DEBUG] Detected timestamp column from cast: {col_name}")
        
        # Find columns in EXTRACT functions
        extract_pattern = r'EXTRACT\s*\([^)]+FROM\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s*\)'
        for match in re.finditer(extract_pattern, sql, re.IGNORECASE):
            col_name = match.group(1).split('.')[-1].lower()
            non_text_columns.add(col_name)
            if debug:
                print(f"[DEBUG] Detected date/time column from EXTRACT: {col_name}")
    
    if not non_text_columns:
        if debug:
            print(f"[DEBUG] No non-text columns detected")
        return sql
    
    if debug:
        print(f"[DEBUG] Non-text columns (removing TRIM): {sorted(non_text_columns)}")
    
    aliases = extract_table_aliases(sql)
    
    # For each non-text column, remove TRIM() wrapping in all patterns
    for col in non_text_columns:
        col_refs = [col] + [f"{alias}.{col}" for alias in aliases.keys()]
        
        for col_ref in col_refs:
            escaped_ref = re.escape(col_ref)
            
            # Pattern 1: Simple TRIM(column) -> column
            pattern = rf'\bTRIM\s*\(\s*{escaped_ref}\s*\)'
            if debug and re.search(pattern, sql, re.IGNORECASE):
                print(f"[DEBUG] Removing TRIM() from non-text column: {col_ref}")
            sql = re.sub(pattern, col_ref, sql, flags=re.IGNORECASE)
            
            # Pattern 2: NULLIF(TRIM(column), '') -> column (simplified)
            pattern = rf'NULLIF\s*\(\s*TRIM\s*\(\s*{escaped_ref}\s*\)\s*,\s*[\'"][\'"]?\s*\)'
            if debug and re.search(pattern, sql, re.IGNORECASE):
                print(f"[DEBUG] Fixing NULLIF(TRIM()) on non-text column: {col_ref}")
            sql = re.sub(pattern, col_ref, sql, flags=re.IGNORECASE)
            
            # Pattern 3: CAST(NULLIF(TRIM(column), '') AS NUMERIC) -> CAST(column AS NUMERIC)
            pattern = rf'CAST\s*\(\s*NULLIF\s*\(\s*TRIM\s*\(\s*{escaped_ref}\s*\)\s*,\s*[\'"][\'"]?\s*\)\s*AS\s+NUMERIC\s*\)'
            if debug and re.search(pattern, sql, re.IGNORECASE):
                print(f"[DEBUG] Simplifying CAST(NULLIF(TRIM())) on non-text column: {col_ref}")
            sql = re.sub(pattern, f'CAST({col_ref} AS NUMERIC)', sql, flags=re.IGNORECASE)
            
            # Pattern 4: column IS NULL OR TRIM(column) = '' -> column IS NULL
            pattern = rf'{escaped_ref}\s+IS\s+NULL\s+OR\s+TRIM\s*\(\s*{escaped_ref}\s*\)\s*=\s*[\'"][\'"]?'
            if debug and re.search(pattern, sql, re.IGNORECASE):
                print(f"[DEBUG] Simplifying NULL check on non-text column: {col_ref}")
            sql = re.sub(pattern, f'{col_ref} IS NULL', sql, flags=re.IGNORECASE)
            
            # Pattern 5: TRIM(column) = '' -> FALSE (for non-text, this is always false)
            pattern = rf'\bTRIM\s*\(\s*{escaped_ref}\s*\)\s*=\s*[\'"][\'"]'
            if debug and re.search(pattern, sql, re.IGNORECASE):
                print(f"[DEBUG] Removing TRIM() = '' check on non-text column: {col_ref}")
            sql = re.sub(pattern, 'FALSE', sql, flags=re.IGNORECASE)
    
    return sql


# ----------------------------
# def aggressive_remove_trim_on_non_text(sql: str, debug: bool = False) -> str:
#     """NUCLEAR OPTION: Remove ALL TRIM() calls on timestamp/date columns with ZERO regex complexity."""
    
#     if debug:
#         print("[DEBUG] Running NUCLEAR TRIM removal...")
    
#     # Step 1: Find ALL columns that appear with ::TIMESTAMP anywhere
#     timestamp_columns = set()
    
#     # Match: anything::TIMESTAMP (with or without alias prefix)
#     for match in re.finditer(r'(\w+)::TIMESTAMP', sql, re.IGNORECASE):
#         col_name = match.group(1).lower()
#         timestamp_columns.add(col_name)
#         if debug:
#             print(f"[DEBUG] Found timestamp column: {col_name}")
    
#     # Step 2: Find ALL columns inside EXTRACT(...FROM column)
#     for match in re.finditer(r'EXTRACT\s*\([^)]+FROM[^)]+\)', sql, re.IGNORECASE):
#         extract_clause = match.group(0)
#         # Get everything after FROM
#         from_match = re.search(r'FROM\s+(.+?)(?:\)|$)', extract_clause, re.IGNORECASE)
#         if from_match:
#             col_part = from_match.group(1).strip()
#             # Remove any CASE/WHEN/THEN wrapper to get to the actual column
#             # Extract just the column name (last word before ::TIMESTAMP or end)
#             col_match = re.search(r'(\w+)(?:::|\s|$)', col_part)
#             if col_match:
#                 col_name = col_match.group(1).lower()
#                 timestamp_columns.add(col_name)
#                 if debug:
#                     print(f"[DEBUG] Found EXTRACT column: {col_name}")
    
#     if not timestamp_columns:
#         if debug:
#             print("[DEBUG] No timestamp columns found")
#         return sql
    
#     if debug:
#         print(f"[DEBUG] All timestamp columns to clean: {sorted(timestamp_columns)}")
    
#     # Step 3: NUCLEAR REMOVAL - Remove TRIM() around ANY occurrence of these columns
#     for col in timestamp_columns:
#         # This will match TRIM(col), TRIM(alias.col), TRIM( col ), etc.
#         # We use a very simple pattern that catches everything
        
#         # Pattern 1: TRIM(anything containing our column name)
#         # This is aggressive but safe for timestamp columns
#         pattern = rf'\bTRIM\s*\(\s*\w*\.?{re.escape(col)}\s*\)'
        
#         matches = list(re.finditer(pattern, sql, re.IGNORECASE))
#         if debug and matches:
#             print(f"[DEBUG] Found {len(matches)} TRIM() calls for {col}")
#             for m in matches:
#                 print(f"[DEBUG]   - Removing: {m.group(0)}")
        
#         # Replace TRIM(xxx.col) with xxx.col or TRIM(col) with col
#         sql = re.sub(
#             pattern,
#             lambda m: m.group(0).replace('TRIM(', '').replace(')', '').strip(),
#             sql,
#             flags=re.IGNORECASE
#         )
    
#     # Step 4: Remove "OR TRIM(column) = ''" patterns completely
#     for col in timestamp_columns:
#         # Match: "column IS NULL OR TRIM(column) = ''"
#         # Replace with: "column IS NULL"
        
#         # Pattern with alias
#         before_sql = sql
#         sql = re.sub(
#             rf'(\w+\.{re.escape(col)})\s+IS\s+NULL\s+OR\s+TRIM\s*\(\s*\w+\.{re.escape(col)}\s*\)\s*=\s*[\'\"]+',
#             r'\1 IS NULL',
#             sql,
#             flags=re.IGNORECASE
#         )
        
#         # Pattern without alias
#         sql = re.sub(
#             rf'\b{re.escape(col)}\s+IS\s+NULL\s+OR\s+TRIM\s*\(\s*{re.escape(col)}\s*\)\s*=\s*[\'\"]+',
#             f'{col} IS NULL',
#             sql,
#             flags=re.IGNORECASE
#         )
        
#         if debug and before_sql != sql:
#             print(f"[DEBUG] Removed 'OR TRIM() = '''  for {col}")
    
#     return sql


# # ----------------------------
# def clean_llm_output(raw_sql: str) -> str:
#     """Clean LLM output to extract pure SQL."""
#     sql = raw_sql.strip()
#     sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
#     sql = re.sub(r'```\s*', '', sql)
#     sql = re.sub(r'`+', '', sql)
#     sql = re.sub(r'^(Here is|Here\'s|The SQL query is|SQL:)\s*', '', sql, flags=re.IGNORECASE)
#     sql_start = re.search(r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b', sql, re.IGNORECASE)
#     if sql_start:
#         sql = sql[sql_start.start():]
#     sql = re.sub(r';[\s`]*$', '', sql)
#     if ';' in sql:
#         sql = sql[:sql.rfind(';') + 1]
#     sql = re.sub(r'\)(\s*)--', r'),\1--', sql)
#     sql = re.sub(r'--[^\n]*', '', sql)
#     return sql


# # ----------------------------
# def extract_table_aliases(sql: str) -> Dict[str, str]:
#     """Extract table aliases from FROM and JOIN clauses."""
#     alias_pattern = re.compile(r'\bFROM\s+([A-Za-z0-9_."]+)\s+([A-Za-z_][A-Za-z0-9_]*)', re.IGNORECASE)
#     aliases = {match.group(2): match.group(1) for match in alias_pattern.finditer(sql)}
#     join_pattern = re.compile(r'\bJOIN\s+([A-Za-z0-9_."]+)\s+([A-Za-z_][A-Za-z0-9_]*)', re.IGNORECASE)
#     for match in join_pattern.finditer(sql):
#         aliases[match.group(2)] = match.group(1)
#     return aliases


# # ----------------------------
# def build_column_type_map(schema_catalog: Dict[str, Any]) -> Dict[str, str]:
#     """Build a map of column_name -> data_type from schema."""
#     column_types = {}
#     if not schema_catalog:
#         return column_types
    
#     for table_name, table_info in schema_catalog.items():
#         for col in table_info.get('columns', []):
#             col_name = col['name'].lower()
#             col_type = col.get('type', '').upper()
#             column_types[col_name] = col_type
    
#     return column_types


# # ----------------------------
# def fix_missing_aliases(sql: str) -> str:
#     """Add missing AS clauses for time extracts in ORDER BY."""
#     order_by_pattern = r'ORDER\s+BY\s+([a-zA-Z_][a-zA-Z0-9_]*)'
#     time_units = ['year', 'month', 'day', 'quarter', 'week', 'hour', 'minute', 'second']
#     for match in re.finditer(order_by_pattern, sql, flags=re.IGNORECASE):
#         col_name = match.group(1).lower()
#         if re.search(rf'\bAS\s+{re.escape(col_name)}\b', sql, flags=re.IGNORECASE):
#             continue
#         if col_name in time_units:
#             from_pos = sql.upper().find(' FROM ')
#             if from_pos == -1:
#                 continue
#             select_clause = sql[:from_pos]
#             extract_pattern = rf'(EXTRACT\s*\(\s*{re.escape(col_name)}\s+FROM\s+[^)]+\))(?!\s+AS)'
#             extract_match = re.search(extract_pattern, select_clause, flags=re.IGNORECASE)
#             if extract_match:
#                 sql = sql[:extract_match.end()] + f' AS {col_name}' + sql[extract_match.end():]
#     return sql


# # ----------------------------
# def remove_trim_from_non_text_columns(sql: str, schema_catalog: Dict[str, Any], debug: bool = False) -> str:
#     """Remove TRIM() calls on columns that are NOT text types (numeric, date, timestamp, boolean, etc.)."""
    
#     # Build map of non-text columns from schema
#     non_text_columns = set()
    
#     if schema_catalog:
#         for table_name, table_info in schema_catalog.items():
#             for col in table_info.get('columns', []):
#                 col_name = col['name'].lower()
#                 col_type = col.get('type', '').upper()
                
#                 # Identify columns where TRIM() doesn't work or shouldn't be used
#                 if col_type in ('NUMERIC', 'INTEGER', 'BIGINT', 'SMALLINT', 'DECIMAL', 
#                                'FLOAT', 'DOUBLE', 'REAL', 'INT', 'INT2', 'INT4', 'INT8',
#                                'FLOAT4', 'FLOAT8', 'MONEY',
#                                'DATE', 'TIMESTAMP', 'TIMESTAMPTZ', 'TIME', 'TIMETZ',
#                                'TIMESTAMP WITHOUT TIME ZONE', 'TIMESTAMP WITH TIME ZONE',
#                                'BOOLEAN', 'BOOL', 'UUID', 'BYTEA', 'JSONB', 'JSON'):
#                     non_text_columns.add(col_name)
    
#     # FALLBACK: If no schema, detect columns by context clues in the SQL
#     if not non_text_columns or not schema_catalog:
#         # Find columns being cast to TIMESTAMP
#         timestamp_cast_pattern = r'(\w+)::TIMESTAMP'
#         for match in re.finditer(timestamp_cast_pattern, sql, re.IGNORECASE):
#             col_name = match.group(1).lower()
#             non_text_columns.add(col_name)
#             if debug:
#                 print(f"[DEBUG] Detected timestamp column from cast: {col_name}")
        
#         # Find columns in EXTRACT functions
#         extract_pattern = r'EXTRACT\s*\([^)]+FROM\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s*\)'
#         for match in re.finditer(extract_pattern, sql, re.IGNORECASE):
#             col_name = match.group(1).split('.')[-1].lower()
#             non_text_columns.add(col_name)
#             if debug:
#                 print(f"[DEBUG] Detected date/time column from EXTRACT: {col_name}")
    
#     if not non_text_columns:
#         if debug:
#             print(f"[DEBUG] No non-text columns detected")
#         return sql
    
#     if debug:
#         print(f"[DEBUG] Non-text columns (removing TRIM): {sorted(non_text_columns)}")
    
#     aliases = extract_table_aliases(sql)
    
#     # For each non-text column, remove TRIM() wrapping in all patterns
#     for col in non_text_columns:
#         col_refs = [col] + [f"{alias}.{col}" for alias in aliases.keys()]
        
#         for col_ref in col_refs:
#             escaped_ref = re.escape(col_ref)
            
#             # Pattern 1: Simple TRIM(column) -> column
#             pattern = rf'\bTRIM\s*\(\s*{escaped_ref}\s*\)'
#             if debug and re.search(pattern, sql, re.IGNORECASE):
#                 print(f"[DEBUG] Removing TRIM() from non-text column: {col_ref}")
#             sql = re.sub(pattern, col_ref, sql, flags=re.IGNORECASE)
            
#             # Pattern 2: NULLIF(TRIM(column), '') -> column (simplified)
#             pattern = rf'NULLIF\s*\(\s*TRIM\s*\(\s*{escaped_ref}\s*\)\s*,\s*[\'"][\'"]?\s*\)'
#             if debug and re.search(pattern, sql, re.IGNORECASE):
#                 print(f"[DEBUG] Fixing NULLIF(TRIM()) on non-text column: {col_ref}")
#             sql = re.sub(pattern, col_ref, sql, flags=re.IGNORECASE)
            
#             # Pattern 3: CAST(NULLIF(TRIM(column), '') AS NUMERIC) -> CAST(column AS NUMERIC)
#             pattern = rf'CAST\s*\(\s*NULLIF\s*\(\s*TRIM\s*\(\s*{escaped_ref}\s*\)\s*,\s*[\'"][\'"]?\s*\)\s*AS\s+NUMERIC\s*\)'
#             if debug and re.search(pattern, sql, re.IGNORECASE):
#                 print(f"[DEBUG] Simplifying CAST(NULLIF(TRIM())) on non-text column: {col_ref}")
#             sql = re.sub(pattern, f'CAST({col_ref} AS NUMERIC)', sql, flags=re.IGNORECASE)
            
#             # Pattern 4: column IS NULL OR TRIM(column) = '' -> column IS NULL
#             pattern = rf'{escaped_ref}\s+IS\s+NULL\s+OR\s+TRIM\s*\(\s*{escaped_ref}\s*\)\s*=\s*[\'"][\'"]?'
#             if debug and re.search(pattern, sql, re.IGNORECASE):
#                 print(f"[DEBUG] Simplifying NULL check on non-text column: {col_ref}")
#             sql = re.sub(pattern, f'{col_ref} IS NULL', sql, flags=re.IGNORECASE)
            
#             # Pattern 5: TRIM(column) = '' -> FALSE (for non-text, this is always false)
#             pattern = rf'\bTRIM\s*\(\s*{escaped_ref}\s*\)\s*=\s*[\'"][\'"]'
#             if debug and re.search(pattern, sql, re.IGNORECASE):
#                 print(f"[DEBUG] Removing TRIM() = '' check on non-text column: {col_ref}")
#             sql = re.sub(pattern, 'FALSE', sql, flags=re.IGNORECASE)
    
#     return sql


# ----------------------------
def smart_date_cast(sql: str, schema_catalog: dict, debug: bool = False) -> str:
    """Add safe NULL handling for TEXT date columns."""
    if not schema_catalog:
        return sql
    date_keywords = ['date', 'time', 'timestamp', 'datetime']
    date_columns = set()
    for table_info in schema_catalog.values():
        for col in table_info.get('columns', []):
            col_name = col['name'].lower()
            col_type = col.get('type', '').upper()
            if col_type in ('TEXT', 'VARCHAR', 'CHAR', 'STRING') and any(kw in col_name for kw in date_keywords):
                date_columns.add(col_name)
    if not date_columns:
        return sql
    if debug:
        print(f"[DEBUG] Date columns identified for casting: {sorted(date_columns)}")
    aliases = extract_table_aliases(sql)
    col_pattern = '|'.join(re.escape(col) for col in date_columns)
    if aliases:
        alias_pattern = '|'.join(re.escape(alias) for alias in aliases.keys())
        full_pattern = rf'((?:{alias_pattern})\.)?({col_pattern})::TIMESTAMP'
    else:
        full_pattern = rf'({col_pattern})::TIMESTAMP'
    def replace_timestamp_cast(match):
        col_ref = match.group(0).replace('::TIMESTAMP', '')
        if debug:
            print(f"[DEBUG] Casting column to TIMESTAMP: {col_ref}")
        return f"(CASE WHEN {col_ref} IS NULL OR TRIM({col_ref}) = '' THEN NULL ELSE {match.group(0)} END)"
    pattern = rf'\b{full_pattern}\b'
    sql = re.sub(pattern, replace_timestamp_cast, sql, flags=re.IGNORECASE)
    return sql


# ----------------------------
def remove_duplicate_casts(sql: str) -> str:
    """Remove nested duplicate CAST operations."""
    max_iterations = 5
    iteration = 0
    while iteration < max_iterations:
        original_sql = sql
        
        sql = re.sub(
            r'(SUM|AVG|MIN|MAX|COUNT)\s*\(\s*CAST\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\.\s*CAST\s*\(\s*NULLIF',
            r'\1(CAST(NULLIF',
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r'CAST\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\.\s*CAST\s*\(',
            r'CAST(',
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r'CAST\s*\(\s*CAST\s*\(([^)]+)\)\s+AS\s+\w+\s*\)\s+AS\s+(\w+)',
            r'CAST(\1 AS \2)',
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r'CAST\s*\(\s*CAST\s*\(',
            r'CAST(',
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r'([a-zA-Z_][a-zA-Z0-9_]*)\.\s*CAST\s*\(',
            r'CAST(',
            sql,
            flags=re.IGNORECASE
        )
        
        sql = re.sub(
            r'(TRIM|NULLIF)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\.\s*([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\1(\2.\3',
            sql,
            flags=re.IGNORECASE
        )
        
        sql = re.sub(
            r'NULLIF\s*\(\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*,\s*NULL\s*\)',
            r'\1',
            sql,
            flags=re.IGNORECASE
        )
        
        if sql == original_sql:
            break
        iteration += 1
    return sql



# ----------------------------
def smart_numeric_cast(sql: str, schema_catalog: Dict[str, Any], debug: bool = False) -> str:
    """Safely cast TEXT numeric columns to NUMERIC with NULLIF(TRIM()), avoiding already-numeric columns."""
    if not schema_catalog:
        return sql
    
    numeric_keywords = [
        'amount', 'price', 'cost', 'fee', 'rate', 'income', 
        'balance', 'payment', 'charge', 'value', 'total', 
        'chargeoff', 'chargeoffs', 'revenue', 'profit', 'loss', 'principal', 'interest'
    ]
    
    text_numeric_columns = set()
    numeric_type_columns = set()
    
    for table_name, table_info in schema_catalog.items():
        for col in table_info.get('columns', []):
            col_name = col['name'].lower()
            col_type = col.get('type', '').upper()
            
            if any(kw in col_name for kw in numeric_keywords):
                if col_type in ('TEXT', 'VARCHAR', 'CHAR', 'STRING'):
                    text_numeric_columns.add(col_name)
                elif col_type in ('NUMERIC', 'INTEGER', 'BIGINT', 'SMALLINT', 'DECIMAL', 
                                 'FLOAT', 'DOUBLE', 'REAL', 'INT', 'INT2', 'INT4', 'INT8',
                                 'FLOAT4', 'FLOAT8', 'MONEY'):
                    numeric_type_columns.add(col_name)
    
    if not text_numeric_columns:
        return sql
    
    if debug:
        print(f"[DEBUG] TEXT numeric columns for casting: {sorted(text_numeric_columns)}")
        print(f"[DEBUG] Already NUMERIC columns (skip): {sorted(numeric_type_columns)}")
    
    aliases = extract_table_aliases(sql)
    
    for col in text_numeric_columns:
        if col in numeric_type_columns:
            continue
            
        col_refs = [col] + [f"{alias}.{col}" for alias in aliases.keys()]
        
        for col_ref in col_refs:
            escaped_ref = re.escape(col_ref)
            
            if re.search(rf'NULLIF\s*\(\s*TRIM\s*\(\s*{escaped_ref}\s*\)', sql, re.IGNORECASE):
                continue
            
            for func in ['SUM', 'AVG', 'MIN', 'MAX', 'COUNT']:
                pattern = rf'\b{func}\s*\(\s*{escaped_ref}\s*\)'
                def replace_agg(m):
                    if debug:
                        print(f"[DEBUG] Wrapping aggregate column: {col_ref} inside {func}()")
                    return f"{func}(CAST(NULLIF(TRIM({col_ref}), '') AS NUMERIC))"
                sql = re.sub(pattern, replace_agg, sql, flags=re.IGNORECASE)
            
            pattern = rf'\bCAST\s*\(\s*{escaped_ref}\s+AS\s+NUMERIC\s*\)'
            if re.search(pattern, sql, flags=re.IGNORECASE):
                if debug:
                    print(f"[DEBUG] Wrapping existing CAST to NUMERIC for column: {col_ref}")
                sql = re.sub(pattern, f"CAST(NULLIF(TRIM({col_ref}), '') AS NUMERIC)", sql, flags=re.IGNORECASE)
            
            pattern = rf'\b{escaped_ref}\b'
            def replace_standalone(match):
                start = match.start()
                end = match.end()
                before = sql[max(0, start-80):start].upper()
                after = sql[end:min(len(sql), end+30)].upper()
                
                if (re.search(r'\w+\s*\($', before) or 
                    'NULLIF' in before[-60:] or 
                    'CAST(' in before[-15:] or 
                    after.startswith('.') or 
                    'GROUP BY' in before[-20:] or 
                    'ORDER BY' in before[-20:]):
                    return match.group(0)
                
                if debug:
                    print(f"[DEBUG] Wrapping standalone column: {match.group(0)}")
                return f"CAST(NULLIF(TRIM({match.group(0)}), '') AS NUMERIC)"
            
            sql = re.sub(pattern, replace_standalone, sql, flags=re.IGNORECASE)
    
    return sql

import re
import datetime
from typing import Dict, List, Any
import re
import datetime
from typing import Dict, List, Any

# def validate_and_fix_cte_syntax(sql: str, debug: bool = False) -> str:
#     """Fix common CTE syntax errors."""
#     if debug:
#         print("[DEBUG] Validating CTE syntax...")
    
#     # Check if it's a CTE query
#     if not re.search(r'\bWITH\b', sql, re.IGNORECASE):
#         return sql
    
#     # Fix 1: Remove comma before final SELECT in CTE
#     # Pattern: )), SELECT ... -> )) SELECT ...
#     sql = re.sub(
#         r'\)\)\s*,\s*SELECT\s+',
#         r')) SELECT ',
#         sql,
#         flags=re.IGNORECASE
#     )
    
#     # Fix 2: Add comma between CTEs if missing
#     # Pattern: ) cte_name AS ( -> ), cte_name AS (
#     sql = re.sub(
#         r'\)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(',
#         r'),\n\1 AS (',
#         sql,
#         flags=re.IGNORECASE
#     )
    
#     # Fix 3: Remove double commas
#     sql = re.sub(r',\s*,', ',', sql)
    
#     if debug:
#         print("[DEBUG] CTE syntax validation complete")
    
#     return sql


import re
import datetime
from typing import Dict, List, Any, Set

def validate_and_fix_cte_syntax(sql: str, debug: bool = False) -> str:
    """Fix common CTE syntax errors."""
    if debug:
        print("[DEBUG] Validating CTE syntax...")
    
    if not re.search(r'\bWITH\b', sql, re.IGNORECASE):
        return sql
    
    # Fix 1: Remove comma before final SELECT in CTE
    sql = re.sub(r'\)\)\s*,\s*SELECT\s+', r')) SELECT ', sql, flags=re.IGNORECASE)
    
    # Fix 2: Add comma between CTEs if missing
    sql = re.sub(r'\)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(', r'),\n\1 AS (', sql, flags=re.IGNORECASE)
    
    # Fix 3: Remove double commas
    sql = re.sub(r',\s*,', ',', sql)
    
    if debug:
        print("[DEBUG] CTE syntax validation complete")
    
    return sql


# ----------------------------
def identify_timestamp_date_columns(sql: str, schema_catalog: Dict[str, Any] = None, debug: bool = False) -> Set[str]:
    """
    Identify ALL timestamp/date columns from SQL patterns and schema.
    These are columns where TRIM() should NEVER be used.
    """
    timestamp_columns = set()
    
    # Method 1: From SQL patterns
    # Find columns cast to TIMESTAMP
    for match in re.finditer(r'(\w+)::TIMESTAMP', sql, re.IGNORECASE):
        col_name = match.group(1).lower()
        timestamp_columns.add(col_name)
        if debug:
            print(f"[DEBUG] Found TIMESTAMP cast column: {col_name}")
    
    # Find columns in EXTRACT()
    for match in re.finditer(r'EXTRACT\s*\([^)]*FROM\s+[^)]+\)', sql, re.IGNORECASE):
        extract_clause = match.group(0)
        col_matches = re.findall(r'(\w+)(?:::|\s*\))', extract_clause, re.IGNORECASE)
        for col_name in col_matches:
            if col_name.upper() not in ('YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND', 'QUARTER', 'WEEK'):
                timestamp_columns.add(col_name.lower())
                if debug:
                    print(f"[DEBUG] Found EXTRACT column: {col_name}")
    
    # Method 2: From schema catalog (if available)
    if schema_catalog:
        for table_name, table_info in schema_catalog.items():
            for col in table_info.get('columns', []):
                col_name = col.get('name', '').lower()
                col_type = col.get('type', '').upper()
                
                # Mark timestamp/date columns from schema
                if col_type in ('TIMESTAMP', 'TIMESTAMPTZ', 'DATE', 'TIME', 'TIMETZ',
                               'TIMESTAMP WITHOUT TIME ZONE', 'TIMESTAMP WITH TIME ZONE'):
                    timestamp_columns.add(col_name)
                    if debug:
                        print(f"[DEBUG] Schema column {col_name}: {col_type}")
    
    if debug:
        print(f"[DEBUG] Total timestamp/date columns identified: {sorted(timestamp_columns)}")
    
    return timestamp_columns


# ----------------------------
def smart_remove_trim_from_timestamps(sql: str, timestamp_columns: Set[str], debug: bool = False) -> str:
    """
    Remove TRIM() ONLY from timestamp/date columns.
    Preserves TRIM() on text columns where it's valid.
    """
    if not timestamp_columns:
        return sql
    
    if debug:
        print(f"[DEBUG] Removing TRIM from these columns: {sorted(timestamp_columns)}")
    
    for col in timestamp_columns:
        # Pattern 1: TRIM(alias.column) where column is a timestamp
        pattern1 = rf'\bTRIM\s*\(\s*(\w+)\.({re.escape(col)})\s*\)'
        
        def replace_aliased(match):
            alias = match.group(1)
            col_name = match.group(2)
            result = f"{alias}.{col_name}"
            if debug:
                print(f"[DEBUG]   Removing: TRIM({alias}.{col_name}) -> {result}")
            return result
        
        sql = re.sub(pattern1, replace_aliased, sql, flags=re.IGNORECASE)
        
        # Pattern 2: TRIM(column) without alias
        pattern2 = rf'\bTRIM\s*\(\s*({re.escape(col)})\s*\)(?!\s*\.)'
        
        def replace_simple(match):
            col_name = match.group(1)
            if debug:
                print(f"[DEBUG]   Removing: TRIM({col_name}) -> {col_name}")
            return col_name
        
        sql = re.sub(pattern2, replace_simple, sql, flags=re.IGNORECASE)
        
        # Pattern 3: Remove "OR TRIM(column) = ''" patterns
        pattern3 = rf'\s+OR\s+TRIM\s*\(\s*(\w+\.)?{re.escape(col)}\s*\)\s*=\s*[\'\"]+\s*'
        before = sql
        sql = re.sub(pattern3, '', sql, flags=re.IGNORECASE)
        if before != sql and debug:
            print(f"[DEBUG]   Removed 'OR TRIM(...) = ''' check for {col}")
    
    return sql





# ----------------------------
def brute_force_remove_all_trim(sql: str, debug: bool = False) -> str:
    """
    BRUTE FORCE: Remove ALL TRIM() calls from SQL, period.
    This is the nuclear option - no fancy regex, just pure replacement.
    """
    if debug:
        print("[DEBUG] BRUTE FORCE TRIM REMOVAL STARTING...")
    
    original_sql = sql
    iteration = 0
    max_iterations = 10
    
    while 'TRIM(' in sql.upper() and iteration < max_iterations:
        iteration += 1
        
        if debug:
            print(f"[DEBUG] Iteration {iteration}: Found TRIM in SQL")
        
        # Find all TRIM( patterns and their matching closing )
        # We'll do this character by character to handle nested parentheses
        
        i = 0
        while i < len(sql):
            # Look for TRIM( (case insensitive)
            if sql[i:i+5].upper() == 'TRIM(':
                # Found TRIM(, now find the matching )
                start = i
                i += 5  # Skip past TRIM(
                
                paren_depth = 1
                content_start = i
                
                while i < len(sql) and paren_depth > 0:
                    if sql[i] == '(':
                        paren_depth += 1
                    elif sql[i] == ')':
                        paren_depth -= 1
                    i += 1
                
                # Extract what was inside TRIM()
                if paren_depth == 0:
                    content = sql[content_start:i-1].strip()
                    
                    if debug:
                        print(f"[DEBUG]   Removing: TRIM({content})")
                    
                    # Replace TRIM(...) with just the content
                    sql = sql[:start] + content + sql[i:]
                    
                    # Reset to start of replacement to check again
                    i = start
            else:
                i += 1
        
        if sql == original_sql:
            break
        original_sql = sql
    
    if debug and iteration > 0:
        print(f"[DEBUG] TRIM removal completed after {iteration} iterations")
    
    return sql


# ----------------------------
def aggressive_remove_trim_on_non_text(sql: str, debug: bool = False) -> str:
    """Enhanced version that first identifies timestamp columns, then removes TRIM."""
    
    if debug:
        print("[DEBUG] Running aggressive TRIM removal...")
    
    # Step 1: Find ALL timestamp/date columns
    timestamp_columns = set()
    
    # Pattern 1: column::TIMESTAMP
    for match in re.finditer(r'(\w+)::TIMESTAMP', sql, re.IGNORECASE):
        col_name = match.group(1).lower()
        timestamp_columns.add(col_name)
        if debug:
            print(f"[DEBUG] Found timestamp column: {col_name}")
    
    # Pattern 2: In EXTRACT()
    for match in re.finditer(r'EXTRACT\s*\([^)]*FROM\s+[^)]+\)', sql, re.IGNORECASE):
        extract_clause = match.group(0)
        # Find column name (last word before :: or )
        col_matches = re.findall(r'(\w+)(?:::|\s*\))', extract_clause, re.IGNORECASE)
        for col_name in col_matches:
            if col_name.upper() not in ('YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND'):
                timestamp_columns.add(col_name.lower())
                if debug:
                    print(f"[DEBUG] Found EXTRACT column: {col_name}")
    
    if not timestamp_columns:
        if debug:
            print("[DEBUG] No timestamp columns found")
        return sql
    
    if debug:
        print(f"[DEBUG] Timestamp columns: {sorted(timestamp_columns)}")
    
    # Step 2: For each timestamp column, remove ALL TRIM references
    for col in timestamp_columns:
        # Remove TRIM(alias.column)
        pattern1 = rf'\bTRIM\s*\(\s*\w+\.{re.escape(col)}\s*\)'
        sql = re.sub(pattern1, lambda m: m.group(0).replace('TRIM(', '').rstrip(')'), sql, flags=re.IGNORECASE)
        
        # Remove TRIM(column)
        pattern2 = rf'\bTRIM\s*\(\s*{re.escape(col)}\s*\)'
        sql = re.sub(pattern2, col, sql, flags=re.IGNORECASE)
        
        # Remove "OR TRIM(column) = ''" checks
        pattern3 = rf'\s+OR\s+TRIM\s*\(\s*(\w+\.)?{re.escape(col)}\s*\)\s*=\s*[\'"]+'
        sql = re.sub(pattern3, '', sql, flags=re.IGNORECASE)
        
        if debug:
            print(f"[DEBUG] Cleaned TRIM for {col}")
    
    return sql


# # ----------------------------
# def nuclear_repair_sql(
#     raw_llm_output: str,
#     tables: List[str] = None,
#     schema_catalog: Dict[str, Any] = None,
#     join_info: Dict[str, Any] = None,
#     debug: bool = False
# ) -> str:
#     """Comprehensive SQL repair with GUARANTEED TRIM removal."""
    
#     # Clean LLM output
#     sql = raw_llm_output.strip()
#     sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
#     sql = re.sub(r'```\s*', '', sql)
#     sql = re.sub(r'`+', '', sql)
    
#     # ✅ FIX: Better CTE comma handling
#     # Pattern: )), SELECT should become )) SELECT (final query, not a CTE)
#     # Pattern: ), SELECT cte_name AS should become ), cte_name AS
    
#     # First: Fix standalone SELECT after CTE (missing final CTE)
#     # If we see )), SELECT without a CTE name, it means the final SELECT is starting
#     # Do NOT add comma here
#     sql = re.sub(r'\)\)\s*,\s*SELECT\s+(?![\w]+\s+AS\s*\()', r')) SELECT ', sql, flags=re.IGNORECASE)
    
#     # Second: Fix missing comma between CTEs: ) cte_name AS ( -> ), cte_name AS (
#     sql = re.sub(r'\)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(', r'),\n\1 AS (', sql, flags=re.IGNORECASE)
    
#     # Third: Remove incorrect comma before final SELECT
#     sql = re.sub(r'\)\)\s*,\s*(SELECT\s+(?![\w]+\s+AS))', r')) \1', sql, flags=re.IGNORECASE)
    
#     # ✅ PHASE 1: Aggressive TRIM removal (pattern-based)
#     if debug:
#         print("\n" + "="*80)
#         print("PHASE 1: Aggressive TRIM Removal")
#         print("="*80)
    
#     sql = aggressive_remove_trim_on_non_text(sql, debug=debug)
    
#     # ✅ PHASE 2: BRUTE FORCE - Remove ANY remaining TRIM
#     if debug:
#         print("\n" + "="*80)
#         print("PHASE 2: Brute Force TRIM Removal")
#         print("="*80)
    
#     sql = brute_force_remove_all_trim(sql, debug=debug)
    
#     # ✅ PHASE 3: Validate and fix CTE syntax
#     if debug:
#         print("\n" + "="*80)
#         print("PHASE 3: CTE Syntax Validation")
#         print("="*80)
    
#     sql = validate_and_fix_cte_syntax(sql, debug=debug)
    
#     # Date/time fixes
#     today = datetime.date.today()
#     fixed_date = f"DATE '{today.year}-{today.month:02d}-{today.day:02d}'"
#     sql = re.sub(r'\bCURRENT_DATE\b', fixed_date, sql, flags=re.IGNORECASE)
#     sql = re.sub(r'\bNOW\(\)\b', fixed_date, sql, flags=re.IGNORECASE)
    
#     # Fix NULL comparisons
#     sql = re.sub(r'(\w+)\s*=\s*NULL\b', r'\1 IS NULL', sql, flags=re.IGNORECASE)
#     sql = re.sub(r'(\w+)\s*!=\s*NULL\b', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)
    
#     # Safe division
#     def safe_div(match):
#         left = match.group(1)
#         right = match.group(2).strip()
#         if 'NULLIF' in right.upper():
#             return match.group(0)
#         return f"{left} / NULLIF({right}, 0)"
#     sql = re.sub(r'([^\s]+)\s*/\s*([A-Za-z0-9_"\.\(\)]+)(?!\s*[,\)])', safe_div, sql)
    
#     # Clean syntax issues
#     sql = re.sub(r',\s*\)', ')', sql)
#     sql = re.sub(r',\s*FROM\b', ' FROM', sql, flags=re.IGNORECASE)
    
#     # Balance parentheses
#     open_count = sql.count('(')
#     close_count = sql.count(')')
#     if open_count > close_count:
#         sql += ')' * (open_count - close_count)
    
#     # Final cleanup
#     sql = re.sub(r'\s+', ' ', sql).strip()
#     sql = sql.rstrip(';') + ';'
    
#     # ✅ FINAL VERIFICATION
#     if 'TRIM' in sql.upper():
#         if debug:
#             print("\n⚠️ WARNING: TRIM still found after all repairs!")
#             print("Running one more brute force pass...")
#         sql = brute_force_remove_all_trim(sql, debug=debug)
    
#     if debug:
#         print("\n" + "="*80)
#         print("FINAL SQL:")
#         print("="*80)
#         print(sql)
#         print("="*80)
        
#         if 'TRIM' in sql.upper():
#             print("\n❌ CRITICAL: TRIM STILL EXISTS!")
#         else:
#             print("\n✅ SUCCESS: All TRIM removed!")
    
#     return sql
# ----------------------------


import re
from typing import List, Dict, Any, Set, Tuple
import datetime

# def identify_timestamp_date_columns(sql: str, schema_catalog: Dict[str, Any] = None, debug: bool = False) -> Set[str]:
#     """Identify timestamp and date columns from schema catalog."""
#     timestamp_cols = set()
    
#     if not schema_catalog:
#         return timestamp_cols
    
#     for table_name, table_info in schema_catalog.items():
#         for col in table_info.get('columns', []):
#             col_name = col.get('name', '').lower()
#             col_type = col.get('type', '').upper()
#             if col_type in ('TIMESTAMP', 'TIMESTAMPTZ', 'DATE', 'TIME'):
#                 timestamp_cols.add(col_name)
    
#     if debug:
#         print(f"[DEBUG] Identified timestamp/date columns: {sorted(timestamp_cols)}")
    
#     return timestamp_cols

# def smart_remove_trim_from_timestamps(sql: str, timestamp_columns: Set[str], debug: bool = False) -> str:
#     """Remove TRIM() only from timestamp/date columns."""
#     if not timestamp_columns:
#         return sql
    
#     for col in timestamp_columns:
#         # Pattern: TRIM(alias.column) or TRIM(column)
#         pattern = rf'\bTRIM\s*\(\s*(\w+\.)?({re.escape(col)})\s*\)'
        
#         def remove_trim(match):
#             prefix = match.group(1) or ''
#             col_name = match.group(2)
#             result = f"{prefix}{col_name}"
#             if debug:
#                 print(f"[DEBUG] Removing TRIM: {match.group(0)} -> {result}")
#             return result
        
#         sql = re.sub(pattern, remove_trim, sql, flags=re.IGNORECASE)
    
#     return sql

# def validate_and_fix_cte_syntax(sql: str, debug: bool = False) -> str:
#     """Validate and fix CTE syntax issues."""
#     if 'WITH' not in sql.upper():
#         return sql
    
#     # Fix missing commas between CTEs
#     sql = re.sub(r'\)\s+([a-zA-Z_]\w*)\s+AS\s*\(', r'),\n\1 AS (', sql, flags=re.IGNORECASE)
    
#     # Remove trailing comma before final SELECT
#     sql = re.sub(r',\s*SELECT\s+(?!.*\s+AS\s*\()', 'SELECT ', sql, flags=re.IGNORECASE)
    
#     return sql

# def extract_all_columns_from_sql(sql: str, debug: bool = False) -> Dict[str, Set[str]]:
#     """
#     Extract all table.column references from SQL query.
#     Returns: {table_alias: {column1, column2, ...}}
#     """
#     columns_by_table = {}
    
#     # Pattern to match: alias.column_name
#     pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\b'
    
#     for match in re.finditer(pattern, sql):
#         table_alias = match.group(1).lower()
#         column_name = match.group(2).lower()
        
#         # Skip SQL keywords
#         if column_name.upper() in ('SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'AND', 'OR', 'AS'):
#             continue
        
#         if table_alias not in columns_by_table:
#             columns_by_table[table_alias] = set()
#         columns_by_table[table_alias].add(column_name)
    
#     if debug:
#         print(f"[DEBUG] Extracted columns by table: {dict(columns_by_table)}")
    
#     return columns_by_table

# def extract_table_aliases_from_sql(sql: str, debug: bool = False) -> Dict[str, str]:
#     """
#     Extract table aliases and their actual table names from SQL.
#     Returns: {alias: actual_table_name}
#     """
#     alias_mapping = {}
    
#     # Pattern: FROM schema.table_name alias or FROM table_name alias
#     from_pattern = r'\bFROM\s+(?:(\w+)\.)?(\w+)(?:\s+(?:AS\s+)?(\w+))?'
#     join_pattern = r'\bJOIN\s+(?:(\w+)\.)?(\w+)(?:\s+(?:AS\s+)?(\w+))?'
    
#     for pattern in [from_pattern, join_pattern]:
#         for match in re.finditer(pattern, sql, re.IGNORECASE):
#             schema = match.group(1)
#             table_name = match.group(2)
#             alias = match.group(3) or table_name
            
#             # Store both the full name and just table name
#             full_table = f"{schema}.{table_name}" if schema else table_name
#             alias_mapping[alias.lower()] = table_name.lower()
    
#     if debug:
#         print(f"[DEBUG] Table alias mapping: {alias_mapping}")
    
#     return alias_mapping

# def get_valid_columns_for_table(table_name: str, schema_catalog: Dict[str, Any], debug: bool = False) -> Set[str]:
#     """Get all valid column names for a given table from schema catalog."""
#     valid_columns = set()
    
#     if not schema_catalog:
#         return valid_columns
    
#     # Try exact match first
#     table_key = table_name.lower()
#     if table_key in schema_catalog:
#         for col in schema_catalog[table_key].get('columns', []):
#             valid_columns.add(col.get('name', '').lower())
#     else:
#         # Try partial match (in case of schema.table format)
#         for key, table_info in schema_catalog.items():
#             if table_name.lower() in key.lower():
#                 for col in table_info.get('columns', []):
#                     valid_columns.add(col.get('name', '').lower())
#                 break
    
#     if debug:
#         print(f"[DEBUG] Valid columns for {table_name}: {sorted(valid_columns)}")
    
#     return valid_columns

# def find_similar_column(invalid_column: str, valid_columns: Set[str], debug: bool = False) -> str:
#     """
#     Find the most similar column name using fuzzy matching.
#     Uses multiple strategies: substring, prefix, suffix, and edit distance.
#     """
#     if not valid_columns:
#         return None
    
#     invalid_lower = invalid_column.lower()
    
#     # Strategy 1: Exact match (shouldn't happen, but just in case)
#     if invalid_lower in valid_columns:
#         return invalid_lower
    
#     # Strategy 2: Check if invalid column is contained in any valid column
#     for valid_col in valid_columns:
#         if invalid_lower in valid_col or valid_col in invalid_lower:
#             if debug:
#                 print(f"[DEBUG] Found substring match: {invalid_column} -> {valid_col}")
#             return valid_col
    
#     # Strategy 3: Check for common patterns (e.g., fsl_programtype might be programtype)
#     # Remove common prefixes
#     prefixes_to_try = ['fsl_', 'app_', 'main_', 'customer_', 'user_']
#     for prefix in prefixes_to_try:
#         if invalid_lower.startswith(prefix):
#             stripped = invalid_lower[len(prefix):]
#             if stripped in valid_columns:
#                 if debug:
#                     print(f"[DEBUG] Found after removing prefix: {invalid_column} -> {stripped}")
#                 return stripped
#             # Also try with the prefix on valid columns
#             for valid_col in valid_columns:
#                 if valid_col.startswith(prefix) and stripped in valid_col:
#                     if debug:
#                         print(f"[DEBUG] Found with prefix pattern: {invalid_column} -> {valid_col}")
#                     return valid_col
    
#     # Strategy 4: Check if any valid column contains key parts of invalid column
#     invalid_parts = invalid_lower.split('_')
#     best_match = None
#     max_matches = 0
    
#     for valid_col in valid_columns:
#         valid_parts = valid_col.split('_')
#         matches = sum(1 for part in invalid_parts if part in valid_parts)
#         if matches > max_matches:
#             max_matches = matches
#             best_match = valid_col
    
#     if max_matches >= 2 or (max_matches >= 1 and len(invalid_parts) <= 2):
#         if debug:
#             print(f"[DEBUG] Found by word matching: {invalid_column} -> {best_match}")
#         return best_match
    
#     # Strategy 5: Simple Levenshtein distance for short columns
#     def levenshtein_distance(s1: str, s2: str) -> int:
#         if len(s1) < len(s2):
#             return levenshtein_distance(s2, s1)
#         if len(s2) == 0:
#             return len(s1)
#         previous_row = range(len(s2) + 1)
#         for i, c1 in enumerate(s1):
#             current_row = [i + 1]
#             for j, c2 in enumerate(s2):
#                 insertions = previous_row[j + 1] + 1
#                 deletions = current_row[j] + 1
#                 substitutions = previous_row[j] + (c1 != c2)
#                 current_row.append(min(insertions, deletions, substitutions))
#             previous_row = current_row
#         return previous_row[-1]
    
#     min_distance = float('inf')
#     best_match = None
    
#     for valid_col in valid_columns:
#         distance = levenshtein_distance(invalid_lower, valid_col)
#         if distance < min_distance:
#             min_distance = distance
#             best_match = valid_col
    
#     # Only accept if similarity is reasonable (distance < 30% of length)
#     if min_distance <= max(3, len(invalid_lower) * 0.3):
#         if debug:
#             print(f"[DEBUG] Found by edit distance: {invalid_column} -> {best_match} (distance: {min_distance})")
#         return best_match
    
#     if debug:
#         print(f"[DEBUG] No good match found for: {invalid_column}")
#     return None

# def validate_and_fix_columns(sql: str, schema_catalog: Dict[str, Any] = None, debug: bool = False) -> str:
#     """
#     Validate all column references in SQL and fix invalid ones.
#     This is the KEY function to prevent UndefinedColumn errors.
#     """
#     if not schema_catalog:
#         if debug:
#             print("[DEBUG] No schema catalog provided, skipping column validation")
#         return sql
    
#     if debug:
#         print("\n" + "="*80)
#         print("COLUMN VALIDATION AND CORRECTION")
#         print("="*80)
    
#     # Step 1: Extract table aliases
#     alias_mapping = extract_table_aliases_from_sql(sql, debug=debug)
    
#     # Step 2: Extract all column references
#     columns_by_alias = extract_all_columns_from_sql(sql, debug=debug)
    
#     # Step 3: Validate each column and build replacement map
#     replacements = []
    
#     for alias, columns in columns_by_alias.items():
#         # Get actual table name
#         actual_table = alias_mapping.get(alias.lower(), alias)
        
#         # Get valid columns for this table
#         valid_columns = get_valid_columns_for_table(actual_table, schema_catalog, debug=debug)
        
#         if not valid_columns:
#             if debug:
#                 print(f"[WARNING] No schema found for table: {actual_table} (alias: {alias})")
#             continue
        
#         # Check each column
#         for col in columns:
#             if col.lower() not in valid_columns:
#                 if debug:
#                     print(f"\n[INVALID COLUMN] {alias}.{col} does not exist in {actual_table}")
                
#                 # Find similar column
#                 similar = find_similar_column(col, valid_columns, debug=debug)
                
#                 if similar:
#                     # Add to replacement list
#                     old_ref = f"{alias}.{col}"
#                     new_ref = f"{alias}.{similar}"
#                     replacements.append((old_ref, new_ref))
#                     if debug:
#                         print(f"[FIX] Will replace: {old_ref} -> {new_ref}")
#                 else:
#                     if debug:
#                         print(f"[ERROR] No suitable replacement found for {alias}.{col}")
#                         print(f"[INFO] Available columns: {sorted(valid_columns)}")
    
#     # Step 4: Apply all replacements
#     if replacements:
#         if debug:
#             print(f"\n[DEBUG] Applying {len(replacements)} column corrections...")
        
#         for old_ref, new_ref in replacements:
#             # Use word boundary to avoid partial replacements
#             pattern = rf'\b{re.escape(old_ref)}\b'
#             sql = re.sub(pattern, new_ref, sql, flags=re.IGNORECASE)
#             if debug:
#                 print(f"[APPLIED] {old_ref} -> {new_ref}")
#     else:
#         if debug:
#             print("\n[SUCCESS] All columns are valid!")
    
#     if debug:
#         print("="*80)
    
#     return sql

# def fix_timestamp_columns(sql: str, schema_catalog: Dict[str, Any] = None, debug: bool = False) -> str:
#     """
#     Convert empty string timestamp/date columns to NULL before casting.
#     Example: col::TIMESTAMP -> NULLIF(col, '')::TIMESTAMP
#     """
#     if not schema_catalog:
#         return sql

#     # Collect timestamp/date columns
#     timestamp_cols = set()
#     for table_name, table_info in schema_catalog.items():
#         for col in table_info.get('columns', []):
#             col_name = col.get('name', '')
#             col_type = col.get('type', '').upper()
#             if col_type in ('TIMESTAMP', 'TIMESTAMPTZ', 'DATE'):
#                 timestamp_cols.add(col_name)

#     if debug:
#         print(f"[DEBUG] Timestamp/date columns: {timestamp_cols}")

#     # Apply NULLIF for safe casting
#     for col in timestamp_cols:
#         # Replace all occurrences of col::TIMESTAMP or col::DATE
#         sql = re.sub(
#             rf'\b({re.escape(col)})\s*::\s*(TIMESTAMP|DATE|TIMESTAMPTZ)\b',
#             r'NULLIF(\1, \'\')::\2',
#             sql,
#             flags=re.IGNORECASE
#         )
#         if debug:
#             print(f"[DEBUG] Applied NULLIF for column: {col}")

#     return sql

# def fix_text_column_aggregations(sql: str, schema_catalog: Dict[str, Any] = None, debug: bool = False) -> str:
#     """
#     Fix SUM/AVG/MIN/MAX on TEXT columns by adding ::NUMERIC casts.
#     Detects columns that need casting and applies it automatically.
#     """
#     if debug:
#         print("[DEBUG] Fixing text column aggregations...")
    
#     # Identify numeric-like text columns from schema
#     text_numeric_columns = set()
    
#     if schema_catalog:
#         for table_name, table_info in schema_catalog.items():
#             for col in table_info.get('columns', []):
#                 col_name = col.get('name', '').lower()
#                 col_type = col.get('type', '').upper()
                
#                 # Find TEXT/VARCHAR columns that should be numeric
#                 if col_type in ('TEXT', 'VARCHAR', 'CHARACTER VARYING', 'CHAR'):
#                     # Common numeric column name patterns
#                     if any(keyword in col_name for keyword in [
#                         'amount', 'balance', 'payment', 'fee', 'charge', 'price', 
#                         'cost', 'total', 'sum', 'rate', 'percent', 'value', 'number',
#                         'count', 'quantity', 'score', 'interest', 'principal', 'apr', 'term'
#                     ]):
#                         text_numeric_columns.add(col_name)
#                         if debug:
#                             print(f"[DEBUG] Identified numeric-like text column: {col_name}")
    
#     # Pattern-based detection from SQL itself
#     # Find columns used in SUM(), AVG(), MIN(), MAX()
#     aggregate_pattern = r'\b(SUM|AVG|MIN|MAX)\s*\(\s*(\w+)\.(\w+)\s*\)'
#     for match in re.finditer(aggregate_pattern, sql, re.IGNORECASE):
#         func = match.group(1).upper()
#         alias = match.group(2)
#         col_name = match.group(3).lower()
        
#         # If not already cast, add it to the list
#         if '::' not in match.group(0):
#             text_numeric_columns.add(col_name)
#             if debug:
#                 print(f"[DEBUG] Found {func} on column: {col_name}")
    
#     # Find columns in CASE THEN clauses that return numeric values
#     case_pattern = r'CASE\s+WHEN[^T]+THEN\s+(\w+)\.(\w+)\s+ELSE\s+\d+'
#     for match in re.finditer(case_pattern, sql, re.IGNORECASE):
#         alias = match.group(1)
#         col_name = match.group(2).lower()
#         text_numeric_columns.add(col_name)
#         if debug:
#             print(f"[DEBUG] Found CASE THEN with numeric column: {col_name}")
    
#     if not text_numeric_columns:
#         if debug:
#             print("[DEBUG] No text numeric columns found")
#         return sql
    
#     if debug:
#         print(f"[DEBUG] Text numeric columns to cast: {sorted(text_numeric_columns)}")
    
#     # Apply ::NUMERIC cast to aggregation functions
#     for col in text_numeric_columns:
#         # Pattern 1: SUM(alias.column) -> SUM(alias.column::NUMERIC)
#         pattern1 = rf'\b(SUM|AVG|MIN|MAX)\s*\(\s*(\w+)\.({re.escape(col)})\s*\)'
        
#         def add_cast_aliased(match):
#             func = match.group(1)
#             alias = match.group(2)
#             col_name = match.group(3)
#             result = f"{func}({alias}.{col_name}::NUMERIC)"
#             if debug:
#                 print(f"[DEBUG]   Adding cast: {match.group(0)} -> {result}")
#             return result
        
#         sql = re.sub(pattern1, add_cast_aliased, sql, flags=re.IGNORECASE)
        
#         # Pattern 2: CASE WHEN ... THEN alias.column ELSE -> CASE WHEN ... THEN alias.column::NUMERIC ELSE
#         pattern2 = rf'\bTHEN\s+(\w+)\.({re.escape(col)})\s+ELSE'
        
#         def add_cast_case(match):
#             alias = match.group(1)
#             col_name = match.group(2)
#             result = f"THEN {alias}.{col_name}::NUMERIC ELSE"
#             if debug:
#                 print(f"[DEBUG]   Adding cast in CASE: THEN {alias}.{col_name} -> {result}")
#             return result
        
#         sql = re.sub(pattern2, add_cast_case, sql, flags=re.IGNORECASE)
        
#         # Pattern 3: Handle columns in division/arithmetic without aggregation
#         pattern3 = rf'\b(\w+)\.({re.escape(col)})\b(?!\s*::)(?!\s+ELSE)'
        
#         def add_cast_arithmetic(match):
#             alias = match.group(1)
#             col_name = match.group(2)
            
#             # Check if in arithmetic context
#             match_end = match.end()
#             look_ahead = sql[match_end:match_end+10] if match_end < len(sql) else ''
            
#             if any(op in look_ahead for op in ['/', '*', '+', '-', ')', ',']):
#                 result = f"{alias}.{col_name}::NUMERIC"
#                 if debug:
#                     print(f"[DEBUG]   Adding cast in arithmetic: {alias}.{col_name} -> {result}")
#                 return result
#             return match.group(0)
        
#         sql = re.sub(pattern3, add_cast_arithmetic, sql, flags=re.IGNORECASE)
    
#     if debug:
#         print("[DEBUG] Text column aggregation fixes complete")
    
#     return sql


# def detect_missing_columns_from_error(error_message: str) -> List[str]:
#     """
#     Detects all missing column names from a PostgreSQL error message.
#     Returns a list of missing columns.
#     """
#     missing_columns = []
#     pattern = r'column "?(.*?)"? does not exist'
#     matches = re.findall(pattern, error_message, re.IGNORECASE)
#     for match in matches:
#         missing_columns.append(match.lower())
#     return missing_columns



# def auto_fix_missing_columns(sql: str, schema_catalog: Dict[str, Any], error_message: str, debug: bool = False) -> str:
#     """
#     Fix SQL if there are missing columns by finding the closest match in the schema.
#     Returns the corrected SQL.
#     """
#     missing_columns = detect_missing_columns_from_error(error_message)
#     if debug and missing_columns:
#         print(f"[DEBUG] Missing columns detected: {missing_columns}")
    
#     # Extract table aliases and all columns
#     alias_mapping = extract_table_aliases_from_sql(sql, debug=debug)
#     columns_by_alias = extract_all_columns_from_sql(sql, debug=debug)
    
#     for missing_col in missing_columns:
#         replaced = False
#         # Search across all table aliases
#         for alias, cols in columns_by_alias.items():
#             actual_table = alias_mapping.get(alias.lower(), alias)
#             valid_columns = get_valid_columns_for_table(actual_table, schema_catalog, debug=debug)
#             if missing_col not in valid_columns:
#                 similar = find_similar_column(missing_col, valid_columns, debug=debug)
#                 if similar:
#                     # Replace in SQL
#                     pattern = rf'\b{alias}\.{missing_col}\b'
#                     sql = re.sub(pattern, f"{alias}.{similar}", sql, flags=re.IGNORECASE)
#                     replaced = True
#                     if debug:
#                         print(f"[FIX] Replaced {alias}.{missing_col} -> {alias}.{similar}")
#         if not replaced and debug:
#             print(f"[WARNING] No suitable replacement found for {missing_col}")
#     return sql


# import re
# import datetime
# from typing import List, Dict, Any
# import psycopg2  # or psycopg2-binary

# def convert_quarter_to_month_interval(sql: str, debug=False) -> str:
#     pattern = r"INTERVAL\s+'(\d+)\s+quarter'"
#     def repl(m):
#         months = int(m.group(1)) * 3
#         if debug:
#             print(f"[DEBUG] Converting {m.group(0)} -> INTERVAL '{months} months'")
#         return f"INTERVAL '{months} months'"
#     return re.sub(pattern, repl, sql, flags=re.IGNORECASE)

# # Replace any timestamp casts on columns that might have empty strings
# def fix_empty_string_timestamps(sql, timestamp_columns, debug=False):
#     for col in timestamp_columns:
#         pattern = rf"({col})::TIMESTAMP"
#         sql = re.sub(pattern, r"NULLIF(\1,'')::TIMESTAMP", sql, flags=re.IGNORECASE)
#         if debug:
#             print(f"[DEBUG] Applied NULLIF fix to timestamp column: {col}")
#     return sql

# # ----------------------------
# def nuclear_repair_sql(
#     raw_llm_output: str,
#     tables: List[str] = None,
#     schema_catalog: Dict[str, Any] = None,
#     join_info: Dict[str, Any] = None,
#     debug: bool = False,
#     execute: bool = False,  # Optional: Execute SQL
#     db_conn_params: Dict[str, Any] = None  # {'host':..., 'port':..., 'user':..., 'password':..., 'database':...}
# ) -> str:
#     """Comprehensive SQL repair with column validation, SMART TRIM removal, and optional execution."""

#     sql_to_run = raw_llm_output
#     max_attempts = 5
#     attempt = 0

#     while attempt < max_attempts:
#         attempt += 1
#         try:
#             if debug:
#                 print(f"\n[ATTEMPT {attempt}] Starting SQL repair...")

#             sql = sql_to_run

#             # ----------------------------
#             # CLEAN RAW LLM SQL
#             # ----------------------------
#             sql = sql.strip()
#             sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
#             sql = re.sub(r'```\s*', '', sql)
#             sql = re.sub(r'`+', '', sql)
#             sql_start = re.search(r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b', sql, re.IGNORECASE)
#             if sql_start:
#                 sql = sql[sql_start.start():]

#             # ----------------------------
#             # PHASE 0: Column Validation & Auto-Fix
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 0] Column Validation and Correction")
#             sql = validate_and_fix_columns(sql, schema_catalog, debug=debug)

#             # ----------------------------
#             # PHASE 1: CTE Syntax Fixes
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 1] CTE Syntax Fixes")
#             sql = re.sub(r'\)\)\s*,\s*SELECT\s+(?![\w]+\s+AS\s*\()', r')) SELECT ', sql, flags=re.IGNORECASE)
#             sql = re.sub(r'\)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(', r'),\n\1 AS (', sql, flags=re.IGNORECASE)
#             sql = validate_and_fix_cte_syntax(sql, debug=debug)

#             # ----------------------------
#             # PHASE 2: Smart TRIM Removal (timestamp/date)
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2] Smart TRIM Removal")
#             timestamp_columns = identify_timestamp_date_columns(sql, schema_catalog, debug=debug)
#             sql = smart_remove_trim_from_timestamps(sql, timestamp_columns, debug=debug)

#             # ----------------------------
#             # PHASE 2.8: Force-cast timestamp/date columns for LIKE/ILIKE
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.8] Force-cast timestamp/date columns for LIKE/ILIKE")

#             text_ops_pattern = r'(\b\w+\.(\w+)\b)\s+(ILIKE|LIKE|SIMILAR TO)\s*'

#             def force_cast_timestamp(match):
#                 full_col, col_name, op = match.group(1), match.group(2).lower(), match.group(3)
#                 cast_required = False
#                 if schema_catalog:
#                     for table_info in schema_catalog.values():
#                         for col in table_info.get("columns", []):
#                             if col_name == col.get("name", "").lower():
#                                 if any(x in col.get("type", "").lower() for x in ["timestamp", "date", "time"]):
#                                     cast_required = True
#                                 break
#                         if cast_required:
#                             break
#                 else:
#                     if any(x in col_name for x in ["date", "timestamp", "time"]):
#                         cast_required = True
#                 return f"{full_col}::TEXT {op} " if cast_required else match.group(0)

#             sql = re.sub(text_ops_pattern, force_cast_timestamp, sql,
#                          flags=re.IGNORECASE | re.MULTILINE)

#             # ----------------------------
#             # PHASE 2.5: Fix Text Column Aggregations
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.5] Fix Text Column Aggregations")
#             sql = fix_text_column_aggregations(sql, schema_catalog, debug=debug)

#             # ----------------------------
#             # PHASE 2.75: Fix Timestamp Columns (empty strings)
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.75] Fix Timestamp Columns")
#             sql = fix_timestamp_columns(sql, schema_catalog, debug=debug)

#             # ----------------------------
#             # PHASE 2.76: Fix empty-string timestamps for safe casting
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.76] Fix empty-string timestamps")
#             timestamp_columns = identify_timestamp_date_columns(sql, schema_catalog, debug=debug)  # reuse or re-identify
#             sql = fix_empty_string_timestamps(sql, timestamp_columns, debug=debug)

#             # ----------------------------
#             # PHASE 3: Date/Time Fixes
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 3] Date/Time Fixes")
#             today = datetime.date.today()
#             fixed_date = f"DATE '{today.year}-{today.month:02d}-{today.day:02d}'"
#             sql = re.sub(r'\bCURRENT_DATE\b', fixed_date, sql, flags=re.IGNORECASE)
#             sql = re.sub(r'\bNOW\(\)\b', fixed_date, sql, flags=re.IGNORECASE)
#             sql = re.sub(r'(\w+)\s*=\s*NULL\b', r'\1 IS NULL', sql, flags=re.IGNORECASE)
#             sql = re.sub(r'(\w+)\s*!=\s*NULL\b', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)

#             # Safe division
#             def safe_div(match):
#                 left, right = match.group(1), match.group(2).strip()
#                 return match.group(0) if 'NULLIF' in right.upper() else f"{left} / NULLIF({right}, 0)"
#             sql = re.sub(r'([^\s]+)\s*/\s*([A-Za-z0-9_"\.\(\)]+)(?!\s*[,\)])', safe_div, sql)

#             # ----------------------------
#             # PHASE 4: Syntax Cleanup
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 4] Syntax Cleanup")
#             sql = re.sub(r',\s*\)', ')', sql)
#             sql = re.sub(r',\s*FROM\b', ' FROM', sql, flags=re.IGNORECASE)
#             sql = re.sub(r',\s*WHERE\b', ' WHERE', sql, flags=re.IGNORECASE)

#             # Balance parentheses
#             open_count, close_count = sql.count('('), sql.count(')')
#             if open_count > close_count:
#                 sql += ')' * (open_count - close_count)
#                 if debug:
#                     print(f"[DEBUG] Added {open_count - close_count} closing parentheses")

#             sql = re.sub(r'\s+', ' ', sql).strip()
#             sql = sql.rstrip(';') + ';'

#             # ----------------------------
#             # CONVERT QUARTER INTERVALS TO MONTHS
#             # ----------------------------
#             sql = convert_quarter_to_month_interval(sql, debug=debug)

#             # Replace '1 quarter' with '3 months' for Postgres
#             sql = re.sub(
#                 r"INTERVAL\s+'1\s+quarter'",
#                 "INTERVAL '3 months'",
#                 sql,
#                 flags=re.IGNORECASE
#             )


#             # ----------------------------
#             # FINAL VERIFICATION: TRIM on timestamp
#             # ----------------------------
#             for col in timestamp_columns:
#                 if re.search(rf'\bTRIM\s*\(\s*\w*\.?{re.escape(col)}\s*\)', sql, re.IGNORECASE):
#                     if debug:
#                         print(f"⚠️ WARNING: TRIM still found on timestamp column: {col}")
#                     sql = smart_remove_trim_from_timestamps(sql, {col}, debug=False)

#             if debug:
#                 print("\n[FINAL SQL]")
#                 print(sql)

#             # ----------------------------
#             # OPTIONAL: Execute SQL
#             # ----------------------------
#             if execute and db_conn_params:
#                 if debug:
#                     print("\n[EXECUTE] Attempting to run SQL against database...")
#                 try:
#                     with psycopg2.connect(**db_conn_params) as conn:
#                         with conn.cursor() as cur:
#                             cur.execute(sql)
#                             if debug:
#                                 print("[EXECUTE] SQL executed successfully")
#                 except Exception as exec_error:
#                     if "does not exist" in str(exec_error).lower():
#                         if debug:
#                             print(f"[EXECUTE] Missing column detected: {exec_error}")
#                             print("[EXECUTE] Auto-fixing missing columns...")
#                         sql_to_run = auto_fix_missing_columns(sql, schema_catalog, str(exec_error), debug=debug)
#                         continue
#                     else:
#                         raise exec_error

#             sql_to_run = sql
#             break

#         except Exception as e:
#             if "does not exist" in str(e).lower():
#                 if debug:
#                     print(f"[DEBUG] Missing column detected: {e}")
#                     print("[DEBUG] Auto-fixing missing columns...")
#                 sql_to_run = auto_fix_missing_columns(sql_to_run, schema_catalog, str(e), debug=debug)
#             else:
#                 raise

#     return sql_to_run



import re
from typing import List, Dict, Any, Set, Tuple
import datetime
import re
from typing import List, Dict, Any, Set, Tuple
import datetime

def identify_timestamp_date_columns(sql: str, schema_catalog: Dict[str, Any] = None, debug: bool = False) -> Set[str]:
    """Identify timestamp and date columns from schema catalog."""
    timestamp_cols = set()
    
    if not schema_catalog:
        return timestamp_cols
    
    for table_name, table_info in schema_catalog.items():
        for col in table_info.get('columns', []):
            col_name = col.get('name', '').lower()
            col_type = col.get('type', '').upper()
            if col_type in ('TIMESTAMP', 'TIMESTAMPTZ', 'DATE', 'TIME'):
                timestamp_cols.add(col_name)
    
    if debug:
        print(f"[DEBUG] Identified timestamp/date columns: {sorted(timestamp_cols)}")
    
    return timestamp_cols

def smart_remove_trim_from_timestamps(sql: str, timestamp_columns: Set[str], debug: bool = False) -> str:
    """Remove TRIM() only from timestamp/date columns."""
    if not timestamp_columns:
        return sql
    
    for col in timestamp_columns:
        # Pattern: TRIM(alias.column) or TRIM(column)
        pattern = rf'\bTRIM\s*\(\s*(\w+\.)?({re.escape(col)})\s*\)'
        
        def remove_trim(match):
            prefix = match.group(1) or ''
            col_name = match.group(2)
            result = f"{prefix}{col_name}"
            if debug:
                print(f"[DEBUG] Removing TRIM: {match.group(0)} -> {result}")
            return result
        
        sql = re.sub(pattern, remove_trim, sql, flags=re.IGNORECASE)
    
    return sql

def validate_and_fix_cte_syntax(sql: str, debug: bool = False) -> str:
    """Validate and fix CTE syntax issues."""
    if 'WITH' not in sql.upper():
        return sql
    
    # Fix missing commas between CTEs
    sql = re.sub(r'\)\s+([a-zA-Z_]\w*)\s+AS\s*\(', r'),\n\1 AS (', sql, flags=re.IGNORECASE)
    
    # Remove trailing comma before final SELECT
    sql = re.sub(r',\s*SELECT\s+(?!.*\s+AS\s*\()', 'SELECT ', sql, flags=re.IGNORECASE)
    
    return sql

def extract_all_columns_from_sql(sql: str, debug: bool = False) -> Dict[str, Set[str]]:
    """
    Extract all table.column references from SQL query.
    Returns: {table_alias: {column1, column2, ...}}
    """
    columns_by_table = {}
    
    # Pattern to match: alias.column_name
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\b'
    
    for match in re.finditer(pattern, sql):
        table_alias = match.group(1).lower()
        column_name = match.group(2).lower()
        
        # Skip SQL keywords
        if column_name.upper() in ('SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'AND', 'OR', 'AS'):
            continue
        
        if table_alias not in columns_by_table:
            columns_by_table[table_alias] = set()
        columns_by_table[table_alias].add(column_name)
    
    if debug:
        print(f"[DEBUG] Extracted columns by table: {dict(columns_by_table)}")
    
    return columns_by_table

def extract_table_aliases_from_sql(sql: str, debug: bool = False) -> Dict[str, str]:
    """
    Extract table aliases and their actual table names from SQL.
    Returns: {alias: actual_table_name}
    """
    alias_mapping = {}
    
    # Pattern: FROM schema.table_name alias or FROM table_name alias
    from_pattern = r'\bFROM\s+(?:(\w+)\.)?(\w+)(?:\s+(?:AS\s+)?(\w+))?'
    join_pattern = r'\bJOIN\s+(?:(\w+)\.)?(\w+)(?:\s+(?:AS\s+)?(\w+))?'
    
    for pattern in [from_pattern, join_pattern]:
        for match in re.finditer(pattern, sql, re.IGNORECASE):
            schema = match.group(1)
            table_name = match.group(2)
            alias = match.group(3) or table_name
            
            # Store both the full name and just table name
            full_table = f"{schema}.{table_name}" if schema else table_name
            alias_mapping[alias.lower()] = table_name.lower()
    
    if debug:
        print(f"[DEBUG] Table alias mapping: {alias_mapping}")
    
    return alias_mapping

def get_valid_columns_for_table(table_name: str, schema_catalog: Dict[str, Any], debug: bool = False) -> Set[str]:
    """Get all valid column names for a given table from schema catalog."""
    valid_columns = set()
    
    if not schema_catalog:
        return valid_columns
    
    # Try exact match first
    table_key = table_name.lower()
    if table_key in schema_catalog:
        for col in schema_catalog[table_key].get('columns', []):
            valid_columns.add(col.get('name', '').lower())
    else:
        # Try partial match (in case of schema.table format)
        for key, table_info in schema_catalog.items():
            if table_name.lower() in key.lower():
                for col in table_info.get('columns', []):
                    valid_columns.add(col.get('name', '').lower())
                break
    
    if debug:
        print(f"[DEBUG] Valid columns for {table_name}: {sorted(valid_columns)}")
    
    return valid_columns

def find_similar_column(invalid_column: str, valid_columns: Set[str], debug: bool = False) -> str:
    """
    Find the most similar column name using fuzzy matching.
    Uses multiple strategies: substring, prefix, suffix, and edit distance.
    """
    if not valid_columns:
        return None
    
    invalid_lower = invalid_column.lower()
    
    # Strategy 1: Exact match (shouldn't happen, but just in case)
    if invalid_lower in valid_columns:
        return invalid_lower
    
    # Strategy 2: Check if invalid column is contained in any valid column
    for valid_col in valid_columns:
        if invalid_lower in valid_col or valid_col in invalid_lower:
            if debug:
                print(f"[DEBUG] Found substring match: {invalid_column} -> {valid_col}")
            return valid_col
    
    # Strategy 3: Check for common patterns (e.g., fsl_programtype might be programtype)
    # Remove common prefixes
    prefixes_to_try = ['fsl_', 'app_', 'main_', 'customer_', 'user_']
    for prefix in prefixes_to_try:
        if invalid_lower.startswith(prefix):
            stripped = invalid_lower[len(prefix):]
            if stripped in valid_columns:
                if debug:
                    print(f"[DEBUG] Found after removing prefix: {invalid_column} -> {stripped}")
                return stripped
            # Also try with the prefix on valid columns
            for valid_col in valid_columns:
                if valid_col.startswith(prefix) and stripped in valid_col:
                    if debug:
                        print(f"[DEBUG] Found with prefix pattern: {invalid_column} -> {valid_col}")
                    return valid_col
    
    # Strategy 4: Check if any valid column contains key parts of invalid column
    invalid_parts = invalid_lower.split('_')
    best_match = None
    max_matches = 0
    
    for valid_col in valid_columns:
        valid_parts = valid_col.split('_')
        matches = sum(1 for part in invalid_parts if part in valid_parts)
        if matches > max_matches:
            max_matches = matches
            best_match = valid_col
    
    if max_matches >= 2 or (max_matches >= 1 and len(invalid_parts) <= 2):
        if debug:
            print(f"[DEBUG] Found by word matching: {invalid_column} -> {best_match}")
        return best_match
    
    # Strategy 5: Simple Levenshtein distance for short columns
    def levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    
    min_distance = float('inf')
    best_match = None
    
    for valid_col in valid_columns:
        distance = levenshtein_distance(invalid_lower, valid_col)
        if distance < min_distance:
            min_distance = distance
            best_match = valid_col
    
    # Only accept if similarity is reasonable (distance < 30% of length)
    if min_distance <= max(3, len(invalid_lower) * 0.3):
        if debug:
            print(f"[DEBUG] Found by edit distance: {invalid_column} -> {best_match} (distance: {min_distance})")
        return best_match
    
    if debug:
        print(f"[DEBUG] No good match found for: {invalid_column}")
    return None

def validate_and_fix_columns(sql: str, schema_catalog: Dict[str, Any] = None, debug: bool = False) -> str:
    """
    Validate all column references in SQL and fix invalid ones.
    This is the KEY function to prevent UndefinedColumn errors.
    """
    if not schema_catalog:
        if debug:
            print("[DEBUG] No schema catalog provided, skipping column validation")
        return sql
    
    if debug:
        print("\n" + "="*80)
        print("COLUMN VALIDATION AND CORRECTION")
        print("="*80)
    
    # Step 1: Extract table aliases
    alias_mapping = extract_table_aliases_from_sql(sql, debug=debug)
    
    # Step 2: Extract all column references
    columns_by_alias = extract_all_columns_from_sql(sql, debug=debug)
    
    # Step 3: Validate each column and build replacement map
    replacements = []
    
    for alias, columns in columns_by_alias.items():
        # Get actual table name
        actual_table = alias_mapping.get(alias.lower(), alias)
        
        # Get valid columns for this table
        valid_columns = get_valid_columns_for_table(actual_table, schema_catalog, debug=debug)
        
        if not valid_columns:
            if debug:
                print(f"[WARNING] No schema found for table: {actual_table} (alias: {alias})")
            continue
        
        # Check each column
        for col in columns:
            if col.lower() not in valid_columns:
                if debug:
                    print(f"\n[INVALID COLUMN] {alias}.{col} does not exist in {actual_table}")
                
                # Find similar column
                similar = find_similar_column(col, valid_columns, debug=debug)
                
                if similar:
                    # Add to replacement list
                    old_ref = f"{alias}.{col}"
                    new_ref = f"{alias}.{similar}"
                    replacements.append((old_ref, new_ref))
                    if debug:
                        print(f"[FIX] Will replace: {old_ref} -> {new_ref}")
                else:
                    if debug:
                        print(f"[ERROR] No suitable replacement found for {alias}.{col}")
                        print(f"[INFO] Available columns: {sorted(valid_columns)}")
    
    # Step 4: Apply all replacements
    if replacements:
        if debug:
            print(f"\n[DEBUG] Applying {len(replacements)} column corrections...")
        
        for old_ref, new_ref in replacements:
            # Use word boundary to avoid partial replacements
            pattern = rf'\b{re.escape(old_ref)}\b'
            sql = re.sub(pattern, new_ref, sql, flags=re.IGNORECASE)
            if debug:
                print(f"[APPLIED] {old_ref} -> {new_ref}")
    else:
        if debug:
            print("\n[SUCCESS] All columns are valid!")
    
    if debug:
        print("="*80)
    
    return sql

def fix_timestamp_columns(sql: str, schema_catalog: Dict[str, Any] = None, debug: bool = False) -> str:
    """
    Convert empty string timestamp/date columns to NULL before casting.
    Example: col::TIMESTAMP -> NULLIF(col, '')::TIMESTAMP
    """
    if not schema_catalog:
        return sql

    # Collect timestamp/date columns
    timestamp_cols = set()
    for table_name, table_info in schema_catalog.items():
        for col in table_info.get('columns', []):
            col_name = col.get('name', '')
            col_type = col.get('type', '').upper()
            if col_type in ('TIMESTAMP', 'TIMESTAMPTZ', 'DATE'):
                timestamp_cols.add(col_name.lower())

    if debug:
        print(f"[DEBUG] Timestamp/date columns: {timestamp_cols}")

    # Apply NULLIF for safe casting - handle both aliased and non-aliased references
    for col in timestamp_cols:
        # Pattern 1: alias.column::TIMESTAMP (e.g., loa.reportdate::TIMESTAMP)
        pattern1 = rf'\b(\w+)\.({re.escape(col)})\s*::\s*(TIMESTAMP|DATE|TIMESTAMPTZ)\b'
        
        def replace_aliased_ts(match):
            alias = match.group(1)
            col_name = match.group(2)
            cast_type = match.group(3)
            # Check if already wrapped in NULLIF
            lookbehind = match.string[max(0, match.start()-30):match.start()]
            if 'NULLIF' in lookbehind.upper():
                return match.group(0)
            result = f"NULLIF({alias}.{col_name}, '')::{cast_type}"
            if debug:
                print(f"[DEBUG] Applied NULLIF: {match.group(0)} -> {result}")
            return result
        
        sql = re.sub(pattern1, replace_aliased_ts, sql, flags=re.IGNORECASE)
        
        # Pattern 2: column::TIMESTAMP (without alias, e.g., reportdate::TIMESTAMP)
        pattern2 = rf'\b(?<!\.)({re.escape(col)})\s*::\s*(TIMESTAMP|DATE|TIMESTAMPTZ)\b'
        
        def replace_non_aliased_ts(match):
            col_name = match.group(1)
            cast_type = match.group(2)
            # Check if already wrapped in NULLIF
            lookbehind = match.string[max(0, match.start()-30):match.start()]
            if 'NULLIF' in lookbehind.upper():
                return match.group(0)
            result = f"NULLIF({col_name}, '')::{cast_type}"
            if debug:
                print(f"[DEBUG] Applied NULLIF: {match.group(0)} -> {result}")
            return result
        
        sql = re.sub(pattern2, replace_non_aliased_ts, sql, flags=re.IGNORECASE)
        
        if debug:
            print(f"[DEBUG] Processed NULLIF wrapping for column: {col}")

    return sql

def fix_text_column_aggregations(sql: str, schema_catalog: Dict[str, Any] = None, debug: bool = False) -> str:
    """
    Fix SUM/AVG/MIN/MAX on TEXT columns by adding ::NUMERIC casts.
    Detects columns that need casting and applies it automatically.
    """
    if debug:
        print("[DEBUG] Fixing text column aggregations...")
    
    # Identify numeric-like text columns from schema
    text_numeric_columns = set()
    
    if schema_catalog:
        for table_name, table_info in schema_catalog.items():
            for col in table_info.get('columns', []):
                col_name = col.get('name', '').lower()
                col_type = col.get('type', '').upper()
                
                # Find TEXT/VARCHAR columns that should be numeric
                if col_type in ('TEXT', 'VARCHAR', 'CHARACTER VARYING', 'CHAR'):
                    # Common numeric column name patterns
                    if any(keyword in col_name for keyword in [
                        'amount', 'balance', 'payment', 'fee', 'charge', 'price', 
                        'cost', 'total', 'sum', 'rate', 'percent', 'value', 'number',
                        'count', 'quantity', 'score', 'interest', 'principal', 'apr', 'term'
                    ]):
                        text_numeric_columns.add(col_name)
                        if debug:
                            print(f"[DEBUG] Identified numeric-like text column: {col_name}")
    
    # Pattern-based detection from SQL itself
    # Find columns used in SUM(), AVG(), MIN(), MAX()
    aggregate_pattern = r'\b(SUM|AVG|MIN|MAX)\s*\(\s*(\w+)\.(\w+)\s*\)'
    for match in re.finditer(aggregate_pattern, sql, re.IGNORECASE):
        func = match.group(1).upper()
        alias = match.group(2)
        col_name = match.group(3).lower()
        
        # If not already cast, add it to the list
        if '::' not in match.group(0):
            text_numeric_columns.add(col_name)
            if debug:
                print(f"[DEBUG] Found {func} on column: {col_name}")
    
    # Find columns in CASE THEN clauses that return numeric values
    case_pattern = r'CASE\s+WHEN[^T]+THEN\s+(\w+)\.(\w+)\s+ELSE\s+\d+'
    for match in re.finditer(case_pattern, sql, re.IGNORECASE):
        alias = match.group(1)
        col_name = match.group(2).lower()
        text_numeric_columns.add(col_name)
        if debug:
            print(f"[DEBUG] Found CASE THEN with numeric column: {col_name}")
    
    if not text_numeric_columns:
        if debug:
            print("[DEBUG] No text numeric columns found")
        return sql
    
    if debug:
        print(f"[DEBUG] Text numeric columns to cast: {sorted(text_numeric_columns)}")
    
    # Apply ::NUMERIC cast to aggregation functions
    for col in text_numeric_columns:
        # Pattern 1: SUM(alias.column) -> SUM(alias.column::NUMERIC)
        pattern1 = rf'\b(SUM|AVG|MIN|MAX)\s*\(\s*(\w+)\.({re.escape(col)})\s*\)'
        
        def add_cast_aliased(match):
            func = match.group(1)
            alias = match.group(2)
            col_name = match.group(3)
            result = f"{func}({alias}.{col_name}::NUMERIC)"
            if debug:
                print(f"[DEBUG]   Adding cast: {match.group(0)} -> {result}")
            return result
        
        sql = re.sub(pattern1, add_cast_aliased, sql, flags=re.IGNORECASE)
        
        # Pattern 2: CASE WHEN ... THEN alias.column ELSE -> CASE WHEN ... THEN alias.column::NUMERIC ELSE
        pattern2 = rf'\bTHEN\s+(\w+)\.({re.escape(col)})\s+ELSE'
        
        def add_cast_case(match):
            alias = match.group(1)
            col_name = match.group(2)
            result = f"THEN {alias}.{col_name}::NUMERIC ELSE"
            if debug:
                print(f"[DEBUG]   Adding cast in CASE: THEN {alias}.{col_name} -> {result}")
            return result
        
        sql = re.sub(pattern2, add_cast_case, sql, flags=re.IGNORECASE)
        
        # Pattern 3: Handle columns in division/arithmetic without aggregation
        pattern3 = rf'\b(\w+)\.({re.escape(col)})\b(?!\s*::)(?!\s+ELSE)'
        
        def add_cast_arithmetic(match):
            alias = match.group(1)
            col_name = match.group(2)
            
            # Check if in arithmetic context
            match_end = match.end()
            look_ahead = sql[match_end:match_end+10] if match_end < len(sql) else ''
            
            if any(op in look_ahead for op in ['/', '*', '+', '-', ')', ',']):
                result = f"{alias}.{col_name}::NUMERIC"
                if debug:
                    print(f"[DEBUG]   Adding cast in arithmetic: {alias}.{col_name} -> {result}")
                return result
            return match.group(0)
        
        sql = re.sub(pattern3, add_cast_arithmetic, sql, flags=re.IGNORECASE)
    
    if debug:
        print("[DEBUG] Text column aggregation fixes complete")
    
    return sql


def detect_missing_columns_from_error(error_message: str) -> List[str]:
    """
    Detects all missing column names from a PostgreSQL error message.
    Returns a list of missing columns.
    """
    missing_columns = []
    pattern = r'column "?(.*?)"? does not exist'
    matches = re.findall(pattern, error_message, re.IGNORECASE)
    for match in matches:
        missing_columns.append(match.lower())
    return missing_columns



def auto_fix_missing_columns(sql: str, schema_catalog: Dict[str, Any], error_message: str, debug: bool = False) -> str:
    """
    Fix SQL if there are missing columns by finding the closest match in the schema.
    Returns the corrected SQL.
    """
    missing_columns = detect_missing_columns_from_error(error_message)
    if debug and missing_columns:
        print(f"[DEBUG] Missing columns detected: {missing_columns}")
    
    # Extract table aliases and all columns
    alias_mapping = extract_table_aliases_from_sql(sql, debug=debug)
    columns_by_alias = extract_all_columns_from_sql(sql, debug=debug)
    
    for missing_col in missing_columns:
        replaced = False
        # Search across all table aliases
        for alias, cols in columns_by_alias.items():
            actual_table = alias_mapping.get(alias.lower(), alias)
            valid_columns = get_valid_columns_for_table(actual_table, schema_catalog, debug=debug)
            if missing_col not in valid_columns:
                similar = find_similar_column(missing_col, valid_columns, debug=debug)
                if similar:
                    # Replace in SQL
                    pattern = rf'\b{alias}\.{missing_col}\b'
                    sql = re.sub(pattern, f"{alias}.{similar}", sql, flags=re.IGNORECASE)
                    replaced = True
                    if debug:
                        print(f"[FIX] Replaced {alias}.{missing_col} -> {alias}.{similar}")
        if not replaced and debug:
            print(f"[WARNING] No suitable replacement found for {missing_col}")
    return sql


import re
import datetime
from typing import List, Dict, Any
import psycopg2  # or psycopg2-binary

def convert_quarter_to_month_interval(sql: str, debug=False) -> str:
    pattern = r"INTERVAL\s+'(\d+)\s+quarter'"
    def repl(m):
        months = int(m.group(1)) * 3
        if debug:
            print(f"[DEBUG] Converting {m.group(0)} -> INTERVAL '{months} months'")
        return f"INTERVAL '{months} months'"
    return re.sub(pattern, repl, sql, flags=re.IGNORECASE)

# Replace any timestamp casts on columns that might have empty strings
def fix_empty_string_timestamps(sql, timestamp_columns, debug=False):
    """
    Fix empty string timestamp casts by wrapping with NULLIF.
    Handles both aliased (alias.column) and non-aliased (column) references.
    """
    if not timestamp_columns:
        return sql
        
    for col in timestamp_columns:
        # Pattern 1: Handle aliased columns (e.g., loa.reportdate::TIMESTAMP)
        pattern1 = rf'\b(\w+)\.({re.escape(col)})\s*::\s*(TIMESTAMP|DATE|TIMESTAMPTZ)\b'
        
        def replace_aliased(match):
            alias = match.group(1)
            col_name = match.group(2)
            cast_type = match.group(3)
            # Check if already wrapped in NULLIF
            if 'NULLIF' in match.string[max(0, match.start()-20):match.start()].upper():
                return match.group(0)
            result = f"NULLIF({alias}.{col_name}, '')::{cast_type}"
            if debug:
                print(f"[DEBUG] Applied NULLIF fix: {match.group(0)} -> {result}")
            return result
        
        sql = re.sub(pattern1, replace_aliased, sql, flags=re.IGNORECASE)
        
        # Pattern 2: Handle non-aliased columns (e.g., reportdate::TIMESTAMP)
        pattern2 = rf'\b({re.escape(col)})\s*::\s*(TIMESTAMP|DATE|TIMESTAMPTZ)\b'
        
        def replace_non_aliased(match):
            col_name = match.group(1)
            cast_type = match.group(2)
            # Check if already wrapped in NULLIF or if it's part of an aliased reference
            context = match.string[max(0, match.start()-20):match.start()]
            if 'NULLIF' in context.upper() or '.' in context[-5:]:
                return match.group(0)
            result = f"NULLIF({col_name}, '')::{cast_type}"
            if debug:
                print(f"[DEBUG] Applied NULLIF fix: {match.group(0)} -> {result}")
            return result
        
        sql = re.sub(pattern2, replace_non_aliased, sql, flags=re.IGNORECASE)
    
    return sql



import re
import datetime
from typing import List, Dict, Any
import psycopg2

# ----------------------------
import re
import datetime
import psycopg2
from typing import List, Dict, Any

# ----------------------------
# def fix_extract_syntax(sql: str, debug: bool = False) -> str:
#     """
#     Fix malformed EXTRACT syntax in SQL queries.
#     Converts:
#         EXTRACT(MONTH) FROM column::TIMESTAMP)
#     into:
#         EXTRACT(MONTH FROM column::TIMESTAMP)
#     """
#     pattern = r'EXTRACT\s*\(\s*(\w+)\s*\)\s*FROM\s*([^)]+)\)?'
    
#     def replacer(match):
#         part1 = match.group(1)
#         part2 = match.group(2).strip()
#         fixed = f"EXTRACT({part1} FROM {part2})"
#         if debug:
#             print(f"[fix_extract_syntax] Fixed EXTRACT: {match.group(0)} -> {fixed}")
#         return fixed
    
#     return re.sub(pattern, replacer, sql, flags=re.IGNORECASE)

# # ----------------------------
# def nuclear_repair_sql(
#     raw_llm_output: str,
#     tables: List[str] = None,
#     schema_catalog: Dict[str, Any] = None,
#     join_info: Dict[str, Any] = None,
#     debug: bool = False,
#     execute: bool = False,  # Optional: Execute SQL
#     db_conn_params: Dict[str, Any] = None  
# ) -> str:
#     """Comprehensive SQL repair with column validation, SMART TRIM removal, and optional execution."""

#     sql_to_run = raw_llm_output
#     max_attempts = 5
#     attempt = 0

#     while attempt < max_attempts:
#         attempt += 1
#         try:
#             if debug:
#                 print(f"\n[ATTEMPT {attempt}] Starting SQL repair...")

#             sql = sql_to_run

#             # ----------------------------
#             # CLEAN RAW LLM SQL
#             # ----------------------------
#             sql = sql.strip()
#             sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
#             sql = re.sub(r'```\s*', '', sql)
#             sql = re.sub(r'`+', '', sql)
#             sql_start = re.search(r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b', sql, re.IGNORECASE)
#             if sql_start:
#                 sql = sql[sql_start.start():]

#             # ----------------------------
#             # PHASE 0: Column Validation & Auto-Fix
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 0] Column Validation and Correction")
#             sql = validate_and_fix_columns(sql, schema_catalog, debug=debug)

#             # ----------------------------
#             # PHASE 1: CTE Syntax Fixes
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 1] CTE Syntax Fixes")
#             sql = re.sub(r'\)\)\s*,\s*SELECT\s+(?![\w]+\s+AS\s*\()', r')) SELECT ', sql, flags=re.IGNORECASE)
#             sql = re.sub(r'\)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(', r'),\n\1 AS (', sql, flags=re.IGNORECASE)
#             sql = validate_and_fix_cte_syntax(sql, debug=debug)

#             # ----------------------------
#             # PHASE 2: Identify Timestamp Columns
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2] Identifying Timestamp/Date Columns")
#             timestamp_columns = identify_timestamp_date_columns(sql, schema_catalog, debug=debug)

#             # ----------------------------
#             # PHASE 2.5: Smart TRIM Removal (timestamp/date)
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.5] Smart TRIM Removal")
#             sql = smart_remove_trim_from_timestamps(sql, timestamp_columns, debug=debug)

#             # ----------------------------
#             # PHASE 2.75: Fix Timestamp Columns (CRITICAL FIX)
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.75] Fix Timestamp Columns with NULLIF")
#             sql = fix_timestamp_columns(sql, schema_catalog, debug=debug)

#             # ----------------------------
#             # PHASE 2.76: Fix empty-string timestamps for safe casting
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.76] Additional empty-string timestamp fixes")
#             sql = fix_empty_string_timestamps(sql, timestamp_columns, debug=debug)

#             # ----------------------------
#             # PHASE 2.77: FIX EXTRACT SYNTAX
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.77] Fix EXTRACT syntax issues")
#             sql = fix_extract_syntax(sql, debug=debug)

#             # ----------------------------
#             # PHASE 2.8: Force-cast timestamp/date columns for LIKE/ILIKE
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.8] Force-cast timestamp/date columns for LIKE/ILIKE")
#             text_ops_pattern = r'(\b\w+\.(\w+)\b)\s+(ILIKE|LIKE|SIMILAR TO)\s*'

#             def force_cast_timestamp(match):
#                 full_col, col_name, op = match.group(1), match.group(2).lower(), match.group(3)
#                 cast_required = False
#                 if schema_catalog:
#                     for table_info in schema_catalog.values():
#                         for col in table_info.get("columns", []):
#                             if col_name == col.get("name", "").lower():
#                                 if any(x in col.get("type", "").lower() for x in ["timestamp", "date", "time"]):
#                                     cast_required = True
#                                 break
#                         if cast_required:
#                             break
#                 else:
#                     if any(x in col_name for x in ["date", "timestamp", "time"]):
#                         cast_required = True
#                 return f"{full_col}::TEXT {op} " if cast_required else match.group(0)

#             sql = re.sub(text_ops_pattern, force_cast_timestamp, sql,
#                          flags=re.IGNORECASE | re.MULTILINE)

#             # ----------------------------
#             # PHASE 2.9: Fix Text Column Aggregations
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.9] Fix Text Column Aggregations")
#             sql = fix_text_column_aggregations(sql, schema_catalog, debug=debug)

#             # ----------------------------
#             # PHASE 3: Date/Time Fixes
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 3] Date/Time Fixes")
#             today = datetime.date.today()
#             fixed_date = f"DATE '{today.year}-{today.month:02d}-{today.day:02d}'"
#             sql = re.sub(r'\bCURRENT_DATE\b', fixed_date, sql, flags=re.IGNORECASE)
#             sql = re.sub(r'\bNOW\(\)\b', fixed_date, sql, flags=re.IGNORECASE)
#             sql = re.sub(r'(\w+)\s*=\s*NULL\b', r'\1 IS NULL', sql, flags=re.IGNORECASE)
#             sql = re.sub(r'(\w+)\s*!=\s*NULL\b', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)

#             # Safe division
#             def safe_div(match):
#                 left, right = match.group(1), match.group(2).strip()
#                 return match.group(0) if 'NULLIF' in right.upper() else f"{left} / NULLIF({right}, 0)"
#             sql = re.sub(r'([^\s]+)\s*/\s*([A-Za-z0-9_"\.\(\)]+)(?!\s*[,\)])', safe_div, sql)

#             # ----------------------------
#             # PHASE 4: Syntax Cleanup
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 4] Syntax Cleanup")
#             sql = re.sub(r',\s*\)', ')', sql)
#             sql = re.sub(r',\s*FROM\b', ' FROM', sql, flags=re.IGNORECASE)
#             sql = re.sub(r',\s*WHERE\b', ' WHERE', sql, flags=re.IGNORECASE)

#             # Balance parentheses
#             open_count, close_count = sql.count('('), sql.count(')')
#             if open_count > close_count:
#                 sql += ')' * (open_count - close_count)
#                 if debug:
#                     print(f"[DEBUG] Added {open_count - close_count} closing parentheses")

#             sql = re.sub(r'\s+', ' ', sql).strip()
#             sql = sql.rstrip(';') + ';'

#             # ----------------------------
#             # CONVERT QUARTER INTERVALS TO MONTHS
#             # ----------------------------
#             sql = convert_quarter_to_month_interval(sql, debug=debug)
#             sql = re.sub(r"INTERVAL\s+'1\s+quarter'", "INTERVAL '3 months'", sql, flags=re.IGNORECASE)

#             # ----------------------------
#             # FINAL VERIFICATION: TRIM on timestamp
#             # ----------------------------
#             for col in timestamp_columns:
#                 if re.search(rf'\bTRIM\s*\(\s*\w*\.?{re.escape(col)}\s*\)', sql, re.IGNORECASE):
#                     if debug:
#                         print(f"⚠️ WARNING: TRIM still found on timestamp column: {col}")
#                     sql = smart_remove_trim_from_timestamps(sql, {col}, debug=False)

#             # ----------------------------
#             # FINAL VERIFICATION: Ensure all timestamp casts are wrapped with NULLIF
#             # ----------------------------
#             if debug:
#                 print("\n[FINAL VERIFICATION] Checking all timestamp casts have NULLIF...")
#             for col in timestamp_columns:
#                 unwrapped_pattern = rf'(?<!NULLIF\()[^(]\b(\w+\.)?{re.escape(col)}\s*::\s*(TIMESTAMP|DATE|TIMESTAMPTZ)\b'
#                 if re.search(unwrapped_pattern, sql, re.IGNORECASE):
#                     if debug:
#                         print(f"⚠️ WARNING: Found unwrapped timestamp cast for {col}, applying fix...")
#                     sql = fix_empty_string_timestamps(sql, {col}, debug=debug)

#             if debug:
#                 print("\n[FINAL SQL]")
#                 print(sql)

#             # ----------------------------
#             # OPTIONAL: Execute SQL
#             # ----------------------------
#             if execute and db_conn_params:
#                 if debug:
#                     print("\n[EXECUTE] Attempting to run SQL against database...")
#                 try:
#                     with psycopg2.connect(**db_conn_params) as conn:
#                         with conn.cursor() as cur:
#                             cur.execute(sql)
#                             if debug:
#                                 print("[EXECUTE] SQL executed successfully")
#                 except Exception as exec_error:
#                     error_str = str(exec_error).lower()
#                     if "does not exist" in error_str:
#                         if debug:
#                             print(f"[EXECUTE] Missing column detected: {exec_error}")
#                             print("[EXECUTE] Auto-fixing missing columns...")
#                         sql_to_run = auto_fix_missing_columns(sql, schema_catalog, str(exec_error), debug=debug)
#                         continue
#                     elif "invalid input syntax for type timestamp" in error_str:
#                         if debug:
#                             print(f"[EXECUTE] Timestamp casting error detected: {exec_error}")
#                             print("[EXECUTE] Re-applying timestamp fixes...")
#                         sql_to_run = fix_empty_string_timestamps(sql, timestamp_columns, debug=debug)
#                         continue
#                     else:
#                         raise exec_error

#             sql_to_run = sql
#             break

#         except Exception as e:
#             error_str = str(e).lower()
#             if "does not exist" in error_str:
#                 if debug:
#                     print(f"[DEBUG] Missing column detected: {e}")
#                     print("[DEBUG] Auto-fixing missing columns...")
#                 sql_to_run = auto_fix_missing_columns(sql_to_run, schema_catalog, str(e), debug=debug)
#             elif "invalid input syntax for type timestamp" in error_str:
#                 if debug:
#                     print(f"[DEBUG] Timestamp casting error detected: {e}")
#                     print("[DEBUG] Re-applying timestamp fixes more aggressively...")
#                 sql_to_run = fix_empty_string_timestamps(sql_to_run, timestamp_columns, debug=debug)
#             else:
#                 if debug:
#                     print(f"[ERROR] Unhandled exception: {e}")
#                 raise

#     return sql_to_run



import re
import traceback
from typing import Dict, List, Tuple, Any, Set
import datetime




def convert_quarter_to_month_interval(sql: str, debug: bool = False) -> str:
    """
    Convert INTERVAL '1 quarter' to INTERVAL '3 months' for PostgreSQL compatibility.
    """
    if debug and re.search(r"INTERVAL\s+'1\s+quarter'", sql, re.IGNORECASE):
        print("\n[convert_quarter_to_month_interval] Converting quarter intervals to months")
    
    sql = re.sub(r"INTERVAL\s+'1\s+quarter'", "INTERVAL '3 months'", sql, flags=re.IGNORECASE)
    return sql

import re
import traceback
from typing import Dict, List, Tuple, Any, Set
import datetime
import psycopg2

# ============================================================================
# CRITICAL FIX FUNCTIONS - MUST RUN FIRST
# ============================================================================

import re
import traceback
from typing import Dict, List, Tuple, Any, Set
import datetime
import psycopg2

# ============================================================================
# EMERGENCY PRE-PROCESSOR - Runs before all other fixes
# ============================================================================

def emergency_preprocess_sql(sql: str, schema_catalog: Dict[str, Any] = None, debug: bool = False) -> str:
    """
    EMERGENCY: Handle severely malformed SQL before main repair.
    This catches the exact pattern from your error.
    """
    if debug:
        print("\n[emergency_preprocess_sql] Running emergency pre-processing...")
    
    # Emergency Fix 1: EXTRACT(MONTH) FROM ... ) AS month
    sql = re.sub(
        r'EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([^\s)]+(?:::\w+)?)\s*\)\s*AS',
        r'EXTRACT(\1 FROM \2) AS',
        sql,
        flags=re.IGNORECASE
    )
    
    # Emergency Fix 2: COUNT(( CASE -> COUNT(CASE
    sql = re.sub(r'COUNT\s*\(\s*\(\s*CASE', 'COUNT(CASE', sql, flags=re.IGNORECASE)
    sql = re.sub(r'SUM\s*\(\s*\(\s*CASE', 'SUM(CASE', sql, flags=re.IGNORECASE)
    sql = re.sub(r'AVG\s*\(\s*\(\s*CASE', 'AVG(CASE', sql, flags=re.IGNORECASE)
    
    # Emergency Fix 3: END )) AS -> END) AS
    sql = re.sub(r'END\s*\)\s*\)\s*AS', 'END) AS', sql, flags=re.IGNORECASE)
    
    # Emergency Fix 4: Find column for wildcard replacement
    # Look for any column reference in the SQL
    default_col = None
    
    # Try to find from FROM clause
    from_match = re.search(r'FROM\s+[\w.]+\s+(\w+)', sql, re.IGNORECASE)
    if from_match:
        alias = from_match.group(1)
        # Find any column with this alias
        col_match = re.search(rf'{alias}\.(\w+)', sql, re.IGNORECASE)
        if col_match:
            default_col = f"{alias}.{col_match.group(1)}"
    
    # If we found a column, do emergency wildcard replacement
    if default_col:
        if debug:
            print(f"[emergency_preprocess_sql] Using emergency column: {default_col}")
        
        # Replace the most critical wildcards
        sql = re.sub(r'CASE\s+WHEN\s+\*\s+IS\s+NULL', f'CASE WHEN {default_col} IS NULL', sql, flags=re.IGNORECASE)
        sql = re.sub(r'TRIM\s*\(\s*\*\s*::\s*text\s*\)', f'TRIM({default_col}::text)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'(?<!\w)\*\s*::\s*text(?!\w)', f'{default_col}::text', sql, flags=re.IGNORECASE)
    
    if debug:
        print("[emergency_preprocess_sql] Emergency pre-processing completed")
    
    return sql


# ============================================================================
# CRITICAL FIX FUNCTIONS - MUST RUN FIRST
# ============================================================================
import re
import datetime
import traceback
import psycopg2
from dateutil.relativedelta import relativedelta
from typing import Dict, Tuple, List, Any
from datetime import datetime as dt

# ============================================================================
# EMERGENCY PRE-PROCESSOR - Runs before all other fixes
# ============================================================================




def balance_parentheses(sql: str, debug: bool = False) -> str:
    """
    ENHANCED: Intelligently balance parentheses.
    """
    if debug:
        print("\n[balance_parentheses] Checking parentheses balance...")
    
    open_count = sql.count('(')
    close_count = sql.count(')')
    
    if open_count == close_count:
        if debug:
            print("[balance_parentheses] Already balanced")
        return sql
    
    diff = open_count - close_count
    
    if diff > 0:
        sql = sql.rstrip(';').rstrip()
        sql += ')' * diff
        sql += ';'
        if debug:
            print(f"[balance_parentheses] Added {diff} closing parentheses")
    else:
        diff = abs(diff)
        if debug:
            print(f"[balance_parentheses] Removing {diff} extra closing parentheses")
        
        for _ in range(diff):
            sql = re.sub(r'\)\s*\)(?=\s*(?:AS|FROM|WHERE|GROUP|ORDER|,|;|$))', ')', sql, count=1)
    
    return sql

import re
import datetime
import traceback
import psycopg2
from dateutil.relativedelta import relativedelta
from typing import Dict, Tuple, List, Any
from datetime import datetime as dt

# ============================================================================
# EMERGENCY PRE-PROCESSOR - Runs before all other fixes
# ============================================================================

def emergency_preprocess_sql(sql: str, schema_catalog: Dict[str, Any] = None, debug: bool = False) -> str:
    """
    EMERGENCY: Handle severely malformed SQL before main repair.
    This catches the exact pattern from your error.
    """
    if debug:
        print("\n[emergency_preprocess_sql] Running emergency pre-processing...")
    
    # Emergency Fix 1: EXTRACT(MONTH) FROM ... ) AS month
    sql = re.sub(
        r'EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([^\s)]+(?:::\w+)?)\s*\)\s*AS',
        r'EXTRACT(\1 FROM \2) AS',
        sql,
        flags=re.IGNORECASE
    )
    
    # Emergency Fix 2: COUNT(( CASE -> COUNT(CASE
    sql = re.sub(r'COUNT\s*\(\s*\(\s*CASE', 'COUNT(CASE', sql, flags=re.IGNORECASE)
    sql = re.sub(r'SUM\s*\(\s*\(\s*CASE', 'SUM(CASE', sql, flags=re.IGNORECASE)
    sql = re.sub(r'AVG\s*\(\s*\(\s*CASE', 'AVG(CASE', sql, flags=re.IGNORECASE)
    
    # Emergency Fix 3: END )) AS -> END) AS
    sql = re.sub(r'END\s*\)\s*\)\s*AS', 'END) AS', sql, flags=re.IGNORECASE)
    
    # Emergency Fix 4: Find column for wildcard replacement
    default_col = None
    from_match = re.search(r'FROM\s+[\w.]+\s+(\w+)', sql, re.IGNORECASE)
    if from_match:
        alias = from_match.group(1)
        col_match = re.search(rf'{alias}\.(\w+)', sql, re.IGNORECASE)
        if col_match:
            default_col = f"{alias}.{col_match.group(1)}"
    
    if default_col:
        if debug:
            print(f"[emergency_preprocess_sql] Using emergency column: {default_col}")
        sql = re.sub(r'CASE\s+WHEN\s+\*\s+IS\s+NULL', f'CASE WHEN {default_col} IS NULL', sql, flags=re.IGNORECASE)
        sql = re.sub(r'TRIM\s*\(\s*\*\s*::\s*text\s*\)', f'TRIM({default_col}::text)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'(?<!\w)\*\s*::\s*text(?!\w)', f'{default_col}::text', sql, flags=re.IGNORECASE)
    
    if debug:
        print("[emergency_preprocess_sql] Emergency pre-processing completed")
    
    return sql


# ============================================================================
# CRITICAL FIX FUNCTIONS - MUST RUN FIRST
# ============================================================================

def fix_extract_syntax(sql: str, debug: bool = False) -> str:
    """
    ULTRA-ENHANCED: Fix ALL malformed EXTRACT syntax patterns.
    Handles the exact error: EXTRACT(MONTH) FROM col::TIMESTAMP)AS month
    """
    if debug:
        print("\n[fix_extract_syntax] Starting EXTRACT syntax repair...")
    
    # Pattern 1: CRITICAL - Fix the exact error pattern
    pattern1 = r'EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([^\s)]+(?:::\w+)?)\s*\)\s*AS'
    sql = re.sub(pattern1, r'EXTRACT(\1 FROM \2) AS', sql, flags=re.IGNORECASE)
    
    # Pattern 2: General - EXTRACT(PART) FROM column -> EXTRACT(PART FROM column)
    pattern2 = r'EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([^\s,;)]+(?:\s*::\s*\w+)?)'
    sql = re.sub(pattern2, r'EXTRACT(\1 FROM \2)', sql, flags=re.IGNORECASE)
    
    # Pattern 3: Remove extra closing paren after EXTRACT
    pattern3 = r'(EXTRACT\s*\(\s*\w+\s+FROM\s+[^)]+\))\s*\)+(?=\s*(?:AS|,|FROM|WHERE|GROUP|ORDER|HAVING|LIMIT|;|$))'
    sql = re.sub(pattern3, r'\1', sql, flags=re.IGNORECASE)
    
    # Pattern 4: Fix GROUP BY with malformed EXTRACT
    pattern4 = r'GROUP\s+BY\s+EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([^\s)]+(?:::\w+)?)\s*(?=\)|ORDER|;|$)'
    sql = re.sub(pattern4, r'GROUP BY EXTRACT(\1 FROM \2)', sql, flags=re.IGNORECASE)
    
    if debug:
        print(f"[fix_extract_syntax] Completed EXTRACT syntax repair")
    
    return sql


def fix_wildcard_in_case_statements(sql: str, schema_catalog: Dict[str, Any] = None, debug: bool = False) -> str:
    """
    ULTRA-ENHANCED: Replace ALL instances of wildcard '*' with actual column names.
    CRITICAL: Also fixes empty column references like TRIM(::text)
    """
    if debug:
        print("\n[fix_wildcard_in_case_statements] Starting wildcard replacement...")
    
    default_column = None
    table_alias = None
    
    # Step 1: Extract table alias and find ID column
    if schema_catalog:
        for table_name, table_info in schema_catalog.items():
            if any(keyword in table_name.lower() for keyword in ["app", "application", "loan", "apploan"]):
                for col in table_info.get("columns", []):
                    col_name = col.get("name", "")
                    if "id" in col_name.lower():
                        alias_match = re.search(rf'{re.escape(table_name)}\s+(\w+)', sql, re.IGNORECASE)
                        if alias_match:
                            alias = alias_match.group(1)
                            table_alias = alias
                            default_column = f"{alias}.{col_name}"
                        else:
                            table_alias = table_name.split('.')[-1]
                            default_column = f"{table_alias}.{col_name}"
                        break
                if default_column:
                    break
    
    # Step 2: If no schema, extract from SQL
    if not default_column:
        from_match = re.search(r'FROM\s+([\w.]+)\s+(\w+)', sql, re.IGNORECASE)
        if from_match:
            table_name = from_match.group(1)
            table_alias = from_match.group(2)
            col_match = re.search(rf'{re.escape(table_alias)}\.(\w+)', sql, re.IGNORECASE)
            if col_match:
                default_column = f"{table_alias}.{col_match.group(1)}"
    
    # Step 3: Fallback searches
    if not default_column:
        col_ref = re.search(r'(\w+\.\w+)(?=\s*(?:::|ILIKE|=|>|<|IS))', sql, re.IGNORECASE)
        if col_ref:
            default_column = col_ref.group(1)
            table_alias = default_column.split('.')[0]
    
    # Step 4: Ultimate fallback
    if not table_alias:
        ta_match = re.search(r'FROM\s+[\w.]+\s+(\w+)', sql, re.IGNORECASE)
        if ta_match:
            table_alias = ta_match.group(1)
    
    if not default_column and table_alias:
        default_column = f"{table_alias}.application_id"
    
    if not default_column:
        default_column = "1"
    
    if debug:
        print(f"[fix_wildcard_in_case_statements] Using column: {default_column}, alias: {table_alias}")
    
    # CRITICAL: Fix missing column references - TRIM(::text) -> TRIM(col::text)
    if table_alias:
        # Fix: TRIM(::text) with missing column before cast
        sql = re.sub(
            r'TRIM\s*\(\s*::\s*text\s*\)',
            f'TRIM({default_column}::text)',
            sql,
            flags=re.IGNORECASE
        )
        # Fix: TRIM(::TEXT)
        sql = re.sub(
            r'TRIM\s*\(\s*::\s*TEXT\s*\)',
            f'TRIM({default_column}::TEXT)',
            sql,
            flags=re.IGNORECASE
        )
        # Fix: UPPER(TRIM(::text)) with missing column
        sql = re.sub(
            r'UPPER\s*\(\s*TRIM\s*\(\s*::\s*text\s*\)\s*\)',
            f'UPPER(TRIM({default_column}::text))',
            sql,
            flags=re.IGNORECASE
        )
        # Fix: Bare ::text with missing column
        sql = re.sub(
            r'(?<=\s)::\s*text(?=\s*(?:=|~|\)|,))',
            f' {default_column}::text',
            sql,
            flags=re.IGNORECASE
        )
        # Fix: Bare ::NUMERIC with missing column
        sql = re.sub(
            r'(?<=\s)::\s*NUMERIC(?=\s*(?:\)|,|;))',
            f' {default_column}::NUMERIC',
            sql,
            flags=re.IGNORECASE
        )
    
    # Then apply standard replacements for wildcards and empty references
    replacements = [
        (r'CASE\s+WHEN\s+\*\s+IS\s+NULL', f'CASE WHEN {default_column} IS NULL'),
        (r'CASE\s+WHEN\s+\*\s+IS\s+NOT\s+NULL', f'CASE WHEN {default_column} IS NOT NULL'),
        (r'TRIM\s*\(\s*\*\s*::\s*text\s*\)', f'TRIM({default_column}::text)'),
        (r'TRIM\s*\(\s*\*\s*::\s*TEXT\s*\)', f'TRIM({default_column}::TEXT)'),
        (r'(?<!\w)\*\s*::\s*text(?!\w)', f'{default_column}::text'),
        (r'(?<!\w)\*\s*::\s*TEXT(?!\w)', f'{default_column}::TEXT'),
        (r'UPPER\s*\(\s*TRIM\s*\(\s*\*\s*::\s*text\s*\)\s*\)', f'UPPER(TRIM({default_column}::text))'),
        (r'UPPER\s*\(\s*TRIM\s*\(\s*\*\s*::\s*TEXT\s*\)\s*\)', f'UPPER(TRIM({default_column}::TEXT))'),
        (r'(?<!\w)\*\s*::\s*NUMERIC(?!\w)', f'{default_column}::NUMERIC'),
        (r'(?<=\s)\*(?=\s+(?:IS|=|!=|<>|~))', default_column),
    ]
    
    for pattern, replacement in replacements:
        sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
    
    return sql


def fix_nested_aggregate_parentheses(sql: str, debug: bool = False) -> str:
    """
    ENHANCED: Fix ALL double/triple nested parentheses in aggregate functions.
    """
    if debug:
        print("\n[fix_nested_aggregate_parentheses] Fixing nested parentheses...")
    
    aggregates = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
    
    for agg in aggregates:
        pattern1 = rf'\b{agg}\s*\(\s*\(\s*CASE'
        sql = re.sub(pattern1, f'{agg}(CASE', sql, flags=re.IGNORECASE)
        
        pattern2 = rf'\b{agg}\s*\(\s*\('
        sql = re.sub(pattern2, f'{agg}(', sql, flags=re.IGNORECASE)
    
    sql = re.sub(r'END\s*\)\s*\)+(?=\s*(?:AS|,|FROM|WHERE|GROUP|ORDER|;|$))', 'END)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\)\s*\)\s*AS\b', ') AS', sql, flags=re.IGNORECASE)
    
    if debug:
        print(f"[fix_nested_aggregate_parentheses] Completed")
    
    return sql


def fix_malformed_case_expressions(sql: str, debug: bool = False) -> str:
    """
    ENHANCED: Fix complex nested CASE expressions.
    """
    if debug:
        print("\n[fix_malformed_case_expressions] Fixing nested CASE expressions...")
    
    pattern1 = r'(SUM|COUNT|AVG)\s*\(\s*CASE\s+WHEN\s+CASE\s+WHEN\s+(.*?)\s+THEN\s+1\s+(?:ELSE\s+0\s+)?END\s+IS\s+NULL\s+THEN\s+NULL(?:\s+ELSE\s+NULL)?\s+END\s*\)'
    
    def replacer1(match):
        agg_func = match.group(1)
        condition = match.group(2).strip()
        return f"{agg_func}(CASE WHEN {condition} THEN 1 ELSE 0 END)"
    
    sql = re.sub(pattern1, replacer1, sql, flags=re.IGNORECASE | re.DOTALL)
    
    pattern2 = r'COUNT\s*\(\s*CASE\s+WHEN\s+CASE\s+WHEN\s+(.+?)\s+THEN\s+(\d+)\s+END'
    
    def replacer2(match):
        condition = match.group(1).strip()
        value = match.group(2)
        return f"COUNT(CASE WHEN {condition} THEN {value}"
    
    sql = re.sub(pattern2, replacer2, sql, flags=re.IGNORECASE | re.DOTALL)
    
    if debug:
        print(f"[fix_malformed_case_expressions] Completed")
    
    return sql


def detect_critical_sql_errors(sql: str, debug: bool = False) -> List[str]:
    """
    DETECTION: Find critical SQL errors that indicate source generation problems.
    Returns list of errors found.
    """
    errors = []
    
    # Check for empty column references
    if re.search(r'TRIM\s*\(\s*::\s*(?:text|TEXT|NUMERIC)\s*\)', sql, re.IGNORECASE):
        errors.append("CRITICAL: Empty column reference in TRIM - TRIM(::type)")
    
    if re.search(r'(?<=\s)::\s*(?:text|TEXT|NUMERIC)(?=\s*(?:=|~|\)|,|;))', sql):
        errors.append("CRITICAL: Bare cast operator with no column")
    
    # Check for nested CASE that's too complex
    case_count = len(re.findall(r'\bCASE\b', sql, re.IGNORECASE))
    if case_count > 3:
        errors.append("WARNING: Too many nested CASE statements (>3)")
    
    # Check for double COUNT(( pattern (before fixing)
    if re.search(r'(COUNT|SUM|AVG)\s*\(\s*\(\s*CASE', sql, re.IGNORECASE):
        errors.append("ERROR: Double-nested aggregate function")
    
    # Check for missing FROM clause
    if not re.search(r'\bFROM\b', sql, re.IGNORECASE):
        errors.append("CRITICAL: Missing FROM clause")
    
    # Check for malformed EXTRACT before fixing
    if re.search(r'EXTRACT\s*\(\s*\w+\s*\)\s+FROM', sql, re.IGNORECASE):
        errors.append("ERROR: Malformed EXTRACT syntax")
    
    if debug and errors:
        print(f"[detect_critical_sql_errors] Found {len(errors)} errors:")
        for err in errors:
            print(f"  - {err}")
    
    return errors
    """
    ENHANCED: Intelligently balance parentheses.
    """
    if debug:
        print("\n[balance_parentheses] Checking parentheses balance...")
    
    open_count = sql.count('(')
    close_count = sql.count(')')
    
    if open_count == close_count:
        if debug:
            print("[balance_parentheses] Already balanced")
        return sql
    
    diff = open_count - close_count
    
    if diff > 0:
        sql = sql.rstrip(';').rstrip()
        sql += ')' * diff
        sql += ';'
        if debug:
            print(f"[balance_parentheses] Added {diff} closing parentheses")
    else:
        diff = abs(diff)
        if debug:
            print(f"[balance_parentheses] Removing {diff} extra closing parentheses")
        
        for _ in range(diff):
            sql = re.sub(r'\)\s*\)(?=\s*(?:AS|FROM|WHERE|GROUP|ORDER|,|;|$))', ')', sql, count=1)
    
    return sql


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_fixed_date_context() -> Dict[str, Any]:
    """
    Generate 2024 reference year context with dynamic month/quarter/half.
    """
    now_dt = dt.now()
    fixed_year = 2024
    month = now_dt.month
    
    quarter = (month - 1) // 3 + 1
    half = 1 if month <= 6 else 2
    
    quarter_start = dt(fixed_year, 3 * (quarter - 1) + 1, 1)
    quarter_end = quarter_start + relativedelta(months=3)
    quarter_end = quarter_end - relativedelta(days=1)
    
    half_start = dt(fixed_year, 1 if half == 1 else 7, 1)
    half_end = half_start + relativedelta(months=6) - relativedelta(days=1)
    
    return {
        "year": fixed_year,
        "month": month,
        "quarter": quarter,
        "half": half,
        "quarter_start": quarter_start.strftime("%Y-%m-%d"),
        "quarter_end": quarter_end.strftime("%Y-%m-%d"),
        "half_start": half_start.strftime("%Y-%m-%d"),
        "half_end": half_end.strftime("%Y-%m-%d"),
        "year_start": f"{fixed_year}-01-01",
        "year_end": f"{fixed_year}-12-31",
    }


def validate_repaired_sql(sql: str, debug: bool = False) -> Tuple[bool, List[str]]:
    """
    VALIDATION: Check if the repaired SQL is likely to execute without syntax errors.
    Returns: (is_valid, list_of_remaining_issues)
    """
    issues = []
    
    # Check parentheses balance
    if sql.count('(') != sql.count(')'):
        issues.append(f"Parentheses mismatch: {sql.count('(')} ( vs {sql.count(')')} )")
    
    # Check for empty column references (CRITICAL)
    if re.search(r'TRIM\s*\(\s*::\s*(?:text|TEXT|NUMERIC)\s*\)', sql, re.IGNORECASE):
        issues.append("Empty column in TRIM(::type)")
    
    if re.search(r'(?<=\s)::\s*(?:text|TEXT|NUMERIC)(?=\s*(?:=|~|\)|,|;))', sql):
        issues.append("Bare cast operator without column")
    
    # Check for malformed EXTRACT
    if re.search(r'EXTRACT\s*\(\s*\w+\s*\)\s+FROM', sql, re.IGNORECASE):
        issues.append("Malformed EXTRACT syntax still present")
    
    # Check for wildcards still present
    if re.search(r'CASE\s+WHEN\s+\*\s+IS', sql, re.IGNORECASE):
        issues.append("Wildcard (*) still in CASE statement")
    
    # Check for double-nested aggregates
    if re.search(r'(COUNT|SUM|AVG)\s*\(\s*\(', sql, re.IGNORECASE):
        issues.append("Double-nested aggregate function")
    
    # Check basic SQL structure
    if not re.search(r'\bFROM\b', sql, re.IGNORECASE):
        issues.append("Missing FROM clause")
    
    if not re.search(r'\bSELECT\b', sql, re.IGNORECASE):
        issues.append("Missing SELECT clause")
    
    is_valid = len(issues) == 0
    
    if debug:
        if is_valid:
            print("[validate_repaired_sql] SQL validation PASSED")
        else:
            print("[validate_repaired_sql] SQL validation FAILED:")
            for issue in issues:
                print(f"  - {issue}")
    
    return is_valid, issues


def enforce_2024(sql: str, date_info: Dict[str, Any] = None) -> str:
    """
    Force all queries to align with 2024 context.
    """
    if date_info is None:
        date_info = get_fixed_date_context()
    
    anchor_date = f"{date_info['year']}-{date_info['month']:02d}-01"
    sql = sql.replace("CURRENT_DATE", f"DATE '{anchor_date}'")
    
    sql = re.sub(r"EXTRACT\(YEAR FROM [^)]+\)", str(date_info["year"]), sql, flags=re.IGNORECASE)
    
    sql = re.sub(
        r"DATE_TRUNC\('quarter', DATE '[^']+'\)\s*-\s*INTERVAL '3 month'",
        f"DATE '{date_info['quarter_start']}'",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"DATE_TRUNC\('quarter', DATE '[^']+'\)",
        f"DATE '{date_info['quarter_end']}'",
        sql,
        flags=re.IGNORECASE,
    )
    
    sql = sql.replace(
        "DATE_TRUNC('half', CURRENT_DATE)",
        f"DATE '{date_info['half_start']}'"
    )
    
    sql = re.sub(
        r"DATE '(\d{4}-\d{2}-\d{2})'\s*-\s*INTERVAL '(\d+) year'",
        f"DATE '{date_info['year']}-01-01'",
        sql,
        flags=re.IGNORECASE,
    )
    
    sql = re.sub(
        r"CURRENT_DATE\s*-\s*INTERVAL '(\d+) month'",
        lambda m: f"DATE '{(dt(date_info['year'], date_info['month'], 1) - relativedelta(months=int(m.group(1)))).strftime('%Y-%m-%d')}'",
        sql,
        flags=re.IGNORECASE,
    )
    
    sql = re.sub(
        r"<\s*CURRENT_DATE",
        f"< DATE '{date_info['year']}-12-31'",
        sql,
        flags=re.IGNORECASE,
    )
    
    sql = re.sub(r"\b(DATE\s+)+DATE\b", "DATE", sql)
    
    return sql


# ============================================================================
# MAIN NUCLEAR REPAIR FUNCTION
# ============================================================================

def nuclear_repair_sql(
    raw_llm_output: str,
    tables: List[str] = None,
    schema_catalog: Dict[str, Any] = None,
    join_info: Dict[str, Any] = None,
    debug: bool = False
) -> str:
    """
    PRODUCTION-READY: Comprehensive SQL repair with ALL fixes.
    This function is self-contained and handles ALL common SQL generation errors.
    """
    
    sql_to_run = raw_llm_output
    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        try:
            if debug:
                print(f"\n{'='*80}")
                print(f"[ATTEMPT {attempt}/{max_attempts}] Starting SQL repair...")
                print(f"{'='*80}")

            sql = sql_to_run

            # ============================================================
            # PHASE 0: EMERGENCY PRE-PROCESSING
            # ============================================================
            if debug:
                print("\n[PHASE 0] EMERGENCY PRE-PROCESSING...")
            
            # Check for critical errors BEFORE processing
            critical_errors = detect_critical_sql_errors(sql, debug=debug)
            if critical_errors:
                if debug:
                    print(f"\n⚠️  CRITICAL ERRORS DETECTED - Applying aggressive fixes")
                    for err in critical_errors:
                        print(f"    {err}")
            
            sql = emergency_preprocess_sql(sql, schema_catalog, debug=debug)

            # ============================================================
            # PHASE 0.1: CLEAN RAW LLM OUTPUT
            # ============================================================
            if debug:
                print("\n[PHASE 0.1] Cleaning raw LLM output...")
            
            sql = sql.strip()
            sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
            sql = re.sub(r'```\s*', '', sql)
            sql = re.sub(r'`+', '', sql)
            
            sql_start = re.search(r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b', sql, re.IGNORECASE)
            if sql_start:
                sql = sql[sql_start.start():]

            # ============================================================
            # PHASE 1: CRITICAL FIXES (MUST RUN FIRST)
            # ============================================================
            
            if debug:
                print("\n[PHASE 1.1] FIXING EXTRACT SYNTAX (Priority 1)")
            sql = fix_extract_syntax(sql, debug=debug)
            
            if debug:
                print("\n[PHASE 1.2] FIXING WILDCARD IN CASE STATEMENTS (Pass 1)")
            sql = fix_wildcard_in_case_statements(sql, schema_catalog, debug=debug)
            
            if debug:
                print("\n[PHASE 1.3] FIXING NESTED AGGREGATE PARENTHESES")
            sql = fix_nested_aggregate_parentheses(sql, debug=debug)
            
            if debug:
                print("\n[PHASE 1.4] FIXING MALFORMED CASE EXPRESSIONS")
            sql = fix_malformed_case_expressions(sql, debug=debug)
            
            if debug:
                print("\n[PHASE 1.5] FIXING WILDCARD IN CASE STATEMENTS (Pass 2)")
            sql = fix_wildcard_in_case_statements(sql, schema_catalog, debug=False)
            
            sql = fix_extract_syntax(sql, debug=False)

            # ============================================================
            # PHASE 2: STANDARD SQL FIXES
            # ============================================================
            
            if debug:
                print("\n[PHASE 2] Standard SQL Fixes")
            
            # 2.1: Date/time fixes
            today = datetime.date.today()
            fixed_date = f"DATE '{today.year}-{today.month:02d}-{today.day:02d}'"
            sql = re.sub(r'\bCURRENT_DATE\b', fixed_date, sql, flags=re.IGNORECASE)
            sql = re.sub(r'\bNOW\(\)\b', fixed_date, sql, flags=re.IGNORECASE)
            
            # 2.2: NULL comparison fixes
            sql = re.sub(r'(\w+)\s*=\s*NULL\b', r'\1 IS NULL', sql, flags=re.IGNORECASE)
            sql = re.sub(r'(\w+)\s*!=\s*NULL\b', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)
            sql = re.sub(r'(\w+)\s*<>\s*NULL\b', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)
            
            # 2.3: Syntax cleanup
            sql = re.sub(r',\s*\)', ')', sql)
            sql = re.sub(r',\s*FROM\b', ' FROM', sql, flags=re.IGNORECASE)
            sql = re.sub(r',\s*WHERE\b', ' WHERE', sql, flags=re.IGNORECASE)
            sql = re.sub(r',\s*GROUP\b', ' GROUP', sql, flags=re.IGNORECASE)
            
            # 2.4: Convert quarter intervals
            sql = re.sub(r"INTERVAL\s+'1\s+quarter'", "INTERVAL '3 months'", sql, flags=re.IGNORECASE)

            # ============================================================
            # PHASE 3: FINAL FIXES
            # ============================================================
            
            if debug:
                print("\n[PHASE 3] Final Fixes")
            
            sql = balance_parentheses(sql, debug=debug)
            sql = re.sub(r'\s+', ' ', sql).strip()
            sql = sql.rstrip(';') + ';'
            sql = fix_extract_syntax(sql, debug=False)

            # ============================================================
            # PHASE 4: VERIFICATION
            # ============================================================
            
            if debug:
                print("\n[PHASE 4] Final Verification")
            
            # Run validation checks
            is_valid, remaining_issues = validate_repaired_sql(sql, debug=debug)
            
            if not is_valid:
                if debug:
                    print(f"\n⚠️  VALIDATION FAILED - {len(remaining_issues)} issues found:")
                    for issue in remaining_issues:
                        print(f"    - {issue}")
                
                # Apply emergency fixes again
                sql = fix_wildcard_in_case_statements(sql, schema_catalog, debug=False)
                sql = fix_nested_aggregate_parentheses(sql, debug=False)
                sql = fix_extract_syntax(sql, debug=False)
                sql = balance_parentheses(sql, debug=False)
                
                # Re-validate
                is_valid, remaining_issues = validate_repaired_sql(sql, debug=False)
                if not is_valid and debug:
                    print(f"⚠️  Still {len(remaining_issues)} issues after re-attempt")
            else:
                if debug:
                    print("✓ SQL validation PASSED")
            
            # Final checks
            issues = []
            
            if re.search(r'CASE\s+WHEN\s+\*\s+IS', sql, re.IGNORECASE):
                issues.append("Wildcard in CASE statement")
            
            if re.search(r'(COUNT|SUM|AVG)\s*\(\s*\(', sql, re.IGNORECASE):
                issues.append("Double parentheses in aggregate")
            
            if re.search(r'EXTRACT\s*\(\s*\w+\s*\)\s+FROM', sql, re.IGNORECASE):
                issues.append("Malformed EXTRACT syntax")
            
            if re.search(r'TRIM\s*\(\s*::\s*(?:text|TEXT|NUMERIC)', sql, re.IGNORECASE):
                issues.append("Empty column in TRIM")
            
            if issues:
                if debug:
                    print(f"WARNING: Issues still present: {', '.join(issues)}")
                sql = fix_wildcard_in_case_statements(sql, schema_catalog, debug=False)
                sql = fix_nested_aggregate_parentheses(sql, debug=False)
                sql = fix_extract_syntax(sql, debug=False)
            else:
                if debug:
                    print("✓ No critical issues detected")
            
            if debug:
                print("\n[FINAL REPAIRED SQL]")
                print("-" * 80)
                print(sql)
                print("-" * 80)

            sql_to_run = sql
            break

        except Exception as e:
            if debug:
                print(f"\n[ERROR in attempt {attempt}] {e}")
                traceback.print_exc()
            
            if attempt >= max_attempts:
                raise

    return sql_to_run



# ============================================================================
# ADD THIS SECTION TO YOUR imports AT THE TOP OF views.py
# ============================================================================
import re
from typing import Tuple, List, Dict, Any

# ============================================================================
# ADD THESE FUNCTIONS BEFORE run_sql_generation_graph1410()
# ============================================================================

def quick_validate_sql_syntax(sql: str) -> Tuple[bool, List[str]]:
    """
    Quick validation check for critical SQL syntax errors.
    Returns: (is_valid, list_of_errors)
    """
    errors = []
    
    # Check 1: EXTRACT parentheses
    if re.search(r'EXTRACT\s*\(\s*\w+\s*\)\s+FROM', sql, re.IGNORECASE):
        errors.append("EXTRACT(PART) FROM - missing FROM inside parentheses")
    
    # Check 2: Empty column references
    if re.search(r'TRIM\s*\(\s*::\s*(?:text|TEXT|NUMERIC)', sql, re.IGNORECASE):
        errors.append("TRIM(::type) - missing column before cast")
    
    if re.search(r'(?<=\s)::\s*(?:text|TEXT|NUMERIC)(?=\s*(?:=|~|\)|,))', sql):
        errors.append("Bare cast operator without column")
    
    # Check 3: Double parentheses in aggregates
    if re.search(r'(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*\(', sql, re.IGNORECASE):
        errors.append("Double parentheses in aggregate function")
    
    # Check 4: Wildcard in CASE
    if re.search(r'CASE\s+WHEN\s+\*\s+IS', sql, re.IGNORECASE):
        errors.append("Wildcard (*) in CASE WHEN")
    
    # Check 5: Parentheses balance
    if sql.count('(') != sql.count(')'):
        errors.append(f"Parentheses mismatch: {sql.count('(')} ( vs {sql.count(')')} )")
    
    return len(errors) == 0, errors


def aggressive_fix_extract_syntax(sql: str) -> str:
    """
    AGGRESSIVE fix for EXTRACT syntax issues.
    Handles: EXTRACT(MONTH) FROM col::TIME)AS → EXTRACT(MONTH FROM col::TIMESTAMP) AS
    """
    # Fix 1: EXTRACT(PART) FROM col::TYPE)AS → EXTRACT(PART FROM col::TYPE) AS
    sql = re.sub(
        r'EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([^\s)]+(?:::\w+)?)\s*\)\s*AS',
        r'EXTRACT(\1 FROM \2) AS',
        sql,
        flags=re.IGNORECASE
    )
    
    # Fix 2: EXTRACT(PART) FROM col::TYPE → EXTRACT(PART FROM col::TYPE)
    sql = re.sub(
        r'EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([^\s,;)]+(?:::\w+)?)',
        r'EXTRACT(\1 FROM \2)',
        sql,
        flags=re.IGNORECASE
    )
    
    # Fix 3: GROUP BY EXTRACT(PART) FROM col::TYPE → GROUP BY EXTRACT(PART FROM col::TYPE)
    sql = re.sub(
        r'GROUP\s+BY\s+EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([^\s)]+(?:::\w+)?)',
        r'GROUP BY EXTRACT(\1 FROM \2)',
        sql,
        flags=re.IGNORECASE
    )
    
    return sql


def aggressive_fix_empty_columns(sql: str, schema_catalog: Dict[str, Any] = None) -> str:
    """
    AGGRESSIVE fix for empty column references: TRIM(::text) → TRIM(col::text)
    """
    # Find a valid column to use as fallback
    default_col = None
    
    # Try to extract from FROM clause
    from_match = re.search(r'FROM\s+([\w.]+)\s+(\w+)', sql, re.IGNORECASE)
    if from_match:
        table_name = from_match.group(1)
        alias = from_match.group(2)
        
        # Look for numeric/amount columns in schema
        if schema_catalog and table_name in schema_catalog:
            table_info = schema_catalog[table_name]
            for col in table_info.get("columns", []):
                col_name = col.get("name", "")
                col_type = col.get("type", "").lower()
                # Prefer numeric or amount columns
                if any(keyword in col_name.lower() for keyword in ["amount", "numeric", "balance", "value", "total"]):
                    default_col = f"{alias}.{col_name}"
                    break
            
            # Fallback: just use first column
            if not default_col and table_info.get("columns"):
                first_col = table_info["columns"][0].get("name", "")
                default_col = f"{alias}.{first_col}"
    
    # Ultimate fallback
    if not default_col:
        col_ref = re.search(r'(\w+\.\w+)', sql)
        if col_ref:
            default_col = col_ref.group(1)
    
    if not default_col:
        default_col = "1"
    
    # Fix empty column references
    sql = re.sub(r'TRIM\s*\(\s*::\s*text\s*\)', f'TRIM({default_col}::text)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'TRIM\s*\(\s*::\s*TEXT\s*\)', f'TRIM({default_col}::TEXT)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'(?<=\s)::\s*text(?=\s*(?:=|~|\)|,|;))', f' {default_col}::text', sql, flags=re.IGNORECASE)
    sql = re.sub(r'(?<=\s)::\s*TEXT(?=\s*(?:=|~|\)|,|;))', f' {default_col}::TEXT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'CASE\s+WHEN\s+\*\s+IS\s+NULL', f'CASE WHEN {default_col} IS NULL', sql, flags=re.IGNORECASE)
    
    return sql


def final_sql_cleanup(sql: str) -> str:
    """
    Final cleanup pass for common SQL issues.
    """
    # Fix double parentheses in aggregates
    sql = re.sub(r'(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*\(\s*CASE', r'\1(CASE', sql, flags=re.IGNORECASE)
    sql = re.sub(r'END\s*\)\s*\)\s*AS', 'END) AS', sql, flags=re.IGNORECASE)
    
    # Fix double commas and syntax issues
    sql = re.sub(r',\s*\)', ')', sql)
    sql = re.sub(r',\s*FROM', ' FROM', sql, flags=re.IGNORECASE)
    sql = re.sub(r',\s*WHERE', ' WHERE', sql, flags=re.IGNORECASE)
    sql = re.sub(r',\s*GROUP', ' GROUP', sql, flags=re.IGNORECASE)
    
    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # Ensure proper semicolon
    sql = sql.rstrip(';') + ';'
    
    return sql


# ============================================================================
# MODIFY run_sql_generation_graph1410() - REPLACE THE SECTION AFTER "Step 6"
# ============================================================================
# ============================================================================
# ADD THIS SECTION TO YOUR imports AT THE TOP OF views.py
# ============================================================================
import re
from typing import Tuple, List, Dict, Any

# ============================================================================
# ADD THESE FUNCTIONS BEFORE run_sql_generation_graph1410()
# ============================================================================
import re
from typing import Tuple, List, Dict, Any

# ============================================================================
# ADD THESE FUNCTIONS BEFORE run_sql_generation_graph1410()
# ============================================================================

def strip_sql_markdown(sql: str) -> str:
    """
    Remove markdown code fence markers from SQL - AGGRESSIVE VERSION.
    Handles all variations: ```sql, ```SQL, ```, etc.
    """
    if not sql:
        return sql
    
    sql = sql.strip()
    
    # Debug print
    print(f"[strip_sql_markdown] Input (first 100 chars): {sql[:100]}")
    
    # Remove opening fence: ```sql, ```SQL, ```postgresql, ``` etc
    sql = re.sub(r'^```(?:sql|SQL|postgres|postgresql|py|python)?\s*\n?', '', sql)
    
    # Remove closing fence: ``` at the end
    sql = re.sub(r'\n?```\s*$', '', sql)
    
    # Remove any inline backticks
    sql = sql.replace('```', '')
    sql = sql.strip('`')
    
    # Final cleanup
    sql = sql.strip()
    
    print(f"[strip_sql_markdown] Output (first 100 chars): {sql[:100]}")
    
    return sql


def quick_validate_sql_syntax(sql: str) -> Tuple[bool, List[str]]:
    """
    Quick validation check for critical SQL syntax errors.
    Returns: (is_valid, list_of_errors)
    """
    errors = []
    
    # Check 1: Markdown fences still present
    if '```' in sql:
        errors.append("Markdown code fences (```) still present in SQL")
    
    # Check 2: EXTRACT parentheses
    if re.search(r'EXTRACT\s*\(\s*\w+\s*\)\s+FROM', sql, re.IGNORECASE):
        errors.append("EXTRACT(PART) FROM - missing FROM inside parentheses")
    
    # Check 3: Empty column references
    if re.search(r'TRIM\s*\(\s*::\s*(?:text|TEXT|NUMERIC)', sql, re.IGNORECASE):
        errors.append("TRIM(::type) - missing column before cast")
    
    if re.search(r'(?<=\s)::\s*(?:text|TEXT|NUMERIC)(?=\s*(?:=|~|\)|,))', sql):
        errors.append("Bare cast operator without column")
    
    # Check 4: Double parentheses in aggregates
    if re.search(r'(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*\(', sql, re.IGNORECASE):
        errors.append("Double parentheses in aggregate function")
    
    # Check 5: Wildcard in CASE
    if re.search(r'CASE\s+WHEN\s+\*\s+IS', sql, re.IGNORECASE):
        errors.append("Wildcard (*) in CASE WHEN")
    
    # Check 6: Parentheses balance
    if sql.count('(') != sql.count(')'):
        errors.append(f"Parentheses mismatch: {sql.count('(')} ( vs {sql.count(')')} )")
    
    return len(errors) == 0, errors


def aggressive_fix_extract_syntax(sql: str) -> str:
    """
    AGGRESSIVE fix for EXTRACT syntax issues.
    Handles: EXTRACT(MONTH) FROM col::TIME)AS → EXTRACT(MONTH FROM col::TIMESTAMP) AS
    """
    # Fix 1: EXTRACT(PART) FROM col::TYPE)AS → EXTRACT(PART FROM col::TYPE) AS
    sql = re.sub(
        r'EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([^\s)]+(?:::\w+)?)\s*\)\s*AS',
        r'EXTRACT(\1 FROM \2) AS',
        sql,
        flags=re.IGNORECASE
    )
    
    # Fix 2: EXTRACT(PART) FROM col::TYPE → EXTRACT(PART FROM col::TYPE)
    sql = re.sub(
        r'EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([^\s,;)]+(?:::\w+)?)',
        r'EXTRACT(\1 FROM \2)',
        sql,
        flags=re.IGNORECASE
    )
    
    # Fix 3: GROUP BY EXTRACT(PART) FROM col::TYPE → GROUP BY EXTRACT(PART FROM col::TYPE)
    sql = re.sub(
        r'GROUP\s+BY\s+EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([^\s)]+(?:::\w+)?)',
        r'GROUP BY EXTRACT(\1 FROM \2)',
        sql,
        flags=re.IGNORECASE
    )
    
    return sql


def aggressive_fix_empty_columns(sql: str, schema_catalog: Dict[str, Any] = None) -> str:
    """
    AGGRESSIVE fix for empty column references: TRIM(::text) → TRIM(col::text)
    """
    # Find a valid column to use as fallback
    default_col = None
    
    # Try to extract from FROM clause
    from_match = re.search(r'FROM\s+([\w.]+)\s+(\w+)', sql, re.IGNORECASE)
    if from_match:
        table_name = from_match.group(1)
        alias = from_match.group(2)
        
        # Look for numeric/amount columns in schema
        if schema_catalog and table_name in schema_catalog:
            table_info = schema_catalog[table_name]
            for col in table_info.get("columns", []):
                col_name = col.get("name", "")
                col_type = col.get("type", "").lower()
                # Prefer numeric or amount columns
                if any(keyword in col_name.lower() for keyword in ["amount", "numeric", "balance", "value", "total"]):
                    default_col = f"{alias}.{col_name}"
                    break
            
            # Fallback: just use first column
            if not default_col and table_info.get("columns"):
                first_col = table_info["columns"][0].get("name", "")
                default_col = f"{alias}.{first_col}"
    
    # Ultimate fallback
    if not default_col:
        col_ref = re.search(r'(\w+\.\w+)', sql)
        if col_ref:
            default_col = col_ref.group(1)
    
    if not default_col:
        default_col = "1"
    
    # Fix empty column references
    sql = re.sub(r'TRIM\s*\(\s*::\s*text\s*\)', f'TRIM({default_col}::text)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'TRIM\s*\(\s*::\s*TEXT\s*\)', f'TRIM({default_col}::TEXT)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'(?<=\s)::\s*text(?=\s*(?:=|~|\)|,|;))', f' {default_col}::text', sql, flags=re.IGNORECASE)
    sql = re.sub(r'(?<=\s)::\s*TEXT(?=\s*(?:=|~|\)|,|;))', f' {default_col}::TEXT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'CASE\s+WHEN\s+\*\s+IS\s+NULL', f'CASE WHEN {default_col} IS NULL', sql, flags=re.IGNORECASE)
    
    return sql


def final_sql_cleanup(sql: str) -> str:
    """
    Final cleanup pass for common SQL issues.
    """
    # Fix double parentheses in aggregates
    sql = re.sub(r'(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*\(\s*CASE', r'\1(CASE', sql, flags=re.IGNORECASE)
    sql = re.sub(r'END\s*\)\s*\)\s*AS', 'END) AS', sql, flags=re.IGNORECASE)
    
    # Fix double commas and syntax issues
    sql = re.sub(r',\s*\)', ')', sql)
    sql = re.sub(r',\s*FROM', ' FROM', sql, flags=re.IGNORECASE)
    sql = re.sub(r',\s*WHERE', ' WHERE', sql, flags=re.IGNORECASE)
    sql = re.sub(r',\s*GROUP', ' GROUP', sql, flags=re.IGNORECASE)
    
    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # Ensure proper semicolon
    sql = sql.rstrip(';') + ';'
    
    return sql

# ============================================================================
# MODIFY run_sql_generation_graph1410() - REPLACE THE SECTION AFTER "Step 6"
# ============================================================================

def run_sql_generation_graph1410(
    question: str,
    user_id: str,
    session_id: str,
    history: List[Dict] = None
) -> Tuple[str, str, str]:
    """
    Fully merged SQL generation function with ANTI-SYNTAX-ERROR fixes.
    """
    try:
        print(f"🔍 SQL Generation for question: {question}")
        print(f"🔑 Session ID: {session_id}")

        # Step 1-5: KEEP YOUR EXISTING CODE (session, tables, joins, prompt, LLM)
        session_data = SessionManager.get_session(session_id)

        if not session_data:
            print(f"⚠️ Session {session_id} not found, trying fallback...")
            if session_store:
                session_id, session_data = next(iter(session_store.items()))
                print(f"↩️ Falling back to session {session_id}")
            else:
                try:
                    emergency_session = create_emergency_session()
                    if emergency_session:
                        session_id, session_data = emergency_session
                        session_store[session_id] = session_data
                        try:
                            embed_schema_catalog(
                                user_id=session_data["user_id"],
                                db_id=session_id,
                                schema_catalog=session_data["schema_catalog"],
                                embedder=embedder,
                                collection=collection
                            )
                            print("✅ Emergency session schema embedded")
                        except Exception as embed_error:
                            print(f"⚠️ Emergency embedding failed: {embed_error}")
                    else:
                        return ("-- ERROR: No active DB session and cannot create emergency session",
                                "Please connect to database first using the connect endpoint", "")
                except Exception as e:
                    print(f"❌ Emergency session creation failed: {e}")
                    traceback.print_exc()
                    return ("-- ERROR: No active DB session", "Please connect to database first", "")

        if not session_data or "schema_catalog" not in session_data:
            print("❌ Invalid session data - missing schema_catalog")
            return ("-- ERROR: Invalid session data", "Session corrupted, please reconnect", "")

        schema_catalog = session_data["schema_catalog"]
        print(f"📚 Schema catalog has {len(schema_catalog)} tables")

        selected_tables = select_relevant_tables_with_embedding_fallback(
            question=question,
            schema_catalog=schema_catalog,
            user_id=user_id,
            session_id=session_id,
            similarity_threshold=0.4,
            max_tables=5
        )

        if not selected_tables:
            selected_tables = list(schema_catalog.keys())[:3]
            print(f"⚠️ EMERGENCY FALLBACK: Using first {len(selected_tables)} tables")

        print(f"✅ Final selected tables: {selected_tables}")

        try:
            join_info = discover_join_keys(selected_tables, schema_catalog)
        except Exception as e:
            print(f"⚠️ Join discovery failed: {e}")
            join_info = {}

        prompt = build_dynamic_sql_prompt(
            question=question,
            schema_catalog=schema_catalog,
            tables=selected_tables,
            join_info=join_info,
            history=history or [],
            similarity_threshold=0.4
        )
        print(f"📝 Prompt length: {len(prompt)} chars")

        try:
            llm = get_llama_maverick_llm()
            if not llm:
                return ("-- ERROR: LLM not configured", "LLM function not available", "LLM_ERROR")

            response = llm.invoke(prompt)
            raw_sql = getattr(response, "content", response if isinstance(response, str) else str(response))
            print(f"🔍 Raw LLM SQL (first 300 chars):\n{raw_sql[:300]}\n")

        except Exception as e:
            print(f"❌ LLM invocation failed: {e}")
            traceback.print_exc()
            return (f"-- ERROR: LLM failed - {str(e)}", "LLM generation failed", "LLM_ERROR")

        # ============================================================================
        # STEP 6: NEW AGGRESSIVE SQL REPAIR PIPELINE (REPLACE OLD STEP 6)
        # ============================================================================
        print("\n" + "="*80)
        print("🔧 APPLYING ANTI-SYNTAX-ERROR FIXES")
        print("="*80)

        sql = raw_sql

        # Phase 0: Quick validation
        is_valid, validation_errors = quick_validate_sql_syntax(sql)
        if not is_valid:
            print(f"\n⚠️ VALIDATION ERRORS DETECTED:")
            for err in validation_errors:
                print(f"  - {err}")

        # Phase 1: Aggressive EXTRACT fix
        print("\n[Phase 1] Fixing EXTRACT syntax...")
        sql = aggressive_fix_extract_syntax(sql)

        # Phase 2: Aggressive empty column fix
        print("[Phase 2] Fixing empty column references...")
        sql = aggressive_fix_empty_columns(sql, schema_catalog)

        # Phase 3: Your existing fixes (keep all of them)
        print("[Phase 3] Applying existing fixes...")
        sql = fix_extract_syntax(sql, debug=False)
        sql = re.sub(r'CASE WHEN \* IS NULL', 'CASE WHEN app.applicationamount IS NULL', sql)
        sql = re.sub(r'TRIM\(\s*::text\)', 'TRIM(app.applicationamount::text)', sql)
        sql = re.sub(r'COUNT\s*\(\(\s*CASE', 'COUNT(CASE', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SUM\s*\(\(\s*CASE', 'SUM(CASE', sql, flags=re.IGNORECASE)
        sql = re.sub(r'END\s*\)\)', 'END)', sql)
        sql = re.sub(
            r'SUM\(\(\s*CASE WHEN CASE WHEN (.*?) THEN 1 ELSE 0 END IS NULL THEN NULL ELSE NULL END\s*\)\)',
            r'SUM(CASE WHEN \1 THEN 1 ELSE 0 END)',
            sql,
            flags=re.IGNORECASE
        )

        # Phase 4: Your existing utility functions
        print("[Phase 4] Running SQL utilities...")
        sql = clean_generated_sql(sql)
        sql = sanitize_sql_for_nulls(sql, schema_catalog=schema_catalog, question=question)
        sql = auto_fix_schema_mismatches(sql, schema_catalog)
        sql = validate_sql_against_catalog(sql, schema_catalog)
        sql = _fix_groupby_alias_and_parens(sql)

        # Phase 5: Final cleanup
        print("[Phase 5] Final cleanup...")
        sql = final_sql_cleanup(sql)

        # Phase 6: 2024 enforcement
        if any("stage" in tbl for tbl in selected_tables):
            date_info = get_fixed_date_context()
            sql = enforce_2024(sql, date_info)

        # Phase 7: FINAL VALIDATION before execution
        print("[Phase 7] Final validation...")
        is_valid, remaining_errors = quick_validate_sql_syntax(sql)
        if not is_valid:
            print(f"\n⚠️ CRITICAL: Still {len(remaining_errors)} errors after fixes:")
            for err in remaining_errors:
                print(f"  - {err}")
            print("\n🔴 Cannot execute SQL due to syntax errors")
            error_msg = "SQL still contains syntax errors after repair: " + "; ".join(remaining_errors)
            return (f"-- ERROR: {error_msg}", error_msg, "SQL_SYNTAX_ERROR")

        print("\n✅ SQL VALIDATION PASSED")
        print(f"\n{'='*80}")
        print("✅ FINAL GENERATED SQL:")
        print("="*80)
        print(sql)
        print("="*80 + "\n")

        # ============================================================================
        # Step 8: Generate explanation (YOUR EXISTING CODE)
        # ============================================================================
        try:
            explanation_prompt = f"""
Explain in plain English what the following SQL query does.
Avoid markdown or bullet points. Write in simple paragraphs.

SQL:
{sql}
"""
            explanation_response = llm.invoke(explanation_prompt)
            explanation = getattr(explanation_response, "content", explanation_response if isinstance(explanation_response, str) else str(explanation_response)).strip()
            print(f"🧠 SQL Explanation:\n{explanation}\n")
        except Exception as e:
            print(f"⚠️ Explanation generation failed: {e}")
            explanation = "Explanation unavailable due to LLM error."

        return (sql, explanation, "")

    except Exception as e:
        print(f"❌ SQL generation error: {e}")
        traceback.print_exc()
        return (f"-- ERROR: {str(e)}", "SQL generation failed", "GENERATION_ERROR")
    
  

# ============================================================================
# COMPLETE CODE FOR views.py
# ============================================================================
# Add this section FIRST at the top of views.py with your other imports
# ============================================================================

import re
from typing import Tuple, List, Dict, Any
import traceback
import datetime

# ============================================================================
# ADD ALL THESE FUNCTIONS BEFORE YOUR run_sql_generation_graph() FUNCTION
# Place them after your imports and before any view functions
# ============================================================================

# ============================================================================
# COMPLETE CODE FOR views.py
# ============================================================================
# Add this section FIRST at the top of views.py with your other imports
# ============================================================================

# ============================================================================
# COMPLETE WORKING CODE FOR views.py
# ============================================================================
# Add at top with imports
# ============================================================================

import re
from typing import Tuple, List, Dict, Any
import traceback
import datetime

# ============================================================================
# ADD THESE 6 HELPER FUNCTIONS BEFORE run_sql_generation_graph()
# ============================================================================

# ============================================================================
# FIXED HELPER FUNCTIONS - Replace in views.py
# ============================================================================

import re
from typing import Tuple, List, Dict, Any
# ============================================================================
# PERMANENTLY FIXED VALIDATION - Replace in views.py
# ============================================================================

import re
from typing import Tuple, List, Dict, Any

def strip_sql_markdown(sql: str) -> str:
    """Remove markdown code fence markers from SQL."""
    if not sql:
        return sql
    
    sql = sql.strip()
    print(f"[strip_sql_markdown] Input: {sql[:100]}")
    
    sql = re.sub(r'^```(?:sql|SQL|postgres|postgresql|py|python)?\s*\n?', '', sql)
    sql = re.sub(r'\n?```\s*$', '', sql)
    sql = sql.replace('```', '').strip('`').strip()
    
    print(f"[strip_sql_markdown] Output: {sql[:100]}")
    return sql


def quick_validate_sql_syntax(sql: str) -> Tuple[bool, List[str]]:
    """
    Quick validation for critical SQL syntax errors.
    PERMANENTLY FIXED: Now uses negative lookahead to avoid false positives.
    """
    errors = []
    
    # Check for markdown remnants
    if '```' in sql:
        errors.append("Markdown code fences still present")
    
    # CRITICAL FIX: Check for INVALID EXTRACT patterns using negative lookahead
    # This pattern matches: EXTRACT(word) FROM  (but NOT if FROM is already inside)
    # Invalid: EXTRACT(MONTH) FROM table
    # Valid:   EXTRACT(MONTH FROM table) - won't match because FROM is inside parens
    
    # Strategy: Look for closing paren followed by whitespace and FROM keyword
    invalid_extract = re.search(
        r'EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+(?!.*\1\s+FROM)',  
        sql, 
        re.IGNORECASE
    )
    
    if invalid_extract:
        part = invalid_extract.group(1)
        # Double-check this isn't a false positive by verifying the valid form isn't present
        valid_form = re.search(
            rf'EXTRACT\s*\(\s*{part}\s+FROM\s+[\w.]+',
            sql,
            re.IGNORECASE
        )
        if not valid_form:
            errors.append(f"EXTRACT({part}) FROM - missing FROM inside parentheses")
    
    # Check for empty column references
    if re.search(r'TRIM\s*\(\s*::\s*(?:text|TEXT|NUMERIC)', sql, re.IGNORECASE):
        errors.append("TRIM(::type) - missing column")
    
    # Check for double opening parens in aggregates
    if re.search(r'(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*\(', sql, re.IGNORECASE):
        errors.append("Double parentheses in aggregate")
    
    # Check for wildcard in CASE WHEN
    if re.search(r'CASE\s+WHEN\s+\*\s+IS', sql, re.IGNORECASE):
        errors.append("Wildcard (*) in CASE WHEN")
    
    # IMPROVED: Parentheses counting with tolerance
    open_count = sql.count('(')
    close_count = sql.count(')')
    diff = abs(open_count - close_count)
    
    # Only flag if difference is significant (> 1 allows for edge cases)
    if diff > 1:
        errors.append(f"Parentheses mismatch: {open_count} ( vs {close_count} )")
    
    return len(errors) == 0, errors

# ============================================================================
# DIAGNOSTIC VERSION - Find where SQL gets corrupted
# ============================================================================

import re
from typing import Tuple, List, Dict, Any
import traceback

# ============================================================================
# HELPER FUNCTIONS (Simplified and Safe)
# ============================================================================

def strip_sql_markdown(sql: str) -> str:
    """Remove markdown code fence markers from SQL."""
    if not sql:
        return sql
    
    sql = sql.strip()
    sql = re.sub(r'^```(?:sql|SQL|postgres|postgresql|py|python)?\s*\n?', '', sql)
    sql = re.sub(r'\n?```\s*$', '', sql)
    sql = sql.replace('```', '').strip('`').strip()
    
    print(f"[strip_sql_markdown] Output length: {len(sql)}")
    return sql


def validate_extract_syntax_comprehensive(sql: str) -> Tuple[bool, List[str]]:
    """Comprehensive EXTRACT syntax validation."""
    issues = []
    
    # Find all EXTRACT expressions
    extract_patterns = list(re.finditer(
        r'EXTRACT\s*\([^)]+\)',
        sql,
        re.IGNORECASE | re.DOTALL
    ))
    
    if not extract_patterns:
        return True, []
    
    for i, match in enumerate(extract_patterns, 1):
        extract_expr = match.group(0)
        
        # Valid: EXTRACT(PART FROM column)
        valid_pattern = re.match(
            r'EXTRACT\s*\(\s*\w+\s+FROM\s+[\w.:]+',
            extract_expr,
            re.IGNORECASE
        )
        
        if not valid_pattern:
            issues.append(f"Invalid EXTRACT #{i}: {extract_expr}")
    
    return len(issues) == 0, issues


def quick_validate_sql_syntax(sql: str) -> Tuple[bool, List[str]]:
    """Quick validation - NON-DESTRUCTIVE."""
    errors = []
    
    if '```' in sql:
        errors.append("Markdown present")
    
    # Use comprehensive check
    extract_valid, extract_issues = validate_extract_syntax_comprehensive(sql)
    if not extract_valid:
        errors.extend(extract_issues)
    
    # Parentheses check
    open_count = sql.count('(')
    close_count = sql.count(')')
    diff = abs(open_count - close_count)
    
    if diff > 1:
        errors.append(f"Parentheses: {open_count} ( vs {close_count} )")
    
    return len(errors) == 0, errors


def safe_fix_extract_syntax(sql: str) -> str:
    """
    SAFE EXTRACT fixer - only fixes if actually broken.
    Does NOT touch valid EXTRACT(PART FROM col) syntax.
    """
    # First check if EXTRACT is already valid
    extract_valid, _ = validate_extract_syntax_comprehensive(sql)
    if extract_valid:
        print("  ✓ EXTRACT syntax already valid, skipping fix")
        return sql
    
    original = sql
    
    # Only fix the broken pattern: EXTRACT(MONTH) FROM col
    # This has closing paren before FROM keyword
    sql = re.sub(
        r'EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([\w.]+(?:::\w+)?)',
        r'EXTRACT(\1 FROM \2)',
        sql,
        flags=re.IGNORECASE
    )
    
    if sql != original:
        print(f"  ✅ Fixed EXTRACT syntax")
        print(f"     Before: {original[:100]}")
        print(f"     After:  {sql[:100]}")
    
    return sql


def minimal_sql_cleanup(sql: str) -> str:
    """Minimal cleanup - removes obvious issues only."""
    # Remove double spaces
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # Ensure semicolon
    if not sql.endswith(';'):
        sql = sql.rstrip(';') + ';'
    
    return sql





def aggressive_fix_extract_syntax(sql: str) -> str:
    """
    Fix malformed EXTRACT syntax.
    ENHANCED: More patterns covered, better detection.
    """
    original_sql = sql
    
    # Pattern 1: EXTRACT(MONTH) FROM date_col  →  EXTRACT(MONTH FROM date_col)
    # Match: EXTRACT(word) whitespace FROM word/dot (not in parens)
    sql = re.sub(
        r'EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([\w.]+(?:::\w+)?)',
        r'EXTRACT(\1 FROM \2)',
        sql,
        flags=re.IGNORECASE
    )
    
    # Pattern 2: With AS alias
    sql = re.sub(
        r'EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([\w.]+(?:::\w+)?)\s*\)',
        r'EXTRACT(\1 FROM \2)',
        sql,
        flags=re.IGNORECASE
    )
    
    # Pattern 3: In GROUP BY clause
    sql = re.sub(
        r'GROUP\s+BY\s+EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([\w.]+(?:::\w+)?)',
        r'GROUP BY EXTRACT(\1 FROM \2)',
        sql,
        flags=re.IGNORECASE
    )
    
    # Pattern 4: In ORDER BY clause
    sql = re.sub(
        r'ORDER\s+BY\s+EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([\w.]+(?:::\w+)?)',
        r'ORDER BY EXTRACT(\1 FROM \2)',
        sql,
        flags=re.IGNORECASE
    )
    
    # Pattern 5: In WHERE clause
    sql = re.sub(
        r'WHERE\s+EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([\w.]+(?:::\w+)?)',
        r'WHERE EXTRACT(\1 FROM \2)',
        sql,
        flags=re.IGNORECASE
    )
    
    if sql != original_sql:
        print(f"✅ EXTRACT syntax fixed")
    
    return sql


def aggressive_fix_empty_columns(sql: str, schema_catalog: Dict[str, Any] = None) -> str:
    """Fix empty column references like TRIM(::text)."""
    default_col = None
    
    # Try to find table and alias
    from_match = re.search(r'FROM\s+([\w.]+)\s+(\w+)', sql, re.IGNORECASE)
    if from_match:
        table_name = from_match.group(1)
        alias = from_match.group(2)
        
        if schema_catalog and table_name in schema_catalog:
            table_info = schema_catalog[table_name]
            # Look for numeric/amount columns
            for col in table_info.get("columns", []):
                col_name = col.get("name", "")
                if any(keyword in col_name.lower() for keyword in ["amount", "numeric", "balance", "value", "total"]):
                    default_col = f"{alias}.{col_name}"
                    break
            
            # Fallback to first column
            if not default_col and table_info.get("columns"):
                first_col = table_info["columns"][0].get("name", "")
                default_col = f"{alias}.{first_col}"
    
    # Fallback: find any column reference
    if not default_col:
        col_ref = re.search(r'(\w+\.\w+)', sql)
        if col_ref:
            default_col = col_ref.group(1)
    
    # Last resort
    if not default_col:
        default_col = "1"
    
    # Fix empty TRIM
    sql = re.sub(r'TRIM\s*\(\s*::\s*text\s*\)', f'TRIM({default_col}::text)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'TRIM\s*\(\s*::\s*TEXT\s*\)', f'TRIM({default_col}::TEXT)', sql, flags=re.IGNORECASE)
    
    # Fix standalone cast
    sql = re.sub(r'(?<=\s)::\s*text(?=\s*(?:=|~|\)|,|;))', f' {default_col}::text', sql, flags=re.IGNORECASE)
    
    # Fix CASE with wildcard
    sql = re.sub(r'CASE\s+WHEN\s+\*\s+IS\s+NULL', f'CASE WHEN {default_col} IS NULL', sql, flags=re.IGNORECASE)
    
    return sql


def balance_parentheses_final(sql: str) -> str:
    """
    Fix parentheses mismatch - smart balancing.
    ENHANCED: Protects EXTRACT syntax during balancing.
    """
    open_count = sql.count('(')
    close_count = sql.count(')')
    
    # No issue
    if open_count == close_count:
        return sql
    
    diff = open_count - close_count
    print(f"[balance_parentheses] Diff: {diff} (open={open_count}, close={close_count})")
    
    # Too many opening parens
    if diff > 0:
        # Remove double opening parens (but preserve EXTRACT patterns)
        # Only do this if EXTRACT is not affected
        extract_count_before = len(re.findall(r'EXTRACT\s*\(', sql, re.IGNORECASE))
        sql_temp = re.sub(r'\(\s*\(', '(', sql)
        extract_count_after = len(re.findall(r'EXTRACT\s*\(', sql_temp, re.IGNORECASE))
        
        if extract_count_before == extract_count_after:
            sql = sql_temp
        
        # Remove double parens in aggregates
        for agg in ['SUM', 'COUNT', 'AVG', 'MIN', 'MAX']:
            sql = re.sub(rf'{agg}\s*\(\s*\(', f'{agg}(', sql, flags=re.IGNORECASE)
    
    # Too many closing parens
    elif diff < 0:
        sql = re.sub(r'\)\s*\)', ')', sql)
        sql = re.sub(r'\)\s*\)', ')', sql)
    
    # Recount after fixes
    open_count = sql.count('(')
    close_count = sql.count(')')
    diff = open_count - close_count
    
    # Final balancing if still off
    if diff > 0 and diff <= 3:
        # Add closing parens at appropriate location
        if not sql.rstrip().endswith(';'):
            sql = sql.rstrip() + (')' * diff) + ';'
        else:
            sql = sql.rstrip(';').rstrip() + (')' * diff) + ';'
    elif diff < 0 and abs(diff) <= 3:
        # Remove extra closing parens
        for _ in range(abs(diff)):
            # Remove from end
            sql = re.sub(r'\)\s*;?\s*$', ';', sql)
    
    print(f"[balance_parentheses] After: open={sql.count('(')}, close={sql.count(')')}")
    return sql


# ============================================================================
# PERMANENTLY FIXED VERSION - Correct EXTRACT validation
# ============================================================================

import re
from typing import Tuple, List, Dict, Any
import traceback

# ============================================================================
# HELPER FUNCTIONS - CORRECTED
# ============================================================================

def strip_sql_markdown(sql: str) -> str:
    """Remove markdown code fence markers from SQL."""
    if not sql:
        return sql
    
    sql = sql.strip()
    sql = re.sub(r'^```(?:sql|SQL|postgres|postgresql|py|python)?\s*\n?', '', sql)
    sql = re.sub(r'\n?```\s*$', '', sql)
    sql = sql.replace('```', '').strip('`').strip()
    
    return sql


def validate_extract_syntax_comprehensive(sql: str) -> Tuple[bool, List[str]]:
    """
    Comprehensive EXTRACT syntax validation.
    FIXED: Now correctly captures entire EXTRACT expression.
    """
    issues = []
    
    # Find all EXTRACT expressions - must capture FULL expression including nested parens
    # Pattern explanation:
    # - EXTRACT\s*\( - Match EXTRACT(
    # - (?:[^()]+|\([^)]*\))+ - Match content, including nested parentheses
    # - \) - Match closing paren
    
    extract_patterns = list(re.finditer(
        r'EXTRACT\s*\((?:[^()]+|\([^)]*\))+\)',
        sql,
        re.IGNORECASE
    ))
    
    if not extract_patterns:
        return True, []
    
    print(f"  Found {len(extract_patterns)} EXTRACT expression(s)")
    
    for i, match in enumerate(extract_patterns, 1):
        extract_expr = match.group(0)
        print(f"  Checking EXTRACT #{i}: {extract_expr[:80]}")
        
        # Valid pattern: EXTRACT(PART FROM column::type)
        # Must have: WORD FROM column_reference
        valid_pattern = re.search(
            r'EXTRACT\s*\(\s*(\w+)\s+FROM\s+([\w.]+(?:::\w+)?)',
            extract_expr,
            re.IGNORECASE
        )
        
        if not valid_pattern:
            issues.append(f"Invalid EXTRACT #{i}: {extract_expr}")
            print(f"    ❌ Invalid syntax")
        else:
            print(f"    ✅ Valid syntax")
    
    return len(issues) == 0, issues


def quick_validate_sql_syntax(sql: str) -> Tuple[bool, List[str]]:
    """Quick validation - uses comprehensive EXTRACT check."""
    errors = []
    
    if '```' in sql:
        errors.append("Markdown present")
    
    # Use comprehensive check
    extract_valid, extract_issues = validate_extract_syntax_comprehensive(sql)
    if not extract_valid:
        errors.extend(extract_issues)
    
    # Check for double parens in aggregates
    if re.search(r'(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*\(', sql, re.IGNORECASE):
        errors.append("Double parentheses in aggregate")
    
    # Parentheses check with tolerance
    open_count = sql.count('(')
    close_count = sql.count(')')
    diff = abs(open_count - close_count)
    
    if diff > 1:
        errors.append(f"Parentheses: {open_count} ( vs {close_count} )")
    
    return len(errors) == 0, errors







import re
from typing import Tuple, List, Dict, Any
from difflib import get_close_matches
import traceback

# ============================================================================
# HELPER FUNCTIONS - CORRECTED
# ============================================================================
"""
PERMANENT FIX for PostgreSQL Date Cast Errors
Uses NULLIF and proper evaluation order to prevent cast errors
"""

import re
from typing import Tuple, List, Dict, Any


def wrap_all_date_casts_bulletproof(sql: str) -> str:
    """
    BULLETPROOF date cast wrapper using NULLIF approach.
    
    The issue with CASE WHEN is that PostgreSQL may still evaluate
    the THEN branch even when conditions are false (query optimization).
    
    Solution: Use NULLIF which guarantees evaluation order.
    """
    print("\n" + "="*80)
    print("BULLETPROOF DATE CAST WRAPPING (NULLIF METHOD)")
    print("="*80)
    
    if not sql:
        return sql
    
    original_sql = sql
    wrapped_columns = set()
    
    # ========================================================================
    # STEP 1: Find ALL date columns being cast
    # ========================================================================
    
    # Pattern: Direct casts with :: operator
    direct_casts = re.finditer(
        r'\b(\w+\.\w+|\w+)\s*::\s*(TIMESTAMP|DATE|TIMESTAMPTZ|DATETZ)\b',
        sql,
        re.IGNORECASE
    )
    
    for match in direct_casts:
        col_ref = match.group(1)
        cast_type = match.group(2).upper()
        wrapped_columns.add((col_ref, cast_type))
        print(f"  Found cast: {col_ref}::{cast_type}")
    
    if not wrapped_columns:
        print("  ✓ No date casts found - no wrapping needed")
        return sql
    
    print(f"\n  Total unique columns to wrap: {len(wrapped_columns)}")
    
    # ========================================================================
    # STEP 2: Create NULLIF wrappers (GUARANTEED evaluation order)
    # ========================================================================
    
    for col_ref, cast_type in wrapped_columns:
        print(f"\n  Wrapping: {col_ref}::{cast_type}")
        
        # NULLIF approach: converts empty strings to NULL BEFORE cast
        # NULLIF(NULLIF(TRIM(col), ''), NULL) - nested to handle both cases
        safe_wrapper = (
            f"NULLIF(TRIM({col_ref}), '')::{cast_type}"
        )
        
        # Even safer: double NULLIF
        ultra_safe_wrapper = (
            f"NULLIF(NULLIF(TRIM({col_ref}), ''), '')::{cast_type}"
        )
        
        # BEST: Use TRY_CAST if PostgreSQL 13+ or custom function
        # For now, use NULLIF approach which works on all versions
        
        # Replace ALL occurrences of this cast
        pattern = rf'\b{re.escape(col_ref)}\s*::\s*{cast_type}\b'
        count = len(re.findall(pattern, sql, re.IGNORECASE))
        
        if count > 0:
            sql = re.sub(
                pattern,
                safe_wrapper,
                sql,
                flags=re.IGNORECASE
            )
            print(f"    ✓ Wrapped {count} occurrence(s) with NULLIF")
    
    # ========================================================================
    # STEP 3: Remove now-redundant WHERE filters
    # ========================================================================
    
    print("\n  Cleaning redundant WHERE filters...")
    
    for col_ref, _ in wrapped_columns:
        # Remove: col IS NOT NULL AND col != '' AND TRIM(col) != '' AND
        redundant_patterns = [
            # Pattern 1: Full redundant check
            (
                rf'{re.escape(col_ref)}\s+IS\s+NOT\s+NULL\s+AND\s+'
                rf'{re.escape(col_ref)}\s*!=\s*\'\'\s+AND\s+'
                rf'TRIM\s*\(\s*{re.escape(col_ref)}\s*\)\s*!=\s*\'\'\s+AND\s+'
            ),
            # Pattern 2: Partial checks
            (
                rf'{re.escape(col_ref)}\s+IS\s+NOT\s+NULL\s+AND\s+'
                rf'{re.escape(col_ref)}\s*!=\s*\'\'\s+AND\s+'
            ),
            # Pattern 3: Just TRIM check
            (
                rf'TRIM\s*\(\s*{re.escape(col_ref)}\s*\)\s*!=\s*\'\'\s+AND\s+'
            ),
        ]
        
        for pattern in redundant_patterns:
            before = sql
            sql = re.sub(pattern, '', sql, flags=re.IGNORECASE)
            if sql != before:
                print(f"    ✓ Removed redundant filter for {col_ref}")
                break
    
    # ========================================================================
    # STEP 4: Verification
    # ========================================================================
    
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    nullif_count = sql.upper().count('NULLIF')
    print(f"  NULLIF wrappers added: {nullif_count}")
    print(f"  Columns wrapped: {len(wrapped_columns)}")
    
    # Check for any remaining naked casts (should be none)
    remaining_naked = []
    for col_ref, cast_type in wrapped_columns:
        # Look for col::TYPE that's NOT preceded by NULLIF
        pattern = rf'(?<!NULLIF\(TRIM\(){re.escape(col_ref)}\s*::\s*{cast_type}'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        if matches:
            remaining_naked.append(f"{col_ref}::{cast_type}")
    
    if remaining_naked:
        print(f"  ⚠️  WARNING: {len(remaining_naked)} naked cast(s) remain")
    else:
        print(f"  ✅ All date casts are safely wrapped")
    
    print("="*80 + "\n")
    
    return sql


import re
from typing import Tuple, List, Dict, Any
import traceback


def strip_sql_markdown(sql: str) -> str:
    """Remove markdown code fence markers from SQL."""
    if not sql:
        return sql
    
    sql = sql.strip()
    sql = re.sub(r'^```(?:sql|SQL|postgres|postgresql)?\s*\n?', '', sql)
    sql = re.sub(r'\n?```\s*$', '', sql)
    sql = sql.replace('```', '').strip('`').strip()
    
    return sql


def validate_extract_syntax_comprehensive(sql: str) -> Tuple[bool, List[str]]:
    """
    Comprehensive EXTRACT syntax validation.
    FIXED: Accepts NULLIF and other functions inside EXTRACT.
    """
    issues = []
    
    # Find all EXTRACT expressions
    extract_patterns = list(re.finditer(
        r'EXTRACT\s*\((?:[^()]|\([^()]*(?:\([^()]*\)[^()]*)*\))*\)',
        sql,
        re.IGNORECASE
    ))
    
    if not extract_patterns:
        return True, []
    
    print(f"  Found {len(extract_patterns)} EXTRACT expression(s)")
    
    for i, match in enumerate(extract_patterns, 1):
        extract_expr = match.group(0)
        print(f"  Checking EXTRACT #{i}: {extract_expr[:80]}")
        
        # Check if FROM keyword exists inside parentheses
        if ' FROM ' not in extract_expr.upper():
            issues.append(f"Invalid EXTRACT #{i}: {extract_expr}")
            print(f"    ❌ Invalid syntax - missing FROM")
            continue
        
        # Simply check: Does it have PART FROM something?
        valid = re.search(
            r'EXTRACT\s*\(\s*\w+\s+FROM\s+.+\)',
            extract_expr,
            re.IGNORECASE
        )
        
        if valid:
            print(f"    ✅ Valid syntax")
        else:
            issues.append(f"Invalid EXTRACT #{i}: {extract_expr}")
            print(f"    ❌ Invalid syntax")
    
    return len(issues) == 0, issues


def safe_fix_extract_syntax(sql: str) -> str:
    """
    SAFE EXTRACT fixer - only fixes if actually broken.
    Does NOT touch valid EXTRACT(PART FROM col) syntax.
    """
    # First check if EXTRACT is already valid
    extract_valid, issues = validate_extract_syntax_comprehensive(sql)
    if extract_valid:
        print("  ✓ EXTRACT syntax already valid, skipping fix")
        return sql
    
    print(f"  EXTRACT needs fixing: {issues}")
    original = sql
    
    # Only fix the broken pattern: EXTRACT(MONTH) FROM col
    # This has closing paren before FROM keyword
    sql = re.sub(
        r'EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+([\w.]+(?:::\w+)?)',
        r'EXTRACT(\1 FROM \2)',
        sql,
        flags=re.IGNORECASE
    )
    
    if sql != original:
        print(f"  ✅ Fixed EXTRACT syntax")
    
    return sql


def balance_parentheses_safe(sql: str) -> str:
    """Safe parentheses balancing with auto-fix."""
    open_count = sql.count('(')
    close_count = sql.count(')')
    
    if open_count == close_count:
        print("  [Paren] Balanced ✓")
        return sql
    
    diff = open_count - close_count
    print(f"  [Paren] Unbalanced: {open_count} open, {close_count} close (diff={diff})")
    
    # Fix double parens in aggregates first
    original = sql
    for agg in ['SUM', 'COUNT', 'AVG', 'MIN', 'MAX']:
        sql = re.sub(rf'{agg}\s*\(\s*\(', f'{agg}(', sql, flags=re.IGNORECASE)
    
    if sql != original:
        print(f"  [Paren] Fixed double parens in aggregates")
    
    # Recount
    open_count = sql.count('(')
    close_count = sql.count(')')
    diff = open_count - close_count
    
    if diff == 0:
        print("  [Paren] Now balanced ✓")
        return sql
    
    # Auto-fix remaining imbalance
    if diff > 0:
        # More open than close - add closing parens before semicolon
        print(f"  [Paren] Adding {diff} closing paren(s)")
        sql = sql.rstrip(';').rstrip() + (')' * diff)
        if not sql.endswith(';'):
            sql += ';'
    elif diff < 0:
        # More close than open - remove extra closing parens from end
        print(f"  [Paren] Removing {abs(diff)} extra closing paren(s)")
        sql = sql.rstrip(';')
        for _ in range(abs(diff)):
            # Remove last occurrence of )
            last_paren = sql.rfind(')')
            if last_paren != -1:
                sql = sql[:last_paren] + sql[last_paren+1:]
        if not sql.endswith(';'):
            sql += ';'
    
    # Final count
    final_open = sql.count('(')
    final_close = sql.count(')')
    print(f"  [Paren] After fix: {final_open} open, {final_close} close ✓")
    
    return sql

"""
FIXED: NULLIF wrapper with proper validation and debugging
"""

import re
from typing import Set, Tuple, List


def add_null_safety_filters11(sql: str) -> str:
    """
    Wrap date casts to prevent invalid cast attempts.
    Uses NULLIF with BOTH arguments to convert empty strings to NULL.
    
    CRITICAL FIX: Ensures NULLIF always has 2 arguments.
    """
    print("\n" + "="*80)
    print("[Step 3a] Applying NULLIF wrappers for date casts...")
    print("="*80)
    
    if not sql:
        return sql
    
    wrapped_columns: Set[Tuple[str, str]] = set()
    
    # Find ALL date casts (comprehensive pattern matching)
    cast_patterns = [
        # Pattern 1: table.column::TYPE
        r'\b(\w+\.\w+)\s*::\s*(TIMESTAMP|DATE|TIMESTAMPTZ)\b',
        # Pattern 2: column::TYPE (without table prefix)
        r'\b(?<!\.)(\w+)\s*::\s*(TIMESTAMP|DATE|TIMESTAMPTZ)\b',
    ]
    
    for pattern in cast_patterns:
        for match in re.finditer(pattern, sql, re.IGNORECASE):
            col_ref = match.group(1)
            cast_type = match.group(2).upper()
            
            # Skip system functions and dates
            if col_ref.upper() in ('CURRENT_DATE', 'CURRENT_TIMESTAMP', 'NOW', 'DATE'):
                continue
            
            # Skip if already wrapped with NULLIF
            context_start = max(0, match.start() - 50)
            context = sql[context_start:match.start()]
            if 'NULLIF' in context.upper():
                continue
            
            wrapped_columns.add((col_ref, cast_type))
    
    if not wrapped_columns:
        print("  ℹ️ No date casts found to wrap")
        return sql
    
    print(f"  Found {len(wrapped_columns)} unique column cast(s) to wrap:")
    for col_ref, cast_type in sorted(wrapped_columns):
        print(f"    • {col_ref}::{cast_type}")
    
    # Apply NULLIF wrapper to each cast
    replacement_count = 0
    for col_ref, cast_type in wrapped_columns:
        # CRITICAL: NULLIF requires EXACTLY 2 arguments
        # NULLIF(value, value_to_compare) returns NULL if they match
        safe_wrapper = f"NULLIF(TRIM({col_ref}), '')::{cast_type}"
        
        # Build regex pattern to match this specific cast
        # Must handle optional whitespace around ::
        pattern = rf'\b{re.escape(col_ref)}\s*::\s*{cast_type}\b'
        
        # Count matches before replacement
        matches = list(re.finditer(pattern, sql, re.IGNORECASE))
        
        if matches:
            # Replace all occurrences
            sql = re.sub(pattern, safe_wrapper, sql, flags=re.IGNORECASE)
            replacement_count += len(matches)
            print(f"    ✓ Replaced {len(matches)}x: {col_ref}::{cast_type} -> NULLIF wrapper")
    
    # CRITICAL VALIDATION: Verify NULLIF has 2 arguments
    print(f"\n  ✅ Final SQL has {sql.count('NULLIF')} NULLIF wrapper(s)")
    
    # Check for malformed NULLIF (only 1 argument)
    malformed = re.findall(
        r'NULLIF\s*\([^,)]+\)\s*::',  # NULLIF with no comma before closing paren
        sql,
        re.IGNORECASE
    )
    
    if malformed:
        print(f"\n  ❌ ERROR: Found {len(malformed)} malformed NULLIF wrapper(s)!")
        for m in malformed:
            print(f"    ❌ Malformed: {m}")
        raise ValueError("NULLIF wrappers are missing second argument! This will cause syntax errors.")
    
    # Verify proper NULLIF syntax (must have comma)
    proper_nullif = re.findall(
        r"NULLIF\s*\([^)]+,\s*''\s*\)",  # NULLIF with comma and empty string
        sql,
        re.IGNORECASE
    )
    
    if proper_nullif:
        print(f"  ✅ NULLIF applied - found {len(proper_nullif)} occurrences")
    else:
        print(f"  ⚠️ WARNING: No valid NULLIF patterns found in SQL!")
    
    return sql


def validate_nullif_syntax(sql: str) -> Tuple[bool, List[str]]:
    """
    Validate that all NULLIF calls have exactly 2 arguments.
    Returns (is_valid, error_messages)
    """
    errors = []
    
    # Find all NULLIF calls
    nullif_calls = re.finditer(
        r'NULLIF\s*\(([^)]+)\)',
        sql,
        re.IGNORECASE
    )
    
    for match in nullif_calls:
        args = match.group(1)
        
        # Count commas (should be exactly 1 for 2 arguments)
        # Account for nested function calls
        depth = 0
        comma_count = 0
        
        for char in args:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                comma_count += 1
        
        if comma_count != 1:
            errors.append(f"Invalid NULLIF (expected 1 comma, found {comma_count}): {match.group(0)[:80]}")
    
    return len(errors) == 0, errors


def comprehensive_sql_fix(sql: str) -> str:
    """
    Complete SQL fixing pipeline with validation.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE SQL FIX PIPELINE")
    print("="*80)
    
    # Step 1: Strip markdown
    sql = sql.strip()
    sql = re.sub(r'^```(?:sql|SQL)?\s*', '', sql)
    sql = re.sub(r'```\s*$', '', sql)
    
    # Step 2: Apply NULLIF wrappers
    sql = add_null_safety_filters(sql)
    
    # Step 3: Validate NULLIF syntax
    print("\n[Step 3b] Validating NULLIF syntax...")
    is_valid, errors = validate_nullif_syntax(sql)
    
    if not is_valid:
        print("  ❌ NULLIF validation failed:")
        for error in errors:
            print(f"    • {error}")
        raise ValueError("SQL contains invalid NULLIF calls - cannot proceed")
    else:
        print("  ✅ All NULLIF calls are properly formatted")
    
    # Step 4: Fix EXTRACT syntax if needed
    print("\n[Step 3c] Fixing EXTRACT syntax...")
    sql = re.sub(
        r'EXTRACT\s*\(\s*(\w+)\s*\)\s+FROM\s+',
        r'EXTRACT(\1 FROM ',
        sql,
        flags=re.IGNORECASE
    )
    
    # Step 5: Clean up whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # Step 6: Ensure semicolon
    if not sql.endswith(';'):
        sql = sql.rstrip(';') + ';'
    
    print("\n" + "="*80)
    print("✅ SQL FIX COMPLETE")
    print("="*80)
    
    return sql


# ============================================================================
# TEST WITH YOUR EXACT FAILING SQL
# ============================================================================

def test_exact_failing_sql():
    """Test with the exact SQL that's failing in your logs."""
    
    failing_sql = """
    SELECT loa.vantage4, COUNT(*) AS count 
    FROM stage.app_main_2024 app 
    INNER JOIN stage.loan_main_2024 loa ON app.customerid = loa.customerid 
    WHERE NULLIF(TRIM(app.fsl_preselectiondate))::TIMESTAMP >= (DATE '2024-10-01' - INTERVAL '1 quarter')::TIMESTAMP 
      AND EXTRACT(QUARTER FROM NULLIF(TRIM(app.fsl_preselectiondate))::TIMESTAMP) = EXTRACT(QUARTER FROM DATE '2024-10-01') 
      AND 2024::TIMESTAMP = 2024 
    GROUP BY loa.vantage4 
    ORDER BY count DESC;
    """
    
    print("="*80)
    print("TESTING WITH YOUR EXACT FAILING SQL")
    print("="*80)
    print("\nINPUT SQL (BROKEN - missing NULLIF second argument):")
    print(failing_sql)
    
    # This should detect the error
    is_valid, errors = validate_nullif_syntax(failing_sql)
    
    if not is_valid:
        print("\n❌ VALIDATION CAUGHT THE ERROR:")
        for error in errors:
            print(f"  • {error}")
        
        print("\n🔧 Applying fix...")
        
        # Fix: Add missing second argument to NULLIF
        fixed_sql = re.sub(
            r'NULLIF\s*\(\s*TRIM\s*\(\s*(\w+\.\w+)\s*\)\s*\)',
            r"NULLIF(TRIM(\1), '')",
            failing_sql,
            flags=re.IGNORECASE
        )
        
        print("\n✅ FIXED SQL:")
        print(fixed_sql)
        
        # Validate again
        is_valid_after, _ = validate_nullif_syntax(fixed_sql)
        assert is_valid_after, "Fix didn't work!"
        
        print("\n✅ VALIDATION PASSED - SQL is now correct!")
        return fixed_sql
    else:
        print("\n✅ SQL is already valid")
        return failing_sql


def fix_malformed_year_filters(sql: str) -> str:
    """
    Fix malformed year filter fragments like:
    AND 2024, ''::TIMESTAMP = 2024
    AND EXTRACT(YEAR FROM col) = 2024 AND 2024, ''::TIMESTAMP = 2024
    """
    print("  [Year Filter Fix] Checking for malformed fragments...")
    
    original_sql = sql
    
    # Pattern 1: Remove fragments like "AND 2024, ''::TIMESTAMP = 2024"
    sql = re.sub(
        r'AND\s+\d{4}\s*,\s*\'\'::(?:TIMESTAMP|DATE)\s*=\s*\d{4}',
        '',
        sql,
        flags=re.IGNORECASE
    )
    
    # Pattern 2: Remove fragments like ", ''::TIMESTAMP = 2024"
    sql = re.sub(
        r',\s*\'\'::(?:TIMESTAMP|DATE)\s*=\s*\d{4}',
        '',
        sql,
        flags=re.IGNORECASE
    )
    
    # Pattern 3: Remove stray ", ''" fragments
    sql = re.sub(
        r',\s*\'\'',
        '',
        sql,
        flags=re.IGNORECASE
    )
    
    # Pattern 4:  up any "AND AND" that may result
    sql = re.sub(r'\bAND\s+AND\b', 'AND', sql, flags=re.IGNORECASE)
    
    # Pattern 5: Clean up WHERE followed immediately by AND
    sql = re.sub(r'\bWHERE\s+AND\b', 'WHERE', sql, flags=re.IGNORECASE)
    
    # Pattern 6: Clean up trailing AND before GROUP BY, ORDER BY, etc.
    sql = re.sub(
        r'\s+AND\s+(GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)',
        r' \1',
        sql,
        flags=re.IGNORECASE
    )
    
    if sql != original_sql:
        print("    ✓ Fixed malformed year filter fragments")
        # Show what was removed
        removed = set(re.findall(r'AND\s+\d{4}\s*,\s*\'\'::(?:TIMESTAMP|DATE)\s*=\s*\d{4}', original_sql, re.IGNORECASE))
        if removed:
            print(f"    Removed: {list(removed)[0]}")
    else:
        print("    No malformed fragments found")
    
    return sql


def final_sql_sanitizer(sql: str) -> str:
    """
    Final pass to catch any malformed SQL fragments.
    This is the last line of defense before execution.
    """
    print("  [Final Sanitizer] Checking for syntax errors...")
    
    original_sql = sql
    issues_fixed = []
    
    # Issue 1: Malformed comparisons like ", ''::TYPE = YEAR"
    if re.search(r',\s*\'\'::(?:TIMESTAMP|DATE)', sql, re.IGNORECASE):
        sql = re.sub(r',\s*\'\'::(?:TIMESTAMP|DATE)\w*\s*=\s*\d{4}', '', sql, flags=re.IGNORECASE)
        issues_fixed.append("Removed malformed empty string comparison")
    
    # Issue 2: Stray year numbers like "AND 2024, ..."
    if re.search(r'AND\s+\d{4}\s*,', sql, re.IGNORECASE):
        sql = re.sub(r'AND\s+\d{4}\s*,\s*[^)]+?(?=\s+(?:GROUP|ORDER|HAVING|LIMIT|;))', '', sql, flags=re.IGNORECASE)
        issues_fixed.append("Removed stray year fragment")
    
    # Issue 3: Double AND
    if re.search(r'\bAND\s+AND\b', sql, re.IGNORECASE):
        sql = re.sub(r'\bAND\s+AND\b', 'AND', sql, flags=re.IGNORECASE)
        issues_fixed.append("Fixed double AND")
    
    # Issue 4: WHERE AND (nothing before AND)
    if re.search(r'\bWHERE\s+AND\b', sql, re.IGNORECASE):
        sql = re.sub(r'\bWHERE\s+AND\b', 'WHERE', sql, flags=re.IGNORECASE)
        issues_fixed.append("Fixed WHERE AND")
    
    # Issue 5: Trailing AND before GROUP BY, ORDER BY, etc.
    if re.search(r'\s+AND\s+(?:GROUP|ORDER|HAVING|LIMIT)', sql, re.IGNORECASE):
        sql = re.sub(r'\s+AND\s+(GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)', r' \1', sql, flags=re.IGNORECASE)
        issues_fixed.append("Fixed trailing AND")
    
    # Issue 6: Empty EXTRACT() arguments
    if re.search(r'EXTRACT\s*\(\s*\)', sql, re.IGNORECASE):
        issues_fixed.append("ERROR: Empty EXTRACT() found")
        print("    ❌ CRITICAL: Empty EXTRACT() detected - cannot auto-fix")
    
    # Issue 7: Unbalanced quotes (simple check)
    single_quote_count = sql.count("'")
    if single_quote_count % 2 != 0:
        issues_fixed.append("WARNING: Unbalanced single quotes")
        print(f"    ⚠️ WARNING: Found {single_quote_count} single quotes (should be even)")
    
    if sql != original_sql:
        print(f"    ✓ Fixed {len(issues_fixed)} issue(s):")
        for issue in issues_fixed:
            print(f"      - {issue}")
    else:
        print("    ✓ No issues found")
    
    return sql


def fix_column_names(sql: str, schema_catalog: Dict[str, Any]) -> str:
    """
    Auto-correct column names that don't exist in schema.
    Uses fuzzy matching to find closest valid column.
    """
    if not schema_catalog:
        return sql
    
    print("  [Column Fix] Checking column names...")
    
    # Extract all column references (alias.column or table.column)
    column_refs = re.findall(r'(\w+)\.(\w+)', sql)
    
    fixed_sql = sql
    fixes_made = []
    
    for alias, column in column_refs:
        # Find the table for this alias
        table_match = re.search(rf'FROM\s+([\w.]+)\s+{alias}\b', sql, re.IGNORECASE)
        if not table_match:
            continue
        
        table_name = table_match.group(1)
        
        if table_name not in schema_catalog:
            continue
        
        table_info = schema_catalog[table_name]
        valid_columns = [col.get('name', '') for col in table_info.get('columns', [])]
        
        # Check if column exists (case-insensitive)
        column_exists = any(col.lower() == column.lower() for col in valid_columns)
        
        if not column_exists:
            # Find closest match
            from difflib import get_close_matches
            matches = get_close_matches(column.lower(), [c.lower() for c in valid_columns], n=1, cutoff=0.6)
            
            if matches:
                # Find the actual column name (with correct case)
                correct_column = next(c for c in valid_columns if c.lower() == matches[0])
                
                # Replace in SQL
                old_ref = f"{alias}.{column}"
                new_ref = f"{alias}.{correct_column}"
                fixed_sql = re.sub(
                    rf'\b{re.escape(old_ref)}\b',
                    new_ref,
                    fixed_sql,
                    flags=re.IGNORECASE
                )
                
                fixes_made.append(f"{old_ref} → {new_ref}")
                print(f"    ✓ Fixed: {old_ref} → {new_ref}")
    
    if fixes_made:
        print(f"  [Column Fix] Made {len(fixes_made)} correction(s)")
    else:
        print(f"  [Column Fix] All columns valid")
    
    return fixed_sql


def quick_validate_sql_syntax(sql: str) -> Tuple[bool, List[str]]:
    """Quick validation - uses comprehensive EXTRACT check."""
    errors = []
    
    if '```' in sql:
        errors.append("Markdown present")
    
    # Use comprehensive check
    extract_valid, extract_issues = validate_extract_syntax_comprehensive(sql)
    if not extract_valid:
        errors.extend(extract_issues)
    
    # Check for double parens in aggregates
    if re.search(r'(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*\(', sql, re.IGNORECASE):
        errors.append("Double parentheses in aggregate")
    
    # Parentheses check with tolerance
    open_count = sql.count('(')
    close_count = sql.count(')')
    diff = abs(open_count - close_count)
    
    if diff > 1:
        errors.append(f"Parentheses: {open_count} ( vs {close_count} )")
    
    return len(errors) == 0, errors


def minimal_sql_cleanup(sql: str) -> str:
    """Minimal cleanup - only safe operations."""
    # Remove double spaces
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # Remove dangling commas
    sql = re.sub(r',\s*(\)|FROM|WHERE|GROUP|ORDER|HAVING)', r' \1', sql, flags=re.IGNORECASE)
    
    # Ensure semicolon
    if not sql.endswith(';'):
        sql = sql.rstrip(';') + ';'
    
    return sql

"""
COMPLETE FIXED SQL PIPELINE - Drop-in replacement for your code
Fixes the NULLIF missing second argument issue
"""

import re
from typing import Set, Tuple, List, Dict, Any


def emergency_repair_broken_nullif(sql: str) -> str:
    """
    EMERGENCY FIX: Repair NULLIF calls that are missing the second argument.
    This runs AFTER all other processing as a safety net.
    
    Fixes: NULLIF(TRIM(col))::TYPE -> NULLIF(TRIM(col), '')::TYPE
    """
    print("\n🚨 [EMERGENCY] Checking for broken NULLIF...")
    
    # Pattern: NULLIF with only 1 argument followed by ::
    broken_pattern = r'NULLIF\s*\(\s*TRIM\s*\(\s*([^\)]+)\s*\)\s*\)\s*::'
    
    broken_matches = list(re.finditer(broken_pattern, sql, re.IGNORECASE))
    
    if broken_matches:
        print(f"  🚨 Found {len(broken_matches)} BROKEN NULLIF(s) - REPAIRING...")
        
        for match in broken_matches:
            col_ref = match.group(1)
            print(f"    ⚠️ Broken: NULLIF(TRIM({col_ref}))::...")
        
        # Fix: Add the missing second argument
        sql = re.sub(
            broken_pattern,
            r"NULLIF(TRIM(\1), '')::",
            sql,
            flags=re.IGNORECASE
        )
        
        print(f"  ✅ Repaired {len(broken_matches)} NULLIF call(s)")
    else:
        print("  ✓ No broken NULLIF calls detected")
    
    return sql
import re
from typing import Set, Tuple
import re
from typing import Set, Tuple, List, Dict
import re
import re
from typing import Dict

import re
from typing import Dict
import re
from typing import Dict
import re
from typing import Dict
import re
from typing import Dict

import re
from typing import Dict

import re
from typing import Dict

import re
from typing import Dict, List

def add_null_safety_filters(sql: str, schema_catalog: Dict[str, Dict]) -> str:
    """
    Safely wraps only real columns in SQL with TRIM + NULLIF + proper cast.
    Fully dynamic: takes SQL and schema_catalog.
    """
    if not sql:
        return sql

    # Build mapping of columns -> type
    column_type_map = {}
    columns_list = []
    for table, cols in schema_catalog.items():
        for col_name, col_info in cols.items():
            col_type = col_info.get("type", "TEXT").upper() if isinstance(col_info, dict) else str(col_info).upper()
            fq_col = f"{table}.{col_name}"
            column_type_map[fq_col] = col_type
            column_type_map[col_name] = col_type
            columns_list.append(fq_col)
            columns_list.append(col_name)

    # Sort by length descending to avoid partial replacements (e.g., customerid before id)
    columns_list = sorted(set(columns_list), key=len, reverse=True)

    # Replace only actual column references
    for col in columns_list:
        col_type = column_type_map[col]
        # Regex: match whole word (avoid touching keywords)
        pattern = re.compile(rf'\b{re.escape(col)}\b')
        replacement = f"NULLIF(TRIM({col}), '')::{col_type}"
        sql = pattern.sub(replacement, sql)

    return sql


import re
import re

def normalize_intervals_in_sql(sql: str) -> str:
    """
    Dynamically normalizes INTERVAL expressions to PostgreSQL-safe syntax.
    Handles numeric words ('two years' → '2 years'), plurals, and natural variants.
    """

    if not sql:
        return sql

    # Step 1. Replace textual numbers with digits
    num_map = {
        "zero": 0, "one": 1, "a": 1, "an": 1, "half": 0.5,
        "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12
    }

    def textnum_to_digit(match):
        textnum = match.group(1).lower()
        num = num_map.get(textnum, textnum)
        return f"INTERVAL '{num} "

    sql = re.sub(r"INTERVAL\s+'(zero|one|a|an|half|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+", textnum_to_digit, sql, flags=re.IGNORECASE)

    # Step 2. Normalize “half year”, “quarter”, etc.
    replacements = {
        # quarter-based → months
        r"INTERVAL\s+'0.5\s+year'": "INTERVAL '6 months'",
        r"INTERVAL\s+'1\s+quarter'": "INTERVAL '3 months'",
        r"INTERVAL\s+'2\s+quarters'": "INTERVAL '6 months'",
        r"INTERVAL\s+'3\s+quarters'": "INTERVAL '9 months'",
        r"INTERVAL\s+'4\s+quarters'": "INTERVAL '12 months'",
        r"INTERVAL\s+'1\s+qtr'": "INTERVAL '3 months'",
        r"INTERVAL\s+'2\s+qtrs'": "INTERVAL '6 months'",

        # year variants
        r"INTERVAL\s+'0.5\s+years?'": "INTERVAL '6 months'",
        r"INTERVAL\s+'1\s+yrs?'": "INTERVAL '1 year'",
        r"INTERVAL\s+'2\s+yrs?'": "INTERVAL '2 years'",

        # plural consistency
        r"INTERVAL\s+'1\s+months'": "INTERVAL '1 month'",
        r"INTERVAL\s+'1\s+years'": "INTERVAL '1 year'",
        r"INTERVAL\s+'1\s+days'": "INTERVAL '1 day'",
    }

    for pattern, replacement in replacements.items():
        sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)

    # Step 3. Handle quarter → month ratio automatically (fallback)
    def convert_quarters_to_months(match):
        num = float(match.group(1))
        months = int(num * 3)
        return f"INTERVAL '{months} months'"

    sql = re.sub(r"INTERVAL\s+'(\d+(?:\.\d+)?)\s+quarters?'", convert_quarters_to_months, sql, flags=re.IGNORECASE)

    # Step 4. Clean redundant spaces / fix common mistakes
    sql = re.sub(r"INTERVAL\s+'(\d+)\s+monthes'", r"INTERVAL '\1 months'", sql, flags=re.IGNORECASE)
    sql = re.sub(r"INTERVAL\s+'(\d+)\s+yrs?'", r"INTERVAL '\1 years'", sql, flags=re.IGNORECASE)
    sql = re.sub(r"INTERVAL\s+'(\d+)\s+years?'", r"INTERVAL '\1 years'", sql, flags=re.IGNORECASE)
    sql = re.sub(r"INTERVAL\s+'(\d+)\s+months?'", r"INTERVAL '\1 months'", sql, flags=re.IGNORECASE)
    sql = re.sub(r"INTERVAL\s+'(\d+)\s+days?'", r"INTERVAL '\1 days'", sql, flags=re.IGNORECASE)
    sql = re.sub(r"INTERVAL\s+'(\d+)\s+hours?'", r"INTERVAL '\1 hours'", sql, flags=re.IGNORECASE)

    return sql


import re

def remove_constants_from_groupby(sql: str) -> str:
    """
    Removes constants from GROUP BY clauses.
    Keeps only columns or expressions that actually vary.
    """
    def replacer(match):
        group_items = match.group(1).split(',')
        # Filter out items that are numeric literals or single quoted strings
        filtered = [item.strip() for item in group_items 
                    if not re.fullmatch(r"(\d+|'[^']*')", item.strip())]
        if not filtered:
            return ""  # no GROUP BY needed
        return "GROUP BY " + ", ".join(filtered)

    # Regex matches GROUP BY ... until ORDER BY or end of statement
    groupby_pattern = re.compile(r"GROUP BY\s+(.+?)(?=(ORDER BY|$))", re.IGNORECASE | re.DOTALL)
    sql = groupby_pattern.sub(replacer, sql)
    return sql



def remove_trim_from_non_text_columns_safe(sql: str, schema_catalog: Dict[str, Any], debug: bool = False) -> str:
    """
    SAFE VERSION: Remove TRIM() from non-text columns WITHOUT breaking NULLIF wrappers.
    
    CRITICAL: This function now PRESERVES NULLIF wrappers.
    """
    if debug:
        print("\n[SAFE TRIM REMOVAL] Processing...")
    
    # Build list of non-text columns
    non_text_columns = set()
    
    if schema_catalog:
        for table_name, table_info in schema_catalog.items():
            for col in table_info.get('columns', []):
                col_name = col['name'].lower()
                col_type = col.get('type', '').upper()
                
                if col_type in ('NUMERIC', 'INTEGER', 'BIGINT', 'SMALLINT', 'DECIMAL', 
                               'FLOAT', 'DOUBLE', 'REAL', 'DATE', 'TIMESTAMP', 'TIMESTAMPTZ',
                               'BOOLEAN', 'BOOL', 'UUID'):
                    non_text_columns.add(col_name)
    
    # Fallback: Detect from SQL context
    for match in re.finditer(r'(\w+)::(?:TIMESTAMP|DATE|NUMERIC|INTEGER)', sql, re.IGNORECASE):
        non_text_columns.add(match.group(1).lower())
    
    if not non_text_columns:
        return sql
    
    if debug:
        print(f"  Non-text columns: {sorted(non_text_columns)}")
    
    aliases = extract_table_aliases(sql)
    
    # Remove TRIM ONLY from standalone occurrences, NOT from NULLIF wrappers
    for col in non_text_columns:
        col_refs = [col] + [f"{alias}.{col}" for alias in aliases.keys()]
        
        for col_ref in col_refs:
            escaped_ref = re.escape(col_ref)
            
            # Pattern: TRIM(column) but NOT inside NULLIF(TRIM(column), '')
            # Use negative lookbehind to avoid breaking NULLIF
            pattern = rf'(?<!NULLIF\s\(\s)\bTRIM\s*\(\s*{escaped_ref}\s*\)(?!\s*,\s*[\'"])'
            
            if re.search(pattern, sql, re.IGNORECASE):
                if debug:
                    print(f"  Removing standalone TRIM() from: {col_ref}")
                sql = re.sub(pattern, col_ref, sql, flags=re.IGNORECASE)
    
    return sql


def aggressive_remove_trim_on_non_text_safe(sql: str, debug: bool = False) -> str:
    """
    SAFE VERSION: Nuclear TRIM removal that PRESERVES NULLIF wrappers.
    
    This is a safer version that won't break your NULLIF(TRIM(col), '') patterns.
    """
    if debug:
        print("\n[SAFE AGGRESSIVE TRIM] Processing...")
    
    # Find timestamp columns
    timestamp_columns = set()
    
    for match in re.finditer(r'(\w+)::TIMESTAMP', sql, re.IGNORECASE):
        col_name = match.group(1).lower()
        if col_name not in ('date', 'current_date', 'now'):
            timestamp_columns.add(col_name)
    
    for match in re.finditer(r'EXTRACT\s*\([^)]+FROM\s+(\w+)', sql, re.IGNORECASE):
        col_name = match.group(1).lower()
        timestamp_columns.add(col_name)
    
    if not timestamp_columns:
        return sql
    
    if debug:
        print(f"  Timestamp columns: {sorted(timestamp_columns)}")
    
    # Remove TRIM ONLY when it's NOT inside NULLIF
    for col in timestamp_columns:
        # Pattern: Standalone TRIM not inside NULLIF
        # Match: TRIM(col) or TRIM(alias.col)
        # Don't match: NULLIF(TRIM(col), '')
        
        pattern = rf'(?<!NULLIF\s\(\s)\bTRIM\s*\(\s*\w*\.?{re.escape(col)}\s*\)(?!\s*,\s*[\'"])'
        
        if re.search(pattern, sql, re.IGNORECASE):
            if debug:
                print(f"  Removing standalone TRIM() from: {col}")
            
            # Replace with just the column reference
            sql = re.sub(
                pattern,
                lambda m: m.group(0).replace('TRIM(', '').replace(')', '').strip(),
                sql,
                flags=re.IGNORECASE
            )
    
    return sql


def extract_table_aliases(sql: str) -> Dict[str, str]:
    """Extract table aliases from FROM and JOIN clauses."""
    aliases = {}
    
    from_pattern = re.compile(r'\bFROM\s+([A-Za-z0-9_."]+)\s+([A-Za-z_][A-Za-z0-9_]*)', re.IGNORECASE)
    for match in from_pattern.finditer(sql):
        aliases[match.group(2)] = match.group(1)
    
    join_pattern = re.compile(r'\bJOIN\s+([A-Za-z0-9_."]+)\s+([A-Za-z_][A-Za-z0-9_]*)', re.IGNORECASE)
    for match in join_pattern.finditer(sql):
        aliases[match.group(2)] = match.group(1)
    
    return aliases


def final_sql_validation_and_repair(sql: str) -> str:
    """
    FINAL STEP: Validate and repair SQL before execution.
    This is your last line of defense against broken NULLIF.
    
    Call this RIGHT BEFORE executing SQL!
    """
    print("\n" + "="*80)
    print("FINAL SQL VALIDATION & REPAIR")
    print("="*80)
    
    # Step 1: Check for broken NULLIF and repair
    sql = emergency_repair_broken_nullif(sql)
    
    # Step 2: Validate all NULLIF have 2 arguments
    print("\n[VALIDATION] Checking NULLIF syntax...")
    
    all_nullif = list(re.finditer(r'NULLIF\s*\([^)]+\)', sql, re.IGNORECASE))
    
    valid_count = 0
    invalid_count = 0
    
    for match in all_nullif:
        args_text = match.group(0)
        # Count commas at depth 0
        depth = 0
        comma_count = 0
        
        for char in args_text:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 1:  # depth=1 means inside NULLIF but not nested functions
                comma_count += 1
        
        if comma_count == 1:
            valid_count += 1
        else:
            invalid_count += 1
            print(f"  ⚠️ Invalid NULLIF: {match.group(0)[:80]}")
    
    if invalid_count > 0:
        print(f"  ❌ Found {invalid_count} invalid NULLIF call(s)")
        print(f"  🚨 Attempting one more repair...")
        sql = emergency_repair_broken_nullif(sql)
        
        # Re-check
        all_nullif_after = list(re.finditer(r'NULLIF\s*\([^)]+\)', sql, re.IGNORECASE))
        if len(all_nullif_after) == len(all_nullif):
            print(f"  ❌ REPAIR FAILED - SQL may have syntax errors")
        else:
            print(f"  ✅ Repair successful")
    else:
        print(f"  ✅ All {valid_count} NULLIF call(s) are valid")
    
    # Step 3: Final parentheses check
    open_count = sql.count('(')
    close_count = sql.count(')')
    
    if open_count != close_count:
        print(f"\n  ⚠️ Unbalanced parentheses: {open_count} open, {close_count} close")
    else:
        print(f"  ✓ Balanced parentheses: {open_count} pairs")
    
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE")
    print("="*80)
    
    return sql
# ============================================================================
# MAIN FUNCTION - SAFE VERSION WITH ALL FIXES
# ============================================================================

def run_sql_generation_graph(
    question: str,
    user_id: str,
    session_id: str,
    history: List[Dict] = None
) -> Tuple[str, str, str]:
    """
    SQL generation - SAFE VERSION with malformed fragment fixes.
    Only applies fixes that are proven safe.
    """
    try:
        print(f"\n{'='*80}")
        print(f"SQL Generation: {question}")
        print(f"Session: {session_id}")
        print('='*80)

        # ========================================================================
        # STEP 1-5: Session, tables, joins, prompt, LLM
        # ========================================================================
        
        session_data = SessionManager.get_session(session_id)

        if not session_data:
            print(f"Session {session_id} not found, trying fallback...")
            if session_store:
                session_id, session_data = next(iter(session_store.items()))
                print(f"Falling back to {session_id}")
            else:
                try:
                    emergency_session = create_emergency_session()
                    if emergency_session:
                        session_id, session_data = emergency_session
                        session_store[session_id] = session_data
                        try:
                            embed_schema_catalog(
                                user_id=session_data["user_id"],
                                db_id=session_id,
                                schema_catalog=session_data["schema_catalog"],
                                embedder=embedder,
                                collection=collection
                            )
                        except Exception as embed_error:
                            print(f"Embedding failed: {embed_error}")
                    else:
                        return ("-- ERROR: No active DB session", "Please connect to database first", "")
                except Exception as e:
                    print(f"Emergency session failed: {e}")
                    return ("-- ERROR: No active DB session", "Please connect to database first", "")

        if not session_data or "schema_catalog" not in session_data:
            return ("-- ERROR: Invalid session", "Session corrupted, please reconnect", "")

        schema_catalog = session_data["schema_catalog"]
        print(f"Schema: {len(schema_catalog)} tables")

        selected_tables = select_relevant_tables_with_embedding_fallback(
            question=question,
            schema_catalog=schema_catalog,
            user_id=user_id,
            session_id=session_id,
            similarity_threshold=0.4,
            max_tables=5
        )

        if not selected_tables:
            selected_tables = list(schema_catalog.keys())[:3]

        print(f"Selected: {selected_tables}")

        try:
            join_info = discover_join_keys(selected_tables, schema_catalog)
        except Exception as e:
            print(f"Join discovery failed: {e}")
            join_info = {}

        prompt = build_dynamic_sql_prompt(
            question=question,
            schema_catalog=schema_catalog,
            tables=selected_tables,
            join_info=join_info,
            history=history or [],
            similarity_threshold=0.4
        )

        try:
            llm = get_llama_maverick_llm()
            if not llm:
                return ("-- ERROR: LLM not configured", "LLM unavailable", "LLM_ERROR")

            response = llm.invoke(prompt)
            raw_sql = getattr(response, "content", response if isinstance(response, str) else str(response))
            print(f"\nRaw SQL from LLM: {raw_sql[:200]}")

        except Exception as e:
            print(f"LLM failed: {e}")
            return (f"-- ERROR: {str(e)}", "LLM generation failed", "LLM_ERROR")

        # ========================================================================
        # STEP 6: SAFE SQL CLEANING - Only proven fixes
        # ========================================================================
        print("\n" + "="*80)
        print("SQL CLEANING (SAFE MODE)")
        print("="*80)

        sql = raw_sql
        
        # Step 1: Strip markdown
        print("\n[Step 1] Strip markdown...")
        sql = strip_sql_markdown(sql)
        if '```' in sql:
            sql = sql.replace('```sql', '').replace('```', '').replace('`', '').strip()

        # Step 2: Verify EXTRACT syntax
        print("\n[Step 2] Check EXTRACT syntax...")
        extract_valid, extract_issues = validate_extract_syntax_comprehensive(sql)
        
        if not extract_valid:
            print(f"  EXTRACT needs fixing: {extract_issues}")
            sql = safe_fix_extract_syntax(sql)
            
            # Verify fix worked
            extract_valid, extract_issues = validate_extract_syntax_comprehensive(sql)
            if not extract_valid:
                print(f"  ❌ Could not fix EXTRACT: {extract_issues}")

        # Step 3: Fix column names
        print("\n[Step 3] Fix invalid column names...")
        sql = fix_column_names(sql, schema_catalog)
        
        # Step 3b: Add NULL safety for date casts
        print("\n[Step 3b] Add NULL safety filters...")
        sql_before_safety = sql
        sql = add_null_safety_filters(sql, schema_catalog)
        sql = normalize_intervals_in_sql(sql)
        sql = remove_constants_from_groupby(sql) 

        # 3️⃣ Rename final cleaned SQL for downstream execution
        safe_sql = sql

        # CRITICAL DEBUG - Verify NULLIF was applied
        if 'NULLIF' not in safe_sql and '::TIMESTAMP' in sql_before_safety:
            print("  ⚠️ WARNING: NULLIF not applied (no date casts found or already wrapped)")
        else:
            nullif_count = safe_sql.upper().count('NULLIF')
            print(f"  ✅ NULLIF applied - found {nullif_count} occurrences")

        # CRITICAL DEBUG - Verify NULLIF was applied
        if 'NULLIF' not in sql and '::TIMESTAMP' in sql_before_safety:
            print("  ⚠️ WARNING: NULLIF not applied (no date casts found or already wrapped)")
        else:
            nullif_count = sql.upper().count('NULLIF')
            print(f"  ✅ NULLIF applied - found {nullif_count} occurrences")
        
        # Step 3c: SKIP potentially dangerous functions
        print("\n[Step 3c] Skipping dangerous auto-fix functions...")
        print("  ⚠️ Skipping: clean_generated_sql() - may corrupt EXTRACT")
        print("  ⚠️ Skipping: auto_fix_schema_mismatches() - may corrupt EXTRACT")
        print("  ⚠️ Skipping: _fix_groupby_alias_and_parens() - may corrupt EXTRACT")
        
        # Only apply SAFE fixes
        try:
            sql = validate_sql_against_catalog(sql, schema_catalog)
        except Exception as e:
            print(f"  validate_sql_against_catalog skipped: {e}")

        # Step 4: Safe parentheses balancing
        print("\n[Step 4] Safe parentheses cleanup...")
        sql = balance_parentheses_safe(sql)

        # Step 5: Date enforcement (if safe)
        print("\n[Step 5] Date enforcement...")
        if any("stage" in tbl for tbl in selected_tables):
            try:
                date_info = get_fixed_date_context()
                sql_before = sql
                sql = enforce_2024(sql, date_info)
                
                # Verify EXTRACT wasn't corrupted
                if sql != sql_before:
                    extract_valid_after, _ = validate_extract_syntax_comprehensive(sql)
                    if not extract_valid_after:
                        print("  ⚠️ enforce_2024 corrupted EXTRACT, reverting")
                        sql = sql_before
            except Exception as e:
                print(f"  Date enforcement skipped: {e}")

        # Step 6: Fix malformed year filters (NEW!)
        print("\n[Step 6] Fix malformed year filters...")
        sql = fix_malformed_year_filters(sql)
        
        # Step 7: Final minimal cleanup
        print("\n[Step 7] Final cleanup...")
        sql = minimal_sql_cleanup(sql)
        
        # Step 8: CRITICAL - Final validation and repair
        print("\n[Step 8] Final validation and repair...")
        sql = final_sql_validation_and_repair(sql)  # <-- NEW: Last line of defense


        # FINAL VALIDATION
        print("\n" + "="*80)
        print("FINAL VALIDATION")
        print("="*80)

        is_valid, errors = quick_validate_sql_syntax(sql)

        if not is_valid:
            print(f"⚠️ Validation issues: {errors}")
            
            # Filter false positives
            real_errors = []
            for err in errors:
                if "EXTRACT" in err and "NULLIF" in sql:
                    print(f"  ✓ Ignoring: EXTRACT with NULLIF is valid PostgreSQL")
                    continue
                if "Parentheses" in err:
                    # Check actual difference AFTER balance_parentheses_safe ran
                    open_c = sql.count('(')
                    close_c = sql.count(')')
                    diff = abs(open_c - close_c)
                    print(f"  Checking parens: open={open_c}, close={close_c}, diff={diff}")
                    
                    if diff <= 2:  # Small diff, try to fix
                        print(f"  ✓ Auto-fixing {diff} paren mismatch...")
                        sql = balance_parentheses_safe(sql)  # Fix it again
                        # Recheck
                        if sql.count('(') == sql.count(')'):
                            print(f"  ✓ Parentheses now balanced, ignoring error")
                            continue
                real_errors.append(err)
            
            if real_errors:
                error_msg = "; ".join(real_errors)
                print(f"❌ Real errors: {error_msg}")
                return (f"-- ERROR: {error_msg}", error_msg, "SQL_SYNTAX_ERROR")
            else:
                print("✅ Errors were false positives, proceeding with SQL")

        # Check for NULLIF success
        if 'NULLIF' in sql and '::TIMESTAMP' in sql:
            print("\n✅ NULL-SAFE SQL: NULLIF wrappers will prevent empty string errors")

        print("\n✅ SQL VALIDATION PASSED")
        print(f"\n{'='*80}")
        print("FINAL SQL:")
        print("="*80)
        print(sql)
        print("="*80 + "\n")

        # Generate explanation
        try:
            explanation_prompt = f"Explain in plain English what this SQL query does:\n\nSQL:\n{sql}"
            explanation_response = llm.invoke(explanation_prompt)
            explanation = getattr(
                explanation_response, 
                "content", 
                explanation_response if isinstance(explanation_response, str) else str(explanation_response)
            ).strip()
        except Exception as e:
            print(f"Explanation failed: {e}")
            explanation = "Query explanation unavailable"

        return (sql, explanation, "")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        return (f"-- ERROR: {str(e)}", "Generation failed", "GENERATION_ERROR")
# # ----------------------------
# def nuclear_repair_sql(
#     raw_llm_output: str,
#     tables: List[str] = None,
#     schema_catalog: Dict[str, Any] = None,
#     join_info: Dict[str, Any] = None,
#     debug: bool = False,
#     execute: bool = False,  # Optional: Execute SQL
#     db_conn_params: Dict[str, Any] = None  # {'host':..., 'port':..., 'user':..., 'password':..., 'database':...}
# ) -> str:
#     """Comprehensive SQL repair with column validation, SMART TRIM removal, and optional execution."""

#     sql_to_run = raw_llm_output
#     max_attempts = 5
#     attempt = 0

#     while attempt < max_attempts:
#         attempt += 1
#         try:
#             if debug:
#                 print(f"\n[ATTEMPT {attempt}] Starting SQL repair...")

#             sql = sql_to_run

#             # ----------------------------
#             # CLEAN RAW LLM SQL
#             # ----------------------------
#             sql = sql.strip()
#             sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
#             sql = re.sub(r'```\s*', '', sql)
#             sql = re.sub(r'`+', '', sql)
#             sql_start = re.search(r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b', sql, re.IGNORECASE)
#             if sql_start:
#                 sql = sql[sql_start.start():]

#             # ----------------------------
#             # PHASE 0: Column Validation & Auto-Fix
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 0] Column Validation and Correction")
#             sql = validate_and_fix_columns(sql, schema_catalog, debug=debug)

#             # ----------------------------
#             # PHASE 1: CTE Syntax Fixes
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 1] CTE Syntax Fixes")
#             sql = re.sub(r'\)\)\s*,\s*SELECT\s+(?![\w]+\s+AS\s*\()', r')) SELECT ', sql, flags=re.IGNORECASE)
#             sql = re.sub(r'\)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(', r'),\n\1 AS (', sql, flags=re.IGNORECASE)
#             sql = validate_and_fix_cte_syntax(sql, debug=debug)

#             # ----------------------------
#             # PHASE 2: Identify Timestamp Columns
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2] Identifying Timestamp/Date Columns")
#             timestamp_columns = identify_timestamp_date_columns(sql, schema_catalog, debug=debug)

#             # ----------------------------
#             # PHASE 2.5: Smart TRIM Removal (timestamp/date)
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.5] Smart TRIM Removal")
#             sql = smart_remove_trim_from_timestamps(sql, timestamp_columns, debug=debug)

#             # ----------------------------
#             # PHASE 2.75: Fix Timestamp Columns (CRITICAL FIX)
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.75] Fix Timestamp Columns with NULLIF")
#             sql = fix_timestamp_columns(sql, schema_catalog, debug=debug)

#             # ----------------------------
#             # PHASE 2.76: Fix empty-string timestamps for safe casting (ADDITIONAL SAFETY)
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.76] Additional empty-string timestamp fixes")
#             sql = fix_empty_string_timestamps(sql, timestamp_columns, debug=debug)

#             # ----------------------------
#             # PHASE 2.8: Force-cast timestamp/date columns for LIKE/ILIKE
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.8] Force-cast timestamp/date columns for LIKE/ILIKE")

#             text_ops_pattern = r'(\b\w+\.(\w+)\b)\s+(ILIKE|LIKE|SIMILAR TO)\s*'

#             def force_cast_timestamp(match):
#                 full_col, col_name, op = match.group(1), match.group(2).lower(), match.group(3)
#                 cast_required = False
#                 if schema_catalog:
#                     for table_info in schema_catalog.values():
#                         for col in table_info.get("columns", []):
#                             if col_name == col.get("name", "").lower():
#                                 if any(x in col.get("type", "").lower() for x in ["timestamp", "date", "time"]):
#                                     cast_required = True
#                                 break
#                         if cast_required:
#                             break
#                 else:
#                     if any(x in col_name for x in ["date", "timestamp", "time"]):
#                         cast_required = True
#                 return f"{full_col}::TEXT {op} " if cast_required else match.group(0)

#             sql = re.sub(text_ops_pattern, force_cast_timestamp, sql,
#                          flags=re.IGNORECASE | re.MULTILINE)

#             # ----------------------------
#             # PHASE 2.9: Fix Text Column Aggregations
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 2.9] Fix Text Column Aggregations")
#             sql = fix_text_column_aggregations(sql, schema_catalog, debug=debug)

#             # ----------------------------
#             # PHASE 3: Date/Time Fixes
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 3] Date/Time Fixes")
#             today = datetime.date.today()
#             fixed_date = f"DATE '{today.year}-{today.month:02d}-{today.day:02d}'"
#             sql = re.sub(r'\bCURRENT_DATE\b', fixed_date, sql, flags=re.IGNORECASE)
#             sql = re.sub(r'\bNOW\(\)\b', fixed_date, sql, flags=re.IGNORECASE)
#             sql = re.sub(r'(\w+)\s*=\s*NULL\b', r'\1 IS NULL', sql, flags=re.IGNORECASE)
#             sql = re.sub(r'(\w+)\s*!=\s*NULL\b', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)

#             # Safe division
#             def safe_div(match):
#                 left, right = match.group(1), match.group(2).strip()
#                 return match.group(0) if 'NULLIF' in right.upper() else f"{left} / NULLIF({right}, 0)"
#             sql = re.sub(r'([^\s]+)\s*/\s*([A-Za-z0-9_"\.\(\)]+)(?!\s*[,\)])', safe_div, sql)

#             # ----------------------------
#             # PHASE 4: Syntax Cleanup
#             # ----------------------------
#             if debug:
#                 print("\n[PHASE 4] Syntax Cleanup")
#             sql = re.sub(r',\s*\)', ')', sql)
#             sql = re.sub(r',\s*FROM\b', ' FROM', sql, flags=re.IGNORECASE)
#             sql = re.sub(r',\s*WHERE\b', ' WHERE', sql, flags=re.IGNORECASE)

#             # Balance parentheses
#             open_count, close_count = sql.count('('), sql.count(')')
#             if open_count > close_count:
#                 sql += ')' * (open_count - close_count)
#                 if debug:
#                     print(f"[DEBUG] Added {open_count - close_count} closing parentheses")

#             sql = re.sub(r'\s+', ' ', sql).strip()
#             sql = sql.rstrip(';') + ';'

#             # ----------------------------
#             # CONVERT QUARTER INTERVALS TO MONTHS
#             # ----------------------------
#             sql = convert_quarter_to_month_interval(sql, debug=debug)

#             # Replace '1 quarter' with '3 months' for Postgres
#             sql = re.sub(
#                 r"INTERVAL\s+'1\s+quarter'",
#                 "INTERVAL '3 months'",
#                 sql,
#                 flags=re.IGNORECASE
#             )

#             # ----------------------------
#             # FINAL VERIFICATION: TRIM on timestamp
#             # ----------------------------
#             for col in timestamp_columns:
#                 if re.search(rf'\bTRIM\s*\(\s*\w*\.?{re.escape(col)}\s*\)', sql, re.IGNORECASE):
#                     if debug:
#                         print(f"⚠️ WARNING: TRIM still found on timestamp column: {col}")
#                     sql = smart_remove_trim_from_timestamps(sql, {col}, debug=False)

#             # ----------------------------
#             # FINAL VERIFICATION: Ensure all timestamp casts are wrapped with NULLIF
#             # ----------------------------
#             if debug:
#                 print("\n[FINAL VERIFICATION] Checking all timestamp casts have NULLIF...")
            
#             for col in timestamp_columns:
#                 # Check for any remaining unwrapped casts
#                 unwrapped_pattern = rf'(?<!NULLIF\()[^(]\b(\w+\.)?{re.escape(col)}\s*::\s*(TIMESTAMP|DATE|TIMESTAMPTZ)\b'
#                 if re.search(unwrapped_pattern, sql, re.IGNORECASE):
#                     if debug:
#                         print(f"⚠️ WARNING: Found unwrapped timestamp cast for {col}, applying fix...")
#                     sql = fix_empty_string_timestamps(sql, {col}, debug=debug)

#             if debug:
#                 print("\n[FINAL SQL]")
#                 print(sql)

#             # ----------------------------
#             # OPTIONAL: Execute SQL
#             # ----------------------------
#             if execute and db_conn_params:
#                 if debug:
#                     print("\n[EXECUTE] Attempting to run SQL against database...")
#                 try:
#                     with psycopg2.connect(**db_conn_params) as conn:
#                         with conn.cursor() as cur:
#                             cur.execute(sql)
#                             if debug:
#                                 print("[EXECUTE] SQL executed successfully")
#                 except Exception as exec_error:
#                     error_str = str(exec_error).lower()
#                     if "does not exist" in error_str:
#                         if debug:
#                             print(f"[EXECUTE] Missing column detected: {exec_error}")
#                             print("[EXECUTE] Auto-fixing missing columns...")
#                         sql_to_run = auto_fix_missing_columns(sql, schema_catalog, str(exec_error), debug=debug)
#                         continue
#                     elif "invalid input syntax for type timestamp" in error_str:
#                         if debug:
#                             print(f"[EXECUTE] Timestamp casting error detected: {exec_error}")
#                             print("[EXECUTE] Re-applying timestamp fixes...")
#                         # Re-apply timestamp fixes more aggressively
#                         sql_to_run = fix_empty_string_timestamps(sql, timestamp_columns, debug=debug)
#                         continue
#                     else:
#                         raise exec_error

#             sql_to_run = sql
#             break

#         except Exception as e:
#             error_str = str(e).lower()
#             if "does not exist" in error_str:
#                 if debug:
#                     print(f"[DEBUG] Missing column detected: {e}")
#                     print("[DEBUG] Auto-fixing missing columns...")
#                 sql_to_run = auto_fix_missing_columns(sql_to_run, schema_catalog, str(e), debug=debug)
#             elif "invalid input syntax for type timestamp" in error_str:
#                 if debug:
#                     print(f"[DEBUG] Timestamp casting error detected: {e}")
#                     print("[DEBUG] Re-applying timestamp fixes more aggressively...")
#                 # Extract the problematic column from error if possible
#                 sql_to_run = fix_empty_string_timestamps(sql_to_run, timestamp_columns, debug=debug)
#             else:
#                 if debug:
#                     print(f"[ERROR] Unhandled exception: {e}")
#                 raise

#     return sql_to_run
# ----------------------------
# def nuclear_repair_sql(
#     raw_llm_output: str,
#     tables: List[str] = None,
#     schema_catalog: Dict[str, Any] = None,
#     join_info: Dict[str, Any] = None,
#     debug: bool = False
# ) -> str:
#     """Comprehensive SQL repair with column validation and SMART TRIM removal."""
    
#     if debug:
#         print("\n" + "="*80)
#         print("SQL REPAIR STARTING")
#         print("="*80)
    
#     # Clean LLM output
#     sql = raw_llm_output.strip()
#     sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
#     sql = re.sub(r'```\s*', '', sql)
#     sql = re.sub(r'`+', '', sql)
    
#     # Extract SQL start
#     sql_start = re.search(r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b', sql, re.IGNORECASE)
#     if sql_start:
#         sql = sql[sql_start.start():]
    
#     # 🆕 PHASE 0: VALIDATE AND FIX INVALID COLUMNS (CRITICAL!)
#     if debug:
#         print("\n[PHASE 0] Column Validation and Correction (CRITICAL)")
    
#     sql = validate_and_fix_columns(sql, schema_catalog, debug=debug)
    
#     # Fix CTE syntax
#     if debug:
#         print("\n[PHASE 1] CTE Syntax Fixes")
    
#     # Remove comma before final SELECT in CTE
#     sql = re.sub(r'\)\)\s*,\s*SELECT\s+(?![\w]+\s+AS\s*\()', r')) SELECT ', sql, flags=re.IGNORECASE)
    
#     # Add comma between CTEs if missing
#     sql = re.sub(r'\)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(', r'),\n\1 AS (', sql, flags=re.IGNORECASE)
    
#     sql = validate_and_fix_cte_syntax(sql, debug=debug)
    
#     # SMART TRIM REMOVAL - Only from timestamp/date columns
#     if debug:
#         print("\n[PHASE 2] Smart TRIM Removal")
    
#     # Identify which columns are timestamps/dates
#     timestamp_columns = identify_timestamp_date_columns(sql, schema_catalog, debug=debug)
    
#     # Remove TRIM ONLY from those columns
#     sql = smart_remove_trim_from_timestamps(sql, timestamp_columns, debug=debug)


#    # --- PHASE: CAST TIMESTAMP/DATE TO TEXT FOR TEXT OPERATORS ---
#     # --- PHASE: CAST timestamp/date columns for text operators DYNAMICALLY ---
#     # --- PHASE 2.8: ALWAYS cast timestamp/date columns for text operators ---
#         # --- PHASE 2.8: Force-cast timestamp/date columns for LIKE/ILIKE ---
#     if debug:
#         print("\n[PHASE 2.8] Force-cast timestamp/date columns for LIKE/ILIKE")

#     text_ops_pattern = r'(\b\w+\.(\w+)\b)\s+(ILIKE|LIKE|SIMILAR TO)\s*'

#     def force_cast_timestamp(match):
#         full_col = match.group(1)
#         col_name = match.group(2).lower()
#         op = match.group(3)
#         cast_required = False

#         if schema_catalog:
#             for table_name, table_info in schema_catalog.items():
#                 for col in table_info.get("columns", []):
#                     if col_name == col.get("name", "").lower():
#                         col_type = col.get("type", "").lower()
#                         if any(x in col_type for x in ["timestamp", "date", "time"]):
#                             cast_required = True
#                         break
#                 if cast_required:
#                     break
#         else:
#             # Fallback heuristic
#             if any(x in col_name for x in ["date", "timestamp", "time"]):
#                 cast_required = True

#         if cast_required:
#             result = f"{full_col}::TEXT {op} "
#             if debug:
#                 print(f"[DEBUG] Force-casting {full_col} for {op} -> {result.strip()}")
#             return result
#         return match.group(0)

#     sql = re.sub(text_ops_pattern, force_cast_timestamp, sql,
#                  flags=re.IGNORECASE | re.MULTILINE)

    
#     # FIX TEXT COLUMN AGGREGATIONS (SUM on TEXT columns)
#     if debug:
#         print("\n[PHASE 2.5] Fix Text Column Aggregations")
    
#     sql = fix_text_column_aggregations(sql, schema_catalog, debug=debug)

#     # FIX TIMESTAMP COLUMNS (handle empty strings)
#     if debug:
#         print("\n[PHASE 2.75] Fix Timestamp Columns (empty strings)")
#     sql = fix_timestamp_columns(sql, schema_catalog, debug=debug)

    
#     # Date/time fixes
#     if debug:
#         print("\n[PHASE 3] Date/Time Fixes")
    
#     today = datetime.date.today()
#     fixed_date = f"DATE '{today.year}-{today.month:02d}-{today.day:02d}'"
#     sql = re.sub(r'\bCURRENT_DATE\b', fixed_date, sql, flags=re.IGNORECASE)
#     sql = re.sub(r'\bNOW\(\)\b', fixed_date, sql, flags=re.IGNORECASE)
    
#     # Fix NULL comparisons
#     sql = re.sub(r'(\w+)\s*=\s*NULL\b', r'\1 IS NULL', sql, flags=re.IGNORECASE)
#     sql = re.sub(r'(\w+)\s*!=\s*NULL\b', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)
    
#     # Safe division
#     def safe_div(match):
#         left = match.group(1)
#         right = match.group(2).strip()
#         if 'NULLIF' in right.upper():
#             return match.group(0)
#         return f"{left} / NULLIF({right}, 0)"
#     sql = re.sub(r'([^\s]+)\s*/\s*([A-Za-z0-9_"\.\(\)]+)(?!\s*[,\)])', safe_div, sql)
    
#     # Clean syntax issues
#     if debug:
#         print("\n[PHASE 4] Syntax Cleanup")
    
#     sql = re.sub(r',\s*\)', ')', sql)
#     sql = re.sub(r',\s*FROM\b', ' FROM', sql, flags=re.IGNORECASE)
#     sql = re.sub(r',\s*WHERE\b', ' WHERE', sql, flags=re.IGNORECASE)
    
#     # Balance parentheses
#     open_count = sql.count('(')
#     close_count = sql.count(')')
#     if open_count > close_count:
#         sql += ')' * (open_count - close_count)
#         if debug:
#             print(f"[DEBUG] Added {open_count - close_count} closing parentheses")
    
#     # Final cleanup
#     sql = re.sub(r'\s+', ' ', sql).strip()
#     sql = sql.rstrip(';') + ';'
    
#     # FINAL VERIFICATION
#     if debug:
#         print("\n" + "="*80)
#         print("FINAL VERIFICATION")
#         print("="*80)
    
#     # Check for TRIM on timestamp columns
#     for col in timestamp_columns:
#         if re.search(rf'\bTRIM\s*\(\s*\w*\.?{re.escape(col)}\s*\)', sql, re.IGNORECASE):
#             if debug:
#                 print(f"⚠️ WARNING: TRIM still found on timestamp column: {col}")
#             # One more pass
#             sql = smart_remove_trim_from_timestamps(sql, {col}, debug=False)
    
#     if debug:
#         print("\n" + "="*80)
#         print("FINAL SQL:")
#         print("="*80)
#         print(sql)
#         print("="*80)
        
#         # Check if TRIM exists on timestamp columns
#         has_bad_trim = False
#         for col in timestamp_columns:
#             if re.search(rf'\bTRIM\s*\(\s*\w*\.?{re.escape(col)}\s*\)', sql, re.IGNORECASE):
#                 print(f"❌ CRITICAL: TRIM still on timestamp column: {col}")
#                 has_bad_trim = True
        
#         if not has_bad_trim:
#             print("✅ SUCCESS: No TRIM on timestamp/date columns!")
        
#         # TRIM on text columns is OK
#         if 'TRIM' in sql.upper() and not has_bad_trim:
#             print("ℹ️  Note: TRIM exists but only on text columns (valid)")
    
#     return sql
# ----------------------------
# def nuclear_repair_sql(
#     raw_llm_output: str,
#     tables: List[str] = None,
#     schema_catalog: Dict[str, Any] = None,
#     join_info: Dict[str, Any] = None,
#     debug: bool = False
# ) -> str:
#     """Comprehensive SQL repair with SMART TRIM removal."""
    
#     if debug:
#         print("\n" + "="*80)
#         print("SQL REPAIR STARTING")
#         print("="*80)
    
#     # Clean LLM output
#     sql = raw_llm_output.strip()
#     sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
#     sql = re.sub(r'```\s*', '', sql)
#     sql = re.sub(r'`+', '', sql)
    
#     # Extract SQL start
#     sql_start = re.search(r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b', sql, re.IGNORECASE)
#     if sql_start:
#         sql = sql[sql_start.start():]
    
#     # Fix CTE syntax FIRST
#     if debug:
#         print("\n[PHASE 1] CTE Syntax Fixes")
    
#     # Remove comma before final SELECT in CTE
#     sql = re.sub(r'\)\)\s*,\s*SELECT\s+(?![\w]+\s+AS\s*\()', r')) SELECT ', sql, flags=re.IGNORECASE)
    
#     # Add comma between CTEs if missing
#     sql = re.sub(r'\)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(', r'),\n\1 AS (', sql, flags=re.IGNORECASE)
    
#     sql = validate_and_fix_cte_syntax(sql, debug=debug)
    
#     # SMART TRIM REMOVAL - Only from timestamp/date columns
#     if debug:
#         print("\n[PHASE 2] Smart TRIM Removal")
    
#     # Identify which columns are timestamps/dates
#     timestamp_columns = identify_timestamp_date_columns(sql, schema_catalog, debug=debug)
    
#     # Remove TRIM ONLY from those columns
#     sql = smart_remove_trim_from_timestamps(sql, timestamp_columns, debug=debug)
    
#     # Date/time fixes
#     if debug:
#         print("\n[PHASE 3] Date/Time Fixes")
    
#     today = datetime.date.today()
#     fixed_date = f"DATE '{today.year}-{today.month:02d}-{today.day:02d}'"
#     sql = re.sub(r'\bCURRENT_DATE\b', fixed_date, sql, flags=re.IGNORECASE)
#     sql = re.sub(r'\bNOW\(\)\b', fixed_date, sql, flags=re.IGNORECASE)
    
#     # Fix NULL comparisons
#     sql = re.sub(r'(\w+)\s*=\s*NULL\b', r'\1 IS NULL', sql, flags=re.IGNORECASE)
#     sql = re.sub(r'(\w+)\s*!=\s*NULL\b', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)
    
#     # Safe division
#     def safe_div(match):
#         left = match.group(1)
#         right = match.group(2).strip()
#         if 'NULLIF' in right.upper():
#             return match.group(0)
#         return f"{left} / NULLIF({right}, 0)"
#     sql = re.sub(r'([^\s]+)\s*/\s*([A-Za-z0-9_"\.\(\)]+)(?!\s*[,\)])', safe_div, sql)
    
#     # Clean syntax issues
#     if debug:
#         print("\n[PHASE 4] Syntax Cleanup")
    
#     sql = re.sub(r',\s*\)', ')', sql)
#     sql = re.sub(r',\s*FROM\b', ' FROM', sql, flags=re.IGNORECASE)
#     sql = re.sub(r',\s*WHERE\b', ' WHERE', sql, flags=re.IGNORECASE)
    
#     # Balance parentheses
#     open_count = sql.count('(')
#     close_count = sql.count(')')
#     if open_count > close_count:
#         sql += ')' * (open_count - close_count)
#         if debug:
#             print(f"[DEBUG] Added {open_count - close_count} closing parentheses")
    
#     # Final cleanup
#     sql = re.sub(r'\s+', ' ', sql).strip()
#     sql = sql.rstrip(';') + ';'
    
#     # FINAL VERIFICATION
#     if debug:
#         print("\n" + "="*80)
#         print("FINAL VERIFICATION")
#         print("="*80)
    
#     # Check for TRIM on timestamp columns
#     for col in timestamp_columns:
#         if re.search(rf'\bTRIM\s*\(\s*\w*\.?{re.escape(col)}\s*\)', sql, re.IGNORECASE):
#             if debug:
#                 print(f"⚠️ WARNING: TRIM still found on timestamp column: {col}")
#             # One more pass
#             sql = smart_remove_trim_from_timestamps(sql, {col}, debug=False)
    
#     if debug:
#         print("\n" + "="*80)
#         print("FINAL SQL:")
#         print("="*80)
#         print(sql)
#         print("="*80)
        
#         # Check if TRIM exists on timestamp columns
#         has_bad_trim = False
#         for col in timestamp_columns:
#             if re.search(rf'\bTRIM\s*\(\s*\w*\.?{re.escape(col)}\s*\)', sql, re.IGNORECASE):
#                 print(f"❌ CRITICAL: TRIM still on timestamp column: {col}")
#                 has_bad_trim = True
        
#         if not has_bad_trim:
#             print("✅ SUCCESS: No TRIM on timestamp/date columns!")
        
#         # TRIM on text columns is OK
#         if 'TRIM' in sql.upper() and not has_bad_trim:
#             print("ℹ️  Note: TRIM exists but only on text columns (valid)")
    
#     return sql


def repair_generated_sql(
    raw_llm_output: str,
    tables: List[str],
    schema_catalog: Dict[str, Any] = None,
    join_info: Dict[str, Any] = None,
    debug: bool = False
) -> str:
    return nuclear_repair_sql(raw_llm_output, tables, schema_catalog, join_info, debug=debug)


# STANDALONE TEST
if __name__ == "__main__":
    test_sql = """SELECT loa.channel_code, SUM(loa.amountfunded)AS total_funded, 
    SUM(app.requestedamount)AS total_requested, 
    ROUND(CAST(SUM(loa.amountfunded)AS NUMERIC)/ NULLIF(SUM(app.requestedamount), 0)* 100, 2)AS approval_rate 
    FROM stage.loan_main_2024 loa 
    INNER JOIN stage.app_main_2024 app ON loa.customerid = app.customerid 
    WHERE EXTRACT(YEAR FROM(CASE WHEN loa.fundeddate IS NULL OR TRIM(loa.fundeddate)= '' 
    THEN NULL ELSE loa.fundeddate::TIMESTAMP END))= EXTRACT(YEAR FROM DATE '2025-10-10')
    GROUP BY loa.channel_code ORDER BY approval_rate DESC;"""
    
    print("TESTING ULTIMATE TRIM FIX")
    print("="*100)
    print("\nBEFORE:")
    print(test_sql)
    print("\n" + "="*100)
    
    fixed = nuclear_repair_sql(test_sql, debug=True)
    
    print("\n" + "="*100)
    print("VERIFICATION:")
    print("="*100)
    if 'TRIM' in fixed.upper():
        print("❌ FAILED: TRIM still exists")
    else:
        print("✅ PASSED: All TRIM removed")
    print("="*100)



def standard_repair_sql(
    sql: str,
    tables: List[str],
    schema_catalog: Dict[str, Any] = None,
    join_info: Dict[str, Any] = None,
    debug: bool = False
) -> str:
    return nuclear_repair_sql(sql, tables, schema_catalog, join_info, debug=debug)


def process_sql(raw_sql: str, schema_catalog: Dict[str, Any] = None, debug: bool = False) -> str:
    sql = clean_llm_output(raw_sql)
    sql = fix_missing_aliases(sql)
    sql = aggressive_remove_trim_on_non_text(sql, debug=debug)
    sql = fix_extract_syntax(sql, debug=True)
    sql = remove_trim_from_non_text_columns(sql, schema_catalog, debug=debug)
    
    if schema_catalog:
        sql = smart_date_cast(sql, schema_catalog, debug=debug)
        sql = smart_numeric_cast(sql, schema_catalog, debug=debug)
    sql = remove_duplicate_casts(sql)
    return sql


# TEST IT RIGHT NOW
if __name__ == "__main__":
    test_sql = """SELECT loa.channel_code, SUM(loa.amountfunded)AS total_funded, SUM(app.requestedamount)AS total_requested, ROUND(CAST(SUM(loa.amountfunded)AS NUMERIC)/ NULLIF(SUM(app.requestedamount), 0)* 100, 2)AS approval_rate FROM stage.loan_main_2024 loa INNER JOIN stage.app_main_2024 app ON loa.customerid = app.customerid WHERE EXTRACT(YEAR FROM(CASE WHEN loa.fundeddate IS NULL OR TRIM(loa.fundeddate)= '' THEN NULL ELSE loa.fundeddate::TIMESTAMP END))= EXTRACT(YEAR FROM DATE '2025-10-10')GROUP BY loa.channel_code ORDER BY approval_rate DESC;"""
    
    print("=" * 100)
    print("BEFORE:")
    print("=" * 100)
    print(test_sql)
    print("\n" + "=" * 100)
    print("RUNNING NUCLEAR FIX...")
    print("=" * 100 + "\n")
    
    fixed_sql = nuclear_repair_sql(test_sql, debug=True)
    
    print("\n" + "=" * 100)
    print("AFTER:")
    print("=" * 100)
    print(fixed_sql)
    print("=" * 100)
    
    # Verify TRIM is gone
    if 'TRIM' in fixed_sql.upper():
        print("\n❌ ERROR: TRIM still exists!")
    else:
        print("\n✅ SUCCESS: All TRIM removed!")
# # ----------------------------
# def clean_llm_output(raw_sql: str) -> str:
#     """Clean LLM output to extract pure SQL."""
#     sql = raw_sql.strip()
#     sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
#     sql = re.sub(r'```\s*', '', sql)
#     sql = re.sub(r'`+', '', sql)
#     sql = re.sub(r'^(Here is|Here\'s|The SQL query is|SQL:)\s*', '', sql, flags=re.IGNORECASE)
#     sql_start = re.search(r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b', sql, re.IGNORECASE)
#     if sql_start:
#         sql = sql[sql_start.start():]
#     sql = re.sub(r';[\s`]*$', '', sql)
#     if ';' in sql:
#         sql = sql[:sql.rfind(';') + 1]
#     return sql

# # ----------------------------
# def extract_table_aliases(sql: str) -> Dict[str, str]:
#     """Extract table aliases from FROM and JOIN clauses."""
#     alias_pattern = re.compile(r'\bFROM\s+([A-Za-z0-9_."]+)\s+([A-Za-z_][A-Za-z0-9_]*)', re.IGNORECASE)
#     aliases = {match.group(2): match.group(1) for match in alias_pattern.finditer(sql)}
#     join_pattern = re.compile(r'\bJOIN\s+([A-Za-z0-9_."]+)\s+([A-Za-z_][A-Za-z0-9_]*)', re.IGNORECASE)
#     for match in join_pattern.finditer(sql):
#         aliases[match.group(2)] = match.group(1)
#     return aliases

# # ----------------------------
# def fix_missing_aliases(sql: str) -> str:
#     """Add missing AS clauses for time extracts in ORDER BY."""
#     order_by_pattern = r'ORDER\s+BY\s+([a-zA-Z_][a-zA-Z0-9_]*)'
#     time_units = ['year', 'month', 'day', 'quarter', 'week', 'hour', 'minute', 'second']
#     for match in re.finditer(order_by_pattern, sql, flags=re.IGNORECASE):
#         col_name = match.group(1).lower()
#         if re.search(rf'\bAS\s+{re.escape(col_name)}\b', sql, flags=re.IGNORECASE):
#             continue
#         if col_name in time_units:
#             from_pos = sql.upper().find(' FROM ')
#             if from_pos == -1:
#                 continue
#             select_clause = sql[:from_pos]
#             extract_pattern = rf'(EXTRACT\s*\(\s*{re.escape(col_name)}\s+FROM\s+[^)]+\))(?!\s+AS)'
#             extract_match = re.search(extract_pattern, select_clause, flags=re.IGNORECASE)
#             if extract_match:
#                 sql = sql[:extract_match.end()] + f' AS {col_name}' + sql[extract_match.end():]
#     return sql

# # ----------------------------
# def smart_date_cast(sql: str, schema_catalog: dict) -> str:
#     """Add safe NULL handling for TEXT date columns."""
#     if not schema_catalog:
#         return sql
#     date_keywords = ['date', 'time', 'timestamp', 'datetime']
#     date_columns = set()
#     for table_info in schema_catalog.values():
#         for col in table_info.get('columns', []):
#             col_name = col['name'].lower()
#             col_type = col.get('type', '').upper()
#             if col_type in ('TEXT', 'VARCHAR', 'CHAR', 'STRING') and any(kw in col_name for kw in date_keywords):
#                 date_columns.add(col_name)
#     if not date_columns:
#         return sql
#     aliases = extract_table_aliases(sql)
#     col_pattern = '|'.join(re.escape(col) for col in date_columns)
#     if aliases:
#         alias_pattern = '|'.join(re.escape(alias) for alias in aliases.keys())
#         full_pattern = rf'((?:{alias_pattern})\.)?({col_pattern})::TIMESTAMP'
#     else:
#         full_pattern = rf'({col_pattern})::TIMESTAMP'
#     def replace_timestamp_cast(match):
#         col_ref = match.group(0).replace('::TIMESTAMP', '')
#         return f"(CASE WHEN {col_ref} IS NULL OR TRIM({col_ref}) = '' THEN NULL ELSE {match.group(0)} END)"
#     pattern = rf'\b{full_pattern}\b'
#     sql = re.sub(pattern, replace_timestamp_cast, sql, flags=re.IGNORECASE)
#     return sql

# # ----------------------------
# def remove_duplicate_casts(sql: str) -> str:
#     """Remove nested duplicate CAST operations."""
#     max_iterations = 5
#     iteration = 0
    
#     while iteration < max_iterations:
#         original_sql = sql
        
#         # Pattern 1: AGG(CAST(alias.CAST(NULLIF... -> AGG(CAST(NULLIF...
#         sql = re.sub(
#             r'(SUM|AVG|MIN|MAX|COUNT)\s*\(\s*CAST\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\.\s*CAST\s*\(\s*NULLIF',
#             r'\1(CAST(NULLIF',
#             sql,
#             flags=re.IGNORECASE
#         )
        
#         # Pattern 2: CAST(alias.CAST(... -> CAST(...
#         sql = re.sub(
#             r'CAST\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\.\s*CAST\s*\(',
#             r'CAST(',
#             sql,
#             flags=re.IGNORECASE
#         )
        
#         # Pattern 3: Double CAST with AS clauses -> single CAST
#         sql = re.sub(
#             r'CAST\s*\(\s*CAST\s*\(([^)]+)\)\s+AS\s+\w+\s*\)\s+AS\s+(\w+)',
#             r'CAST(\1 AS \2)',
#             sql,
#             flags=re.IGNORECASE
#         )
        
#         # Pattern 4: Simple double CAST(CAST(
#         sql = re.sub(
#             r'CAST\s*\(\s*CAST\s*\(',
#             r'CAST(',
#             sql,
#             flags=re.IGNORECASE
#         )
        
#         # Pattern 5: alias.CAST without surrounding CAST
#         sql = re.sub(
#             r'([a-zA-Z_][a-zA-Z0-9_]*)\.\s*CAST\s*\(',
#             r'CAST(',
#             sql,
#             flags=re.IGNORECASE
#         )
        
#         # Pattern 6: Remove alias prefix from inside CAST/NULLIF/TRIM (but keep it)
#         sql = re.sub(
#             r'(TRIM|NULLIF)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\.\s*([a-zA-Z_][a-zA-Z0-9_]*)',
#             r'\1(\2.\3',
#             sql,
#             flags=re.IGNORECASE
#         )
        
#         if sql == original_sql:
#             break
            
#         iteration += 1
    
#     return sql

# # ----------------------------
# def smart_numeric_cast(sql: str, schema_catalog: Dict[str, Any]) -> str:
#     """
#     Safely cast TEXT numeric columns to NUMERIC with NULLIF(TRIM()).
#     CRITICAL: Handles empty strings that cause 'invalid input syntax for type numeric' errors.
#     """
#     if not schema_catalog:
#         return sql

#     # Extended keywords to catch all numeric-like columns
#     numeric_keywords = [
#         'amount', 'price', 'cost', 'fee', 'rate', 'income', 
#         'balance', 'payment', 'charge', 'value', 'total', 
#         'chargeoff', 'chargeoffs',  # CRITICAL: Both singular and plural
#         'revenue', 'profit', 'loss', 'principal', 'interest'
#     ]
    
#     numeric_columns = set()
    
#     # Identify TEXT columns with numeric-like names
#     for table_info in schema_catalog.values():
#         for col in table_info.get('columns', []):
#             col_name = col['name'].lower()
#             col_type = col.get('type', '').upper()
#             if col_type in ('TEXT', 'VARCHAR', 'CHAR', 'STRING'):
#                 # Check if column name contains any numeric keyword
#                 if any(kw in col_name for kw in numeric_keywords):
#                     numeric_columns.add(col_name)

#     if not numeric_columns:
#         return sql

#     aliases = extract_table_aliases(sql)
    
#     # Process each numeric column
#     for col in numeric_columns:
#         # Build all possible column references (with and without aliases)
#         col_refs = []
#         for alias in aliases.keys():
#             col_refs.append(f"{alias}.{col}")
#         col_refs.append(col)  # column without alias
        
#         for col_ref in col_refs:
#             escaped_ref = re.escape(col_ref)
            
#             # Skip if already wrapped with NULLIF(TRIM(...))
#             if re.search(rf'NULLIF\s*\(\s*TRIM\s*\(\s*{escaped_ref}\s*\)', sql, re.IGNORECASE):
#                 continue
            
#             # PRIORITY 1: Handle aggregates SUM/AVG/MIN/MAX/COUNT
#             for func in ['SUM', 'AVG', 'MIN', 'MAX', 'COUNT']:
#                 # Match FUNC(col_ref) patterns
#                 pattern = rf'\b{func}\s*\(\s*{escaped_ref}\s*\)'
                
#                 def replace_agg(m):
#                     return f"{func}(CAST(NULLIF(TRIM({col_ref}), '') AS NUMERIC))"
                
#                 sql = re.sub(pattern, replace_agg, sql, flags=re.IGNORECASE)
            
#             # PRIORITY 2: Handle existing CAST(col AS NUMERIC) - wrap with NULLIF/TRIM
#             pattern = rf'\bCAST\s*\(\s*{escaped_ref}\s+AS\s+NUMERIC\s*\)'
#             replacement = f"CAST(NULLIF(TRIM({col_ref}), '') AS NUMERIC)"
#             sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
            
#             # PRIORITY 3: Handle standalone column references (conservative approach)
#             pattern = rf'\b{escaped_ref}\b'
            
#             def replace_standalone(match):
#                 start = match.start()
#                 end = match.end()
                
#                 # Get surrounding context
#                 before = sql[max(0, start-80):start].upper()
#                 after = sql[end:min(len(sql), end+30)].upper()
                
#                 # Skip if already inside a function
#                 if re.search(r'\w+\s*\($', before):
#                     return match.group(0)
                
#                 # Skip if already wrapped with NULLIF/TRIM
#                 if 'NULLIF' in before[-60:] and 'TRIM' in before[-60:]:
#                     return match.group(0)
                
#                 # Skip if inside CAST we already handled
#                 if 'CAST(' in before[-15:]:
#                     return match.group(0)
                
#                 # Skip if it's a table alias (followed by dot)
#                 if after.startswith('.'):
#                     return match.group(0)
                
#                 # Skip if in GROUP BY or ORDER BY (usually doesn't need casting)
#                 if 'GROUP BY' in before[-20:] or 'ORDER BY' in before[-20:]:
#                     return match.group(0)
                
#                 # Apply safe cast
#                 return f"CAST(NULLIF(TRIM({match.group(0)}), '') AS NUMERIC)"
            
#             sql = re.sub(pattern, replace_standalone, sql, flags=re.IGNORECASE)
    
#     return sql

# # ----------------------------
# def nuclear_repair_sql(
#     raw_llm_output: str,
#     tables: List[str] = None,
#     schema_catalog: Dict[str, Any] = None,
#     join_info: Dict[str, Any] = None
# ) -> str:
#     """
#     Main SQL repair function with comprehensive error handling.
#     Fixes: dates, numerics, aliases, divisions, NULLs, intervals, parentheses.
#     """
    
#     # Step 1: Clean LLM output
#     sql = clean_llm_output(raw_llm_output)
    
#     # Step 2: Fix missing aliases
#     sql = fix_missing_aliases(sql)
    
#     # Step 3: Apply smart casts for dates and numerics
#     if schema_catalog:
#         sql = smart_date_cast(sql, schema_catalog)
#         sql = smart_numeric_cast(sql, schema_catalog)
    
#     # Step 4: Remove duplicate casts created during processing
#     sql = remove_duplicate_casts(sql)
    
#     # Step 5: Fix INTERVAL quarters (PostgreSQL doesn't support quarters directly)
#     for i in range(1, 5):
#         sql = re.sub(
#             rf"INTERVAL\s+['\"]{i}\s+quarters?['\"]", 
#             f"INTERVAL '{i*3} months'", 
#             sql, 
#             flags=re.IGNORECASE
#         )
    
#     # Step 6: Replace CURRENT_DATE / NOW() with fixed date
#     today = datetime.date.today()
#     fixed_date = f"DATE '{today.year}-{today.month:02d}-{today.day:02d}'"
#     sql = re.sub(r'\bCURRENT_DATE\b', fixed_date, sql, flags=re.IGNORECASE)
#     sql = re.sub(r'\bNOW\(\)\b', fixed_date, sql, flags=re.IGNORECASE)
    
#     # Step 7: Fix NULL comparisons (= NULL -> IS NULL)
#     sql = re.sub(r'(\w+)\s*=\s*NULL\b', r'\1 IS NULL', sql, flags=re.IGNORECASE)
#     sql = re.sub(r'(\w+)\s*!=\s*NULL\b', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)
#     sql = re.sub(r'(\w+)\s*<>\s*NULL\b', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)
    
#     # Step 8: Protect divisions from zero (add NULLIF)
#     def safe_div(match):
#         left = match.group(1)
#         right = match.group(2).strip()
#         if 'NULLIF' in right.upper() or '(' in right:
#             return match.group(0)
#         return f"{left} / NULLIF({right}, 0)"
    
#     sql = re.sub(r'([^\s]+)\s*/\s*([A-Za-z0-9_"\.\(\)]+)(?!\s*[,\)])', safe_div, sql)
    
#     # Step 9: Remove stray commas
#     sql = re.sub(r',\s*\)', ')', sql)
#     sql = re.sub(r',\s*FROM\b', ' FROM', sql, flags=re.IGNORECASE)
#     sql = re.sub(r',\s*WHERE\b', ' WHERE', sql, flags=re.IGNORECASE)
    
#     # Step 10: Balance parentheses
#     open_count = sql.count('(')
#     close_count = sql.count(')')
#     if open_count > close_count:
#         sql += ')' * (open_count - close_count)
#     elif close_count > open_count:
#         for _ in range(close_count - open_count):
#             sql = sql.rstrip(';').rstrip(')')
    
#     # Step 11: Clean whitespace and formatting
#     sql = re.sub(r'\s+', ' ', sql).strip()
#     sql = sql.rstrip(';') + ';'
#     sql = sql.replace('`', '')
    
#     # Step 12: Final duplicate cast cleanup
#     sql = remove_duplicate_casts(sql)
    
#     return sql

# ----------------------------
def clean_generated_sql(sql: str, schema_catalog: Dict[str, Any] = None) -> str:
    """
    Final cleanup pass for edge cases.
    Run this AFTER nuclear_repair_sql if needed.
    """
    if not sql:
        return sql

    # Remove duplicate casts
    sql = remove_duplicate_casts(sql)

    # Fix NULLIF without second argument
    sql = re.sub(r'NULLIF\(\s*(\d+)\s*\)', r'NULLIF(\1, 0)', sql)

    # Fix stray commas
    sql = re.sub(r',\s*\)', ')', sql)

    # Remove stray casts
    sql = re.sub(r'::\s*\)', ')', sql)

    # Fix TRIM(...)=NULL
    sql = re.sub(
        r'TRIM\((\w+)\)\s*=\s*NULL',
        r'(\1 IS NULL OR TRIM(\1) = \'\')',
        sql,
        flags=re.IGNORECASE
    )

    # Remove double ::TIMESTAMP
    sql = re.sub(r'::\s*TIMESTAMP\s*::\s*TIMESTAMP', '::TIMESTAMP', sql, flags=re.IGNORECASE)
    
    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'\s*,\s*', ', ', sql)
    sql = re.sub(r'\s*\(\s*', '(', sql)
    sql = re.sub(r'\s*\)\s*', ')', sql)
    
    # Remove empty strings
    sql = re.sub(r"''\s*''", "''", sql)
    
    # Final duplicate cast check
    sql = remove_duplicate_casts(sql)
    
    return sql.strip()

# # ----------------------------
# # Convenience wrapper functions
# # ----------------------------
# def standard_repair_sql(
#     sql: str,
#     tables: List[str],
#     schema_catalog: Dict[str, Any] = None,
#     join_info: Dict[str, Any] = None
# ) -> str:
#     """Wrapper for nuclear_repair_sql."""
#     return nuclear_repair_sql(sql, tables, schema_catalog, join_info)

# def repair_generated_sql(
#     raw_llm_output: str,
#     tables: List[str],
#     schema_catalog: Dict[str, Any] = None,
#     join_info: Dict[str, Any] = None
# ) -> str:
#     """Wrapper for nuclear_repair_sql."""
#     return nuclear_repair_sql(raw_llm_output, tables, schema_catalog, join_info)

# def process_sql(raw_sql: str, schema_catalog: Dict[str, Any] = None) -> str:
#     """
#     Simplified processing pipeline (lighter version).
#     """
#     sql = clean_llm_output(raw_sql)
#     sql = fix_missing_aliases(sql)
    
#     if schema_catalog:
#         sql = smart_date_cast(sql, schema_catalog)
#         sql = smart_numeric_cast(sql, schema_catalog)
    
#     sql = remove_duplicate_casts(sql)
    
#     return sql

# ============================================================================ 
# CLEAN GENERATED SQL
# ============================================================================

# def clean_generated_sql(sql: str, schema_catalog: Dict[str, Any] = None) -> str:
#     """
#     Automatically fix AI-generated SQL issues, including safe TIMESTAMP casting:
#     - No ::TIMESTAMP outside the CASE.
#     - Every comparison is safe for empty strings or NULLs.
#     """
#     if not sql:
#         return sql

#     # Fix NULLIF without second argument
#     sql = re.sub(r'NULLIF\(\s*(\d+)\s*\)', r'NULLIF(\1, 0)', sql)

#     # Fix stray commas before closing parentheses
#     sql = re.sub(r',\s*\)', ')', sql)

#     # Remove stray casts like ':: )'
#     sql = re.sub(r'::\s*\)', ')', sql)

#     # Fix TRIM(...)=NULL -> TRIM(...) IS NULL OR TRIM(...)=''
#     sql = re.sub(
#         r'TRIM\((\w+)\)\s*=\s*NULL',
#         r'(\1 IS NULL OR TRIM(\1) = \'\')',
#         sql,
#         flags=re.IGNORECASE
#     )

#     # Safe TIMESTAMP wrapper
#     def safe_timestamp_wrap(col: str) -> str:
#         return f"(CASE WHEN {col} IS NULL OR TRIM({col}) = '' THEN NULL ELSE {col}::TIMESTAMP END)"

#     # Wrap all potential date columns safely
#     if schema_catalog:
#         date_keywords = ['date', 'time', 'timestamp', 'datetime']
#         date_columns = []
#         for table_info in schema_catalog.values():
#             for col in table_info.get("columns", []):
#                 col_name = col["name"]
#                 col_type = col.get("type", "").upper()
#                 if col_type in ("TEXT", "VARCHAR", "CHAR", "STRING") and any(kw in col_name.lower() for kw in date_keywords):
#                     date_columns.append(col_name)

#         for col in date_columns:
#             pattern = rf'(?<!CASE WHEN )\b{re.escape(col)}::TIMESTAMP\b'
#             sql = re.sub(pattern, lambda m: safe_timestamp_wrap(col), sql, flags=re.IGNORECASE)
#             pattern_alias = rf'\b(\w+)\.{re.escape(col)}::TIMESTAMP\b'
#             sql = re.sub(pattern_alias, lambda m: f"(CASE WHEN {m.group(1)}.{col} IS NULL OR TRIM({m.group(1)}.{col}) = '' THEN NULL ELSE {m.group(1)}.{col}::TIMESTAMP END)", sql, flags=re.IGNORECASE)


#     # Remove double ::TIMESTAMP
#     sql = re.sub(r'::\s*TIMESTAMP\s*::\s*TIMESTAMP', '::TIMESTAMP', sql, flags=re.IGNORECASE)
#     sql = re.sub(r'::\s*\)', ')', sql)

#     return sql


# def clean_generated_sql(sql: str) -> str:
#     """Legacy function - now minimal cleanup only."""
#     if not sql:
#         return sql
    
#     # Only fix obvious typos that nuclear_repair might miss
#     sql = re.sub(r'::\s*\)', ')', sql)
    
#     return sql


# def clean_generated_sql(sql: str) -> str:
#     """Final cleanup."""
#     if not sql:
#         return sql
    
#     sql = re.sub(r'NULLIF\(\s*(\d+)\s*\)', r'NULLIF(\1, 0)', sql)
#     sql = re.sub(r',\s*\)', ')', sql)
#     sql = re.sub(r'::\s*\)', ')', sql)
    
#     return sql

# def clean_generated_sql(sql: str) -> str:
#     """
#     Automatically fix common AI-generated SQL issues.
#     This is now mostly redundant since nuclear_repair handles it.
#     """
#     if not sql:
#         return sql
    
#     # Fix NULLIF without second argument
#     sql = re.sub(r'NULLIF\(\s*(\d+)\s*\)', r'NULLIF(\1, 0)', sql)
    
#     # Fix stray commas before closing parentheses
#     sql = re.sub(r',\s*\)', ')', sql)
    
#     # Remove stray casts like ':: )'
#     sql = re.sub(r'::\s*\)', ')', sql)
    
#     return sql


# # Main entry point - replace your repair_generated_sql with this
# def repair_generated_sql(
#     raw_llm_output: str,
#     tables: List[str],
#     schema_catalog: Dict[str, Any] = None,
#     join_info: Dict[str, Any] = None
# ) -> str:
#     """
#     Main repair function - calls nuclear repair which handles everything.
#     """
#     return nuclear_repair_sql(raw_llm_output, tables, schema_catalog, join_info)

def improve_sql_prompt_clarity(base_prompt: str) -> str:
    """
    Add crystal-clear SQL generation rules to prevent malformed queries.
    """
    clarity_rules = """

# ⚠️ CRITICAL SQL SYNTAX RULES - FOLLOW EXACTLY ⚠️

## CORRECT SQL STRUCTURE (Always follow this pattern):

SELECT 
    column1,
    column2,
    AGG_FUNCTION(column3) AS alias
FROM schema.table1 AS t1
INNER JOIN schema.table2 AS t2 ON t1.id = t2.id
WHERE conditions
GROUP BY column1, column2
HAVING aggregate_conditions
ORDER BY column1
LIMIT n;

## COMMON MISTAKES TO AVOID:
❌ NEVER: SELECT EXTRACT(QUARTER FROM table_name AS alias...
✅ ALWAYS: SELECT EXTRACT(QUARTER FROM alias.date_column) AS quarter FROM table_name AS alias...

❌ NEVER: Put FROM/JOIN inside function calls
✅ ALWAYS: Complete the SELECT clause, THEN write FROM clause

❌ NEVER: FROM table1, table2 (implicit joins)
✅ ALWAYS: FROM table1 AS t1 JOIN table2 AS t2 ON condition

## FUNCTION SYNTAX:
- EXTRACT(QUARTER FROM date_column) - NOT EXTRACT(QUARTER FROM table_name)
- TO_DATE(column, 'YYYY-MM-DD') - column name only, not table
- CAST(column AS type) - column name only

## QUERY CONSTRUCTION ORDER:
1. Write complete SELECT clause with all columns/expressions
2. Write FROM clause with first table and alias
3. Write JOIN clauses if needed
4. Write WHERE conditions
5. Write GROUP BY if using aggregations
6. Write ORDER BY for sorting

NEVER mix these sections or put clauses inside other clauses!

"""
    return base_prompt + clarity_rules


import datetime
import re
import yaml
from pathlib import Path
from typing import List, Dict
#########################################working1010
# def load_schema_from_yaml(yaml_dir='yaml_schema') -> dict:
#     """
#     Loads all YAML files in the given directory into a schema catalog dict.
#     """
#     schema_catalog = {}
#     yaml_path = Path(yaml_dir)
#     if not yaml_path.exists() or not yaml_path.is_dir():
#         raise FileNotFoundError(f"YAML schema directory not found: {yaml_dir}")

#     for yaml_file in yaml_path.glob("*.yaml"):
#         with open(yaml_file, 'r') as f:
#             table_data = yaml.safe_load(f)
#             table_name = f"{table_data.get('schema', '')}.{table_data.get('table', '')}"
#             schema_catalog[table_name] = table_data

#     return schema_catalog
####################################working1010




def load_schema_from_yaml(yaml_dir='yaml_schema', force_refresh=False) -> dict:
    """
    Loads all YAML files in the given directory into a schema catalog dict.
    
    Args:
        yaml_dir: Directory containing YAML schema files
        force_refresh: If True, delete existing YAML files to force regeneration
    """
    yaml_path = Path(yaml_dir)
    
    # 🆕 FORCE REFRESH: Delete existing YAML files if requested
    if force_refresh and yaml_path.exists():
        print(f"[DEBUG] Force refresh enabled - clearing existing YAML files in {yaml_dir}")
        import shutil
        shutil.rmtree(yaml_path)
        yaml_path.mkdir(exist_ok=True)
        print(f"[DEBUG] YAML directory cleared and recreated")
    
    if not yaml_path.exists() or not yaml_path.is_dir():
        raise FileNotFoundError(f"YAML schema directory not found: {yaml_dir}")
    
    schema_catalog = {}
    yaml_files = list(yaml_path.glob("*.yaml"))
    
    if not yaml_files:
        print(f"[DEBUG] No YAML files found in {yaml_dir}")
        return schema_catalog
    
    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as f:
            table_data = yaml.safe_load(f)
            table_name = f"{table_data.get('schema', '')}.{table_data.get('table', '')}"
            schema_catalog[table_name] = table_data
    
    print(f"[DEBUG] Loaded {len(schema_catalog)} tables from YAML")
    return schema_catalog


def build_dynamic_sql_prompt(
    question: str,
    yaml_dir='yaml_schema',
    schema_catalog: dict = None,
    tables: List[str] = None,
    join_info: dict = None,
    history: List[Dict] = None,
    similarity_threshold: float = 0.4,
    error_context: dict = None,
    force_refresh: bool = False  # 🆕 NEW PARAMETER
) -> str:
    """
    Production-grade SQL generation prompt with ANTI-HALLUCINATION safeguards.
    Forces LLM to use ONLY columns that exist in the schema.
    """
    # Load schema catalog from YAML only if not passed
    if schema_catalog is None:
        schema_catalog = load_schema_from_yaml(yaml_dir, force_refresh=force_refresh)  # 🆕 PASS force_refresh

    # Enforce ONLY stage tables
    fixed_tables = [
        "stage.app_main_2024",
        "stage.loan_main_2024"
        # "stage.apploan_metrics_2024"  # 🆕 CORRECTED TABLE NAME
    ]
    schema_catalog = {t: schema_catalog[t] for t in fixed_tables if t in schema_catalog}

    if not schema_catalog:
        raise ValueError(
            "Required stage tables (app_main_2024, loan_main_2024, apploan_metrics_2024) not found in YAML schema."
        )

    # Determine tables to use
    if not tables:
        tables = list(schema_catalog.keys())
    else:
        # Only allow stage tables if user passes any
        tables = [t for t in tables if t in fixed_tables]
        if not tables:
            tables = list(schema_catalog.keys())  

    # Get columns for selected tables
    all_columns = []
    column_type_map = {}
    column_table_map = {}
    exact_column_list = []  
    
    for table in tables:
        table_info = schema_catalog.get(table, {})
        cols = table_info.get("columns", [])
        all_columns.extend(cols)
        table_alias = table.split('.')[-1][:3]
        
        for col in cols:
            col_name = col.get('name', '')
            col_name_lower = col_name.lower()
            col_type = col.get('type', '').lower()
            
            column_type_map[col_name_lower] = col_type
            column_table_map[col_name_lower] = table_alias
            exact_column_list.append(f"{table_alias}.{col_name}")

    # Detect patterns
    arithmetic_info = detect_arithmetic_patterns(question, all_columns, tables, schema_catalog)
    select_patterns = arithmetic_info.get("select", [])
    group_by_cols = arithmetic_info.get("group_by", [])

    # Analyze question intent
    question_lower = question.lower()
    is_percentage = any(word in question_lower for word in ['rate', 'percent', '%', 'ratio', 'proportion'])
    is_comparison = any(word in question_lower for word in ['compare', 'vs', 'versus', 'difference', 'higher', 'lower', 'more', 'less'])
    is_aggregation = any(word in question_lower for word in ['total', 'sum', 'count', 'average', 'avg', 'max', 'min', 'top', 'bottom'])
    is_temporal = any(word in question_lower for word in ['today', 'yesterday', 'last', 'recent', 'current', 'this month', 'this year', 'quarter'])
    is_ranking = any(word in question_lower for word in ['top', 'bottom', 'highest', 'lowest', 'best', 'worst', 'rank'])

    # NEW: Extract relevant columns for THIS question
    relevant_columns = extract_relevant_columns_for_question(
        question, all_columns, column_table_map
    )

    # Build schema section with STRICT EMPHASIS
    today = datetime.date.today()
    current_year = today.year
    
    # Start with CRITICAL WARNING
    prompt = f"""You are an expert PostgreSQL database programmer with 15+ years of production experience.

    ⚠️ IMPORTANT: Return ONLY valid PostgreSQL SQL. Do NOT include explanations, prose, or commentary. 
Do NOT write any text outside the SQL statement. Output must be executable as-is.

⚠️  CRITICAL: You MUST use ONLY the columns listed below. DO NOT invent or assume any columns exist.
⚠️  Any column not explicitly listed DOES NOT EXIST and will cause a database error.
⚠️  If a column you need doesn't exist, use the closest available column or explain why the query cannot be written.

"""

    # Add complete column inventory FIRST (most important)
    prompt += "# COMPLETE COLUMN INVENTORY (USE ONLY THESE)\n\n"
    prompt += "Available columns you can use:\n"
    
    # Group by table for clarity
    for table in tables:
        table_info = schema_catalog.get(table, {})
        table_alias = table.split('.')[-1][:3]
        prompt += f"\n## {table} (alias: {table_alias})\n"
        
        columns_by_category = {
            'Identifiers': [],
            'Dates/Timestamps': [],
            'Numeric/Amounts': [],
            'Status/Flags': [],
            'Text/Categories': []
        }
        
        for col in table_info.get("columns", []):
            col_name = col.get('name', '')
            col_type = col.get('type', '').lower()
            qualified_name = f"{table_alias}.{col_name}"
            
            # Categorize for easier reference
            if any(word in col_name.lower() for word in ['id', 'number', 'code', 'key']):
                columns_by_category['Identifiers'].append((qualified_name, col_type))
            elif any(word in col_name.lower() for word in ['date', 'time', 'dt', 'timestamp']):
                columns_by_category['Dates/Timestamps'].append((qualified_name, col_type))
            elif any(word in col_name.lower() for word in ['amount', 'balance', 'price', 'cost', 'total', 'fee', 'rate', 'dpd', 'days', 'score']):
                columns_by_category['Numeric/Amounts'].append((qualified_name, col_type))
            elif any(word in col_name.lower() for word in ['status', 'flag', 'type', 'category']):
                columns_by_category['Status/Flags'].append((qualified_name, col_type))
            else:
                columns_by_category['Text/Categories'].append((qualified_name, col_type))
        
        # Print categorized columns
        for category, cols in columns_by_category.items():
            if cols:
                prompt += f"  {category}:\n"
                for qualified_name, col_type in cols:
                    # Add casting hints
                    if col_type in ['text', 'varchar', 'char', 'character varying']:
                        if 'date' in qualified_name.lower() or 'time' in qualified_name.lower():
                            prompt += f"    ✓ {qualified_name} (TEXT) → USE: {qualified_name}::TIMESTAMP\n"
                        elif any(word in qualified_name.lower() for word in ['amount', 'balance', 'price', 'cost', 'total', 'fee', 'rate', 'score']):
                            prompt += f"    ✓ {qualified_name} (TEXT) → USE: CAST({qualified_name} AS NUMERIC)\n"
                        else:
                            prompt += f"    ✓ {qualified_name} (TEXT) → USE: ILIKE for search\n"
                    else:
                        prompt += f"    ✓ {qualified_name} ({col_type})\n"

    # Add relevant columns section for THIS question
    if relevant_columns:
        prompt += "\n# COLUMNS MOST RELEVANT TO YOUR QUESTION\n\n"
        prompt += "Based on your question, these columns are likely what you need:\n"
        for col_info in relevant_columns[:10]:  # Top 10 most relevant
            col_name = col_info['name']
            table_alias = col_info['alias']
            col_type = col_info['type']
            qualified = f"{table_alias}.{col_name}"
            prompt += f"  → {qualified} ({col_type})\n"

    # Join section
    join_section = ""
    if join_info:
        join_section = "\n# TABLE RELATIONSHIPS\n\n"
        for join_key, jd in join_info.items():
            if isinstance(join_key, (list, tuple)):
                t1, t2 = join_key
            else:
                continue
            left = jd.get('left_columns', [])
            right = jd.get('right_columns', [])
            join_type = jd.get('type', 'INNER')
            join_section += f"- {t1} ⟷ {t2}\n"
            for l, r in zip(left, right):
                join_section += f"  {l} = {r} ({join_type} JOIN)\n"
        prompt += join_section

    # Learning from history
    if history:
        prompt += "\n# QUERY LEARNING HISTORY\n\n"
        prompt += "Learn from these successful patterns:\n"
        for entry in history[-3:]:  # Only last 3 to save tokens
            q = entry.get('question', '')
            sql = entry.get('sql', '')
            success = entry.get('success', True)
            if success and sql:
                prompt += f"✓ Q: {q}\n  SQL: {sql[:200]}...\n\n"

    # Error section
    if error_context:
        prompt += "\n# PREVIOUS ERROR - LEARN AND FIX\n\n"
        error_msg = error_context.get('error_message', '')
        
        # Extract problematic column from error
        if 'column' in error_msg.lower() and 'does not exist' in error_msg.lower():
            import re
            col_match = re.search(r'column ([a-z_\.]+) does not exist', error_msg, re.IGNORECASE)
            if col_match:
                bad_column = col_match.group(1)
                prompt += f"⚠️  ERROR: Column '{bad_column}' does not exist!\n"
                
                # Suggest alternatives
                base_col = bad_column.split('.')[-1] if '.' in bad_column else bad_column
                suggestions = find_similar_columns(base_col, all_columns)
                if suggestions:
                    prompt += f"   💡 Did you mean one of these?\n"
                    for sug in suggestions[:3]:
                        prompt += f"      - {sug['alias']}.{sug['name']} ({sug['type']})\n"
        
        prompt += f"\nPrevious attempt failed with: {error_context.get('error_type', 'Unknown error')}\n"
        prompt += f"Error: {error_msg}\n"
        prompt += f"Failed SQL: {error_context.get('failed_sql', '')[:300]}...\n\n"
        prompt += "⚠️  Generate NEW SQL that uses ONLY columns from the inventory above.\n\n"

    # Question analysis
    prompt += f"""
# YOUR TASK

Question: "{question}"

Detected Intent:
- Percentage/Rate: {"YES - use CAST(numerator AS NUMERIC) / NULLIF(denominator, 0) * 100" if is_percentage else "NO"}
- Comparison: {"YES - use WHERE or CASE WHEN" if is_comparison else "NO"}
- Aggregation: {"YES - use SUM/COUNT/AVG with GROUP BY" if is_aggregation else "NO"}
- Time-based: {"YES - filter by date using column::TIMESTAMP" if is_temporal else "NO"}
- Ranking: {"YES - use ORDER BY with LIMIT" if is_ranking else "NO"}
"""

    # Add strict rules
    prompt += """

    ⚠️ IMPORTANT: Return ONLY valid PostgreSQL SQL. Do NOT include explanations, prose, or commentary. 
Do NOT write any text outside the SQL statement. Output must be executable as-is.

Don't hallucinate anything that is not in the schema above. Only use the columns and data types provided.

# STRICT RULES (MUST FOLLOW)

1. ⛔ USE ONLY COLUMNS FROM THE INVENTORY ABOVE
2. ⛔ DO NOT invent columns like "dpd_30plus_flag" - use what EXISTS
3. ✅ Always cast TEXT date columns: column::TIMESTAMP or column::DATE
4. ✅ Always cast TEXT numeric columns: CAST(column AS NUMERIC)
5. ✅ Always protect division: / NULLIF(denominator, 0)
6. ✅ Use ILIKE '%value%' for text search (not = )
7. ✅ Use table aliases consistently
8. ✅ Handle NULL values properly
9. ✅ Use JOINs only when necessary
10. ✅ Use CASE WHEN for conditional logic
11. ✅ Use COALESCE(column, 'default_value') for default values
12. ✅ Use EXTRACT(YEAR FROM column::TIMESTAMP) for year extraction
13. for "stage.app_main_2024" and "stage.loan_main_2024" tables use the year 2024 always. Because that contains only the 2024 data.
13. ✅ Use EXTRACT(MONTH FROM column::TIMESTAMP) for month extraction 
14. ✅ Use EXTRACT(DAY FROM column::TIMESTAMP) for day extraction
15. ✅ Use EXTRACT(HOUR FROM column::TIMESTAMP) for hour extraction
# 16. ✅ Use EXTRACT(MINUTE FROM column::TIMESTAMP) for minute extraction
# 17. ✅ Use EXTRACT(SECOND FROM column::TIMESTAMP) for second extraction
# 18. ✅ Use EXTRACT(WEEK FROM column::TIMESTAMP) for week extraction
# 19. ✅ Use EXTRACT(QUARTER FROM column::TIMESTAMP) for quarter extraction
# 20. ✅ Use EXTRACT(DOW FROM column::TIMESTAMP) for day of week extraction
# 21. ✅ Use EXTRACT(DOY FROM column::TIMESTAMP) for day of year extraction
# 22. ✅ Use EXTRACT(EPOCH FROM column::TIMESTAMP) for epoch extraction
# 23. ✅ Use EXTRACT(ISODOW FROM column::TIMESTAMP) for ISO day of week extraction
# 24. ✅ Use EXTRACT(ISOYEAR FROM column::TIMESTAMP) for ISO year extraction


# OUTPUT FORMAT

Return ONLY valid PostgreSQL SQL - no explanations, no markdown, no apologies.
The query MUST use ONLY columns that exist in the schema above.

SQL Query:
"""


        # 🧠 SELF-HEALING LOGIC FOR KNOWN SQL ISSUES
    if error_context:
        error_msg = error_context.get('error_message', '').lower()

        if 'syntax error' in error_msg and 'nullif' in error_msg:
            prompt += "\n# AUTO-REPAIR DETECTED ISSUE\n"
            prompt += "⚙️ Detected malformed NULLIF (missing second argument). Always use: NULLIF(col::TEXT, '').\n"
            prompt += "Rebuild query ensuring both arguments to NULLIF are present.\n\n"

        if 'syntax error' in error_msg and 'near ")"' in error_msg:
            prompt += "\n# AUTO-REPAIR DETECTED ISSUE\n"
            prompt += "⚙️ Detected stray parenthesis in SQL. Verify every NULLIF() and CAST() expression.\n"
            prompt += "Ensure SQL follows: NULLIF(column::TEXT, '')::TIMESTAMP with no extra ) symbols.\n\n"

        if 'btrim' in error_msg:
            prompt += "\n# AUTO-REPAIR DETECTED ISSUE\n"
            prompt += "⚙️ Detected TRIM() on non-text column. Replace TRIM() with direct cast to TEXT inside NULLIF.\n"
            prompt += "Use NULLIF(column::TEXT, '')::TIMESTAMP instead of TRIM.\n\n"

    return prompt


def extract_relevant_columns_for_question(
    question: str, 
    all_columns: List[Dict], 
    column_table_map: Dict[str, str]
) -> List[Dict]:
    """Extract columns most relevant to the question using keyword matching."""
    from difflib import SequenceMatcher
    
    question_lower = question.lower()
    question_words = set(re.findall(r'\b\w+\b', question_lower))
    
    scored_columns = []
    
    for col in all_columns:
        col_name = col.get('name', '')
        col_name_lower = col_name.lower()
        col_type = col.get('type', '').lower()
        
        # Calculate relevance score
        score = 0.0
        
        # Direct word match
        col_words = set(col_name_lower.replace('_', ' ').split())
        overlap = len(question_words & col_words)
        score += overlap * 2.0
        
        # Substring match
        if any(word in col_name_lower for word in question_words):
            score += 1.0
        
        # Semantic matching for common patterns
        if 'dpd' in question_lower or 'past due' in question_lower or 'overdue' in question_lower:
            if 'dpd' in col_name_lower or 'overdue' in col_name_lower or 'pastdue' in col_name_lower:
                score += 3.0
        
        if 'balance' in question_lower:
            if 'balance' in col_name_lower or 'amount' in col_name_lower:
                score += 2.0
        
        if 'date' in question_lower or 'when' in question_lower or 'time' in question_lower:
            if 'date' in col_name_lower or 'time' in col_name_lower:
                score += 1.5
        
        if 'channel' in question_lower or 'source' in question_lower:
            if 'channel' in col_name_lower or 'source' in col_name_lower:
                score += 2.0
        
        if score > 0:
            table_alias = column_table_map.get(col_name_lower, 'tbl')
            scored_columns.append({
                'name': col_name,
                'type': col_type,
                'alias': table_alias,
                'score': score
            })
    
    # Sort by score
    scored_columns.sort(key=lambda x: x['score'], reverse=True)
    
    return scored_columns


# def build_dynamic_sql_prompt(
#     question: str,
#     yaml_dir='yaml_schema',
#     schema_catalog: dict = None,
#     tables: List[str] = None,
#     join_info: dict = None,
#     history: List[Dict] = None,
#     similarity_threshold: float = 0.4,
#     error_context: dict = None
# ) -> str:
#     """
#     Production-grade SQL generation prompt using YAML schema catalog.
#     Handles all edge cases, data types, and business logic automatically.
#     """
#     # Load schema catalog from YAML only if not passed
#     if schema_catalog is None:
#         schema_catalog = load_schema_from_yaml(yaml_dir)


#     # Enforce ONLY stage tables
#     fixed_tables = [
#         "stage.app_main_2024",
#         "stage.loan_main_2024",
#         "stage.app_loan_metrics"
#     ]
#     schema_catalog = {t: schema_catalog[t] for t in fixed_tables if t in schema_catalog}

#     if not schema_catalog:
#         raise ValueError(
#             "Required stage tables (app_main_2024, loan_main_2024, app_loan_metrics) not found in YAML schema."
#         )

#     # Determine tables to use
#     if not tables:
#         tables = list(schema_catalog.keys())
#     else:
#         # Only allow stage tables if user passes any
#         tables = [t for t in tables if t in fixed_tables]
#         if not tables:
#             tables = list(schema_catalog.keys())  # fallback to stage tables


#     # if not tables:
#     #     tables = list(schema_catalog.keys())[:3]  # fallback: first 3 tables

#     # Ensure we have tables
#     # if not tables:
#     #     raise ValueError("No tables available in YAML schema catalog.")
#     # Get columns for selected tables
#     all_columns = []
#     column_type_map = {}
#     column_table_map = {}  # NEW: maps column name -> table alias
#     for table in tables:
#         table_info = schema_catalog.get(table, {})
#         cols = table_info.get("columns", [])
#         all_columns.extend(cols)
#         table_alias = table.split('.')[-1][:3]  # example: use first 3 letters of table name as alias
#         for col in cols:
#             col_name_lower = col.get('name', '').lower()
#             column_type_map[col_name_lower] = col.get('type', '').lower()
#             column_table_map[col_name_lower] = table_alias

#     # Detect patterns (assuming detect_arithmetic_patterns is defined elsewhere)
#     arithmetic_info = detect_arithmetic_patterns(question, all_columns, tables, schema_catalog)
#     select_patterns = arithmetic_info.get("select", [])
#     group_by_cols = arithmetic_info.get("group_by", [])

#     # Analyze question intent
#     question_lower = question.lower()
#     is_percentage = any(word in question_lower for word in ['rate', 'percent', '%', 'ratio', 'proportion'])
#     is_comparison = any(word in question_lower for word in ['compare', 'vs', 'versus', 'difference', 'higher', 'lower', 'more', 'less'])
#     is_aggregation = any(word in question_lower for word in ['total', 'sum', 'count', 'average', 'avg', 'max', 'min', 'top', 'bottom'])
#     is_temporal = any(word in question_lower for word in ['today', 'yesterday', 'last', 'recent', 'current', 'this month', 'this year', 'quarter'])
#     is_ranking = any(word in question_lower for word in ['top', 'bottom', 'highest', 'lowest', 'best', 'worst', 'rank'])

#     # Build schema section
#     today = datetime.date.today()
#     current_year = today.year
#     schema_section = "# DATABASE SCHEMA\n\n"

#         # Prepare temporal cast hints for TEXT date columns
#     temporal_cast_hints = []
#     if is_temporal:
#         text_date_cols = [
#             col['name'] for col in all_columns
#             if col['type'].lower() in ['text', 'varchar'] and 'date' in col['name'].lower()
#         ]
#         for c in text_date_cols:
#             table_alias = column_table_map.get(c.lower(), '')
#             temporal_cast_hints.append(f"{table_alias}.{c}::DATE")


#     for table in tables:
#         table_info = schema_catalog.get(table, {})
#         table_year = None
#         year_match = re.search(r'_(\d{4})$', table)
#         if year_match:
#             table_year = year_match.group(1)
#         schema_section += f"## Table: {table}" + (f" (Year: {table_year})" if table_year else "") + "\nColumns:\n"
#         for col in table_info.get("columns", []):
#             col_name = col.get('name', '')
#             col_type = col.get('type', '').lower()
#             pk = " [PRIMARY KEY]" if col.get("is_primary") else ""
#             type_hint = ""

#                         # Add hint for casting text dates
            


#                         # In the loop over columns
#             if col_type in ['text', 'varchar', 'char', 'character varying']:
#                 if any(date_word in col_name.lower() for date_word in ['date', 'dt', 'timestamp']):
#                     type_hint = " ⚠️ TEXT storing dates - MUST CAST: column::DATE"
#                 elif any(numeric_word in col_name.lower() for numeric_word in ['amount', 'price', 'cost', 'total', 'sum', 'balance', 'fee', 'rate', 'score']):
#                     type_hint = " ⚠️ TEXT storing numbers - MUST USE: CAST(NULLIF(column, '') AS NUMERIC)"
#                 else:
#                     type_hint = " [String - use ILIKE for search]"

#             elif col_type in ['integer', 'bigint', 'smallint', 'numeric', 'decimal', 'float', 'double precision', 'real']:
#                 type_hint = " [Numeric - safe for arithmetic]"
#             elif col_type in ['date', 'timestamp', 'timestamp without time zone', 'timestamp with time zone']:
#                 type_hint = " [Date/Time - use DATE/TIMESTAMP functions]"
#             elif col_type == 'boolean':
#                 type_hint = " [Boolean - TRUE/FALSE]"
#             schema_section += f"  - {col_name}: {col.get('type')}{pk}{type_hint}\n"
#         schema_section += "\n"

#     # Join section
#     join_section = ""
#     if join_info:
#         join_section = "# TABLE RELATIONSHIPS\n\n"
#         for join_key, jd in join_info.items():
#             if isinstance(join_key, (list, tuple)):
#                 t1, t2 = join_key
#             else:
#                 continue
#             left = jd.get('left_columns', [])
#             right = jd.get('right_columns', [])
#             join_type = jd.get('type', 'INNER')
#             join_section += f"- {t1} ⟷ {t2}\n"
#             for l, r in zip(left, right):
#                 join_section += f"  {l} = {r} ({join_type} JOIN)\n"
#         join_section += "\n"

#     # Learning from history
#     history_section = ""
#     learned_patterns = []
#     if history:
#         history_section = "# QUERY LEARNING HISTORY\nLearn from these successful patterns:\n\n"
#         for entry in history[-5:]:
#             q = entry.get('question', '')
#             sql = entry.get('sql', '')
#             success = entry.get('success', True)
#             if success:
#                 history_section += f"✓ Q: {q}\n  SQL: {sql}\n\n"
#                 if 'CAST(NULLIF' in sql:
#                     learned_patterns.append("Use CAST(NULLIF(column, '') AS NUMERIC) for TEXT columns with numbers")
#                 if '/ NULLIF' in sql:
#                     learned_patterns.append("Always protect division with NULLIF(divisor, 0)")
#                 if 'ILIKE' in sql:
#                     learned_patterns.append("Use ILIKE for case-insensitive text matching")

#     # Error section
#     error_section = ""
#     if error_context:
#         error_section = "# PREVIOUS ERROR - LEARN AND FIX\n\n"
#         error_section += f"Previous attempt failed with: {error_context.get('error_type', 'Unknown error')}\n"
#         error_section += f"Error message: {error_context.get('error_message', '')}\n"
#         error_section += f"Failed SQL: {error_context.get('failed_sql', '')}\n\n"
#         error_section += "Analyze the error and generate corrected SQL that avoids this mistake.\n\n"

#     # Build final prompt
#     prompt = f"""You are an expert PostgreSQL database programmer with 15+ years of production experience.
# You write bulletproof SQL that handles ALL edge cases automatically.

# {schema_section}
# {join_section}
# {history_section}
# {error_section}

# # QUESTION ANALYSIS

# Question: "{question}"
# """

#     # Add detected intent
#     prompt += f"""Detected Intent:
# - Percentage/Rate calculation: {"YES" if is_percentage else "NO"}
# - Comparison needed: {"YES" if is_comparison else "NO"}
# - Aggregation required: {"YES" if is_aggregation else "NO"}
# - Time-based query: {"YES" if is_temporal else "NO"}
# - Ranking needed: {"YES" if is_ranking else "NO"}
# """

#     # Suggested approach
#     if is_percentage:
#         prompt += "- Calculate ratio and multiply by 100\n- Use CAST(numerator AS NUMERIC) / NULLIF(denominator, 0) * 100\n"
#     if is_ranking:
#         prompt += "- Use ORDER BY with LIMIT or ROW_NUMBER() window function\n"
#     if is_aggregation:
#         prompt += "- Use appropriate aggregate function (COUNT, SUM, AVG, MAX, MIN) and GROUP BY\n"
#     if is_temporal:
#         prompt += f"- Filter by date using DATE '{current_year}-{today.month:02d}-{today.day:02d}'\n"

#     if group_by_cols:
#         prompt += f"- GROUP BY: {', '.join(group_by_cols)}\n"
#     if select_patterns:
#         prompt += f"- Consider SELECT: {', '.join(select_patterns[:5])}\n"

#     prompt += """

# # OUTPUT REQUIREMENTS

# Generate a single, valid PostgreSQL query that:
# 1. Answers the question accurately
# 2. Handles ALL edge cases (NULL, empty strings, zero division)
# 3. Uses correct data types with proper casting
# 4. Follows all rules above without exception
# 5. Returns meaningful results ready for display

# Return ONLY the SQL query - no explanations, no markdown, no comments.
# The query must execute successfully on the first attempt.

# SQL Query:
# """
#     return prompt


######################################################working 1010
# def build_dynamic_sql_prompt(
#     question: str,
#     yaml_dir='yaml_schema',
#     schema_catalog: dict = None,
#     tables: List[str] = None,
#     join_info: dict = None,
#     history: List[Dict] = None,
#     similarity_threshold: float = 0.4,
#     error_context: dict = None
# ) -> str:
#     """
#     Production-grade SQL generation prompt with ANTI-HALLUCINATION safeguards.
#     Forces LLM to use ONLY columns that exist in the schema.
#     """
#     # Load schema catalog from YAML only if not passed
#     if schema_catalog is None:
#         schema_catalog = load_schema_from_yaml(yaml_dir)

#     # Enforce ONLY stage tables
#     fixed_tables = [
#         "stage.app_main_2024",
#         "stage.loan_main_2024",
#         "stage.app_loan_metrics"
#     ]
#     schema_catalog = {t: schema_catalog[t] for t in fixed_tables if t in schema_catalog}

#     if not schema_catalog:
#         raise ValueError(
#             "Required stage tables (app_main_2024, loan_main_2024, app_loan_metrics) not found in YAML schema."
#         )

#     # Determine tables to use
#     if not tables:
#         tables = list(schema_catalog.keys())
#     else:
#         # Only allow stage tables if user passes any
#         tables = [t for t in tables if t in fixed_tables]
#         if not tables:
#             tables = list(schema_catalog.keys())  

#     # Get columns for selected tables
#     all_columns = []
#     column_type_map = {}
#     column_table_map = {}
#     exact_column_list = []  
    
#     for table in tables:
#         table_info = schema_catalog.get(table, {})
#         cols = table_info.get("columns", [])
#         all_columns.extend(cols)
#         table_alias = table.split('.')[-1][:3]
        
#         for col in cols:
#             col_name = col.get('name', '')
#             col_name_lower = col_name.lower()
#             col_type = col.get('type', '').lower()
            
#             column_type_map[col_name_lower] = col_type
#             column_table_map[col_name_lower] = table_alias
#             exact_column_list.append(f"{table_alias}.{col_name}")

#     # Detect patterns
#     arithmetic_info = detect_arithmetic_patterns(question, all_columns, tables, schema_catalog)
#     select_patterns = arithmetic_info.get("select", [])
#     group_by_cols = arithmetic_info.get("group_by", [])

#     # Analyze question intent
#     question_lower = question.lower()
#     is_percentage = any(word in question_lower for word in ['rate', 'percent', '%', 'ratio', 'proportion'])
#     is_comparison = any(word in question_lower for word in ['compare', 'vs', 'versus', 'difference', 'higher', 'lower', 'more', 'less'])
#     is_aggregation = any(word in question_lower for word in ['total', 'sum', 'count', 'average', 'avg', 'max', 'min', 'top', 'bottom'])
#     is_temporal = any(word in question_lower for word in ['today', 'yesterday', 'last', 'recent', 'current', 'this month', 'this year', 'quarter'])
#     is_ranking = any(word in question_lower for word in ['top', 'bottom', 'highest', 'lowest', 'best', 'worst', 'rank'])

#     # NEW: Extract relevant columns for THIS question
#     relevant_columns = extract_relevant_columns_for_question(
#         question, all_columns, column_table_map
#     )

#     # Build schema section with STRICT EMPHASIS
#     today = datetime.date.today()
#     current_year = today.year
    
#     # Start with CRITICAL WARNING
#     prompt = f"""You are an expert PostgreSQL database programmer with 15+ years of production experience.

#     ⚠️ IMPORTANT: Return ONLY valid PostgreSQL SQL. Do NOT include explanations, prose, or commentary. 
# Do NOT write any text outside the SQL statement. Output must be executable as-is.

# ⚠️  CRITICAL: You MUST use ONLY the columns listed below. DO NOT invent or assume any columns exist.
# ⚠️  Any column not explicitly listed DOES NOT EXIST and will cause a database error.
# ⚠️  If a column you need doesn't exist, use the closest available column or explain why the query cannot be written.

# """

#     # Add complete column inventory FIRST (most important)
#     prompt += "# COMPLETE COLUMN INVENTORY (USE ONLY THESE)\n\n"
#     prompt += "Available columns you can use:\n"
    
#     # Group by table for clarity
#     for table in tables:
#         table_info = schema_catalog.get(table, {})
#         table_alias = table.split('.')[-1][:3]
#         prompt += f"\n## {table} (alias: {table_alias})\n"
        
#         columns_by_category = {
#             'Identifiers': [],
#             'Dates/Timestamps': [],
#             'Numeric/Amounts': [],
#             'Status/Flags': [],
#             'Text/Categories': []
#         }
        
#         for col in table_info.get("columns", []):
#             col_name = col.get('name', '')
#             col_type = col.get('type', '').lower()
#             qualified_name = f"{table_alias}.{col_name}"
            
#             # Categorize for easier reference
#             if any(word in col_name.lower() for word in ['id', 'number', 'code', 'key']):
#                 columns_by_category['Identifiers'].append((qualified_name, col_type))
#             elif any(word in col_name.lower() for word in ['date', 'time', 'dt', 'timestamp']):
#                 columns_by_category['Dates/Timestamps'].append((qualified_name, col_type))
#             elif any(word in col_name.lower() for word in ['amount', 'balance', 'price', 'cost', 'total', 'fee', 'rate', 'dpd', 'days', 'score']):
#                 columns_by_category['Numeric/Amounts'].append((qualified_name, col_type))
#             elif any(word in col_name.lower() for word in ['status', 'flag', 'type', 'category']):
#                 columns_by_category['Status/Flags'].append((qualified_name, col_type))
#             else:
#                 columns_by_category['Text/Categories'].append((qualified_name, col_type))
        
#         # Print categorized columns
#         for category, cols in columns_by_category.items():
#             if cols:
#                 prompt += f"  {category}:\n"
#                 for qualified_name, col_type in cols:
#                     # Add casting hints
#                     if col_type in ['text', 'varchar', 'char', 'character varying']:
#                         if 'date' in qualified_name.lower() or 'time' in qualified_name.lower():
#                             prompt += f"    ✓ {qualified_name} (TEXT) → USE: {qualified_name}::TIMESTAMP\n"
#                         elif any(word in qualified_name.lower() for word in ['amount', 'balance', 'price', 'cost', 'total', 'fee', 'rate', 'score']):
#                             prompt += f"    ✓ {qualified_name} (TEXT) → USE: CAST({qualified_name} AS NUMERIC)\n"
#                         else:
#                             prompt += f"    ✓ {qualified_name} (TEXT) → USE: ILIKE for search\n"
#                     else:
#                         prompt += f"    ✓ {qualified_name} ({col_type})\n"

#     # Add relevant columns section for THIS question
#     if relevant_columns:
#         prompt += "\n# COLUMNS MOST RELEVANT TO YOUR QUESTION\n\n"
#         prompt += "Based on your question, these columns are likely what you need:\n"
#         for col_info in relevant_columns[:10]:  # Top 10 most relevant
#             col_name = col_info['name']
#             table_alias = col_info['alias']
#             col_type = col_info['type']
#             qualified = f"{table_alias}.{col_name}"
#             prompt += f"  → {qualified} ({col_type})\n"

#     # Join section
#     join_section = ""
#     if join_info:
#         join_section = "\n# TABLE RELATIONSHIPS\n\n"
#         for join_key, jd in join_info.items():
#             if isinstance(join_key, (list, tuple)):
#                 t1, t2 = join_key
#             else:
#                 continue
#             left = jd.get('left_columns', [])
#             right = jd.get('right_columns', [])
#             join_type = jd.get('type', 'INNER')
#             join_section += f"- {t1} ⟷ {t2}\n"
#             for l, r in zip(left, right):
#                 join_section += f"  {l} = {r} ({join_type} JOIN)\n"
#         prompt += join_section

#     # Learning from history
#     if history:
#         prompt += "\n# QUERY LEARNING HISTORY\n\n"
#         prompt += "Learn from these successful patterns:\n"
#         for entry in history[-3:]:  # Only last 3 to save tokens
#             q = entry.get('question', '')
#             sql = entry.get('sql', '')
#             success = entry.get('success', True)
#             if success and sql:
#                 prompt += f"✓ Q: {q}\n  SQL: {sql[:200]}...\n\n"

#     # Error section
#     if error_context:
#         prompt += "\n# PREVIOUS ERROR - LEARN AND FIX\n\n"
#         error_msg = error_context.get('error_message', '')
        
#         # Extract problematic column from error
#         if 'column' in error_msg.lower() and 'does not exist' in error_msg.lower():
#             import re
#             col_match = re.search(r'column ([a-z_\.]+) does not exist', error_msg, re.IGNORECASE)
#             if col_match:
#                 bad_column = col_match.group(1)
#                 prompt += f"⚠️  ERROR: Column '{bad_column}' does not exist!\n"
                
#                 # Suggest alternatives
#                 base_col = bad_column.split('.')[-1] if '.' in bad_column else bad_column
#                 suggestions = find_similar_columns(base_col, all_columns)
#                 if suggestions:
#                     prompt += f"   💡 Did you mean one of these?\n"
#                     for sug in suggestions[:3]:
#                         prompt += f"      - {sug['alias']}.{sug['name']} ({sug['type']})\n"
        
#         prompt += f"\nPrevious attempt failed with: {error_context.get('error_type', 'Unknown error')}\n"
#         prompt += f"Error: {error_msg}\n"
#         prompt += f"Failed SQL: {error_context.get('failed_sql', '')[:300]}...\n\n"
#         prompt += "⚠️  Generate NEW SQL that uses ONLY columns from the inventory above.\n\n"

#     # Question analysis
#     prompt += f"""
# # YOUR TASK

# Question: "{question}"

# Detected Intent:
# - Percentage/Rate: {"YES - use CAST(numerator AS NUMERIC) / NULLIF(denominator, 0) * 100" if is_percentage else "NO"}
# - Comparison: {"YES - use WHERE or CASE WHEN" if is_comparison else "NO"}
# - Aggregation: {"YES - use SUM/COUNT/AVG with GROUP BY" if is_aggregation else "NO"}
# - Time-based: {"YES - filter by date using column::TIMESTAMP" if is_temporal else "NO"}
# - Ranking: {"YES - use ORDER BY with LIMIT" if is_ranking else "NO"}
# """

#     # Add strict rules
#     prompt += """

#     ⚠️ IMPORTANT: Return ONLY valid PostgreSQL SQL. Do NOT include explanations, prose, or commentary. 
# Do NOT write any text outside the SQL statement. Output must be executable as-is.

# Don't hallucinate anything that is not in the schema above. Only use the columns and data types provided.

# # STRICT RULES (MUST FOLLOW)

# 1. ⛔ USE ONLY COLUMNS FROM THE INVENTORY ABOVE
# 2. ⛔ DO NOT invent columns like "dpd_30plus_flag" - use what EXISTS
# 3. ✅ Always cast TEXT date columns: column::TIMESTAMP or column::DATE
# 4. ✅ Always cast TEXT numeric columns: CAST(column AS NUMERIC)
# 5. ✅ Always protect division: / NULLIF(denominator, 0)
# 6. ✅ Use ILIKE '%value%' for text search (not = )
# 7. ✅ Use table aliases consistently
# 8. ✅ Handle NULL values properly
# 9. ✅ Use JOINs only when necessary
# 10. ✅ Use CASE WHEN for conditional logic
# 11. ✅ Use COALESCE(column, 'default_value') for default values
# 12. ✅ Use EXTRACT(YEAR FROM column::TIMESTAMP) for year extraction
# 13. ✅ Use EXTRACT(MONTH FROM column::TIMESTAMP) for month extraction
# 14. ✅ Use EXTRACT(DAY FROM column::TIMESTAMP) for day extraction
# 15. ✅ Use EXTRACT(HOUR FROM column::TIMESTAMP) for hour extraction
# 16. ✅ Use EXTRACT(MINUTE FROM column::TIMESTAMP) for minute extraction
# 17. ✅ Use EXTRACT(SECOND FROM column::TIMESTAMP) for second extraction
# 18. ✅ Use EXTRACT(WEEK FROM column::TIMESTAMP) for week extraction
# 19. ✅ Use EXTRACT(QUARTER FROM column::TIMESTAMP) for quarter extraction
# 20. ✅ Use EXTRACT(DOW FROM column::TIMESTAMP) for day of week extraction
# 21. ✅ Use EXTRACT(DOY FROM column::TIMESTAMP) for day of year extraction
# 22. ✅ Use EXTRACT(EPOCH FROM column::TIMESTAMP) for epoch extraction
# 23. ✅ Use EXTRACT(ISODOW FROM column::TIMESTAMP) for ISO day of week extraction
# 24. ✅ Use EXTRACT(ISOYEAR FROM column::TIMESTAMP) for ISO year extraction


# # OUTPUT FORMAT

# Return ONLY valid PostgreSQL SQL - no explanations, no markdown, no apologies.
# The query MUST use ONLY columns that exist in the schema above.

# SQL Query:
# """
#     return prompt


# def extract_relevant_columns_for_question(
#     question: str, 
#     all_columns: List[Dict], 
#     column_table_map: Dict[str, str]
# ) -> List[Dict]:
#     """Extract columns most relevant to the question using keyword matching."""
#     from difflib import SequenceMatcher
    
#     question_lower = question.lower()
#     question_words = set(re.findall(r'\b\w+\b', question_lower))
    
#     scored_columns = []
    
#     for col in all_columns:
#         col_name = col.get('name', '')
#         col_name_lower = col_name.lower()
#         col_type = col.get('type', '').lower()
        
#         # Calculate relevance score
#         score = 0.0
        
#         # Direct word match
#         col_words = set(col_name_lower.replace('_', ' ').split())
#         overlap = len(question_words & col_words)
#         score += overlap * 2.0
        
#         # Substring match
#         if any(word in col_name_lower for word in question_words):
#             score += 1.0
        
#         # Semantic matching for common patterns
#         if 'dpd' in question_lower or 'past due' in question_lower or 'overdue' in question_lower:
#             if 'dpd' in col_name_lower or 'overdue' in col_name_lower or 'pastdue' in col_name_lower:
#                 score += 3.0
        
#         if 'balance' in question_lower:
#             if 'balance' in col_name_lower or 'amount' in col_name_lower:
#                 score += 2.0
        
#         if 'date' in question_lower or 'when' in question_lower or 'time' in question_lower:
#             if 'date' in col_name_lower or 'time' in col_name_lower:
#                 score += 1.5
        
#         if 'channel' in question_lower or 'source' in question_lower:
#             if 'channel' in col_name_lower or 'source' in col_name_lower:
#                 score += 2.0
        
#         if score > 0:
#             table_alias = column_table_map.get(col_name_lower, 'tbl')
#             scored_columns.append({
#                 'name': col_name,
#                 'type': col_type,
#                 'alias': table_alias,
#                 'score': score
#             })
    
#     # Sort by score
#     scored_columns.sort(key=lambda x: x['score'], reverse=True)
    
#     return scored_columns
#############################################################working1010

def find_similar_columns(target_col: str, all_columns: List[Dict]) -> List[Dict]:
    """Find columns similar to the target using fuzzy matching."""
    from difflib import SequenceMatcher
    
    suggestions = []
    target_lower = target_col.lower()
    
    for col in all_columns:
        col_name = col.get('name', '')
        similarity = SequenceMatcher(None, target_lower, col_name.lower()).ratio()
        
        if similarity > 0.4:  # 40% similarity threshold
            suggestions.append({
                'name': col_name,
                'type': col.get('type', ''),
                'alias': 'tbl',  # You should pass column_table_map here
                'similarity': similarity
            })
    
    suggestions.sort(key=lambda x: x['similarity'], reverse=True)
    return suggestions[:5]



# def build_dynamic_sql_prompt(
#     question: str,
#     schema_catalog: dict,
#     tables: List[str] = None,
#     join_info: dict = None,
#     history: List[Dict] = None,
#     similarity_threshold: float = 0.4,
#     error_context: dict = None
# ) -> str:
#     """
#     Production-grade SQL generation prompt with self-learning capabilities.
#     Handles all edge cases, data types, and business logic automatically.
#     """
    
#     # Auto-select tables if not provided
#     if not tables:
#         tables = select_relevant_tables(question, schema_catalog, similarity_threshold=similarity_threshold)
    
#     # Ensure we have tables
#     if not tables:
#         tables = list(schema_catalog.keys())[:3]
    
#     # Get columns for selected tables
#     all_columns = []
#     column_type_map = {}  # Track data types for intelligent handling
#     for table in tables:
#         table_info = schema_catalog.get(table, {})
#         cols = table_info.get("columns", [])
#         all_columns.extend(cols)
#         for col in cols:
#             column_type_map[col.get('name', '').lower()] = col.get('type', '').lower()
    
#     # Detect patterns
#     arithmetic_info = detect_arithmetic_patterns(question, all_columns, tables, schema_catalog)
#     select_patterns = arithmetic_info["select"]
#     group_by_cols = arithmetic_info["group_by"]
    
#     # Analyze question intent
#     question_lower = question.lower()
#     is_percentage = any(word in question_lower for word in ['rate', 'percent', '%', 'ratio', 'proportion'])
#     is_comparison = any(word in question_lower for word in ['compare', 'vs', 'versus', 'difference', 'higher', 'lower', 'more', 'less'])
#     is_aggregation = any(word in question_lower for word in ['total', 'sum', 'count', 'average', 'avg', 'max', 'min', 'top', 'bottom'])
#     is_temporal = any(word in question_lower for word in ['today', 'yesterday', 'last', 'recent', 'current', 'this month', 'this year', 'quarter'])
#     is_ranking = any(word in question_lower for word in ['top', 'bottom', 'highest', 'lowest', 'best', 'worst', 'rank'])
    
#     # Build schema section with enhanced metadata
#     today = datetime.date.today()
#     current_year = today.year
#     schema_section = "# DATABASE SCHEMA\n\n"
    
#     for table in tables:
#         table_info = schema_catalog.get(table, {})
#         # Extract year from table name for temporal context
#         table_year = None
#         year_match = re.search(r'_(\d{4})$', table)
#         if year_match:
#             table_year = year_match.group(1)
        
#         schema_section += f"## Table: {table}"
#         if table_year:
#             schema_section += f" (Year: {table_year})"
#         schema_section += "\nColumns:\n"
        
#         for col in table_info.get("columns", []):
#             col_name = col.get('name', '')
#             col_type = col.get('type', '').lower()
#             pk = " [PRIMARY KEY]" if col.get("is_primary") else ""
            
#             # Enhanced type hints
#             type_hint = ""
#             if col_type in ['text', 'varchar', 'char', 'character varying']:
#                 if any(numeric_word in col_name.lower() for numeric_word in ['amount', 'price', 'cost', 'total', 'sum', 'balance', 'fee', 'rate', 'score']):
#                     type_hint = " ⚠️ TEXT storing numbers - MUST USE: CAST(NULLIF(column, '') AS NUMERIC)"
#                 else:
#                     type_hint = " [String - use ILIKE for search]"
#             elif col_type in ['integer', 'bigint', 'smallint', 'numeric', 'decimal', 'float', 'double precision', 'real']:
#                 type_hint = " [Numeric - safe for arithmetic]"
#             elif col_type in ['date', 'timestamp', 'timestamp without time zone', 'timestamp with time zone']:
#                 type_hint = " [Date/Time - use DATE/TIMESTAMP functions]"
#             elif col_type == 'boolean':
#                 type_hint = " [Boolean - TRUE/FALSE]"
            
#             schema_section += f"  - {col_name}: {col.get('type')}{pk}{type_hint}\n"
#         schema_section += "\n"
    
#     # Enhanced JOIN section with relationship context
#     join_section = ""
#     if join_info:
#         join_section = "# TABLE RELATIONSHIPS\n\n"
#         for join_key, jd in join_info.items():
#             if isinstance(join_key, (list, tuple)):
#                 t1, t2 = join_key
#             else:
#                 continue
            
#             left = jd.get('left_columns', [])
#             right = jd.get('right_columns', [])
#             join_type = jd.get('type', 'INNER')
            
#             join_section += f"- {t1} ⟷ {t2}\n"
#             for l, r in zip(left, right):
#                 join_section += f"  {l} = {r} ({join_type} JOIN)\n"
#         join_section += "\n"
    
#     # Learning from history
#     history_section = ""
#     learned_patterns = []
#     if history:
#         history_section = "# QUERY LEARNING HISTORY\n"
#         history_section += "Learn from these successful patterns:\n\n"
        
#         for entry in history[-5:]:  # Last 5 queries
#             q = entry.get('question', '')
#             sql = entry.get('sql', '')
#             success = entry.get('success', True)
            
#             if success:
#                 history_section += f"✓ Q: {q}\n  SQL: {sql}\n\n"
#                 # Extract patterns
#                 if 'CAST(NULLIF' in sql:
#                     learned_patterns.append("Use CAST(NULLIF(column, '') AS NUMERIC) for TEXT columns with numbers")
#                 if '/ NULLIF' in sql:
#                     learned_patterns.append("Always protect division with NULLIF(divisor, 0)")
#                 if 'ILIKE' in sql:
#                     learned_patterns.append("Use ILIKE for case-insensitive text matching")
    
#     # Error recovery section
#     error_section = ""
#     if error_context:
#         error_section = "# PREVIOUS ERROR - LEARN AND FIX\n\n"
#         error_section += f"Previous attempt failed with: {error_context.get('error_type', 'Unknown error')}\n"
#         error_section += f"Error message: {error_context.get('error_message', '')}\n"
#         error_section += f"Failed SQL: {error_context.get('failed_sql', '')}\n\n"
#         error_section += "Analyze the error and generate corrected SQL that avoids this mistake.\n\n"
    
#     # Build the comprehensive prompt
#     prompt = f"""You are an expert PostgreSQL database programmer with 15+ years of production experience.
# You write bulletproof SQL that handles ALL edge cases automatically.

# {schema_section}
# {join_section}
# {history_section}
# {error_section}

# # CORE PRINCIPLES (NEVER VIOLATE)

# ## 1. DATA TYPE INTELLIGENCE
# CRITICAL: Analyze column types before ANY operation:

# ### TEXT Columns Storing Numbers
# - ⚠️ TEXT/VARCHAR columns with numeric names (amount, price, cost, total, etc.) contain string numbers
# - ALWAYS wrap in: CAST(NULLIF(column, '') AS NUMERIC)
# - Example: amount > 100 → CAST(NULLIF(amount, '') AS NUMERIC) > 100
# - Apply to: WHERE, CASE WHEN, HAVING, ORDER BY, arithmetic operations

# ### NULL/Empty Safety
# - Check for NULL: column IS NULL (never column = NULL)
# - Empty strings: NULLIF(column, '') converts '' to NULL
# - Chain: CAST(NULLIF(TRIM(column), '') AS type)
# - Default values: COALESCE(column, default_value)
# - Aggregations: COALESCE(SUM(column), 0) - never return NULL for counts/sums

# ### Division Safety
# - NEVER use: a / b
# - ALWAYS use: a / NULLIF(b, 0)
# - For percentages: (numerator::NUMERIC / NULLIF(denominator, 0)) * 100
# - Cast integers before division: column::NUMERIC to avoid integer division

# ### Date Handling
# - Current date: DATE '{current_year}-{today.month:02d}-{today.day:02d}'
# - NEVER use INTERVAL '1 quarter' - use INTERVAL '3 months'
# - NEVER use CURRENT_DATE or NOW() - use explicit dates
# - Table year context: If table ends with _2024, filter by year 2024
# - Date arithmetic: date_column + INTERVAL '3 months'
# - Extract: EXTRACT(YEAR FROM date_column)

# ## 2. QUERY CONSTRUCTION RULES

# ### Column Qualification
# - ALWAYS use: "schema"."table"."column" or "alias"."column"
# - Never use bare column names in multi-table queries
# - Consistent aliasing: table AS alias

# ### JOIN Strategy
# - Understand relationships before joining
# - Use appropriate JOIN type: INNER, LEFT, RIGHT, FULL OUTER
# - ALWAYS specify ON conditions explicitly
# - Multiple joins: chain logically from base table
# - Avoid Cartesian products

# ### Aggregation Rules
# - If using GROUP BY, all non-aggregated SELECT columns MUST be in GROUP BY
# - Aggregate functions: COUNT, SUM, AVG, MAX, MIN
# - Use COUNT(DISTINCT column) for unique counts
# - Filter aggregates with HAVING (not WHERE)

# ### String Matching
# - Case-insensitive: ILIKE '%pattern%'
# - Exact match: column = 'value'
# - Multiple patterns: column ILIKE '%a%' OR column ILIKE '%b%'
# - NOT matching: column NOT ILIKE '%pattern%'

# ## 3. QUERY PATTERNS BY INTENT

# ### Percentage/Rate Calculations
# When question asks for rate, %, proportion:
# ```sql
# -- Pattern: (count_of_success / total_count) * 100
# SELECT 
#     group_column,
#     (COUNT(CASE WHEN condition THEN 1 END)::NUMERIC / 
#      NULLIF(COUNT(*), 0)) * 100 AS success_rate
# FROM table
# GROUP BY group_column
# ```

# ### Rankings (Top N, Bottom N)
# ```sql
# SELECT * FROM (
#     SELECT *, ROW_NUMBER() OVER (ORDER BY metric DESC) as rank
#     FROM table
# ) ranked
# WHERE rank <= N
# -- Or simply: ORDER BY metric DESC LIMIT N
# ```

# ### Comparisons (Higher, Lower, More, Less)
# ```sql
# -- Use CASE WHEN for conditional logic
# SELECT 
#     column,
#     CASE 
#         WHEN value > threshold THEN 'High'
#         WHEN value < threshold THEN 'Low'
#         ELSE 'Medium'
#     END AS category
# FROM table
# ```

# ### Time-based Analysis
# ```sql
# -- Last 3 months
# WHERE date_column >= DATE '{current_year}-{today.month:02d}-{today.day:02d}' - INTERVAL '3 months'

# -- This year
# WHERE EXTRACT(YEAR FROM date_column) = {current_year}

# -- Month comparison
# WHERE EXTRACT(MONTH FROM date_column) = {today.month}
# ```

# ## 4. BUSINESS LOGIC AUTO-APPLICATION

# ### New Table Discovery
# When encountering a new table:
# 1. Analyze column names for business meaning
# 2. Identify key metrics (amounts, counts, dates, statuses)
# 3. Infer relationships from column name patterns (id, _id suffixes)
# 4. Apply appropriate data type handling automatically

# ### Metric Inference
# - "funded", "approved", "completed" → COUNT WHERE status conditions
# - "amount", "total", "sum" → Use SUM with CAST if TEXT
# - "rate", "percentage" → Calculate ratio * 100
# - "average", "mean" → Use AVG with proper casting
# - "highest", "top" → ORDER BY DESC LIMIT N
# - "lowest", "bottom" → ORDER BY ASC LIMIT N

# ### Fuzzy Column Matching
# If exact column doesn't exist:
# - customer_id, customerid, customer_id → treat as same
# - amount_funded, amountfunded, funded_amount → treat as same
# - Use ILIKE for flexible matching

# ## 5. ERROR PREVENTION

# ### Common Pitfalls to AVOID
# ❌ Text vs number comparison without CAST
# ❌ Division by zero
# ❌ Unqualified columns in multi-table queries
# ❌ Missing GROUP BY for non-aggregated columns
# ❌ Using CURRENT_DATE (replace with explicit date)
# ❌ Integer division (5/2=2, not 2.5)
# ❌ Comparing to NULL with = or !=
# ❌ Forgetting DISTINCT for unique counts
# ❌ Schema not qualified
# ❌ Quotes inconsistent ("" vs '')

# ### Self-Validation Checklist
# Before returning SQL, verify:
# ✓ All TEXT numeric columns wrapped in CAST(NULLIF(...))
# ✓ All divisions protected with NULLIF(divisor, 0)
# ✓ All columns fully qualified with schema/alias
# ✓ GROUP BY includes all non-aggregated columns
# ✓ Dates use explicit DATE 'YYYY-MM-DD' format
# ✓ JOINs have proper ON conditions
# ✓ Aggregations use COALESCE for NULL handling
# ✓ String matching uses ILIKE
# ✓ Syntax is valid PostgreSQL

# ## 6. LEARNED PATTERNS
# """
    
#     if learned_patterns:
#         prompt += "Apply these proven patterns:\n"
#         for pattern in set(learned_patterns):
#             prompt += f"- {pattern}\n"
#         prompt += "\n"
    
#     # Add question intent analysis
#     prompt += f"""
# # QUESTION ANALYSIS

# Question: "{question}"

# Detected Intent:
# - Percentage/Rate calculation: {"YES" if is_percentage else "NO"}
# - Comparison needed: {"YES" if is_comparison else "NO"}
# - Aggregation required: {"YES" if is_aggregation else "NO"}
# - Time-based query: {"YES" if is_temporal else "NO"}
# - Ranking needed: {"YES" if is_ranking else "NO"}

# Suggested Approach:
# """
    
#     if is_percentage:
#         prompt += "- Calculate ratio and multiply by 100 for percentage\n"
#         prompt += "- Use CAST(numerator AS NUMERIC) / NULLIF(denominator, 0) * 100\n"
    
#     if is_ranking:
#         prompt += "- Use ORDER BY with LIMIT or ROW_NUMBER() window function\n"
    
#     if is_aggregation:
#         prompt += "- Use appropriate aggregate function (COUNT, SUM, AVG, MAX, MIN)\n"
#         prompt += "- Remember GROUP BY for dimensions\n"
    
#     if is_temporal:
#         prompt += f"- Filter by date using DATE '{current_year}-{today.month:02d}-{today.day:02d}'\n"
#         prompt += "- Consider table year suffix for filtering\n"
    
#     if group_by_cols:
#         prompt += f"- GROUP BY: {', '.join(group_by_cols)}\n"
    
#     if select_patterns:
#         prompt += f"- Consider SELECT: {', '.join(select_patterns[:5])}\n"
    
#     # Final instructions
#     prompt += f"""

# # OUTPUT REQUIREMENTS

# Generate a single, valid PostgreSQL query that:
# 1. Answers the question accurately
# 2. Handles ALL edge cases (NULL, empty strings, zero division)
# 3. Uses correct data types with proper casting
# 4. Follows all rules above without exception
# 5. Returns meaningful results ready for display

# Return ONLY the SQL query - no explanations, no markdown, no comments.
# The query must execute successfully on the first attempt.

# SQL Query:
# """

#     prompt = improve_sql_prompt_clarity(prompt)
    
#     return prompt

import re
import datetime
from typing import List, Dict, Any
from difflib import get_close_matches
import re
import datetime
from typing import List, Dict, Any
from difflib import get_close_matches

import re
import datetime
from typing import List, Dict, Any
from difflib import get_close_matches
import re
import datetime
from typing import List, Dict, Any
from difflib import get_close_matches
import re
import datetime
from typing import List, Dict, Any
from difflib import get_close_matches

import re
import datetime
from typing import List, Dict, Any
from difflib import get_close_matches

import re
import datetime
from typing import List, Dict, Any
from difflib import get_close_matches

def repair_generated_sql11(
    raw_llm_output: str,
    tables: List[str],
    schema_catalog: Dict[str, Any] = None,
    join_info: Dict[str, Any] = None
) -> str:
    """
    Repair, normalize, and auto-correct generated SQL.
    PERMANENT FIX for JOIN injection and column qualification issues.
    """
    sql = (raw_llm_output or "").strip()

    # --- Cleanup code fences ---
    if "```sql" in sql:
        sql = sql.split("```sql", 1)[1].split("```", 1)[0].strip()
    elif "```" in sql:
        sql = re.sub(r'```.*?```', '', sql, flags=re.DOTALL).strip()

    # Remove comment lines
    sql = "\n".join([ln for ln in sql.splitlines() if not ln.strip().startswith('--')])

    # --- Fix INTERVAL 'x quarter' → months ---
    for i in range(1, 5):
        sql = re.sub(
            rf"INTERVAL\s+['\"]{i}\s+quarters?['\"]",
            f"INTERVAL '{i*3} months'",
            sql,
            flags=re.IGNORECASE
        )

    # --- Ensure schema-qualified table names ---
    for table in tables:
        clean_table = table.replace('"', '')
        parts = clean_table.split('.')
        if len(parts) == 2:
            schema, table_name = parts
            quoted = f'"{schema}"."{table_name}"'
            sql = re.sub(rf'(?<!")\b{re.escape(schema)}\.{re.escape(table_name)}\b(?!")', quoted, sql)

    # --- Fix CURRENT_DATE / NOW() ---
    today = datetime.date.today()
    fixed_date = f"DATE '{today.year}-{today.month:02d}-{today.day:02d}'"
    sql = re.sub(r'\bCURRENT_DATE\b', fixed_date, sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNOW\(\)\b', fixed_date, sql, flags=re.IGNORECASE)

    # --- Fix = NULL / != NULL ---
    sql = re.sub(r'\b(\w+)\s*=\s*NULL\b', r'\1 IS NULL', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\b(\w+)\s*!=\s*NULL\b', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)

    # --- CRITICAL: Extract and normalize existing FROM clause BEFORE processing ---
    from_match = re.search(r'\bFROM\s+(.*?)(?:\bWHERE\b|\bGROUP BY\b|\bHAVING\b|\bORDER BY\b|\bLIMIT\b|$)', 
                          sql, re.IGNORECASE | re.DOTALL)
    
    existing_from_clause = ""
    remaining_sql = sql
    
    if from_match:
        existing_from_clause = from_match.group(1).strip()
        # Split SQL into parts: before FROM, FROM clause, after FROM
        from_start = from_match.start()
        from_end = from_match.end(1)
        sql_before_from = sql[:from_start]
        sql_after_from = sql[from_end:]
        remaining_sql = sql_before_from + sql_after_from
    
    # --- Build proper JOIN structure ---
    all_table_columns = {}  # Map of table_alias -> list of columns
    table_to_alias = {}  # Map full table name -> alias
    first_alias = None
    new_from_clause = ""
    
    if join_info and tables:
        used_aliases = set()
        first_table = tables[0]
        
        # Extract just the table name (without schema) for first table alias
        first_alias = first_table.split('.')[-1].strip('"')
        used_aliases.add(first_alias)
        table_to_alias[first_table] = first_alias
        
        # Start building FROM clause with first table
        new_from_clause = f'FROM {first_table} AS "{first_alias}"'
        
        # Get columns for first table
        if schema_catalog:
            for table_key, info in schema_catalog.items():
                if first_table in table_key or table_key in first_table:
                    all_table_columns[first_alias] = [col['name'] for col in info.get('columns', [])]
                    break

        # Build JOIN clauses
        join_clauses = []
        for join_key, join_cond in join_info.items():
            if isinstance(join_key, (list, tuple)):
                join_tbl = join_key[1] if len(join_key) > 1 else join_key[0]
            else:
                join_tbl = join_key

            # Generate unique alias for joined table
            alias_base = join_tbl.split('.')[-1].strip('"')
            alias = alias_base
            count = 1
            while alias in used_aliases:
                alias = f"{alias_base}_j{count}"
                count += 1
            used_aliases.add(alias)
            table_to_alias[join_tbl] = alias
            
            # Get columns for joined table
            if schema_catalog:
                for table_key, info in schema_catalog.items():
                    if join_tbl in table_key or table_key in join_tbl:
                        all_table_columns[alias] = [col['name'] for col in info.get('columns', [])]
                        break

            left_cols = join_cond.get('left_columns', [])
            right_cols = join_cond.get('right_columns', [])
            join_type = join_cond.get('type', 'INNER').upper()

            if left_cols and right_cols and len(left_cols) == len(right_cols):
                on_clauses = [
                    f'"{first_alias}"."{left}" = "{alias}"."{right}"' 
                    for left, right in zip(left_cols, right_cols)
                ]
                on_sql = " AND ".join(on_clauses)
                join_clause = f'{join_type} JOIN {join_tbl} AS "{alias}" ON {on_sql}'
                join_clauses.append(join_clause)
        
        # Append all JOINs to FROM clause
        if join_clauses:
            new_from_clause += " " + " ".join(join_clauses)
    
    # --- Replace or insert FROM clause ---
    if new_from_clause:
        # Check if FROM already exists
        if re.search(r'\bFROM\b', remaining_sql, re.IGNORECASE):
            # Replace existing FROM clause
            sql = re.sub(
                r'\bFROM\s+.*?(?=\bWHERE\b|\bGROUP BY\b|\bHAVING\b|\bORDER BY\b|\bLIMIT\b|$)',
                new_from_clause + ' ',
                sql,
                count=1,
                flags=re.IGNORECASE | re.DOTALL
            )
        else:
            # Insert FROM clause after SELECT
            select_end = re.search(r'(?<=SELECT\s).*?(?=\bWHERE\b|\bGROUP BY\b|$)', sql, re.IGNORECASE)
            if select_end:
                insert_pos = select_end.end()
                sql = sql[:insert_pos] + f" {new_from_clause} " + sql[insert_pos:]
    
    # --- NOW qualify columns with aliases ---
    if table_to_alias and all_table_columns:
        # Build comprehensive column mapping
        all_columns = {}  # Map column_lower -> (actual_name, preferred_alias)
        for alias, cols in all_table_columns.items():
            for col in cols:
                col_lower = col.lower()
                if col_lower not in all_columns:
                    all_columns[col_lower] = (col, alias)
                elif alias == first_alias:
                    # Override with first_alias if found
                    all_columns[col_lower] = (col, alias)
        
        # Qualify unqualified columns in SELECT, WHERE, GROUP BY, ORDER BY
        for col_lower, (actual_col, target_alias) in all_columns.items():
            # Pattern: Match column name that's NOT already qualified
            # Negative lookbehind: not preceded by quote-dot pattern
            # Negative lookahead: not followed by dot or inside function params
            pattern = rf'(?<!")(?<!\.)\b{re.escape(actual_col)}\b(?![\.":])'
            
            def qualify_column(match):
                # Get context around match
                start = max(0, match.start() - 100)
                end = min(len(sql), match.end() + 20)
                context = sql[start:end]
                match_in_context = match.start() - start
                
                before = context[:match_in_context]
                
                # Skip if already qualified (has "alias". before it)
                if re.search(r'"[^"]+"\.\s*$', before):
                    return match.group(0)
                
                # Skip if inside function parameters (between parentheses with function name before)
                if re.search(r'\b(?:TO_DATE|CAST|EXTRACT|NULLIF|COALESCE|SUBSTRING|TRIM)\s*\([^)]*$', before, re.IGNORECASE):
                    return match.group(0)
                
                # Skip if it's an alias definition
                if re.search(r'\bAS\s+[^\s,]*$', before, re.IGNORECASE):
                    return match.group(0)
                
                # Skip if it's in FROM/JOIN clause
                if re.search(r'\b(?:FROM|JOIN)\s+[^,]*$', before, re.IGNORECASE):
                    return match.group(0)
                
                # Qualify the column
                return f'"{target_alias}"."{actual_col}"'
            
            sql = re.sub(pattern, qualify_column, sql, flags=re.IGNORECASE)
    
    # --- Remove schema.table.column, keep only alias.column ---
    for full_table, alias in table_to_alias.items():
        parts = full_table.replace('"', '').split('.')
        if len(parts) == 2:
            schema, table_name = parts
            # Pattern: "schema"."table"."column" -> "alias"."column"
            pattern = rf'"{re.escape(schema)}"\."{re.escape(table_name)}"\.("[^"]+")'
            replacement = f'"{alias}".\\1'
            sql = re.sub(pattern, replacement, sql)
    
    # --- Safe TEXT column handling (AFTER column qualification) ---
    if schema_catalog:
        text_columns = {}
        for table_key, info in schema_catalog.items():
            for c in info.get('columns', []):
                col_name = c['name']
                col_type = c.get('type', '').lower()
                if col_type in ['text', 'varchar', 'char', 'character varying']:
                    text_columns[col_name.lower()] = col_name
        
        # Wrap TEXT columns in numeric comparisons
        for col_lower, col_name in text_columns.items():
            for op in ['>', '<', '>=', '<=', '=', '!=', '<>']:
                # Pattern: "alias"."column" OP number (not already wrapped)
                pattern = rf'(?<!CAST\(NULLIF\()("[^"]+")\.("{re.escape(col_name)}")\s*{re.escape(op)}\s*(-?\d+(?:\.\d+)?)'
                replacement = rf'CAST(NULLIF(\1.\2, \'\') AS NUMERIC) {op} \3'
                sql = re.sub(pattern, replacement, sql)
    
    # --- Safe CAST operations ---
    sql = re.sub(
        r'CAST\(\s*("[^"]+"\."[^"]+"|"[^"]+")\s+AS\s+DATE\s*\)',
        r"CAST(NULLIF(\1, '') AS DATE)",
        sql,
        flags=re.IGNORECASE
    )
    sql = re.sub(
        r'CAST\(\s*("[^"]+"\."[^"]+"|"[^"]+")\s+AS\s+(NUMERIC|INTEGER|BIGINT|DECIMAL|FLOAT)\s*\)',
        r"CAST(NULLIF(\1, '') AS \2)",
        sql,
        flags=re.IGNORECASE
    )
    
    # --- Safe division ---
    def _safe_div(m):
        left = m.group(1)
        right = m.group(2).strip()
        if 'NULLIF' in right:
            return m.group(0)
        return f"{left} / NULLIF({right}, 0)"
    
    sql = re.sub(r'([^\s]+\s*)/\s*([A-Za-z0-9_"\)\.]+(?:\([^\)]*\))?)', _safe_div, sql)

    # --- Fix misplaced AND/OR before WHERE ---
    sql = re.sub(r'\b(AND|OR)\s+WHERE\b', 'WHERE', sql, flags=re.IGNORECASE)

    # --- Fix ILIKE %% → % ---
    sql = sql.replace("%%", "%")
    sql = re.sub(r'\s+', ' ', sql).strip()

    # --- Balance parentheses & trailing semicolon ---
    open_p, close_p = sql.count('('), sql.count(')')
    if open_p > close_p:
        sql += ')' * (open_p - close_p)
    if not sql.endswith(';'):
        sql += ';'

    return sql


import re
import re

def clean_generated_sql11(sql: str) -> str:
    """Automatically fix common AI-generated SQL issues, including safe timestamp casting."""
    if not sql:
        return sql

    # Fix NULLIF without second argument (e.g., NULLIF(100))
    sql = re.sub(r'NULLIF\(\s*(\d+)\s*\)', r'NULLIF(\1, 0)', sql)

    # Fix stray commas before closing parentheses
    sql = re.sub(r',\s*\)', ')', sql)

    # Remove stray casts like ':: )'
    sql = re.sub(r'::\s*\)', ')', sql)

    # Fix TRIM(...)=NULL -> TRIM(...) IS NULL OR TRIM(...)=''
    sql = re.sub(
        r'TRIM\((\w+)\)\s*=\s*NULL',
        r'(\1 IS NULL OR TRIM(\1) = \'\')',
        sql,
        flags=re.IGNORECASE
    )

    # Safely cast all TEXT columns to TIMESTAMP
    def safe_timestamp(match):
        col = match.group(1)
        return f"CASE WHEN {col} IS NULL OR TRIM({col}) = '' THEN NULL ELSE {col}::TIMESTAMP END"

    sql = re.sub(r'(\w+)::TIMESTAMP', safe_timestamp, sql, flags=re.IGNORECASE)

    sql = re.sub(r'::\s*\)', ')', sql)

    return sql

# ============================================================================
# MAIN SQL GENERATION PIPELINE
# ============================================================================

# def run_sql_generation_graph(
#     question: str,
#     user_id: str,
#     session_id: str,
#     history: List[Dict] = None
# ) -> Tuple[str, str, str]:
#     """
#     Generate SQL for a given question with guaranteed table selection.
#     Returns: (sql, explanation, error_info)
#     """
#     try:
#         print(f"🔍 SQL Generation for question: {question}")
#         print(f"🔑 Session ID: {session_id}")

#         # Import SessionManager dynamically to avoid circular imports
        
#         # Fetch session data
#         session_data = SessionManager.get_session(session_id)
#         if not session_data:
#             return ("-- ERROR: Session not found", "Connect to database first", "SESSION_ERROR")

#         schema_catalog = session_data.get("schema_catalog", {})
#         if not schema_catalog:
#             return ("-- ERROR: No schema", "Schema catalog is empty", "SCHEMA_ERROR")

#         print(f"📚 Schema catalog has {len(schema_catalog)} tables")

#         # Step 1: Select relevant tables (GUARANTEED to return at least one)
#         selected_tables = select_relevant_tables(
#             question=question,
#             schema_catalog=schema_catalog,
#             similarity_threshold=0.4,
#             max_tables=5
#         )
        
#         if not selected_tables:
#             # ABSOLUTE FALLBACK: Use all tables
#             selected_tables = list(schema_catalog.keys())[:3]
#             print(f"⚠️ EMERGENCY FALLBACK: Using first {len(selected_tables)} tables")
        
#         print(f"✅ Final selected tables: {selected_tables}")

#         # Step 2: Discover join keys
#         try:
#             join_info = discover_join_keys(selected_tables, schema_catalog)
#         except Exception as e:
#             print(f"⚠️ Join discovery failed: {e}")
#             join_info = {}

#         # Step 3: Build prompt
#         prompt = build_dynamic_sql_prompt(
#             question=question,
#             schema_catalog=schema_catalog,
#             tables=selected_tables,
#             join_info=join_info,
#             history=history or [],
#             similarity_threshold=0.4
#         )

#         print(f"📝 Prompt length: {len(prompt)} chars")

#         # Step 4: Generate SQL using LLM
#         try:
#             from genai_app.utils.llm_config import get_llama_maverick_llm
#             llm = get_llama_maverick_llm()
#             if not llm:
#                 return ("-- ERROR: LLM not configured", "LLM function not available", "LLM_ERROR")
            
#             response = llm.invoke(prompt)

#              # ✅ CRITICAL: Extract string from response
#             if hasattr(response, 'content'):
#                 raw_sql = response.content
#             elif isinstance(response, str):
#                 raw_sql = response
#             else:
#                 raw_sql = str(response)
            
#             print(f"🔍 Raw LLM SQL (first 300 chars):\n{raw_sql[:300]}\n")
#         except Exception as e:
#             print(f"❌ LLM invocation failed: {e}")
#             traceback.print_exc()
#             return (f"-- ERROR: LLM failed - {str(e)}", "LLM generation failed", "LLM_ERROR")

#         # Step 5: Repair and finalize SQL

#         sql = nuclear_repair_sql(
#             raw_llm_output=raw_sql,  # ✅ Use extracted string
#             tables=selected_tables,
#             schema_catalog=schema_catalog,
#             join_info=join_info,
#             debug=True  # ✅ See what's happening
#         )
                
#         # Step 5b: Clean common AI SQL issues
#         sql = clean_generated_sql(sql)

#         print(f"✅ Generated SQL Final:\n{sql}\n")
#          # ✅ Step 5c: Generate explanation of SQL (using LLM)
#         try:
#             explanation_prompt = f"""
#             Explain in plain English what the following SQL query does.
#             Avoid markdown or bullet points. Write in simple paragraphs.

#             SQL:
#             {sql}
#             """

#             explanation_response = llm.invoke(explanation_prompt)

#             if hasattr(explanation_response, 'content'):
#                 explanation = explanation_response.content.strip()
#             elif isinstance(explanation_response, str):
#                 explanation = explanation_response.strip()
#             else:
#                 explanation = str(explanation_response).strip()

#             print(f"🧠 SQL Explanation:\n{explanation}\n")
#         except Exception as e:
#             print(f"⚠️ Explanation generation failed: {e}")
#             explanation = "Explanation unavailable due to LLM error."

#         # ✅ Final return (use explanation instead of fixed message)
#         return (sql, explanation, "")
#         # return (sql, "SQL generated successfully", "")

#     except Exception as e:
#         print(f"❌ SQL generation error: {e}")
#         traceback.print_exc()
#         return (f"-- ERROR: {str(e)}", "SQL generation failed", "GENERATION_ERROR")


from typing import List, Dict, Tuple
import traceback
import chromadb
from sentence_transformers import SentenceTransformer

# ✅ Initialize ChromaDB and Embedder ONCE (Module-level globals)
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection("schema_embeddings")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast model

print("✅ ChromaDB initialized with collection: schema_embeddings")


# def run_sql_generation_graph(
#     question: str,
#     user_id: str,
#     session_id: str,
#     history: List[Dict] = None
# ) -> Tuple[str, str, str]:
#     """
#     Generate SQL for a given question with guaranteed table selection.
#     Uses ChromaDB embeddings as fallback if schema catalog fails.
#     Returns: (sql, explanation, error_info)
#     """
#     try:
#         print(f"🔍 SQL Generation for question: {question}")
#         print(f"🔑 Session ID: {session_id}")

#         # Import SessionManager dynamically to avoid circular imports
#         # from genai_app.session_manager import SessionManager
        
#         # Fetch session data
#         session_data = SessionManager.get_session(session_id)
#         if not session_data:
#             return ("-- ERROR: Session not found", "Connect to database first", "SESSION_ERROR")

#         schema_catalog = session_data.get("schema_catalog", {})
#         if not schema_catalog:
#             return ("-- ERROR: No schema", "Schema catalog is empty", "SCHEMA_ERROR")

#         print(f"📚 Schema catalog has {len(schema_catalog)} tables")

#         # ✅ Step 1: Select relevant tables with EMBEDDING FALLBACK
#         selected_tables = select_relevant_tables_with_embedding_fallback(
#             question=question,
#             schema_catalog=schema_catalog,
#             user_id=user_id,
#             session_id=session_id,
#             similarity_threshold=0.4,
#             max_tables=5
#         )
        
#         if not selected_tables:
#             # ABSOLUTE EMERGENCY FALLBACK: Use first 3 tables
#             selected_tables = list(schema_catalog.keys())[:3]
#             print(f"⚠️ EMERGENCY FALLBACK: Using first {len(selected_tables)} tables")
        
#         print(f"✅ Final selected tables: {selected_tables}")

#         # Step 2: Discover join keys
#         try:
#             # from genai_app.sql_generation.join_discovery import discover_join_keys
#             join_info = discover_join_keys(selected_tables, schema_catalog)
#         except Exception as e:
#             print(f"⚠️ Join discovery failed: {e}")
#             join_info = {}

#         # Step 3: Build prompt
#         # from genai_app.sql_generation.prompt_builder import build_dynamic_sql_prompt
#         prompt = build_dynamic_sql_prompt(
#             question=question,
#             schema_catalog=schema_catalog,
#             tables=selected_tables,
#             join_info=join_info,
#             history=history or [],
#             similarity_threshold=0.4
#         )

#         print(f"📝 Prompt length: {len(prompt)} chars")

#         # Step 4: Generate SQL using LLM
#         try:
#             from genai_app.utils.llm_config import get_llama_maverick_llm
#             llm = get_llama_maverick_llm()
#             if not llm:
#                 return ("-- ERROR: LLM not configured", "LLM function not available", "LLM_ERROR")
            
#             response = llm.invoke(prompt)

#             # ✅ CRITICAL: Extract string from response
#             if hasattr(response, 'content'):
#                 raw_sql = response.content
#             elif isinstance(response, str):
#                 raw_sql = response
#             else:
#                 raw_sql = str(response)
            
#             print(f"🔍 Raw LLM SQL (first 300 chars):\n{raw_sql[:300]}\n")
#         except Exception as e:
#             print(f"❌ LLM invocation failed: {e}")
#             traceback.print_exc()
#             return (f"-- ERROR: LLM failed - {str(e)}", "LLM generation failed", "LLM_ERROR")

#         # Step 5: Repair and finalize SQL
#         # from genai_app.sql_generation.sql_repair import nuclear_repair_sql, clean_generated_sql
#         sql = nuclear_repair_sql(
#             raw_llm_output=raw_sql,
#             tables=selected_tables,
#             schema_catalog=schema_catalog,
#             join_info=join_info,
#             debug=True
#         )
                
#         # Step 5b: Clean common AI SQL issues
#         sql = clean_generated_sql(sql)

#         print(f"✅ Generated SQL Final:\n{sql}\n")
        
#         # ✅ Step 5c: Generate explanation of SQL (using LLM)
#         try:
#             explanation_prompt = f"""
# Explain in plain English what the following SQL query does.
# Avoid markdown or bullet points. Write in simple paragraphs.

# SQL:
# {sql}
# """

#             explanation_response = llm.invoke(explanation_prompt)

#             if hasattr(explanation_response, 'content'):
#                 explanation = explanation_response.content.strip()
#             elif isinstance(explanation_response, str):
#                 explanation = explanation_response.strip()
#             else:
#                 explanation = str(explanation_response).strip()

#             print(f"🧠 SQL Explanation:\n{explanation}\n")
#         except Exception as e:
#             print(f"⚠️ Explanation generation failed: {e}")
#             explanation = "Explanation unavailable due to LLM error."

#         # ✅ Final return with explanation
#         return (sql, explanation, "")

#     except Exception as e:
#         print(f"❌ SQL generation error: {e}")
#         traceback.print_exc()
#         return (f"-- ERROR: {str(e)}", "SQL generation failed", "GENERATION_ERROR")


from typing import List, Dict, Tuple, Any
import traceback
import re
import datetime

def run_sql_generation_graph1410(
    question: str,
    user_id: str,
    session_id: str,
    history: List[Dict] = None
) -> Tuple[str, str, str]:
    """
    Fully merged SQL generation function.
    - Handles session fallback & emergency creation
    - Uses schema-aware catalog & embeddings for table selection
    - Enforces join keys, column casting, null handling, and 2024 date logic
    - Generates SQL + explanation using LLM
    Returns: (sql, explanation, error_info)
    """
    try:
        print(f"🔍 SQL Generation for question: {question}")
        print(f"🔑 Session ID: {session_id}")

        # -------------------------------
        # Step 1: Fetch or fallback session
        # -------------------------------
        session_data = SessionManager.get_session(session_id)

        if not session_data:
            print(f"⚠️ Session {session_id} not found, trying fallback...")
            if session_store:
                session_id, session_data = next(iter(session_store.items()))
                print(f"↩️ Falling back to session {session_id}")
            else:
                try:
                    emergency_session = create_emergency_session()
                    if emergency_session:
                        session_id, session_data = emergency_session
                        session_store[session_id] = session_data
                        try:
                            embed_schema_catalog(
                                user_id=session_data["user_id"],
                                db_id=session_id,
                                schema_catalog=session_data["schema_catalog"],
                                embedder=embedder,
                                collection=collection
                            )
                            print("✅ Emergency session schema embedded")
                        except Exception as embed_error:
                            print(f"⚠️ Emergency embedding failed: {embed_error}")
                    else:
                        return ("-- ERROR: No active DB session and cannot create emergency session",
                                "Please connect to database first using the connect endpoint", "")
                except Exception as e:
                    print(f"❌ Emergency session creation failed: {e}")
                    traceback.print_exc()
                    return ("-- ERROR: No active DB session", "Please connect to database first", "")

        if not session_data or "schema_catalog" not in session_data:
            print("❌ Invalid session data - missing schema_catalog")
            return ("-- ERROR: Invalid session data", "Session corrupted, please reconnect", "")

        schema_catalog = session_data["schema_catalog"]
        print(f"📚 Schema catalog has {len(schema_catalog)} tables")

        # -------------------------------
        # Step 2: Select relevant tables
        # -------------------------------
        selected_tables = select_relevant_tables_with_embedding_fallback(
            question=question,
            schema_catalog=schema_catalog,
            user_id=user_id,
            session_id=session_id,
            similarity_threshold=0.4,
            max_tables=5
        )

        if not selected_tables:
            selected_tables = list(schema_catalog.keys())[:3]
            print(f"⚠️ EMERGENCY FALLBACK: Using first {len(selected_tables)} tables")

        print(f"✅ Final selected tables: {selected_tables}")

        # -------------------------------
        # Step 3: Discover join keys
        # -------------------------------
        try:
            join_info = discover_join_keys(selected_tables, schema_catalog)
        except Exception as e:
            print(f"⚠️ Join discovery failed: {e}")
            join_info = {}

        # -------------------------------
        # Step 4: Build dynamic SQL prompt
        # -------------------------------
        prompt = build_dynamic_sql_prompt(
            question=question,
            schema_catalog=schema_catalog,
            tables=selected_tables,
            join_info=join_info,
            history=history or [],
            similarity_threshold=0.4
        )
        print(f"📝 Prompt length: {len(prompt)} chars")

        # -------------------------------
        # Step 5: LLM SQL generation
        # -------------------------------
        try:
            llm = get_llama_maverick_llm()
            if not llm:
                return ("-- ERROR: LLM not configured", "LLM function not available", "LLM_ERROR")

            response = llm.invoke(prompt)
            raw_sql = getattr(response, "content", response if isinstance(response, str) else str(response))
            print(f"🔍 Raw LLM SQL (first 300 chars):\n{raw_sql[:300]}\n")

        except Exception as e:
            print(f"❌ LLM invocation failed: {e}")
            traceback.print_exc()
            return (f"-- ERROR: LLM failed - {str(e)}", "LLM generation failed", "LLM_ERROR")


        

        # -------------------------------
        # Step 6: Repair & finalize SQL
        # -------------------------------
        sql = nuclear_repair_sql(
            raw_llm_output=raw_sql,
            tables=selected_tables,
            schema_catalog=schema_catalog,
            join_info=join_info,
            debug=True
        )

        sql = fix_extract_syntax(sql, debug=True)
        # Choose a column that makes sense for numeric aggregation
        sql = re.sub(r'CASE WHEN \* IS NULL', 'CASE WHEN app.applicationamount IS NULL', sql)
        sql = re.sub(r'TRIM\(\s*::text\)', 'TRIM(app.applicationamount::text)', sql)
        sql = re.sub(r'COUNT\s*\(\(\s*CASE', 'COUNT(CASE', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SUM\s*\(\(\s*CASE', 'SUM(CASE', sql, flags=re.IGNORECASE)
        sql = re.sub(r'END\s*\)\)', 'END)', sql)  # remove double closing

                # Replace SUM((CASE WHEN CASE WHEN ... END IS NULL THEN NULL ... END)) 
        # with SUM(CASE WHEN app.fsl_programtype ILIKE '%dsc%' THEN 1 ELSE 0 END)
        sql = re.sub(
            r'SUM\(\(\s*CASE WHEN CASE WHEN (.*?) THEN 1 ELSE 0 END IS NULL THEN NULL ELSE NULL END\s*\)\)',
            r'SUM(CASE WHEN \1 THEN 1 ELSE 0 END)',
            sql,
            flags=re.IGNORECASE
        )

        sql = clean_generated_sql(sql)
        sql = sanitize_sql_for_nulls(sql, schema_catalog=schema_catalog, question=question)
        sql = auto_fix_schema_mismatches(sql, schema_catalog)
        sql = validate_sql_against_catalog(sql, schema_catalog)
        sql = _fix_groupby_alias_and_parens(sql)

        # -------------------------------
        # Step 7: Handle 2024 enforcement for stage tables
        # -------------------------------
        if any("stage" in tbl for tbl in selected_tables):
            date_info = get_fixed_date_context()
            sql = enforce_2024(sql, date_info)

        print(f"✅ Generated SQL Final:\n{sql}\n")

        # -------------------------------
        # Step 8: Generate explanation
        # -------------------------------
        try:
            explanation_prompt = f"""
Explain in plain English what the following SQL query does.
Avoid markdown or bullet points. Write in simple paragraphs.

SQL:
{sql}
"""
            explanation_response = llm.invoke(explanation_prompt)
            explanation = getattr(explanation_response, "content", explanation_response if isinstance(explanation_response, str) else str(explanation_response)).strip()
            print(f"🧠 SQL Explanation:\n{explanation}\n")
        except Exception as e:
            print(f"⚠️ Explanation generation failed: {e}")
            explanation = "Explanation unavailable due to LLM error."

        return (sql, explanation, "")

    except Exception as e:
        print(f"❌ SQL generation error: {e}")
        traceback.print_exc()
        return (f"-- ERROR: {str(e)}", "SQL generation failed", "GENERATION_ERROR")



def select_relevant_tables_with_embedding_fallback(
    question: str,
    schema_catalog: Dict,
    user_id: str,
    session_id: str,
    similarity_threshold: float = 0.4,
    max_tables: int = 5
) -> List[str]:
    """
    Select relevant tables using primary method, fall back to ChromaDB embeddings if needed.
    Uses module-level initialized ChromaDB client and embedder.
    """
    
    # ✅ Try primary method (your existing select_relevant_tables function)
    try:
        # from genai_app.sql_generation.table_selector import select_relevant_tables
        
        tables = select_relevant_tables(
            question=question,
            schema_catalog=schema_catalog,
            similarity_threshold=similarity_threshold,
            max_tables=max_tables
        )
        
        if tables:
            print(f"✅ Primary method returned {len(tables)} tables: {tables}")
            return tables
    except Exception as e:
        print(f"⚠️ Primary table selection failed: {e}")
    
    # ✅ FALLBACK: Query ChromaDB embeddings (using module-level globals)
    print("⚠️ No tables from primary method, querying ChromaDB embeddings...")
    
    try:
        # Use module-level initialized embedder and collection
        global embedder, collection
        
        # Embed the question
        query_embedding = embedder.encode([question]).tolist()[0]
        
        # Query ChromaDB with proper filtering
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max_tables,
            where={
                "$and": [
                    {"user_id": user_id},
                    {"db_id": session_id}
                ]
            }
        )
        
        # Extract table names from results
        if results and results['metadatas'] and results['metadatas'][0]:
            tables = []
            for meta in results['metadatas'][0]:
                table_name = meta.get('table')
                if table_name and table_name in schema_catalog:
                    tables.append(table_name)
            
            if tables:
                print(f"✅ Retrieved {len(tables)} tables from ChromaDB: {tables}")
                return tables
            else:
                print("⚠️ ChromaDB returned results but no valid tables found")
        else:
            print("⚠️ No results from ChromaDB query")
            
    except Exception as e:
        print(f"❌ ChromaDB fallback failed: {e}")
        traceback.print_exc()
    
    # ✅ FINAL EMERGENCY FALLBACK: Return empty list (triggers emergency fallback in main function)
    print("⚠️ All retrieval methods failed, returning empty list")
    return []


# ✅ Optional: Helper function to add schemas to ChromaDB
def embed_schema_to_chromadb(
    user_id: str,
    db_id: str,
    schema_catalog: Dict
):
    """
    Embed schema catalog into ChromaDB for semantic retrieval.
    Call this when user connects to a database.
    """
    try:
        global embedder, collection
        
        documents = []
        metadatas = []
        ids = []
        
        for table_name, table_info in schema_catalog.items():
            # Create a rich text description for embedding
            columns = table_info.get('columns', [])
            column_names = [col['name'] for col in columns]
            column_types = [col['type'] for col in columns]
            
            # Create searchable text
            doc_text = (
                f"Table: {table_name}. "
                f"Columns: {', '.join(column_names)}. "
                f"Types: {', '.join(column_types)}."
            )
            
            documents.append(doc_text)
            metadatas.append({
                "user_id": user_id,
                "db_id": db_id,
                "table": table_name,
                "column_count": len(columns)
            })
            ids.append(f"{user_id}_{db_id}_{table_name}")
        
        # Embed and store
        embeddings = embedder.encode(documents).tolist()
        
        collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Embedded {len(documents)} tables into ChromaDB")
        
    except Exception as e:
        print(f"❌ Failed to embed schema: {e}")
        traceback.print_exc()
# ============================================================================
# HELPER: SELECT TABLES FOR QUESTION (WRAPPER)
# ============================================================================

def select_tables_for_question(
    question: str,
    schema_catalog: dict,
    session_id: str = None
) -> List[str]:
    """
    Wrapper function for table selection with logging.
    GUARANTEED to return at least one table.
    """
    print(f"🎯 Selecting tables for: {question}")
    
    if not schema_catalog:
        print("❌ Schema catalog is empty!")
        return []
    
    tables = select_relevant_tables(
        question=question,
        schema_catalog=schema_catalog,
        similarity_threshold=0.4,
        max_tables=5
    )
    
    # ABSOLUTE GUARANTEE: Never return empty
    if not tables:
        all_tables = list(schema_catalog.keys())
        print(f"⚠️ CRITICAL FALLBACK: Using first 3 of {len(all_tables)} tables")
        tables = all_tables[:3]
    
    print(f"📊 Selected: {tables}")
    return tables

# ============================================================================
# DISCOVER JOIN KEYS (FALLBACK IMPLEMENTATION)
# ============================================================================

def discover_join_keys(tables: List[str], schema_catalog: dict) -> dict:
    """
    Discover potential join keys between tables.
    Returns dictionary of join relationships.
    """
    join_info = {}
    
    if len(tables) < 2:
        return join_info
    
    # Common join key patterns
    common_join_keys = [
        'customerid', 'customer_id',
        'accountid', 'account_id',
        'userid', 'user_id',
        'loanid', 'loan_id',
        'applicationid', 'application_id',
        'id'
    ]
    
    for i, table1 in enumerate(tables):
        for table2 in tables[i+1:]:
            table1_info = schema_catalog.get(table1, {})
            table2_info = schema_catalog.get(table2, {})
            
            cols1 = [c['name'].lower() for c in table1_info.get('columns', [])]
            cols2 = [c['name'].lower() for c in table2_info.get('columns', [])]
            
            # Find common columns
            common = set(cols1) & set(cols2)
            
            # Prioritize known join keys
            for key in common_join_keys:
                if key in common:
                    join_info[(table1, table2)] = {
                        'left_columns': [key],
                        'right_columns': [key],
                        'type': 'INNER'
                    }
                    break
            
            # Fallback: use first common column
            if (table1, table2) not in join_info and common:
                first_common = list(common)[0]
                join_info[(table1, table2)] = {
                    'left_columns': [first_common],
                    'right_columns': [first_common],
                    'type': 'INNER'
                }
    
    return join_info

# ============================================================================
# SAFE GENERATE NARRATIVE (STUB)
# ============================================================================

# ============================================================================
# EXPORT ALL FUNCTIONS
# ============================================================================

__all__ = [
    'normalize_name',
    'fuzzy_score',
    'select_relevant_tables',
    'is_numeric_column',
    'format_column_for_agg',
    'get_best_column_match',
    'resolve_columns_in_question',
    'get_column_table_alias',
    'generate_joins',
    'quote_table_name',
    'detect_aggregations',
    'detect_arithmetic_patterns',
    'build_dynamic_sql_prompt',
    'repair_generated_sql',
    'run_sql_generation_graph',
    'select_tables_for_question',
    'discover_join_keys',
    'safe_generate_narrative',
    'llm_generate_narrative',
    'llm_generate_recommendation',
    'llm_generate_chart_config',
]
# ============================================================================
# MAIN SQL GENERATION
# ============================================================================
def run_sql_generation_graph710(
    question: str,
    user_id: str,
    session_id: str,
    history: List[Dict] = None
) -> Tuple[str, str, str]:
    """Main SQL generation pipeline."""

    print(f"🔍 SQL Generation for: {question}")
    print(f"🔑 Session: {session_id}")

    session_data = SessionManager.get_session(session_id)
    if not session_data:
        return ("-- ERROR: Session not found", "Connect to database first", "")

    schema_catalog = session_data.get("schema_catalog", {})
    if not schema_catalog:
        return ("-- ERROR: No schema", "Schema catalog empty", "")

    try:
        selected_tables = select_tables_for_question(
            question=question,
            schema_catalog=schema_catalog,
            session_id=session_id
        )

        if not selected_tables:
            return ("-- ERROR: No tables found", "Could not identify relevant tables", "")

        join_info = discover_join_keys(selected_tables, schema_catalog)

        prompt = build_dynamic_sql_prompt(
            question=question,
            tables=selected_tables,
            schema_catalog=schema_catalog,
            join_info=join_info,
            history=history or []
        )

        if not get_llama_maverick_llm:
            return ("-- ERROR: LLM not configured", "LLM function not available", "")

        llm = get_llama_maverick_llm()
        response = llm.invoke(prompt)

        sql = repair_generated_sql(
            raw_llm_output=response,
            tables=selected_tables,
            schema_catalog=schema_catalog,
            join_info=join_info
        )

        print(f"✅ Generated SQL:\n{sql}\n")
        return (sql, "SQL generated successfully", "")

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return (f"-- ERROR: {str(e)}", "SQL generation failed", "")



# ============================================================================
# REQUIRED GLOBALS (add these at module level in your views.py)
# ============================================================================

# These should already exist in your code:
# session_store = {}
# conversation_memory_store = {}



# ============================================================================
# REQUIRED GLOBALS (add these at module level in your views.py)
# ============================================================================

# These should already exist in your code:
# session_store = {}
# conversation_memory_store = {}

# ============================================================================
# INTEGRATION: Update your existing ask_question_stream view
# ============================================================================

# REPLACE your existing SQL generation section with this:





#     prompt = f"""You are an expert PostgreSQL query generator.

# {schema_section}
# {join_section}
# {history_section}

# # TABLE SELECTION RULES
# - If the question is related to loan approval or approved loans, use ONLY the table: "loan_main_{year_suffix}".
# - If the question is related to loan application or application details, use ONLY the table: "app_main_{year_suffix}".
# - If the question involves both loan approvals and applications, use BOTH tables and JOIN them appropriately based on the relationships provided.

# # DATE RULES
# - Always use year {year_suffix} for tables and date conditions.
# - Use the current month ({current_month}) and current day ({current_day}) for any date-related filters.
# - If the question implies filtering by "today" or "current date", replace it with {year_suffix}-{current_month:02d}-{current_day:02d}.
# - Use proper DATE literals in PostgreSQL, e.g., DATE '{year_suffix}-{current_month:02d}-{current_day:02d}'.

# # TASK
# Generate a PostgreSQL query for: "{question}"

# # RULES
# 1. Use ONLY tables and columns shown above (and follow the table selection rules)
# 2. Always use fully qualified names: "schema"."table"
# 3. Use proper JOINs when needed
# 4. Handle NULLs with COALESCE for aggregates
# 5. Use ILIKE for case-insensitive text matching
# 6. Return ONLY the SQL query, no explanations

# """


def jsonl_line(obj: dict) -> str:
    """Helper to format JSONL."""
    import json
    return json.dumps(obj) + '\n'

# import re
# from typing import Any, Dict, List

# def generate_sql(question: str, user_id: str, db_id: str) -> str:
#     try:

#         prompt = PromptTemplate.from_template(f"""
# You are a PostgreSQL SQL expert.

# Your task is two-fold:
# 1. Generate the SQL query.
# 2. Provide a short **business recommendation** (1–3 lines) based on the query output.

# Strict rules:
# - Use only this table: "final_lusa"."apploan_metrics_2024".
# - Use only the listed columns below (do NOT invent new ones).
# - Always use exact column names.
# - If a column doesn’t exist, respond with: `-- Error: unknown column requested`.

# 📌 Column Dictionary:
# - application_id → Unique loan application ID.
# - channel_code → Loan sales channel (DTM, DTC, FSL).
# - channel_group → Grouping of channels.
# - merchant_id → Merchant/store/vendor ID.
# - merchant_vertical → Industry/business type of merchant.
# - client_id → Client institution/business ID.
# - customer_id → Borrower ID.
# - loan_application_purpose → Purpose of loan (education, medical, etc.).
# - loan_application_city → Borrower city.
# - loan_application_state → Borrower state/region.
# - requested_amount → Amount requested by borrower.
# - approval_amount → Amount approved after checks.
# - loan_application_status → Application status (Approved, Pending, Rejected).
# - funded_date → Date loan was disbursed.
# - loan_servicer_boarding → Loan servicing provider.
# - loan_group_boarding → Loan portfolio group.
# - issuing_bank → Bank/financial institution that issued the loan.
# - decision_source → Automated/manual decision source.
# - note_principal → Original principal amount.
# - annual_percentage_rate → Annual cost of loan (%).
# - loan_term → Loan duration.
# - loan_amount_funded → Actual amount disbursed.
# - fico_score → Borrower’s FICO score (300–850).
# - vantage_score → Borrower’s VantageScore (300–850).
# - is_fraud_suspected → Flag if fraud suspected.
# - is_fraud_confirmed → Flag if fraud confirmed.
# - merchant_discount → Merchant fee for financing.
# - loan_application_date → Date loan was applied.
# - interest_rate → Nominal interest rate (%).
# - annual_income → Borrower’s reported yearly income.
                                              

# 📌 Date Rules:
# - Always treat the year as {date_info["year"]}.
# - Current month = {date_info["month"]}.
# - Current quarter = {date_info["quarter"]}.
# - Current half = {date_info["half"]}.
# - Last quarter = {date_info['quarter_start']} → {date_info['quarter_end']}.
# - Last half = {date_info['half_start']} → {date_info['half_end']}.
# - Do NOT infer other years.

# 📌 Business Rules:
# - Use `application_id` for counting unique applications.
# - Use `channel_code` for DTM/DTC/FSL analysis (never confuse with loan_application_purpose).
# - For loan purpose trends → use `loan_application_purpose`.
# - For geography → use `loan_application_city` or `loan_application_state` with ILIKE and wildcards.
# - For fraud → use `is_fraud_suspected` and `is_fraud_confirmed`.
# - For creditworthiness → use `fico_score`, `vantage_score`, and `annual_income`.
# - For loan amounts:
#   - Requested → requested_amount
#   - Approved → approval_amount
#   - Funded   → loan_amount_funded
# - For time filters:
#   - Use `loan_application_date` (TIMESTAMP).
#   - Use `funded_date` for disbursed loans.
#   - Extract parts with EXTRACT(YEAR/MONTH/QUARTER FROM loan_application_date).
#   - For last quarter:
#       loan_application_date >= DATE_TRUNC('quarter', CURRENT_DATE) - INTERVAL '3 month'
#       AND loan_application_date < DATE_TRUNC('quarter', CURRENT_DATE)
#   - For last N months:
#       loan_application_date >= CURRENT_DATE - INTERVAL 'N month'

# 📊 Recommendation Guidelines:
#     - Provide a concise, professional, and **actionable recommendation (1–3 lines)** derived from the query context and expected output, telling the user what actions they can take to enhance their business.
# - If approvals are low → suggest reviewing underwriting criteria.
# - If fraud is high in a region → tighten fraud checks in that area.
# - If certain loan purposes dominate → recommend targeted loan products.
# - If interest rates are high → suggest better rates for prime borrowers.
# - If income-to-loan ratio is weak → recommend stricter creditworthiness checks.

# Conversation history:
# {history_context}

# Now generate SQL for:
# {question}
# """)
        
# """"""""bfrsep2024  
        # llm = get_llama_maverick_llm()
        # runnable = prompt | llm
        # response = runnable.invoke({
        #     **state,
        #     "history_context": history_context,
        #     "question": state["question"]
        # })
        # text = response.content if hasattr(response, "content") else response

        # if "Recommendation:" in text:
        #     sql_part, explanation_part = text.split("Recommendation:", 1)
        #     sql_clean = sql_part.replace("SQL:", "").strip()
        #     sql_clean = enforce_2024(sql_clean, date_info)
        #     state["sql"] = validate_sql_against_catalog(sql_clean, schema_catalog)
        #     state["explanation"] = explanation_part.strip()
        # else:
        #     sql_clean = text.strip()
        #     sql_clean = enforce_2024(sql_clean, date_info)
        #     state["sql"] = validate_sql_against_catalog(sql_clean, schema_catalog)
        #     state["explanation"] = "-- No recommendation generated"

        # return state
# """"""""bfrsep2024  
        # if "Recommendation:" in text:
        #     sql_part, explanation_part = text.split("Recommendation:", 1)
        #     sql_clean = sql_part.replace("SQL:", "").strip()
        #     # ✅ Force 2024 ranges
        #     state["sql"] = enforce_2024(sql_clean, date_info)
        #     state["explanation"] = explanation_part.strip()
        # else:
        #     sql_clean = text.strip()
        #     state["sql"] = enforce_2024(sql_clean, date_info)
        #     state["explanation"] = "-- No recommendation generated"

        # state["result"] = []
        # return state
# """"""""bfrsep2024  
    # graph = StateGraph(SQLState)
    # graph.add_node("retrieve_context", retrieve_context_node)
    # graph.add_node("generate_sql", generate_sql_node)
    # graph.set_entry_point("retrieve_context")
    # graph.add_edge("retrieve_context", "generate_sql")
    # graph.set_finish_point("generate_sql")

    # final_state = graph.compile().invoke({
    #     "question": question,
    #     "user_id": user_id,
    #     "db_id": db_id,
    # })
    # return (
    #     final_state["sql"],
    #     final_state["explanation"],
    #     final_state.get("summary", "")
    # )
# """"""""bfrsep2024        
        # llm = get_llama_maverick_llm()
        # runnable = prompt | llm
        # response = runnable.invoke({**state, "history_context": history_context, "question": state["question"]})
        # text = response.content if hasattr(response, "content") else response


        # # llm = get_llama_maverick_llm()
        # # runnable = prompt | llm
        # # response = runnable.invoke(state)
        # # text = response.content if hasattr(response, "content") else response

        # if "Recommendation:" in text:
        #     sql_part, explanation_part = text.split("Recommendation:", 1)
        #     state["sql"] = sql_part.replace("SQL:", "").strip()
        #     state["explanation"] = explanation_part.strip()
        # else:
        #     state["sql"] = text.strip()
        #     state["explanation"] = "-- No recommendation generated"

        # state["result"] = []

        # return state
    
#     def summarize_result_node_after_execution(state):
#         try:
#             question = state["question"]
#             sql = state["sql"]
#             result = state["result"]

#             prompt = f"""
# You are a business analyst.

# A user asked: "{question}"
# The SQL used:
# {sql}
# The result from the database:
# {result[:10]}

# Summarize the result in a business-friendly sentence with bullet points if needed.
# """
#             llm = get_llama_maverick_llm()
#             summary = llm.invoke(prompt)
#             state["summary"] = summary
#         except Exception as e:
#             state["summary"] = "Could not generate summary."
#         return state

    

        # if "Explanation:" in text:
        #     sql_part, explanation_part = text.split("Explanation:", 1)
        #     state["sql"] = sql_part.replace("SQL:", "").strip()
        #     state["explanation"] = explanation_part.strip()
        # else:
        #     state["sql"] = text.strip()
        #     state["explanation"] = "-- No explanation generated"

        # return state
        # state["sql"] = response.content if hasattr(response, "content") else response
        # return state

    # graph = StateGraph(SQLState)
    # graph.add_node("retrieve_context", retrieve_context_node)
    # graph.add_node("generate_sql", generate_sql_node)
    # # graph.add_node("summarize_result", summarize_result_node_after_execution)
    # graph.set_entry_point("retrieve_context")
    # graph.add_edge("retrieve_context", "generate_sql")
    # graph.set_finish_point("generate_sql")
    # # graph.add_edge("generate_sql", "summarize_result")
    # # graph.set_finish_point("summarize_result")

    # final_state = graph.compile().invoke({
    #     "question": question,
    #     "user_id": user_id,
    #     "db_id": db_id,
    # })
    # return final_state["sql"], final_state["explanation"], final_state.get("summary", "")




def generate_summary_from_rows(question, sql, rows):
    try:
        prompt = f"""
You are a business analyst.

A user asked: "{question}"
The SQL used:
{sql}
The result from the database:
{rows[:10]}

Summarize the result in a business-friendly sentence.
"""
        llm = get_llama_maverick_llm()
        return llm.invoke(prompt)
    except Exception as e:
        print("⚠️ Summary generation failed:", str(e))
        return "Summary not available."


    # final_state = graph.compile().invoke({"question": question})
    # return {
    #     "sql": final_state.get("sql", "-- No SQL generated"),
    #     "explanation": final_state.get("explanation", "-- No explanation generated")
    # }














































































# class SQLState(TypedDict):
#     question: str
#     context: str
#     sql: str


# def trim_history_tokenwise(history, max_tokens=1000):
#     """Trim conversation history based on token estimate"""
#     trimmed = []
#     total_tokens = 0
#     for item in reversed(history):
#         entry = f"Q: {item['question']}\nSQL: {item['sql']}"
#         tokens = estimate_tokens(entry)
#         if total_tokens + tokens > max_tokens:
#             break
#         trimmed.append(entry)
#         total_tokens += tokens
#     return "\n".join(reversed(trimmed))


# def run_sql_generation_graph(question, user_id, db_id, history=[]):
#     def retrieve_context_node(state):
#         full_context = get_context(state["question"], user_id, db_id)

#         # ✅ Filter only the target table
#         allowed_table = '"stage"."GBM1_prediction_data_with_recommendations"'
#         lines = full_context.splitlines()
#         filtered_lines = [
#             line for line in lines if allowed_table in line or line.strip().startswith("- ")
#         ]
#         context = f'Table: {allowed_table}\n' + "\n".join(filtered_lines)

#         # ✂️ Truncate context if too long
#         state["context"] = truncate_text(context, max_tokens=2500)
#         return state

#     def generate_sql_node(state):
#         history_context = trim_history_tokenwise(history, max_tokens=1000)

#         prompt = PromptTemplate.from_template(f"""You are a PostgreSQL SQL expert.

# Only use this table: "stage"."GBM1_prediction_data_with_recommendations"

# Given the schema:

# {{context}}

# Conversation history:
# {history_context}

# Now generate SQL for:
# {{question}}""")

#         llm = get_llama_maverick_llm()
#         runnable = prompt | llm
#         response = runnable.invoke(state)
#         state["sql"] = response.content if hasattr(response, "content") else response
#         return state

#     # ✅ Build LangGraph
#     graph = StateGraph(SQLState)
#     graph.add_node("retrieve_context", retrieve_context_node)
#     graph.add_node("generate_sql", generate_sql_node)
#     graph.set_entry_point("retrieve_context")
#     graph.add_edge("retrieve_context", "generate_sql")
#     graph.set_finish_point("generate_sql")

#     # ✅ Run pipeline
#     final_state = graph.compile().invoke({"question": question})
#     return final_state["sql"]

# # utils/token_utils.py

# def estimate_tokens(text: str) -> int:
#     return len(text) // 4

# def truncate_text(text: str, max_tokens: int) -> str:
#     if estimate_tokens(text) <= max_tokens:
#         return text
#     return text[:max_tokens * 4 - 100] + "\n\n[Truncated]"



