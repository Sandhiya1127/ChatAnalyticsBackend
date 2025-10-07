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
def get_fixed_date_context():
    """
    Always return 2024 as the reference year,
    but align month/quarter/half dynamically to system month.
    """
    # now_dt = datetime.now()
    now_dt = datetime.datetime.now()
    fixed_year = 2024
    month = now_dt.month

    # Quarter & Half
    quarter = (month - 1) // 3 + 1
    half = 1 if month <= 6 else 2

    # Quarter ranges
    quarter_start = datetime(fixed_year, 3 * (quarter - 1) + 1, 1)
    quarter_end = quarter_start + relativedelta(months=3)

    # Half-year ranges
    half_start = datetime(fixed_year, 1 if half == 1 else 7, 1)
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

def repair_generated_sql(
    raw_llm_output: Any,
    tables: List[str],
    schema_catalog: Dict[str, List[str]],
    join_info: Dict[str, Any]
) -> str:
    """
    Complete repair pipeline for LLM-generated SQL.
    Handles all common issues in one pass.
    """
    
    # Extract SQL from various formats
    if isinstance(raw_llm_output, tuple):
        sql = raw_llm_output[0] if raw_llm_output else ""
    elif hasattr(raw_llm_output, 'content'):
        sql = raw_llm_output.content
    else:
        sql = str(raw_llm_output)
    
    # Remove markdown fences
    sql = re.sub(r'```(?:sql)?\s*', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'```\s*$', '', sql)
    
    # Remove explanatory text
    lines = sql.split('\n')
    sql_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that look like explanations
        if line.lower().startswith(('note:', 'explanation:', 'this query', 'the above')):
            continue
        sql_lines.append(line)
    
    sql = '\n'.join(sql_lines)
    
    # Fix common syntax errors
    sql = sql.replace('EXTEXTRACT', 'EXTRACT')
    sql = sql.replace('FROM FROM', 'FROM')
    sql = re.sub(r'\bLIMIT(\d+)', r'LIMIT \1', sql, flags=re.IGNORECASE)
    
    # Remove rogue parentheses
    if sql.startswith('(SELECT') and not sql.endswith(')'):
        sql = sql[1:]
    
    # Fix missing aliases in COUNT/SUM/AVG
    sql = re.sub(r'\b(COUNT|SUM|AVG)\s*\([^)]+\)\s+AS\s*$', r'\1(\2) AS total', sql, flags=re.IGNORECASE)
    
    # Repair column references to match actual schema
    aliases = join_info['aliases']
    
    for table in tables:
        alias = aliases[table]
        cols = schema_catalog.get(table, [])
        
        # Fix common column name issues
        for col in cols:
            col_lower = col.lower()
            # Find variations in SQL and fix them
            pattern = r'\b' + alias + r'\.' + col_lower + r'\b'
            replacement = f'{alias}.{col}'
            sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
    
    # Ensure table names are properly quoted
    for table in tables:
        # If table isn't already quoted, quote it
        if table.startswith('"') and table.endswith('"'):
            continue
        # Extract schema and table name
        parts = table.split('.')
        if len(parts) == 2:
            schema, tbl = parts
            quoted = f'"{schema}"."{tbl}"'
            sql = sql.replace(table, quoted)
    
    # Fix GROUP BY with aliases (replace alias with expression)
    select_aliases = extract_select_aliases(sql)
    sql = replace_group_by_aliases(sql, select_aliases)
    
    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # Ensure space between GROUP BY and ORDER BY
    sql = re.sub(r'(GROUP\s+BY[^;]+?)(ORDER\s+BY)', r'\1 \2', sql, flags=re.IGNORECASE)
    
    # Add semicolon if missing
    if not sql.endswith(';'):
        sql += ';'
    
    return sql


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

from sqlalchemy import create_engine, inspect, text
from typing import Dict, List, Any, Optional

def build_schema_catalog(
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

def select_relevant_tables(
    question: str,
    schema_catalog: Dict[str, dict],
    similarity_threshold: float = 0.6,
    max_tables: int = 3
) -> List[str]:
    """
    Automatically select relevant tables based on question content,
    with fuzzy column matching and numeric detection.
    """

    question_lower = question.lower()
    table_scores = {}

    for full_table_name, table_info in schema_catalog.items():
        score = 0
        table_name = table_info.get("table", "").lower()

        # ---- Table name matching ----
        table_keywords = table_name.replace("_", " ").split()
        for kw in table_keywords:
            if kw in question_lower:
                score += 10

        # ---- Column-based scoring ----
        columns = table_info.get("columns", [])
        for col in columns:
            col_name = col.get("name", "").lower()
            col_type = col.get("type", "").lower()

            # Exact match
            if col_name in question_lower:
                score += 15

            # Partial keyword match
            col_keywords = col_name.replace("_", " ").split()
            for kw in col_keywords:
                if kw in question_lower:
                    score += 5

            # Fuzzy match
            close_matches = get_close_matches(question_lower, [col_name], n=1, cutoff=similarity_threshold)
            if close_matches:
                score += 7

            # Boost for numeric-like columns if question mentions amounts, charge-offs, rates, etc.
            numeric_keywords = ["amount", "charge", "balance", "fee", "rate", "price", "cost", "value"]
            if any(k in question_lower for k in numeric_keywords) and col_type in ("text", "varchar", "char") and any(k in col_name for k in numeric_keywords):
                score += 10

        # ---- Boost for common patterns ----
        if "status" in question_lower and "status" in table_name:
            score += 20
        if "income" in question_lower and any("income" in str(c.get("name", "")).lower() for c in columns):
            score += 20
        if "approved" in question_lower and "app" in table_name:
            score += 15
        if "loan" in question_lower and "loan" in table_name:
            score += 15

        if score > 0:
            table_scores[full_table_name] = score

    # ---- Sort tables by score and return top N ----
    sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
    selected = [table for table, _ in sorted_tables[:max_tables]]

    print(f"🎯 Question: {question}")
    print(f"📊 Selected tables: {selected}")

    return selected


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

def nuclear_repair_sql(
    raw_llm_output: str,
    tables: List[str],
    schema_catalog: Dict[str, Any] = None,
    join_info: Dict[str, Any] = None
) -> str:
    """
    NUCLEAR OPTION: Completely rebuild SQL from broken pieces.
    Handles even catastrophically malformed SQL.
    """
    sql = (raw_llm_output or "").strip()
    
    # Remove code fences
    if "```sql" in sql:
        sql = sql.split("```sql", 1)[1].split("```", 1)[0].strip()
    elif "```" in sql:
        sql = re.sub(r'```.*?```', '', sql, flags=re.DOTALL).strip()
    
    # Remove comments
    sql = "\n".join([ln for ln in sql.splitlines() if not ln.strip().startswith('--')])
    
    print(f"🔧 NUCLEAR REPAIR - Input SQL:\n{sql[:300]}...\n")
    
    # Check if it's completely broken (FROM/JOIN in wrong places)
    if re.search(r'EXTRACT\([^)]+\s+(?:FROM|AS|JOIN)\s+["\w]+\.', sql, re.IGNORECASE):
        print("⚠️ CATASTROPHIC ERROR: FROM/JOIN inside function - rebuilding from scratch")
        return rebuild_sql_from_scratch(sql, tables, schema_catalog, join_info)
    
    # If not catastrophic, try normal repair
    return standard_repair_sql(sql, tables, schema_catalog, join_info)


def rebuild_sql_from_scratch(
    broken_sql: str,
    tables: List[str],
    schema_catalog: Dict[str, Any] = None,
    join_info: Dict[str, Any] = None
) -> str:
    """
    When SQL is completely broken, rebuild it from scratch by extracting intent.
    """
    print("🏗️ REBUILDING SQL FROM SCRATCH")
    
    # Extract what we can understand from broken SQL
    has_extract_quarter = 'EXTRACT' in broken_sql.upper() and 'QUARTER' in broken_sql.upper()
    has_sum = 'SUM' in broken_sql.upper()
    has_count = 'COUNT' in broken_sql.upper()
    has_group_by = 'GROUP BY' in broken_sql.upper()
    has_order_by = 'ORDER BY' in broken_sql.upper()
    has_where = 'WHERE' in broken_sql.upper()
    
    # Find date columns
    date_cols = []
    amount_cols = []
    
    if schema_catalog and tables:
        for table in tables:
            for table_key, info in schema_catalog.items():
                if table in table_key or table_key in table:
                    for col in info.get('columns', []):
                        col_type = col.get('type', '').lower()
                        col_name = col['name']
                        
                        if any(dt in col_type for dt in ['date', 'timestamp']):
                            date_cols.append(col_name)
                        
                        if any(amt in col_name.lower() for amt in ['amount', 'total', 'sum', 'balance']):
                            amount_cols.append(col_name)
    
    # Build fresh SQL
    first_table = tables[0] if tables else "unknown_table"
    alias = first_table.split('.')[-1].strip('"')
    
    # SELECT clause
    select_parts = []
    
    if has_extract_quarter and date_cols:
        date_col = date_cols[0]
        select_parts.append(f'EXTRACT(QUARTER FROM "{alias}"."{date_col}") AS application_quarter')
    
    if has_sum and amount_cols:
        amount_col = amount_cols[0]
        # Check if it's TEXT type
        is_text = False
        if schema_catalog:
            for table_key, info in schema_catalog.items():
                if first_table in table_key:
                    for col in info.get('columns', []):
                        if col['name'] == amount_col:
                            col_type = col.get('type', '').lower()
                            if any(t in col_type for t in ['text', 'varchar', 'char']):
                                is_text = True
                            break
        
        if is_text:
            select_parts.append(f'COALESCE(SUM(CAST(NULLIF("{alias}"."{amount_col}", \'\') AS NUMERIC)), 0) AS total_amount')
        else:
            select_parts.append(f'COALESCE(SUM("{alias}"."{amount_col}"), 0) AS total_amount')
    
    if has_count and not has_sum:
        select_parts.append(f'COUNT(*) AS total_count')
    
    if not select_parts:
        select_parts.append('*')
    
    select_clause = "SELECT " + ", ".join(select_parts)
    
    # FROM clause
    from_clause = f'FROM {first_table} AS "{alias}"'
    
    # Add JOINs if available
    if join_info and len(tables) > 1:
        for join_key, join_cond in join_info.items():
            if isinstance(join_key, (list, tuple)) and len(join_key) >= 2:
                join_table = join_key[1]
                join_alias = join_table.split('.')[-1].strip('"')
                
                left_cols = join_cond.get('left_columns', [])
                right_cols = join_cond.get('right_columns', [])
                
                if left_cols and right_cols:
                    on_clause = " AND ".join([
                        f'"{alias}"."{l}" = "{join_alias}"."{r}"'
                        for l, r in zip(left_cols, right_cols)
                    ])
                    from_clause += f' INNER JOIN {join_table} AS "{join_alias}" ON {on_clause}'
    
    # WHERE clause
    where_parts = []
    
    # Extract WHERE conditions from broken SQL
    where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|;|$)', broken_sql, re.IGNORECASE | re.DOTALL)
    if where_match:
        where_text = where_match.group(1).strip()
        # Clean up any misplaced FROM/JOIN in WHERE
        where_text = re.sub(r'\s+(?:FROM|JOIN|AS)\s+["\w.]+', '', where_text, flags=re.IGNORECASE)
        
        # Extract simple conditions
        conditions = re.findall(r'(["\w.]+)\s*(IS NOT NULL|IS NULL|!=|<>|>=|<=|>|<|=)\s*(["\w\s\'-]+)', where_text, re.IGNORECASE)
        
        for col, op, val in conditions:
            col_clean = col.strip().strip('"')
            val_clean = val.strip()
            
            # Qualify column if needed
            if '.' not in col_clean:
                col_qualified = f'"{alias}"."{col_clean}"'
            else:
                col_qualified = col_clean
            
            where_parts.append(f"{col_qualified} {op} {val_clean}")
    
    # Add date range if we have date column
    if date_cols and has_where:
        today = datetime.date.today()
        # Last quarter
        if 'last' in broken_sql.lower() or '3 month' in broken_sql.lower():
            start_date = today - datetime.timedelta(days=90)
            where_parts.append(f'"{alias}"."{date_cols[0]}" >= DATE \'{start_date.strftime("%Y-%m-%d")}\'')
            where_parts.append(f'"{alias}"."{date_cols[0]}" <= DATE \'{today.strftime("%Y-%m-%d")}\'')
    
    where_clause = ""
    if where_parts:
        where_clause = "WHERE " + " AND ".join(where_parts)
    
    # GROUP BY clause
    group_by_clause = ""
    if has_group_by and has_extract_quarter and date_cols:
        group_by_clause = f'GROUP BY EXTRACT(QUARTER FROM "{alias}"."{date_cols[0]}")'
    
    # ORDER BY clause
    order_by_clause = ""
    if has_order_by:
        if has_extract_quarter:
            order_by_clause = "ORDER BY application_quarter"
        elif has_sum:
            order_by_clause = "ORDER BY total_amount DESC"
        elif has_count:
            order_by_clause = "ORDER BY total_count DESC"
    
    # Combine all parts
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


def standard_repair_sql(
    sql: str,
    tables: List[str],
    schema_catalog: Dict[str, Any] = None,
    join_info: Dict[str, Any] = None
) -> str:
    """
    Standard repair for SQL that's mostly correct.
    """
    # Fix quarters
    for i in range(1, 5):
        sql = re.sub(
            rf"INTERVAL\s+['\"]{i}\s+quarters?['\"]",
            f"INTERVAL '{i*3} months'",
            sql,
            flags=re.IGNORECASE
        )
    
    # Fix dates
    today = datetime.date.today()
    fixed_date = f"DATE '{today.year}-{today.month:02d}-{today.day:02d}'"
    sql = re.sub(r'\bCURRENT_DATE\b', fixed_date, sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNOW\(\)\b', fixed_date, sql, flags=re.IGNORECASE)
    
    # Fix NULL comparisons
    sql = re.sub(r'(\w+)\s*=\s*NULL', r'\1 IS NULL', sql, flags=re.IGNORECASE)
    sql = re.sub(r'(\w+)\s*!=\s*NULL', r'\1 IS NOT NULL', sql, flags=re.IGNORECASE)
    
    # Safe division
    def safe_div(m):
        left = m.group(1)
        right = m.group(2).strip()
        if 'NULLIF' in right:
            return m.group(0)
        return f"{left} / NULLIF({right}, 0)"
    
    sql = re.sub(r'([^\s]+)\s*/\s*([A-Za-z0-9_"\.\)]+)', safe_div, sql)
    
    # Cleanup
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # Balance parentheses
    open_p = sql.count('(')
    close_p = sql.count(')')
    if open_p > close_p:
        sql += ')' * (open_p - close_p)
    elif close_p > open_p:
        excess = close_p - open_p
        for _ in range(excess):
            sql = sql.rstrip(';').rstrip(')')
    
    if not sql.endswith(';'):
        sql += ';'
    
    return sql


# Main entry point - replace your repair_generated_sql with this
def repair_generated_sql(
    raw_llm_output: str,
    tables: List[str],
    schema_catalog: Dict[str, Any] = None,
    join_info: Dict[str, Any] = None
) -> str:
    """
    Main repair function - calls nuclear repair which handles everything.
    """
    return nuclear_repair_sql(raw_llm_output, tables, schema_catalog, join_info)

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



def build_dynamic_sql_prompt(
    question: str,
    schema_catalog: dict,
    tables: List[str] = None,
    join_info: dict = None,
    history: List[Dict] = None,
    similarity_threshold: float = 0.4,
    error_context: dict = None
) -> str:
    """
    Production-grade SQL generation prompt with self-learning capabilities.
    Handles all edge cases, data types, and business logic automatically.
    """
    
    # Auto-select tables if not provided
    if not tables:
        tables = select_relevant_tables(question, schema_catalog, similarity_threshold=similarity_threshold)
    
    # Ensure we have tables
    if not tables:
        tables = list(schema_catalog.keys())[:3]
    
    # Get columns for selected tables
    all_columns = []
    column_type_map = {}  # Track data types for intelligent handling
    for table in tables:
        table_info = schema_catalog.get(table, {})
        cols = table_info.get("columns", [])
        all_columns.extend(cols)
        for col in cols:
            column_type_map[col.get('name', '').lower()] = col.get('type', '').lower()
    
    # Detect patterns
    arithmetic_info = detect_arithmetic_patterns(question, all_columns, tables, schema_catalog)
    select_patterns = arithmetic_info["select"]
    group_by_cols = arithmetic_info["group_by"]
    
    # Analyze question intent
    question_lower = question.lower()
    is_percentage = any(word in question_lower for word in ['rate', 'percent', '%', 'ratio', 'proportion'])
    is_comparison = any(word in question_lower for word in ['compare', 'vs', 'versus', 'difference', 'higher', 'lower', 'more', 'less'])
    is_aggregation = any(word in question_lower for word in ['total', 'sum', 'count', 'average', 'avg', 'max', 'min', 'top', 'bottom'])
    is_temporal = any(word in question_lower for word in ['today', 'yesterday', 'last', 'recent', 'current', 'this month', 'this year', 'quarter'])
    is_ranking = any(word in question_lower for word in ['top', 'bottom', 'highest', 'lowest', 'best', 'worst', 'rank'])
    
    # Build schema section with enhanced metadata
    today = datetime.date.today()
    current_year = today.year
    schema_section = "# DATABASE SCHEMA\n\n"
    
    for table in tables:
        table_info = schema_catalog.get(table, {})
        # Extract year from table name for temporal context
        table_year = None
        year_match = re.search(r'_(\d{4})$', table)
        if year_match:
            table_year = year_match.group(1)
        
        schema_section += f"## Table: {table}"
        if table_year:
            schema_section += f" (Year: {table_year})"
        schema_section += "\nColumns:\n"
        
        for col in table_info.get("columns", []):
            col_name = col.get('name', '')
            col_type = col.get('type', '').lower()
            pk = " [PRIMARY KEY]" if col.get("is_primary") else ""
            
            # Enhanced type hints
            type_hint = ""
            if col_type in ['text', 'varchar', 'char', 'character varying']:
                if any(numeric_word in col_name.lower() for numeric_word in ['amount', 'price', 'cost', 'total', 'sum', 'balance', 'fee', 'rate', 'score']):
                    type_hint = " ⚠️ TEXT storing numbers - MUST USE: CAST(NULLIF(column, '') AS NUMERIC)"
                else:
                    type_hint = " [String - use ILIKE for search]"
            elif col_type in ['integer', 'bigint', 'smallint', 'numeric', 'decimal', 'float', 'double precision', 'real']:
                type_hint = " [Numeric - safe for arithmetic]"
            elif col_type in ['date', 'timestamp', 'timestamp without time zone', 'timestamp with time zone']:
                type_hint = " [Date/Time - use DATE/TIMESTAMP functions]"
            elif col_type == 'boolean':
                type_hint = " [Boolean - TRUE/FALSE]"
            
            schema_section += f"  - {col_name}: {col.get('type')}{pk}{type_hint}\n"
        schema_section += "\n"
    
    # Enhanced JOIN section with relationship context
    join_section = ""
    if join_info:
        join_section = "# TABLE RELATIONSHIPS\n\n"
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
        join_section += "\n"
    
    # Learning from history
    history_section = ""
    learned_patterns = []
    if history:
        history_section = "# QUERY LEARNING HISTORY\n"
        history_section += "Learn from these successful patterns:\n\n"
        
        for entry in history[-5:]:  # Last 5 queries
            q = entry.get('question', '')
            sql = entry.get('sql', '')
            success = entry.get('success', True)
            
            if success:
                history_section += f"✓ Q: {q}\n  SQL: {sql}\n\n"
                # Extract patterns
                if 'CAST(NULLIF' in sql:
                    learned_patterns.append("Use CAST(NULLIF(column, '') AS NUMERIC) for TEXT columns with numbers")
                if '/ NULLIF' in sql:
                    learned_patterns.append("Always protect division with NULLIF(divisor, 0)")
                if 'ILIKE' in sql:
                    learned_patterns.append("Use ILIKE for case-insensitive text matching")
    
    # Error recovery section
    error_section = ""
    if error_context:
        error_section = "# PREVIOUS ERROR - LEARN AND FIX\n\n"
        error_section += f"Previous attempt failed with: {error_context.get('error_type', 'Unknown error')}\n"
        error_section += f"Error message: {error_context.get('error_message', '')}\n"
        error_section += f"Failed SQL: {error_context.get('failed_sql', '')}\n\n"
        error_section += "Analyze the error and generate corrected SQL that avoids this mistake.\n\n"
    
    # Build the comprehensive prompt
    prompt = f"""You are an expert PostgreSQL database programmer with 15+ years of production experience.
You write bulletproof SQL that handles ALL edge cases automatically.

{schema_section}
{join_section}
{history_section}
{error_section}

# CORE PRINCIPLES (NEVER VIOLATE)

## 1. DATA TYPE INTELLIGENCE
CRITICAL: Analyze column types before ANY operation:

### TEXT Columns Storing Numbers
- ⚠️ TEXT/VARCHAR columns with numeric names (amount, price, cost, total, etc.) contain string numbers
- ALWAYS wrap in: CAST(NULLIF(column, '') AS NUMERIC)
- Example: amount > 100 → CAST(NULLIF(amount, '') AS NUMERIC) > 100
- Apply to: WHERE, CASE WHEN, HAVING, ORDER BY, arithmetic operations

### NULL/Empty Safety
- Check for NULL: column IS NULL (never column = NULL)
- Empty strings: NULLIF(column, '') converts '' to NULL
- Chain: CAST(NULLIF(TRIM(column), '') AS type)
- Default values: COALESCE(column, default_value)
- Aggregations: COALESCE(SUM(column), 0) - never return NULL for counts/sums

### Division Safety
- NEVER use: a / b
- ALWAYS use: a / NULLIF(b, 0)
- For percentages: (numerator::NUMERIC / NULLIF(denominator, 0)) * 100
- Cast integers before division: column::NUMERIC to avoid integer division

### Date Handling
- Current date: DATE '{current_year}-{today.month:02d}-{today.day:02d}'
- NEVER use INTERVAL '1 quarter' - use INTERVAL '3 months'
- NEVER use CURRENT_DATE or NOW() - use explicit dates
- Table year context: If table ends with _2024, filter by year 2024
- Date arithmetic: date_column + INTERVAL '3 months'
- Extract: EXTRACT(YEAR FROM date_column)

## 2. QUERY CONSTRUCTION RULES

### Column Qualification
- ALWAYS use: "schema"."table"."column" or "alias"."column"
- Never use bare column names in multi-table queries
- Consistent aliasing: table AS alias

### JOIN Strategy
- Understand relationships before joining
- Use appropriate JOIN type: INNER, LEFT, RIGHT, FULL OUTER
- ALWAYS specify ON conditions explicitly
- Multiple joins: chain logically from base table
- Avoid Cartesian products

### Aggregation Rules
- If using GROUP BY, all non-aggregated SELECT columns MUST be in GROUP BY
- Aggregate functions: COUNT, SUM, AVG, MAX, MIN
- Use COUNT(DISTINCT column) for unique counts
- Filter aggregates with HAVING (not WHERE)

### String Matching
- Case-insensitive: ILIKE '%pattern%'
- Exact match: column = 'value'
- Multiple patterns: column ILIKE '%a%' OR column ILIKE '%b%'
- NOT matching: column NOT ILIKE '%pattern%'

## 3. QUERY PATTERNS BY INTENT

### Percentage/Rate Calculations
When question asks for rate, %, proportion:
```sql
-- Pattern: (count_of_success / total_count) * 100
SELECT 
    group_column,
    (COUNT(CASE WHEN condition THEN 1 END)::NUMERIC / 
     NULLIF(COUNT(*), 0)) * 100 AS success_rate
FROM table
GROUP BY group_column
```

### Rankings (Top N, Bottom N)
```sql
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (ORDER BY metric DESC) as rank
    FROM table
) ranked
WHERE rank <= N
-- Or simply: ORDER BY metric DESC LIMIT N
```

### Comparisons (Higher, Lower, More, Less)
```sql
-- Use CASE WHEN for conditional logic
SELECT 
    column,
    CASE 
        WHEN value > threshold THEN 'High'
        WHEN value < threshold THEN 'Low'
        ELSE 'Medium'
    END AS category
FROM table
```

### Time-based Analysis
```sql
-- Last 3 months
WHERE date_column >= DATE '{current_year}-{today.month:02d}-{today.day:02d}' - INTERVAL '3 months'

-- This year
WHERE EXTRACT(YEAR FROM date_column) = {current_year}

-- Month comparison
WHERE EXTRACT(MONTH FROM date_column) = {today.month}
```

## 4. BUSINESS LOGIC AUTO-APPLICATION

### New Table Discovery
When encountering a new table:
1. Analyze column names for business meaning
2. Identify key metrics (amounts, counts, dates, statuses)
3. Infer relationships from column name patterns (id, _id suffixes)
4. Apply appropriate data type handling automatically

### Metric Inference
- "funded", "approved", "completed" → COUNT WHERE status conditions
- "amount", "total", "sum" → Use SUM with CAST if TEXT
- "rate", "percentage" → Calculate ratio * 100
- "average", "mean" → Use AVG with proper casting
- "highest", "top" → ORDER BY DESC LIMIT N
- "lowest", "bottom" → ORDER BY ASC LIMIT N

### Fuzzy Column Matching
If exact column doesn't exist:
- customer_id, customerid, customer_id → treat as same
- amount_funded, amountfunded, funded_amount → treat as same
- Use ILIKE for flexible matching

## 5. ERROR PREVENTION

### Common Pitfalls to AVOID
❌ Text vs number comparison without CAST
❌ Division by zero
❌ Unqualified columns in multi-table queries
❌ Missing GROUP BY for non-aggregated columns
❌ Using CURRENT_DATE (replace with explicit date)
❌ Integer division (5/2=2, not 2.5)
❌ Comparing to NULL with = or !=
❌ Forgetting DISTINCT for unique counts
❌ Schema not qualified
❌ Quotes inconsistent ("" vs '')

### Self-Validation Checklist
Before returning SQL, verify:
✓ All TEXT numeric columns wrapped in CAST(NULLIF(...))
✓ All divisions protected with NULLIF(divisor, 0)
✓ All columns fully qualified with schema/alias
✓ GROUP BY includes all non-aggregated columns
✓ Dates use explicit DATE 'YYYY-MM-DD' format
✓ JOINs have proper ON conditions
✓ Aggregations use COALESCE for NULL handling
✓ String matching uses ILIKE
✓ Syntax is valid PostgreSQL

## 6. LEARNED PATTERNS
"""
    
    if learned_patterns:
        prompt += "Apply these proven patterns:\n"
        for pattern in set(learned_patterns):
            prompt += f"- {pattern}\n"
        prompt += "\n"
    
    # Add question intent analysis
    prompt += f"""
# QUESTION ANALYSIS

Question: "{question}"

Detected Intent:
- Percentage/Rate calculation: {"YES" if is_percentage else "NO"}
- Comparison needed: {"YES" if is_comparison else "NO"}
- Aggregation required: {"YES" if is_aggregation else "NO"}
- Time-based query: {"YES" if is_temporal else "NO"}
- Ranking needed: {"YES" if is_ranking else "NO"}

Suggested Approach:
"""
    
    if is_percentage:
        prompt += "- Calculate ratio and multiply by 100 for percentage\n"
        prompt += "- Use CAST(numerator AS NUMERIC) / NULLIF(denominator, 0) * 100\n"
    
    if is_ranking:
        prompt += "- Use ORDER BY with LIMIT or ROW_NUMBER() window function\n"
    
    if is_aggregation:
        prompt += "- Use appropriate aggregate function (COUNT, SUM, AVG, MAX, MIN)\n"
        prompt += "- Remember GROUP BY for dimensions\n"
    
    if is_temporal:
        prompt += f"- Filter by date using DATE '{current_year}-{today.month:02d}-{today.day:02d}'\n"
        prompt += "- Consider table year suffix for filtering\n"
    
    if group_by_cols:
        prompt += f"- GROUP BY: {', '.join(group_by_cols)}\n"
    
    if select_patterns:
        prompt += f"- Consider SELECT: {', '.join(select_patterns[:5])}\n"
    
    # Final instructions
    prompt += f"""

# OUTPUT REQUIREMENTS

Generate a single, valid PostgreSQL query that:
1. Answers the question accurately
2. Handles ALL edge cases (NULL, empty strings, zero division)
3. Uses correct data types with proper casting
4. Follows all rules above without exception
5. Returns meaningful results ready for display

Return ONLY the SQL query - no explanations, no markdown, no comments.
The query must execute successfully on the first attempt.

SQL Query:
"""

    prompt = improve_sql_prompt_clarity(prompt)
    
    return prompt

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
# ============================================================================
# MAIN SQL GENERATION PIPELINE
# ============================================================================

def run_sql_generation_graph(
    question: str,
    user_id: str,
    session_id: str,
    history: List[Dict] = None
) -> Tuple[str, str, str]:
    """
    Generate SQL for a given question with guaranteed table selection.
    Returns: (sql, explanation, error_info)
    """
    try:
        print(f"🔍 SQL Generation for question: {question}")
        print(f"🔑 Session ID: {session_id}")

        # Import SessionManager dynamically to avoid circular imports
        
        # Fetch session data
        session_data = SessionManager.get_session(session_id)
        if not session_data:
            return ("-- ERROR: Session not found", "Connect to database first", "SESSION_ERROR")

        schema_catalog = session_data.get("schema_catalog", {})
        if not schema_catalog:
            return ("-- ERROR: No schema", "Schema catalog is empty", "SCHEMA_ERROR")

        print(f"📚 Schema catalog has {len(schema_catalog)} tables")

        # Step 1: Select relevant tables (GUARANTEED to return at least one)
        selected_tables = select_relevant_tables(
            question=question,
            schema_catalog=schema_catalog,
            similarity_threshold=0.4,
            max_tables=5
        )
        
        if not selected_tables:
            # ABSOLUTE FALLBACK: Use all tables
            selected_tables = list(schema_catalog.keys())[:3]
            print(f"⚠️ EMERGENCY FALLBACK: Using first {len(selected_tables)} tables")
        
        print(f"✅ Final selected tables: {selected_tables}")

        # Step 2: Discover join keys
        try:
            join_info = discover_join_keys(selected_tables, schema_catalog)
        except Exception as e:
            print(f"⚠️ Join discovery failed: {e}")
            join_info = {}

        # Step 3: Build prompt
        prompt = build_dynamic_sql_prompt(
            question=question,
            schema_catalog=schema_catalog,
            tables=selected_tables,
            join_info=join_info,
            history=history or [],
            similarity_threshold=0.4
        )

        print(f"📝 Prompt length: {len(prompt)} chars")

        # Step 4: Generate SQL using LLM
        try:
            from genai_app.utils.llm_config import get_llama_maverick_llm
            llm = get_llama_maverick_llm()
            if not llm:
                return ("-- ERROR: LLM not configured", "LLM function not available", "LLM_ERROR")
            
            response = llm.invoke(prompt)
        except Exception as e:
            print(f"❌ LLM invocation failed: {e}")
            traceback.print_exc()
            return (f"-- ERROR: LLM failed - {str(e)}", "LLM generation failed", "LLM_ERROR")

        # Step 5: Repair and finalize SQL
        sql = repair_generated_sql(
            raw_llm_output=response,
            tables=selected_tables,
            schema_catalog=schema_catalog,
            join_info=join_info
        )

        print(f"✅ Generated SQL Final:\n{sql}\n")
        return (sql, "SQL generated successfully", "")

    except Exception as e:
        print(f"❌ SQL generation error: {e}")
        traceback.print_exc()
        return (f"-- ERROR: {str(e)}", "SQL generation failed", "GENERATION_ERROR")

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



