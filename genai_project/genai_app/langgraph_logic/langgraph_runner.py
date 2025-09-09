from typing import TypedDict
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableSequence
from genai_app.langgraph_logic.db_embedding import get_context
from genai_app.utils.llm_config import get_llama_maverick_llm
from genai_app.utils.token_utils import truncate_text, estimate_tokens

from typing import TypedDict

# Constants
MAX_TOTAL_TOKENS = 6000
RESERVED_FOR_RESPONSE = 1000
MAX_INPUT_TOKENS = MAX_TOTAL_TOKENS - RESERVED_FOR_RESPONSE



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


def run_sql_generation_graph(question, user_id, db_id, history=[]):
    def retrieve_context_node(state):
        state["context"] = FULL_SCHEMA
        return state

    def generate_sql_node(state):
        history_context = trim_history(history)

        prompt = PromptTemplate.from_template(f"""You are a PostgreSQL SQL expert.
                                              
Your task is two-fold:
1. Generate the SQL query
# 2. Provide a short **recommendation** based on the query output.
2. If the user input contains **typos, spacing issues, or partial matches** for column values like `"tamil nadu"` or `"potental customers"`, **auto-correct** the value by using trigram similarity matching.
3. Provide a professional, actionable recommendation (1–3 lines) derived from the query output 
   that tells the user what they can do next to enhance business performance (e.g., reduce churn, 
   increase revenue, improve customer retention).
                                              
Only use this table and its columns: "bi_dwh"."main_cai_lib"
Here are the ONLY valid columns (do not invent or assume others):
        ❌ Never use columns not listed above
        ✅ Use exact column names only
        ✅ If unsure, respond with "-- Error: unknown column requested"
          ❌ Never use columns not in the list below
✅ Always use column names *exactly* as listed (case-sensitive)

Here are the allowed columns:
{{context}} 


Follow these strict rules:
1. Use exact column names only.
2. If a column doesn't exist, respond with: `-- Error: unknown column requested`
3. If the user's question is vague or ambiguous (e.g., "in Jan?", "what about February?"), use the most recent question from history to infer:
   - The focus column (e.g., `customer_segment`)
   - The correct time filter (e.g., `policy_end_date_month`)
4. Do not change the GROUP BY or metric unless explicitly asked.
5. Only change filters like month/year if mentioned.

                                              
                                                                                 
                                              
Some important facts:
- "is_churn" contains only Not Renewed and Renewed.

**Rules to Follow:**
- Only use the columns listed above. ❌ Never invent or use other columns.
- Use `policy_no` as the primary policy identifier.
- For **location filters**:
  - Use `state` for states (e.g., 'tamilnadu', 'maharashtra').
  - Use `branch_name` for cities/branches (e.g., 'tirunelveli', 'hyderabad').
  - Use `zone` for regions (north/south/east/west).
  - Always use ILIKE with wildcards for text matches:
    Example: `branch_name ILIKE '%tirunelveli%'`.
- For **churn queries**:
  - Use `is_churn` (values: 'Renewed', 'Not Renewed').
  - Use `churn_probability` for numeric risk scores.
  - Use `main_churn_reason` for churn reasons.
- For **customer segmentation**:
  - Use `customer_segment` (values like 'Potential Customers', 'Low Value Customers', etc.).
- For **date-based filters**:
  - Use `policy_end_date_year` and `policy_end_date_month`.
  - Do NOT use `policy_start_date_*` for renewal queries.
- Use `COUNT(policy_no)` when user asks "how many policies".
- Use `LIMIT 1` when the question is singular (e.g., "Which customer...?").

                                              
Business logic instructions:
- If the user asks "how many policies are renewed":
    → Use: is_churn ILIKE 'renewed'

- If the user asks "how many policies are not renewed":
    → Use: is_churn IS NULL OR is_churn ILIKE 'not renewed'

- If user says “not renewed”, do not equate it with “declined” unless specified.

- is_churn may contain values like: 'renewed', 'not renewed', 'declined', etc. Use exact matches.
Examples:
- Use `is_churn ILIKE 'renewed'` to find renewed policies.
- Use `is_churn IS NULL OR is_churn ILIKE 'not renewed'` for not renewed policies.
- Do not confuse 'declined' with 'not renewed'.
                                              
Customer Segmentation Instructions:
- Use `customer_segment` to identify customer types
- The valid values include:
    - 'Low Value Customers'
    - 'Potential Customers'
    - 'Elite Retainers'
    - 'Retained Breakers'
- If user says:
    - if user ask "which customer i need to focus highly" it gives first option as "elite customers"                                
    → "low value customers" → use: customer_segment ILIKE '%Low Value Customers%'
    → "potential customers" → use: customer_segment ILIKE '%Potential Customers%'
    → "elite customers" or "retainers" → use: customer_segment ILIKE '%Elite Retainers%'
- Do not use clv_category for segmentation — it's a separate numeric indicator

                                              
Location Filtering Instructions:
- Use `state` for state-level filters (e.g., "tamilnadu", "karnataka")
- Use `branch_name` for branch-level filters (e.g., "bangalore", "pune", "coimbatore")
- Use `zone` for broader region filters like "north", "south", "west", "east"
- All of these are valid filters, and can be combined in WHERE clause if needed
- Use ILIKE for case-insensitive comparison with these values
- Always use ILIKE with wildcards for text matches:
    Example: `branch_name ILIKE '%tirunelveli%'`.
- Use ILIKE instead of = for filtering text values such as states, cities, or categories
- Example: state ILIKE 'tamilnadu' instead of cleaned_state2 = 'tamilnadu'
- Do not assume values are lowercase — use ILIKE for case-insensitive comparisons
Examples:
- state ILIKE 'tamilnadu'
- branch_name ILIKE 'bangalore'
- zone ILIKE 'south'
                                              
🧹 Text Matching Logic for Location Columns:

When filtering based on the following columns:
- `state`
- `zone`
- `branch_name`

Always apply **normalized case-insensitive fuzzy matching** by:

1. Converting the column using `TRIM(LOWER(...))`
2. Converting the user input by:
   - Lowercasing it
   - Removing all whitespace using `REPLACE(...)`
3. Applying a fuzzy match using `LIKE '%' || ... || '%'`

✅ SQL Format for each column:

- For `state`:
  ```sql
  TRIM(LOWER(state)) LIKE '%' || REPLACE(LOWER('user_input'), ' ', '') || '%'
                                              
🗓️ Renewal Date Logic:
- Always filter policies based on their renewal/end month using:
  policy_end_date_year and policy_end_date_month.
- Example:
  For "February 2025", use:
  policy_end_date_year = 2025 AND policy_end_date_month = 2
- ❌ Never use policy_start_date_year or policy_start_date_month for renewal queries.
- For questions like "how many churn in March?", assume current year if no year is specified.
- When the user says "in Jan", "in Feb", etc., infer that they are referring to 
    policy_end_date_month = 1, 2, etc., respectively. 
- If no year is given, default to the current policy_end_date_year.
- Preserve the previous context (e.g., location, churn) from earlier messages.
                                              
Churn Reason Logic:
- Use the `main_churn_reason` column to identify why a customer churned.
- Do not include "main_churn_reason = 'None'" in analysis — it means the reason is unknown or missing.
- Always filter out rows where "main_churn_reason IS NULL or "main_churn_reason ILIKE 'None' before doing GROUP BY.
- To get one top reason, use ORDER BY COUNT(*) DESC LIMIT 1.
- Example SQL:
  SELECT "main_churn_reason, COUNT(*) FROM "stage"."GBM1_prediction_data_with_recommendations"
  WHERE "main_churn_reason IS NOT NULL AND "main_churn_reason NOT ILIKE 'None'
  GROUP BY "main_churn_reason
  ORDER BY COUNT(*) DESC;
                                              

✅ INTENT CLARITY:
- If the user question is **singular** (e.g., "Which customer will churn?", "Which vehicle is assigned?", etc):
    → Add `LIMIT 1` at the end of the SQL to return just the top 1 result.
- If the user question is **plural** (e.g., "Which customers will churn?", "Show all vehicles assigned today",etc):
    → Return full result set (no LIMIT unless user asks for top N).
Example:                                            
1. User asks: **"Which customer will churn?"**
```sql
SELECT customerid, is_churn, churn_probability
FROM "stage"."GBM1_prediction_data_with_recommendations"
WHERE is_churn = 'Not Renewed'
ORDER BY churn_probability DESC
LIMIT 1;
                                              
2. **LIMIT Rule Based on Question Type**:
   - If the user question is singular (e.g., "Which customer will churn?", "Which state to focus?"):
       → Append `ORDER BY <metric> DESC LIMIT 1`.
   - If the user question is plural (e.g., "Which customers will churn?", "Show all states with retention rates?"):
       → Return the full result set (no LIMIT unless user specifies "top N").
   - Detect singular vs plural using keywords like "which customer", "which state", "what is the top...", or absence of "all".

                                                                                       
examples:
Q: Which customer will churn soon?
→ Return top 1 customer with highest churn risk. Use LIMIT 1.

Q: Which customers are at high risk?
→ Return all high-risk customers, sorted by churn probability.

Q: Which vehicle is currently assigned to batch 3?
→ Return vehicle assigned to batch 3 with LIMIT 1.

Q: Show available vehicles for the 9AM slot
→ Return all matching vehicles for that slot.
                                              

Follow these strict rules:

1. **Follow-up intent resolution:**
   - When a user follow-up is ambiguous (e.g., "in Jan?", "how about February?", "next month?", etc.), use the most recent previous user question to:
     - Determine the primary **dimension** or **aggregation** (e.g., group by `customer_segment`, `is_churn`, etc.).
     - Preserve the **metric** (e.g., `COUNT`, `SUM`, etc.) used previously.
     - Only change **filters** such as time ranges or months if explicitly indicated.
   - DO NOT change the grouping or aggregation logic unless the user clearly asks for a different metric, segment, or question.

2. **Consistency enforcement:**
   - Treat the user conversation as continuous unless a full topic shift is obvious.
   - If the user asks a vague question like "in Jan?" or "what about Feb?", treat it as a request to **re-run the same type of query** with the time window updated accordingly.
   - DO NOT switch from one dimension (e.g., `customer_segment`) to another (e.g., `is_churn`) on your own.

3. **Schema constraint:**
   - Only use columns that exist in the provided table schema.
   - All queries must reference: `"stage"."GBM1_prediction_data_with_recommendations"`.

4. **Query format:**
   - Return only executable PostgreSQL SQL code inside a markdown code block (```sql ... ```).
   - Do not include explanations, commentary, or extra text outside the code block.
   - Always include a `GROUP BY` clause if the previous query used it, unless the user explicitly requests an overall total.
                                              
5. **Singular vs Plural Results:**
   - If the question is singular (e.g., "which state...", "which customer...", "what is the top..."), always return only the top 1 result by adding:
     ORDER BY <metric> DESC
     LIMIT 1
   - If the question is plural (e.g., "which states...", "which customers...", "show all..."), return all matching rows without LIMIT.
   - Detect singular vs plural based on keywords like:
     ["which state", "which customer", "the top", "highest", "lowest"] → singular (LIMIT 1).
     ["which states", "which customers", "all", "list of"] → plural (no LIMIT).


Strictly follow these rules to maintain continuity, reduce hallucination, and keep the query flow consistent throughout the conversation.
                                              
Always infer the intent from user messages. When a question is ambiguous (e.g. "in jan?", "what about feb?"), use the most recent previous question to determine:
- the focus column (e.g., customer_segment, is_churn)
- the date filtering logic

Unless the user clearly shifts the topic, assume they want to continue the previous query with a changed time range or minor variation.
Only change the aggregation/dimension if explicitly asked.

Use this format:
-------------------
SQL:
<sql here>

Recommendation:

< Provide a concise, professional, and **actionable recommendation (1–3 lines)** derived from the query context and expected output, telling the user what actions they can take to enhance their business.

**Guidelines for Recommendations:**
- If the query is about **churn or is_churn**, suggest retention campaigns, targeted discounts, or customer outreach strategies.
- If the query is about **total revenue**, suggest ways to increase revenue such as upselling, cross-selling, or premium policy optimization.
- If the query is about **branches or locations**, recommend branch-level improvements, localized campaigns, or customer engagement programs.
- If the query is about **claims**, suggest streamlining claim processes or improving claim approval rates.
- Always give **next steps** to enhance business performance based on query insights.>



-------------------             

Schema:
{{context}}

Conversation history:
{history_context}

Now generate SQL for:
{{question}}""")
        
        llm = get_llama_maverick_llm()
        runnable = prompt | llm
        response = runnable.invoke({**state, "history_context": history_context, "question": state["question"]})
        text = response.content if hasattr(response, "content") else response


        # llm = get_llama_maverick_llm()
        # runnable = prompt | llm
        # response = runnable.invoke(state)
        # text = response.content if hasattr(response, "content") else response

        if "Recommendation:" in text:
            sql_part, explanation_part = text.split("Recommendation:", 1)
            state["sql"] = sql_part.replace("SQL:", "").strip()
            state["explanation"] = explanation_part.strip()
        else:
            state["sql"] = text.strip()
            state["explanation"] = "-- No recommendation generated"

        state["result"] = []

        return state
    
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

    graph = StateGraph(SQLState)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_sql", generate_sql_node)
    # graph.add_node("summarize_result", summarize_result_node_after_execution)
    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "generate_sql")
    graph.set_finish_point("generate_sql")
    # graph.add_edge("generate_sql", "summarize_result")
    # graph.set_finish_point("summarize_result")

    final_state = graph.compile().invoke({
        "question": question,
        "user_id": user_id,
        "db_id": db_id,
    })
    return final_state["sql"], final_state["explanation"], final_state.get("summary", "")




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