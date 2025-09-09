


# import os
# import time
# import logging
# from typing import Optional

# from azure.ai.inference import ChatCompletionsClient
# from azure.ai.inference.models import SystemMessage, UserMessage
# from azure.core.credentials import AzureKeyCredential
# from azure.core.exceptions import HttpResponseError, ServiceRequestError, ServiceResponseError

# logger = logging.getLogger(__name__)

# AZURE_ENDPOINT = os.getenv("AZURE_INFERENCE_ENDPOINT")  # e.g., https://genaiprochurn.services.ai.azure.com/models
# AZURE_API_KEY = os.getenv("AZURE_INFERENCE_API_KEY")
# AZURE_MODEL = os.getenv("AZURE_INFERENCE_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8-prochurn-demo")
# AZURE_API_VERSION = "2024-05-01-preview"
# MAX_PROMPT_TOKENS = int(os.getenv("AZURE_MAX_PROMPT_TOKENS", "120000"))  # 128k-context model -> ~120k safety

# _SYSTEM_PROMPT = os.getenv("AZURE_SYSTEM_PROMPT", "You are a helpful SQL assistant.")

# _client: Optional[ChatCompletionsClient] = None


# def _get_client() -> ChatCompletionsClient:
#     global _client
#     if _client is None:
#         if not AZURE_ENDPOINT or not AZURE_API_KEY:
#             logger.error("Azure credentials not configured")
#             raise RuntimeError("Set AZURE_INFERENCE_ENDPOINT and AZURE_INFERENCE_API_KEY.")
#         _client = ChatCompletionsClient(
#             endpoint=AZURE_ENDPOINT,
#             credential=AzureKeyCredential(AZURE_API_KEY),
#             api_version=AZURE_API_VERSION,
#         )
#     return _client

# def _yes_no(text: str) -> str:
#     t = (text or "").strip().upper()
#     if t.startswith("Y"):
#         return "YES"
#     if t.startswith("N"):
#         return "NO"
#     return "NO"





# from urllib.parse import quote_plus
# from sqlalchemy import create_engine

# def ensure_session(session_id: str, user_id: str = "admin"):
#     sess = session_store.get(session_id)
#     if sess and sess.get("engine"):
#         return sess["engine"]

#     encoded_pwd = quote_plus(POSTGRES_PASSWORD)
#     conn_str = f"postgresql://{POSTGRES_USER}:{encoded_pwd}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
#     engine = create_engine(conn_str, pool_pre_ping=True, pool_recycle=1800, future=True)

#     session_store[session_id] = {"engine": engine, "user_id": user_id}
#     conversation_memory_store.setdefault(session_id, [])

#     # Optional: first-time worker bootstrap for embeddings
#     try:
#         schema_text = extract_schema_from_sqlalchemy(engine)
#         embed_schema_for_user(user_id=user_id, db_id=session_id, schema_text=schema_text)
#     except Exception as _:
#         # If embedding already exists or you don't need it every time, ignore
#         pass

#     print(f"✅ ensure_session: created engine for {session_id}")
#     return engine








# def check_intent(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         question = data.get("question", "")

#         prompt = f"""
# Classify the user's intent strictly as YES or NO.

# If the question is a general greeting, chit-chat, or general knowledge (e.g. "hi", "hello", "how are you", "who is the PM of India"), respond with NO.

# If it is about querying the connected database schema or fetching data from tables, respond with YES.

# Question: "{question}"
# Only respond YES or NO.
# """

#         try:
#             response = requests.post(
#                 "https://openrouter.ai/api/v1/chat/completions",
#                 headers={
#                     "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#                     "Content-Type": "application/json"
#                 },
#                 json={
#                     # "model": "meta-llama/llama-4-maverick:free",
#                     "model": "qwen/qwen-2.5-72b-instruct:free",
#                     "messages": [
#                         {"role": "system", "content": "You are an intent classifier."},
#                         {"role": "user", "content": prompt}
#                     ]
#                 }
#             )
#             result = response.json()
#             answer = result["choices"][0]["message"]["content"].strip()
#             return JsonResponse({"answer": answer})

#         except Exception as e:
#             return JsonResponse({"answer": "No", "error": str(e)}, status=500)




# ++++++++++++++++AZURE code


# def call_llm_with_retry(prompt: str, max_retries: int = 3, base_delay: float = 1.0) -> str:
#     # Rough token estimate; keep margin below model context
#     estimated_tokens = int(len(prompt.split()) * 1.5)
#     logger.info(f"Estimated prompt tokens: {estimated_tokens}")
#     if estimated_tokens > MAX_PROMPT_TOKENS:
#         logger.error(f"Prompt too long: {estimated_tokens} tokens (max: {MAX_PROMPT_TOKENS})")
#         return "The prompt is too long for the current model. Please try with a shorter question."

#     logger.info(f"Starting Azure Inference call with {max_retries} max retries")

#     for attempt in range(max_retries):
#         try:
#             logger.info(f"Attempt {attempt + 1}/{max_retries}")
#             resp = _get_client().complete(
#                 messages=[
#                     SystemMessage(content=_SYSTEM_PROMPT),
#                     UserMessage(content=prompt),
#                 ],
#                 model=AZURE_MODEL,
#                 temperature=0.1,
#                 top_p=0.9,
#                 frequency_penalty=0.0,
#                 presence_penalty=0.0,
#                 max_tokens=4000,
#             )

#             if resp.choices:
#                 content = (resp.choices[0].message.content or "").strip()
#                 if content:
#                     return content
#             logger.warning("Empty response from Azure Inference; retrying if attempts remain.")

#         except HttpResponseError as e:
#             status = getattr(e, "status_code", None)
#             logger.warning(f"Azure HttpResponseError (status={status}): {e}")
#             if status in (429, 502, 503, 504) and attempt < max_retries - 1:
#                 time.sleep(base_delay * (2 ** attempt))
#                 continue
#             break  # non-retryable or retries exhausted

#         except (ServiceRequestError, ServiceResponseError, TimeoutError, ConnectionError) as e:
#             logger.warning(f"Transient Azure error: {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(base_delay * (2 ** attempt))
#                 continue
#             break

#         except Exception as e:
#             logger.warning(f"Retry {attempt + 1} failed: {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(base_delay * (2 ** attempt))
#                 continue
#             break

#     logger.error("All retry attempts failed")
#     return "I'm currently experiencing issues reaching the AI service. Please try again later."


# @csrf_exempt
# def connect_database(request):
#     try:
#         # Optional user_id from frontend, fallback to 'admin'
#         try:
#             data = json.loads(request.body.decode("utf-8") or '{}')
#         except json.JSONDecodeError:
#             data = {}

#         user_id = data.get("user_id", "admin")

#         # Encode password and form connection string
#         encoded_pwd = quote_plus(POSTGRES_PASSWORD)
#         conn_str = f"postgresql://{POSTGRES_USER}:{encoded_pwd}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
#         engine = create_engine(conn_str)

#         # Test DB connection
#         with engine.connect() as conn:
#             conn.execute(text("SELECT 1"))

#         # Extract schema and initialize session
#         schema_text = extract_schema_from_sqlalchemy(engine)
#         session_id = data.get("session_id", str(uuid.uuid4()))
#         session_store[session_id] = {"engine": engine, "user_id": user_id}
#         conversation_memory_store[session_id] = []
#         print(session_id, session_store[session_id],"Connect database session_id")

#         # Embed schema for this session
#         embed_schema_for_user(user_id=user_id, db_id=session_id, schema_text=schema_text)

#         return JsonResponse({
#             "message": "Connected and embedded successfully (static credentials)",
#             "session_id": session_id
#         })

#     except Exception as e:
#         print("❌ connect_database failed:", str(e))
#         traceback.print_exc()
#         return JsonResponse({"error": str(e)}, status=500)
    

# @csrf_exempt
# def check_intent(request):
#     if request.method != "POST":
#         return JsonResponse({"answer": "NO", "error": "Only POST allowed"}, status=405)

#     try:
#         data = json.loads(request.body or "{}")
#     except Exception:
#         return JsonResponse({"answer": "NO", "error": "Invalid JSON body"}, status=400)

#     question = (data.get("question") or "").strip()
#     if not question:
#         return JsonResponse({"answer": "NO"})

#     prompt = f"""
# Classify the user's intent strictly as YES or NO.

# If the question is a general greeting, chit-chat, or general knowledge (e.g., "hi", "hello", "how are you", "who is the PM of India"), respond with NO.

# If it is about querying the connected database schema or fetching data from tables, respond with YES.
# If the question is trying to analyze, query, or summarize tabular or structured data (e.g., Excel, PDF tables, database), respond with YES.

# Question: "{question}"
# Only respond YES or NO.
# """.strip()

#     try:
#         resp = _get_client().complete(
#             messages=[
#                 SystemMessage(content="You are an intent classifier. Reply only YES or NO."),
#                 UserMessage(content=prompt),
#             ],
#             model=AZURE_MODEL,
#             temperature=0.0,
#             top_p=1.0, 
#             max_tokens=3,
#         )
#         raw = resp.choices[0].message.content if resp.choices else ""
#         answer = _yes_no(raw)
#         return JsonResponse({"answer": answer})
#     except (HttpResponseError, ServiceRequestError, ServiceResponseError, TimeoutError, ConnectionError) as e:
#         logger.warning(f"Azure inference error: {e}")
#         return JsonResponse({"answer": "NO", "error": str(e)}, status=502)
#     except Exception as e:
#         logger.exception("check_intent failed")
#         return JsonResponse({"answer": "NO", "error": str(e)}, status=500)




# def _clean_model_text(text: str) -> str:
#     if not text:
#         return ""
#     # Strip SQL/code blocks and inline code
#     text = re.sub(r"```sql.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)
#     text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
#     text = re.sub(r"`[^`]+`", "", text)
#     # Collapse whitespace
#     text = re.sub(r"\s+", " ", text).strip()
#     return text


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

# @csrf_exempt
# def ask_question(request):
#     try:
#         start_time = now()

     
#         if request.method == "GET" and request.GET.get("export") == "true":
#             session_id = request.GET.get("session_id")
#             question = request.GET.get("question")

#             print("🔍 Export request received")
#             print(f"🔎 Query Session ID requested: {session_id}")
#             print(f"🔎 Question: {question}")
#             print("🔍 Current session_store keys:", list(session_store.keys()))

#             if not session_id:
#                 return JsonResponse({"error": "Missing session_id parameter"}, status=400)
#             if not question:
#                 return JsonResponse({"error": "Missing question parameter"}, status=400)

#              # >>> ADD THIS BLOCK <<<
#             if session_id and session_id not in session_store:
#                 try:
#                     ensure_session(session_id, user_id="export_user")
#                     print(f"✅ ensure_session: export path recovered session {session_id}")
#                 except Exception as rec_err:
#                     print(f"❌ ensure_session (export) failed: {rec_err}")
#                     return JsonResponse({
#                         "error": f"Session ID {session_id} could not be initialized for export"
#                     }, status=404)
#             # <<< END ADD >>>

          
#             session_data = None
#             actual_session_id = None
       
#             if session_id in session_store:
#                 session_data = session_store[session_id]
#                 actual_session_id = session_id
#                 print(f"✅ Found exact session match: {session_id}")
#             else:
#                 # 🔧 FIX 2: Fallback - look for any session with the same user query in history
#                 print(f"❌ Session {session_id} not found, searching in conversation history...")
#                 for stored_session_id, stored_data in session_store.items():
#                     history = conversation_memory_store.get(stored_session_id, [])
#                     for entry in history:
#                         if entry.get("question", "").strip().lower() == question.strip().lower():
#                             session_data = stored_data
#                             actual_session_id = stored_session_id
#                             print(f"✅ Found session via question match: {stored_session_id}")
#                             break
#                     if session_data:
#                         break

#             if not session_data:
#                 print(f"❌ No session found for question: {question}")
#                 print(f"🔍 Available sessions: {list(session_store.keys())}")
#                 return JsonResponse({
#                     "error": f"Session ID {session_id} not found. Please run the query first via chat.",
#                     "available_sessions": list(session_store.keys()),
#                     "debug_info": f"Searched for question: '{question}'"
#                 }, status=404)

#             try:
#                 user_id = session_data.get("user_id", "export_user")
#                 engine = session_data.get("engine")

#                 if not engine:
#                     return JsonResponse({"error": "Database engine not found in session"}, status=500)

#                 print(f"✅ Session found. User ID: {user_id}, Actual Session: {actual_session_id}")

#                 history = conversation_memory_store.get(actual_session_id, [])
#                 sql = None

#                 # 🔧 FIX 3: More flexible question matching
#                 for entry in reversed(history):
#                     stored_question = entry.get("question", "").strip().lower()
#                     search_question = question.strip().lower()
                    
#                     # Try exact match first, then partial match
#                     if stored_question == search_question or search_question in stored_question:
#                         sql = entry.get("sql")
#                         print(f"📋 Found SQL in history: {sql}")
#                         break

#                 if not sql:
#                     print("🔄 Generating new SQL for export...")
#                     try:
#                         raw_response = run_sql_generation_graph(question, user_id=user_id, db_id=actual_session_id, history=history)
#                         sql, _ = raw_response if isinstance(raw_response, tuple) else (extract_sql_block(raw_response), None)
#                         print(f"🆕 Generated SQL: {sql}")
#                     except Exception as gen_error:
#                         print(f"❌ SQL generation failed: {str(gen_error)}")
#                         return JsonResponse({"error": f"Failed to generate SQL: {str(gen_error)}"}, status=500)

#                 if not sql or not sql.strip():
#                     return JsonResponse({"error": "No SQL query generated"}, status=400)

#                 sql_lower = sql.strip().lower()
#                 if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
#                     return JsonResponse({"error": "Invalid SQL query type"}, status=400)

#                 print(f"🚀 Executing SQL query...")
#                 with engine.connect() as conn:
#                     result = conn.execute(text(sql))
#                     rows = [dict(row._mapping) for row in result]

#                 print(f"✅ Query executed successfully. Found {len(rows)} rows")

#                 if not rows:
#                     # 🔧 FIX 5: Return empty CSV instead of plain text
#                     response = HttpResponse(content_type='text/csv')
#                     response['Content-Disposition'] = 'attachment; filename="export_no_data.csv"'
#                     response['Access-Control-Allow-Origin'] = '*'
#                     response.write("No data found for this query")
#                     return response

              
#                 from urllib.parse import quote
#                 timestamp = now().strftime("%Y%m%d_%H%M%S")
#                 filename = f"export_{timestamp}.csv"
                
#                 response = HttpResponse(content_type='text/csv')
#                 response['Content-Disposition'] = f'attachment; filename="{filename}"'
#                 response['Access-Control-Allow-Origin'] = '*'
#                 response['Access-Control-Allow-Headers'] = 'Content-Type'
#                 response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'

#                 writer = csv.DictWriter(response, fieldnames=rows[0].keys())
#                 writer.writeheader()
#                 writer.writerows(rows)

#                 print(f"📁 CSV file created successfully with {len(rows)} rows")
#                 return response

#             except Exception as e:
#                 print(f"❌ Export execution error: {str(e)}")
#                 traceback.print_exc()
#                 return JsonResponse({
#                     "error": f"Export failed: {str(e)}",
#                     "details": "Check server logs for more information",
#                     "session_used": actual_session_id
#                 }, status=500)

#         elif request.method == "POST":
#             data = json.loads(request.body)
#             session_id = data.get("session_id")
#             question = data.get("question")
#             chart_config = None
#             user_id = data.get("user_id", "admin")

#             print(f"📨 POST request received")
#             print(f"🔎 Post request Session ID: {session_id}")
#             print(f"🔎 Question: {question}")
#             print(f"🔎 User ID: {user_id}")

#             if not all([session_id, question]):
#                 return JsonResponse({"error": "Missing session_id or question"}, status=400)

#             # 🔹 ADD: auto-recover session if missing (survives restarts / multi-workers)
#             try:
#                 engine = ensure_session(session_id, user_id=user_id)
#                 print(ensure_session, "post ensure_session")
#             except Exception as rec_err:
#                 print(f"❌ ensure_session failed: {rec_err}")
#                 return JsonResponse({
#                     "answer": "Session not found",
#                     "success": False,
#                     "error": f"Session ID {session_id} could not be initialized",
#                     "rows": [],
#                     "row_count": 0,
#                     "session_id": session_id,
#                     "response_time": "0.00s",
#                     "code": "SESSION_INIT_FAILED"
#                 }, status=404)

#             if session_id not in session_store:
#                 print(f"⚠️ Session {session_id} not found")

#             history = conversation_memory_store.get(session_id, [])
#             print(f"🧠 Running SQL generation with session_id={session_id}, user_id={user_id}")

#             try:
#                 # 🔧 FIX 7: Handle column validation gracefully
#                 try:
#                     VALID_COLUMNS = extract_columns_from_schema(FULL_SCHEMA)
#                     validation_enabled = True
#                 except NameError:
#                     print("⚠️ Column validation not available")
#                     validation_enabled = False

#                 raw_response = run_sql_generation_graph(question, user_id=user_id, db_id=session_id, history=history)

#                 if isinstance(raw_response, tuple) and len(raw_response) == 3:
#                     sql_raw, recommendation, summary = raw_response
#                 else:
#                     sql_raw, recommendation = raw_response if isinstance(raw_response, tuple) else (extract_sql_block(raw_response), None)
#                     summary = ""

#                 # if isinstance(raw_response, tuple):
#                 #     sql_raw, recommendation = raw_response
#                 # else:
#                 #     sql_raw, recommendation = raw_response, None

#                 sql = extract_sql_block(sql_raw)
#                 print("📝 Extracted SQL:", sql)
#                 # ✅ Fix spacing issues in LIMIT clauses (e.g., LIMIT1 → LIMIT 1)
#                 sql = re.sub(r'\bLIMIT(\d+)', r'LIMIT \1', sql, flags=re.IGNORECASE)


#                 if validation_enabled:
#                     invalid_cols = validate_sql_columns(sql, VALID_COLUMNS)
#                     if invalid_cols:
#                         return JsonResponse({
#                             "answer": f"Invalid columns in SQL: {', '.join(invalid_cols)}",
#                             "success": False,
#                             "query_used": sql,
#                             "rows": [],
#                             "row_count": 0,
#                             "session_id": session_id,
#                             "response_time": "0.00s"
#                         }, status=400)

#             except Exception as sql_gen_error:
#                 print(f"❌ SQL generation error: {str(sql_gen_error)}")
#                 return JsonResponse({
#                     "answer": "Failed to generate SQL query",
#                     "success": False,
#                     "error": str(sql_gen_error),
#                     "rows": [],
#                     "row_count": 0,
#                     "session_id": session_id,
#                     "response_time": "0.00s"
#                 }, status=500)

#             if not sql or not sql.strip().lower().startswith(("select", "with")):
#                 return JsonResponse({
#                     "answer": "Invalid or failed SQL generation",
#                     "success": False,
#                     "query_used": sql or "No SQL generated",
#                     "rows": [],
#                     "row_count": 0,
#                     "session_id": session_id,
#                     "response_time": "0.00s"
#                 }, status=500)

#             if session_id not in session_store:
#                 return JsonResponse({
#                     "answer": "Session not found",
#                     "success": False,
#                     "error": f"Session ID {session_id} not found in session_store",
#                     "rows": [],
#                     "row_count": 0,
#                     "session_id": session_id,
#                     "response_time": "0.00s"
#                 }, status=404)

#             engine = session_store[session_id]["engine"]

#             try:
#                 print(f"✅ Executing SQL on session: {session_id}")
#                 with engine.connect() as conn:
#                     result = conn.execute(text(sql))
#                     rows = [dict(row._mapping) for row in result]
#                 print(f"✅ SQL executed successfully. Row count: {len(rows)}")

#                 summary = generate_summary_from_rows(question, sql, rows)
#             except Exception as e:
#                 print("❌ SQL Execution Error:", str(e))
#                 traceback.print_exc()
#                 return JsonResponse({
#                     "answer": "SQL execution failed.",
#                     "success": False,
#                     "query_used": sql,
#                     "error": str(e),
#                     "rows": [],
#                     "row_count": 0,
#                     "response_time": "0.00s",
#                     "session_id": session_id
#                 }, status=500)

#             # chart_config = None 
#             answer = ""
#             if rows:
#                 if len(rows) > 50:
#                     answer = f"Found {len(rows)} results. Too many to display here - please download the full results using the download button."
#                 else:
#                     formatted_rows = [", ".join(str(v) for v in row.values()) for row in rows[:3]]
#                     answer = "\n".join(formatted_rows)
#                     if len(rows) > 3:
#                         answer += f"\n...and {len(rows) - 3} more rows."
#             else:
#                 answer = "No data found."


           
#             try:
#                     chart_config = llm_generate_chart_config(question, rows)
#                     print("📊 Generated chart config:", json.dumps(chart_config, indent=2))
                    
#                     # Validate chart config before sending
#                     if chart_config and isinstance(chart_config, dict):
#                         # Ensure required fields exist
#                         if 'series' not in chart_config or not chart_config['series']:
#                             print("⚠️ Invalid chart config - missing or empty series")
#                             chart_config = None
#                         else:
#                             # Validate each series has data
#                             valid_series = []
#                             for series in chart_config['series']:
#                                 if 'data' in series and series['data']:
#                                     valid_series.append(series)
                            
#                             if valid_series:
#                                 chart_config['series'] = valid_series
#                             else:
#                                 chart_config = None
                                
#             except Exception as chart_err:
#                     print("⚠️ Chart generation failed:", chart_err)
#                     chart_config = None


#             try:
#                 if not recommendation:
#                     recommendation = llm_generate_recommendation(question, rows)
#             except Exception as rec_err:
#                 print("⚠️ Recommendation generation failed:", rec_err)
#                 recommendation = "Could not generate recommendation at this time."

       
#             conversation_memory_store.setdefault(session_id, []).append({
#                 "question": question,
#                 "sql": sql,
#                 "row_count": len(rows),
#                 "timestamp": now().isoformat()
#             })

#             if session_id in session_store:
#                 session_store[session_id]["user_id"] = user_id

#             total_time = (now() - start_time).total_seconds()

#             return JsonResponse({
#                 "answer": answer,
#                 "success": True,
#                 "query_used": sql,
#                 "rows": rows,
#                 "summary": summary,
#                 "chart_config": chart_config,
#                 "row_count": len(rows),
#                 "recommendation": recommendation,
#                 "response_time": f"{total_time:.2f}s",
#                 "session_id": session_id,
#                 "history": conversation_memory_store[session_id]
#             })

#         elif request.method == "OPTIONS":
#             response = HttpResponse()
#             response['Access-Control-Allow-Origin'] = '*'
#             response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
#             response['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
#             return response

#         else:
#             return JsonResponse({"error": "Method not allowed. Use GET for export or POST for queries."}, status=405)

#     except Exception as e:
#             print("💥 Unexpected error in ask_question:")
#             traceback.print_exc()

#             # ✅ Don’t re-parse request.body here; it can raise again.
#             sid = None
#             try:
#                 if request.method == "GET":
#                     sid = request.GET.get("session_id")
#                 elif request.method == "POST":
#                     # use already-parsed 'data' if available
#                     if 'data' in locals() and isinstance(data, dict):
#                         sid = data.get("session_id")
#             except Exception:
#                 sid = None

#             return JsonResponse({
#                 "answer": "Something went wrong.",
#                 "success": False,
#                 "error": str(e),
#                 "rows": [],
#                 "row_count": 0,
#                 "response_time": "0.00s",
#                 "session_id": sid or "unknown"
#             }, status=500)


    # except Exception as e:
    #     print("💥 Unexpected error in ask_question:")
    #     traceback.print_exc()
    #     return JsonResponse({
    #         "answer": "Something went wrong.",
    #         "success": False,
    #         "error": str(e),
    #         "rows": [],
    #         "row_count": 0,
    #         "response_time": "0.00s",
    #         "session_id": request.GET.get("session_id") if request.method == "GET" else json.loads(request.body).get("session_id", "unknown") if request.method == "POST" else "unknown"
    #     }, status=500)
