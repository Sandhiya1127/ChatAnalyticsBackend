# import json
# from django.views.decorators.csrf import csrf_exempt
# from django.http import JsonResponse
# # from langchain.vectorstores import Chroma
# # from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import CrossEncoder
# # Load or initialize your vector store
# # embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device": "cpu"})


# # Initialize models
# # Initialize models
# try:
#     embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5",model_kwargs={"device": "cpu"})

#     # embedding_model = OpenAIEmbeddings(
#     #     model="text-embedding-3-large",
#     #     openai_api_key=OPENROUTER_API_KEY
#     # )
#     rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
# except Exception as e:
#     # logger.error(f"Failed to initialize models: {e}")
#     embedding_model = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-en-v1.5",
#     model_kwargs={"device_map": "cpu", "torch_dtype": "float32"}
# )
#     rerank_model = None

# vector_db = Chroma(persist_directory="db/chroma_feedback", embedding_function=embedding_model)

# @csrf_exempt
# def feedback_view(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "Invalid method"}, status=405)

#     try:
#         data = json.loads(request.body)
#         feedback = data.get("feedback")
#         question = data.get("question", "")
#         answer = data.get("answer", "")
#         sql = data.get("sql", "")
#         summary = data.get("summary", "")
#         recommendation = data.get("recommendation", "")
#         session_id = data.get("session_id", "")

#         if feedback == "yes":
#             print(f"✅ Storing feedback for question: {question}")
#             vector_db.add_texts(
#                 texts=[question],
#                 metadatas=[{
#                     "answer": answer,
#                     "sql": sql,
#                     "summary": summary,
#                     "recommendation": recommendation,
#                     "session_id": session_id,
#                     "feedback": "positive"
#                 }]
#             )
#             return JsonResponse({"status": "stored", "success": True})

#         elif feedback == "no":
#             print(f"🔁 Triggering re-generation for question: {question}")
#             return JsonResponse({
#                 "status": "regenerating",
#                 "success": True,
#                 "message": "Re-run the query on frontend"
#             })

#         return JsonResponse({"error": "Invalid feedback option"}, status=400)

#     except Exception as e:
#         print("❌ Error in feedback_view:", str(e))
#         return JsonResponse({"error": str(e)}, status=500)



import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
# from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings   # ✅ keep only this
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain.schema import Document
# from .utils.stream import store_general_feedback
from django.utils.timezone import now
from sqlalchemy import text
from .utils.stream import jsonl_line

# Initialize models
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"}   # ✅ correct way
    )
    rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

except Exception as e:
    print(f"⚠️ Falling back due to error: {e}")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}
    )
    rerank_model = None


# Vector DB
vector_db = Chroma(
    # persist_directory="db/chroma_feedback",
    persist_directory="chroma/chroma_feedback",
    embedding_function=embedding_model
)

general_db = Chroma(  # General questions (ask_qwen)
    persist_directory="chroma_general_db/chroma_general",
    embedding_function=embedding_model
)
# ✅ Helper for normalization
def normalize_question(q: str) -> str:
    """Lowercase, strip, collapse spaces, remove trailing punctuation."""
    if not q:
        return ""
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)         # collapse multiple spaces
    q = re.sub(r"[?!.;,]+$", "", q)    # remove trailing ? . , etc.
    return q

# =====================================
# 🔹 Store General Feedback
# =====================================
def store_general_feedback(question, answer, session_id, feedback="auto"):
    try:
        q_norm = normalize_question(question)
        doc_id = f"{session_id}:{q_norm}"
        metadata = {
            "answer": answer,
            "session_id": session_id,
            "feedback": feedback,
            "doc_id": doc_id,
            "timestamp": now().isoformat()
        }
        general_db.add_texts(
            texts=[q_norm],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(f"💾 Stored/updated GENERAL doc_id={doc_id} (feedback={feedback})")
    except Exception as e:
        print(f"⚠️ Could not store in General DB: {e}")

# =====================================
# 🔹 Feedback View
# =====================================
@csrf_exempt
def feedback_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    try:
        data = json.loads(request.body)
        feedback = data.get("feedback")
        question = data.get("question", "")
        answer = data.get("answer", "")
        sql = data.get("sql", "")
        summary = data.get("summary", "")
        recommendation = data.get("recommendation", "")
        session_id = data.get("session_id", "")
        is_general = data.get("is_general", False)
        source = data.get("source", "")

        if not question:
            return JsonResponse({"error": "Missing question"}, status=400)

        # ✅ Normalize the question for dedup consistency
        q_norm = normalize_question(question)
        doc_id = f"{session_id}:{q_norm}"

        # ✅ Explicit rule: ask_qwen responses go to general DB
        if source == "ask_qwen":
            is_general = True

        # ✅ Fallback: no SQL/summary/recommendation → treat as general
        if not is_general and not sql and not summary and not recommendation:
            is_general = True

        if is_general:
            store_general_feedback(
                question=q_norm,
                answer=answer,
                session_id=session_id,
                feedback="positive" if feedback == "yes" else "negative"
            )
            return JsonResponse({
                "status": "stored/updated",
                "db": "general",
                "success": True
            })

        # Else → SQL feedback DB
        metadata = {
            "answer": answer,
            "sql": sql,
            "summary": summary,
            "recommendation": recommendation,
            "session_id": session_id,
            "feedback": "positive" if feedback == "yes" else "negative",
            "doc_id": doc_id,
            "timestamp": now().isoformat()
        }

        vector_db.add_texts(
            texts=[q_norm],
            metadatas=[metadata],
            ids=[doc_id]
        )

        if feedback == "yes":
            print(f"✅ Stored SQL {doc_id} as positive")
            return JsonResponse({"status": "stored/updated", "db": "sql_feedback", "success": True})
        elif feedback == "no":
            print(f"🔴 Stored SQL {doc_id} as negative")
            return JsonResponse({
                "status": "stored/updated",
                "db": "sql_feedback",
                "success": True,
                "message": "Re-run the query on frontend"
            })

        return JsonResponse({"error": "Invalid feedback option"}, status=400)

    except Exception as e:
        print("❌ Error in feedback_view:", str(e))
        return JsonResponse({"error": str(e)}, status=500)

def store_general_feedback159(question, answer, session_id, feedback="auto"):
    """Store Q/A for general questions into a separate Chroma DB"""
    try:
        doc_id = f"{session_id}:{question.lower().strip()}"
        metadata = {
            "answer": answer,
            "session_id": session_id,
            "feedback": feedback,
            "doc_id": doc_id,
            "timestamp": now().isoformat()
        }
        general_db.add_texts(
            texts=[question],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(f"💾 Stored/updated GENERAL doc_id={doc_id} (feedback={feedback})")
    except Exception as e:
        print(f"⚠️ Could not store in General DB: {e}")
# =====================================
# 🔹 Feedback View (SQL + General)
# =====================================
@csrf_exempt
def feedback_view159(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    try:
        data = json.loads(request.body)
        feedback = data.get("feedback")   # yes / no
        question = data.get("question", "")
        answer = data.get("answer", "")
        sql = data.get("sql", "")
        summary = data.get("summary", "")
        recommendation = data.get("recommendation", "")
        session_id = data.get("session_id", "")
        is_general = data.get("is_general", False)  # 🔹 flag for general feedback

        if not question:
            return JsonResponse({"error": "Missing question"}, status=400)

        doc_id = f"{session_id}:{question.lower().strip()}"

        if is_general:
            # store in GENERAL DB
            store_general_feedback(
                question,
                answer,
                session_id,
                feedback="positive" if feedback == "yes" else "negative"
            )
            return JsonResponse({"status": "stored/updated", "db": "general", "success": True})

        else:
            # store in SQL feedback DB
            metadata = {
                "answer": answer,
                "sql": sql,
                "summary": summary,
                "recommendation": recommendation,
                "session_id": session_id,
                "feedback": "positive" if feedback == "yes" else "negative",
                "doc_id": doc_id,
                "timestamp": now().isoformat()
            }

            vector_db.add_texts(
                texts=[question],
                metadatas=[metadata],
                ids=[doc_id]
            )

            if feedback == "yes":
                print(f"✅ Marked SQL {doc_id} as positive")
                return JsonResponse({"status": "stored/updated", "db": "sql_feedback", "success": True})
            elif feedback == "no":
                print(f"🔴 Marked SQL {doc_id} as negative")
                return JsonResponse({
                    "status": "regenerating",
                    "db": "sql_feedback",
                    "success": True,
                    "message": "Re-run the query on frontend"
                })

        return JsonResponse({"error": "Invalid feedback option"}, status=400)

    except Exception as e:
        print("❌ Error in feedback_view:", str(e))
        return JsonResponse({"error": str(e)}, status=500)

#workingcode bfr generaldb
@csrf_exempt
def feedback_view11(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    try:
        from langchain.schema import Document
        data = json.loads(request.body)
        feedback = data.get("feedback")
        question = data.get("question", "")
        answer = data.get("answer", "")
        sql = data.get("sql", "")
        summary = data.get("summary", "")
        recommendation = data.get("recommendation", "")
        session_id = data.get("session_id", "")

        if not question:
            return JsonResponse({"error": "Missing question"}, status=400)

        doc_id = f"{session_id}:{question.lower().strip()}"
        metadata = {
            "answer": answer,
            "sql": sql,
            "summary": summary,
            "recommendation": recommendation,
            "session_id": session_id,
            "feedback": "positive" if feedback == "yes" else "negative",
            "doc_id": doc_id
        }

        if feedback == "yes":
            vector_db.add_texts(
                texts=[question],
                metadatas=[metadata],
                ids=[doc_id]
            )
            print(f"✅ Marked {doc_id} as positive")
            return JsonResponse({"status": "stored/updated", "success": True})

        elif feedback == "no":
            vector_db.add_texts(
                texts=[question],
                metadatas=[metadata],
                ids=[doc_id]
            )
            print(f"🔴 Marked {doc_id} as negative")
            return JsonResponse({
                "status": "regenerating",
                "success": True,
                "message": "Re-run the query on frontend"
            })

        return JsonResponse({"error": "Invalid feedback option"}, status=400)

    except Exception as e:
        print("❌ Error in feedback_view:", str(e))
        return JsonResponse({"error": str(e)}, status=500)


# @csrf_exempt
# def feedback_view(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "Invalid method"}, status=405)

#     try:
#         data = json.loads(request.body)
#         feedback = data.get("feedback")
#         question = data.get("question", "")
#         answer = data.get("answer", "")
#         sql = data.get("sql", "")
#         summary = data.get("summary", "")
#         recommendation = data.get("recommendation", "")
#         session_id = data.get("session_id", "")

#         if feedback == "yes":
#             print(f"✅ Storing feedback for question: {question}")
#             vector_db.add_texts(
#                 texts=[question],
#                 metadatas=[{
#                     "answer": answer,
#                     "sql": sql,
#                     "summary": summary,
#                     "recommendation": recommendation,
#                     "session_id": session_id,
#                     "feedback": "positive"
#                 }]
#             )
#             return JsonResponse({"status": "stored", "success": True})

#         elif feedback == "no":
#             print(f"🔁 Triggering re-generation for question: {question}")
#             return JsonResponse({
#                 "status": "regenerating",
#                 "success": True,
#                 "message": "Re-run the query on frontend"
#             })

#         return JsonResponse({"error": "Invalid feedback option"}, status=400)

#     except Exception as e:
#         print("❌ Error in feedback_view:", str(e))

#         return JsonResponse({"error": str(e)}, status=500)


#without doc id
# @csrf_exempt
# def feedback_view(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "Invalid method"}, status=405)

#     try:
#         data = json.loads(request.body)
#         feedback = data.get("feedback")
#         question = data.get("question", "")
#         answer = data.get("answer", "")
#         sql = data.get("sql", "")
#         summary = data.get("summary", "")
#         recommendation = data.get("recommendation", "")
#         session_id = data.get("session_id", "")

#         if not question:
#             return JsonResponse({"error": "Missing question"}, status=400)

#         # ✅ Check if this question already exists in vector DB
#         try:
#             matches = vector_db.similarity_search(question, k=1)
#         except Exception as e:
#             print(f"⚠️ Search failed: {e}")
#             matches = []

#         if feedback == "yes":
#             if matches and matches[0].page_content.strip().lower() == question.strip().lower():
#                 # ✅ Update existing metadata (positive feedback)
#                 doc_id = matches[0].metadata.get("doc_id")
#                 if doc_id:
#                     vector_db.update_document(
#                         ids=[doc_id],
#                         documents=[Document(
#                             page_content=question,
#                             metadata={
#                                 "answer": answer,
#                                 "sql": sql,
#                                 "summary": summary,
#                                 "recommendation": recommendation,
#                                 "session_id": session_id,
#                                 "feedback": "positive"
#                             }
#                         )]
#                     )
#                     print(f"🔄 Updated existing doc {doc_id} with positive feedback")
#                 else:
#                     # fallback to add if no id
#                     vector_db.add_texts(
#                         texts=[question],
#                         metadatas=[{
#                             "answer": answer,
#                             "sql": sql,
#                             "summary": summary,
#                             "recommendation": recommendation,
#                             "session_id": session_id,
#                             "feedback": "positive"
#                         }]
#                     )
#                     print("➕ Inserted new doc (no ID found in existing)")
#             else:
#                 # ➕ Insert new doc if not found
#                 vector_db.add_texts(
#                     texts=[question],
#                     metadatas=[{
#                         "answer": answer,
#                         "sql": sql,
#                         "summary": summary,
#                         "recommendation": recommendation,
#                         "session_id": session_id,
#                         "feedback": "positive"
#                     }]
#                 )
#                 print("➕ Stored new doc with positive feedback")
#             return JsonResponse({"status": "stored/updated", "success": True})

#         elif feedback == "no":
#             if matches:
#                 doc_id = matches[0].metadata.get("doc_id")
#                 if doc_id:
#                     # ✅ Mark as negative
#                     vector_db.update_document(
#                         ids=[doc_id],
#                         documents=[Document(
#                             page_content=question,
#                             metadata={
#                                 "answer": answer,
#                                 "sql": sql,
#                                 "summary": summary,
#                                 "recommendation": recommendation,
#                                 "session_id": session_id,
#                                 "feedback": "negative"
#                             }
#                         )]
#                     )
#                     print(f"🔴 Marked doc {doc_id} as negative")
#                 else:
#                     print("⚠️ Could not update doc (no ID), keeping old one")
#             # Trigger regeneration on frontend
#             return JsonResponse({
#                 "status": "regenerating",
#                 "success": True,
#                 "message": "Re-run the query on frontend"
#             })

#         return JsonResponse({"error": "Invalid feedback option"}, status=400)

#     except Exception as e:
#         print("❌ Error in feedback_view:", str(e))
#         return JsonResponse({"error": str(e)}, status=500)
