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
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings   # ✅ keep only this
from sentence_transformers import CrossEncoder


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
    persist_directory="db/chroma_feedback",
    embedding_function=embedding_model
)


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

        if feedback == "yes":
            print(f"✅ Storing feedback for question: {question}")
            vector_db.add_texts(
                texts=[question],
                metadatas=[{
                    "answer": answer,
                    "sql": sql,
                    "summary": summary,
                    "recommendation": recommendation,
                    "session_id": session_id,
                    "feedback": "positive"
                }]
            )
            return JsonResponse({"status": "stored", "success": True})

        elif feedback == "no":
            print(f"🔁 Triggering re-generation for question: {question}")
            return JsonResponse({
                "status": "regenerating",
                "success": True,
                "message": "Re-run the query on frontend"
            })

        return JsonResponse({"error": "Invalid feedback option"}, status=400)

    except Exception as e:
        print("❌ Error in feedback_view:", str(e))
        return JsonResponse({"error": str(e)}, status=500)
