# # narrative.py
# import os, json, requests

# def llm_generate_narrative(question, rows):
#     preview = rows[:20]
#     preview_str = "\n".join(", ".join(f"{k}: {v}" for k, v in r.items()) for r in preview)

#     system_prompt = (
#         "You are a business analyst. Produce a SHORT, executive-friendly narrative in STRICT JSON with keys:\n"
#         "opener (string), insights (string[]), recommendations (string[]), next_step (string).\n"
#         "No SQL, no markdown, no extra keys."
#     )
#     user_prompt = f"""Question: {question}

# First 20 rows:
# {preview_str}

# Return ONLY JSON with keys: opener, insights, recommendations, next_step.
# """.strip()

#     headers = {
#         "Authorization": f"Bearer {os.getenv('GROQ_API_KEY','')}",
#         "Content-Type": "application/json",
#     }
#     payload = {
#         "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         "temperature": 0.2,
#         "max_tokens": 600,
#     }

#     try:
#         resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
#                              headers=headers, json=payload, timeout=45)
#         resp.raise_for_status()
#         txt = resp.json()["choices"][0]["message"]["content"].strip()
#         txt = txt.strip().strip("```json").strip("```").strip()
#         data = json.loads(txt)
#         return {
#             "opener": str(data.get("opener", "")).strip()[:300],
#             "insights": [str(x).strip() for x in (data.get("insights") or [])][:8],
#             "recommendations": [str(x).strip() for x in (data.get("recommendations") or [])][:8],
#             "next_step": str(data.get("next_step", "")).strip()[:200],
#         }
#     except Exception:
#         return {"opener": "Here are the top findings at a glance.",
#                 "insights": [], "recommendations": [], "next_step": ""}


# narrative.py
import os, json, requests, re

from django.utils.timezone import now

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError, ServiceResponseError
#groq running on 9000

# def llm_generate_narrative(question, rows):
#     preview = rows[:20]
#     preview_str = "\n".join(", ".join(f"{k}: {v}" for k, v in r.items()) for r in preview)

#     system_prompt = (
#     "You are a business analyst. Produce a SHORT, executive-friendly narrative in STRICT JSON with keys:\n"
#     "opener (string), insights (string[]), recommendations (string[]), next_step (string).\n"
#     "The opener must dynamically reference the user’s question and context. "
#     "Examples: 'Let’s take a closer look at the churn probability across customer segments…', "
#     "'The analysis of retention by state reveals…', "
#     "'Examining churn patterns by policy tenure shows…'. "
#     "Avoid generic openings like 'Here is…' or 'Here are…'.\n"
#     "No SQL, no markdown, no extra keys, no code fences."
# )

#     user_prompt = f"""Question: {question}

# First 20 rows:
# {preview_str}

# Return ONLY JSON with keys: opener, insights, recommendations, next_step.
# """.strip()

#     headers = {
#         "Authorization": f"Bearer {os.getenv('GROQ_API_KEY','')}",
#         "Content-Type": "application/json",
#     }
#     payload = {
#         "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         "temperature": 0.2,
#         "max_tokens": 600,
#     }

#     try:
#         resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
#                              headers=headers, json=payload, timeout=45)
#         resp.raise_for_status()
#         txt = resp.json()["choices"][0]["message"]["content"].strip()

#         # Clean out code fences if the model adds them
#         txt = re.sub(r"^```(?:json)?\s*", "", txt)
#         txt = re.sub(r"\s*```$", "", txt)

#         data = json.loads(txt)

#         opener_text = str(data.get("opener", "")).strip()[:300]

#         # --- Validation: auto-fix if opener starts with weak phrases ---
#         banned_starts = ("Here are", "Here is", "This is", "These are")
#         if opener_text.startswith(banned_starts):
#             opener_text = "Let’s take a closer look at the results."

#         return {
#             "opener": opener_text,
#             "insights": [str(x).strip() for x in (data.get("insights") or [])][:8],
#             "recommendations": [str(x).strip() for x in (data.get("recommendations") or [])][:8],
#             "next_step": str(data.get("next_step", "")).strip()[:200],
#         }
#     except Exception:
#         return {
#             "opener": "Let’s take a closer look at the results.",
#             "insights": [],
#             "recommendations": [],
#             "next_step": ""
#         }




def llm_generate_narrative(question, rows):
    preview = rows[:20]
    preview_str = "\n".join(", ".join(f"{k}: {v}" for k, v in r.items()) for r in preview)

    system_prompt = (
        "You are a business analyst. Produce a SHORT, executive-friendly narrative in STRICT JSON with keys:\n"
        "opener (string), insights (string[]), recommendations (string[]), next_step (string).\n"
        "The opener must dynamically reference the user’s question and context. "
        "Examples: 'Let’s take a closer look at the churn probability across customer segments…', "
        "'The analysis of retention by state reveals…', "
        "'Examining churn patterns by policy tenure shows…'. "
        "Avoid generic openings like 'Here is…' or 'Here are…'.\n"
        "No SQL, no markdown, no extra keys, no code fences."
    )

    user_prompt = f"""Question: {question}

First 20 rows:
{preview_str}

Return ONLY JSON with keys: opener, insights, recommendations, next_step.
""".strip()

    # --- Azure configs (reuse your environment variables) ---
    endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT", "").rstrip("/")
    if not endpoint.endswith("/models"):
        endpoint = f"{endpoint}/models"

    api_key = os.getenv("AZURE_INFERENCE_API_KEY", "")
    model = os.getenv("AZURE_INFERENCE_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8-prochurn-demo")
    api_version = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")

    try:
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
            api_version=api_version,
        )

        resp = client.complete(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )

        txt = resp.choices[0].message.content.strip()

        # Clean out code fences if the model adds them
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)

        data = json.loads(txt)

        opener_text = str(data.get("opener", "")).strip()[:300]

        # --- Validation: auto-fix if opener starts with weak phrases ---
        banned_starts = ("Here are", "Here is", "This is", "These are")
        if opener_text.startswith(banned_starts):
            opener_text = "Let’s take a closer look at the results."

        return {
            "opener": opener_text,
            "insights": [str(x).strip() for x in (data.get("insights") or [])][:8],
            "recommendations": [str(x).strip() for x in (data.get("recommendations") or [])][:8],
            "next_step": str(data.get("next_step", "")).strip()[:200],
        }
    except Exception:
        return {
            "opener": "Let’s take a closer look at the results.",
            "insights": [],
            "recommendations": [],
            "next_step": ""
        }


# def llm_generate_narrative(question, rows):
#     preview = rows[:20]
#     preview_str = "\n".join(", ".join(f"{k}: {v}" for k, v in r.items()) for r in preview)

#     system_prompt = (
#         "You are a business analyst. Produce a SHORT, executive-friendly narrative in STRICT JSON with keys:\n"
#         "opener (string), insights (string[]), recommendations (string[]), next_step (string).\n"
#         "The opener must dynamically reference the user’s question and context. "
#         "Examples: 'Let’s take a closer look at the churn probability across customer segments…', "
#         "'The analysis of retention by state reveals…', "
#         "'Examining churn patterns by policy tenure shows…'. "
#         "Avoid generic openings like 'Here is…' or 'Here are…'.\n"
#         "No SQL, no markdown, no extra keys, no code fences."
#     )

#     user_prompt = f"""Question: {question}

# First 20 rows:
# {preview_str}

# Return ONLY JSON with keys: opener, insights, recommendations, next_step.
# """.strip()

#     # --- Azure configs (reuse your environment variables) ---
#     endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT", "").rstrip("/")
#     if not endpoint.endswith("/models"):
#         endpoint = f"{endpoint}/models"

#     api_key = os.getenv("AZURE_INFERENCE_API_KEY", "")
#     model = os.getenv("AZURE_INFERENCE_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8-prochurn-demo")
#     api_version = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")

#     try:
#         client = ChatCompletionsClient(
#             endpoint=endpoint,
#             credential=AzureKeyCredential(api_key),
#             api_version=api_version,
#         )

#         resp = client.complete(
#             model=model,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             temperature=0.2,
#             max_tokens=600,
#         )

#         txt = resp.choices[0].message.content.strip()

#         # Clean out code fences if the model adds them
#         txt = re.sub(r"^```(?:json)?\s*", "", txt)
#         txt = re.sub(r"\s*```$", "", txt)

#         data = json.loads(txt)

#         opener_text = str(data.get("opener", "")).strip()[:300]

#         # --- Validation: auto-fix if opener starts with weak phrases ---
#         banned_starts = ("Here are", "Here is", "This is", "These are")
#         if opener_text.startswith(banned_starts):
#             opener_text = "Let’s take a closer look at the results."

#         return {
#             "opener": opener_text,
#             "insights": [str(x).strip() for x in (data.get("insights") or [])][:8],
#             "recommendations": [str(x).strip() for x in (data.get("recommendations") or [])][:8],
#             "next_step": str(data.get("next_step", "")).strip()[:200],
#         }
#     except Exception:
#         return {
#             "opener": "Let’s take a closer look at the results.",
#             "insights": [],
#             "recommendations": [],
#             "next_step": ""
#         }
