# from django.shortcuts import render



from django.shortcuts import render
import os
import uuid
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import requests
import json
import traceback


@csrf_exempt
def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        filename = default_storage.save(file.name, file)
        filepath = os.path.join(settings.MEDIA_ROOT, filename)

        try:
            #  Load file based on extension
            if filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(filepath)
            elif filename.endswith('.csv'):
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(filepath, encoding='ISO-8859-1')
            elif filename.endswith('.tsv'):
                df = pd.read_csv(filepath, sep='\t')
            elif filename.endswith('.xlsb'):
                df = pd.read_excel(filepath, engine='pyxlsb')
            else:
                return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)

            df = df.head(100)  # limit rows
            # Convert NaN/NaT to null-safe for JSON
            df = df.where(pd.notnull(df), None)

        except Exception as e:
            print("❌ Uncaught error in upload_file:", traceback.format_exc())
            return JsonResponse({'error': f'Server crashed: {str(e)}'}, status=500)

        preview = df.head(10).to_dict(orient='records')

        chart_path = None
        try:
            if df.shape[1] >= 2:
                chart_path = os.path.join(settings.MEDIA_ROOT, f"chart_{uuid.uuid4()}.png")
                df.iloc[:, :2].value_counts().plot(kind='bar')
                plt.title("Auto Chart from First 2 Columns")
                plt.tight_layout()
                plt.savefig(chart_path)
                plt.close()
        except Exception as e:
            chart_path = None

        return JsonResponse({
            'message': 'File uploaded and chart generated.',
            'filename': filename,
            'columns': df.columns.tolist(),
            'preview': preview,
            'chart_url': settings.MEDIA_URL + os.path.basename(chart_path) if chart_path else None
        }, json_dumps_params={"allow_nan": False})  # ✅ prevent NaN in final response

    return JsonResponse({'error': 'Invalid request'}, status=400)


# @csrf_exempt
# def ask(request):
#     print("🚀 Starting Ollama (Qwen2.5) in background...")
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             question = data.get('question', '').strip()
#             filename = data.get('filename', '').strip()

#             if not question:
#                 return JsonResponse({'error': '❗ Question is required.'}, status=400)

#             prompt = ""

#             if filename:
#                 filepath = os.path.join(settings.MEDIA_ROOT, filename)

#                 if not os.path.exists(filepath):
#                     return JsonResponse({'error': '❌ Uploaded file not found.'}, status=400)

#                 if filename.endswith(('.xls', '.xlsx')):
#                     df = pd.read_excel(filepath)
#                 elif filename.endswith('.csv'):
#                     try:
#                         df = pd.read_csv(filepath, encoding='utf-8')
#                     except UnicodeDecodeError:
#                         df = pd.read_csv(filepath, encoding='ISO-8859-1')
#                 elif filename.endswith('.tsv'):
#                     df = pd.read_csv(filepath, sep='\t')
#                 elif filename.endswith('.xlsb'):
#                     df = pd.read_excel(filepath, engine='pyxlsb')
#                 else:
#                     return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)

#                 df = df.where(pd.notnull(df), None)
#   # clean NaN
#                 context = df.head(20).to_csv(index=False)
#                 prompt = f"Given the following data:\n{context}\n\nAnswer this: {question}"

#             else:
#                 prompt = f"Answer the following question:\n{question}"

#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": prompt,
#                 "stream": False
#             })

#             json_response = response.json()
#             answer = json_response.get('response', '❌ No response from Qwen model.')

#             return JsonResponse({
#                 'question': question,
#                 'answer': answer
#             })

#         except Exception as e:
#             print("❌ ask_qwen failed:", traceback.format_exc())
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Invalid request'}, status=400)

import matplotlib.pyplot as plt
import uuid
import seaborn as sns

# @csrf_exempt
# def ask(request):
#     print("🚀 Starting Qwen2.5 model...")
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             question = data.get('question', '').strip()
#             filename = data.get('filename', '').strip()

#             if not question or not filename:
#                 return JsonResponse({'error': '❗ Question and filename are required.'}, status=400)

#             filepath = os.path.join(settings.MEDIA_ROOT, filename)
#             if not os.path.exists(filepath):
#                 return JsonResponse({'error': '❌ Uploaded file not found.'}, status=400)

#             # Load file
#             if filename.endswith(('.xls', '.xlsx')):
#                 df = pd.read_excel(filepath)
#             elif filename.endswith('.csv'):
#                 try:
#                     df = pd.read_csv(filepath, encoding='utf-8')
#                 except UnicodeDecodeError:
#                     df = pd.read_csv(filepath, encoding='ISO-8859-1')
#             elif filename.endswith('.tsv'):
#                 df = pd.read_csv(filepath, sep='\t')
#             elif filename.endswith('.xlsb'):
#                 df = pd.read_excel(filepath, engine='pyxlsb')
#             else:
#                 return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)

#             df = df.where(pd.notnull(df), None)
#             df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

#             question_lower = question.lower()

#             # ✅ Auto Chart Trigger Logic
#             if any(k in question_lower for k in ["chart", "plot", "graph", "visualize"]):
#                 numeric_cols = df.select_dtypes(include='number').columns.tolist()
#                 non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()

#                 if numeric_cols and non_numeric_cols:
#                     chart_file = f"chart_{uuid.uuid4()}.png"
#                     chart_path = os.path.join(settings.MEDIA_ROOT, chart_file)

#                     # Use first category and first numeric for plotting
#                     cat_col = non_numeric_cols[0]
#                     val_col = numeric_cols[0]

#                     plt.figure(figsize=(10, 6))
#                     chart_df = df[[cat_col, val_col]].dropna().groupby(cat_col)[val_col].mean().sort_values(ascending=False).head(10)
#                     chart_df.plot(kind='bar', color='teal')
#                     plt.title(f"Top 10 {cat_col.title()} by Avg {val_col.title()}")
#                     plt.xticks(rotation=45, ha='right')
#                     plt.ylabel(val_col.title())
#                     plt.tight_layout()
#                     plt.savefig(chart_path)
#                     plt.close()

#                     return JsonResponse({
#                         'question': question,
#                         'answer': f'📊 Chart generated using {cat_col} and {val_col}.',
#                         'chart_url': settings.MEDIA_URL + chart_file
#                     })

#                 else:
#                     return JsonResponse({'answer': '❌ No suitable numeric and categorical columns found to plot.'})

#             # ✅ Fallback to LLM (Qwen)
#             rows_for_prompt = df.head(100).to_csv(index=False)
#             prompt = f"""
# You are a smart analyst.
# Here is a sample of the uploaded data:

# {rows_for_prompt}

# Answer this question: {question}
# """

#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": prompt,
#                 "stream": False
#             })

#             json_response = response.json()
#             answer = json_response.get('response', '❌ No response from the model.')

#             return JsonResponse({
#                 'question': question,
#                 'answer': answer
#             })

#         except Exception as e:
#             print("❌ ask() crashed:", traceback.format_exc())
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Invalid request method'}, status=400)


import json
import os
import traceback
import pandas as pd
import requests
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json


import os
import traceback
import pandas as pd
import requests
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt



from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import os
import traceback
import json
import requests
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import os
import traceback
import json
import requests
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import os
import traceback
import json
import requests
from django.conf import settings

@csrf_exempt
def askl(request):
    print("\U0001F680 Starting Qwen2.5 model...")

    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)

    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        filename = data.get('filename', '').strip()
        model_name = data.get('model', 'qwen2.5:7b').strip()

        if model_name not in ['qwen2.5:1.5b', 'qwen2.5:7b']:
            return JsonResponse({'error': f'❌ Unsupported model: {model_name}'}, status=400)

        if not question:
            return JsonResponse({'error': '❗ Question is required.'}, status=400)

        df = None
        if filename:
            filepath = os.path.join(settings.MEDIA_ROOT, filename)
            if not os.path.exists(filepath):
                return JsonResponse({'error': '❌ Uploaded file not found.'}, status=400)

            try:
                if filename.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(filepath)
                elif filename.endswith('.csv'):
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(filepath, encoding='ISO-8859-1')
                elif filename.endswith('.tsv'):
                    df = pd.read_csv(filepath, sep='\t')
                elif filename.endswith('.xlsb'):
                    df = pd.read_excel(filepath, engine='pyxlsb')
                else:
                    return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)
            except Exception as file_err:
                return JsonResponse({'error': f'❌ Failed to load file: {file_err}'}, status=400)

            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            df = df.where(pd.notnull(df), None)

        question_lower = question.lower()

        # ✅ Chart Generation based on question and file
        chart_keywords = ["chart", "plot", "graph", "visualize", "sales", "trend", "compare"]
        chart_types = ["bar", "line", "column", "pie", "area"]

        matched_chart = next((t for t in chart_types if t in question_lower), None)
        chart_requested = df is not None and any(k in question_lower for k in chart_keywords)

        if chart_requested:
            schema_info = "\n".join([f"- {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
            max_rows = min(1000, len(df))
            rows_info = df.head(max_rows).to_string(index=False)

            chart_prompt = f"""
You are a data visualization assistant. Based on the following schema and data, generate the best chart to answer the user's question.

If the user has specified a chart type in the question, strictly use that chart type.

Include:
1. chart_type: one of ['bar', 'line', 'column', 'pie', 'area']
2. x_axis: column name
3. y_axis: column name
4. A list of chart data as dictionaries

Always respond ONLY with the requested JSON format. Do not explain or add extra text.

\ud83d\udcd8 SCHEMA:
{schema_info}

\ud83d\udcca DATA SAMPLE ({max_rows} rows):
{rows_info}

❓ QUESTION:
{question}

Reply in JSON format like:
{{
  "chart_type": "line",
  "x_axis": "month",
  "y_axis": "sales",
  "data": [
    {{"month": "Jan", "sales": 100}},
    ...
  ]
}}
"""

            response = requests.post("http://localhost:11434/api/generate", json={
                "model": model_name,
                "prompt": chart_prompt,
                "stream": False
            })

            try:
                json_text = response.json().get("response", "")
                print("📤 Raw response for chart:", json_text)
                chart_info = json.loads(json_text)

                required_keys = ["chart_type", "y_axis", "data"]
                if not all(k in chart_info for k in required_keys):
                    return JsonResponse({'error': f'❌ Missing keys in chart data: {chart_info.keys()}'}, status=400)

                chart_type = chart_info.get("chart_type", "bar")

                data = chart_info["data"]

                if chart_type == "pie":
                    series_data = []
                    for row in data:
                        try:
                            name = row.get("name") or next(iter(row.values()))
                            value = row.get("y") or row.get("value") or list(row.values())[-1]
                            value = eval(str(value).split('=')[-1].strip()) if '+' in str(value) or '=' in str(value) else float(value)
                            series_data.append({"name": name, "y": value})
                        except Exception as e:
                            print("⚠️ Pie chart parse error:", e)
                    return JsonResponse({
                        'question': question,
                        'answer': f'📊 Pie chart generated.',
                        'chart_type': "pie",
                        'x_categories': [],
                        'y_key': chart_info["y_axis"],
                        'series_data': series_data
                    })

                # For all other charts
                x_axis = chart_info.get("x_axis")
                y_axis = chart_info.get("y_axis")
                if not x_axis or not y_axis:
                    return JsonResponse({'error': '❌ x_axis or y_axis missing.'}, status=400)

                return JsonResponse({
                    'question': question,
                    'answer': f'📊 {chart_type.capitalize()} chart generated.',
                    'chart_type': chart_type,
                    'x_categories': [item[x_axis] for item in data],
                    'y_key': y_axis,
                    'series_data': [
                        {"name": item[x_axis], "y": item[y_axis]} for item in data
                    ]
                })

            except Exception as chart_err:
                return JsonResponse({'error': f'❌ Failed to parse chart info: {chart_err}'}, status=400)

        # Fallback
        fallback_prompt = f"""
You are a helpful and knowledgeable assistant.

Answer the following question clearly and concisely:

❓ {question}
"""

        response = requests.post("http://localhost:11434/api/generate", json={
            "model": model_name,
            "prompt": fallback_prompt,
            "stream": False
        })

        json_response = response.json()
        answer = json_response.get('response', '❌ No response from the model.')

        return JsonResponse({
            'question': question,
            'answer': answer
        })

    except Exception as e:
        print("❌ ask() crashed:", traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def asknow(request):
    print("🚀 Starting Qwen2.5 model...")

    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)

    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        filename = data.get('filename', '').strip()

        if not question:
            return JsonResponse({'error': '❗ Question is required.'}, status=400)

        df = None
        if filename:
            filepath = os.path.join(settings.MEDIA_ROOT, filename)
            if not os.path.exists(filepath):
                return JsonResponse({'error': '❌ Uploaded file not found.'}, status=400)

            try:
                if filename.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(filepath)
                elif filename.endswith('.csv'):
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(filepath, encoding='ISO-8859-1')
                elif filename.endswith('.tsv'):
                    df = pd.read_csv(filepath, sep='\t')
                elif filename.endswith('.xlsb'):
                    df = pd.read_excel(filepath, engine='pyxlsb')
                else:
                    return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)
            except Exception as file_err:
                return JsonResponse({'error': f'❌ Failed to load file: {file_err}'}, status=400)

            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            df = df.where(pd.notnull(df), None)

        question_lower = question.lower()

        # ✅ Chart Generation
        if df is not None and any(k in question_lower for k in ["chart", "plot", "graph", "visualize", "sales", "trend", "compare"]):
            schema_info = "\n".join([f"- {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
            max_rows = min(1000, len(df))
            rows_info = df.head(max_rows).to_string(index=False)

            chart_prompt = f"""
You are a data visualization assistant. Based on the following schema and data, generate the best chart to answer the user's question.

Include:
1. chart_type: one of ['bar', 'line', 'column', 'pie', 'area']
2. x_axis: column name
3. y_axis: column name
4. A list of chart data as dictionaries

📘 SCHEMA:
{schema_info}

📊 DATA SAMPLE ({max_rows} rows):
{rows_info}

❓ QUESTION:
{question}

Reply in JSON format like:
{{
  "chart_type": "line",
  "x_axis": "month",
  "y_axis": "sales",
  "data": [
    {{"month": "Jan", "sales": 100}},
    ...
  ]
}}
"""

            response = requests.post("http://localhost:11434/api/generate", json={
                "model": "qwen2.5:7b",
                "prompt": chart_prompt,
                "stream": False
            })

            try:
                json_text = response.json().get("response", "")
                chart_info = json.loads(json_text)

                return JsonResponse({
                    'question': question,
                    'answer': f'📊 {chart_info["chart_type"].capitalize()} chart generated.',
                    'chart_type': chart_info["chart_type"],
                    'x_categories': [item[chart_info["x_axis"]] for item in chart_info["data"]] if chart_info["chart_type"] != "pie" else [],
                    'y_key': chart_info["y_axis"],
                    'series_data': [
                        {"name": item[chart_info["x_axis"]], "y": item[chart_info["y_axis"]]}
                        for item in chart_info["data"]
                    ]
                })

            except Exception as chart_err:
                return JsonResponse({'error': f'❌ Failed to parse chart info: {chart_err}'}, status=400)

        # ✅ Summary Generation
        if df is not None and any(k in question_lower for k in ["summary", "summarize", "overview", "insight", "brief"]):
            schema_info = "\n".join([f"- {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
            max_rows = min(1000, len(df))
            rows_info = df.head(max_rows).to_string(index=False)

            summary_prompt = f"""
You are a smart data analyst AI. Based on the schema and sample data below, provide a short summary with key insights and trends.

📘 SCHEMA:
{schema_info}

📊 DATA SAMPLE ({max_rows} rows):
{rows_info}

🧠 SUMMARY:
"""

            response = requests.post("http://localhost:11434/api/generate", json={
                "model": "qwen2.5:7b",
                "prompt": summary_prompt,
                "stream": False
            })

            json_response = response.json()
            answer = json_response.get('response', '❌ No summary returned.')

            return JsonResponse({
                'question': question,
                'answer': answer,
                'summary': True
            })

        # ✅ General Data Q&A
        if df is not None:
            schema_info = "\n".join([f"- {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
            max_rows = min(1000, len(df))
            rows_info = df.head(max_rows).to_string(index=False)

            prompt = f"""
You are a smart data analyst. Use the schema and data sample to answer the question.

📘 SCHEMA:
{schema_info}

📊 DATA SAMPLE ({max_rows} rows):
{rows_info}

❓ QUESTION:
{question}

Answer only based on the data.
"""

            response = requests.post("http://localhost:11434/api/generate", json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False
            })

            json_response = response.json()
            answer = json_response.get('response', '❌ No response from the model.')

            return JsonResponse({
                'question': question,
                'answer': answer
            })

        return JsonResponse({'error': '❌ No data available to analyze.'}, status=400)

    except Exception as e:
        print("❌ ask() crashed:", traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)
    
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import os
import traceback
import json
import requests
from django.conf import settings

@csrf_exempt
def ask(request):
    print("🚀 Starting Qwen2.5 model...")

    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)

    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        filename = data.get('filename', '').strip()

        if not question:
            return JsonResponse({'error': 'Question is required.'}, status=400)

        df = None
        if filename:
            filepath = os.path.join(settings.MEDIA_ROOT, filename)
            if not os.path.exists(filepath):
                return JsonResponse({'error': 'Uploaded file not found.'}, status=400)

            if filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(filepath)
            elif filename.endswith('.csv'):
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(filepath, encoding='ISO-8859-1')
            elif filename.endswith('.tsv'):
                df = pd.read_csv(filepath, sep='\t')
            elif filename.endswith('.xlsb'):
                df = pd.read_excel(filepath, engine='pyxlsb')
            else:
                return JsonResponse({'error': 'Unsupported file format.'}, status=400)

            df = df.where(pd.notnull(df), None)
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        question_lower = question.lower()

        # ✅ Handle direct table preview
        if df is not None and any(k in question_lower for k in ["show table", "show data", "print table", "preview"]):
            sample = df.head(20)
            markdown_table = sample.to_markdown(index=False)
            return JsonResponse({
                'question': question,
                'answer': f"Here is a preview of your data:\n\n{markdown_table}"
            })


        # ✅ Handle row count questions
        if df is not None and any(k in question_lower for k in ["how many rows", "row count", "total rows"]):
            return JsonResponse({
                'question': question,
                'answer': f" The uploaded sheet has {len(df)} rows."
            })

        # ✅ Handle column count questions
        if df is not None and any(k in question_lower for k in ["how many columns", "column count", "total columns"]):
            return JsonResponse({
                'question': question,
                'answer': f"The uploaded sheet has {df.shape[1]} columns: {', '.join(df.columns)}"
            })

        # ✅ Auto Chart Trigger (Highcharts JSON)
        if df is not None and any(k in question_lower for k in ["chart", "plot", "graph", "visualize"]):
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()

            cat_col = next((col for col in non_numeric_cols if col in question_lower), non_numeric_cols[0] if non_numeric_cols else None)
            val_col = next((col for col in numeric_cols if col in question_lower), numeric_cols[0] if numeric_cols else None)

            if not cat_col or not val_col:
                return JsonResponse({'error': 'Could not find suitable columns for chart generation.'}, status=400)

            if "pie" in question_lower:
                chart_type = "pie"
            elif "line" in question_lower:
                chart_type = "line"
            elif "bar" in question_lower:
                chart_type = "bar"
            elif "column" in question_lower:
                chart_type = "column"
            elif "hist" in question_lower or "histogram" in question_lower:
                chart_type = "column"
            elif "area" in question_lower:
                chart_type = "area"
            else:
                chart_type = "column"

            chart_df = df[[cat_col, val_col]].dropna().groupby(cat_col)[val_col].mean().sort_values(ascending=False).head(10)
            chart_data = chart_df.reset_index().to_dict(orient='records')

            series_data = [{"name": row[cat_col], "y": row[val_col]} for row in chart_data]

            return JsonResponse({
                'question': question,
                'answer': f' {chart_type.capitalize()} chart generated using {cat_col} and {val_col}.',
                'chart_type': chart_type,
                'series_data': series_data,
                'x_categories': [row[cat_col] for row in chart_data] if chart_type != "pie" else [],
                'y_key': val_col,
            })

        # ✅ Fallback to LLM (Qwen)
        rows_for_prompt = df.head(100).to_csv(index=False) if df is not None else ""
        prompt = f"""
You are a smart analyst.
{f"Here is a sample of the uploaded data:\n\n{rows_for_prompt}" if df is not None else "No file uploaded."}

Answer this question: {question}
"""

        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "qwen2.5:1.5b",
            "prompt": prompt,
            "stream": False
        })

        json_response = response.json()
        answer = json_response.get('response', 'No response from the model.')

        return JsonResponse({
            'question': question,
            'answer': answer
        })

    except Exception as e:
        print("ask() crashed:", traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)
    
import os
import json
import uuid
import traceback
import requests
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.conf import settings
# 4-6-mrg
@csrf_exempt
def ask11(request):
    print("🚀 Starting Qwen2.5 model...")
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            question = data.get('question', '').strip()
            filename = data.get('filename', '').strip()

            if not question:
                return JsonResponse({'error': '❗ Question is required.'}, status=400)

            df = None
            if filename:
                filepath = os.path.join(settings.MEDIA_ROOT, filename)
                if not os.path.exists(filepath):
                    return JsonResponse({'error': '❌ Uploaded file not found.'}, status=400)

                if filename.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(filepath)
                elif filename.endswith('.csv'):
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(filepath, encoding='ISO-8859-1')
                elif filename.endswith('.tsv'):
                    df = pd.read_csv(filepath, sep='\t')
                elif filename.endswith('.xlsb'):
                    df = pd.read_excel(filepath, engine='pyxlsb')
                else:
                    return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)

                df = df.where(pd.notnull(df), None)
                df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

            question_lower = question.lower()

            # ✅ Handle row count directly without LLM
            if df is not None and "how many rows" in question_lower:
                return JsonResponse({
                    'question': question,
                    'answer': f"There are {len(df)} rows in your sheet."
                })

            # ✅ Auto Chart Trigger Logic
            if df is not None and any(k in question_lower for k in ["chart", "plot", "graph", "visualize"]):
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()

                if numeric_cols and non_numeric_cols:
                    chart_type = "column"
                    if "pie" in question_lower:
                        chart_type = "pie"
                    elif "line" in question_lower:
                        chart_type = "line"
                    elif "bar" in question_lower:
                        chart_type = "bar"
                    elif "hist" in question_lower or "histogram" in question_lower:
                        chart_type = "column"
                    elif "area" in question_lower:
                        chart_type = "area"

                    cat_col = non_numeric_cols[0]
                    val_col = numeric_cols[0]

                    chart_df = df[[cat_col, val_col]].dropna().groupby(cat_col)[val_col].mean().sort_values(ascending=False).head(10)
                    chart_data = chart_df.reset_index().to_dict(orient='records')

                    series_data = [{"name": row[cat_col], "y": row[val_col]} for row in chart_data]

                    return JsonResponse({
                        'question': question,
                        'answer': f'📊 {chart_type.capitalize()} chart generated using {cat_col} and {val_col}.',
                        'chart_type': chart_type,
                        'series_data': series_data,
                        'x_categories': [row[cat_col] for row in chart_data] if chart_type != "pie" else [],
                        'y_key': val_col,
                    })

            # ✅ Prompt Construction for General Questions
            if df is not None:
                schema_info = "\n".join([f"- {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])

                try:
                    head_rows = df.head(30)
                    tail_rows = df.tail(30)
                    sample_data = pd.concat([head_rows, tail_rows])
                    sample_info = sample_data.to_string(index=False)
                except Exception:
                    sample_info = "⚠️ Could not extract sample rows."

                try:
                    summary_stats = df.describe(include='all').to_string()
                except Exception:
                    summary_stats = "⚠️ Could not generate summary statistics."

                prompt = f"""
You are a smart data analyst. Use the schema, sample rows, and summary stats below to answer the user's question.

📘 SCHEMA:
{schema_info}

📊 SAMPLE ROWS (Head + Tail):
{sample_info}

📈 SUMMARY STATS:
{summary_stats}

❓ QUESTION:
{question}

Answer only based on this info. If the information is missing, say so.
"""

                # Optional: Limit prompt size to prevent crash (token budget)
                if len(prompt) > 10000:
                    prompt = prompt[:10000] + "\n⚠️ Prompt truncated due to size."

                response = requests.post("http://localhost:11434/api/generate", json={
                    "model": "qwen2.5:7b",
                    "prompt": prompt,
                    "stream": False
                })

                json_response = response.json()
                answer = json_response.get('response', '❌ No response from the model.')

                return JsonResponse({
                    'question': question,
                    'answer': answer
                })

            else:
                return JsonResponse({'error': '❌ No data available to analyze.'}, status=400)

        except Exception as e:
            print("❌ ask() crashed:", traceback.format_exc())
            return JsonResponse({'error': str(e)}, status=500)
# 4-6-mrg
# @csrf_exempt
# def ask(request):
#     print("🚀 Starting Qwen2.5 model...")
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             question = data.get('question', '').strip()
#             filename = data.get('filename', '').strip()

#             if not question:
#                 return JsonResponse({'error': '❗ Question is required.'}, status=400)

#             df = None
#             if filename:
#                 filepath = os.path.join(settings.MEDIA_ROOT, filename)
#                 if not os.path.exists(filepath):
#                     return JsonResponse({'error': '❌ Uploaded file not found.'}, status=400)

#                 # Load file based on extension
#                 if filename.endswith(('.xls', '.xlsx')):
#                     df = pd.read_excel(filepath)
#                 elif filename.endswith('.csv'):
#                     try:
#                         df = pd.read_csv(filepath, encoding='utf-8')
#                     except UnicodeDecodeError:
#                         df = pd.read_csv(filepath, encoding='ISO-8859-1')
#                 elif filename.endswith('.tsv'):
#                     df = pd.read_csv(filepath, sep='\t')
#                 elif filename.endswith('.xlsb'):
#                     df = pd.read_excel(filepath, engine='pyxlsb')
#                 else:
#                     return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)

#                 df = df.where(pd.notnull(df), None)
#                 df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

#             question_lower = question.lower()

#             # ✅ If file uploaded and user asks row count
#             if df is not None and "how many rows" in question_lower:
#                 return JsonResponse({
#                     'question': question,
#                     'answer': f"There are {len(df)} rows in your sheet."
#                 })

#             # ✅ Auto Chart generation if user asks for plot
#             if df is not None and any(k in question_lower for k in ["chart", "plot", "graph", "visualize"]):
#                 numeric_cols = df.select_dtypes(include='number').columns.tolist()
#                 non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()

#                 if numeric_cols and non_numeric_cols:
#                     chart_type = "column"
#                     if "pie" in question_lower:
#                         chart_type = "pie"
#                     elif "line" in question_lower:
#                         chart_type = "line"
#                     elif "bar" in question_lower:
#                         chart_type = "bar"
#                     elif "hist" in question_lower or "histogram" in question_lower:
#                         chart_type = "column"
#                     elif "area" in question_lower:
#                         chart_type = "area"

#                     cat_col = non_numeric_cols[0]
#                     val_col = numeric_cols[0]

#                     chart_df = df[[cat_col, val_col]].dropna().groupby(cat_col)[val_col].mean().sort_values(ascending=False).head(10)
#                     chart_data = chart_df.reset_index().to_dict(orient='records')

#                     series_data = [{"name": row[cat_col], "y": row[val_col]} for row in chart_data]

#                     return JsonResponse({
#                         'question': question,
#                         'answer': f'📊 {chart_type.capitalize()} chart generated using {cat_col} and {val_col}.',
#                         'chart_type': chart_type,
#                         'series_data': series_data,
#                         'x_categories': [row[cat_col] for row in chart_data] if chart_type != "pie" else [],
#                         'y_key': val_col,
#                     })

#             # ✅ If file is uploaded: construct prompt from schema + data
#             if df is not None:
#                 schema_info = "\n".join([f"- {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])

#                 try:
#                     head_rows = df.head(30)
#                     tail_rows = df.tail(30)
#                     sample_data = pd.concat([head_rows, tail_rows])
#                     sample_info = sample_data.to_string(index=False)
#                 except Exception:
#                     sample_info = "⚠️ Could not extract sample rows."

#                 try:
#                     summary_stats = df.describe(include='all').to_string()
#                 except Exception:
#                     summary_stats = "⚠️ Could not generate summary statistics."

#                 prompt = f"""
# You are a smart data analyst. Use the schema, sample rows, and summary stats below to answer the user's question.

# 📘 SCHEMA:
# {schema_info}

# 📊 SAMPLE ROWS (Head + Tail):
# {sample_info}

# 📈 SUMMARY STATS:
# {summary_stats}

# ❓ QUESTION:
# {question}

# Answer only based on this info. If the information is missing, say so.
# """
#             else:
#                 # ✅ If no file is uploaded, treat it as a general question
#                 prompt = f"""
# You are a helpful assistant. Answer the following user question in a clear and friendly way.

# ❓ QUESTION:
# {question}
# """

#             # ✅ Truncate prompt if too long
#             if len(prompt) > 10000:
#                 prompt = prompt[:10000] + "\n⚠️ Prompt truncated due to size."

#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:7b",
#                 "prompt": prompt,
#                 "stream": False
#             })

#             json_response = response.json()
#             answer = json_response.get('response', '❌ No response from the model.')

#             return JsonResponse({
#                 'question': question,
#                 'answer': answer
#             })

#         except Exception as e:
#             print("❌ ask() crashed:", traceback.format_exc())
#             return JsonResponse({'error': str(e)}, status=500)

# running
# @csrf_exempt
# def ask(request):
#     print("🚀 Starting Qwen2.5 model...")
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             question = data.get('question', '').strip()
#             filename = data.get('filename', '').strip()

#             if not question:
#                 return JsonResponse({'error': '❗ Question is required.'}, status=400)

#             df = None
#             if filename:
#                 filepath = os.path.join(settings.MEDIA_ROOT, filename)
#                 if not os.path.exists(filepath):
#                     return JsonResponse({'error': '❌ Uploaded file not found.'}, status=400)

#                 if filename.endswith(('.xls', '.xlsx')):
#                     df = pd.read_excel(filepath)
#                 elif filename.endswith('.csv'):
#                     try:
#                         df = pd.read_csv(filepath, encoding='utf-8')
#                     except UnicodeDecodeError:
#                         df = pd.read_csv(filepath, encoding='ISO-8859-1')
#                 elif filename.endswith('.tsv'):
#                     df = pd.read_csv(filepath, sep='\t')
#                 elif filename.endswith('.xlsb'):
#                     df = pd.read_excel(filepath, engine='pyxlsb')
#                 else:
#                     return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)

#                 df = df.where(pd.notnull(df), None)
#                 df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

#             question_lower = question.lower()

#             # ✅ Auto Chart Trigger Logic (Highcharts JSON)
#             if df is not None and any(k in question_lower for k in ["chart", "plot", "graph", "visualize"]):
#                 numeric_cols = df.select_dtypes(include='number').columns.tolist()
#                 non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()

#                 if numeric_cols and non_numeric_cols:
#                     if "pie" in question_lower:
#                         chart_type = "pie"
#                     elif "line" in question_lower:
#                         chart_type = "line"
#                     elif "bar" in question_lower:
#                         chart_type = "bar"
#                     elif "column" in question_lower:
#                         chart_type = "column"
#                     elif "hist" in question_lower or "histogram" in question_lower:
#                         chart_type = "column"
#                     elif "area" in question_lower:
#                         chart_type = "area"
#                     else:
#                         chart_type = "column"

#                     cat_col = non_numeric_cols[0]
#                     val_col = numeric_cols[0]

#                     chart_df = df[[cat_col, val_col]].dropna().groupby(cat_col)[val_col].mean().sort_values(ascending=False).head(10)
#                     chart_data = chart_df.reset_index().to_dict(orient='records')

#                     if chart_type == "pie":
#                         series_data = [{"name": row[cat_col], "y": row[val_col]} for row in chart_data]
#                     else:
#                         series_data = [{"name": row[cat_col], "y": row[val_col]} for row in chart_data]

#                     return JsonResponse({
#                         'question': question,
#                         'answer': f'📊 {chart_type.capitalize()} chart generated using {cat_col} and {val_col}.',
#                         'chart_type': chart_type,
#                         'series_data': series_data,
#                         'x_categories': [row[cat_col] for row in chart_data] if chart_type != "pie" else [],
#                         'y_key': val_col,
#                     })

#             # ✅ Fallback to LLM (Qwen)
#             rows_for_prompt = df.head(100).to_csv(index=False) if df is not None else ""
#             prompt = f"""
# You are a smart analyst.
# {f"Here is a sample of the uploaded data:\n\n{rows_for_prompt}" if df is not None else "No file uploaded."}

# Answer this question: {question}
# """

#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": prompt,
#                 "stream": False
#             })

#             json_response = response.json()
#             answer = json_response.get('response', '❌ No response from the model.')

#             return JsonResponse({
#                 'question': question,
#                 'answer': answer
#             })

#         except Exception as e:
#             print("❌ ask() crashed:", traceback.format_exc())
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Invalid request method'}, status=400)
# running
# from django.views.decorators.csrf import csrf_exempt
# from django.http import JsonResponse
# from django.conf import settings
# import traceback
# import requests
# import pandas as pd
# import os
# import json

# @csrf_exempt
# def ask(request):
#     print("🚀 Starting Qwen model...")
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             question = data.get('question', '').strip()
#             filename = data.get('filename', '').strip()

#             if not question:
#                 return JsonResponse({'error': '❗ Question is required.'}, status=400)

#             df = None
#             if filename:
#                 filepath = os.path.join(settings.MEDIA_ROOT, filename)
#                 if not os.path.exists(filepath):
#                     return JsonResponse({'error': '❌ Uploaded file not found.'}, status=400)

#                 if filename.endswith(('.xls', '.xlsx')):
#                     df = pd.read_excel(filepath)
#                 elif filename.endswith('.csv'):
#                     try:
#                         df = pd.read_csv(filepath, encoding='utf-8')
#                     except UnicodeDecodeError:
#                         df = pd.read_csv(filepath, encoding='ISO-8859-1')
#                 elif filename.endswith('.tsv'):
#                     df = pd.read_csv(filepath, sep='\t')
#                 elif filename.endswith('.xlsb'):
#                     df = pd.read_excel(filepath, engine='pyxlsb')
#                 else:
#                     return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)

#                 df = df.where(pd.notnull(df), None)
#                 df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

#             question_lower = question.lower()

#             # Detect if it's a query for total quantity (e.g. how many x bought by y)
#             if df is not None and any(k in question_lower for k in ["how many", "total", "bought", "sold", "purchased"]):
#                 keywords = question_lower.split()
#                 colnames = df.columns.tolist()
#                 product_col = next((col for col in colnames if "product" in col or "item" in col), None)
#                 person_col = next((col for col in colnames if "salesperson" in col or "name" in col), None)
#                 units_col = next((col for col in colnames if "unit" in col and "sold" in col), None)

#                 if product_col and person_col and units_col:
#                     matching_rows = df[
#                         df[product_col].str.lower().isin([word for word in keywords]) &
#                         df[person_col].str.lower().isin([word for word in keywords])
#                     ]

#                     total_units = matching_rows[units_col].sum()
#                     if not matching_rows.empty:
#                         return JsonResponse({
#                             'answer': f"✅ Total {product_col.replace('_',' ')}: {int(total_units)}",
#                             'table_markdown': matching_rows.to_markdown(index=False)
#                         })

#             # ✅ Auto Chart Trigger Logic
#             if df is not None and any(k in question_lower for k in ["chart", "plot", "graph", "visualize"]):
#                 numeric_cols = df.select_dtypes(include='number').columns.tolist()
#                 non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()

#                 if numeric_cols and non_numeric_cols:
#                     chart_type = "column"
#                     if "pie" in question_lower:
#                         chart_type = "pie"
#                     elif "line" in question_lower:
#                         chart_type = "line"
#                     elif "bar" in question_lower:
#                         chart_type = "bar"
#                     elif "area" in question_lower:
#                         chart_type = "area"

#                     cat_col = non_numeric_cols[0]
#                     val_col = numeric_cols[0]

#                     chart_df = df[[cat_col, val_col]].dropna().groupby(cat_col)[val_col].mean().sort_values(ascending=False).head(10)
#                     chart_data = chart_df.reset_index().to_dict(orient='records')
#                     series_data = [{"name": row[cat_col], "y": row[val_col]} for row in chart_data]

#                     return JsonResponse({
#                         'question': question,
#                         'answer': f'📊 {chart_type.capitalize()} chart generated using {cat_col} and {val_col}.',
#                         'chart_type': chart_type,
#                         'series_data': series_data,
#                         'x_categories': [row[cat_col] for row in chart_data] if chart_type != "pie" else [],
#                         'y_key': val_col,
#                     })

#             # Fallback to LLM
#             rows_for_prompt = df.head(50).to_csv(index=False) if df is not None else ""
#             prompt = f"""
# You are a smart data analyst.

# {f"Here is a sample of the uploaded data:\n\n{rows_for_prompt}" if df is not None else "No file uploaded."}

# Answer the user's question:
# - If the user asks to show, table or list data, respond with a markdown table. Align the table using consistent column widths.
# - If the user asks for a report, provide a structured summary (overview, trends, recommendations).
# - Otherwise, give a helpful text answer.

# Ensure the markdown table uses proper alignment and spacing for readability.

# Question: {question}
# """

#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": prompt,
#                 "stream": False
#             })

#             json_response = response.json()
#             answer = json_response.get('response', '❌ No response from the model.')

#             return JsonResponse({
#                 'question': question,
#                 'answer': answer
#             })

#         except Exception as e:
#             print("❌ ask() crashed:", traceback.format_exc())
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Invalid request method'}, status=400)

# @csrf_exempt
# def ask(request):
#     print("🚀 Starting Qwen model...")
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Invalid request method'}, status=400)

#     try:
#         data = json.loads(request.body)
#         question = data.get('question', '').strip()
#         filename = data.get('filename', '').strip()

#         if not question:
#             return JsonResponse({'error': '❗ Question is required.'}, status=400)

#         df = None
#         if filename:
#             filepath = os.path.join(settings.MEDIA_ROOT, filename)
#             if not os.path.exists(filepath):
#                 return JsonResponse({'error': '❌ Uploaded file not found.'}, status=400)

#             # Read file
#             if filename.endswith(('.xls', '.xlsx')):
#                 df = pd.read_excel(filepath)
#             elif filename.endswith('.csv'):
#                 try:
#                     df = pd.read_csv(filepath, encoding='utf-8')
#                 except UnicodeDecodeError:
#                     df = pd.read_csv(filepath, encoding='ISO-8859-1')
#             elif filename.endswith('.tsv'):
#                 df = pd.read_csv(filepath, sep='\t')
#             elif filename.endswith('.xlsb'):
#                 df = pd.read_excel(filepath, engine='pyxlsb')
#             else:
#                 return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)

#             df = df.where(pd.notnull(df), None)
#             df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

#         question_lower = question.lower()

#         # ✅ Product Category Count (e.g., how many products)
#         if df is not None and "product" in df.columns:
#             if "how many" in question_lower and "product" in question_lower and "present" in question_lower:
#                 product_counts = df["product"].value_counts().reset_index()
#                 product_counts.columns = ["Product", "Count"]
#                 return JsonResponse({
#                     'answer': "This table shows the number of each product category present in the uploaded data.",
#                     'table_markdown': product_counts.to_markdown(index=False)
#                 })

#         # ✅ Specific Unit Sold (e.g., how many monitors sold)
#         if df is not None and any(k in question_lower for k in ["how many", "total", "bought", "sold", "purchased"]):
#             colnames = df.columns.tolist()
#             product_col = next((col for col in colnames if "product" in col), None)
#             person_col = next((col for col in colnames if "salesperson" in col or "person" in col), None)
#             units_col = next((col for col in colnames if "unit" in col and "sold" in col), None)

#             if product_col and units_col:
#                 keywords = question_lower.split()

#                 product_filter = df[product_col].astype(str).str.lower().apply(lambda x: any(k in x for k in keywords))
#                 person_filter = df[person_col].astype(str).str.lower().apply(lambda x: any(k in x for k in keywords)) if person_col else True

#                 matched_df = df[product_filter & person_filter]

#                 if not matched_df.empty:
#                     total_units = matched_df[units_col].sum()
#                     return JsonResponse({
#                         'answer': f"{matched_df[product_col].iloc[0].capitalize()} total units sold: {int(total_units)}",
#                         'table_markdown': matched_df.to_markdown(index=False)
#                     })

#         # 📊 Chart Trigger
#         if df is not None and any(k in question_lower for k in ["chart", "plot", "graph", "visualize"]):
#             numeric_cols = df.select_dtypes(include='number').columns.tolist()
#             non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()

#             if numeric_cols and non_numeric_cols:
#                 chart_type = "column"
#                 if "pie" in question_lower: chart_type = "pie"
#                 elif "line" in question_lower: chart_type = "line"
#                 elif "bar" in question_lower: chart_type = "bar"
#                 elif "area" in question_lower: chart_type = "area"

#                 cat_col = non_numeric_cols[0]
#                 val_col = numeric_cols[0]

#                 chart_df = df[[cat_col, val_col]].dropna().groupby(cat_col)[val_col].sum().sort_values(ascending=False).head(10)
#                 chart_data = chart_df.reset_index().to_dict(orient='records')
#                 series_data = [{"name": row[cat_col], "y": row[val_col]} for row in chart_data]

#                 return JsonResponse({
#                     'question': question,
#                     'answer': f'📊 {chart_type.capitalize()} chart generated using {cat_col} and {val_col}.',
#                     'chart_type': chart_type,
#                     'series_data': series_data,
#                     'x_categories': [row[cat_col] for row in chart_data] if chart_type != "pie" else [],
#                 })

#         # 🧠 LLM fallback with strict instructions
#         rows_for_prompt = df.head(50).to_csv(index=False) if df is not None else ""
#         prompt = f"""
# You are a helpful AI data assistant. Only use the dataset provided.

# {f"The uploaded dataset sample:\n\n{rows_for_prompt}" if df is not None else "No file uploaded."}

# Instructions:
# - If the user asks for a TABLE, respond in markdown table.
# - If the user asks a numeric or aggregation question (e.g. "how many", "total", "sum"), compute using the data.
# - NEVER make up data. Base answers strictly on the dataset.
# - If uncertain, say "Cannot determine based on uploaded file."

# Question: {question}
# """

#         response = requests.post("http://localhost:11434/api/generate", json={
#             "model": "qwen2.5:1.5b",
#             "prompt": prompt,
#             "stream": False
#         })

#         json_response = response.json()
#         answer = json_response.get("response", "❌ No response from the model.")
#         is_markdown_table = answer.strip().startswith("|") and "---" in answer

#         return JsonResponse({
#             'question': question,
#             'answer': answer,
#             'table_markdown': answer if is_markdown_table else None,
#         })

#     except Exception as e:
#         print("❌ ask() crashed:", traceback.format_exc())
#         return JsonResponse({'error': str(e)}, status=500)

# @csrf_exempt
# def ask(request):
#     print("🚀 Starting Qwen model...")
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             question = data.get('question', '').strip()
#             filename = data.get('filename', '').strip()

#             if not question:
#                 return JsonResponse({'error': '❗ Question is required.'}, status=400)

#             df = None
#             if filename:
#                 filepath = os.path.join(settings.MEDIA_ROOT, filename)
#                 if not os.path.exists(filepath):
#                     return JsonResponse({'error': '❌ Uploaded file not found.'}, status=400)

#                 if filename.endswith(('.xls', '.xlsx')):
#                     df = pd.read_excel(filepath)
#                 elif filename.endswith('.csv'):
#                     try:
#                         df = pd.read_csv(filepath, encoding='utf-8')
#                     except UnicodeDecodeError:
#                         df = pd.read_csv(filepath, encoding='ISO-8859-1')
#                 elif filename.endswith('.tsv'):
#                     df = pd.read_csv(filepath, sep='\t')
#                 elif filename.endswith('.xlsb'):
#                     df = pd.read_excel(filepath, engine='pyxlsb')
#                 else:
#                     return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)

#                 df = df.where(pd.notnull(df), None)
#                 df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

#             question_lower = question.lower()

#             # ✅ Auto Chart Trigger Logic (Highcharts JSON)
#             if df is not None and any(k in question_lower for k in ["chart", "plot", "graph", "visualize"]):
#                 numeric_cols = df.select_dtypes(include='number').columns.tolist()
#                 non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()

#                 if numeric_cols and non_numeric_cols:
#                     chart_type = "column"
#                     if "pie" in question_lower:
#                         chart_type = "pie"
#                     elif "line" in question_lower:
#                         chart_type = "line"
#                     elif "bar" in question_lower:
#                         chart_type = "bar"
#                     elif "area" in question_lower:
#                         chart_type = "area"

#                     cat_col = non_numeric_cols[0]
#                     val_col = numeric_cols[0]

#                     chart_df = df[[cat_col, val_col]].dropna().groupby(cat_col)[val_col].mean().sort_values(ascending=False).head(10)
#                     chart_data = chart_df.reset_index().to_dict(orient='records')

#                     series_data = [{"name": row[cat_col], "y": row[val_col]} for row in chart_data]

#                     return JsonResponse({
#                         'question': question,
#                         'answer': f'📊 {chart_type.capitalize()} chart generated using {cat_col} and {val_col}.',
#                         'chart_type': chart_type,
#                         'series_data': series_data,
#                         'x_categories': [row[cat_col] for row in chart_data] if chart_type != "pie" else [],
#                         'y_key': val_col,
#                     })

#             # ✅ Fallback to LLM (Qwen) for Table or Report
#             rows_for_prompt = df.head(100).to_csv(index=False) if df is not None else ""
#             prompt = f"""
# You are a smart data analyst.

# {f"Here is a sample of the uploaded data:\n\n{rows_for_prompt}" if df is not None else "No file uploaded."}

# Answer the user's question:
# - If the user asks to show, table or list data, respond with a markdown table. Align the table using consistent column widths.
# - If the user asks for a report, provide a structured summary (overview, trends, recommendations).
# - Otherwise, give a helpful text answer.

# Ensure the markdown table uses proper alignment and spacing for readability.

# Question: {question}
# """

#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": prompt,
#                 "stream": False
#             })

#             json_response = response.json()
#             answer = json_response.get('response', '❌ No response from the model.')

#             return JsonResponse({
#                 'question': question,
#                 'answer': answer
#             })

#         except Exception as e:
#             print("❌ ask() crashed:", traceback.format_exc())
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Invalid request method'}, status=400)


# @csrf_exempt
# def ask(request):
#     print("🚀 Starting Qwen2.5 model...")
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             question = data.get('question', '').strip()
#             filename = data.get('filename', '').strip()

#             if not question or not filename:
#                 return JsonResponse({'error': '❗ Question and filename are required.'}, status=400)

#             filepath = os.path.join(settings.MEDIA_ROOT, filename)
#             if not os.path.exists(filepath):
#                 return JsonResponse({'error': '❌ Uploaded file not found.'}, status=400)

#             # Load file
#             if filename.endswith(('.xls', '.xlsx')):
#                 df = pd.read_excel(filepath)
#             elif filename.endswith('.csv'):
#                 try:
#                     df = pd.read_csv(filepath, encoding='utf-8')
#                 except UnicodeDecodeError:
#                     df = pd.read_csv(filepath, encoding='ISO-8859-1')
#             elif filename.endswith('.tsv'):
#                 df = pd.read_csv(filepath, sep='\t')
#             elif filename.endswith('.xlsb'):
#                 df = pd.read_excel(filepath, engine='pyxlsb')
#             else:
#                 return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)

#             df = df.where(pd.notnull(df), None)
#             df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

#             question_lower = question.lower()

#             # ✅ Auto Chart Trigger Logic (Highcharts-friendly JSON instead of PNG)
#             if any(k in question_lower for k in ["chart", "plot", "graph", "visualize"]):
#                 numeric_cols = df.select_dtypes(include='number').columns.tolist()
#                 non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()

#                 if numeric_cols and non_numeric_cols:
#                     # Determine chart type
#                     if "pie" in question_lower:
#                         chart_type = "pie"
#                     elif "line" in question_lower:
#                         chart_type = "line"
#                     elif "bar" in question_lower:
#                         chart_type = "bar"
#                     elif "hist" in question_lower or "histogram" in question_lower:
#                         chart_type = "column"
#                     elif "area" in question_lower:
#                         chart_type = "area"
#                     else:
#                         chart_type = "bar"

#                     cat_col = non_numeric_cols[0]
#                     val_col = numeric_cols[0]

#                     chart_df = df[[cat_col, val_col]].dropna().groupby(cat_col)[val_col].mean().sort_values(ascending=False).head(10)
#                     chart_data = chart_df.reset_index().to_dict(orient='records')

#                     return JsonResponse({
#                         'question': question,
#                         'answer': f'📊 {chart_type.capitalize()} chart generated using {cat_col} and {val_col}.',
#                         'chart_type': chart_type,
#                         'xKey': cat_col,
#                         'yKey': val_col,
#                         'chart_data': chart_data
#                     })
#                 else:
#                     return JsonResponse({'answer': '❌ No suitable numeric and categorical columns found to plot.'})

#             # ✅ Fallback to LLM (Qwen)
#             rows_for_prompt = df.head(100).to_csv(index=False)
#             prompt = f"""
# You are a smart analyst.
# Here is a sample of the uploaded data:

# {rows_for_prompt}

# Answer this question: {question}
# """

#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": prompt,
#                 "stream": False
#             })

#             json_response = response.json()
#             answer = json_response.get('response', '❌ No response from the model.')

#             return JsonResponse({
#                 'question': question,
#                 'answer': answer
#             })

#         except Exception as e:
#             print("❌ ask() crashed:", traceback.format_exc())
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Invalid request method'}, status=400)


# @csrf_exempt
# def ask(request):
#     print("🚀 Starting Qwen2.5 model...")
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             question = data.get('question', '').strip()
#             filename = data.get('filename', '').strip()

#             if not question:
#                 return JsonResponse({'error': '❗ Question is required.'}, status=400)
#             if not filename:
#                 return JsonResponse({'error': '❗ Filename is required.'}, status=400)

#             filepath = os.path.join(settings.MEDIA_ROOT, filename)
#             if not os.path.exists(filepath):
#                 return JsonResponse({'error': '❌ Uploaded file not found.'}, status=400)

#             # 🧠 Load the file into a DataFrame
#             if filename.endswith(('.xls', '.xlsx')):
#                 df = pd.read_excel(filepath)
#             elif filename.endswith('.csv'):
#                 try:
#                     df = pd.read_csv(filepath, encoding='utf-8')
#                 except UnicodeDecodeError:
#                     df = pd.read_csv(filepath, encoding='ISO-8859-1')
#             elif filename.endswith('.tsv'):
#                 df = pd.read_csv(filepath, sep='\t')
#             elif filename.endswith('.xlsb'):
#                 df = pd.read_excel(filepath, engine='pyxlsb')
#             else:
#                 return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)

#             df = df.where(pd.notnull(df), None)  # Convert NaNs

#             # 🧠 Normalize column names
#             df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
#             fuel_col = "fuel_type" if "fuel_type" in df.columns else None
#             city_col = "rto_location" if "rto_location" in df.columns else None

#             # ✅ Backend-driven logic for specific questions
#             question_lower = question.lower()
#             if fuel_col and "diesel" in question_lower and "how many" in question_lower:
#                 diesel_count = df[df[fuel_col].str.upper() == "DIESEL"].shape[0]
#                 return JsonResponse({
#                     'question': question,
#                     'answer': f'There are {diesel_count} diesel cars in the dataset.'
#                 })

#             if city_col and "city" in question_lower:
#                 city_count = df[city_col].nunique()
#                 return JsonResponse({
#                     'question': question,
#                     'answer': f'There are {city_count} distinct cities in the dataset.'
#                 })

#             # 🧠 For general LLM prompt-based answers
#             rows_for_prompt = df.head(100).to_csv(index=False)

#             prompt = f"""
# You are a smart insurance data analyst.
# Analyze the following dataset (first 100 rows):

# {rows_for_prompt}

# Question: {question}

# Please give a clear, data-supported answer.
# """

#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": prompt,
#                 "stream": False
#             })

#             json_response = response.json()
#             answer = json_response.get('response', '❌ No response from the model.')

#             return JsonResponse({
#                 'question': question,
#                 'answer': answer
#             })

#         except Exception as e:
#             print("❌ ask() crashed:", traceback.format_exc())
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Invalid request method'}, status=400)


def index(request):
    return render(request, 'index.html')

import json
import re
import urllib.parse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sqlalchemy import create_engine, inspect, text
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"  # Or use qwen2.5:3b if 7b is unavailable

# Step 1: Inspect schema and collect tables and columns
def get_schema_structure(engine, schema_filter=None):
    inspector = inspect(engine)
    structure = {}
    for schema in inspector.get_schema_names():
        if schema in ["information_schema", "pg_catalog"]:
            continue
        if schema_filter and schema != schema_filter:
            continue
        structure[schema] = {}
        for table in inspector.get_table_names(schema=schema):
            columns = inspector.get_columns(table, schema=schema)
            structure[schema][table] = [col["name"] for col in columns]
    return structure

# Step 2: Format prompt for LLM
def format_schema_prompt(schema_dict):
    lines = []
    for schema, tables in schema_dict.items():
        for table, columns in tables.items():
            lines.append(f"{schema}.{table}: {', '.join(columns)}")
    return "\n".join(lines)

# Step 3: Generate SQL from question + schema prompt
def generate_sql_from_question(question, schema_text):
    prompt = f"""### Database Schema ###
{schema_text}

### Task ###
Convert the following natural language request into an SQL query:
Question: {question}
SQL:"""

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })

    result = response.json()
    sql_code = result.get("response", "").strip()
    sql_code = re.split(r';?\n', sql_code)[0]  # Get first SQL line
    return sql_code

# # Step 4: Django API endpoint
# @csrf_exempt
# def ask_qwen_db(request):
#     if request.method != "POST":
#         return JsonResponse({"message": "Please use POST method to query the database via AI."}, status=405)

#     try:
#         data = json.loads(request.body)
#         question = data.get("question")
#         db_config = data.get("db_config")

#         if not question or not db_config:
#             return JsonResponse({"error": "Missing question or db_config"}, status=400)

#         # DB credentials
#         host = db_config.get("host")
#         port = int(db_config.get("port", 5432))
#         database = db_config.get("database")
#         user = db_config.get("user")
#         password = db_config.get("password")
#         schema = db_config.get("schema")  # Optional

#         # Safe password URL encoding
#         safe_password = urllib.parse.quote_plus(password)
#         db_url = f"postgresql://{user}:{safe_password}@{host}:{port}/{database}"
#         engine = create_engine(db_url)

#         # Step 0: Fast connection test
#         if question.lower().strip() == "test connection":
#             with engine.connect() as conn:
#                 conn.execute(text("SELECT 1"))  # Safe ping
#             return JsonResponse({
#                 "question": question,
#                 "result": [{"status": "connected"}]
#             })

#         # Step 1: Extract schema (entire or filtered)
#         schema_info = get_schema_structure(engine, schema_filter=schema)
#         formatted_schema = format_schema_prompt(schema_info)

#         # Step 2: Handle meta-question manually — "how many tables in schema"
#         schema_match = re.search(r'how many tables.*\b(\w+)\b', question, re.IGNORECASE)
#         if schema_match:
#             schema_name = schema_match.group(1)
#             if schema_name in schema_info:
#                 table_count = len(schema_info[schema_name])
#                 return JsonResponse({
#                     "question": question,
#                     "generated_sql": None,
#                     "result": [{"table_count": table_count}]
#                 })

#         # Step 3: Ask Qwen2.5 to generate SQL
#         generated_sql = generate_sql_from_question(question, formatted_schema)
#         print("🧠 Generated SQL:", generated_sql)

#         # Safety: Only allow SELECT queries
#         if not any(generated_sql.lower().strip().startswith(cmd) for cmd in ["select", "insert", "update"]):

#             return JsonResponse({"error": "Only SELECT queries are allowed."}, status=400)

#         # Step 4: Execute query
#         with engine.connect() as conn:
#             result = conn.execute(text(generated_sql))
#             columns = result.keys()
#             rows = [dict(zip(columns, row)) for row in result]

#         return JsonResponse({
#             "question": question,
#             "generated_sql": generated_sql,
#             "result": rows
#         })

#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)


# views.py

import json, urllib.parse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
# from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType
from langchain_ollama import OllamaLLM
import json, urllib.parse, traceback
import warnings

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# @csrf_exempt
# def ask_qwen_db(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "Only POST allowed"}, status=405)

#     try:
#         data = json.loads(request.body)
#         question = data.get("question")
#         db_config = data.get("db_config")  # dict with host, port, user, password, db

#         if not question or not db_config:
#             return JsonResponse({"error": "Missing question or db_config"}, status=400)

#         # Construct connection URL
#         safe_pwd = urllib.parse.quote_plus(db_config['password'])
#         db_url = f"postgresql://{db_config['user']}:{safe_pwd}@{db_config['host']}:{db_config.get('port', 5432)}/{db_config['database']}"

#         # Initialize LLM + LangChain SQL agent
#         # llm = Ollama(model="qwen2.5:7b", base_url="http://localhost:11434")
#         llm = OllamaLLM(model="qwen2.5:7b", base_url="http://localhost:11434",temperature=0.0,
#     max_tokens=512,streaming=True)

#         db = SQLDatabase.from_uri(db_url)
#         toolkit = SQLDatabaseToolkit(llm=llm, db=db)
#         agent = initialize_agent(
#             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
#             tools=toolkit.get_tools(), 
#             llm=llm, 
#             verbose=False)

#         # response = agent.run(question)
#     #     response = agent.invoke(question)

#     #     return JsonResponse({"answer": response})

#     # except Exception as e:
#     #     return JsonResponse({"error": str(e)}, status=500)
#         result = agent.invoke({"input": question})

#         final_answer = (
#             result.get("output") or result.get("answer") or str(result)
#             if isinstance(result, dict)
#             else str(result)
#         )

#         return JsonResponse({"answer": final_answer})

#     except Exception as e:
#         print("Exception:", traceback.format_exc())
#         return JsonResponse({"error": str(e)}, status=500)
import json, urllib.parse, traceback, warnings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

warnings.filterwarnings("ignore", category=DeprecationWarning)
import json, urllib.parse, traceback, warnings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json, urllib.parse, traceback, warnings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType
from sqlalchemy import create_engine, inspect

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import re
import json, urllib.parse, traceback, warnings, re
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType
from sqlalchemy import create_engine, inspect, text

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def format_schema_details(db_url):
    engine = create_engine(db_url)
    inspector = inspect(engine)
    lines = []
    for schema in inspector.get_schema_names():
        if schema in ["information_schema", "pg_catalog"]:
            continue
        for table in inspector.get_table_names(schema=schema):
            cols = inspector.get_columns(table, schema=schema)
            col_names = ', '.join([c['name'] for c in cols])
            lines.append(f"Schema: {schema}, Table: {table}, Columns: {col_names}")
    return "\n".join(lines)

def extract_sql(text_response):
    """Extract SQL code from markdown-style ```sql blocks"""
    match = re.search(r"```sql\s*(.*?)\s*```", text_response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

@csrf_exempt
def ask_qwen_db(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
        question = data.get("question")
        db_config = data.get("db_config")

        if not question or not db_config:
            return JsonResponse({"error": "Missing question or db_config"}, status=400)

        # Database URI
        safe_pwd = urllib.parse.quote_plus(db_config['password'])
        db_url = f"postgresql://{db_config['user']}:{safe_pwd}@{db_config['host']}:{db_config.get('port', 5432)}/{db_config['database']}"

        # Schema context
        schema_context = format_schema_details(db_url)
        prompt = (
            f"You are a PostgreSQL expert. Below is the database schema:\n\n"
            f"{schema_context}\n\n"
            f"Answer the following user question based on this schema. If answering requires a query, return only the SQL query in ```sql block. Don't explain anything:\n{question}"
            f"You are a PostgreSQL expert. Below is the database schema:\n\n"
            f"{schema_context}\n\n"
            f"When writing SQL queries, always qualify columns using table names "
            f"(e.g., 'orders.product_id' not just 'product_id'). "
            f"If answering requires a query, return it in ```sql block. "
            f"Don't explain anything:\n{question}"
        )

        # LLM + Agent setup
        llm = OllamaLLM(model="qwen2.5:7b", base_url="http://localhost:11434", temperature=0.0, max_tokens=512)
        db = SQLDatabase.from_uri(db_url)
        toolkit = SQLDatabaseToolkit(llm=llm, db=db)
        agent = initialize_agent(
            tools=toolkit.get_tools(),
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
        )

        result = agent.invoke({"input": prompt})
        final_output = (
            result.get("output") or result.get("answer") or str(result)
            if isinstance(result, dict)
            else str(result)
        )

        sql_query = extract_sql(final_output)

        if sql_query:
            with db._engine.connect() as conn:
                query_result = conn.execute(text(sql_query)).fetchone()
                if query_result:
                    return JsonResponse({"answer": str(query_result[0])})
                else:
                    return JsonResponse({"answer": "No result returned."})
        else:
            return JsonResponse({"answer": final_output})

    except Exception as e:
        print("❌ Exception:", traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)



# # ✅ Fix: Use SQLAlchemy inspect properly
# def format_schema_details(db_url):
#     engine = create_engine(db_url)
#     inspector = inspect(engine)  # <-- corrected line
#     lines = []
#     for schema in inspector.get_schema_names():
#         if schema in ["information_schema", "pg_catalog"]:
#             continue
#         for table in inspector.get_table_names(schema=schema):
#             cols = inspector.get_columns(table, schema=schema)
#             col_names = ', '.join([c['name'] for c in cols])
#             lines.append(f"Schema: {schema}, Table: {table}, Columns: {col_names}")
#     return "\n".join(lines)

# @csrf_exempt
# def ask_qwen_db(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "Only POST allowed"}, status=405)

#     try:
#         data = json.loads(request.body)
#         question = data.get("question")
#         db_config = data.get("db_config")

#         if not question or not db_config:
#             return JsonResponse({"error": "Missing question or db_config"}, status=400)

#         # Build DB URL
#         safe_pwd = urllib.parse.quote_plus(db_config['password'])
#         db_url = f"postgresql://{db_config['user']}:{safe_pwd}@{db_config['host']}:{db_config.get('port', 5432)}/{db_config['database']}"

#         # Get schema-aware context
#         schema_context = format_schema_details(db_url)
#         # prompt = (
#         #     f"You are a PostgreSQL expert. Below is the database schema:\n\n"
#         #     f"{schema_context}\n\n"
#         #     f"Answer the following user question based only on this schema:\n{question}"
#         # )
#         prompt = (
#             f"You are a PostgreSQL expert. Below is the database schema:\n\n"
#             f"{schema_context}\n\n"
#             f"Answer the following user question based on this schema. Do not return SQL code. Only provide the final numerical or textual result. Be precise:\n{question}"
#         )


#         # Initialize model and agent
#         llm = OllamaLLM(
#             model="qwen2.5:7b",
#             base_url="http://localhost:11434",
#             temperature=0.0,
#             max_tokens=512,
#         )

#         db = SQLDatabase.from_uri(db_url)
#         toolkit = SQLDatabaseToolkit(llm=llm, db=db)
#         agent = initialize_agent(
#             tools=toolkit.get_tools(),
#             llm=llm,
#             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#             verbose=False,
#         )

#         result = agent.invoke({"input": prompt})

#         final_answer = (
#             result.get("output") or result.get("answer") or str(result)
#             if isinstance(result, dict)
#             else str(result)
#         )

#         return JsonResponse({"answer": final_answer})

#     except Exception as e:
#         print("❌ Exception:", traceback.format_exc())
#         return JsonResponse({"error": str(e)}, status=500)




# import pandas as pd
# import psycopg2
# import requests
# import json
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from sqlalchemy import create_engine



# from sqlalchemy import inspect

# def get_full_schema_structure(engine):
#     inspector = inspect(engine)
#     structure = []

#     for schema in inspector.get_schema_names():
#         if schema in ['information_schema', 'pg_catalog']:  # skip system schemas
#             continue
#         for table_name in inspector.get_table_names(schema=schema):
#             columns = inspector.get_columns(table_name, schema=schema)
#             column_names = [col['name'] for col in columns]
#             structure.append({
#                 "schema": schema,
#                 "table": table_name,
#                 "columns": column_names
#             })
#     return structure


import re
import pandas as pd
import urllib.parse
import requests
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sqlalchemy import create_engine, inspect

import re
import pandas as pd
import urllib.parse
import requests
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sqlalchemy import create_engine, inspect



# def get_full_schema_structure(engine, filter_schema=None):
#     inspector = inspect(engine)
#     structure = []
#     for schema in inspector.get_schema_names():
#         if schema in ['information_schema', 'pg_catalog']:
#             continue
#         if filter_schema and schema != filter_schema:
#             continue
#         for table_name in inspector.get_table_names(schema=schema):
#             columns = inspector.get_columns(table_name, schema=schema)
#             column_names = [col['name'] for col in columns]
#             structure.append({
#                 "schema": schema,
#                 "table": table_name,
#                 "columns": column_names
#             })
#     return structure


# def format_schema_for_prompt(schema_data):
#     prompt = "Available database structure:\n"
#     for item in schema_data:
#         prompt += f"- Schema: {item['schema']}, Table: {item['table']}, Columns: {', '.join(item['columns'])}\n"
#     return prompt


# def format_schema_for_prompt(schema_data):
#     prompt = "Available database structure:\n"
#     for item in schema_data:
#         prompt += f"- Schema: {item['schema']}, Table: {item['table']}, Columns: {', '.join(item['columns'])}\n"
#     return prompt
# def format_schema_for_prompt(schema_data, engine):
#     prompt = "Available database structure:\n"
#     conn = engine.connect()

#     for item in schema_data:
#         schema = item['schema']
#         table = item['table']
#         prompt += f"- Schema: {schema}, Table: {table}, Columns:\n"

#         for col in item['columns']:
#             desc = ""
#             try:
#                 # Fetch 2 sample values
#                 result = conn.execute(text(f'SELECT DISTINCT "{col}" FROM "{schema}"."{table}" WHERE "{col}" IS NOT NULL LIMIT 2'))
#                 values = [str(r[0]) for r in result]
#                 if values:
#                     desc = f" (e.g., {', '.join(values)})"
#             except:
#                 pass
#             prompt += f"    - {col}{desc}\n"

#     conn.close()
#     return prompt
# +++++++++++++

# Helper: Fetch tables from DB
# def list_available_tables(engine):
#     try:
#         with engine.connect() as conn:
#             result = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
#             return [row[0] for row in result.fetchall()]
#     except Exception as e:
#         print("❌ Failed to fetch table list:", str(e))
#         return []

# Helper: Fetch data from DB with generated query
# def fetch_table_data(query, host, port, db_name, db_user, db_password):
#     try:
#         print("🟢 Connecting to database...")
#         db_url = f"postgresql://{db_user}:{db_password}@{host}:{port}/{db_name}"
#         engine = create_engine(db_url)
#         df = pd.read_sql_query(query, engine)
#         return df
#     except Exception as e:
#         print("❌ Database Error:", str(e))
#         return None

# # Build prompt from DB result

# def build_prompt_from_db(df, user_question):
#     if df is None or df.empty:
#         return "The database returned no results."
#     table_sample = df.head(20).to_csv(index=False)
#     column_info = ', '.join(df.columns)
#     prompt = f"""
# You are an intelligent data assistant.

# Table Schema: {column_info}

# Here is the sample data:
# {table_sample}

# Question: {user_question}

# Answer in plain English using only the data above.
# """
#     return prompt

# # Ask Qwen 2.5

# def ask_qwen(prompt):
#     try:
#         response = requests.post("http://localhost:11434/api/generate", json={
#             "model": "qwen2.5:7b",
#             "prompt": prompt,
#             "stream": False
#         })
#         return response.json().get('response', '❌ No response from Qwen.')
#     except Exception as e:
#         return f"❌ Error calling Qwen: {str(e)}"

# # views.py
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from sqlalchemy import create_engine
# import pandas as pd
# import requests
# import json
# from sqlalchemy import inspect
# import urllib.parse

# import json
# import re
# import urllib.parse
# import pandas as pd
# import matplotlib.pyplot as plt
# import base64
# from io import BytesIO
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from sqlalchemy import create_engine, inspect
# import requests


# def get_full_schema_structure(engine, filter_schema=None):
#     inspector = inspect(engine)
#     structure = {}
#     schemas = inspector.get_schema_names()
#     for schema in schemas:
#         if schema in ['information_schema', 'pg_catalog']:
#             continue
#         if filter_schema and schema != filter_schema:
#             continue
#         tables = inspector.get_table_names(schema=schema)
#         structure[schema] = {}
#         for table in tables:
#             columns = inspector.get_columns(table, schema=schema)
#             structure[schema][table] = [col['name'] for col in columns]
#     return structure


# def format_schema_for_prompt(schema_dict):
#     lines = []
#     for schema, tables in schema_dict.items():
#         for table, columns in tables.items():
#             lines.append(f"{schema}.{table}: {', '.join(columns)}")
#     return "\n".join(lines)


# def guess_numeric_column(df):
#     for col in df.columns:
#         if pd.api.types.is_numeric_dtype(df[col]):
#             return col
#     return None


# def generate_chart_from_df(df, chart_type='bar', x_col=None, y_col=None):
#     plt.figure(figsize=(10, 5))
#     if chart_type == 'bar':
#         plt.bar(df[x_col], df[y_col])
#     elif chart_type == 'line':
#         plt.plot(df[x_col], df[y_col], marker='o')
#     elif chart_type == 'pie':
#         plt.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%')
#     else:
#         raise ValueError("Unsupported chart type. Use 'bar', 'line', or 'pie'.")

#     plt.xlabel(x_col)
#     plt.ylabel(y_col)
#     plt.title(f"{y_col} by {x_col}")
#     plt.xticks(rotation=45)
#     plt.tight_layout()

#     buffer = BytesIO()
#     plt.savefig(buffer, format='png')
#     plt.close()
#     buffer.seek(0)
#     image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
#     return image_base64


# @csrf_exempt
# def ask_qwen_db(request):
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Only POST allowed'}, status=405)

#     try:
#         data = json.loads(request.body)
#         question = data.get('question')
#         db_config = data.get('db_config')

#         if not question or not db_config:
#             return JsonResponse({'error': 'Missing question or db_config.'}, status=400)

#         host = db_config.get('host')
#         port = int(db_config.get('port', 5432))
#         database = db_config.get('database')
#         user = db_config.get('user')
#         password = db_config.get('password')
#         safe_password = urllib.parse.quote_plus(password)
#         db_url = f"postgresql://{user}:{safe_password}@{host}:{port}/{database}"
#         engine = create_engine(db_url)

#         all_schemas = [s for s in inspect(engine).get_schema_names() if s not in ['information_schema', 'pg_catalog']]
#         requested_schema = None

#         # Match schema by presence in question (e.g., "customer schema", "in customer", etc.)
#         for schema in all_schemas:
#             if schema.lower() in question.lower():
#                 requested_schema = schema
#                 break


#         if "schema" in question.lower() and "how many" in question.lower():
#             return JsonResponse({
#                 "answer": f"There are {len(all_schemas)} user-defined schemas: {', '.join(all_schemas)}.",
#                 "query": None, "columns": [], "preview": []
#             })

#         if "how many tables" in question.lower() or "total tables" in question.lower():
#             if not requested_schema:
#                 return JsonResponse({
#                     "answer": "Please specify which schema you're referring to.",
#                     "query": None, "columns": [], "preview": []
#                 })
#             tables = inspect(engine).get_table_names(schema=requested_schema)
#             return JsonResponse({
#                 "answer": f"There are {len(tables)} tables in schema '{requested_schema}'.",
#                 "query": None, "columns": [], "preview": []
#             })

#         if any(x in question.lower() for x in ["table name", "table names", "show tables", "what tables", "list tables"]):
#             if not requested_schema:
#                 return JsonResponse({
#                     "answer": "Please specify which schema you want to list tables from.",
#                     "query": None, "columns": [], "preview": []
#                 })
#             tables = inspect(engine).get_table_names(schema=requested_schema)
#             return JsonResponse({
#                 "answer": f"The tables in schema '{requested_schema}' are: {', '.join(tables)}.",
#                 "query": None, "columns": [], "preview": []
#             })

#         schema_structure = get_full_schema_structure(engine, filter_schema=requested_schema)
#         structure_text = format_schema_for_prompt(schema_structure)

#         sql_prompt = f"""
# You are a PostgreSQL expert. Generate ONLY a SQL SELECT query based on the schema and user question.

# ## DO NOT:
# - Do not explain.
# - Do not include anything other than a SQL SELECT query.
# - Do not guess tables or columns.

# ## DO:
# - Use only the following schema and table definitions.
# - Use exact names and fully-qualified names like schema.table.
# - If string match is required, use = or ILIKE properly.

# --- SCHEMA ---
# {structure_text}

# --- QUESTION ---
# {question}

# SQL:
# """

#         sql_response = requests.post("http://localhost:11434/api/generate", json={
#             "model": "qwen2.5:7b", "prompt": sql_prompt, "stream": False
#         })

#         raw_output = sql_response.json().get("response", "")
#         sql_only = None
#         for line in raw_output.splitlines():
#             line = line.strip().rstrip(";")
#             if line.lower().startswith("select"):
#                 sql_only = line + ";"
#                 break

#         if not sql_only:
#             return JsonResponse({"answer": "Could not generate SQL.", "query": None, "columns": [], "preview": []})

#         df = pd.read_sql_query(sql_only, engine)

#         if df.empty:
#             return JsonResponse({
#                 "answer": "Query returned no results.",
#                 "query": sql_only, "columns": [], "preview": []
#             })

#         # Chart generation if applicable
#         if df.shape[1] >= 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
#             x_col, y_col = df.columns[:2]
#             chart_base64 = generate_chart_from_df(df, chart_type='bar', x_col=x_col, y_col=y_col)
#             return JsonResponse({
#                 "answer": f"Here's a chart for {y_col} by {x_col}.",
#                 "query": sql_only,
#                 "columns": df.columns.tolist(),
#                 "preview": df.head(10).to_dict(orient="records"),
#                 "chart_base64": chart_base64
#             })

#         return JsonResponse({
#             "answer": "✅ Query executed successfully.",
#             "query": sql_only,
#             "columns": df.columns.tolist(),
#             "preview": df.head(10).to_dict(orient="records")
#         })

#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)
# ++++++++++

# def guess_numeric_column(df):
#     print(df.dtypes)

#     numeric_cols =df.select_dtypes(include=['number']).columns
#     return numeric_cols[0] if len(numeric_cols) > 0 else None

# def guess_numeric_column(df):
#     print(df.dtypes)

#     numeric_cols = df.select_dtypes(include='number').columns

#     return numeric_cols[0] if len(numeric_cols) > 0 else None



# def guess_numeric_column(df):
#     numeric_cols = df.select_dtypes(include='number').columns
#     return numeric_cols[0] if len(numeric_cols) > 0 else None

# def call_qwen(prompt, model="qwen2.5:7b", stream=False):
#     try:
#         response = requests.post("http://localhost:11434/api/generate", json={
#             "model": model,
#             "prompt": prompt,
#             "stream": stream
#         })
#         if response.status_code != 200:
#             return None, f"❌ Qwen error: {response.status_code} {response.text}"
#         return response.json().get('response', None), None
#     except Exception as e:
#         return None, f"❌ Qwen connection error: {str(e)}"

# ========================== MAIN VIEW ==========================

# @csrf_exempt
# def ask_qwen_db(request):
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Only POST allowed'}, status=405)

#     try:
#         data = json.loads(request.body)
#         question = data.get('question')
#         db_config = data.get('db_config')

#         if not question or not db_config:
#             return JsonResponse({'error': 'Missing question or db_config.'}, status=400)

#         # Database Config
#         host = db_config.get('host')
#         port = int(db_config.get('port', 5432))
#         database = db_config.get('database')
#         user = db_config.get('user')
#         password = urllib.parse.quote_plus(db_config.get('password'))
#         db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
#         engine = create_engine(db_url)

#         inspector = inspect(engine)

#         # Extract schema if specified in question
#         schema_match = re.search(r'from\s+([\w]+)[.]', question, re.IGNORECASE) or \
#                        re.search(r'in\s+schema\s+([\w]+)', question, re.IGNORECASE)
#         requested_schema = schema_match.group(1) if schema_match else None

#         # Greeting Case
#         greetings = [
#             "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening",
#             "how are you", "what's up", "howdy", "nice to meet you", "good to see you",
#             "yo", "sup", "hey there", "hiya", "😊", "👋", "🙋"
#         ]
#         if any(greet in question.lower() for greet in greetings):
#             prompt = f"You are a helpful assistant.\n\nUser: {question}"
#             answer, err = call_qwen(prompt)
#             return JsonResponse({
#                 "answer": answer or err or "🤖 No answer.",
#                 "query": None, "columns": [], "preview": []
#             })

#         # Ask for number of schemas
#         if "schema" in question.lower() and "how many" in question.lower():
#             schemas = [s for s in inspector.get_schema_names() if s not in ['information_schema', 'pg_catalog']]
#             return JsonResponse({
#                 "answer": f"There are {len(schemas)} user-defined schemas: {', '.join(schemas)}.",
#                 "query": None, "columns": [], "preview": []
#             })

#         # Ask for number of tables in specific schema
#         table_count_match = re.search(r'how many tables.*schema\s+(\w+)', question.lower())
#         if table_count_match:
#             target_schema = table_count_match.group(1)
#             tables = inspector.get_table_names(schema=target_schema)
#             return JsonResponse({
#                 "answer": f"Schema '{target_schema}' contains {len(tables)} tables: {', '.join(tables)}.",
#                 "query": None, "columns": [], "preview": []
#             })

#         # Schema structure prompt for SQL generation
#         schema_structure = get_full_schema_structure(engine, filter_schema=requested_schema)
#         structure_text = format_schema_for_prompt(schema_structure)

#         sql_prompt = f"""You are a PostgreSQL expert. Given the table and column definitions below, write a SQL SELECT query that answers the user's question.

# Only use the exact columns and tables listed.
# Always qualify tables as schema.table (e.g., public.car_sales).
# Use aggregation (SUM, COUNT, etc.) if asked.
# Include WHERE clauses to filter (e.g., WHERE car = 'Nissan') if needed.
# Only return SQL, no explanation.

# {structure_text}

# User Question: {question}
# """

#         generated_sql, err = call_qwen(sql_prompt)
#         if err or not generated_sql:
#             return JsonResponse({'error': err or '❌ No SQL generated by Qwen.'}, status=500)

#         generated_sql = generated_sql.replace("sql", "").replace("", "").strip()
#         print( generated_sql," generated_sql")

#         if not generated_sql.lower().startswith("select"):
#             fallback_prompt = f"User asked: {question}\n\nContext: {requested_schema or 'No schema specified'}.\nProvide a clear answer in plain English."
#             answer, _ = call_qwen(fallback_prompt)
#             return JsonResponse({
#                 "answer": answer or "🤖 Could not answer.",
#                 "query": None, "columns": [], "preview": []
#             })

#         # Run the SQL
#         df = pd.read_sql_query(generated_sql, engine)
#         if df.empty:
#             return JsonResponse({
#                 "answer": f"The query ran but returned no results.",
#                 "query": generated_sql, "columns": [], "preview": []
#             })

#         # If GROUP BY — summarize
#         if 'group by' in generated_sql.lower():
#             table_sample = df.head(20).to_csv(index=False)
#             analysis_prompt = f"""You are a smart assistant. Analyze this SQL output and provide a final answer based on the user's question.

# SQL: {generated_sql}

# Sample Data:
# {table_sample}

# Question: {question}
# """
#             answer, _ = call_qwen(analysis_prompt)
#             return JsonResponse({
#                 "answer": answer or "✅ Query executed.",
#                 "query": generated_sql,
#                 "columns": df.columns.tolist(),
#                 "preview": df.head(10).to_dict(orient="records")
#             })

#         # Numeric column summary
#         numeric_col = guess_numeric_column(df)
#         if numeric_col:
#             total = df[numeric_col].sum()
#             return JsonResponse({
#                 "answer": f"The total of {numeric_col} is {total:,.2f}.",
#                 "query": generated_sql,
#                 "columns": df.columns.tolist(),
#                 "preview": df.head(10).to_dict(orient="records")
#             })

#         # Fallback: natural language summary
#         table_sample = df.head(20).to_csv(index=False)
#         summary_prompt = f"""You are a helpful assistant. Based on the SQL result below, give a plain English summary.

# SQL: {generated_sql}

# Result Data:
# {table_sample}

# User's Question: {question}
# """
#         answer, _ = call_qwen(summary_prompt)

#         return JsonResponse({
#             "answer": answer or "✅ Query executed.",
#             "query": generated_sql,
#             "columns": df.columns.tolist(),
#             "preview": df.head(10).to_dict(orient="records")
#         })

#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)

# ++++++++

# @csrf_exempt
# def ask_qwen_db(request):
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Only POST allowed'}, status=405)

#     try:
#         data = json.loads(request.body)
#         question = data.get('question')
#         db_config = data.get('db_config')

#         if not question or not db_config:
#             return JsonResponse({'error': 'Missing question or db_config.'}, status=400)

#         host = db_config.get('host')
#         port = int(db_config.get('port', 5432))
#         database = db_config.get('database')
#         user = db_config.get('user')
#         password = db_config.get('password')
#         safe_password = urllib.parse.quote_plus(password)
#         db_url = f"postgresql://{user}:{safe_password}@{host}:{port}/{database}"
#         engine = create_engine(db_url)

#         schema_match = re.search(r'from\s+([\w]+)[.]', question, re.IGNORECASE) or \
#                         re.search(r'in\s+schema\s+([\w]+)', question, re.IGNORECASE)
#         requested_schema = schema_match.group(1) if schema_match else None

#         if "schema" in question.lower() and "how many" in question.lower():
#             schemas = [s for s in inspect(engine).get_schema_names() if s not in ['information_schema', 'pg_catalog']]
#             return JsonResponse({
#                 "answer": f"There are {len(schemas)} user-defined schemas: {', '.join(schemas)}.",
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         if "how many tables" in question.lower() or "total tables" in question.lower():
#             target_schema = requested_schema or 'public'
#             tables = inspect(engine).get_table_names(schema=target_schema)
#             return JsonResponse({
#                 "answer": f"There are {len(tables)} tables in schema '{target_schema}'.",
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         if any(x in question.lower() for x in ["table name", "table names", "show tables", "what tables", "list tables"]):
#             target_schema = requested_schema or 'public'
#             tables = inspect(engine).get_table_names(schema=target_schema)
#             return JsonResponse({
#                 "answer": f"The tables in schema '{target_schema}' are: {', '.join(tables)}.",
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening",
#                      "how are you", "what's up", "howdy", "nice to meet you", "good to see you", "yo", "sup",
#                      "hey there", "hiya", "😊", "👋", "🙋"]
#         if any(greet in question.lower() for greet in greetings):
#             prompt = f"You are a helpful assistant. {question}"
#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:7b",
#                 "prompt": prompt,
#                 "stream": False
#             })
#             return JsonResponse({
#                 "answer": response.json().get("response", "🤖 No answer."),
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         schema_structure = get_full_schema_structure(engine, filter_schema=requested_schema)
#         # structure_text = format_schema_for_prompt(schema_structure)
#         structure_text = format_schema_for_prompt(schema_structure, engine)


#         sql_prompt = f"""
# You are a highly skilled PostgreSQL assistant.

# Your task is to generate an accurate and valid SQL `SELECT` query that answers the user's question using the provided database schema.

# ### IMPORTANT RULES:
# Use only the tables and columns listed below.
# - Use fully qualified table names in the format: schema.table
# - Use exact column names — DO NOT assume columns exist.
# - Use the most appropriate table and column based on the sample values and descriptions.
# - Prefer columns like `sales`, `amount`, `price`, or `total` for aggregations.
# - Use `car` (not `model`) if the question refers to brand names like Honda, BMW, etc.
# - Do not use JOINs unless the user specifically asks for it (e.g., “combine employee and sales”).
# - If a string match is needed, use `= 'value'` for exact or `ILIKE '%value%'` for flexible matching.
# - Avoid referencing tables or columns that are not in the provided schema.
# - Do not explain anything — only return a raw SQL query.
# - ✅ Use only the tables and columns listed in the schema below.
# - ✅ Always reference columns using **exact names**.
# - ✅ Use the most contextually appropriate column (e.g., `car` for brand names like Honda, `sales` for totals).
# - ✅ Use fully qualified table names: `schema.table`
# - ✅ Use string match conditions like `= 'value'` or `ILIKE '%value%'` depending on what the question implies.
# - ❌ DO NOT use JOINs unless the user specifically asks to combine tables (e.g., “combine”, “merge”, “join”, “from both”).
# - ❌ NEVER use columns that are not explicitly listed.
# - ❌ DO NOT explain anything. Return only the raw SQL query.

# ### ADDITIONAL CONTEXT INSTRUCTIONS:
# - If the question refers to brands like "BMW", "Honda", "Toyota", match using the `car` column (not `model`) if present.
# - If the question asks for totals, counts, or aggregates, prefer numeric fields such as `sales`, `amount`, `salary`, `count`, or `price`.
# - If the question asks for employee details, check `employee` tables and relevant text fields like `name`, `department`, or `salary`.
# - For time-based queries (e.g., "this year", "recent", "2023"), use columns like `year`, `date`, or `created_at`.

# ---

# ### DATABASE STRUCTURE:

# {structure_text}

# ---

# ### USER QUESTION:
# {question}

# Return ONLY the raw SQL query (no explanation).
# """




#         sql_response = requests.post("http://localhost:11434/api/generate", json={
#             "model": "qwen2.5:7b",
#             "prompt": sql_prompt,
#             "stream": False
#         })

#         generated_sql = sql_response.json().get("response", "").replace("```sql", "").replace("```", "").strip()
#         print(f"Generated SQL: {generated_sql}")

#         if not generated_sql.lower().startswith("select"):
#             fallback_prompt = f"""
# You are a PostgreSQL assistant. Answer the user's question below based on PostgreSQL data.

# If SQL cannot be generated, provide a natural language explanation.
# Avoid making assumptions. Do not refer to the internet or market conditions.

# User Question: {question}
# """
#             fallback_response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:7b",
#                 "prompt": fallback_prompt,
#                 "stream": False
#             })
#             return JsonResponse({
#                 "answer": fallback_response.json().get("response", "🤖 Could not answer."),
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         try:
#             df = pd.read_sql_query(generated_sql, engine)
#         except Exception as sql_err:
#             return JsonResponse({'error': f'SQL execution failed: {str(sql_err)}', 'query': generated_sql}, status=500)

#         if df.empty:
#             return JsonResponse({
#                 "answer": f"The query ran but returned no matching results for: {question}",
#                 "query": generated_sql,
#                 "columns": [],
#                 "preview": []
#             })
        
#         # Optional block for unwanted joins
#         if " join " in generated_sql.lower() and "join" not in question.lower():
#             return JsonResponse({
#                 "error": "Unsafe JOIN generated by model (no JOIN requested).",
#                 "query": generated_sql
#             }, status=400)


#         if 'group by' in generated_sql.lower():
#             table_sample = df.head(20).to_csv(index=False)
#             final_prompt = f"""
# You are a data assistant. Analyze the SQL output and provide a final clear answer in natural language based on the user's question.

# SQL: {generated_sql}

# Table Output:
# {table_sample}

# User Question: {question}
# """
#             answer_response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:7b",
#                 "prompt": final_prompt,
#                 "stream": False
#             })
#             return JsonResponse({
#                 "answer": answer_response.json().get("response", "✅ Query executed."),
#                 "query": generated_sql,
#                 "columns": df.columns.tolist(),
#                 "preview": df.head(10).to_dict(orient="records")
#             })

#         if df.shape[1] == 1 and str(df.columns[0]).lower() in ('count', 'cnt') and pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
#             count_val = df.iloc[0, 0]
#             if float(count_val).is_integer():
#                 count_val = int(count_val)
#             return JsonResponse({
#                 "answer": f"The total count is {count_val}.",
#                 "query": generated_sql,
#                 "columns": df.columns.tolist(),
#                 "preview": df.head(10).to_dict(orient="records")
#             })

#         numeric_col = guess_numeric_column(df)
#         if numeric_col:
#             total = df[numeric_col].sum()
#             return JsonResponse({
#                 "answer": f"The total of `{numeric_col}` is {total:,.2f}.",
#                 "query": generated_sql,
#                 "columns": df.columns.tolist(),
#                 "preview": df.head(10).to_dict(orient="records")
#             })

#         table_sample = df.head(20).to_csv(index=False)
#         final_prompt = f"""
# Based on the following SQL output and user's question, provide a clear and final answer only — no SQL.

# SQL: {generated_sql}

# Table Output:
# {table_sample}

# User Question: {question}
# """
#         answer_response = requests.post("http://localhost:11434/api/generate", json={
#             "model": "qwen2.5:7b",
#             "prompt": final_prompt,
#             "stream": False
#         })

#         return JsonResponse({
#             "answer": answer_response.json().get("response", "✅ Query executed."),
#             "query": generated_sql,
#             "columns": df.columns.tolist(),
#             "preview": df.head(10).to_dict(orient="records")
#         })

#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)

# +++++++++

# import json
# import re
# import urllib.parse
# import pandas as pd
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from sqlalchemy import create_engine, inspect
# import requests

# # Dummy helper – use your actual logic
# def get_full_schema_structure(engine, filter_schema=None):
#     inspector = inspect(engine)
#     structure = {}
#     schemas = inspector.get_schema_names()
#     for schema in schemas:
#         if schema in ['information_schema', 'pg_catalog']:
#             continue
#         if filter_schema and schema != filter_schema:
#             continue
#         tables = inspector.get_table_names(schema=schema)
#         structure[schema] = {}
#         for table in tables:
#             columns = inspector.get_columns(table, schema=schema)
#             structure[schema][table] = [col['name'] for col in columns]
#     return structure

# def format_schema_for_prompt(schema_dict, engine=None):
#     lines = []
#     for schema, tables in schema_dict.items():
#         for table, columns in tables.items():
#             lines.append(f"{schema}.{table}: {', '.join(columns)}")
#     return "\n".join(lines)

# def guess_numeric_column(df):
#     for col in df.columns:
#         if pd.api.types.is_numeric_dtype(df[col]):
#             return col
#     return None

# @csrf_exempt
# def ask_qwen_db(request):
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Only POST allowed'}, status=405)

#     try:
#         data = json.loads(request.body)
#         question = data.get('question')
#         db_config = data.get('db_config')

#         if not question or not db_config:
#             return JsonResponse({'error': 'Missing question or db_config.'}, status=400)

#         host = db_config.get('host')
#         port = int(db_config.get('port', 5432))
#         database = db_config.get('database')
#         user = db_config.get('user')
#         password = db_config.get('password')
#         safe_password = urllib.parse.quote_plus(password)
#         db_url = f"postgresql://{user}:{safe_password}@{host}:{port}/{database}"
#         engine = create_engine(db_url)

#         schema_match = re.search(r'from\s+([\w]+)[.]', question, re.IGNORECASE) or \
#                        re.search(r'in\s+schema\s+([\w]+)', question, re.IGNORECASE)
#         requested_schema = schema_match.group(1) if schema_match else None

#         # Schema-level queries
#         if "schema" in question.lower() and "how many" in question.lower():
#             schemas = [s for s in inspect(engine).get_schema_names() if s not in ['information_schema', 'pg_catalog']]
#             return JsonResponse({
#                 "answer": f"There are {len(schemas)} user-defined schemas: {', '.join(schemas)}.",
#                 "query": None, "columns": [], "preview": []
#             })

#         if "how many tables" in question.lower() or "total tables" in question.lower():
#             target_schema = requested_schema or 'public'
#             tables = inspect(engine).get_table_names(schema=target_schema)
#             return JsonResponse({
#                 "answer": f"There are {len(tables)} tables in schema '{target_schema}'.",
#                 "query": None, "columns": [], "preview": []
#             })

#         if any(x in question.lower() for x in ["table name", "table names", "show tables", "what tables", "list tables"]):
#             target_schema = requested_schema or 'public'
#             tables = inspect(engine).get_table_names(schema=target_schema)
#             return JsonResponse({
#                 "answer": f"The tables in schema '{target_schema}' are: {', '.join(tables)}.",
#                 "query": None, "columns": [], "preview": []
#             })

#         # Greeting handler
#         greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
#         if any(greet in question.lower() for greet in greetings):
#             prompt = f"You are a helpful assistant. {question}"
#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:7b", "prompt": prompt, "stream": False
#             })
#             return JsonResponse({
#                 "answer": response.json().get("response", "🤖 No answer."),
#                 "query": None, "columns": [], "preview": []
#             })

#         schema_structure = get_full_schema_structure(engine, filter_schema=None)
#         structure_text = format_schema_for_prompt(schema_structure)

#         sql_prompt = f"""
# You are a highly skilled PostgreSQL assistant.

# Your task is to generate an accurate SQL SELECT query that answers the user's question using the database schema.

# ### RULES:
# - Use only the tables and columns listed below.
# - Use fully qualified table names like schema.table
# - Do NOT assume columns exist — use exact names only.
# - Do NOT include JOIN unless the question asks for it.
# - Return ONLY a raw SQL SELECT query.

# --- SCHEMA ---
# {structure_text}

# --- USER QUESTION ---
# {question}

# Return only a valid raw SQL SELECT query.
# """

#         sql_response = requests.post("http://localhost:11434/api/generate", json={
#             "model": "qwen2.5:7b", "prompt": sql_prompt, "stream": False
#         })

#         # Extract first SELECT line from response
#         raw_output = sql_response.json().get("response", "")
#         sql_only = None
#         for line in raw_output.splitlines():
#             line = line.strip().rstrip(";")
#             if line.lower().startswith("select"):
#                 sql_only = line + ";"
#                 break

#         if not sql_only:
#             fallback_prompt = f"""
# You are a PostgreSQL assistant. Provide a natural answer to the question if no SQL is possible.

# User Question: {question}
# """
#             fallback_response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:7b", "prompt": fallback_prompt, "stream": False
#             })
#             return JsonResponse({
#                 "answer": fallback_response.json().get("response", "🤖 Could not answer."),
#                 "query": None, "columns": [], "preview": []
#             })

#         print(f"Generated SQL: {sql_only}")
#         if " join " in sql_only.lower() and "join" not in question.lower():
#             return JsonResponse({
#                 "error": "Unsafe JOIN generated without user requesting join.",
#                 "query": sql_only
#             }, status=400)

#         try:
#             df = pd.read_sql_query(sql_only, engine)
#         except Exception as sql_err:
#             return JsonResponse({'error': f'SQL execution failed: {str(sql_err)}', 'query': sql_only}, status=500)

#         if df.empty:
#             return JsonResponse({
#                 "answer": "Query returned no results.",
#                 "query": sql_only, "columns": [], "preview": []
#             })

#         if 'group by' in sql_only.lower():
#             sample = df.head(20).to_csv(index=False)
#             final_prompt = f"""
# You are a smart assistant. Answer the user's question clearly based on the SQL output.

# SQL: {sql_only}

# Table Output:
# {sample}

# User Question: {question}
# """
#             answer_response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:7b", "prompt": final_prompt, "stream": False
#             })
#             return JsonResponse({
#                 "answer": answer_response.json().get("response", "✅ Query executed."),
#                 "query": sql_only,
#                 "columns": df.columns.tolist(),
#                 "preview": df.head(10).to_dict(orient="records")
#             })

#         if df.shape[1] == 1 and str(df.columns[0]).lower() in ('count', 'cnt') and pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
#             val = df.iloc[0, 0]
#             return JsonResponse({
#                 "answer": f"The total count is {int(val) if float(val).is_integer() else val}.",
#                 "query": sql_only,
#                 "columns": df.columns.tolist(),
#                 "preview": df.head(10).to_dict(orient="records")
#             })

#         numeric_col = guess_numeric_column(df)
#         if numeric_col:
#             total = df[numeric_col].sum()
#             return JsonResponse({
#                 "answer": f"The total of {numeric_col} is {total:,.2f}.",
#                 "query": sql_only,
#                 "columns": df.columns.tolist(),
#                 "preview": df.head(10).to_dict(orient="records")
#             })

#         return JsonResponse({
#             "answer": "✅ Query executed successfully.",
#             "query": sql_only,
#             "columns": df.columns.tolist(),
#             "preview": df.head(10).to_dict(orient="records")
#         })

#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)


# @csrf_exempt
# def ask_qwen_db(request):
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Only POST allowed'}, status=405)

#     try:
#         data = json.loads(request.body)
#         question = data.get('question')
#         db_config = data.get('db_config')

#         if not question or not db_config:
#             return JsonResponse({'error': 'Missing question or db_config.'}, status=400)

#         host = db_config.get('host')
#         port = int(db_config.get('port', 5432))
#         database = db_config.get('database')
#         user = db_config.get('user')
#         password = db_config.get('password')
#         safe_password = urllib.parse.quote_plus(password)
#         db_url = f"postgresql://{user}:{safe_password}@{host}:{port}/{database}"
#         engine = create_engine(db_url)

#         schema_match = re.search(r'from\s+([\w]+)[.]', question, re.IGNORECASE) or \
#                         re.search(r'in\s+schema\s+([\w]+)', question, re.IGNORECASE)
#         requested_schema = schema_match.group(1) if schema_match else None

#         if "schema" in question.lower() and "how many" in question.lower():
#             schemas = [s for s in inspect(engine).get_schema_names() if s not in ['information_schema', 'pg_catalog']]
#             return JsonResponse({
#                 "answer": f"There are {len(schemas)} user-defined schemas: {', '.join(schemas)}.",
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         if "how many tables" in question.lower():
#             target_schema = requested_schema or 'public'
#             tables = inspect(engine).get_table_names(schema=target_schema)
#             return JsonResponse({
#                 "answer": f"There are {len(tables)} tables in schema '{target_schema}'.",
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         if "table name" in question.lower() or "table names" in question.lower():
#             target_schema = requested_schema or 'public'
#             tables = inspect(engine).get_table_names(schema=target_schema)
#             return JsonResponse({
#                 "answer": f"The tables in schema '{target_schema}' are: {', '.join(tables)}.",
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening",
#                      "how are you", "what's up", "howdy", "nice to meet you", "good to see you", "yo", "sup",
#                      "hey there", "hiya", "😊", "👋", "🙋"]
#         if any(greet in question.lower() for greet in greetings):
#             prompt = f"You are a helpful assistant. {question}"
#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": prompt,
#                 "stream": False
#             })
#             return JsonResponse({
#                 "answer": response.json().get("response", "🤖 No answer."),
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         schema_structure = get_full_schema_structure(engine, filter_schema=requested_schema)
#         structure_text = format_schema_for_prompt(schema_structure)

#         sql_prompt = f"""
# You are a PostgreSQL expert. The user has asked a question. Based on the schema and tables below, write a valid SELECT SQL query to answer it.

# Instructions:
# - Only use the columns and tables listed below.
# - Use fully qualified table names like schema.table.
# - Prefer numeric columns like `sales`, `amount`, or `total` if aggregation is needed.
# - Do not explain anything. Only return raw SQL.

# {structure_text}

# Question: {question}
# """

#         sql_response = requests.post("http://localhost:11434/api/generate", json={
#             "model": "qwen2.5:1.5b",
#             "prompt": sql_prompt,
#             "stream": False
#         })

#         generated_sql = sql_response.json().get("response", "").replace("```sql", "").replace("```", "").strip()
#         print(f"Generated SQL: {generated_sql}")

#         if not generated_sql.lower().startswith("select"):
#             fallback_prompt = f"The user asked: {question}\n\n{f'Note: The context is the `{requested_schema}` schema.' if requested_schema else ''}\n\nAnswer in natural language."
#             fallback_response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": fallback_prompt,
#                 "stream": False
#             })
#             return JsonResponse({
#                 "answer": fallback_response.json().get("response", "🤖 Could not answer."),
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         try:
#             df = pd.read_sql_query(generated_sql, engine)
#         except Exception as sql_err:
#             return JsonResponse({
#                 'error': f'SQL execution failed: {str(sql_err)}',
#                 'query': generated_sql
#             }, status=500)

#         if df.empty:
#             return JsonResponse({
#                 "answer": f"The query ran but returned no matching results for: {question}",
#                 "query": generated_sql,
#                 "columns": [],
#                 "preview": []
#             })

#         if 'group by' in generated_sql.lower():
#             table_sample = df.head(20).to_csv(index=False)
#             final_prompt = f"""
# You are a data assistant. Analyze the SQL output and provide a final clear answer in natural language based on the user's question.

# SQL: {generated_sql}

# Table Output:
# {table_sample}

# User Question: {question}
# """
#             answer_response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": final_prompt,
#                 "stream": False
#             })
#             return JsonResponse({
#                 "answer": answer_response.json().get("response", "✅ Query executed."),
#                 "query": generated_sql,
#                 "columns": df.columns.tolist(),
#                 "preview": df.head(10).to_dict(orient="records")
#             })

#         # Special case: if the result is a count
#         if df.shape[1] == 1 and str(df.columns[0]).lower() in ('count', 'cnt') and pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
#             # count_val = int(df.iloc[0, 0])
#             count_val = df.iloc[0, 0]
#             if float(count_val).is_integer():
#                 count_val = int(count_val)

#             return JsonResponse({
#                 "answer": f"The total count is {count_val}.",
#                 "query": generated_sql,
#                 "columns": df.columns.tolist(),
#                 "preview": df.head(10).to_dict(orient="records")
#             })

#         numeric_col = guess_numeric_column(df)
#         if numeric_col:
#             total = df[numeric_col].sum()
#             return JsonResponse({
#                 "answer": f"The total of `{numeric_col}` is {total:,.2f}.",
#                 "query": generated_sql,
#                 "columns": df.columns.tolist(),
#                 "preview": df.head(10).to_dict(orient="records")
#             })

#         table_sample = df.head(20).to_csv(index=False)
#         final_prompt = f"""
# Based on the following SQL output and user's question, provide a clear and final answer only — no SQL.

# SQL: {generated_sql}

# Table Output:
# {table_sample}

# User Question: {question}
# """
#         answer_response = requests.post("http://localhost:11434/api/generate", json={
#             "model": "qwen2.5:1.5b",
#             "prompt": final_prompt,
#             "stream": False
#         })

#         return JsonResponse({
#             "answer": answer_response.json().get("response", "✅ Query executed."),
#             "query": generated_sql,
#             "columns": df.columns.tolist(),
#             "preview": df.head(10).to_dict(orient="records")
#         })

#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)

# @csrf_exempt
# def ask_qwen_db(request):
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Only POST allowed'}, status=405)

#     try:
#         data = json.loads(request.body)
#         question = data.get('question')
#         db_config = data.get('db_config')

#         if not question or not db_config:
#             return JsonResponse({'error': 'Missing question or db_config.'}, status=400)

#         host = db_config.get('host')
#         port = int(db_config.get('port', 5432))
#         database = db_config.get('database')
#         user = db_config.get('user')
#         password = db_config.get('password')
#         safe_password = urllib.parse.quote_plus(password)
#         db_url = f"postgresql://{user}:{safe_password}@{host}:{port}/{database}"
#         engine = create_engine(db_url)

#         schema_match = re.search(r'from\s+([\w]+)[.]', question, re.IGNORECASE) or \
#                         re.search(r'in\s+schema\s+([\w]+)', question, re.IGNORECASE)
#         requested_schema = schema_match.group(1) if schema_match else None

#         if "schema" in question.lower() and "how many" in question.lower():
#             schemas = [s for s in inspect(engine).get_schema_names() if s not in ['information_schema', 'pg_catalog']]
#             return JsonResponse({
#                 "answer": f"There are {len(schemas)} user-defined schemas: {', '.join(schemas)}.",
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         table_count_match = re.search(r'how many tables.*schema\s+(\w+)', question.lower())
#         if table_count_match:
#             target_schema = table_count_match.group(1)
#             tables = inspect(engine).get_table_names(schema=target_schema)
#             return JsonResponse({
#                 "answer": f"Schema '{target_schema}' contains {len(tables)} tables: {', '.join(tables)}.",
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         greetings = ["hi", "hello", "hey", "greetings",
#                      "good morning", "good afternoon", "good evening",
#                      "how are you", "what's up", "howdy", "nice to meet you",
#                      "good to see you", "yo", "sup", "hey there", "hiya", "😊", "👋", "🙋"]
#         if any(greet in question.lower() for greet in greetings):
#             prompt = f"You are a helpful assistant. {question}"
#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": prompt,
#                 "stream": False
#             })
#             return JsonResponse({
#                 "answer": response.json().get("response", "🤖 No answer."),
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         schema_structure = get_full_schema_structure(engine, filter_schema=requested_schema)
#         structure_text = format_schema_for_prompt(schema_structure)

#         sql_prompt = f"""
# You are a PostgreSQL expert. The user has asked a question. Based on the schema and tables below, write a valid SELECT SQL query to answer it.

# Instructions:
# - Only use the columns and tables listed below.
# - Use fully qualified table names like schema.table.
# - Prefer numeric columns like `sales`, `amount`, or `total` if aggregation is needed.
# - Do not explain anything. Only return raw SQL.

# {structure_text}

# Question: {question}
# """



#         sql_response = requests.post("http://localhost:11434/api/generate", json={
#             "model": "qwen2.5:1.5b",
#             "prompt": sql_prompt,
#             "stream": False
#         })

#         generated_sql = sql_response.json().get("response", "").replace("```sql", "").replace("```", "").strip()
#         print(f"Generated SQL: {generated_sql}")


#         if not generated_sql.lower().startswith("select"):
#             fallback_prompt = f"The user asked: {question}\n\n{f'Note: The context is the `{requested_schema}` schema.' if requested_schema else ''}\n\nAnswer in natural language."
#             fallback_response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": fallback_prompt,
#                 "stream": False
#             })
#             return JsonResponse({
#                 "answer": fallback_response.json().get("response", "🤖 Could not answer."),
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         df = pd.read_sql_query(generated_sql, engine)

#         if df.empty:
#             return JsonResponse({
#                 "answer": f"The query ran but returned no matching results for: {question}",
#                 "query": generated_sql,
#                 "columns": [],
#                 "preview": []
#             })

#         # Attempt to compute numeric summary directly for grouped results
#         if 'group by' in generated_sql.lower():
#             table_sample = df.head(20).to_csv(index=False)
#             final_prompt = f"""
# You are a data assistant. Analyze the SQL output and provide a final clear answer in natural language based on the user's question.

# SQL: {generated_sql}

# Table Output:
# {table_sample}

# User Question: {question}
# """
#             answer_response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": final_prompt,
#                 "stream": False
#             })
#             return JsonResponse({
#                 "answer": answer_response.json().get("response", "✅ Query executed."),
#                 "query": generated_sql,
#                 "columns": df.columns.tolist(),
#                 "preview": df.head(10).to_dict(orient="records")
#             })

#         numeric_col = guess_numeric_column(df)
#         if numeric_col:
#             total = df[numeric_col].sum()
#             return JsonResponse({
#                 "answer": f"The total of `{numeric_col}` is {total:,.2f}.",
#                 "query": generated_sql,
#                 "columns": df.columns.tolist(),
#                 "preview": df.head(10).to_dict(orient="records")
#             })

#         table_sample = df.head(20).to_csv(index=False)
#         final_prompt = f"""
# Based on the following SQL output and user's question, provide a clear and final answer only — no SQL.

# SQL: {generated_sql}

# Table Output:
# {table_sample}

# User Question: {question}
# """
#         answer_response = requests.post("http://localhost:11434/api/generate", json={
#             "model": "qwen2.5:1.5b",
#             "prompt": final_prompt,
#             "stream": False
#         })

#         return JsonResponse({
#             "answer": answer_response.json().get("response", "✅ Query executed."),
#             "query": generated_sql,
#             "columns": df.columns.tolist(),
#             "preview": df.head(10).to_dict(orient="records")
#         })

#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)


# @csrf_exempt
# def ask_qwen_db(request):
#     print("Received request:")
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Only POST allowed'}, status=405)

#     try:
#         data = json.loads(request.body)
#         question = data.get('question')
#         db_config = data.get('db_config')

#         if not question or not db_config:
#             return JsonResponse({'error': 'Missing question or db_config.'}, status=400)

#         host = db_config.get('host')
#         port = int(db_config.get('port', 5432))
#         database = db_config.get('database')
#         user = db_config.get('user')
#         password = db_config.get('password')
#         safe_password = urllib.parse.quote_plus(password)
#         db_url = f"postgresql://{user}:{safe_password}@{host}:{port}/{database}"
#         engine = create_engine(db_url)

#         # ⛳ Handle "how many schemas" question directly
#         if "schema" in question.lower() and "how many" in question.lower():
#             schemas = [s for s in inspect(engine).get_schema_names() if s not in ['information_schema', 'pg_catalog']]
#             return JsonResponse({
#                 "answer": f"There are {len(schemas)} user-defined schemas: {', '.join(schemas)}.",
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         # 💡 Greeting/general response
#         general_phrases = [
#                 "hi",
#                 "hello",
#                 "hey",
#                 "good morning",
#                 "good afternoon",
#                 "good evening",
#                 "greetings",
#                 "how are you",
#                 "what's up",
#                 "howdy",
#                 "nice to meet you",
#                 "good to see you",  "yo", "sup", "hey there", "hiya", "😊", "👋", "🙋"
#             ]
#         if any(greet in question.lower() for greet in general_phrases):
#             prompt = f"You are a helpful assistant. {question}"
#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": prompt,
#                 "stream": False
#             })
#             return JsonResponse({
#                 "answer": response.json().get("response", "🤖 No answer."),
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         # 🧠 Full schema for SQL generation
#         schema_structure = get_full_schema_structure(engine)
#         structure_text = format_schema_for_prompt(schema_structure)

#         sql_prompt = f"""
# Convert the following natural language question into a raw SQL SELECT statement that runs on PostgreSQL. 
# Do not explain, just return the SQL.

# {structure_text}

# Question: {question}
# """
#         sql_response = requests.post("http://localhost:11434/api/generate", json={
#             "model": "qwen2.5:1.5b",
#             "prompt": sql_prompt,
#             "stream": False
#         })

#         generated_sql = sql_response.json().get("response", "").replace("```sql", "").replace("```", "").strip()

#         # Fallback to natural answer if no valid SQL
#         if not generated_sql.lower().startswith("select"):
#             fallback_prompt = f"User asked: {question}\nAnswer naturally:"
#             fallback_response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": fallback_prompt,
#                 "stream": False
#             })
#             return JsonResponse({
#                 "answer": fallback_response.json().get("response", "🤖 Could not answer."),
#                 "query": None,
#                 "columns": [],
#                 "preview": []
#             })

#         # 🔄 Execute SQL
#         df = pd.read_sql_query(generated_sql, engine)
#         table_sample = df.head(20).to_csv(index=False)
#         final_prompt = f"""
# Use the below data to answer the user's question.

# SQL: {generated_sql}

# Data:
# {table_sample}

# Question: {question}
# """
#         answer_response = requests.post("http://localhost:11434/api/generate", json={
#             "model": "qwen2.5:1.5b",
#             "prompt": final_prompt,
#             "stream": False
#         })

#         return JsonResponse({
#             "answer": answer_response.json().get("response", "✅ Query executed."),
#             "query": generated_sql,
#             "columns": df.columns.tolist(),
#             "preview": df.head(10).to_dict(orient="records")
#         })

#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)


# @csrf_exempt
# @csrf_exempt
# def ask_qwen_db(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             question = data.get('question')
#             db_config = data.get('db_config')

#             if not question or not db_config:
#                 return JsonResponse({'error': 'Missing question or db_config.'}, status=400)

#             host = db_config.get('host')
#             port = int(db_config.get('port', 5432))
#             database = db_config.get('database')
#             user = db_config.get('user')
#             password = db_config.get('password')

#             db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
#             engine = create_engine(db_url)

#             # ✅ If it's just a test connection, skip LLM and return a simple query
#             if question.strip().lower() == "test connection":
#                 df = pd.read_sql_query("SELECT version();", engine)
#                 return JsonResponse({
#                     "answer": "✅ Database connected successfully.",
#                     "query": "SELECT version();",
#                     "columns": df.columns.tolist(),
#                     "preview": df.head(5).to_dict(orient="records")
#                 })

#             # 🔄 Rest of your LLM logic continues below...


#             tables = inspect(engine).get_table_names()

#             if not tables:
#                 return JsonResponse({'error': 'No tables found in database.'}, status=400)

#             sql_prompt = f"Convert this natural language question to SQL:\nQuestion: {question}\nTables: {', '.join(tables)}"
#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": sql_prompt,
#                 "stream": False
#             })

#             sql = response.json().get("response", "").strip().replace("```sql", "").replace("```", "").strip()

#             if not sql.lower().startswith("select"):
#                 return JsonResponse({'error': f"Invalid SQL generated: {sql}"}, status=400)

#             df = pd.read_sql_query(sql, engine)

#             preview = df.head(10).to_dict(orient='records')

#             return JsonResponse({
#                 'query': sql,
#                 'answer': f"Fetched {len(df)} rows.",
#                 'columns': df.columns.tolist(),
#                 'preview': preview
#             })

#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Only POST allowed'}, status=405)




# import pandas as pd
# import psycopg2
# import requests
# import json
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt

# from sqlalchemy import create_engine

# def fetch_table_data(query, host, port, db_name, db_user, db_password):
#     try:
#         print("🟢 Connecting to database...")
#         db_url = f"postgresql://{db_user}:{db_password}@{host}:{port}/{db_name}"
#         engine = create_engine(db_url)
#         df = pd.read_sql_query(query, engine)
#         return df
#     except Exception as e:
#         print("❌ Database Error:", str(e))
#         return None


# # 🔹 Build prompt for Qwen using table data and user question
# def build_prompt_from_db(df, user_question):
#     if df is None or df.empty:
#         return "The database returned no results."

#     table_sample = df.head(20).to_csv(index=False)
#     column_info = ', '.join(df.columns)

#     prompt = f"""
# You are an intelligent data assistant.

# Table Schema: {column_info}

# Here is the sample data:
# {table_sample}

# Question: {user_question}

# Answer in plain English using only the data above.
# """
#     return prompt


# # 🔹 Call Qwen via Ollama
# def ask_qwen(prompt):
#     try:
#         response = requests.post("http://localhost:11434/api/generate", json={
#             "model": "qwen2.5:1.5b",
#             "prompt": prompt,
#             "stream": False
#         })
#         return response.json().get('response', '❌ No response from Qwen.')
#     except Exception as e:
#         return f"❌ Error calling Qwen: {str(e)}"


# # 🔹 Django endpoint to accept DB config, query, and user question
# @csrf_exempt
# def ask_from_database_view(request):
#     print("Starting to connect to database...")
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)

#             # Extract required fields
#             question = data.get('question')
#             query = data.get('query')
#             host = data.get('host')
#             port = int(data.get('port', 5432))  # default to 5432
#             database = data.get('database')
#             user = data.get('user')
#             password = data.get('password')

#             # Validation
#             if not all([question, query, host, port, database, user, password]):
#                 return JsonResponse({'error': 'Missing required fields (question, query, host, port, database, user, password).'}, status=400)

#             # Fetch data from DB
#             df = fetch_table_data(query, host, port, database, user, password)

#             # Build prompt and ask Qwen
#             prompt = build_prompt_from_db(df, question)
#             answer = ask_qwen(prompt)

#             return JsonResponse({
#                 'answer': answer,
#                 'prompt': prompt,
#                 'columns': df.columns.tolist() if df is not None else [],
#                 'preview': df.head(10).to_dict(orient='records') if df is not None else []
#             })

#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Only POST method is allowed.'}, status=405)



# # Create your views here.
# import os
# import uuid
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')  # ✅ Use non-GUI backend
# import matplotlib.pyplot as plt
# from django.conf import settings
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.core.files.storage import default_storage
# import requests
# import json
# import traceback


# # === Ensure these are set in settings.py ===
# # MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
# # MEDIA_URL = '/media/'
# import os
# import uuid
# import pandas as pd
# import matplotlib.pyplot as plt
# from django.conf import settings
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.core.files.storage import default_storage
# import requests
# import json

# @csrf_exempt
# def upload_file(request):
#     if request.method == 'POST' and request.FILES.get('file'):
#         file = request.FILES['file']
#         filename = default_storage.save(file.name, file)
#         filepath = os.path.join(settings.MEDIA_ROOT, filename)

#         try:
#             if filename.endswith(('.xls', '.xlsx')):
#                 df = pd.read_excel(filepath)
#             elif filename.endswith('.csv'):
#                 try:
#                     df = pd.read_csv(filepath, encoding='utf-8')
#                 except UnicodeDecodeError:
#                     df = pd.read_csv(filepath, encoding='ISO-8859-1')
#             elif filename.endswith('.tsv'):
#                 df = pd.read_csv(filepath, sep='\t')
#             elif filename.endswith('.xlsb'):
#                 df = pd.read_excel(filepath, engine='pyxlsb')
#             else:
#                 return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)

#             df = df.head(100)  # 👈 Keep only first 100 rows

#         except Exception as e:
#             import traceback
#             print("❌ Uncaught error in upload_file:", traceback.format_exc())
#             return JsonResponse({'error': f'Server crashed: {str(e)}'}, status=500)

#         preview = df.head(10).to_dict(orient='records')

#         chart_path = None
#         try:
#             if df.shape[1] >= 2:
#                 chart_path = os.path.join(settings.MEDIA_ROOT, f"chart_{uuid.uuid4()}.png")
#                 df.iloc[:, :2].value_counts().plot(kind='bar')
#                 plt.title("Auto Chart from First 2 Columns")
#                 plt.tight_layout()
#                 plt.savefig(chart_path)
#                 plt.close()
#         except Exception as e:
#             chart_path = None

#         return JsonResponse({
#             'message': 'File uploaded and chart generated.',
#             'filename': filename,
#             'columns': df.columns.tolist(),
#             'preview': preview,
#             'chart_url': settings.MEDIA_URL + os.path.basename(chart_path) if chart_path else None
#         })

#     return JsonResponse({'error': 'Invalid request'}, status=400)



# @csrf_exempt
# def ask_qwen(request):
#     print("🚀 Starting Ollama (Qwen2.5) in background...")
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             question = data.get('question', '').strip()
#             filename = data.get('filename', '').strip()

#             if not question:
#                 return JsonResponse({'error': '❗ Question is required.'}, status=400)

#             prompt = ""

#             # ✅ Case 1: With file context
#             if filename:
#                 filepath = os.path.join(settings.MEDIA_ROOT, filename)

#                 if not os.path.exists(filepath):
#                     return JsonResponse({'error': '❌ Uploaded file not found.'}, status=400)

#                 if filename.endswith(('.xls', '.xlsx')):
#                     df = pd.read_excel(filepath)
#                 elif filename.endswith('.csv'):
#                     try:
#                         df = pd.read_csv(filepath, encoding='utf-8')
#                     except UnicodeDecodeError:
#                         df = pd.read_csv(filepath, encoding='ISO-8859-1')
#                 elif filename.endswith('.tsv'):
#                     df = pd.read_csv(filepath, sep='\t')

#                 else:
#                     return JsonResponse({'error': '❌ Unsupported file format.'}, status=400)
#                 context = df.head(20).to_csv(index=False)
#                 prompt = f"Given the following data:\n{context}\n\nAnswer this: {question}"

#             else:
#                 # ✅ Case 2: No file, general question
#                 prompt = f"Answer the following question:\n{question}"

#             response = requests.post("http://localhost:11434/api/generate", json={
#                 "model": "qwen2.5:1.5b",
#                 "prompt": prompt,
#                 "stream": False
#             })

#             json_response = response.json()
#             answer = json_response.get('response', '❌ No response from Qwen model.')

#             return JsonResponse({
#                 'question': question,
#                 'answer': answer
#             })

#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Invalid request'}, status=400)


# from django.shortcuts import render

# def index(request):
#     return render(request, 'index.html')

