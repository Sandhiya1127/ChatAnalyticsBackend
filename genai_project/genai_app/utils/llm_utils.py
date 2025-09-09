# import json
# import os
# import requests
# import re
# from decimal import Decimal
# from json import JSONDecodeError

# # ✅ Use environment variable in production
# GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# def sanitize_for_json(data):
#     """Recursively convert Decimal to float for LLM compatibility."""
#     if isinstance(data, dict):
#         return {k: sanitize_for_json(v) for k, v in data.items()}
#     elif isinstance(data, list):
#         return [sanitize_for_json(v) for v in data]
#     elif isinstance(data, Decimal):
#         return float(data)
#     return data

# def clean_llm_json(raw_output):
#     """Remove markdown code fences or common formatting issues."""
#     raw_output = raw_output.strip()
#     if raw_output.startswith("```json"):
#         raw_output = raw_output[7:]
#     elif raw_output.startswith("```"):
#         raw_output = raw_output[3:]
#     if raw_output.endswith("```"):
#         raw_output = raw_output[:-3]
#     return raw_output.strip()

# def call_llm(prompt):
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": 0.1,
#         "max_tokens": 1000
#     }

#     response = requests.post(f"{GROQ_BASE_URL}/chat/completions", headers=headers, json=payload)
#     response.raise_for_status()

#     content = response.json()["choices"][0]["message"]["content"]
#     return clean_llm_json(content)

# def llm_generate_chart_config(question, rows):
#     try:
#         if len(rows) == 1 and len(rows[0]) == 1:
#             # Skip chart generation if it's just one value
#             return None

#         sanitized_rows = sanitize_for_json(rows[:10])
#         table_preview = json.dumps(sanitized_rows, indent=2)

#         chart_prompt = f"""
# You are a BI dashboard expert. Based on the user question and the following data (JSON array of rows),
# return a valid Highcharts JSON configuration. Automatically choose the best chart type:

# Rules:
# - Column chart for category comparisons
# - Line chart for time trends
# - Pie chart for proportions
# - Combo chart (e.g., line + bar) for dual metrics
# - Use `xAxis.categories` from data keys like dates or categories
# - Fill `series` with names and numeric data arrays
# - Do not add explanations or comments—return ONLY pure JSON.

# Question: {question}
# Data:
# {table_preview}
# """.strip()

#         raw_output = call_llm(chart_prompt)
#         return json.loads(raw_output)

#     except JSONDecodeError as json_err:
#         print("❌ JSON Parsing Error:", json_err)
#         print("⚠️ Raw Output was:", raw_output)
#         return {"error": "Chart generation failed", "raw": raw_output}
#     except Exception as e:
#         print("❌ LLM Chart Generation Error:", e)
#         return {"error": "Chart generation failed", "raw": str(e)}



import json
import os
import requests
import re
from decimal import Decimal
from json import JSONDecodeError

# ✅ Use environment variable in production
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# def sanitize_for_json(data):
#     """Recursively convert Decimal to float for LLM compatibility."""
#     if isinstance(data, dict):
#         return {k: sanitize_for_json(v) for k, v in data.items()}
#     elif isinstance(data, list):
#         return [sanitize_for_json(v) for v in data]
#     elif isinstance(data, Decimal):
#         return float(data)
#     return data

# def clean_llm_json(raw_output):
#     """Remove markdown code fences or common formatting issues."""
#     raw_output = raw_output.strip()
#     if raw_output.startswith("```json"):
#         raw_output = raw_output[7:]
#     elif raw_output.startswith("```"):
#         raw_output = raw_output[3:]
#     if raw_output.endswith("```"):
#         raw_output = raw_output[:-3]
#     return raw_output.strip()

# def call_llm(prompt):
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": 0.1,
#         "max_tokens": 1000
#     }

#     response = requests.post(f"{GROQ_BASE_URL}/chat/completions", headers=headers, json=payload)
#     response.raise_for_status()

#     content = response.json()["choices"][0]["message"]["content"]
#     return clean_llm_json(content)

# def create_chart_config_from_data(question, rows):
#     """Create chart config directly from data without LLM if LLM fails"""
#     try:
#         if not rows or len(rows) == 0:
#             return None
            
#         # Analyze the data structure
#         first_row = rows[0]
#         columns = list(first_row.keys())
        
#         # Find numeric columns
#         numeric_columns = []
#         category_columns = []
        
#         for col in columns:
#             sample_values = [row.get(col) for row in rows[:5]]  # Check first 5 rows
#             if all(isinstance(val, (int, float)) and not isinstance(val, bool) for val in sample_values if val is not None):
#                 numeric_columns.append(col)
#             else:
#                 category_columns.append(col)
        
#         if not numeric_columns:
#             return None
            
#         # Determine chart type and structure based on data
#         if len(category_columns) >= 2 and len(numeric_columns) >= 1:
#             # Grouped data (like policy_tenure + is_churn + policy_count)
#             primary_category = category_columns[0]  # policy_tenure
#             secondary_category = category_columns[1]  # is_churn
#             value_column = numeric_columns[0]  # policy_count
            
#             # Group data for series
#             categories = sorted(list(set(row[primary_category] for row in rows)))
#             series_names = sorted(list(set(row[secondary_category] for row in rows)))
            
#             series_data = []
#             for series_name in series_names:
#                 data_points = []
#                 for category in categories:
#                     # Find matching row
#                     matching_row = next((row for row in rows 
#                                        if row[primary_category] == category 
#                                        and row[secondary_category] == series_name), None)
#                     data_points.append(matching_row[value_column] if matching_row else 0)
                
#                 series_data.append({
#                     "name": str(series_name),
#                     "data": data_points
#                 })
            
#             return {
#                 "chart": {"type": "column"},
#                 "title": {"text": f"{question}"},
#                 "xAxis": {
#                     "categories": [str(cat) for cat in categories],
#                     "title": {"text": primary_category.replace('_', ' ').title()}
#                 },
#                 "yAxis": {
#                     "title": {"text": value_column.replace('_', ' ').title()}
#                 },
#                 "series": series_data
#             }
        
#         elif len(category_columns) == 1 and len(numeric_columns) >= 1:
#             # Simple category vs value chart
#             category_col = category_columns[0]
#             value_col = numeric_columns[0]
            
#             categories = [str(row[category_col]) for row in rows]
#             values = [row[value_col] for row in rows]
            
#             return {
#                 "chart": {"type": "column"},
#                 "title": {"text": f"{question}"},
#                 "xAxis": {
#                     "categories": categories,
#                     "title": {"text": category_col.replace('_', ' ').title()}
#                 },
#                 "yAxis": {
#                     "title": {"text": value_col.replace('_', ' ').title()}
#                 },
#                 "series": [{
#                     "name": value_col.replace('_', ' ').title(),
#                     "data": values
#                 }]
#             }
            
#         return None
        
#     except Exception as e:
#         print("❌ Fallback chart creation error:", e)
#         return None

# def llm_generate_chart_config(question, rows):
#     try:
#         if len(rows) == 1 and len(rows[0]) == 1:
#             # Skip chart generation if it's just one value
#             return None

#         sanitized_rows = sanitize_for_json(rows[:10])
#         table_preview = json.dumps(sanitized_rows, indent=2)

#         chart_prompt = f"""
# You are a BI dashboard expert. Based on the user question and the following data (JSON array of rows),
# return a valid Highcharts JSON configuration. 

# IMPORTANT RULES:
# 1. Return ONLY valid JSON - no explanations, no comments, no markdown
# 2. Use "column" type for category comparisons (like churn analysis)
# 3. Use "line" type for time series data
# 4. Use "pie" type only for proportions of a whole
# 5. For grouped data, create multiple series
# 6. Ensure xAxis.categories is an array of strings or numbers
# 7. Ensure series data arrays contain only numbers
# 8. Include proper titles and axis labels

# Question: {question}
# Data:
# {table_preview}

# Return valid Highcharts config JSON:
# """.strip()

#         raw_output = call_llm(chart_prompt)
#         chart_config = json.loads(raw_output)
        
#         # Validate the chart config structure
#         if not isinstance(chart_config, dict) or 'series' not in chart_config:
#             print("⚠️ Invalid chart config from LLM, using fallback")
#             return create_chart_config_from_data(question, rows)
            
#         # Ensure series data is properly formatted
#         if 'series' in chart_config:
#             for series in chart_config['series']:
#                 if 'data' in series:
#                     # Convert all data points to numbers
#                     series['data'] = [float(x) if x is not None else 0 for x in series['data']]
        
#         return chart_config

#     except JSONDecodeError as json_err:
#         print("❌ JSON Parsing Error:", json_err)
#         print("⚠️ Raw Output was:", raw_output)
#         # Fallback to programmatic chart creation
#         return create_chart_config_from_data(question, rows)
#     except Exception as e:
#         print("❌ LLM Chart Generation Error:", e)
#         # Fallback to programmatic chart creation
#         return create_chart_config_from_data(question, rows)

# # Update the main function call
# def generate_chart_and_response(question, rows):
#     """Main function to generate chart config with fallbacks"""
#     try:
#         chart_config = llm_generate_chart_config(question, rows)
#         print("📊 Chart config:", chart_config)
#         return chart_config
#     except Exception as chart_err:
#         print("⚠️ Chart generation failed:", chart_err)
#         return None

# .........runningchart
# def sanitize_for_json(data):
#     """Recursively convert Decimal to float for LLM compatibility."""
#     if isinstance(data, dict):
#         return {k: sanitize_for_json(v) for k, v in data.items()}
#     elif isinstance(data, list):
#         return [sanitize_for_json(v) for v in data]
#     elif isinstance(data, Decimal):
#         return float(data)
#     return data

# def clean_llm_json(raw_output):
#     """Remove markdown code fences or common formatting issues."""
#     raw_output = raw_output.strip()
#     if raw_output.startswith("```json"):
#         raw_output = raw_output[7:]
#     elif raw_output.startswith("```"):
#         raw_output = raw_output[3:]
#     if raw_output.endswith("```"):
#         raw_output = raw_output[:-3]
#     return raw_output.strip()

# def call_llm(prompt):
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": 0.1,
#         "max_tokens": 1000
#     }

#     response = requests.post(f"{GROQ_BASE_URL}/chat/completions", headers=headers, json=payload)
#     response.raise_for_status()

#     content = response.json()["choices"][0]["message"]["content"]
#     return clean_llm_json(content)

# def detect_chart_type(question, rows, columns, numeric_columns, category_columns):
#     """Intelligently detect the best chart type based on question and data"""
#     question_lower = question.lower()
    
#     # Time series detection
#     time_keywords = ['trend', 'over time', 'monthly', 'yearly', 'daily', 'weekly', 'timeline', 'growth', 'decline']
#     date_columns = [col for col in columns if any(keyword in col.lower() for keyword in ['date', 'time', 'month', 'year', 'day'])]
    
#     # Distribution/correlation detection
#     scatter_keywords = ['correlation', 'relationship', 'scatter', 'distribution', 'vs', 'against']
    
#     # Proportion detection
#     pie_keywords = ['percentage', 'proportion', 'share', 'distribution', 'breakdown', 'composition']
    
#     # Comparison detection
#     bar_keywords = ['compare', 'comparison', 'versus', 'top', 'bottom', 'ranking', 'highest', 'lowest']
    
#     # Geographic detection
#     geo_keywords = ['by state', 'by region', 'by country', 'by city', 'geographic', 'location']
    
#     # Check for time series
#     if any(keyword in question_lower for keyword in time_keywords) or date_columns:
#         return 'line'
    
#     # Check for scatter plot
#     if (any(keyword in question_lower for keyword in scatter_keywords) and 
#         len(numeric_columns) >= 2):
#         return 'scatter'
    
#     # Check for pie chart (only if we have proportional data and single category)
#     if (any(keyword in question_lower for keyword in pie_keywords) and 
#         len(category_columns) == 1 and len(numeric_columns) == 1 and len(rows) <= 10):
#         return 'pie'
    
#     # Check for horizontal bar chart
#     if (any(keyword in question_lower for keyword in bar_keywords) or 
#         any(keyword in question_lower for keyword in geo_keywords)):
#         return 'bar'
    
#     # Default to column chart for comparisons
#     return 'column'

# def create_chart_config_from_data(question, rows):
#     """Create chart config directly from data without LLM if LLM fails"""
#     try:
#         if not rows or len(rows) == 0:
#             return None
            
#         # Analyze the data structure
#         first_row = rows[0]
#         columns = list(first_row.keys())
        
#         # Find numeric columns
#         numeric_columns = []
#         category_columns = []
        
#         for col in columns:
#             sample_values = [row.get(col) for row in rows[:5]]  # Check first 5 rows
#             if all(isinstance(val, (int, float)) and not isinstance(val, bool) for val in sample_values if val is not None):
#                 numeric_columns.append(col)
#             else:
#                 category_columns.append(col)
        
#         if not numeric_columns:
#             return None
        
#         # Detect the best chart type
#         chart_type = detect_chart_type(question, rows, columns, numeric_columns, category_columns)
        
#         # Handle different chart types
#         if chart_type == 'pie':
#             # Pie chart for single category and value
#             if len(category_columns) >= 1 and len(numeric_columns) >= 1:
#                 category_col = category_columns[0]
#                 value_col = numeric_columns[0]
                
#                 pie_data = []
#                 for row in rows:
#                     pie_data.append({
#                         "name": str(row[category_col]),
#                         "y": float(row[value_col])
#                     })
                
#                 return {
#                     "chart": {"type": "pie"},
#                     "title": {"text": f"{question}"},
#                     "series": [{
#                         "name": value_col.replace('_', ' ').title(),
#                         "data": pie_data
#                     }],
#                     "plotOptions": {
#                         "pie": {
#                             "allowPointSelect": True,
#                             "cursor": "pointer",
#                             "dataLabels": {
#                                 "enabled": True,
#                                 "format": '<b>{point.name}</b>: {point.percentage:.1f} %'
#                             }
#                         }
#                     }
#                 }
        
#         elif chart_type == 'scatter':
#             # Scatter plot for correlation analysis
#             if len(numeric_columns) >= 2:
#                 x_col = numeric_columns[0]
#                 y_col = numeric_columns[1]
                
#                 scatter_data = []
#                 for row in rows:
#                     scatter_data.append([float(row[x_col]), float(row[y_col])])
                
#                 return {
#                     "chart": {"type": "scatter", "zoomType": "xy"},
#                     "title": {"text": f"{question}"},
#                     "xAxis": {
#                         "title": {"text": x_col.replace('_', ' ').title()}
#                     },
#                     "yAxis": {
#                         "title": {"text": y_col.replace('_', ' ').title()}
#                     },
#                     "series": [{
#                         "name": f"{y_col} vs {x_col}",
#                         "data": scatter_data
#                     }]
#                 }
        
#         elif chart_type == 'line':
#             # Line chart for time series or trends
#             if len(category_columns) >= 1 and len(numeric_columns) >= 1:
#                 if len(category_columns) >= 2:
#                     # Multiple series line chart
#                     primary_category = category_columns[0]
#                     secondary_category = category_columns[1]  
#                     value_column = numeric_columns[0]
                    
#                     categories = sorted(list(set(row[primary_category] for row in rows)))
#                     series_names = sorted(list(set(row[secondary_category] for row in rows)))
                    
#                     series_data = []
#                     for series_name in series_names:
#                         data_points = []
#                         for category in categories:
#                             matching_row = next((row for row in rows 
#                                                if row[primary_category] == category 
#                                                and row[secondary_category] == series_name), None)
#                             data_points.append(float(matching_row[value_column]) if matching_row else 0)
                        
#                         series_data.append({
#                             "name": str(series_name),
#                             "data": data_points
#                         })
                    
#                     return {
#                         "chart": {"type": "line"},
#                         "title": {"text": f"{question}"},
#                         "xAxis": {
#                             "categories": [str(cat) for cat in categories],
#                             "title": {"text": primary_category.replace('_', ' ').title()}
#                         },
#                         "yAxis": {
#                             "title": {"text": value_column.replace('_', ' ').title()}
#                         },
#                         "series": series_data
#                     }
#                 else:
#                     # Single series line chart
#                     category_col = category_columns[0]
#                     value_col = numeric_columns[0]
                    
#                     categories = [str(row[category_col]) for row in rows]
#                     values = [float(row[value_col]) for row in rows]
                    
#                     return {
#                         "chart": {"type": "line"},
#                         "title": {"text": f"{question}"},
#                         "xAxis": {
#                             "categories": categories,
#                             "title": {"text": category_col.replace('_', ' ').title()}
#                         },
#                         "yAxis": {
#                             "title": {"text": value_col.replace('_', ' ').title()}
#                         },
#                         "series": [{
#                             "name": value_col.replace('_', ' ').title(),
#                             "data": values
#                         }]
#                     }
        
#         # Default handling for column/bar charts
#         if len(category_columns) >= 2 and len(numeric_columns) >= 1:
#             # Grouped data (like policy_tenure + is_churn + policy_count)
#             primary_category = category_columns[0]  # policy_tenure
#             secondary_category = category_columns[1]  # is_churn
#             value_column = numeric_columns[0]  # policy_count
            
#             # Group data for series
#             categories = sorted(list(set(row[primary_category] for row in rows)))
#             series_names = sorted(list(set(row[secondary_category] for row in rows)))
            
#             series_data = []
#             for series_name in series_names:
#                 data_points = []
#                 for category in categories:
#                     # Find matching row
#                     matching_row = next((row for row in rows 
#                                        if row[primary_category] == category 
#                                        and row[secondary_category] == series_name), None)
#                     data_points.append(float(matching_row[value_column]) if matching_row else 0)
                
#                 series_data.append({
#                     "name": str(series_name),
#                     "data": data_points
#                 })
            
#             return {
#                 "chart": {"type": chart_type},
#                 "title": {"text": f"{question}"},
#                 "xAxis": {
#                     "categories": [str(cat) for cat in categories],
#                     "title": {"text": primary_category.replace('_', ' ').title()}
#                 },
#                 "yAxis": {
#                     "title": {"text": value_column.replace('_', ' ').title()}
#                 },
#                 "series": series_data
#             }
        
#         elif len(category_columns) == 1 and len(numeric_columns) >= 1:
#             # Simple category vs value chart
#             category_col = category_columns[0]
#             value_col = numeric_columns[0]
            
#             categories = [str(row[category_col]) for row in rows]
#             values = [float(row[value_col]) for row in rows]
            
#             return {
#                 "chart": {"type": chart_type},
#                 "title": {"text": f"{question}"},
#                 "xAxis": {
#                     "categories": categories,
#                     "title": {"text": category_col.replace('_', ' ').title()}
#                 },
#                 "yAxis": {
#                     "title": {"text": value_col.replace('_', ' ').title()}
#                 },
#                 "series": [{
#                     "name": value_col.replace('_', ' ').title(),
#                     "data": values
#                 }]
#             }
            
#         return None
        
#     except Exception as e:
#         print("❌ Fallback chart creation error:", e)
#         return None

# def llm_generate_chart_config(question, rows):
#     try:
#         if len(rows) == 1 and len(rows[0]) == 1:
#             # Skip chart generation if it's just one value
#             return None

#         sanitized_rows = sanitize_for_json(rows[:10])
#         table_preview = json.dumps(sanitized_rows, indent=2)

#         chart_prompt = f"""
# You are a BI dashboard expert. Based on the user question and the following data (JSON array of rows),
# return a valid Highcharts JSON configuration. 

# IMPORTANT RULES:
# 1. Return ONLY valid JSON - no explanations, no comments, no markdown
# 2. Choose the BEST chart type based on question intent and data:
#    - "column" for category comparisons and counts
#    - "bar" for rankings, geographic data, or long category names
#    - "line" for trends, time series, or growth analysis
#    - "pie" for proportions/percentages of a whole (max 8 categories)
#    - "scatter" for correlations between two numeric variables
#    - "area" for cumulative data or filled trends
# 3. For grouped data, create multiple series
# 4. Ensure xAxis.categories is an array of strings or numbers
# 5. Ensure series data arrays contain only numbers
# 6. For pie charts, use format: [{{"name": "Category", "y": value}}]
# 7. For scatter plots, use format: [[x, y], [x, y], ...]
# 8. Include proper titles and axis labels
# 9. Add appropriate plotOptions for enhanced visualization

# Question: {question}
# Data:
# {table_preview}

# Return valid Highcharts config JSON:
# """.strip()

#         raw_output = call_llm(chart_prompt)
#         chart_config = json.loads(raw_output)
        
#         # Validate the chart config structure
#         if not isinstance(chart_config, dict) or 'series' not in chart_config:
#             print("⚠️ Invalid chart config from LLM, using fallback")
#             return create_chart_config_from_data(question, rows)
            
#         # Ensure series data is properly formatted based on chart type
#         if 'series' in chart_config:
#             chart_type = chart_config.get('chart', {}).get('type', 'column')
            
#             for series in chart_config['series']:
#                 if 'data' in series:
#                     if chart_type == 'pie':
#                         # Pie chart data should be objects with name and y
#                         if isinstance(series['data'][0], dict):
#                             for point in series['data']:
#                                 if 'y' in point:
#                                     point['y'] = float(point['y']) if point['y'] is not None else 0
#                     elif chart_type == 'scatter':
#                         # Scatter plot data should be [x, y] arrays
#                         series['data'] = [[float(point[0]) if point[0] is not None else 0, 
#                                          float(point[1]) if point[1] is not None else 0] 
#                                         for point in series['data'] if len(point) >= 2]
#                     else:
#                         # Regular charts - convert all data points to numbers
#                         series['data'] = [float(x) if x is not None else 0 for x in series['data']]
        
#         return chart_config

#     except JSONDecodeError as json_err:
#         print("❌ JSON Parsing Error:", json_err)
#         print("⚠️ Raw Output was:", raw_output)
#         # Fallback to programmatic chart creation
#         return create_chart_config_from_data(question, rows)
#     except Exception as e:
#         print("❌ LLM Chart Generation Error:", e)
#         # Fallback to programmatic chart creation
#         return create_chart_config_from_data(question, rows)

# # Update the main function call
# def generate_chart_and_response(question, rows):
#     """Main function to generate chart config with fallbacks"""
#     try:
#         chart_config = llm_generate_chart_config(question, rows)
#         print("📊 Chart config:", chart_config)
#         return chart_config
#     except Exception as chart_err:
#         print("⚠️ Chart generation failed:", chart_err)
#         return None

# .........runningchart


def sanitize_for_json(data):
    """Recursively convert Decimal to float for LLM compatibility."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    elif isinstance(data, Decimal):
        return float(data)
    return data

def clean_llm_json(raw_output):
    """Remove markdown code fences or common formatting issues."""
    raw_output = raw_output.strip()
    if raw_output.startswith("```json"):
        raw_output = raw_output[7:]
    elif raw_output.startswith("```"):
        raw_output = raw_output[3:]
    if raw_output.endswith("```"):
        raw_output = raw_output[:-3]

    raw_output = raw_output.replace('" + "', '')
    return raw_output.strip()

def call_llm(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1000
    }

    response = requests.post(f"{GROQ_BASE_URL}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    return clean_llm_json(content)

def detect_chart_type(question, rows, columns, numeric_columns, category_columns):
    """Intelligently detect the best chart type based on question and data"""
    question_lower = question.lower()
    
    # Time series detection
    time_keywords = ['trend', 'over time', 'monthly', 'yearly', 'daily', 'weekly', 'timeline', 'growth', 'decline']
    date_columns = [col for col in columns if any(keyword in col.lower() for keyword in ['date', 'time', 'month', 'year', 'day'])]
    
    # Distribution/correlation detection
    scatter_keywords = ['correlation', 'relationship', 'scatter', 'distribution', 'vs', 'against']
    histogram_keywords = ['histogram', 'distribution', 'frequency', 'bins', 'range']
    
    # Proportion detection
    pie_keywords = ['percentage', 'proportion', 'share', 'distribution', 'breakdown', 'composition']
    donut_keywords = ['donut', 'doughnut'] + pie_keywords
    
    # Comparison detection
    bar_keywords = ['compare', 'comparison', 'versus', 'top', 'bottom', 'ranking', 'highest', 'lowest', 'horizontal']
    
    # Geographic/Heatmap detection
    geo_keywords = ['by state', 'by region', 'by country', 'by city', 'geographic', 'location']
    heatmap_keywords = ['heatmap', 'heat map', 'intensity', 'correlation matrix', 'density']
    
    # Process/Flow detection
    funnel_keywords = ['funnel', 'conversion', 'process', 'stages', 'pipeline']
    waterfall_keywords = ['waterfall', 'cumulative', 'breakdown', 'contribution', 'bridge']
    gantt_keywords = ['gantt', 'timeline', 'project', 'schedule', 'tasks']
    
    # Hierarchical detection
    treemap_keywords = ['treemap', 'tree map', 'hierarchical', 'nested', 'proportional']
    
    # Multi-dimensional detection
    bubble_keywords = ['bubble', 'three variables', '3d', 'size represents']
    radar_keywords = ['radar', 'spider', 'multi-axis', 'performance']
    
    # Check for specific chart types
    if any(keyword in question_lower for keyword in donut_keywords):
        return 'donut'
    elif any(keyword in question_lower for keyword in histogram_keywords):
        return 'histogram'
    elif any(keyword in question_lower for keyword in heatmap_keywords):
        return 'heatmap'
    elif any(keyword in question_lower for keyword in funnel_keywords):
        return 'funnel'
    elif any(keyword in question_lower for keyword in waterfall_keywords):
        return 'waterfall'
    elif any(keyword in question_lower for keyword in gantt_keywords):
        return 'gantt'
    elif any(keyword in question_lower for keyword in treemap_keywords):
        return 'treemap'
    elif any(keyword in question_lower for keyword in bubble_keywords) and len(numeric_columns) >= 3:
        return 'bubble'
    elif any(keyword in question_lower for keyword in radar_keywords):
        return 'radar'
    elif any(keyword in question_lower for keyword in time_keywords) or date_columns:
        return 'line'
    elif (any(keyword in question_lower for keyword in scatter_keywords) and 
        len(numeric_columns) >= 2):
        return 'scatter'
    elif (any(keyword in question_lower for keyword in pie_keywords) and 
        len(category_columns) == 1 and len(numeric_columns) == 1 and len(rows) <= 10):
        return 'pie'
    elif (any(keyword in question_lower for keyword in bar_keywords) or 
        any(keyword in question_lower for keyword in geo_keywords)):
        return 'bar'
    
    # Default to column chart for comparisons
    return 'column'

def create_histogram_data(values, bins=10):
    """Create histogram data from numeric values"""
    import numpy as np
    
    hist, bin_edges = np.histogram(values, bins=bins)
    categories = []
    data = []
    
    for i in range(len(hist)):
        categories.append(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}")
        data.append(int(hist[i]))
    
    return categories, data

def create_chart_config_from_data(question, rows):
    """Create chart config directly from data without LLM if LLM fails"""
    try:
        if not rows or len(rows) == 0:
            return None
            
        # Analyze the data structure
        first_row = rows[0]
        columns = list(first_row.keys())
        
        # Find numeric columns
        numeric_columns = []
        category_columns = []
        
        for col in columns:
            sample_values = [row.get(col) for row in rows[:5]]  # Check first 5 rows
            if all(isinstance(val, (int, float)) and not isinstance(val, bool) for val in sample_values if val is not None):
                numeric_columns.append(col)
            else:
                category_columns.append(col)
        
        if not numeric_columns:
            return None
        
        # Detect the best chart type
        chart_type = detect_chart_type(question, rows, columns, numeric_columns, category_columns)
        
        # Handle different chart types
        if chart_type == 'histogram':
            # Histogram for distribution analysis
            if len(numeric_columns) >= 1:
                value_col = numeric_columns[0]
                values = [float(row[value_col]) for row in rows if row[value_col] is not None]
                
                categories, data = create_histogram_data(values)
                
                return {
                    "chart": {"type": "column"},
                    "title": {"text": f"Distribution of {value_col.replace('_', ' ').title()}"},
                    "xAxis": {
                        "categories": categories,
                        "title": {"text": f"{value_col.replace('_', ' ').title()} Ranges"}
                    },
                    "yAxis": {
                        "title": {"text": "Frequency"}
                    },
                    "series": [{
                        "name": "Frequency",
                        "data": data,
                        "color": "#ff6b6b"
                    }],
                    "plotOptions": {
                        "column": {
                            "pointPadding": 0,
                            "borderWidth": 0,
                            "groupPadding": 0,
                            "shadow": False
                        }
                    }
                }
        
        elif chart_type == 'bubble':
            # Bubble chart for 3-variable analysis
            if len(numeric_columns) >= 3:
                x_col = numeric_columns[0]
                y_col = numeric_columns[1]
                size_col = numeric_columns[2]
                
                bubble_data = []
                for row in rows:
                    bubble_data.append({
                        "x": float(row[x_col]),
                        "y": float(row[y_col]),
                        "z": float(row[size_col]),
                        "name": str(row.get(category_columns[0], f"Point {len(bubble_data)+1}")) if category_columns else f"Point {len(bubble_data)+1}"
                    })
                
                return {
                    "chart": {"type": "bubble", "plotBorderWidth": 1, "zoomType": "xy"},
                    "title": {"text": f"{question}"},
                    "xAxis": {
                        "title": {"text": x_col.replace('_', ' ').title()}
                    },
                    "yAxis": {
                        "title": {"text": y_col.replace('_', ' ').title()}
                    },
                    "series": [{
                        "name": size_col.replace('_', ' ').title(),
                        "data": bubble_data
                    }]
                }
        
        elif chart_type == 'heatmap':
            # Heatmap for correlation or intensity data
            if len(numeric_columns) >= 1 and len(category_columns) >= 2:
                x_col = category_columns[0]
                y_col = category_columns[1]
                value_col = numeric_columns[0]
                
                # Create matrix data
                x_categories = sorted(list(set(row[x_col] for row in rows)))
                y_categories = sorted(list(set(row[y_col] for row in rows)))
                
                heatmap_data = []
                for i, y_cat in enumerate(y_categories):
                    for j, x_cat in enumerate(x_categories):
                        matching_row = next((row for row in rows 
                                           if row[x_col] == x_cat and row[y_col] == y_cat), None)
                        if matching_row:
                            heatmap_data.append([j, i, float(matching_row[value_col])])
                
                return {
                    "chart": {"type": "heatmap", "marginTop": 40, "marginBottom": 80},
                    "title": {"text": f"{question}"},
                    "xAxis": {
                        "categories": [str(cat) for cat in x_categories],
                        "title": {"text": x_col.replace('_', ' ').title()}
                    },
                    "yAxis": {
                        "categories": [str(cat) for cat in y_categories],
                        "title": {"text": y_col.replace('_', ' ').title()}
                    },
                    "colorAxis": {
                        "min": 0,
                        "minColor": '#FFFFFF',
                        "maxColor": '#ff6b6b'
                    },
                    "series": [{
                        "name": value_col.replace('_', ' ').title(),
                        "data": heatmap_data,
                        "dataLabels": {
                            "enabled": True,
                            "color": '#000000'
                        }
                    }]
                }
        
        elif chart_type == 'treemap':
            # Treemap for hierarchical data
            if len(category_columns) >= 1 and len(numeric_columns) >= 1:
                category_col = category_columns[0]
                value_col = numeric_columns[0]
                
                treemap_data = []
                for row in rows:
                    treemap_data.append({
                        "name": str(row[category_col]),
                        "value": float(row[value_col])
                    })
                
                return {
                    "chart": {"type": "treemap"},
                    "title": {"text": f"{question}"},
                    "series": [{
                        "name": value_col.replace('_', ' ').title(),
                        "data": treemap_data,
                        "layoutAlgorithm": 'squarified'
                    }]
                }
        
        elif chart_type == 'funnel':
            # Funnel chart for conversion processes
            if len(category_columns) >= 1 and len(numeric_columns) >= 1:
                category_col = category_columns[0]
                value_col = numeric_columns[0]
                
                funnel_data = []
                for row in rows:
                    funnel_data.append([str(row[category_col]), float(row[value_col])])
                
                return {
                    "chart": {"type": "funnel", "marginRight": 100},
                    "title": {"text": f"{question}"},
                    "plotOptions": {
                        "series": {
                            "dataLabels": {
                                "enabled": True,
                                "format": '<b>{point.name}</b> ({point.y:,.0f})',
                                "softConnector": True
                            },
                            "neckWidth": "30%",
                            "neckHeight": "25%"
                        }
                    },
                    "series": [{
                        "name": value_col.replace('_', ' ').title(),
                        "data": funnel_data
                    }]
                }
        
        elif chart_type == 'waterfall':
            # Waterfall chart for cumulative effects
            if len(category_columns) >= 1 and len(numeric_columns) >= 1:
                category_col = category_columns[0]
                value_col = numeric_columns[0]
                
                waterfall_data = []
                for i, row in enumerate(rows):
                    waterfall_data.append({
                        "name": str(row[category_col]),
                        "y": float(row[value_col]),
                        "isSum": i == len(rows) - 1  # Last item as sum
                    })
                
                return {
                    "chart": {"type": "waterfall"},
                    "title": {"text": f"{question}"},
                    "xAxis": {
                        "type": "category"
                    },
                    "yAxis": {
                        "title": {"text": value_col.replace('_', ' ').title()}
                    },
                    "series": [{
                        "name": value_col.replace('_', ' ').title(),
                        "data": waterfall_data,
                        "dataLabels": {
                            "enabled": True,
                            "style": {
                                "fontWeight": "bold"
                            }
                        },
                        "pointPadding": 0
                    }]
                }
        
        elif chart_type == 'radar':
            # Radar chart for multi-variable comparison
            if len(numeric_columns) >= 3:
                categories = [col.replace('_', ' ').title() for col in numeric_columns]
                
                if len(category_columns) >= 1:
                    # Multiple series radar
                    series_col = category_columns[0]
                    series_names = list(set(row[series_col] for row in rows))
                    
                    series_data = []
                    for series_name in series_names:
                        matching_rows = [row for row in rows if row[series_col] == series_name]
                        if matching_rows:
                            data_points = [float(matching_rows[0][col]) for col in numeric_columns]
                            series_data.append({
                                "name": str(series_name),
                                "data": data_points,
                                "pointPlacement": 'on'
                            })
                else:
                    # Single series radar
                    data_points = [float(rows[0][col]) for col in numeric_columns]
                    series_data = [{
                        "name": "Values",
                        "data": data_points,
                        "pointPlacement": 'on'
                    }]
                
                return {
                    "chart": {"polar": True, "type": "line"},
                    "title": {"text": f"{question}"},
                    "xAxis": {
                        "categories": categories,
                        "tickmarkPlacement": 'on',
                        "lineWidth": 0
                    },
                    "yAxis": {
                        "gridLineInterpolation": 'polygon',
                        "lineWidth": 0,
                        "min": 0
                    },
                    "series": series_data
                }
        
        elif chart_type == 'donut':
            # Donut chart (pie with inner radius)
            if len(category_columns) >= 1 and len(numeric_columns) >= 1:
                category_col = category_columns[0]
                value_col = numeric_columns[0]
                
                donut_data = []
                for row in rows:
                    donut_data.append({
                        "name": str(row[category_col]),
                        "y": float(row[value_col])
                    })
                
                return {
                    "chart": {"type": "pie"},
                    "title": {"text": f"{question}"},
                    "series": [{
                        "name": value_col.replace('_', ' ').title(),
                        "data": donut_data,
                        "innerSize": "50%",
                        "dataLabels": {
                            "enabled": True,
                            "format": '<b>{point.name}</b>: {point.percentage:.1f} %'
                        }
                    }]
                }
        
        elif chart_type == 'pie':
            # Pie chart for single category and value
            if len(category_columns) >= 1 and len(numeric_columns) >= 1:
                category_col = category_columns[0]
                value_col = numeric_columns[0]
                
                pie_data = []
                for row in rows:
                    pie_data.append({
                        "name": str(row[category_col]),
                        "y": float(row[value_col])
                    })
                
                return {
                    "chart": {"type": "pie"},
                    "title": {"text": f"{question}"},
                    "series": [{
                        "name": value_col.replace('_', ' ').title(),
                        "data": pie_data
                    }],
                    "plotOptions": {
                        "pie": {
                            "allowPointSelect": True,
                            "cursor": "pointer",
                            "dataLabels": {
                                "enabled": True,
                                "format": '<b>{point.name}</b>: {point.percentage:.1f} %'
                            }
                        }
                    }
                }
        
        elif chart_type == 'scatter':
            # Scatter plot for correlation analysis
            if len(numeric_columns) >= 2:
                x_col = numeric_columns[0]
                y_col = numeric_columns[1]
                
                scatter_data = []
                for row in rows:
                    scatter_data.append([float(row[x_col]), float(row[y_col])])
                
                return {
                    "chart": {"type": "scatter", "zoomType": "xy"},
                    "title": {"text": f"{question}"},
                    "xAxis": {
                        "title": {"text": x_col.replace('_', ' ').title()}
                    },
                    "yAxis": {
                        "title": {"text": y_col.replace('_', ' ').title()}
                    },
                    "series": [{
                        "name": f"{y_col} vs {x_col}",
                        "data": scatter_data
                    }]
                }
        
        elif chart_type == 'line':
            # Line chart for time series or trends
            if len(category_columns) >= 1 and len(numeric_columns) >= 1:
                if len(category_columns) >= 2:
                    # Multiple series line chart
                    primary_category = category_columns[0]
                    secondary_category = category_columns[1]  
                    value_column = numeric_columns[0]
                    
                    categories = sorted(list(set(row[primary_category] for row in rows)))
                    series_names = sorted(list(set(row[secondary_category] for row in rows)))
                    
                    series_data = []
                    for series_name in series_names:
                        data_points = []
                        for category in categories:
                            matching_row = next((row for row in rows 
                                               if row[primary_category] == category 
                                               and row[secondary_category] == series_name), None)
                            data_points.append(float(matching_row[value_column]) if matching_row else 0)
                        
                        series_data.append({
                            "name": str(series_name),
                            "data": data_points
                        })
                    
                    return {
                        "chart": {"type": "line"},
                        "title": {"text": f"{question}"},
                        "xAxis": {
                            "categories": [str(cat) for cat in categories],
                            "title": {"text": primary_category.replace('_', ' ').title()}
                        },
                        "yAxis": {
                            "title": {"text": value_column.replace('_', ' ').title()}
                        },
                        "series": series_data
                    }
                else:
                    # Single series line chart
                    category_col = category_columns[0]
                    value_col = numeric_columns[0]
                    
                    categories = [str(row[category_col]) for row in rows]
                    values = [float(row[value_col]) for row in rows]
                    
                    return {
                        "chart": {"type": "line"},
                        "title": {"text": f"{question}"},
                        "xAxis": {
                            "categories": categories,
                            "title": {"text": category_col.replace('_', ' ').title()}
                        },
                        "yAxis": {
                            "title": {"text": value_col.replace('_', ' ').title()}
                        },
                        "series": [{
                            "name": value_col.replace('_', ' ').title(),
                            "data": values
                        }]
                    }
        
        # Default handling for column/bar charts
        if len(category_columns) >= 2 and len(numeric_columns) >= 1:
            # Grouped data (like policy_tenure + is_churn + policy_count)
            primary_category = category_columns[0]  # policy_tenure
            secondary_category = category_columns[1]  # is_churn
            value_column = numeric_columns[0]  # policy_count
            
            # Group data for series
            categories = sorted(list(set(row[primary_category] for row in rows)))
            series_names = sorted(list(set(row[secondary_category] for row in rows)))
            
            series_data = []
            for series_name in series_names:
                data_points = []
                for category in categories:
                    # Find matching row
                    matching_row = next((row for row in rows 
                                       if row[primary_category] == category 
                                       and row[secondary_category] == series_name), None)
                    data_points.append(float(matching_row[value_column]) if matching_row else 0)
                
                series_data.append({
                    "name": str(series_name),
                    "data": data_points
                })
            
            return {
                "chart": {"type": chart_type},
                "title": {"text": f"{question}"},
                "xAxis": {
                    "categories": [str(cat) for cat in categories],
                    "title": {"text": primary_category.replace('_', ' ').title()}
                },
                "yAxis": {
                    "title": {"text": value_column.replace('_', ' ').title()}
                },
                "series": series_data
            }
        
        elif len(category_columns) == 1 and len(numeric_columns) >= 1:
            # Simple category vs value chart
            category_col = category_columns[0]
            value_col = numeric_columns[0]
            
            categories = [str(row[category_col]) for row in rows]
            values = [float(row[value_col]) for row in rows]
            
            return {
                "chart": {"type": chart_type},
                "title": {"text": f"{question}"},
                "xAxis": {
                    "categories": categories,
                    "title": {"text": category_col.replace('_', ' ').title()}
                },
                "yAxis": {
                    "title": {"text": value_col.replace('_', ' ').title()}
                },
                "series": [{
                    "name": value_col.replace('_', ' ').title(),
                    "data": values
                }]
            }
            
        return None
        
    except Exception as e:
        print("❌ Fallback chart creation error:", e)
        return None
    
    
import json
from json import JSONDecodeError

def llm_generate_chart_config(question, rows):
    try:
        if len(rows) == 1 and len(rows[0]) == 1:
            # Skip chart generation if it's just one value
            return None

        sanitized_rows = sanitize_for_json(rows[:10])
        table_preview = json.dumps(sanitized_rows, indent=2)

        chart_prompt = f"""
You are a BI dashboard expert. Based on the user question and the following data (JSON array of rows),
return a valid Highcharts JSON configuration. 

IMPORTANT RULES:
1. Return ONLY valid JSON - no explanations, no comments, no markdown
2. Choose the BEST chart type based on question intent and data:
   - "column" for category comparisons and counts
   - "bar" for rankings, geographic data, or long category names
   - "line" for trends, time series, or growth analysis
   - "pie" for proportions/percentages of a whole (max 8 categories)
   - "scatter" for correlations between two numeric variables
   - "area" for cumulative data or filled trends
   - "histogram" for distribution analysis (use column type with special binning)
   - "bubble" for 3-variable analysis (x, y, size)
   - "treemap" for hierarchical data and proportions
   - "heatmap" for correlation matrices or intensity data
   - "funnel" for conversion processes or stages
   - "gantt" for project timelines and dependencies
   - "waterfall" for cumulative effects or breakdowns
   - "radar" for multi-variable comparisons
   - "donut" for pie charts with inner radius (set innerSize: "50%")
3. For grouped data, create multiple series
4. Ensure xAxis.categories is an array of strings or numbers
5. Ensure series data arrays contain only numbers (except for special types)
6. For pie/donut charts, use format: [{{"name": "Category", "y": value}}]
7. For scatter plots, use format: [[x, y], [x, y], ...]
8. For bubble charts, use format: [{{"x": x, "y": y, "z": size, "name": "label"}}]
9. For heatmap, use format: [[x_index, y_index, value], [x_index, y_index, value], ...]
10. For treemap, use format: [{{"name": "Category", "value": size}}]
11. For funnel/waterfall, use format: [[name, value], [name, value], ...]
12. Include proper titles and axis labels
13. Add appropriate plotOptions for enhanced visualization
14. Never use '+' for string concatenation in JSON. Always return complete strings.
15. All strings (like tooltip formats) must be returned as one continuous JSON-safe string.

Question: {question}
Data:
{table_preview}

Return valid Highcharts config JSON:
""".strip()

        raw_output = call_llm(chart_prompt)
        chart_config = json.loads(raw_output)

        # Validate the chart config structure
        if not isinstance(chart_config, dict) or 'series' not in chart_config:
            print("⚠️ Invalid chart config from LLM, using fallback")
            return create_chart_config_from_data(question, rows)

        return chart_config

    except JSONDecodeError as json_err:
        print("❌ JSON Parsing Error:", json_err)
        print("⚠️ Raw Output was:", raw_output)
        return create_chart_config_from_data(question, rows)
    except Exception as e:
        print("❌ LLM Chart Generation Error:", e)
        return create_chart_config_from_data(question, rows)


def generate_chart_and_response(question, rows):
    """Main function to generate chart config with fallbacks"""
    try:
        chart_config = llm_generate_chart_config(question, rows)
        print("📊 Chart config:", chart_config)
    except Exception as chart_err:
        print("⚠️ Chart generation failed:", chart_err)
        return None, rows

    # Ensure series data is properly formatted based on chart type
    if chart_config and 'series' in chart_config:
        chart_type = chart_config.get('chart', {}).get('type', 'column')

        for series in chart_config['series']:
            if 'data' in series:
                if chart_type in ['pie', 'donut']:
                    if isinstance(series['data'][0], dict):
                        for point in series['data']:
                            if 'y' in point:
                                point['y'] = float(point['y']) if point['y'] is not None else 0

                elif chart_type == 'scatter':
                    series['data'] = [
                        [float(point[0]) if point[0] is not None else 0,
                         float(point[1]) if point[1] is not None else 0]
                        for point in series['data'] if len(point) >= 2
                    ]

                elif chart_type == 'bubble':
                    if isinstance(series['data'][0], dict):
                        for point in series['data']:
                            point['x'] = float(point['x']) if point['x'] is not None else 0
                            point['y'] = float(point['y']) if point['y'] is not None else 0
                            point['z'] = float(point['z']) if point['z'] is not None else 0

                elif chart_type == 'heatmap':
                    if isinstance(series['data'][0], list) and len(series['data'][0]) >= 3:
                        series['data'] = [
                            [int(point[0]), int(point[1]), float(point[2])]
                            for point in series['data']
                        ]

                elif chart_type in ['treemap', 'funnel', 'waterfall']:
                    pass  # These formats are assumed to be handled correctly

                else:
                    series['data'] = [
                        float(x) if x is not None else 0 for x in series['data']
                    ]

    return chart_config











# pip install azure-ai-inference azure-core

# import os
# import time
# from typing import Optional, List
# from azure.ai.inference import ChatCompletionsClient
# from azure.ai.inference.models import SystemMessage, UserMessage
# from azure.core.credentials import AzureKeyCredential
# from azure.core.exceptions import HttpResponseError, ServiceRequestError, ServiceResponseError

# AZURE_ENDPOINT = os.getenv("AZURE_INFERENCE_ENDPOINT")  # e.g. https://genaiprochurn.services.ai.azure.com/models
# AZURE_API_KEY = os.getenv("AZURE_INFERENCE_API_KEY")
# AZURE_MODEL = os.getenv("AZURE_INFERENCE_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8-prochurn-demo")
# AZURE_API_VERSION = "2024-05-01-preview"

# _client: Optional[ChatCompletionsClient] = None

# def _client_instance() -> ChatCompletionsClient:
#     global _client
#     if _client is None:
#         if not AZURE_ENDPOINT or not AZURE_API_KEY:
#             raise RuntimeError("Set AZURE_INFERENCE_ENDPOINT and AZURE_INFERENCE_API_KEY.")
#         _client = ChatCompletionsClient(
#             endpoint=AZURE_ENDPOINT,
#             credential=AzureKeyCredential(AZURE_API_KEY),
#             api_version=AZURE_API_VERSION,
#         )
#     return _client

# def call_llm(
#     prompt: str,
#     *,
#     system_prompt: str = "You are a helpful SQL assistant.",
#     temperature: float = 0.1,
#     max_tokens: int = 1000,
#     top_p: float = 0.9,
#     stop: Optional[List[str]] = None,
# ):
#     """
#     Calls Azure AI Inference Chat Completions and returns cleaned JSON text via clean_llm_json().
#     Mirrors your previous signature and defaults.
#     """
#     msgs = [SystemMessage(content=system_prompt), UserMessage(content=prompt)]

#     max_retries = 3
#     for attempt in range(max_retries):
#         try:
#             resp = _client_instance().complete(
#                 messages=msgs,
#                 model=AZURE_MODEL,
#                 max_tokens=max_tokens,
#                 temperature=temperature,
#                 top_p=top_p,
#                 stop=stop,
#             )
#             content = resp.choices[0].message.content.strip()
#             return clean_llm_json(content)

#         except HttpResponseError as e:
#             status = getattr(e, "status_code", None)
#             if status == 429 and attempt < max_retries - 1:
#                 time.sleep(2 ** attempt)
#                 continue
#             raise
#         except (ServiceRequestError, ServiceResponseError, TimeoutError):
#             if attempt < max_retries - 1:
#                 time.sleep(2 ** attempt)
#                 continue
#             raise
