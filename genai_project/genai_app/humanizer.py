# Fully Dynamic LLM-Generated Human Speech
from .utils.llm_config import get_llama_maverick_llm
import json, re
from typing import Dict, Any, List

NARRATIVE_SCHEMA = {
    "opener": "",
    "insights": [],
    "recommendations": [],
    "next_step": ""
}

def _to_preview_rows(rows: List[Dict[str, Any]], limit: int = 25) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return rows[:limit]

def _derive_contextual_metrics(question: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract rich contextual data for the LLM to use in generating dynamic responses"""
    preview = _to_preview_rows(rows, 100)
    
    metrics = {
        "row_count": len(rows),
        "columns": list(preview[0].keys()) if preview else [],
        "has_data": len(rows) > 0,
        "data_size": "small" if len(rows) < 10 else "medium" if len(rows) < 100 else "large",
    }
    
    # Extract actual data patterns for LLM analysis
    if preview:
        # Identify column types dynamically
        column_analysis = {}
        for col in metrics["columns"]:
            sample_values = [r.get(col) for r in preview[:10] if r.get(col) is not None]
            if sample_values:
                # Check if numeric
                numeric_values = []
                for val in sample_values:
                    try:
                        numeric_values.append(float(val))
                    except:
                        break
                
                if len(numeric_values) == len(sample_values):
                    column_analysis[col] = {
                        "type": "numeric",
                        "sample_values": numeric_values[:5],
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                        "sum": sum(numeric_values),
                        "avg": sum(numeric_values) / len(numeric_values)
                    }
                else:
                    # Categorical/text
                    unique_values = list(set(str(v) for v in sample_values))
                    column_analysis[col] = {
                        "type": "categorical",
                        "sample_values": sample_values[:5],
                        "unique_count": len(unique_values),
                        "unique_values": unique_values[:10]
                    }
        
        metrics["column_analysis"] = column_analysis
        
        # Extract top patterns for LLM insight generation
        metrics["data_patterns"] = []
        
        # Find the most interesting numeric patterns
        numeric_cols = [col for col, info in column_analysis.items() if info.get("type") == "numeric"]
        if numeric_cols:
            # Get top values by first numeric column
            main_numeric = numeric_cols[0]
            categorical_cols = [col for col, info in column_analysis.items() if info.get("type") == "categorical"]
            
            if categorical_cols:
                # Create aggregations for LLM to analyze
                aggregations = {}
                for row in preview:
                    cat_val = str(row.get(categorical_cols[0], "Unknown"))
                    num_val = row.get(main_numeric, 0)
                    try:
                        num_val = float(num_val)
                        if cat_val not in aggregations:
                            aggregations[cat_val] = []
                        aggregations[cat_val].append(num_val)
                    except:
                        continue
                
                # Summarize aggregations
                agg_summary = {}
                for cat, values in aggregations.items():
                    if values:
                        agg_summary[cat] = {
                            "count": len(values),
                            "total": sum(values),
                            "avg": sum(values) / len(values),
                            "max": max(values)
                        }
                
                metrics["aggregation_summary"] = dict(sorted(agg_summary.items(), 
                                                           key=lambda x: x[1].get("total", 0), 
                                                           reverse=True)[:10])
    
    return metrics

def _clean_and_extract_json(raw_text: str) -> dict:
    """Robustly extract JSON from LLM response"""
    # Clean the text
    text = (raw_text or "").strip()
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)  # Remove emojis
    
    # Try to find JSON block
    json_patterns = [
        r'\{.*\}',  # Basic JSON
        r'```json\s*(\{.*\})\s*```',  # Markdown code block
        r'```\s*(\{.*\})\s*```',  # Generic code block
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_text = match.group(1) if match.groups() else match.group(0)
            try:
                # Clean common JSON issues
                json_text = re.sub(r',\s*}', '}', json_text)
                json_text = re.sub(r',\s*]', ']', json_text)
                return json.loads(json_text)
            except:
                continue
    
    # If no valid JSON found, return empty structure
    return {}
def safe_generate_narrative(question, sql, rows):
    """
    Safe wrapper: Try fully dynamic humanized narrative first,
    then intelligent LLM fallback, and finally static narration.
    Always avoids negative or apologetic phrasing.
    """
    try:
        # --- 1. Try fully dynamic narrative ---
        return humanize_narrative(
            question=question,
            rows=rows,
            summary="",
            recommendation="",
            sql=sql,
        )

    except Exception as e:
        logger.warning(f"Dynamic narrative generation failed: {e}")
        row_count = len(rows)

        try:
            # --- 2. Intelligent fallback (still uses LLM) ---
            return _generate_intelligent_fallback(
                question, rows, _derive_contextual_metrics(question, rows)
            )

        except Exception as e2:
            logger.error(f"Intelligent fallback also failed: {e2}")

            # --- 3. Static narration (neutral & constructive) ---
            if row_count == 0:
                return {
                    "opener": "This query did not return any matching records under the current filters.",
                    "insights": ["No results are available with the applied filters."],
                    "recommendations": [
                        "Try broadening the filters or adjusting the time period.",
                        "Consider checking another branch or product category."
                    ],
                    "next_step": "Would you like me to expand the scope and re-run the analysis?"
                }
            else:
                return {
                    "opener": f"I analyzed {row_count} records from your query and here’s what stands out.",
                    "insights": [
                        f"The dataset includes {len(rows[0]) if rows else 0} fields.",
                        "The results align with the filters you applied."
                    ],
                    "recommendations": [
                        "You could segment the results further by branch, product, or customer type.",
                        "It may also be useful to compare this period with a previous one."
                    ],
                    "next_step": "Would you like me to highlight key drivers or trends next?"
                }




# def generate_dynamic_conversational_opener(question: str, metrics: Dict[str, Any]) -> str:
#     """Use LLM to generate a completely dynamic conversational opener"""
#     llm = get_llama_maverick_llm()
    
#     system_prompt = (
#         "You are a friendly, experienced data analyst. Generate a natural, conversational opener "
#         "that sounds like how you would actually start explaining results to a colleague. "
#         "Be warm, professional, and engaging. NO emojis. 2-3 sentences max. "
#         "Make it feel like natural human speech, not a template."
#     )
    
#     context = {
#         "user_question": question,
#         "data_overview": {
#             "found_records": metrics.get("row_count", 0),
#             "data_size": metrics.get("data_size", "unknown"),
#             "has_results": metrics.get("has_data", False)
#         }
#     }
    
#     user_prompt = (
#         f"Generate a conversational opener for this data analysis situation:\n"
#         f"{json.dumps(context, indent=2)}\n\n"
#         "Respond with ONLY the opener text - no JSON, no formatting, just natural speech."
#     )
    
#     try:
#         opener = llm._call(
#             prompt=user_prompt,
#             system_prompt=system_prompt,
#             temperature=0.7,  # Higher for more variety
#             max_tokens=150,
#         )
        
#         # Clean and return
#         opener = (opener or "").strip().strip('"').strip("'")
#         if len(opener) < 10:  # Fallback if too short
#             if metrics.get("has_data"):
#                 return f"Great question! I've analyzed your data and found some interesting patterns."
#             else:
#                 return "I've looked into your question, and here's what I can tell you."
        
#         return opener
        
#     except Exception as e:
#         print(f"⚠️ Dynamic opener generation failed: {e}")
#         # Minimal fallback
#         return "Let me walk you through what I found in the data."
# Update your generate_dynamic_conversational_opener function
def generate_dynamic_conversational_opener(question: str, metrics: Dict[str, Any]) -> str:
    """Use LLM to generate a completely dynamic conversational opener - ALWAYS POSITIVE during loading"""
    llm = get_llama_maverick_llm()
    
    # Extract key entities/topics from the question for context
    question_lower = question.lower().strip()
    
    # Determine the topic/domain for better context
    if any(word in question_lower for word in ['churn', 'retention', 'customer', 'policy']):
        domain = "customer analytics"
    elif any(word in question_lower for word in ['branch', 'location', 'office']):
        domain = "branch performance"
    elif any(word in question_lower for word in ['premium', 'revenue', 'sales']):
        domain = "revenue analysis"
    elif any(word in question_lower for word in ['campaign', 'marketing']):
        domain = "marketing insights"
    else:
        domain = "business intelligence"
    
    system_prompt = (
        "You are a friendly, experienced data analyst generating an opener BEFORE data analysis is complete. "
        "This opener shows during loading, so NEVER mention results, findings, or data availability. "
        "Focus on what you're ABOUT TO DO, not what you found. "
        "Be warm, professional, and encouraging. NO emojis. 2-3 sentences max. "
        "CRITICAL: Never say 'no data', 'no records', 'nothing found', or anything negative about results."
    )
    
    # For loading phase, don't pass actual results - focus on intent
    context = {
        "user_question": question,
        "analysis_domain": domain,
        "loading_phase": True  # Key indicator this is pre-results
    }
    
    user_prompt = (
        f"Generate a conversational opener for starting this data analysis:\n"
        f"{json.dumps(context, indent=2)}\n\n"
        "This opener shows WHILE loading, so focus on:\n"
        "- What you're about to analyze\n"
        "- Your enthusiasm for helping\n"
        "- The process you'll follow\n\n"
        "DO NOT mention any results or data availability since analysis hasn't finished yet.\n"
        "Respond with ONLY the opener text - no JSON, no formatting."
    )
    
    try:
        opener = llm._call(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=150,
        )
        
        # Clean and validate
        opener = (opener or "").strip().strip('"').strip("'")
        
        # Filter out any negative phrases that might slip through
        negative_patterns = [
            "no data", "no records", "nothing found", "don't have", "aren't any",
            "not available", "no results", "no information", "bit of a surprise",
            "unfortunately", "however", "but when I", "found that there"
        ]
        
        opener_lower = opener.lower()
        if any(pattern in opener_lower for pattern in negative_patterns) or len(opener) < 10:
            # Use positive fallback
            positive_fallbacks = [
                f"Let me analyze your {domain} question and gather the relevant insights for you.",
                f"I'm diving into your {domain} data to provide you with comprehensive analysis.",
                f"Great question about {domain}! I'm processing your request to deliver detailed insights.",
                f"I'm excited to help with your {domain} analysis and will have results for you shortly.",
                f"Let me examine your {domain} data thoroughly to give you the best possible insights."
            ]
            import random
            opener = random.choice(positive_fallbacks)
        
        return opener
        
    except Exception as e:
        print(f"⚠️ Dynamic opener generation failed: {e}")
        # Safe fallback that's always positive
        return f"I'm analyzing your {domain} question and will have comprehensive insights for you shortly."
# def humanize_narrative(
#     question: str,
#     rows: List[Dict[str, Any]],
#     summary: str,
#     recommendation: str,
#     sql: str,
#     data_window: Dict[str, str] | None = None,
#     model_temperature: float = 0.6,
#     opener: str = None
# ) -> Dict[str, Any]:
#     """
#     Fully dynamic narrative generation - LLM creates all content naturally
#     """
#     llm = get_llama_maverick_llm()
    
#     # Extract rich contextual data
#     metrics = _derive_contextual_metrics(question, rows)
#     preview = _to_preview_rows(rows, 20)
    
#     # Enhanced system prompt for natural human speech
#     system = (
#     "You are a skilled data analyst having a natural conversation with a colleague. "
#     "Explain results warmly and professionally.\n\n"

#     "Important rules:\n"
#     "- NEVER say: 'no records', 'no data', 'nothing to pull', 'bit of a surprise', or anything apologetic/negative.\n"
#     "- If results are empty, simply say: 'This query did not return matching records under the current filters.'\n"
#     "- If results exist, focus only on what was found — start with what you did analyze.\n"
#     "- Always keep the tone constructive and forward-looking.\n"
#     "- NO emojis.\n\n"

#     "Response format (JSON only):\n"
#     "{\n"
#     '  "opener": "Natural 2-3 sentence conversation starter",\n'
#     '  "insights": ["Specific finding 1", "Specific finding 2", ...],\n'
#     '  "recommendations": ["Action item 1", "Action item 2", ...],\n'
#     '  "next_step": "Natural offer for follow-up"\n'
#     "}"
# )


    
#     # Rich context for LLM
#     analysis_context = {
#         "user_question": question,
#         "data_summary": {
#             "total_rows": len(rows),
#             "columns_available": metrics.get("columns", []),
#             "data_patterns": metrics.get("aggregation_summary", {}),
#             "column_insights": metrics.get("column_analysis", {}),
#             "time_period": data_window or {}
#         },
#         "sample_data": preview,
#         "existing_summary": summary,
#         "existing_recommendation": recommendation,
#         "sql_context": "Data was queried from business intelligence database"
#     }
    
#     user_prompt = (
#         "Please analyze this data and create a natural, conversational explanation. "
#         "Write like you're a human analyst talking to a colleague - be specific, insightful, and engaging.\n\n"
#         f"ANALYSIS CONTEXT:\n{json.dumps(analysis_context, ensure_ascii=False, indent=2)}\n\n"
#         "Generate a natural narrative that sounds authentically human. Use the actual data patterns and numbers."
#     )
    
#     try:
#         # Generate with higher temperature for natural variety
#         raw_response = llm._call(
#             prompt=user_prompt,
#             system_prompt=system,
#             temperature=model_temperature,
#             max_tokens=800,
#         )
        
#         # Extract JSON
#         narrative_obj = _clean_and_extract_json(raw_response)
        
#         # Validate structure
#         if not isinstance(narrative_obj, dict) or not narrative_obj:
#             raise ValueError("Invalid or empty response structure")
        
#         # Ensure all required keys exist with natural fallbacks
#         if not narrative_obj.get("opener"):
#             if opener:
#                 narrative_obj["opener"] = opener
#             else:
#                 narrative_obj["opener"] = generate_dynamic_conversational_opener(question, metrics)
        
#         if not isinstance(narrative_obj.get("insights"), list):
#             narrative_obj["insights"] = _generate_dynamic_fallback_insights(question, rows, metrics)
            
#         if not isinstance(narrative_obj.get("recommendations"), list):
#             narrative_obj["recommendations"] = _generate_dynamic_fallback_recommendations(question, rows, metrics)
            
#         if not narrative_obj.get("next_step"):
#             narrative_obj["next_step"] = _generate_dynamic_next_step(question, metrics)
        
#         return narrative_obj
        
#     except Exception as e:
#         print(f"⚠️ Dynamic narrative generation failed: {e}")
#         return _generate_intelligent_fallback(question, rows, metrics, opener)

# def humanize_narrative(
#     question: str,
#     rows: List[Dict[str, Any]],
#     summary: str,
#     recommendation: str,
#     sql: str,
#     data_window: Dict[str, str] | None = None,
#     model_temperature: float = 0.6,
#     opener: str = None,
#     corpus_insights: Dict[str, Any] | None = None   # ✅ NEW
# ) -> Dict[str, Any]:
#     """
#     Fully dynamic narrative generation - LLM creates all content naturally.
#     If corpus insights exist, they will be blended with SQL analysis.
#     """
#     llm = get_llama_maverick_llm()
    
#     # Extract rich contextual data
#     metrics = _derive_contextual_metrics(question, rows)
#     preview = _to_preview_rows(rows, 20)
    
#     # Enhanced system prompt
#     system = (
#         "You are a skilled data analyst having a natural conversation with a colleague. "
#         "Blend curated knowledge (if available) with fresh SQL data. "
#         "Explain results warmly and professionally.\n\n"

#         "Important rules:\n"
#         "- NEVER say: 'no records', 'no data', 'nothing to pull', 'bit of a surprise', or anything apologetic/negative.\n"
#         "- If results are empty, simply say: 'This query did not return matching records under the current filters.'\n"
#         "- If results exist, focus only on what was found — start with what you did analyze.\n"
#         "- Always keep the tone constructive and forward-looking.\n"
#         "- NO emojis.\n\n"

#         "Response format (JSON only):\n"
#         "{\n"
#         '  "opener": "Natural 2-3 sentence conversation starter",\n'
#         '  "insights": ["Specific finding 1", "Specific finding 2", ...],\n'
#         '  "recommendations": ["Action item 1", "Action item 2", ...],\n'
#         '  "next_step": "Natural offer for follow-up"\n'
#         "}"
#     )
    
#     # Build analysis context with both SQL + corpus
#     analysis_context = {
#         "user_question": question,
#         "data_summary": {
#             "total_rows": len(rows),
#             "columns_available": metrics.get("columns", []),
#             "data_patterns": metrics.get("aggregation_summary", {}),
#             "column_insights": metrics.get("column_analysis", {}),
#             "time_period": data_window or {}
#         },
#         "sample_data": preview,
#         "existing_summary": summary,
#         "existing_recommendation": recommendation,
#         "sql_context": sql,
#     }
    
#     # ✅ Blend corpus insights if provided
#     if corpus_insights:
#         analysis_context["corpus_knowledge"] = {
#             "summary": corpus_insights.get("summary"),
#             "recommendations": corpus_insights.get("recommendations", []),
#             "sql_used_before": corpus_insights.get("sql"),
#             "chart_config": corpus_insights.get("chart_config"),
#             "row_count_previous": corpus_insights.get("row_count")
#         }
    
#     user_prompt = (
#         "Please analyze this data and create a natural, conversational explanation. "
#         "Blend any provided corpus knowledge with fresh SQL insights. "
#         "Write like you're a human analyst talking to a colleague.\n\n"
#         f"ANALYSIS CONTEXT:\n{json.dumps(analysis_context, ensure_ascii=False, indent=2)}\n\n"
#         "Generate a natural narrative that sounds authentically human. "
#         "If both corpus and SQL data are present, merge them seamlessly."
#     )
    
#     try:
#         raw_response = llm._call(
#             prompt=user_prompt,
#             system_prompt=system,
#             temperature=model_temperature,
#             max_tokens=800,
#         )
        
#         narrative_obj = _clean_and_extract_json(raw_response)
        
#         if not isinstance(narrative_obj, dict) or not narrative_obj:
#             raise ValueError("Invalid or empty response structure")
        
#         if not narrative_obj.get("opener"):
#             if opener:
#                 narrative_obj["opener"] = opener
#             else:
#                 narrative_obj["opener"] = generate_dynamic_conversational_opener(question, metrics)
        
#         if not isinstance(narrative_obj.get("insights"), list):
#             narrative_obj["insights"] = _generate_dynamic_fallback_insights(question, rows, metrics)
            
#         if not isinstance(narrative_obj.get("recommendations"), list):
#             narrative_obj["recommendations"] = _generate_dynamic_fallback_recommendations(question, rows, metrics)
            
#         if not narrative_obj.get("next_step"):
#             narrative_obj["next_step"] = _generate_dynamic_next_step(question, metrics)
        
#         return narrative_obj
        
#     except Exception as e:
#         print(f"⚠️ Dynamic narrative generation failed: {e}")
#         return _generate_intelligent_fallback(question, rows, metrics, opener)
# Update the system prompt in humanize_narrative function
def humanize_narrative(
    question: str,
    rows: List[Dict[str, Any]],
    summary: str,
    recommendation: str,
    sql: str,
    data_window: Dict[str, str] | None = None,
    model_temperature: float = 0.6,
    opener: str = None,
    corpus_insights: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Fully dynamic narrative generation - LLM creates all content naturally.
    If corpus insights exist, they will be blended with SQL analysis.
    """
    llm = get_llama_maverick_llm()
    
    # Extract rich contextual data
    metrics = _derive_contextual_metrics(question, rows)
    preview = _to_preview_rows(rows, 20)
    
    # Enhanced system prompt - STRONGER negative phrase prevention
    system = (
        "You are a skilled data analyst having a natural conversation with a colleague. "
        "Blend curated knowledge (if available) with fresh SQL data. "
        "Explain results warmly and professionally.\n\n"

        "CRITICAL RULES FOR TONE:\n"
        "- ABSOLUTELY NEVER use these phrases: 'no records', 'no data', 'nothing to pull', "
        "'bit of a surprise', 'unfortunately', 'however', 'but when I dug in', 'aren't actually any', "
        "'don't have any data', 'nothing found', 'no results to show'\n"
        "- If results are empty, ONLY say: 'This query did not return matching records under the current filters.'\n"
        "- If results exist, START with what you DID find, never what you didn't find\n"
        "- Always maintain constructive, forward-looking tone\n"
        "- NO emojis or apologetic language\n\n"

        "Response format (JSON only):\n"
        "{\n"
        '  "opener": "Natural 2-3 sentence conversation starter (POSITIVE ONLY)",\n'
        '  "insights": ["Specific finding 1", "Specific finding 2", ...],\n'
        '  "recommendations": ["Action item 1", "Action item 2", ...],\n'
        '  "next_step": "Natural offer for follow-up"\n'
        "}"
    )
    
    # Build analysis context with both SQL + corpus
    analysis_context = {
        "user_question": question,
        "data_summary": {
            "total_rows": len(rows),
            "columns_available": metrics.get("columns", []),
            "data_patterns": metrics.get("aggregation_summary", {}),
            "column_insights": metrics.get("column_analysis", {}),
            "time_period": data_window or {}
        },
        "sample_data": preview,
        "existing_summary": summary,
        "existing_recommendation": recommendation,
        "sql_context": sql,
    }
    
    # Blend corpus insights if provided
    if corpus_insights:
        analysis_context["corpus_knowledge"] = {
            "summary": corpus_insights.get("summary"),
            "recommendations": corpus_insights.get("recommendations", []),
            "sql_used_before": corpus_insights.get("sql"),
            "chart_config": corpus_insights.get("chart_config"),
            "row_count_previous": corpus_insights.get("row_count")
        }
    
    user_prompt = (
        "Please analyze this data and create a natural, conversational explanation. "
        "Blend any provided corpus knowledge with fresh SQL insights. "
        "Write like you're a human analyst talking to a colleague.\n\n"
        "TONE REQUIREMENTS:\n"
        "- Be constructive and solution-focused\n"
        "- If no data found, suggest broadening scope rather than dwelling on the lack of results\n"
        "- Focus on what CAN be done, not what couldn't be found\n\n"
        f"ANALYSIS CONTEXT:\n{json.dumps(analysis_context, ensure_ascii=False, indent=2)}\n\n"
        "Generate a natural narrative that sounds authentically human. "
        "If both corpus and SQL data are present, merge them seamlessly."
    )
    
    try:
        raw_response = llm._call(
            prompt=user_prompt,
            system_prompt=system,
            temperature=model_temperature,
            max_tokens=800,
        )
        
        narrative_obj = _clean_and_extract_json(raw_response)
        
        if not isinstance(narrative_obj, dict) or not narrative_obj:
            raise ValueError("Invalid or empty response structure")
        
        # Enhanced validation and filtering for opener
        if not narrative_obj.get("opener"):
            if opener:
                narrative_obj["opener"] = opener
            else:
                narrative_obj["opener"] = generate_dynamic_conversational_opener(question, metrics)
        else:
            # Filter negative phrases from generated opener
            opener_text = narrative_obj["opener"]
            negative_patterns = [
                "no records", "no data", "nothing to pull", "bit of a surprise",
                "unfortunately", "however", "but when I", "aren't actually any",
                "don't have any", "nothing found", "found that there"
            ]
            
            opener_lower = opener_text.lower()
            if any(pattern in opener_lower for pattern in negative_patterns):
                # Replace with positive version
                narrative_obj["opener"] = generate_dynamic_conversational_opener(question, metrics)
        
        if not isinstance(narrative_obj.get("insights"), list):
            narrative_obj["insights"] = _generate_dynamic_fallback_insights(question, rows, metrics)
            
        if not isinstance(narrative_obj.get("recommendations"), list):
            narrative_obj["recommendations"] = _generate_dynamic_fallback_recommendations(question, rows, metrics)
            
        if not narrative_obj.get("next_step"):
            narrative_obj["next_step"] = _generate_dynamic_next_step(question, metrics)
        
        return narrative_obj
        
    except Exception as e:
        print(f"⚠️ Dynamic narrative generation failed: {e}")
        return _generate_intelligent_fallback(question, rows, metrics, opener)

# def _generate_dynamic_fallback_insights(question: str, rows: List[Dict], metrics: Dict) -> List[str]:
#     """Generate insights using LLM even in fallback mode"""
#     llm = get_llama_maverick_llm()
    
#     if not rows:
#         return ["No matching data was found for your query."]
    
#     context = {
#         "question": question,
#         "row_count": len(rows),
#         "key_data": metrics.get("aggregation_summary", {}),
#         "columns": metrics.get("columns", [])
#     }
    
#     prompt = (
#         f"Generate 2-3 natural insights about this data analysis:\n"
#         f"{json.dumps(context, indent=2)}\n\n"
#         "Write like a human analyst. Be specific. Return as a JSON array of strings."
#         "Don't return like this below:So it looks like we don't have any records to analyze, but interestingly, the query still returned results, which is a bit unexpected."
#     )
    
#     try:
#         response = llm._call(prompt=prompt, temperature=0.6, max_tokens=300)
#         insights = json.loads(response)
#         if isinstance(insights, list):
#             return insights[:4]
#     except:
#         pass
    
#     # Simple fallback
#     return [
#         f"I analyzed {len(rows)} records from your query.",
#         f"The data includes {len(metrics.get('columns', []))} different data points.",
#         "Here are the key patterns I identified."
#     ]


# def _generate_dynamic_fallback_insights(question: str, rows: List[Dict], metrics: Dict) -> List[str]:
#     """Generate insights using LLM even in fallback mode"""
#     llm = get_llama_maverick_llm()

#     if not rows:
#         # Neutral fallback for empty results
#         return ["This query did not return any matching records under the current filters."]

#     context = {
#         "question": question,
#         "row_count": len(rows),
#         "key_data": metrics.get("aggregation_summary", {}),
#         "columns": metrics.get("columns", [])
#     }

#     prompt = (
#         f"Generate 2-3 natural insights about this data analysis:\n"
#         f"{json.dumps(context, indent=2)}\n\n"
#         "Write like a human analyst. Be specific. "
#         "Do not use phrases like 'no data', 'no records', 'nothing to analyze', "
#         "or 'bit of a surprise'. Keep tone constructive and professional.\n\n"
#         "Return as a JSON array of strings."
#     )

#     try:
#         response = llm._call(prompt=prompt, temperature=0.6, max_tokens=300)
#         insights = json.loads(response)
#         if isinstance(insights, list):
#             return insights[:4]
#     except:
#         pass

#     # Simple static fallback (neutral wording only)
#     return [
#         f"I analyzed {len(rows)} records from your query.",
#         f"The dataset includes {len(metrics.get('columns', []))} different fields.",
#         "Here are the key patterns I identified."
#     ]
# Update the fallback insights function to be more positive
def _generate_dynamic_fallback_insights(question: str, rows: List[Dict], metrics: Dict) -> List[str]:
    """Generate insights using LLM even in fallback mode"""
    llm = get_llama_maverick_llm()

    if not rows:
        # Completely neutral, solution-focused fallback for empty results
        return [
            "This query did not return matching records under the current filters.",
            "Consider adjusting the time period or expanding the geographic scope.",
            "The query structure is valid and ready for broader parameter testing."
        ]

    context = {
        "question": question,
        "row_count": len(rows),
        "key_data": metrics.get("aggregation_summary", {}),
        "columns": metrics.get("columns", [])
    }

    prompt = (
        f"Generate 2-3 natural insights about this data analysis:\n"
        f"{json.dumps(context, indent=2)}\n\n"
        "Write like a human analyst. Be specific and constructive. "
        "NEVER use phrases like: 'no data', 'no records', 'nothing to analyze', "
        "'bit of a surprise', 'unfortunately', 'however', 'but when I'.\n"
        "Focus on what WAS found and analyzed.\n\n"
        "Return as a JSON array of strings."
    )

    try:
        response = llm._call(prompt=prompt, temperature=0.6, max_tokens=300)
        insights = json.loads(response)
        if isinstance(insights, list):
            # Filter out any negative insights that might slip through
            filtered_insights = []
            negative_phrases = ["no data", "no records", "nothing", "unfortunately", "however", "but when"]
            
            for insight in insights[:4]:
                insight_lower = insight.lower()
                if not any(phrase in insight_lower for phrase in negative_phrases):
                    filtered_insights.append(insight)
            
            if filtered_insights:
                return filtered_insights
    except:
        pass

    # Simple static fallback (positive wording only)
    return [
        f"I analyzed {len(rows)} records from your query.",
        f"The dataset includes {len(metrics.get('columns', []))} different fields with useful information.",
        "Here are the key patterns and trends I identified from the data."
    ]

def _generate_dynamic_fallback_recommendations(question: str, rows: List[Dict], metrics: Dict) -> List[str]:
    """Generate recommendations using LLM even in fallback mode"""
    llm = get_llama_maverick_llm()
    
    context = {
        "question_intent": question,
        "data_size": len(rows),
        "patterns_found": bool(metrics.get("aggregation_summary"))
    }
    
    prompt = (
        f"Based on this analysis context, suggest 2-3 natural next actions:\n"
        f"{json.dumps(context, indent=2)}\n\n"
        "Sound like a helpful analyst colleague. Return as JSON array of strings."
    )
    
    try:
        response = llm._call(prompt=prompt, temperature=0.6, max_tokens=200)
        recs = json.loads(response)
        if isinstance(recs, list):
            return recs[:4]
    except:
        pass
    
    # Simple fallback
    return [
        "Consider exploring different time periods for comparison.",
        "You might want to segment this data by additional categories."
    ]

def _generate_dynamic_next_step(question: str, metrics: Dict) -> str:
    """Generate next step using LLM"""
    llm = get_llama_maverick_llm()
    
    prompt = (
        f"Given this analysis of '{question}' with {metrics.get('row_count', 0)} results, "
        "what would you naturally offer as a follow-up? One conversational sentence."
    )
    
    try:
        response = llm._call(prompt=prompt, temperature=0.7, max_tokens=100)
        next_step = response.strip().strip('"').strip("'")
        if len(next_step) > 10:
            return next_step
    except:
        pass
    
    return "Want to dive deeper into any of these findings?"

def _generate_intelligent_fallback(question: str, rows: List[Dict], metrics: Dict, opener: str = None) -> Dict[str, Any]:
    """Even the fallback uses dynamic LLM generation"""
    return {
        "opener": opener or generate_dynamic_conversational_opener(question, metrics),
        "insights": _generate_dynamic_fallback_insights(question, rows, metrics),
        "recommendations": _generate_dynamic_fallback_recommendations(question, rows, metrics),
        "next_step": _generate_dynamic_next_step(question, metrics)
    }