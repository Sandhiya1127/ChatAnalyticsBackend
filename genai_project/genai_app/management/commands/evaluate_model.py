from django.core.management.base import BaseCommand
from sentence_transformers import SentenceTransformer, util
from genai_app.views import run_sql_generation_graph
from genai_app.views import extract_sql_block, session_store, conversation_memory_store
from sqlalchemy import text
import os, json, re

model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_sql(sql):
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    return sql

def compare_sql(a, b):
    return clean_sql(a) == clean_sql(b)

def compare_output(a, b):
    return str(a).lower() == str(b).lower()

def compare_text(a, b, threshold=0.85):
    if not a or not b:
        return False
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item() >= threshold

def execute_sql(sql, engine):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            return [dict(row._mapping) for row in result]
    except Exception as e:
        print("❌ SQL Execution Error:", str(e))
        return []

class Command(BaseCommand):
    help = "Evaluate model accuracy using actual session questions and outputs"

    def handle(self, *args, **kwargs):
        session_stats = {}

        for session_id, session_data in session_store.items():
            engine = session_data.get("engine")
            if not engine:
                continue

            history = conversation_memory_store.get(session_id, [])
            if not history:
                continue

            self.stdout.write(f"\n📦 Evaluating Session: {session_id} ({len(history)} questions)")
            results = []

            for i, message in enumerate(history):
                question = message.get("question")
                expected_sql = message.get("sql")
                expected_output = message.get("output")
                expected_summary = message.get("summary")
                expected_recommendation = message.get("recommendation")

                if not question or not expected_sql:
                    continue

                self.stdout.write(f"\n🔍 Question {i + 1}: {question}")

                try:
                    sql, recommendation, summary = run_sql_generation_graph(
                        question=question, user_id=session_id, db_id=session_id, history=[]
                    )

                    sql = extract_sql_block(sql)
                    result_rows = execute_sql(sql, engine)

                    correct_sql = compare_sql(sql, expected_sql)
                    correct_output = compare_output(result_rows, expected_output)
                    correct_summary = compare_text(summary, expected_summary)
                    correct_recommendation = compare_text(recommendation, expected_recommendation)

                    results.append({
                        "question": question,
                        "correct_sql": correct_sql,
                        "correct_output": correct_output,
                        "correct_summary": correct_summary,
                        "correct_recommendation": correct_recommendation
                    })

                    if not correct_sql:
                        self.stdout.write("❌ SQL mismatch")
                    if not correct_output:
                        self.stdout.write("❌ Output mismatch")
                    if not correct_summary:
                        self.stdout.write("❌ Summary mismatch")
                    if not correct_recommendation:
                        self.stdout.write("❌ Recommendation mismatch")

                except Exception as e:
                    self.stderr.write(f"💥 Error in question {i + 1}: {e}")

            session_stats[session_id] = results

        # Per-session summary
        self.stdout.write("\n📊 Per-Session Accuracy Report:")
        for session, results in session_stats.items():
            total = len(results)
            if total == 0:
                continue
            sql_acc = sum(r["correct_sql"] for r in results) / total * 100
            out_acc = sum(r["correct_output"] for r in results) / total * 100
            sum_acc = sum(r["correct_summary"] for r in results) / total * 100
            rec_acc = sum(r["correct_recommendation"] for r in results) / total * 100
            overall = (sql_acc + out_acc + sum_acc + rec_acc) / 4

            self.stdout.write(f"\nSession: {session}")
            self.stdout.write(f"Total Questions:         {total}")
            self.stdout.write(f"SQL Accuracy:            {sql_acc:.2f}%")
            self.stdout.write(f"Output Accuracy:         {out_acc:.2f}%")
            self.stdout.write(f"Summary Accuracy:        {sum_acc:.2f}%")
            self.stdout.write(f"Recommendation Accuracy: {rec_acc:.2f}%")
            self.stdout.write(f"Overall Accuracy:        {overall:.2f}%")
