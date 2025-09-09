# from django.apps import AppConfig
# import subprocess
# import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# class GenaiAppConfig(AppConfig):
#     default_auto_field = 'django.db.models.BigAutoField'
#     name = 'genai_app'


#     def ready(self):
#         """
#         Automatically run the Qwen2.5 model using Ollama when Django starts.
#         """
#         try:
#             # Ping Ollama locally
#             requests.get("http://localhost:11434", timeout=2)
#             print("✅ Ollama is already running.")
#         except requests.exceptions.RequestException:
#             try:
#                 print("🚀 Starting Ollama (Qwen2.5) in background...")
#                 subprocess.Popen(["ollama", "run", "qwen2.5:1.5b","qwen2.5:7b"])
#             except Exception as inner:
#                 print(f"❌ Failed to start Ollama: {inner}")


# In your apps.py (e.g., genai_app/apps.py)
from django.apps import AppConfig
import subprocess
import threading
import requests
import os

class GenaiAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'genai_app'

    def ready(self):
        try:
            # ✅ Check if Ollama is running
            requests.get("http://localhost:11434", timeout=2)
            print("✅ Ollama is already running.")
        except requests.exceptions.RequestException:
            try:
                print("🚀 Starting Ollama (Qwen2.5) in background...")
                subprocess.Popen(["ollama", "run", "qwen2.5:1.5b", "qwen2.5:7b"])
            except Exception as inner:
                print(f"❌ Failed to start Ollama: {inner}")

        # ✅ Only run evaluation ONCE during development server start
        if os.environ.get("RUN_MAIN") == "true":
            from django.core.management import call_command
            threading.Thread(target=self.run_evaluation).start()

    def run_evaluation(self):
        try:
            print("📊 Auto-running evaluate_model...")
            from django.core.management import call_command
            call_command("evaluate_model")
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
