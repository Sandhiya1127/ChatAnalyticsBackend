
from django.utils.timezone import now
from sqlalchemy import text
# from ..feedback_view import general_db
from langchain.schema import Document

def jsonl_line(data):
    return str(data) + "\n"