
from django.urls import path
from .views import upload_file, ask_qwen,get_session_info,connect_database,ask_questionpdf, upload_pdf, ask_questionbot,upload_pdfbot,ask_question,check_intent,ask_question_stream,inspect_chroma
from .views import inspect_general_chroma
from genai_app.feedback.feedback_view import feedback_view 
from genai_app.report.report import generate_response_report

from django.views.generic import RedirectView
from django.templatetags.static import static

from django.http import HttpResponse


urlpatterns = [

    path('upload/', upload_file, name='upload_file'),
    path('api/ask_qwen/', ask_qwen, name='ask_qwen'),
    path('session/', get_session_info, name='get_session_info'),
    path('api/upload_pdf/', upload_pdf),
    path('api/upload_pdfbot/', upload_pdfbot),
    path('api/ask/', ask_questionpdf),
    path('api/askbot/', ask_questionbot),
    path('api/connect_database/', connect_database, name='connect_database'),
    path('api/ask_question/', ask_question, name='ask_question'),
    path('api/check_intent/', check_intent, name='check_intent'),
    path("api/feedback/", feedback_view),
    path("api/generate_response_report/", generate_response_report),
    path('api/ask_question_stream/', ask_question_stream, name='ask_question_stream'),
    path("api/inspect_chroma/", inspect_chroma, name="inspect_chroma"),
    path("api/inspect_general_chroma/", inspect_general_chroma, name="inspect_general_chroma"),
    path("favicon.ico", RedirectView.as_view(url=static("favicon.ico"))),



]