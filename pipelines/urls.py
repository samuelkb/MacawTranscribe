from django.urls import path

from pipelines.views import upload_recording_ui, upload_normalize_recording

app_name = "pipelines"

urlpatterns = [
    path('upload-and-normalize/', upload_normalize_recording, name='upload_and_normalize_recording'),
    path('upload-ui/', upload_recording_ui, name='main_pipeline_ui'),
]