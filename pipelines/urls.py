from django.urls import path

from pipelines.views import upload_recording_ui, upload_normalize_recording, \
    upload_normalize_and_diarize_recording_view, upload_normalize_diarize_and_vad_recording_view

app_name = "pipelines"

urlpatterns = [
    path('upload-and-normalize/', upload_normalize_recording, name='upload_and_normalize_recording'),
    path('upload-ui/', upload_recording_ui, name='main_pipeline_ui'),
    path("upload-normalize-diarize/", upload_normalize_and_diarize_recording_view, name="upload_normalize_and_diarize_recording"),
    path("upload-normalize-diarize-vad/", upload_normalize_diarize_and_vad_recording_view, name="upload_normalize_diarize_and_vad_recording"),
]