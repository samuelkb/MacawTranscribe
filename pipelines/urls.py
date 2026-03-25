from django.urls import path

from pipelines.views import upload_recording_ui, upload_normalize_recording, \
    upload_normalize_and_diarize_recording_view, upload_normalize_diarize_and_vad_recording_view, \
    upload_normalize_diarize_vad_and_chunk_recording_view, start_full_transcription_view, recording_progress_view

app_name = "pipelines"

urlpatterns = [
    path('upload-and-normalize/', upload_normalize_recording, name='upload_and_normalize_recording'),
    path('upload-ui/', upload_recording_ui, name='main_pipeline_ui'),
    path("upload-normalize-diarize/", upload_normalize_and_diarize_recording_view, name="upload_normalize_and_diarize_recording"),
    path("upload-normalize-diarize-vad/", upload_normalize_diarize_and_vad_recording_view, name="upload_normalize_diarize_and_vad_recording"),
    path("upload-normalize-diarize-vad-chunk/", upload_normalize_diarize_vad_and_chunk_recording_view, name="upload_normalize_diarize_vad_and_chunk_recording"),
    path("transcribe/", start_full_transcription_view, name='start_full_transcription'),
    path("recordings/<uuid:recording_id>/progress/", recording_progress_view, name='recording_progress'),
]