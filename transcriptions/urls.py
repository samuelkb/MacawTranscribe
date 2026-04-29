from django.urls import path

from transcriptions.views import transcribe_chunk_on_demand_view, full_recording_transcription_view, \
    assembled_recording_page_view, download_assembled_recording_view

app_name = 'transcriptions'

urlpatterns = [
    path("chunks/<uuid:chunk_id>/transcribe/", transcribe_chunk_on_demand_view, name="transcribe_chunk"),
    path("recordings/<uuid:recording_id>/full-transcript", full_recording_transcription_view, name="full_recording_transcript"),
    path("recordings/<uuid:recording_id>/assembled/", assembled_recording_page_view, name="assembled_recording_page"),
    path("recordings/<uuid:recording_id>/assembled/download/", download_assembled_recording_view, name="download_assembled_recording"),
]
