from django.urls import path

from transcriptions.views import transcribe_chunk_view

app_name = 'transcriptions'

urlpatterns = [
    path("chunks/<uuid:chunk_id>/transcribe/", transcribe_chunk_view, name="transcribe_chunk")
]