from django.urls import path

from recordings.views import upload_recording, create_chunks_view, recording_detail_view, normalized_audio_view

app_name = "recordings"

urlpatterns = [
    path('upload/', upload_recording, name='upload_recording'),
    path('<uuid:recording_id>/', recording_detail_view, name='recording_detail'),
    path('<uuid:recording_id>/normalized-audio/', normalized_audio_view, name='normalized_audio'),
    path('<uuid:recording_id>/chunks/', create_chunks_view, name='create_chunks')
]
