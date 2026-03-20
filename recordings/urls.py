from django.urls import path

from recordings.views import upload_recording, create_chunks_view

app_name = "recordings"

urlpatterns = [
    path('upload/', upload_recording, name='upload_recording'),
    path('<uuid:recording_id>/chunks/', create_chunks_view, name='create_chunks')
]