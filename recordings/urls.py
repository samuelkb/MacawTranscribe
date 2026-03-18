from django.urls import path

from recordings.views import upload_recording, upload_recording_ui

app_name = "recordings"

urlpatterns = [
    path('upload/', upload_recording, name='upload_recording'),
    path('upload-ui/', upload_recording_ui, name='upload_recording_ui'),
]