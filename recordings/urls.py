from django.urls import path

from recordings.views import upload_recording

app_name = "recordings"

urlpatterns = [
    path('upload/', upload_recording, name='upload_recording'),
]