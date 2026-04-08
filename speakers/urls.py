from django.urls import path

from speakers.views import save_speaker_label_view

app_name = "speakers"

urlpatterns = [
    path("recordings/<uuid:recording_id>/labels/", save_speaker_label_view, name="save_speaker_label"),
]
