from django.urls import path

from user_settings.views import transcription_runtime_settings_view

urlpatterns = [
    path("transcription-runtime/", transcription_runtime_settings_view, name="transcription-runtime-settings"),
]