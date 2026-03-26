from django.contrib import messages
from django.shortcuts import redirect, render

from user_settings.forms import TranscriptionRuntimeSettingsForm
from user_settings.services import get_runtime_settings, update_runtime_settings


def transcription_runtime_settings_view(request):
    settings_object = get_runtime_settings()
    if request.method == "POST":
        form = TranscriptionRuntimeSettingsForm(request.POST, instance=settings_object)
        if form.is_valid():
            update_runtime_settings(**form.cleaned_data)
            messages.success(request, "Transcription runtime settings updated successfully")
            return redirect("transcription-runtime-settings")
    else:
        form = TranscriptionRuntimeSettingsForm(instance=settings_object)

    return render(
        request,
        "user_settings/transcription_runtime_settings.html",
        {
            "form": form,
            "settings_object": settings_object,
        }
    )