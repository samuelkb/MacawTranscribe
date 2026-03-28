from django import forms
from django.utils.translation import gettext_lazy

from user_settings.models import TranscriptionRuntimeSettings


class TranscriptionRuntimeSettingsForm(forms.ModelForm):
    class Meta:
        model = TranscriptionRuntimeSettings
        fields = [
            "supervisor_enabled",
            "desired_worker_count",
            "max_worker_count",
            "max_job_per_worker",
            "max_idle_seconds",
            "supervisor_poll_seconds",
            "guarded_scaling_enabled",
            "max_memory_percent",
            "min_available_memory_mb",
            "estimated_memory_per_worker_mb",
            "cpu_guard_enabled",
            "max_cpu_percent",
            "default_backend",
            "default_model",
        ]
        labels ={
            "supervisor_enabled": gettext_lazy("Supervisor enabled"),
            "desired_worker_count": gettext_lazy("Desired worker count"),
            "max_worker_count": gettext_lazy("Max worker count"),
            "max_job_per_worker": gettext_lazy("Max jobs per worker"),
            "max_idle_seconds": gettext_lazy("Max idle seconds"),
            "supervisor_poll_seconds": gettext_lazy("Supervisor poll seconds"),
            "guarded_scaling_enabled": gettext_lazy("Guarded scaling enabled"),
            "max_memory_percent": gettext_lazy("Max memory percent per worker"),
            "min_available_memory_mb": gettext_lazy("Min memory per worker in mb"),
        }
        widgets = {
            "supervisor_enabled": forms.CheckboxInput(),
            "guarded_scaling_enabled": forms.CheckboxInput(),
            "cpu_guard_enabled": forms.CheckboxInput(),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "supervisor_enabled" in self.fields:
            self.fields["supervisor_enabled"].disabled = True

        advanced_fields = {
            "max_idle_seconds",
            "supervisor_poll_seconds",
            "guarded_scaling_enabled",
            "max_memory_percent",
            "min_available_memory_mb",
            "estimated_memory_per_worker_mb",
            "cpu_guard_enabled",
            "max_cpu_percent",
            "default_backend",
            "default_model",
        }
        for name, field in self.fields.items():
            css = "form-input"
            if name in advanced_fields:
                css += " advanced-setting"
            field.widget.attrs["class"] = css
