from django.db import migrations


def create_transcription_runtime_settings(apps, schema_editor):
    TranscriptionRuntimeSettings = apps.get_model("user_settings", "TranscriptionRuntimeSettings")
    TranscriptionRuntimeSettings.objects.get_or_create(
        pk=1,
        defaults={
            "singleton_enforcer": True,
            "supervisor_enabled": True,
            "desired_worker_count": 2,
            "max_worker_count": 4,
            "max_job_per_worker": 3,
            "max_idle_seconds": 300,
            "supervisor_poll_seconds": 5,
            "guarded_scaling_enabled": True,
            "max_memory_percent": 85,
            "min_available_memory_mb": 12000,
            "estimated_memory_per_worker_mb": 12000,
            "cpu_guard_enabled": False,
            "max_cpu_percent": 85,
            "default_backend": "mlx-whisper",
            "default_model": "medium",
        },
    )


def delete_transcription_runtime_settings(apps, schema_editor):
    TranscriptionRuntimeSettings = apps.get_model("user_settings", "TranscriptionRuntimeSettings")
    TranscriptionRuntimeSettings.objects.filter(pk=1).delete()


class Migration(migrations.Migration):
    dependencies = [("user_settings", "0001_initial")]
    operations = [
        migrations.RunPython(
            create_transcription_runtime_settings,
            delete_transcription_runtime_settings,
        ),
    ]