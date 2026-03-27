from django.contrib import admin

from user_settings.models import TranscriptionRuntimeSettings, WorkerProcessState


@admin.register(TranscriptionRuntimeSettings)
class TranscriptionRuntimeSettingsAdmin(admin.ModelAdmin):
    list_display = [
        "supervisor_enabled",
        "desired_worker_count",
        "max_worker_count",
        "guarded_scaling_enabled",
        "updated_at"
    ]
    readonly_fields = ["created_at", "updated_at"]
    fieldsets = (
    (
        "Supervisor",
        {
            "fields": (
                "supervisor_enabled",
                "supervisor_poll_seconds"
            )
        }
    ),
    (
        "Worker Pool",
        {
            "fields": (
                "desired_worker_count",
                "max_worker_count",
                "max_job_per_worker",
                "max_idle_seconds"
            )
        }
    ),
    (
        "Guarded Scaling",
        {
            "fields": (
                "guarded_scaling_enabled",
                "min_available_memory_mb",
                "estimated_memory_per_worker_mb",
                "cpu_guard_enabled",
                "max_cpu_percent"
            )
        }
    ),
    (
        "Defaults",
        {
            "fields": (
                "default_backend",
                "default_model"
            )
        }
    ),
    (
        "Metadata",
        {
            "fields": (
                "created_at",
                "updated_at",
            )
        }
    )
    )

    def has_add_permission(self, request) -> bool:
        return not TranscriptionRuntimeSettings.objects.exists()

    def has_delete_permission(self, request, obj=None) -> bool:
        return False


@admin.register(WorkerProcessState)
class WorkerProcessStateAdmin(admin.ModelAdmin):
    list_display = (
        "worker_id",
        "role",
        "pid",
        "status",
        "backend",
        "model",
        "jobs_processed",
        "current_chunk_id",
        "last_heartbeat_at",
        "idle_since",
        "started_at",
        "stopped_at",
    )

    list_filter = (
        "role",
        "status",
        "backend",
        "model",
    )

    search_fields = (
        "worker_id",
        "pid",
        "current_chunk_id",
        "exit_reason",
        "last_error",
        "hostname"
    )

    readonly_fields = (
        "worker_id",
        "pid",
        "role",
        "backend",
        "model",
        "jobs_processed",
        "current_chunk_id",
        "exit_reason",
        "last_error",
        "hostname",
        "started_at",
        "last_heartbeat_at",
        "idle_since",
        "stopped_at",
        "created_at",
        "updated_at",
    )

    def has_add_permission(self, request) -> bool:
        return False
