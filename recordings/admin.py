from django.contrib import admin

from .models import Recording, Chunk

@admin.register(Recording)
class RecordingAdmin(admin.ModelAdmin):
    """
    Recording admin model for recordings
    """
    list_display = (
        "id",
        "original_file_name",
        "duration_milliseconds",
        "status",
        "created_at",
        "updated_at",
    )
    list_filter = (
        "status",
        "created_at",
    )
    search_fields = ["id", "original_file_name"]
    readonly_fields = ("id", "created_at", "updated_at")


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    """
    Chunk admin model for recordings
    """
    list_display = (
        "id",
        "recording",
        "chunk_index",
        "start_time",
        "end_time",
        "status",
        "worker_id",
        "attempt_count",
        "has_pending_candidate",
        "updated_at",
    )
    list_filter = ("status", "has_pending_candidate", "created_at")
    search_fields = ("id", "recording__id", "recording__original_file_name", "worker_id")
    readonly_fields = ("id", "created_at", "updated_at")
