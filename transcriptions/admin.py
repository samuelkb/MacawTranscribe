from django.contrib import admin

from transcriptions.models import TranscriptWord, Transcript, Edit, TranscriptCandidate


@admin.register(TranscriptWord)
class TranscriptWordAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "chunk",
        "word_index",
        "text",
        "start_time",
        "end_time",
        "model_used",
        "created_at",
    )
    list_filter = ("model_used", "created_at")
    search_fields = ("id","chunk__id", "text")
    readonly_fields = ("id","created_at",)


@admin.register(Transcript)
class TranscriptAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "chunk",
        "model_used",
        "created_at",
        "updated_at",
    )
    list_filter = ("model_used", "created_at", "updated_at")
    search_fields = ("id", "chunk__id", "accepted_text")
    readonly_fields = ("id","created_at","updated_at")


@admin.register(Edit)
class EditAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "transcript",
        "editor",
        "created_at",
    )
    list_filter = ("editor", "created_at")
    search_fields = ("id", "transcript__id", "edited_text")
    readonly_fields = ("id", "created_at")


@admin.register(TranscriptCandidate)
class TranscriptCandidateAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "chunk",
        "model_used",
        "is_from_retry",
        "accepted",
        "rejected",
        "created_at",
    )
    list_filter = ("model_used", "is_from_retry", "accepted", "rejected", "created_at")
    search_fields = ("id", "chunk__id", "candidate_text")
    readonly_fields = ("id", "created_at", "accepted_at", "rejected_at")
