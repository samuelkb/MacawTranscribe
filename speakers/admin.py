from django.contrib import admin

from speakers.models import SpeakerSegment, SilenceSegment, SpeakerLabel


@admin.register(SpeakerSegment)
class SpeakerSegmentAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "recording_id",
        "speaker_id",
        "start_time",
        "end_time",
        "created_at"
    )
    list_filter = ("speaker_id", "created_at")
    search_fields = ("id", "recording__id", "speaker_id")
    readonly_fields = ("id","created_at")


@admin.register(SilenceSegment)
class SilenceSegmentAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "recording_id",
        "start_time",
        "end_time",
        "created_at",
    )
    list_filter = ("created_at",)
    search_fields = ("id", "recording__id")
    readonly_fields = ("id","created_at")


@admin.register(SpeakerLabel)
class ModelNameAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "recording_id",
        "speaker_id",
        "display_name",
        "created_at",
        "updated_at",
    )
    list_filter = ("created_at", "updated_at")
    search_fields = ("id", "recording__id", "speaker_id", "display_name")
    readonly_fields = ("id","created_at", "updated_at")
