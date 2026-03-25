from django.apps import AppConfig


class PipelinesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'pipelines'

    def ready(self) -> None:
        from pipelines.runtime import start_embedded_workers
        start_embedded_workers()