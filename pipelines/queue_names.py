from ml.types import BackendName, ModelName


def build_transcription_partition_key(*, backend: BackendName, model: ModelName) -> str:
    """
    Build the canonical worker/job partition key.
    """
    return f"{backend.name}__{model.name}"

def build_diarization_partition_key(*, backend: str, model: str) -> str:
    """
    Build the canonical worker/job partition key.
    """
    return f"{backend}__{model}"

def build_transcription_queue_name(*, backend: BackendName, model: ModelName) -> str:
    """
    Build the Redis queue name for transcription jobs.
    """
    partition_key = build_transcription_partition_key(backend=backend, model=model)
    return f"transcription_jobs:{partition_key}"

def build_diarization_queue_name(*, backend: str, model: str) -> str:
    """
    Build the Redis queue name for diarization jobs.
    """
    partition_key = build_diarization_partition_key(backend=backend, model=model)
    return f"diarization_jobs:{partition_key}"