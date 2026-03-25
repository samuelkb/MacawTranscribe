from dataclasses import dataclass, asdict
from uuid import UUID

from ml.types import BackendName, ModelName


@dataclass(frozen=True)
class TranscriptionJob:
    """
    Job payload for chunk transcription
    """
    chunk_id: UUID
    backend: BackendName
    model: ModelName

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        return {
            "chunk_id": str(payload["chunk_id"]),
            "backend": payload["backend"].value,
            "model": payload["model"].value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, str]) -> "TranscriptionJob":
        return cls(
            chunk_id=UUID(payload["chunk_id"]),
            backend=BackendName(payload["backend"]),
            model=ModelName(payload["model"]),
        )
