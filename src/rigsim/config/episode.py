"""Episode export schema for Project 2 (EpisodeForge)."""

from pydantic import BaseModel, Field

from rigsim.config.session import Stream
from rigsim.config.types import QAVerdict


class Episode(BaseModel):
    """Handoff package for Project 2: EpisodeForge."""

    schema_version: str = "1.0.0"
    episode_id: str
    source_session_id: str
    t_start_ns: int
    t_end_ns: int
    duration_s: float
    streams: list[Stream]
    frames_count: dict[str, int]
    calibration_hash: str
    git_commit: str
    config_hash: str
    qa_verdict: QAVerdict = QAVerdict.PASS
    data_uris: dict[str, str] = Field(
        default_factory=dict, description="Pointers to synchronized arrays or mcap slices"
    )
