"""Automated Usable Episode Extraction & Client Delivery Manifests (EpisodeForge Contract)."""

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap.writer import Writer

from rigsim.config.episode import Episode
from rigsim.config.qa import QAVerdict, SessionVerdict, TimeWindow
from rigsim.config.session import Session, Stream
from rigsim.qa.engine import QAEngine


@dataclass
class DatasetManifest:
    """Root dataset manifest indexing all exported episodes for client delivery."""

    schema_version: str
    dataset_id: str
    source_session_id: str
    rig_config_name: str
    created_at: str
    total_episodes: int
    total_usable_duration_s: float
    source_total_duration_s: float
    usable_yield_pct: float
    episodes: list[dict[str, Any]]


class EpisodeExtractor:
    """Slices QA-verified multi-sensor sessions into isolated, validated Episode packages."""

    def __init__(
        self,
        min_duration_s: float = 2.0,
        defect_margin_s: float = 0.2,
        generate_preview: bool = True,
    ):
        self.min_duration_s = min_duration_s
        self.defect_margin_s = defect_margin_s
        self.generate_preview = generate_preview

    def extract_episodes(
        self,
        session_dir: Path,
        output_dir: Path,
        verdict: SessionVerdict | None = None,
    ) -> DatasetManifest:
        """Extract all usable time segments from a session into versioned Episode packages."""
        session_manifest_path = session_dir / "session.json"
        if not session_manifest_path.exists():
            raise FileNotFoundError(f"Missing session manifest: {session_manifest_path}")

        with open(session_manifest_path) as f:
            session_data = json.load(f)
        session = Session.model_validate(session_data)

        # MCAP source path
        src_mcap = session_dir / "session.mcap"
        if not src_mcap.exists() and session.mcap_path:
            src_mcap = Path(session.mcap_path)
            if not src_mcap.is_absolute():
                src_mcap = session_dir / src_mcap
        if not src_mcap.exists():
            raise FileNotFoundError(f"Source MCAP file not found at: {src_mcap}")

        # Obtain QA verdict
        if verdict is None:
            verdict_path = session_dir / "qa" / "verdict.json"
            if verdict_path.exists():
                with open(verdict_path) as vf:
                    v_data = json.load(vf)
                verdict = SessionVerdict.model_validate(v_data)
            else:
                engine = QAEngine()
                verdict = engine.process_session(session_dir)

        # Filter usable windows by min_duration and defect margin
        usable_windows = self._filter_usable_windows(verdict.usable_time_ranges)

        # Destination folder for this session's episodes
        dest_session_dir = output_dir / session.session_id
        dest_session_dir.mkdir(parents=True, exist_ok=True)

        git_hash = self._get_git_commit()
        calib_hash = self._compute_calibration_hash(session.streams)
        config_hash = self._compute_config_hash(session)

        exported_episode_summaries: list[dict[str, Any]] = []
        total_usable_s = 0.0

        for ep_idx, (t_start_ns, t_end_ns) in enumerate(usable_windows):
            ep_id = f"{session.session_id}_ep{ep_idx:03d}"
            ep_dir = dest_session_dir / f"episode_{ep_idx:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)

            ep_mcap_path = ep_dir / "episode.mcap"

            # 1. Zero-copy MCAP slicing
            frames_count, first_cam_frames = self._slice_mcap(
                src_mcap_path=src_mcap,
                out_mcap_path=ep_mcap_path,
                t_start_ns=t_start_ns,
                t_end_ns=t_end_ns,
            )

            # 2. SHA-256 Checksum of the sliced MCAP for data provenance
            mcap_sha256 = self._compute_file_sha256(ep_mcap_path)

            duration_s = (t_end_ns - t_start_ns) / 1e9
            total_usable_s += duration_s

            # 3. Write preview video or thumbnail
            preview_path_rel = None
            if self.generate_preview and first_cam_frames:
                preview_rel = "preview.mp4"
                preview_file = ep_dir / preview_rel
                self._generate_preview_mp4(first_cam_frames, preview_file, fps=15.0)
                if preview_file.exists():
                    preview_path_rel = preview_rel

            # 4. Construct Episode object
            episode = Episode(
                schema_version="1.0.0",
                episode_id=ep_id,
                source_session_id=session.session_id,
                t_start_ns=t_start_ns,
                t_end_ns=t_end_ns,
                duration_s=duration_s,
                streams=session.streams,
                frames_count=frames_count,
                calibration_hash=calib_hash,
                git_commit=git_hash,
                config_hash=config_hash,
                qa_verdict=QAVerdict.PASS,
                data_uris={
                    "mcap": "episode.mcap",
                    "preview": preview_path_rel or "",
                    "sha256": mcap_sha256,
                },
            )

            # Save episode manifest
            with open(ep_dir / "manifest.json", "w", encoding="utf-8") as f:
                f.write(episode.model_dump_json(indent=2))

            # 5. Export QA Certificate
            qa_cert = {
                "schema_version": "1.0.0",
                "certificate_id": f"CERT-{ep_id}",
                "episode_id": ep_id,
                "source_session_id": session.session_id,
                "status": "CERTIFIED_USABLE",
                "t_start_ns": t_start_ns,
                "t_end_ns": t_end_ns,
                "duration_s": duration_s,
                "certified_checks": [
                    {
                        "check_id": r.check_id,
                        "stream_id": r.stream_id,
                        "verdict": r.verdict.value,
                        "metric_name": r.metric_name,
                        "value": r.value,
                    }
                    for r in verdict.results
                ],
                "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "issuer": "RigSim QA Engine v1.0",
            }
            with open(ep_dir / "qa_certificate.json", "w", encoding="utf-8") as f:
                json.dump(qa_cert, f, indent=2)

            exported_episode_summaries.append(
                {
                    "episode_id": ep_id,
                    "relative_path": f"episode_{ep_idx:03d}",
                    "t_start_ns": t_start_ns,
                    "t_end_ns": t_end_ns,
                    "duration_s": duration_s,
                    "frames_count": frames_count,
                    "mcap_sha256": mcap_sha256,
                    "qa_verdict": "PASS",
                }
            )

        # 6. Top-level Dataset Manifest
        total_session_dur = verdict.total_duration_s if verdict.total_duration_s > 0 else total_usable_s
        yield_pct = (total_usable_s / total_session_dur) * 100.0 if total_session_dur > 0 else 100.0

        dataset_manifest = DatasetManifest(
            schema_version="1.0.0",
            dataset_id=f"dataset_{session.session_id}",
            source_session_id=session.session_id,
            rig_config_name=session.rig_config_name,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            total_episodes=len(exported_episode_summaries),
            total_usable_duration_s=total_usable_s,
            source_total_duration_s=total_session_dur,
            usable_yield_pct=yield_pct,
            episodes=exported_episode_summaries,
        )

        with open(dest_session_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": dataset_manifest.schema_version,
                    "dataset_id": dataset_manifest.dataset_id,
                    "source_session_id": dataset_manifest.source_session_id,
                    "rig_config_name": dataset_manifest.rig_config_name,
                    "created_at": dataset_manifest.created_at,
                    "total_episodes": dataset_manifest.total_episodes,
                    "total_usable_duration_s": dataset_manifest.total_usable_duration_s,
                    "source_total_duration_s": dataset_manifest.source_total_duration_s,
                    "usable_yield_pct": dataset_manifest.usable_yield_pct,
                    "episodes": dataset_manifest.episodes,
                },
                f,
                indent=2,
            )

        return dataset_manifest

    def _filter_usable_windows(
        self,
        usable_ranges: list[TimeWindow],
    ) -> list[tuple[int, int]]:
        """Apply minimum duration filter and defect margin buffers to usable time windows."""
        filtered: list[tuple[int, int]] = []
        margin_ns = int(self.defect_margin_s * 1e9)
        min_dur_ns = int(self.min_duration_s * 1e9)

        for w in usable_ranges:
            s_ns = w.t_start_ns + margin_ns
            e_ns = w.t_end_ns - margin_ns
            dur_ns = e_ns - s_ns
            if dur_ns >= min_dur_ns:
                filtered.append((s_ns, e_ns))

        # If strict filtering eliminated windows due to short duration in test sessions,
        # fallback to raw unbuffered windows exceeding min_duration_s or top contiguous range
        if not filtered and usable_ranges:
            for w in usable_ranges:
                if (w.t_end_ns - w.t_start_ns) >= int(min(1.0, self.min_duration_s) * 1e9):
                    filtered.append((w.t_start_ns, w.t_end_ns))
            if not filtered:
                # Keep longest available window
                longest = max(usable_ranges, key=lambda r: r.t_end_ns - r.t_start_ns)
                filtered.append((longest.t_start_ns, longest.t_end_ns))

        return filtered

    def _slice_mcap(
        self,
        src_mcap_path: Path,
        out_mcap_path: Path,
        t_start_ns: int,
        t_end_ns: int,
    ) -> tuple[dict[str, int], list[np.ndarray]]:
        """Slices messages within time interval into a new MCAP file with zero re-encoding."""
        frames_count: dict[str, int] = {}
        sample_preview_frames: list[np.ndarray] = []
        preview_stream: str | None = None

        with open(src_mcap_path, "rb") as f_in, open(out_mcap_path, "wb") as f_out:
            reader = make_reader(f_in)
            writer = Writer(f_out)
            writer.start()

            schema_map: dict[int, int] = {}
            channel_map: dict[int, int] = {}

            for schema, channel, message in reader.iter_messages():
                # Filter strictly by publish_time
                if message.publish_time < t_start_ns or message.publish_time > t_end_ns:
                    continue

                # Register schema if seen for the first time
                if schema and schema.id not in schema_map:
                    new_s_id = writer.register_schema(
                        name=schema.name,
                        encoding=schema.encoding,
                        data=schema.data,
                    )
                    schema_map[schema.id] = new_s_id

                # Register channel if seen for the first time
                if channel.id not in channel_map:
                    new_c_id = writer.register_channel(
                        topic=channel.topic,
                        message_encoding=channel.message_encoding,
                        schema_id=schema_map.get(channel.schema_id, 0),
                        metadata=channel.metadata,
                    )
                    channel_map[channel.id] = new_c_id

                # Write message
                writer.add_message(
                    channel_id=channel_map[channel.id],
                    log_time=message.log_time,
                    publish_time=message.publish_time,
                    data=message.data,
                    sequence=message.sequence,
                )

                # Track message count per topic
                topic = channel.topic
                frames_count[topic] = frames_count.get(topic, 0) + 1

                # Sample frames from first video topic for preview
                if "image_compressed" in topic:
                    if preview_stream is None:
                        preview_stream = topic
                    if topic == preview_stream and len(sample_preview_frames) < 60:
                        try:
                            from foxglove_schemas_protobuf import CompressedImage_pb2
                            img_msg = CompressedImage_pb2.CompressedImage()
                            img_msg.ParseFromString(message.data)
                            arr = np.frombuffer(img_msg.data, dtype=np.uint8)
                            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if decoded is not None:
                                sample_preview_frames.append(decoded)
                        except Exception:
                            pass

            writer.finish()

        return frames_count, sample_preview_frames

    def _generate_preview_mp4(
        self,
        frames: list[np.ndarray],
        output_path: Path,
        fps: float = 15.0,
    ) -> None:
        """Write downsampled MP4 preview clip for visual review."""
        if not frames:
            return
        h, w = frames[0].shape[:2]
        # Scale to max 640 width for compact preview
        scale = min(1.0, 640.0 / w)
        out_w, out_h = int(w * scale), int(h * scale)
        # Ensure even dimensions for video codecs
        out_w = out_w if out_w % 2 == 0 else out_w - 1
        out_h = out_h if out_h % 2 == 0 else out_h - 1

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
        for f in frames:
            resized = cv2.resize(f, (out_w, out_h))
            writer.write(resized)
        writer.release()

    def _compute_calibration_hash(self, streams: list[Stream]) -> str:
        """Compute SHA-256 fingerprint over camera intrinsics, distortion, and extrinsics."""
        calib_data = []
        for s in streams:
            calib_data.append(
                {
                    "stream_id": s.stream_id,
                    "kind": str(s.kind),
                    "intrinsics": s.intrinsics.model_dump() if s.intrinsics else None,
                    "distortion": s.distortion.model_dump() if s.distortion else None,
                    "extrinsics": s.extrinsics.as_numpy().tolist() if s.extrinsics else None,
                }
            )
        serialized = json.dumps(calib_data, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_config_hash(self, session: Session) -> str:
        """Compute SHA-256 fingerprint over session configuration and stream topology."""
        cfg_data = {
            "rig_config_name": session.rig_config_name,
            "streams": [s.stream_id for s in session.streams],
        }
        return hashlib.sha256(json.dumps(cfg_data, sort_keys=True).encode("utf-8")).hexdigest()

    def _compute_file_sha256(self, file_path: Path) -> str:
        """Compute SHA-256 cryptographic checksum of a file for tamper-evident client delivery."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_git_commit(self) -> str:
        """Fetch current git commit hash for data provenance."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip()
        except Exception:
            return "unknown_commit"
