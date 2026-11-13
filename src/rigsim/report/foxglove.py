"""Foxglove event and log annotation exporter for MCAP recordings."""

import shutil
import tempfile
from pathlib import Path

import google.protobuf.timestamp_pb2 as pb_ts
from foxglove_schemas_protobuf import Event_pb2, Log_pb2
from mcap.reader import make_reader
from mcap.writer import Writer

from rigsim.config.qa import QAResult, QAVerdict, SessionVerdict


def _ns_to_pb_timestamp(timestamp_ns: int) -> pb_ts.Timestamp:
    ts = pb_ts.Timestamp()
    ts.seconds = timestamp_ns // 1_000_000_000
    ts.nanos = timestamp_ns % 1_000_000_000
    return ts


def export_foxglove_annotations(
    mcap_path: Path,
    qa_results: list[QAResult],
    session_verdict: SessionVerdict | None = None,
    output_mcap_path: Path | None = None,
) -> Path:
    """Annotate an existing MCAP file with Foxglove QA event spans and log diagnostics.

    Writes to `/qa/events` (foxglove.Event) for visual timeline highlighting,
    and `/qa/logs` (foxglove.Log) for interactive log filtering in Foxglove Studio.
    """
    mcap_path = Path(mcap_path)
    in_place = output_mcap_path is None or output_mcap_path == mcap_path
    target_out = mcap_path if in_place else Path(output_mcap_path)  # type: ignore[arg-type]

    # Create temporary file in the same directory for atomic rename
    temp_dir = target_out.parent
    temp_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".mcap", delete=False) as tmp_f:
        temp_path = Path(tmp_f.name)

    try:
        with open(mcap_path, "rb") as f_in, open(temp_path, "wb") as f_out:
            reader = make_reader(f_in)
            writer = Writer(f_out)
            writer.start()

            # 1. Map and copy existing schemas
            schema_map: dict[int, int] = {}
            for schema_id, schema in reader.get_summary().schemas.items():
                new_schema_id = writer.register_schema(
                    name=schema.name,
                    encoding=schema.encoding,
                    data=schema.data,
                )
                schema_map[schema_id] = new_schema_id

            # 2. Map and copy existing channels
            channel_map: dict[int, int] = {}
            for channel_id, channel in reader.get_summary().channels.items():
                new_schema_id = schema_map.get(channel.schema_id, 0)
                new_channel_id = writer.register_channel(
                    topic=channel.topic,
                    message_encoding=channel.message_encoding,
                    schema_id=new_schema_id,
                    metadata=dict(channel.metadata),
                )
                channel_map[channel_id] = new_channel_id

            # 3. Register Foxglove Event and Log schemas
            event_schema_id = writer.register_schema(
                name=Event_pb2.Event.DESCRIPTOR.full_name,
                encoding="protobuf",
                data=Event_pb2.Event.DESCRIPTOR.file.serialized_pb,
            )
            log_schema_id = writer.register_schema(
                name=Log_pb2.Log.DESCRIPTOR.full_name,
                encoding="protobuf",
                data=Log_pb2.Log.DESCRIPTOR.file.serialized_pb,
            )

            # 4. Register QA channels
            qa_events_channel_id = writer.register_channel(
                topic="/qa/events",
                message_encoding="protobuf",
                schema_id=event_schema_id,
                metadata={"description": "RigSim Automated Data QA Flagged Defect Events"},
            )
            qa_logs_channel_id = writer.register_channel(
                topic="/qa/logs",
                message_encoding="protobuf",
                schema_id=log_schema_id,
                metadata={"description": "RigSim Automated Data QA Diagnostics and Explanations"},
            )

            # 5. Copy all existing messages
            first_ts = None
            last_ts = None
            for schema, channel, message in reader.iter_messages():
                if first_ts is None or message.publish_time < first_ts:
                    first_ts = message.publish_time
                if last_ts is None or message.publish_time > last_ts:
                    last_ts = message.publish_time

                writer.add_message(
                    channel_id=channel_map[channel.id],
                    log_time=message.log_time,
                    publish_time=message.publish_time,
                    data=message.data,
                    sequence=message.sequence,
                )

            session_t_start = first_ts or 1_700_000_000_000_000_000
            session_t_end = last_ts or (session_t_start + 1_000_000_000)

            # 6. Write QA flagged window events and logs
            for r in qa_results:
                if r.verdict == QAVerdict.PASS:
                    continue

                level = (
                    Log_pb2.Log.Level.ERROR
                    if r.verdict == QAVerdict.FAIL
                    else Log_pb2.Log.Level.WARNING
                )

                if r.flagged_windows:
                    for w in r.flagged_windows:
                        w_start = (
                            w.t_start_ns
                            if hasattr(w, "t_start_ns")
                            else w.get("t_start_ns", session_t_start)
                        )
                        w_end = (
                            w.t_end_ns
                            if hasattr(w, "t_end_ns")
                            else w.get("t_end_ns", w_start + 50_000_000)
                        )

                        # Write Event
                        event = Event_pb2.Event()
                        event.start_time.CopyFrom(_ns_to_pb_timestamp(w_start))
                        event.end_time.CopyFrom(_ns_to_pb_timestamp(w_end))

                        m1 = event.metadata.add()
                        m1.key = "check_id"
                        m1.value = str(r.check_id)

                        m2 = event.metadata.add()
                        m2.key = "stream_id"
                        m2.value = str(r.stream_id)

                        m3 = event.metadata.add()
                        m3.key = "verdict"
                        m3.value = str(r.verdict.value)

                        m4 = event.metadata.add()
                        m4.key = "metric_value"
                        m4.value = f"{r.value:.4f}"

                        m5 = event.metadata.add()
                        m5.key = "threshold"
                        m5.value = f"{r.threshold:.4f}"

                        m6 = event.metadata.add()
                        m6.key = "explanation"
                        m6.value = str(r.explanation)

                        writer.add_message(
                            channel_id=qa_events_channel_id,
                            log_time=w_start,
                            publish_time=w_start,
                            data=event.SerializeToString(),
                        )

                        # Write Log message
                        log = Log_pb2.Log()
                        log.timestamp.CopyFrom(_ns_to_pb_timestamp(w_start))
                        log.level = level
                        log.name = "RigSimQA"
                        log.message = f"[{r.check_id}] ({r.stream_id}): {r.explanation}"

                        writer.add_message(
                            channel_id=qa_logs_channel_id,
                            log_time=w_start,
                            publish_time=w_start,
                            data=log.SerializeToString(),
                        )
                else:
                    # Check failed/warned globally without discrete frame windows
                    log = Log_pb2.Log()
                    log.timestamp.CopyFrom(_ns_to_pb_timestamp(session_t_start))
                    log.level = level
                    log.name = "RigSimQA"
                    log.message = f"[{r.check_id}] ({r.stream_id}): {r.explanation}"

                    writer.add_message(
                        channel_id=qa_logs_channel_id,
                        log_time=session_t_start,
                        publish_time=session_t_start,
                        data=log.SerializeToString(),
                    )

            # 7. Write overall verdict summary log
            if session_verdict:
                v_level = (
                    Log_pb2.Log.Level.ERROR
                    if session_verdict.overall_verdict == QAVerdict.FAIL
                    else (
                        Log_pb2.Log.Level.WARNING
                        if session_verdict.overall_verdict == QAVerdict.WARN
                        else Log_pb2.Log.Level.INFO
                    )
                )

                summary_log = Log_pb2.Log()
                summary_log.timestamp.CopyFrom(_ns_to_pb_timestamp(session_t_start))
                summary_log.level = v_level
                summary_log.name = "RigSimQA.Summary"
                summary_log.message = (
                    f"Overall Verdict: {session_verdict.overall_verdict.value} | "
                    f"Data Yield: {session_verdict.usable_percentage:.1f}% "
                    f"({session_verdict.usable_duration_s:.2f}s usable) | "
                    f"Summary: {session_verdict.summary_sentence}"
                )

                writer.add_message(
                    channel_id=qa_logs_channel_id,
                    log_time=session_t_start,
                    publish_time=session_t_start,
                    data=summary_log.SerializeToString(),
                )

            writer.finish()

        # Atomic replacement
        shutil.move(str(temp_path), str(target_out))
        return target_out

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
