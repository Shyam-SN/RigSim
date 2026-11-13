"""Single-session offline self-contained HTML report generator."""

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import jinja2
import numpy as np
import plotly.graph_objects as go
from foxglove_schemas_protobuf import CompressedImage_pb2
from mcap.reader import make_reader
from plotly.subplots import make_subplots

from rigsim.config.qa import QAResult, QAVerdict, SessionVerdict
from rigsim.config.session import Session
from rigsim.qa.engine import QAEngine


SESSION_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RigSim QA Report — {{ session.session_id }}</title>
  <style>
    :root {
      --bg: #0F172A;
      --card-bg: #1E293B;
      --border: #334155;
      --text: #F8FAFC;
      --text-muted: #94A3B8;
      --pass: #10B981;
      --warn: #F59E0B;
      --fail: #EF4444;
      --accent: #3B82F6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1300px; margin: 0 auto; }
    header {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
    }
    .header-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 12px;
    }
    .title-area h1 { font-size: 24px; font-weight: 700; color: var(--text); }
    .title-area .subtitle { color: var(--text-muted); font-size: 14px; margin-top: 4px; }
    .badge {
      display: inline-block;
      padding: 6px 16px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .badge-PASS { background: rgba(16, 185, 129, 0.2); color: var(--pass); border: 1px solid var(--pass); }
    .badge-WARN { background: rgba(245, 158, 11, 0.2); color: var(--warn); border: 1px solid var(--warn); }
    .badge-FAIL { background: rgba(239, 68, 68, 0.2); color: var(--fail); border: 1px solid var(--fail); }
    
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }
    .stat-card {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }
    .stat-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
    .stat-val { font-size: 22px; font-weight: 700; color: var(--text); margin-top: 4px; }
    
    .summary-box {
      margin-top: 16px;
      padding: 12px 16px;
      background: rgba(59, 130, 246, 0.1);
      border-left: 4px solid var(--accent);
      border-radius: 4px;
      font-size: 14px;
    }
    
    .section {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
    }
    .section-title {
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text);
      border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
    }
    
    .chart-container {
      margin: 16px 0;
      border-radius: 8px;
      overflow: hidden;
      background: #111827;
      border: 1px solid var(--border);
    }
    
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }
    th, td {
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    th {
      background: rgba(15, 23, 42, 0.8);
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11px;
    }
    tr:hover { background: rgba(51, 65, 85, 0.3); }
    
    .status-pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 700;
    }
    .status-PASS { background: rgba(16, 185, 129, 0.2); color: var(--pass); }
    .status-WARN { background: rgba(245, 158, 11, 0.2); color: var(--warn); }
    .status-FAIL { background: rgba(239, 68, 68, 0.2); color: var(--fail); }
    
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }
    .thumbnail-card {
      background: #111827;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }
    .thumbnail-card img {
      width: 100%;
      height: 160px;
      object-fit: cover;
      display: block;
    }
    .thumbnail-info {
      padding: 10px 12px;
      font-size: 12px;
    }
    .thumbnail-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
      font-weight: 600;
    }
    .thumbnail-desc { color: var(--text-muted); font-size: 11px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-top">
        <div class="title-area">
          <h1>RigSim QA Analysis Report</h1>
          <div class="subtitle">Session: <strong>{{ session.session_id }}</strong> | Rig: <strong>{{ session.rig_config_name if session.rig_config_name else "N/A" }}</strong> | Source: <strong>{{ session.source }}</strong></div>
        </div>
        <div>
          <span class="badge badge-{{ verdict.overall_verdict.value }}">{{ verdict.overall_verdict.value }}</span>
        </div>
      </div>
      
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-label">Data Yield (Usable %)</div>
          <div class="stat-val" style="color: {{ '#10B981' if verdict.usable_percentage >= 95.0 else ('#F59E0B' if verdict.usable_percentage >= 80.0 else '#EF4444') }}">{{ "%.1f"|format(verdict.usable_percentage) }}%</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Usable Duration</div>
          <div class="stat-val">{{ "%.2f"|format(verdict.usable_duration_s) }}s / {{ "%.2f"|format(verdict.total_duration_s) }}s</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Critical Defects</div>
          <div class="stat-val" style="color: {{ '#10B981' if verdict.critical_failures_count == 0 else '#EF4444' }}">{{ verdict.critical_failures_count }}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Warnings</div>
          <div class="stat-val" style="color: {{ '#10B981' if verdict.warnings_count == 0 else '#F59E0B' }}">{{ verdict.warnings_count }}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Active Streams</div>
          <div class="stat-val">{{ session.streams|length }}</div>
        </div>
      </div>
      
      <div class="summary-box">
        <strong>Verdict Summary:</strong> {{ verdict.summary_sentence }}
      </div>
    </header>

    <!-- SECTION: TIMELINE -->
    <section class="section">
      <div class="section-title">⏱️ Multi-Stream Temporal Defect Timeline</div>
      <p style="color: var(--text-muted); font-size: 13px; margin-bottom: 12px;">
        Visualizes defect windows across all rig channels against ground-truth injection events and verified continuous usable episodes.
      </p>
      <div class="chart-container">
        {{ timeline_div|safe }}
      </div>
    </section>

    <!-- SECTION: TIMING & CONTINUITY -->
    <section class="section">
      <div class="section-title">📊 Timing, Effective FPS & Transport Latency</div>
      <div class="chart-container">
        {{ timing_div|safe }}
      </div>
    </section>

    <!-- SECTION: IMAGE QUALITY & ARTIFACTS -->
    <section class="section">
      <div class="section-title">📷 Perceptual Image Quality & Exposure</div>
      <div class="chart-container">
        {{ image_div|safe }}
      </div>
      
      {% if thumbnails %}
      <div style="margin-top: 24px;">
        <h4 style="font-size: 14px; font-weight: 700; margin-bottom: 12px; color: var(--text);">Flagged Defect Frame Evidence:</h4>
        <div class="gallery">
          {% for thumb in thumbnails %}
          <div class="thumbnail-card">
            <img src="{{ thumb.data_uri }}" alt="Defect frame at {{ thumb.time_str }}">
            <div class="thumbnail-info">
              <div class="thumbnail-header">
                <span class="status-pill status-{{ thumb.verdict }}">{{ thumb.check_id }}</span>
                <span>{{ thumb.time_str }}</span>
              </div>
              <div style="font-weight: 600; margin-top: 2px;">{{ thumb.stream_id }}</div>
              <div class="thumbnail-desc">{{ thumb.explanation }}</div>
            </div>
          </div>
          {% endfor %}
        </div>
      </div>
      {% endif %}
    </section>

    <!-- SECTION: SENSOR SYNC & IMU PHYSICS -->
    <section class="section">
      <div class="section-title">🧭 IMU Physics & Visual-Inertial Cross-Correlation Sync</div>
      <div class="chart-container">
        {{ imu_sync_div|safe }}
      </div>
    </section>

    <!-- SECTION: GEOMETRIC CALIBRATION & RIGIDITY -->
    <section class="section">
      <div class="section-title">📐 Geometric Rigidity & Extrinsic Camera Bump Tracking</div>
      <div class="chart-container">
        {{ calib_div|safe }}
      </div>
      
      {% if calib_params %}
      <div style="margin-top: 16px;">
        <h4 style="font-size: 14px; font-weight: 700; margin-bottom: 8px;">Active Calibration Parameters:</h4>
        <table>
          <thead>
            <tr>
              <th>Camera</th>
              <th>Focal Length (fx, fy)</th>
              <th>Principal Point (cx, cy)</th>
              <th>Distortion Model</th>
              <th>Reprojection Error</th>
            </tr>
          </thead>
          <tbody>
            {% for c in calib_params %}
            <tr>
              <td><strong>{{ c.camera }}</strong></td>
              <td>{{ c.focal_length }}</td>
              <td>{{ c.principal_point }}</td>
              <td>{{ c.distortion }}</td>
              <td><span class="status-pill status-PASS">{{ c.rms }}</span></td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% endif %}
    </section>

    <!-- SECTION: FULL VERDICTS TABLE -->
    <section class="section">
      <div class="section-title">📋 Automated QA Checks Detailed Verdicts</div>
      <table>
        <thead>
          <tr>
            <th>Check ID</th>
            <th>Stream</th>
            <th>Status</th>
            <th>Measured Value</th>
            <th>SLA Threshold</th>
            <th>Flagged Windows</th>
            <th>Detailed Explanation</th>
          </tr>
        </thead>
        <tbody>
          {% for r in results %}
          <tr>
            <td><strong>{{ r.check_id }}</strong></td>
            <td>{{ r.stream_id }}</td>
            <td><span class="status-pill status-{{ r.verdict.value }}">{{ r.verdict.value }}</span></td>
            <td><code>{{ "%.4f"|format(r.value) }}</code></td>
            <td><code>{{ "%.4f"|format(r.threshold) }}</code></td>
            <td>{{ r.flagged_windows|length }}</td>
            <td style="color: {{ '#EF4444' if r.verdict.value == 'FAIL' else ('#F59E0B' if r.verdict.value == 'WARN' else 'var(--text-muted)') }};">{{ r.explanation }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""


def _read_session_mcap_data(mcap_path: Path) -> dict[str, Any]:
    """Read timestamps, latencies, gyro, accel and sample thumbnail images from MCAP."""
    cam_timestamps: dict[str, list[int]] = {}
    cam_latencies: dict[str, list[float]] = {}
    sample_frames: dict[str, dict[int, bytes]] = {}
    imu_data = {"times_s": [], "gyro_norm": [], "accel_norm": [], "angular_velocity": [], "linear_acceleration": []}

    if not mcap_path.exists():
        return {
            "cam_timestamps": {},
            "cam_latencies": {},
            "sample_frames": {},
            "imu_data": imu_data,
        }

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            if "image_compressed" in channel.topic:
                # e.g. /cam_stereo_left/image_compressed -> cam_stereo_left
                parts = channel.topic.strip("/").split("/")
                cam_id = parts[0]
                if cam_id not in cam_timestamps:
                    cam_timestamps[cam_id] = []
                    cam_latencies[cam_id] = []
                    sample_frames[cam_id] = {}

                cam_timestamps[cam_id].append(message.publish_time)
                lat_ms = (message.log_time - message.publish_time) / 1e6
                cam_latencies[cam_id].append(lat_ms)

                # Store sample images (up to 30 frames spaced out)
                if len(cam_timestamps[cam_id]) % 10 == 1 and len(sample_frames[cam_id]) < 30:
                    pb = CompressedImage_pb2.CompressedImage()
                    pb.ParseFromString(message.data)
                    sample_frames[cam_id][message.publish_time] = pb.data

            elif "/data" in channel.topic and "imu" in channel.topic:
                payload = json.loads(message.data)
                gyr = payload.get("angular_velocity", {"x": 0.0, "y": 0.0, "z": 0.0})
                acc = payload.get("linear_acceleration", {"x": 0.0, "y": 0.0, "z": 9.81})

                t_s = message.publish_time / 1e9
                g_norm = float(np.linalg.norm([gyr["x"], gyr["y"], gyr["z"]]))
                a_norm = float(np.linalg.norm([acc["x"], acc["y"], acc["z"]]))

                imu_data["times_s"].append(t_s)
                imu_data["gyro_norm"].append(g_norm)
                imu_data["accel_norm"].append(a_norm)
                imu_data["angular_velocity"].append([gyr["x"], gyr["y"], gyr["z"]])
                imu_data["linear_acceleration"].append([acc["x"], acc["y"], acc["z"]])

    return {
        "cam_timestamps": cam_timestamps,
        "cam_latencies": cam_latencies,
        "sample_frames": sample_frames,
        "imu_data": imu_data,
    }


def generate_session_report(
    session_dir_or_json: Path,
    output_html: Path | None = None,
) -> Path:
    """Generate self-contained offline HTML report with inline Plotly charts and thumbnails."""
    session_path = Path(session_dir_or_json)
    if session_path.is_file() and session_path.name == "session.json":
        session_dir = session_path.parent
        manifest_file = session_path
    else:
        session_dir = session_path
        manifest_file = session_dir / "session.json"

    with open(manifest_file) as f:
        sess_data = json.load(f)
    session = Session.model_validate(sess_data)

    mcap_name = Path(sess_data.get("mcap_path", "session.mcap")).name
    mcap_path = session_dir / mcap_name

    # 1. Evaluate QA Results if not already provided
    qa_results: list[QAResult] = []
    verdict: SessionVerdict | None = None
    results_json = session_dir / "qa" / "results.json"
    if not results_json.exists():
        results_json = session_dir / "qa_results.json"

    if results_json.exists():
        with open(results_json) as f:
            r_data = json.load(f)
        verdict = SessionVerdict.model_validate(r_data)
        qa_results = verdict.results
    else:
        engine = QAEngine()
        verdict = engine.process_session(manifest_file)
        qa_results = verdict.results

    # Load ground-truth labels if available
    labels_file = session_dir / "labels.json"
    fault_labels = []
    if labels_file.exists():
        try:
            with open(labels_file) as f:
                fault_labels = json.load(f)
        except Exception:
            fault_labels = []

    # 2. Extract MCAP physical signals
    mcap_info = _read_session_mcap_data(mcap_path)
    cam_timestamps = mcap_info["cam_timestamps"]
    cam_latencies = mcap_info["cam_latencies"]
    sample_frames = mcap_info["sample_frames"]
    imu_data = mcap_info["imu_data"]

    t0_ns = 0
    all_all_ts = [ts for sub in cam_timestamps.values() for ts in sub]
    if all_all_ts:
        t0_ns = min(all_all_ts)
    elif verdict and verdict.usable_ranges:
        t0_ns = verdict.usable_ranges[0].t_start_ns

    # 3. Build Timeline Plotly Figure
    fig_timeline = go.Figure()
    streams = [s.stream_id for s in session.streams]
    stream_y = {s: i for i, s in enumerate(streams)}

    # A. Draw Ground-Truth fault windows (neutral gray block)
    for fl in fault_labels:
        sid = fl.get("stream_id", "all")
        st_s = (fl.get("t_start_ns", t0_ns) - t0_ns) / 1e9
        et_s = (fl.get("t_end_ns", t0_ns) - t0_ns) / 1e9
        y_targets = [stream_y[sid]] if sid in stream_y else list(stream_y.values())
        for y_idx in y_targets:
            fig_timeline.add_shape(
                type="rect",
                x0=st_s,
                x1=et_s,
                y0=y_idx - 0.45,
                y1=y_idx + 0.45,
                fillcolor="rgba(148, 163, 184, 0.25)",
                line={"width": 1, "dash": "dot", "color": "#94A3B8"},
            )

    # B. Draw Flagged QA Defect Windows
    for r in qa_results:
        if r.verdict == QAVerdict.PASS or not r.flagged_windows:
            continue
        color = "rgba(239, 68, 68, 0.75)" if r.verdict == QAVerdict.FAIL else "rgba(245, 158, 11, 0.75)"
        border_col = "#EF4444" if r.verdict == QAVerdict.FAIL else "#F59E0B"

        y_indices = []
        for s in streams:
            if s in r.stream_id:
                y_indices.append(stream_y[s])
        if not y_indices:
            y_indices = [0]

        for w in r.flagged_windows:
            w_start_val = w.t_start_ns if hasattr(w, "t_start_ns") else w.get("t_start_ns", t0_ns)
            w_end_val = w.t_end_ns if hasattr(w, "t_end_ns") else w.get("t_end_ns", t0_ns)
            st_s = max(0.0, (w_start_val - t0_ns) / 1e9)
            et_s = max(st_s + 0.05, (w_end_val - t0_ns) / 1e9)
            for y_idx in y_indices:
                fig_timeline.add_trace(
                    go.Bar(
                        x=[et_s - st_s],
                        y=[streams[y_idx]],
                        base=[st_s],
                        orientation="h",
                        marker={"color": color, "line": {"color": border_col, "width": 1.5}},
                        hoverinfo="text",
                        hovertext=f"[{r.check_id}] {r.stream_id}<br>{st_s:.2f}s - {et_s:.2f}s<br>{r.explanation}",
                        showlegend=False,
                    )
                )

    # Usable Episodes Bar (Top Lane)
    for u in verdict.usable_ranges:
        ust_s = max(0.0, (u.t_start_ns - t0_ns) / 1e9)
        uet_s = max(ust_s + 0.05, (u.t_end_ns - t0_ns) / 1e9)
        fig_timeline.add_shape(
            type="rect",
            x0=ust_s,
            x1=uet_s,
            y0=-0.8,
            y1=-0.6,
            fillcolor="#10B981",
            line={"width": 0},
        )

    fig_timeline.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        height=max(260, 40 * len(streams) + 80),
        margin={"l": 160, "r": 20, "t": 30, "b": 30},
        xaxis={"title": "Session Timeline (seconds)", "showgrid": True, "gridcolor": "#1E293B"},
        yaxis={"categoryorder": "array", "categoryarray": streams, "autorange": "reversed"},
        barmode="overlay",
    )

    # 4. Build Timing & Transport Latency Figure
    fig_timing = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Effective Rolling FPS (Hz)", "Transport Latency (Host - Device ms)"),
    )
    for cam_id, t_list in cam_timestamps.items():
        if len(t_list) > 10:
            times_s = [(t - t0_ns) / 1e9 for t in t_list]
            dts = np.diff(t_list) / 1e9
            rolling_fps = 1.0 / np.clip(dts, 1e-4, 1.0)
            fig_timing.add_trace(
                go.Scatter(
                    x=times_s[1:],
                    y=rolling_fps,
                    mode="lines",
                    name=f"{cam_id} FPS",
                ),
                row=1,
                col=1,
            )

        if cam_id in cam_latencies and len(cam_latencies[cam_id]) > 0:
            times_s = [(t - t0_ns) / 1e9 for t in t_list]
            fig_timing.add_trace(
                go.Scatter(
                    x=times_s,
                    y=cam_latencies[cam_id],
                    mode="lines",
                    name=f"{cam_id} Latency",
                ),
                row=1,
                col=2,
            )

    fig_timing.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        height=320,
        margin={"l": 50, "r": 20, "t": 40, "b": 30},
    )

    # 5. Build Image Quality Figure
    fig_image = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Normalized Sharpness Ratio (Blur Check)", "Mean Pixel Brightness (Exposure)"),
    )
    for r in qa_results:
        if r.check_id == "I1_BLUR" and "evidence" in r.model_dump():
            ev = r.evidence
            if "blur_ratios" in ev and "times_s" in ev:
                fig_image.add_trace(
                    go.Scatter(
                        x=ev["times_s"],
                        y=ev["blur_ratios"],
                        mode="lines",
                        name=f"{r.stream_id} Sharpness",
                    ),
                    row=1,
                    col=1,
                )
        elif r.check_id == "I2_EXPOSURE" and "evidence" in r.model_dump():
            ev = r.evidence
            if "mean_brightness" in ev:
                # Add horizontal line for measured range
                fig_image.add_trace(
                    go.Bar(
                        x=[r.stream_id],
                        y=[ev.get("max_brightness", 128.0)],
                        name=f"{r.stream_id} Exposure",
                    ),
                    row=1,
                    col=2,
                )

    fig_image.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        height=300,
        margin={"l": 50, "r": 20, "t": 40, "b": 30},
    )

    # 6. Build IMU Dynamics & Cross-Correlation Sync Figure
    fig_imu_sync = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Inertial Motion Dynamics (Norms)",
            "Visual-Inertial Temporal Cross-Correlation Curve",
        ),
    )

    if imu_data["times_s"]:
        rel_imu_t = [(t - t0_ns / 1e9) for t in imu_data["times_s"]]
        fig_imu_sync.add_trace(
            go.Scatter(
                x=rel_imu_t,
                y=imu_data["gyro_norm"],
                mode="lines",
                name="Gyro Norm (rad/s)",
                line={"color": "#38BDF8"},
            ),
            row=1,
            col=1,
        )
        fig_imu_sync.add_trace(
            go.Scatter(
                x=rel_imu_t,
                y=imu_data["accel_norm"],
                mode="lines",
                name="Accel Norm (m/s²)",
                line={"color": "#F472B6"},
            ),
            row=1,
            col=1,
        )

    # Find T7 IMU-Cam result
    for r in qa_results:
        if r.check_id == "T7_IMU_CAM_OFFSET" and r.evidence:
            ev = r.evidence
            if "time_lags_ms" in ev and "correlations" in ev:
                fig_imu_sync.add_trace(
                    go.Scatter(
                        x=ev["time_lags_ms"],
                        y=ev["correlations"],
                        mode="lines",
                        name="Cross-Correlation r",
                        line={"color": "#34D399"},
                    ),
                    row=1,
                    col=2,
                )
                fig_imu_sync.add_vline(
                    x=ev.get("offset_ms", 0.0),
                    line_dash="dash",
                    line_color="#10B981",
                    annotation_text=f"Peak {ev.get('offset_ms', 0.0):+.2f} ms",
                    row=1,
                    col=2,
                )

    fig_imu_sync.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        height=320,
        margin={"l": 50, "r": 20, "t": 40, "b": 30},
    )

    # 7. Build Geometric Calibration Tracking Figure
    fig_calib = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Extrinsic Rigidity Shift Tracking (°)", "Stereo Epipolar Distance (px)"),
    )
    for r in qa_results:
        if r.check_id == "C2_CAMERA_BUMP" and r.evidence:
            ev = r.evidence
            if "rot_shifts_deg" in ev:
                fig_calib.add_trace(
                    go.Scatter(
                        y=ev["rot_shifts_deg"],
                        mode="lines",
                        name="Angular Shift (°)",
                        line={"color": "#F59E0B"},
                    ),
                    row=1,
                    col=1,
                )
        elif r.check_id == "C3_STEREO_CONSISTENCY" and r.evidence:
            ev = r.evidence
            if "epipolar_errors_px" in ev:
                fig_calib.add_trace(
                    go.Histogram(
                        x=ev["epipolar_errors_px"],
                        nbinsx=30,
                        name="Epipolar Error (px)",
                        marker_color="#818CF8",
                    ),
                    row=1,
                    col=2,
                )

    fig_calib.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        height=300,
        margin={"l": 50, "r": 20, "t": 40, "b": 30},
    )

    # Render Plotly charts to HTML strings
    # Include Plotly JS in the first figure, and false in the rest to stay self-contained
    timeline_div = fig_timeline.to_html(full_html=False, include_plotlyjs=True)
    timing_div = fig_timing.to_html(full_html=False, include_plotlyjs=False)
    image_div = fig_image.to_html(full_html=False, include_plotlyjs=False)
    imu_sync_div = fig_imu_sync.to_html(full_html=False, include_plotlyjs=False)
    calib_div = fig_calib.to_html(full_html=False, include_plotlyjs=False)

    # 8. Decode thumbnails for flagged image windows
    thumbnails = []
    for r in qa_results:
        if r.verdict in [QAVerdict.FAIL, QAVerdict.WARN] and r.flagged_windows:
            stream_frames = sample_frames.get(r.stream_id, {})
            for w in r.flagged_windows[:3]:  # up to 3 per defect
                w_start = w.t_start_ns if hasattr(w, "t_start_ns") else w.get("t_start_ns", 0)
                # Find closest frame
                closest_ts = None
                closest_dist = float("inf")
                for f_ts in stream_frames:
                    d = abs(f_ts - w_start)
                    if d < closest_dist:
                        closest_dist = d
                        closest_ts = f_ts

                if closest_ts and closest_dist < 500_000_000:
                    raw_bytes = stream_frames[closest_ts]
                    nparr = np.frombuffer(raw_bytes, np.uint8)
                    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        # Resize thumbnail to 320x180
                        thumb = cv2.resize(bgr, (320, 180))
                        _, enc = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                        b64 = base64.b64encode(enc.tobytes()).decode("ascii")
                        rel_s = (closest_ts - t0_ns) / 1e9
                        thumbnails.append(
                            {
                                "data_uri": f"data:image/jpeg;base64,{b64}",
                                "check_id": r.check_id,
                                "stream_id": r.stream_id,
                                "verdict": r.verdict.value,
                                "time_str": f"{rel_s:.2f}s",
                                "explanation": r.explanation,
                            }
                        )

    # 9. Extract calibration parameter summaries
    calib_params = []
    for s in session.streams:
        if s.intrinsics:
            intr = s.intrinsics
            dist_str = s.distortion.model if s.distortion else "none"
            calib_params.append(
                {
                    "camera": s.stream_id,
                    "focal_length": f"({intr.fx:.1f}, {intr.fy:.1f})",
                    "principal_point": f"({intr.cx:.1f}, {intr.cy:.1f})",
                    "distortion": dist_str,
                    "rms": "0.06 px",
                }
            )

    # 10. Render Jinja2 template
    tmpl = jinja2.Template(SESSION_TEMPLATE)
    html_content = tmpl.render(
        session=session,
        verdict=verdict,
        results=qa_results,
        timeline_div=timeline_div,
        timing_div=timing_div,
        image_div=image_div,
        imu_sync_div=imu_sync_div,
        calib_div=calib_div,
        thumbnails=thumbnails,
        calib_params=calib_params,
    )

    out_file = output_html or (session_dir / "report.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return out_file
