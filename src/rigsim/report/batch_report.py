"""Multi-session fleet summary HTML report generator for operations and QA leadership."""

import json
from pathlib import Path
from typing import Any

import jinja2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rigsim.config.qa import QAVerdict, SessionVerdict
from rigsim.config.session import Session
from rigsim.qa.engine import QAEngine


BATCH_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RigSim Fleet QA Operations Summary — {{ batch_name }}</title>
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
    .container { max-width: 1350px; margin: 0 auto; }
    header {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
    }
    .header-title h1 { font-size: 26px; font-weight: 700; color: var(--text); }
    .header-title .subtitle { color: var(--text-muted); font-size: 14px; margin-top: 4px; }
    
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-top: 20px;
    }
    .stat-card {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }
    .stat-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
    .stat-val { font-size: 24px; font-weight: 700; color: var(--text); margin-top: 4px; }
    
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
    
    a { color: var(--accent); text-decoration: none; font-weight: 600; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-title">
        <h1>RigSim Fleet QA Operations Summary</h1>
        <div class="subtitle">Multi-Session Fleet Batch Analysis | Target Directory: <strong>{{ batch_dir }}</strong></div>
      </div>
      
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-label">Total Sessions</div>
          <div class="stat-val">{{ summary.total_sessions }}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Fleet Pass Rate</div>
          <div class="stat-val" style="color: {{ '#10B981' if summary.pass_rate >= 80.0 else '#EF4444' }}">{{ "%.1f"|format(summary.pass_rate) }}%</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Fleet Data Yield</div>
          <div class="stat-val" style="color: {{ '#10B981' if summary.yield_rate >= 85.0 else '#F59E0B' }}">{{ "%.1f"|format(summary.yield_rate) }}%</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Recorded / Usable Time</div>
          <div class="stat-val">{{ "%.2f"|format(summary.total_usable_duration_h) }}h / {{ "%.2f"|format(summary.total_recorded_duration_h) }}h</div>
        </div>
      </div>
    </header>

    <!-- SECTION: CHARTS -->
    <section class="section">
      <div class="section-title">📊 Fleet Operations & Failure Distribution</div>
      <div class="chart-container">
        {{ fleet_charts_div|safe }}
      </div>
    </section>

    <!-- SECTION: PER-RIG BREAKDOWN -->
    <section class="section">
      <div class="section-title">🤖 Rig Breakdown & Quality Comparison</div>
      <table>
        <thead>
          <tr>
            <th>Rig Identifier</th>
            <th>Sessions Captured</th>
            <th>Pass Rate (%)</th>
            <th>Data Yield (%)</th>
            <th>Total Usable (hours)</th>
            <th>Dominant Failure Cause</th>
          </tr>
        </thead>
        <tbody>
          {% for r in rig_breakdown %}
          <tr>
            <td><strong>{{ r.rig_name }}</strong></td>
            <td>{{ r.sessions_count }}</td>
            <td><span class="status-pill status-{{ 'PASS' if r.pass_rate >= 80.0 else 'FAIL' }}">{{ "%.1f"|format(r.pass_rate) }}%</span></td>
            <td><strong>{{ "%.1f"|format(r.yield_rate) }}%</strong></td>
            <td>{{ "%.2f"|format(r.usable_h) }}h</td>
            <td style="color: var(--text-muted);">{{ r.dominant_defect }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

    <!-- SECTION: ALL SESSIONS TABLE -->
    <section class="section">
      <div class="section-title">📁 Processed Sessions Listing</div>
      <table>
        <thead>
          <tr>
            <th>Session ID</th>
            <th>Source</th>
            <th>Rig</th>
            <th>Operator</th>
            <th>Verdict</th>
            <th>Data Yield</th>
            <th>Duration</th>
            <th>Primary QA Issue</th>
            <th>Report</th>
          </tr>
        </thead>
        <tbody>
          {% for s in session_rows %}
          <tr>
            <td><strong>{{ s.session_id }}</strong></td>
            <td><code>{{ s.source }}</code></td>
            <td>{{ s.rig_name }}</td>
            <td>{{ s.operator }}</td>
            <td><span class="status-pill status-{{ s.verdict }}">{{ s.verdict }}</span></td>
            <td><strong>{{ "%.1f"|format(s.yield_pct) }}%</strong></td>
            <td>{{ "%.1f"|format(s.duration_s) }}s</td>
            <td style="color: {{ '#EF4444' if s.verdict == 'FAIL' else 'var(--text-muted)' }};">{{ s.top_issue }}</td>
            <td>
              {% if s.report_exists %}
              <a href="{{ s.report_rel_path }}" target="_blank">View Report ↗</a>
              {% else %}
              <span style="color: var(--text-muted);">No Report</span>
              {% endif %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""


def generate_batch_report(
    batch_dir: Path,
    output_html: Path | None = None,
) -> Path:
    """Generate multi-session fleet summary HTML report for operations leadership."""
    batch_path = Path(batch_dir)
    session_dirs = []

    # Find all session folders with session.json
    for manifest_path in batch_path.glob("**/session.json"):
        session_dirs.append(manifest_path.parent)

    session_rows: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    verdict_counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    total_recorded_s = 0.0
    total_usable_s = 0.0

    rig_stats: dict[str, dict[str, Any]] = {}

    for s_dir in session_dirs:
        try:
            with open(s_dir / "session.json") as f:
                sess_dict = json.load(f)
            sess = Session.model_validate(sess_dict)

            # Load or compute verdict
            verdict: SessionVerdict | None = None
            qa_results = []
            res_json = s_dir / "qa" / "results.json"
            if not res_json.exists():
                res_json = s_dir / "qa_results.json"

            if res_json.exists():
                with open(res_json) as f:
                    r_data = json.load(f)
                verdict = SessionVerdict.model_validate(r_data)
                qa_results = [r.model_dump() for r in verdict.results]
            else:
                engine = QAEngine()
                verdict = engine.process_session(s_dir / "session.json")
                qa_results = [r.model_dump() for r in verdict.results]

            v_str = verdict.overall_verdict.value
            verdict_counts[v_str] = verdict_counts.get(v_str, 0) + 1

            total_recorded_s += verdict.total_duration_s
            total_usable_s += verdict.usable_duration_s

            # Rig stats
            r_name = sess.rig_config_name if hasattr(sess, "rig_config_name") else "default"
            if r_name not in rig_stats:
                rig_stats[r_name] = {
                    "sessions": 0,
                    "pass": 0,
                    "usable_s": 0.0,
                    "total_s": 0.0,
                    "defects": {},
                }
            rig_stats[r_name]["sessions"] += 1
            if v_str == "PASS":
                rig_stats[r_name]["pass"] += 1
            rig_stats[r_name]["usable_s"] += verdict.usable_duration_s
            rig_stats[r_name]["total_s"] += verdict.total_duration_s

            # Defect count
            primary_issue = "None (Clean)"
            for r in qa_results:
                cid = r.get("check_id", "")
                v = r.get("verdict", "")
                if v == "FAIL":
                    failure_counts[cid] = failure_counts.get(cid, 0) + 1
                    rig_stats[r_name]["defects"][cid] = (
                        rig_stats[r_name]["defects"].get(cid, 0) + 1
                    )
                    if primary_issue == "None (Clean)":
                        primary_issue = f"{cid}: {r.get('explanation', '')[:60]}..."

            # Report link
            report_file = s_dir / "report.html"
            rel_report = None
            if report_file.exists():
                try:
                    rel_report = str(report_file.relative_to(batch_path))
                except Exception:
                    rel_report = str(report_file)

            session_rows.append(
                {
                    "session_id": sess.session_id,
                    "source": sess.source.value if hasattr(sess.source, "value") else str(sess.source),
                    "rig_name": r_name,
                    "operator": sess.operator or "automation",
                    "verdict": v_str,
                    "yield_pct": verdict.usable_percentage,
                    "duration_s": verdict.total_duration_s,
                    "top_issue": primary_issue,
                    "report_exists": report_file.exists(),
                    "report_rel_path": rel_report,
                }
            )
        except Exception:
            continue

    total_sessions = len(session_rows)
    pass_rate = (verdict_counts["PASS"] / total_sessions * 100.0) if total_sessions > 0 else 0.0
    yield_rate = (total_usable_s / total_recorded_s * 100.0) if total_recorded_s > 0 else 0.0

    # Rig breakdown list
    rig_breakdown = []
    for r_name, r_data in rig_stats.items():
        sess_c = r_data["sessions"]
        p_rate = (r_data["pass"] / sess_c * 100.0) if sess_c > 0 else 0.0
        y_rate = (r_data["usable_s"] / r_data["total_s"] * 100.0) if r_data["total_s"] > 0 else 0.0
        top_def = "None"
        if r_data["defects"]:
            top_def = max(r_data["defects"].items(), key=lambda x: x[1])[0]
        rig_breakdown.append(
            {
                "rig_name": r_name,
                "sessions_count": sess_c,
                "pass_rate": p_rate,
                "yield_rate": y_rate,
                "usable_h": r_data["usable_s"] / 3600.0,
                "dominant_defect": top_def,
            }
        )

    # Plotly figures: Fleet Summary
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Fleet Session Verdicts", "Top QA Failure Root Causes"),
        specs=[[{"type": "domain"}, {"type": "xy"}]],
    )

    # Donut chart
    labels = ["PASS", "WARN", "FAIL"]
    values = [verdict_counts.get(k, 0) for k in labels]
    colors = ["#10B981", "#F59E0B", "#EF4444"]
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker={"colors": colors},
            hole=0.45,
            textinfo="label+percent",
        ),
        row=1,
        col=1,
    )

    # Top failures horizontal bar chart
    sorted_fails = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    if sorted_fails:
        fail_names = [f[0] for f in sorted_fails][::-1]
        fail_vals = [f[1] for f in sorted_fails][::-1]
        fig.add_trace(
            go.Bar(
                x=fail_vals,
                y=fail_names,
                orientation="h",
                marker_color="#EF4444",
                name="Failure Count",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        height=320,
        margin={"l": 140, "r": 20, "t": 40, "b": 30},
    )

    fleet_charts_div = fig.to_html(full_html=False, include_plotlyjs=True)

    summary_info = {
        "total_sessions": total_sessions,
        "pass_rate": pass_rate,
        "yield_rate": yield_rate,
        "total_recorded_duration_h": total_recorded_s / 3600.0,
        "total_usable_duration_h": total_usable_s / 3600.0,
    }

    tmpl = jinja2.Template(BATCH_TEMPLATE)
    html_content = tmpl.render(
        batch_name=batch_path.name,
        batch_dir=str(batch_path),
        summary=summary_info,
        fleet_charts_div=fleet_charts_div,
        rig_breakdown=rig_breakdown,
        session_rows=session_rows,
    )

    out_file = output_html or (batch_path / "summary_report.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return out_file
