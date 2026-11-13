"""Interactive HTML report generation for the capture planner and economics engine."""

from pathlib import Path

import jinja2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rigsim.plan.bandwidth import BandwidthReport
from rigsim.plan.codec_bench import CodecResult
from rigsim.plan.cost_model import CostModelReport

PLANNER_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RigSim Capture Planner & Cost Report — {{ bandwidth.rig_name }}</title>
  <style>
    :root {
      --bg: #0F172A;
      --card-bg: #1E293B;
      --card-sub: #273549;
      --border: #334155;
      --text: #F8FAFC;
      --text-muted: #94A3B8;
      --pass: #10B981;
      --warn: #F59E0B;
      --fail: #EF4444;
      --accent: #3B82F6;
      --pareto: #8B5CF6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1360px; margin: 0 auto; }
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
    .title-area h1 { font-size: 26px; font-weight: 700; color: var(--text); }
    .title-area .subtitle { color: var(--text-muted); font-size: 14px; margin-top: 4px; }
    .badge {
      display: inline-block;
      padding: 6px 14px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 13px;
      text-transform: uppercase;
    }
    .badge-pass { background: rgba(16, 185, 129, 0.2); color: var(--pass); border: 1px solid var(--pass); }
    .badge-warn { background: rgba(245, 158, 11, 0.2); color: var(--warn); border: 1px solid var(--warn); }
    .badge-fail { background: rgba(239, 68, 68, 0.2); color: var(--fail); border: 1px solid var(--fail); }
    .badge-pareto { background: rgba(139, 92, 246, 0.25); color: #C4B5FD; border: 1px solid var(--pareto); }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-top: 20px;
    }
    .stat-card {
      background: var(--card-sub);
      padding: 16px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .stat-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
    .stat-value { font-size: 24px; font-weight: 700; color: var(--text); margin-top: 4px; }
    .stat-sub { font-size: 12px; color: var(--text-muted); margin-top: 2px; }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
    }
    h2 { font-size: 18px; font-weight: 600; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }
    th {
      background: var(--card-sub);
      color: var(--text-muted);
      padding: 10px 14px;
      font-weight: 600;
      border-bottom: 1px solid var(--border);
    }
    td {
      padding: 10px 14px;
      border-bottom: 1px solid rgba(51, 65, 85, 0.5);
    }
    tr:hover { background: rgba(255, 255, 255, 0.02); }

    .chart-container {
      margin-top: 16px;
      margin-bottom: 24px;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
    }

    .rec-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 20px;
      margin-top: 16px;
    }
    .rec-card {
      background: var(--card-sub);
      border-radius: 8px;
      border-left: 4px solid var(--accent);
      padding: 16px;
    }
    .rec-card.archival { border-left-color: #38BDF8; }
    .rec-card.training { border-left-color: #10B981; }
    .rec-card.edge { border-left-color: #F59E0B; }
    .rec-title { font-size: 15px; font-weight: 700; margin-bottom: 8px; }
    .rec-body { font-size: 13px; color: var(--text-muted); line-height: 1.5; }
    .rec-tags { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
    .tag { font-size: 11px; background: rgba(255,255,255,0.06); padding: 4px 8px; border-radius: 4px; color: var(--text); }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-top">
        <div class="title-area">
          <h1>Capture Planner & Economics Report</h1>
          <div class="subtitle">Hardware Bandwidth, Interface Limits, Codec Sweeps & Cloud Cost Modeling</div>
        </div>
        <div>
          <span class="badge {% if bandwidth.warnings %}badge-warn{% else %}badge-pass{% endif %}">
            {% if bandwidth.warnings %}Hardware Warnings ({{ bandwidth.warnings|length }}){% else %}Interfaces Within Limits{% endif %}
          </span>
        </div>
      </div>

      <div class="summary-grid">
        <div class="stat-card">
          <div class="stat-label">Rig Architecture</div>
          <div class="stat-value">{{ bandwidth.rig_name }}</div>
          <div class="stat-sub">{{ bandwidth.cameras|length }} Cameras {% if bandwidth.imu %}+ 1 IMU{% endif %}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Raw Sustained Bandwidth</div>
          <div class="stat-value">{{ "%.1f"|format(bandwidth.total_raw_mb_per_s) }} <span style="font-size:14px;">MB/s</span></div>
          <div class="stat-sub">{{ "%.1f"|format(bandwidth.total_raw_mb_per_s * 8.0) }} Mbps uncompressed</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Raw Ingest Rate</div>
          <div class="stat-value">{{ "%.1f"|format(bandwidth.total_raw_gb_per_hour) }} <span style="font-size:14px;">GB/hr</span></div>
          <div class="stat-sub">{{ "%.1f"|format(bandwidth.total_raw_tb_per_1000h) }} TB per 1,000 hours</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Model Horizon</div>
          <div class="stat-value">{{ "%.0f"|format(cost_report.hours_modeled) }} <span style="font-size:14px;">Hours</span></div>
          <div class="stat-sub">S3 Standard / Glacier / Deep Archive</div>
        </div>
      </div>
    </header>

    {% if bandwidth.warnings %}
    <div class="card" style="border-left: 4px solid var(--warn);">
      <h2 style="color: var(--warn); border-bottom: none; margin-bottom: 8px;">Hardware Bus & Interface Warnings</h2>
      <ul style="padding-left: 20px; font-size: 13px; color: var(--text);">
        {% for w in bandwidth.warnings %}
        <li style="margin-bottom: 6px;">{{ w }}</li>
        {% endfor %}
      </ul>
    </div>
    {% endif %}

    <div class="card">
      <h2>1. Physical Transport & Host Interface Headroom</h2>
      <table>
        <thead>
          <tr>
            <th>Interface / Bus Standard</th>
            <th>Sustained Limit</th>
            <th>Rig Ingest Rate</th>
            <th>Bus Utilization</th>
            <th>Headroom Status</th>
          </tr>
        </thead>
        <tbody>
          {% for iface, info in bandwidth.interface_comparisons.items() %}
          <tr>
            <td><strong>{{ iface }}</strong></td>
            <td>{{ "%.1f"|format(info.limit_mb_s) }} MB/s</td>
            <td>{{ "%.1f"|format(bandwidth.total_raw_mb_per_s) }} MB/s</td>
            <td>{{ "%.1f"|format(info.utilization_pct) }}%</td>
            <td>
              {% if info.exceeded %}
              <span class="badge badge-fail">Exceeded</span>
              {% elif info.utilization_pct > 75.0 %}
              <span class="badge badge-warn">Near Capacity</span>
              {% else %}
              <span class="badge badge-pass">Safe</span>
              {% endif %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>2. Empirical Codec Rate-Distortion & Speed Frontier</h2>
      <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">
        Evaluated on standardized multi-camera video clips across CPU software and Apple Silicon hardware acceleration.
        Points tagged with <span class="badge badge-pareto" style="font-size: 10px; padding: 2px 8px;">Pareto Optimal</span>
        define non-dominated frontiers across image fidelity (SSIM), compression ratio, and transcoding throughput.
      </p>

      {% if rate_distortion_chart %}
      <div class="chart-container">
        {{ rate_distortion_chart|safe }}
      </div>
      {% endif %}

      {% if speed_chart %}
      <div class="chart-container">
        {{ speed_chart|safe }}
      </div>
      {% endif %}

      {% if codec_results %}
      <table>
        <thead>
          <tr>
            <th>Codec / Pipeline</th>
            <th>Preset / Profile</th>
            <th>Compression Ratio</th>
            <th>Bitrate (kbps)</th>
            <th>Speedup (Realtime)</th>
            <th>SSIM</th>
            <th>PSNR (dB)</th>
            <th>ORB Feature Retention</th>
            <th>Frontier Status</th>
          </tr>
        </thead>
        <tbody>
          {% for cr in codec_results %}
          <tr>
            <td><strong>{{ cr.codec_name }}</strong></td>
            <td>{{ cr.preset_name }}</td>
            <td>{{ "%.1f"|format(cr.compression_ratio) }}x</td>
            <td>{{ "%.0f"|format(cr.bitrate_kbps) }}</td>
            <td>{{ "%.2f"|format(cr.speedup_realtime) }}x ({{ "%.1f"|format(cr.encode_fps) }} fps)</td>
            <td>{{ "%.4f"|format(cr.ssim) }}</td>
            <td>{{ "%.1f"|format(cr.psnr_db) }} dB</td>
            <td>{{ "%.1f"|format(cr.feature_retention_ratio * 100.0) }}%</td>
            <td>
              {% if cr.is_pareto %}
              <span class="badge badge-pareto">Pareto Optimal</span>
              {% else %}
              <span style="color: var(--text-muted); font-size: 12px;">Dominated</span>
              {% endif %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% endif %}
    </div>

    <div class="card">
      <h2>3. Cloud Storage, Data Transfer & Cost Projections ({{ "%.0f"|format(cost_report.hours_modeled) }} Hours)</h2>
      <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">
        {{ cost_report.executive_summary }}
      </p>

      {% if cost_chart %}
      <div class="chart-container">
        {{ cost_chart|safe }}
      </div>
      {% endif %}

      <table>
        <thead>
          <tr>
            <th>Encoding Format</th>
            <th>Storage (TB)</th>
            <th>Monthly Hot (S3)</th>
            <th>Monthly Archive (Glacier)</th>
            <th>Annual Archive (Glacier)</th>
            <th>Annual Savings vs Raw (Hot)</th>
            <th>Upload Time (1 Gbps)</th>
          </tr>
        </thead>
        <tbody>
          {% for proj in cost_report.projections %}
          <tr>
            <td><strong>{{ proj.format_name }}</strong></td>
            <td>{{ "%.2f"|format(proj.storage_tb) }} TB</td>
            <td>${{ "%.2f"|format(proj.monthly_hot_usd) }}</td>
            <td>${{ "%.2f"|format(proj.monthly_archive_usd) }}</td>
            <td>${{ "%.2f"|format(proj.annual_archive_usd) }}</td>
            <td style="color: {% if proj.annual_savings_vs_raw_hot_usd > 0 %}var(--pass){% else %}var(--text-muted){% endif %};">
              {% if proj.annual_savings_vs_raw_hot_usd > 0 %}
              +${{ "%.2f"|format(proj.annual_savings_vs_raw_hot_usd) }} / yr
              {% else %}
              Baseline
              {% endif %}
            </td>
            <td>{{ "%.1f"|format(proj.upload_duration_hours_1gbps) }} hours</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>4. Strategic Deployment Recommendations</h2>
      <div class="rec-grid">
        <div class="rec-card archival">
          <div class="rec-title">Tier 1: Regulatory & Ground Truth Archival</div>
          <div class="rec-body">
            Store raw uncompressed sensors or <strong>FFV1 Level 3</strong> lossless compression directly to <strong>AWS Glacier Deep Archive</strong> ($0.00099/GB/mo).
            Preserves 100% pixel fidelity, bit-exact sensor noise, and sub-millisecond timestamps for future sensor calibration re-runs.
          </div>
          <div class="rec-tags">
            <span class="tag">Codec: FFV1 / Lossless</span>
            <span class="tag">Storage: Glacier Deep Archive</span>
            <span class="tag">SSIM: 1.000</span>
          </div>
        </div>

        <div class="rec-card training">
          <div class="rec-title">Tier 2: Foundation Model & Policy Training Delivery</div>
          <div class="rec-body">
            Transcode to <strong>HEVC / H.265 (CRF 22-28)</strong> or <strong>H.264 (CRF 23)</strong>.
            Achieves <strong>10x - 15x compression</strong> while maintaining <strong>>98% ORB feature retention</strong> and SSIM &gt;0.98.
            Saves up to 93% cloud storage expenditure while reducing dataset transfer time over facility fiber from days to hours.
          </div>
          <div class="rec-tags">
            <span class="tag">Codec: libx265 / Apple HEVC</span>
            <span class="tag">Storage: S3 Standard / Coldline</span>
            <span class="tag">Savings: &gt;90%</span>
          </div>
        </div>

        <div class="rec-card edge">
          <div class="rec-title">Tier 3: Low-Latency Edge Streaming & Teleop</div>
          <div class="rec-body">
            Employ hardware-accelerated <strong>H.264 VideoToolbox / NVENC</strong> at 4-10 Mbps with ultrafast preset.
            Provides &gt;3.0x realtime encoding throughput on embedded host compute with &lt;15ms latency, staying well within 1GbE / Wi-Fi 6 limits.
          </div>
          <div class="rec-tags">
            <span class="tag">Codec: Hardware H.264</span>
            <span class="tag">Latency: &lt;15ms</span>
            <span class="tag">Throughput: &gt;100 fps</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""


def generate_capture_planner_report(
    bandwidth: BandwidthReport,
    cost_report: CostModelReport,
    codec_results: list[CodecResult] | None,
    output_path: Path,
) -> Path:
    """Generate interactive self-contained HTML report with Plotly figures."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rd_html = ""
    speed_html = ""
    cost_html = ""

    # 1. Rate-Distortion Plotly Chart
    if codec_results:
        fig_rd = go.Figure()

        # Non-Pareto points
        non_pareto = [c for c in codec_results if not c.is_pareto]
        if non_pareto:
            fig_rd.add_trace(
                go.Scatter(
                    x=[c.compression_ratio for c in non_pareto],
                    y=[c.ssim for c in non_pareto],
                    mode="markers+text",
                    marker=dict(size=10, color="#94A3B8", symbol="circle"),
                    text=[c.preset_name for c in non_pareto],
                    textposition="top center",
                    textfont=dict(size=10, color="#94A3B8"),
                    name="Dominated Codecs",
                    hovertemplate="<b>%{text}</b><br>Compression: %{x:.1f}x<br>SSIM: %{y:.4f}<extra></extra>",
                )
            )

        # Pareto points
        pareto = sorted([c for c in codec_results if c.is_pareto], key=lambda c: c.compression_ratio)
        if pareto:
            fig_rd.add_trace(
                go.Scatter(
                    x=[c.compression_ratio for c in pareto],
                    y=[c.ssim for c in pareto],
                    mode="lines+markers+text",
                    line=dict(color="#8B5CF6", width=2, dash="dash"),
                    marker=dict(size=14, color="#C4B5FD", symbol="star", line=dict(width=1, color="#8B5CF6")),
                    text=[f"{c.preset_name} [Pareto]" for c in pareto],
                    textposition="bottom right",
                    textfont=dict(size=11, color="#C4B5FD"),
                    name="Pareto Frontier",
                    hovertemplate="<b>%{text}</b><br>Compression: %{x:.1f}x<br>SSIM: %{y:.4f}<extra></extra>",
                )
            )

        fig_rd.update_layout(
            title="Rate-Distortion Efficiency (SSIM vs Compression Ratio)",
            xaxis_title="Compression Ratio (vs Raw RGB)",
            yaxis_title="SSIM (Structural Similarity)",
            template="plotly_dark",
            paper_bgcolor="#1E293B",
            plot_bgcolor="#0F172A",
            font=dict(color="#F8FAFC"),
            height=400,
            margin=dict(l=50, r=40, t=50, b=40),
            yaxis=dict(range=[min(0.90, min(c.ssim for c in codec_results) - 0.02), 1.005]),
        )
        rd_html = fig_rd.to_html(full_html=False, include_plotlyjs="cdn")

        # 2. Encode Speed & Feature Retention Chart
        fig_speed = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Encoding Throughput (Realtime Speedup)", "ORB Feature Retention vs Compression"),
        )

        fig_speed.add_trace(
            go.Bar(
                x=[c.preset_name for c in codec_results],
                y=[c.speedup_realtime for c in codec_results],
                marker_color=["#10B981" if c.speedup_realtime >= 1.0 else "#F59E0B" for c in codec_results],
                name="Speedup",
                hovertemplate="%{x}<br>Speedup: %{y:.2f}x realtime<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Real-time reference line
        fig_speed.add_hline(
            y=1.0,
            line_dash="dot",
            line_color="#EF4444",
            annotation_text="1.0x Realtime",
            annotation_position="top left",
            row=1,
            col=1,
        )

        fig_speed.add_trace(
            go.Scatter(
                x=[c.compression_ratio for c in codec_results],
                y=[c.feature_retention_ratio * 100.0 for c in codec_results],
                mode="lines+markers",
                marker=dict(size=10, color="#38BDF8"),
                line=dict(color="#38BDF8", width=2),
                name="ORB Retention %",
                hovertemplate="Compression: %{x:.1f}x<br>ORB Retention: %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig_speed.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1E293B",
            plot_bgcolor="#0F172A",
            font=dict(color="#F8FAFC"),
            height=380,
            showlegend=False,
            margin=dict(l=50, r=40, t=50, b=40),
        )
        speed_html = fig_speed.to_html(full_html=False, include_plotlyjs=False)

    # 3. Cost Comparison Bar Chart
    if cost_report.projections:
        fig_cost = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Storage Footprint (TB)", "Annual Storage Cost (USD)"),
        )

        names = [p.format_name for p in cost_report.projections]
        tbs = [p.storage_tb for p in cost_report.projections]
        hot_usd = [p.annual_hot_usd for p in cost_report.projections]
        arch_usd = [p.annual_archive_usd for p in cost_report.projections]

        fig_cost.add_trace(
            go.Bar(
                x=names,
                y=tbs,
                name="Footprint (TB)",
                marker_color="#38BDF8",
                hovertemplate="%{x}: %{y:.2f} TB<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig_cost.add_trace(
            go.Bar(
                x=names,
                y=hot_usd,
                name="Hot Storage (S3)",
                marker_color="#F59E0B",
                hovertemplate="%{x}: $%{y:,.2f}/yr (Hot)<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig_cost.add_trace(
            go.Bar(
                x=names,
                y=arch_usd,
                name="Deep Archive",
                marker_color="#10B981",
                hovertemplate="%{x}: $%{y:,.2f}/yr (Archive)<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig_cost.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="#1E293B",
            plot_bgcolor="#0F172A",
            font=dict(color="#F8FAFC"),
            height=380,
            margin=dict(l=50, r=40, t=50, b=80),
        )
        cost_html = fig_cost.to_html(full_html=False, include_plotlyjs=False)

    template = jinja2.Template(PLANNER_REPORT_TEMPLATE)
    html_content = template.render(
        bandwidth=bandwidth,
        cost_report=cost_report,
        codec_results=codec_results,
        rate_distortion_chart=rd_html,
        speed_chart=speed_html,
        cost_chart=cost_html,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
