"""Interactive HTML report generation for QA engine evaluation benchmarks."""

from pathlib import Path

import jinja2
import numpy as np
import plotly.graph_objects as go

from rigsim.eval.benchmark import BenchmarkResult

EVAL_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RigSim QA Evaluation Benchmark Report — {{ result.suite_name }}</title>
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

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
    .stat-value { font-size: 26px; font-weight: 700; color: var(--text); margin-top: 4px; }
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
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-top">
        <div class="title-area">
          <h1>QA Engine Precision & Recall Evaluation Benchmark</h1>
          <div class="subtitle">Confusion Matrix, Temporal Localization (IoU) & Detection Sensitivity</div>
        </div>
        <div>
          <span class="badge {% if result.macro_f1 >= 0.90 %}badge-pass{% else %}badge-warn{% endif %}">
            Macro F1: {{ "%.1f"|format(result.macro_f1 * 100.0) }}%
          </span>
        </div>
      </div>

      <div class="summary-grid">
        <div class="stat-card">
          <div class="stat-label">Overall Accuracy</div>
          <div class="stat-value">{{ "%.1f"|format(result.overall_metrics.accuracy * 100.0) }}%</div>
          <div class="stat-sub">{{ result.total_sessions_evaluated }} Sessions Evaluated</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Macro Precision</div>
          <div class="stat-value">{{ "%.1f"|format(result.macro_precision * 100.0) }}%</div>
          <div class="stat-sub">Across 5 Primary Categories</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Macro Recall</div>
          <div class="stat-value">{{ "%.1f"|format(result.macro_recall * 100.0) }}%</div>
          <div class="stat-sub">True Positive Rate (Sensitivity)</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">False Positive Rate (FPR)</div>
          <div class="stat-value" style="color: {% if result.overall_metrics.false_positive_rate <= 0.05 %}var(--pass){% else %}var(--warn){% endif %};">
            {{ "%.1f"|format(result.overall_metrics.false_positive_rate * 100.0) }}%
          </div>
          <div class="stat-sub">Clean Session False Alarms</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Mean Temporal IoU</div>
          <div class="stat-value">{{ "%.1f"|format(result.mean_temporal_iou * 100.0) }}%</div>
          <div class="stat-sub">Defect Window Localization</div>
        </div>
      </div>
    </header>

    <div class="card">
      <h2>1. Multiclass Confusion Matrix & Performance Curves</h2>
      <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">
        Rows represent ground-truth defect categories; columns represent categories predicted by the QA engine.
      </p>

      {% if confusion_matrix_chart %}
      <div class="chart-container">
        {{ confusion_matrix_chart|safe }}
      </div>
      {% endif %}

      {% if category_metrics_chart %}
      <div class="chart-container">
        {{ category_metrics_chart|safe }}
      </div>
      {% endif %}
    </div>

    <div class="card">
      <h2>2. High-Level Category Performance Breakdown</h2>
      <table>
        <thead>
          <tr>
            <th>Category</th>
            <th>True Positives (TP)</th>
            <th>False Positives (FP)</th>
            <th>True Negatives (TN)</th>
            <th>False Negatives (FN)</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
          </tr>
        </thead>
        <tbody>
          {% for cat, m in result.category_metrics.items() %}
          <tr>
            <td><strong>{{ cat }}</strong></td>
            <td>{{ m.tp }}</td>
            <td>{{ m.fp }}</td>
            <td>{{ m.tn }}</td>
            <td>{{ m.fn }}</td>
            <td>{{ "%.1f"|format(m.precision * 100.0) }}%</td>
            <td>{{ "%.1f"|format(m.recall * 100.0) }}%</td>
            <td><strong>{{ "%.1f"|format(m.f1 * 100.0) }}%</strong></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>3. Evaluated Session Ledger ({{ result.session_summaries|length }} Sessions)</h2>
      <table>
        <thead>
          <tr>
            <th>Session ID</th>
            <th>Ground Truth Category</th>
            <th>Predicted Category</th>
            <th>Injected Faults</th>
            <th>Overall Verdict</th>
            <th>Failing Checks Triggered</th>
            <th>Usable Yield %</th>
          </tr>
        </thead>
        <tbody>
          {% for s in result.session_summaries %}
          <tr>
            <td><code>{{ s.session_id }}</code></td>
            <td>{{ s.true_category }}</td>
            <td>
              <span class="badge {% if s.true_category == s.pred_category %}badge-pass{% else %}badge-fail{% endif %}">
                {{ s.pred_category }}
              </span>
            </td>
            <td>{{ s.gt_fault_count }}</td>
            <td>
              <span class="badge {% if s.verdict == 'PASS' %}badge-pass{% elif s.verdict == 'WARN' %}badge-warn{% else %}badge-fail{% endif %}">
                {{ s.verdict }}
              </span>
            </td>
            <td>{{ s.failing_checks|join(", ") if s.failing_checks else "None (All Clean)" }}</td>
            <td>{{ "%.1f"|format(s.usable_percentage) }}%</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def generate_evaluation_report(
    result: BenchmarkResult,
    output_path: Path,
) -> Path:
    """Generate self-contained HTML evaluation report with Plotly figures."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm_html = ""
    metrics_html = ""

    # 1. Plotly Confusion Matrix Heatmap
    cm = result.confusion_matrix_5x5
    labels = result.category_labels

    # Normalize by row sum for percentage annotations
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=f"{cm[i, j]}<br>({cm_norm[i, j]:.0f}%)",
                    font=dict(color="#FFFFFF" if cm_norm[i, j] > 40 else "#94A3B8", size=12),
                    showarrow=False,
                )
            )

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm_norm,
            x=labels,
            y=labels,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Accuracy %"),
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Percentage: %{z:.1f}%<extra></extra>",
        )
    )
    fig_cm.update_layout(
        title="5×5 Confusion Matrix (Normalized % by Ground Truth Row)",
        xaxis_title="Predicted Category",
        yaxis_title="Ground Truth Category",
        annotations=annotations,
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        height=450,
        margin=dict(l=80, r=40, t=50, b=50),
        yaxis=dict(autorange="reversed"),
    )
    cm_html = fig_cm.to_html(full_html=False, include_plotlyjs="cdn")

    # 2. Category Metrics Bar Chart (Precision, Recall, F1)
    fig_metrics = go.Figure()
    cats = list(result.category_metrics.keys())
    precs = [result.category_metrics[c].precision * 100.0 for c in cats]
    recs = [result.category_metrics[c].recall * 100.0 for c in cats]
    f1s = [result.category_metrics[c].f1 * 100.0 for c in cats]

    fig_metrics.add_trace(go.Bar(x=cats, y=precs, name="Precision %", marker_color="#38BDF8"))
    fig_metrics.add_trace(go.Bar(x=cats, y=recs, name="Recall %", marker_color="#10B981"))
    fig_metrics.add_trace(go.Bar(x=cats, y=f1s, name="F1-Score %", marker_color="#8B5CF6"))

    fig_metrics.update_layout(
        barmode="group",
        title="Per-Category Statistical Performance (Precision, Recall, F1)",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 105]),
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        height=380,
        margin=dict(l=50, r=40, t=50, b=50),
    )
    metrics_html = fig_metrics.to_html(full_html=False, include_plotlyjs=False)

    template = jinja2.Template(EVAL_REPORT_TEMPLATE)
    html_content = template.render(
        result=result,
        confusion_matrix_chart=cm_html,
        category_metrics_chart=metrics_html,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
