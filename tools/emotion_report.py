"""
Emotion session report generator.

Reads a CSV session file and produces a self-contained HTML report with:
  - Session summary (duration, dominant emotions, stats)
  - Emotion distribution pie chart (Plotly)
  - Stacked timeline (dominance switches over time)
  - Per-emotion bar chart (avg intensity)
  - Raw data table (scrollable)

Usage:
    python emotion_report.py emotions_alice_20260611_120000.csv
    python emotion_report.py output/emotions_alice_20260611_120000.csv
    python emotion_report.py --label alice           (auto-finds latest)

Output:
    output/report_<label>_<timestamp>.html
"""
import sys
import os
import csv
import json
from datetime import datetime
from collections import Counter

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SRC_DIR, "..", "output")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
EMOTION_COLORS = {
    "neutral":    "#aaaaaa",
    "happy":      "#00cc00",
    "sad":        "#4488ff",
    "angry":      "#cc0000",
    "surprised":  "#ffaa00",
    "fearful":    "#6600cc",
    "disgusted":  "#006600",
}

# --- Helpers ------------------------------------------------------
def load_csv(path):
    """Return {t, emotions, dominance, label, n_faces}."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = rows[1:]
    if not data:
        return None

    t = [float(r[0]) for r in data]
    scores = {e: [float(r[header.index(e)]) for r in data] for e in EMOTIONS}
    dominance = [r[header.index("dominant_emotion")] for r in data]
    n_faces = [int(r[header.index("n_faces")]) for r in data]

    label = os.path.splitext(os.path.basename(path))[0]
    if label.startswith("emotions_"):
        label = label[9:]
    elif label.startswith("audio_emotions_"):
        label = label[15:]

    return {"t": t, "scores": scores, "dominance": dominance, "n_faces": n_faces, "label": label}


def find_latest(label=None):
    """Find the most recent CSV in output/csv/ matching optional label."""
    csv_dir = os.path.join(OUTPUT_DIR, "csv")
    candidates = []
    try:
        files = os.listdir(csv_dir)
    except FileNotFoundError:
        files = []
    for f in files:
        if f.endswith(".csv") and f.startswith("emotions"):
            path = os.path.join(csv_dir, f)
            mtime = os.path.getmtime(path)
            if label and label not in f:
                continue
            candidates.append((mtime, path))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None


# --- Report generation --------------------------------------------
def build_report(path):
    d = load_csv(path)
    if d is None:
        print(f"Empty or invalid CSV: {path}")
        return None

    is_audio = os.path.basename(path).startswith("audio_emotions")
    report_title = "Audio Emotion Session Report" if is_audio else "Emotion Session Report"
    footer_text = "Generated with speech emotion recognition" if is_audio else "Generated with HSEmotion (EfficientNet-B0, AffectNet)"

    total_s = d["t"][-1]
    n_samples = len(d["t"])
    sample_rate = n_samples / total_s if total_s > 0 else 0

    # Emotion distribution (weighted by dominance count)
    dom_counts = Counter(d["dominance"])
    total = sum(dom_counts.values())

    # Summary stats
    summary_rows = ""
    for emo in EMOTIONS:
        vals = d["scores"][emo]
        avg = sum(vals) / len(vals)
        peak = max(vals)
        dom_pct = dom_counts.get(emo, 0) / max(total, 1) * 100
        summary_rows += f"""
        <tr>
            <td style="color:{EMOTION_COLORS[emo]};font-weight:bold">{emo}</td>
            <td>{avg:.1f}%</td>
            <td>{peak:.1f}%</td>
            <td>{dom_pct:.1f}%</td>
        </tr>"""

    # Pie chart data
    pie_labels = json.dumps([f"{e} ({dom_counts.get(e,0)})" for e in EMOTIONS])
    pie_values = json.dumps([dom_counts.get(e, 0) for e in EMOTIONS])
    pie_colors = json.dumps([EMOTION_COLORS[e] for e in EMOTIONS])

    # Timeline data (downsample to ~200 points for HTML)
    step = max(1, n_samples // 200)
    tl_t = [round(d["t"][i], 1) for i in range(0, n_samples, step)]
    tl_dom = [d["dominance"][i] for i in range(0, n_samples, step)]

    # Per-frame dominance encoded as numeric indices for the heatmap
    dom_idx_map = {e: i for i, e in enumerate(EMOTIONS)}
    tl_dom_idx = [dom_idx_map.get(x, 0) for x in tl_dom]

    # Emotion traces for timeline (each emotion's intensity over time)
    traces_js = ""
    for emo in EMOTIONS:
        vals = [d["scores"][emo][i] for i in range(0, n_samples, step)]
        traces_js += f"""
        {{
            x: timeline_t,
            y: {json.dumps(vals)},
            name: '{emo}',
            type: 'scatter',
            mode: 'lines',
            stackgroup: 'one',
            line: {{width: 0}},
            fillcolor: '{EMOTION_COLORS[emo]}',
            opacity: 0.6,
        }},"""

    # Bar chart: avg emotion intensity
    bar_vals = []
    for emo in EMOTIONS:
        vals = d["scores"][emo]
        bar_vals.append(sum(vals) / len(vals))

    # N faces over time
    nf_vals = [d["n_faces"][i] for i in range(0, n_samples, step)]

    report_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Emotion Report - {d["label"]}</title>
<script src="../plotly-2.32.0.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
.container {{ max-width: 1100px; margin: 0 auto; }}
h1 {{ color: #fff; margin-bottom: 4px; }}
.meta {{ color: #888; font-size: 14px; margin-bottom: 24px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.card {{ background: #16213e; border-radius: 10px; padding: 20px; }}
.card.full {{ grid-column: 1 / -1; }}
.card h2 {{ font-size: 16px; color: #ccc; margin-bottom: 12px; border-bottom: 1px solid #2a3a5a; padding-bottom: 8px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #2a3a5a; font-size: 14px; }}
th {{ color: #aaa; font-weight: 600; }}
.chart {{ width: 100%; height: 350px; }}
.timeline-chart {{ width: 100%; height: 400px; }}
.raw-table {{ max-height: 400px; overflow-y: auto; }}
.raw-table table {{ font-size: 12px; }}
.raw-table th {{ position: sticky; top: 0; background: #16213e; z-index: 1; }}
.footer {{ margin-top: 24px; color: #555; font-size: 12px; text-align: center; }}
</style>
</head>
<body>
<div class="container">
<h1>{report_title}</h1>
<div class="meta">
    Label: <strong>{d["label"]}</strong> &bull;
    Generated: {report_ts} &bull;
    Duration: {total_s:.0f}s &bull;
    Samples: {n_samples} ({sample_rate:.1f}/s)
</div>

<!-- Summary -->
<div class="card" style="margin-bottom:20px">
    <h2>Per-Emotion Summary</h2>
    <table>
    <tr><th>Emotion</th><th>Avg Intensity</th><th>Peak</th><th>Dominance %</th></tr>
    {summary_rows}
    </table>
</div>

<!-- Charts grid -->
<div class="grid">
    <div class="card">
        <h2>Emotion Distribution (dominance count)</h2>
        <div id="pie" class="chart"></div>
    </div>
    <div class="card">
        <h2>Average Emotion Intensity</h2>
        <div id="bar" class="chart"></div>
    </div>
    <div class="card full">
        <h2>Emotion Timeline (stacked area)</h2>
        <div id="timeline" class="timeline-chart"></div>
    </div>
    <div class="card full">
        <h2>Raw Data (first 500 rows)</h2>
        <div class="raw-table" id="raw"></div>
    </div>
</div>

<div class="footer">
    {footer_text} &bull; Claude Code &amp; Happy
</div>
</div>

<script>
const timeline_t = {json.dumps(tl_t)};

// --- Pie Chart ---
Plotly.newPlot('pie', [{{
    values: {pie_values},
    labels: {pie_labels},
    type: 'pie',
    hole: 0.4,
    marker: {{colors: {pie_colors}}},
    textinfo: 'label+percent',
    textposition: 'outside',
}}], {{
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{color: '#aaa', size: 12}},
    margin: {{t: 10, b: 10, l: 10, r: 10}},
    showlegend: false,
}});

// --- Bar Chart ---
Plotly.newPlot('bar', [{{
    x: {json.dumps(EMOTIONS)},
    y: {json.dumps(bar_vals)},
    type: 'bar',
    marker: {{color: {json.dumps([EMOTION_COLORS[e] for e in EMOTIONS])}}},
    text: {json.dumps([f"{v:.1f}%" for v in bar_vals])},
    textposition: 'auto',
}}], {{
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{color: '#aaa', size: 12}},
    xaxis: {{color: '#aaa'}},
    yaxis: {{title: 'Avg Intensity (%)', color: '#aaa', range: [0, 100]}},
    margin: {{t: 10, b: 50, l: 50, r: 10}},
}});

// --- Timeline stacked area ---
Plotly.newPlot('timeline', [
    {traces_js}
], {{
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{color: '#aaa', size: 11}},
    xaxis: {{title: 'Time (s)', color: '#aaa', gridcolor: '#2a3a5a'}},
    yaxis: {{title: '%', color: '#aaa', gridcolor: '#2a3a5a'}},
    margin: {{t: 10, b: 40, l: 50, r: 10}},
    showlegend: true,
    legend: {{orientation: 'h', y: 1.1, font: {{size: 11}}}},
    hovermode: 'x unified',
}});

// --- Raw table ---
(function() {{
    const container = document.getElementById('raw');
    // Build from Python data directly
    const emotionKeys = {json.dumps(EMOTIONS)};
    const emotionColors = {json.dumps(EMOTION_COLORS)};
    const allT = {json.dumps([round(d["t"][i],1) for i in range(0, min(n_samples, 500))])};
    const allDom = {json.dumps([d["dominance"][i] for i in range(0, min(n_samples, 500))])};
    const allScores = {json.dumps({e: [round(d["scores"][e][i],1) for i in range(0, min(n_samples, 500))] for e in EMOTIONS})};

    let html = '<table><thead><tr><th>Time</th><th>Dominant</th>';
    for (const e of emotionKeys) html += '<th>' + e + '</th>';
    html += '</tr></thead><tbody>';

    for (let i = 0; i < allT.length; i++) {{
        html += '<tr><td>' + allT[i].toFixed(1) + '</td>';
        html += '<td style="color:' + (emotionColors[allDom[i]] || '#e0e0e0') + '">' + allDom[i] + '</td>';
        for (const e of emotionKeys) {{
            html += '<td>' + allScores[e][i].toFixed(0) + '</td>';
        }}
        html += '</tr>';
    }}
    html += '</tbody></table>';
    container.innerHTML = html;
}})();
</script>

</body>
</html>"""

    return html


# --- Main ---------------------------------------------------------
def main():
    path = None
    label = None

    for a in sys.argv[1:]:
        if a in ("--label", "-l"):
            label = ""
        elif label == "":
            label = a
        else:
            path = a

    if path is None:
        if label:
            path = find_latest(label)
        else:
            path = find_latest()
        if path is None:
            print("No CSV found. Use: python emotion_report.py <file.csv>")
            sys.exit(1)

    html = build_report(path)
    if html is None:
        sys.exit(1)

    # Write to output/
    basename = os.path.splitext(os.path.basename(path))[0]
    report_path = os.path.join(REPORTS_DIR, f"report_{basename}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
