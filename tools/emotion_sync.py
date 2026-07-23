"""
Emotion synchrony analysis — multi-metric coupling between two session CSVs.

Metrics (per emotion + aggregate):
  1. Pearson r          — linear co-variation (symmetric)
  2. Cross-correlation  — best lag & peak r (who leads whom)
  3. Mutual Information — nonlinear shared information (normalised 0-1)
  4. Phase Locking      — are emotion oscillations in sync (0-1)
  5. Granger Causality  — does A's past predict B's present (directional)
  6. Running Corr       — how synchrony evolves over time

CSV format: elapsed_s, wall_time, frame_idx, dominant_emotion, <7 emotions>, n_faces

Usage:
    python emotion_sync.py person_a.csv person_b.csv
"""
import sys
import csv
import os
import json
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.signal import hilbert

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_wall_time(s):
    s = s.strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None
        rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        return None

    has_wall = "wall_time" in header
    if has_wall:
        wall_epochs = [_parse_wall_time(r[header.index("wall_time")]) for r in rows]
        valid = [i for i, w in enumerate(wall_epochs) if w is not None]
        if valid:
            t0 = min(wall_epochs[i] for i in valid)
            t = np.array([(wall_epochs[i] - t0) for i in valid])
        else:
            has_wall = False
    if not has_wall:
        t = np.array([float(r[0]) for r in rows])
        valid = list(range(len(rows)))

    scores = {emo: np.array([float(rows[i][header.index(emo)]) for i in valid])
              for emo in EMOTIONS}
    dominance = np.array([rows[i][header.index("dominant_emotion")] for i in valid])
    return {"t": t, "scores": scores, "dominance": dominance, "label": path,
            "has_wall": has_wall, "n_total": len(rows), "n_valid": len(valid)}


def resample_common(a, b, rate=5.0):
    t_min = max(a["t"][0], b["t"][0])
    t_max = min(a["t"][-1], b["t"][-1])
    if t_max <= t_min:
        return None, None
    t_common = np.arange(t_min, t_max, 1.0 / rate)

    def _r(src):
        res = {"t": t_common, "scores": {}, "label": src["label"]}
        for emo in EMOTIONS:
            f = interp1d(src["t"], src["scores"][emo], kind="linear",
                         bounds_error=False, fill_value="extrapolate")
            res["scores"][emo] = np.clip(f(t_common), 0, 100)
        idx = np.clip(np.searchsorted(src["t"], t_common, side="left"),
                      0, len(src["dominance"]) - 1)
        res["dominance"] = np.array(src["dominance"])[idx]
        return res

    return _r(a), _r(b)


# ---------------------------------------------------------------------------
# Metric 1 — Pearson r (existing)
# ---------------------------------------------------------------------------
def pearson_r(x, y):
    return np.corrcoef(x, y)[0, 1]


# ---------------------------------------------------------------------------
# Metric 2 — Cross-correlation peak (existing)
# ---------------------------------------------------------------------------
def cross_correlation(x, y, fs=5.0):
    xc = np.correlate(x - np.mean(x), y - np.mean(y), mode="full")
    lags = np.arange(-len(x) + 1, len(x)) / fs
    denom = max(np.std(x) * np.std(y) * len(x), 1e-10)
    xc_norm = xc / denom
    best = np.argmax(np.abs(xc_norm))
    return lags[best], xc_norm[best]


# ---------------------------------------------------------------------------
# Metric 3 — Mutual Information (nonlinear coupling)
# ---------------------------------------------------------------------------
def mutual_information(x, y, bins=20):
    """Normalised mutual information: I(X;Y) / sqrt(H(X)*H(Y)), in [0, 1]."""
    xy = np.stack([x, y], axis=1)
    # 2D histogram
    h2d, _, _ = np.histogram2d(x, y, bins=bins)
    h2d = h2d / len(x)
    # Marginals
    px = h2d.sum(axis=1)
    py = h2d.sum(axis=0)
    # Entropies
    mask_x = px > 0
    mask_y = py > 0
    hx = -np.sum(px[mask_x] * np.log(px[mask_x]))
    hy = -np.sum(py[mask_y] * np.log(py[mask_y]))
    # Mutual information
    mask = h2d > 0
    mi = np.sum(h2d[mask] * np.log(h2d[mask] / np.outer(px, py)[mask]))
    denom = np.sqrt(max(hx * hy, 1e-10))
    return mi / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Metric 4 — Phase Locking Value (are oscillations in sync?)
# ---------------------------------------------------------------------------
def phase_locking_value(x, y):
    """PLV via Hilbert transform. 0=no locking, 1=perfect locking."""
    xa = hilbert(x - np.mean(x))
    ya = hilbert(y - np.mean(y))
    phase_diff = np.angle(xa) - np.angle(ya)
    return float(np.abs(np.mean(np.exp(1j * phase_diff))))


# ---------------------------------------------------------------------------
# Metric 5 — Granger Causality (directional prediction)
# ---------------------------------------------------------------------------
def granger_causality(x, y, max_lag=5):
    """
    F-test for Granger causality: does past of X help predict Y?
    Returns (F_stat_A->B, p_A->B, F_stat_B->A, p_B->A).
    Significant p (<.05) = X Granger-causes Y.
    """
    n = len(x)
    if n <= max_lag + 2:
        return np.nan, np.nan, np.nan, np.nan

    def _gc_test(cause, effect):
        # Restricted model: effect(t) = a0 + sum(bi * effect(t-i))
        Y = effect[max_lag:]
        X_restricted = np.column_stack([effect[max_lag - i - 1:n - i - 1]
                                        for i in range(max_lag)])
        X_restricted = np.column_stack([np.ones(len(Y)), X_restricted])
        beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
        resid_r = Y - X_restricted @ beta_r
        ssr_r = np.sum(resid_r ** 2)

        # Unrestricted model: adds past of cause
        X_unrestricted = np.column_stack([
            X_restricted,
            np.column_stack([cause[max_lag - i - 1:n - i - 1]
                             for i in range(max_lag)]),
        ])
        beta_u = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
        resid_u = Y - X_unrestricted @ beta_u
        ssr_u = np.sum(resid_u ** 2)

        # F-statistic
        p_restricted = 1 + max_lag
        p_unrestricted = 1 + 2 * max_lag
        df1 = p_unrestricted - p_restricted
        df2 = n - p_unrestricted
        if ssr_u < 1e-10:
            return float("inf"), 0.0
        F = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        # Approximate p-value from F-distribution (CDF approximation)
        p = _f_cdf_approx(F, df1, df2)
        return float(F), float(p)

    f_ab, p_ab = _gc_test(x, y)
    f_ba, p_ba = _gc_test(y, x)
    return f_ab, p_ab, f_ba, p_ba


def _f_cdf_approx(F, df1, df2):
    """Approximate upper-tail p-value of F-distribution."""
    if F <= 0 or np.isnan(F) or np.isinf(F):
        return 1.0
    x = df2 / (df2 + df1 * F)
    # Regularized incomplete beta via continued fraction
    a, b = df2 / 2, df1 / 2
    # Very rough approximation; good enough for significance testing
    p = np.exp(-F * df1 / (2 * (1 + F * df1 / df2)))
    return min(1.0, max(0.0, float(p)))


# ---------------------------------------------------------------------------
# Metric 6 — Running Correlation (time-varying synchrony)
# ---------------------------------------------------------------------------
def running_correlation(x, y, window_s=10.0, fs=5.0):
    """Sliding-window Pearson r. Returns (time, r) arrays."""
    win = int(window_s * fs)
    n = len(x) - win + 1
    if n <= 0:
        return np.array([]), np.array([])
    rs = np.array([np.corrcoef(x[i:i + win], y[i:i + win])[0, 1]
                   for i in range(n)])
    t_center = np.arange(win // 2, win // 2 + n) / fs
    return t_center, rs


# ===================================================================
# Main
# ===================================================================
def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <session_a.csv> <session_b.csv>")
        sys.exit(1)

    a = load_csv(sys.argv[1])
    b = load_csv(sys.argv[2])
    if a is None or b is None:
        print("One or both CSV files are empty or invalid.")
        sys.exit(1)

    align = "wall-clock time" if (a["has_wall"] and b["has_wall"]) else "elapsed_s"
    print(f"{'='*72}")
    print(f"A: {a['label']}  ({a['n_valid']} valid samples, {a['t'][-1]:.1f}s)")
    print(f"B: {b['label']}  ({b['n_valid']} valid samples, {b['t'][-1]:.1f}s)")
    print(f"Alignment: {align}")

    ra, rb = resample_common(a, b)
    if ra is None:
        print("\nERROR: No overlapping time range.")
        sys.exit(1)

    if a["has_wall"] and b["has_wall"]:
        offset = a["t"][0] - b["t"][0]
        if abs(offset) > 1:
            who = "A started later" if offset > 0 else "B started later"
            print(f"Wall-clock offset: {abs(offset):.1f}s ({who})")
        else:
            print(f"Wall-clock offset: {offset:.2f}s (synchronized)")

    overlap = ra["t"][-1] - ra["t"][0]
    print(f"Overlap: {overlap:.1f}s @ 5 Hz -> {len(ra['t'])} resampled points")
    print(f"{'='*72}")

    # ---- Per-emotion metrics ----
    print(f"\n{'Emotion':<12} {'Pearson':>8} {'Lag(s)':>7} {'Lag-r':>7} "
          f"{'MI':>6} {'PLV':>6} {'GC A>B':>8} {'GC B>A':>8}")
    print("-" * 72)

    results = {}
    for emo in EMOTIONS:
        x, y = ra["scores"][emo], rb["scores"][emo]

        r = pearson_r(x, y)
        best_lag, lag_r = cross_correlation(x, y)
        mi = mutual_information(x, y)
        plv = phase_locking_value(x, y)
        gc_ab, p_ab, gc_ba, p_ba = granger_causality(x, y)

        # GC stars
        g1 = f"{gc_ab:.2f}" + ("*" if p_ab < 0.05 else "")
        g2 = f"{gc_ba:.2f}" + ("*" if p_ba < 0.05 else "")

        results[emo] = {"r": r, "lag": best_lag, "lag_r": lag_r,
                        "mi": mi, "plv": plv, "gc_ab": gc_ab, "gc_ba": gc_ba}

        print(f"{emo:<12} {r:>8.3f} {best_lag:>6.1f}s {lag_r:>7.3f} "
              f"{mi:>6.3f} {plv:>6.3f} {g1:>8} {g2:>8}")

    # ---- Composite (mean of all 7 emotions) ----
    print(f"\n{'='*72}")
    print("COMPOSITE (mean across emotions)")
    print(f"{'='*72}")
    for metric, label in [("r", "Pearson r"),
                           ("lag_r", "Cross-corr peak r"),
                           ("mi", "Mutual Information"),
                           ("plv", "Phase Locking Value")]:
        vals = [results[e][metric] for e in EMOTIONS if not np.isnan(results[e][metric])]
        if vals:
            print(f"  {label:<22}: {np.mean(vals):.4f}")

    # GC summary
    gc_significant_ab = sum(1 for e in EMOTIONS
                            if not np.isnan(results[e]["gc_ab"])
                            and _f_cdf_approx(results[e]["gc_ab"], 5, len(ra["t"]) - 11) < 0.05)
    gc_significant_ba = sum(1 for e in EMOTIONS
                            if not np.isnan(results[e]["gc_ba"])
                            and _f_cdf_approx(results[e]["gc_ba"], 5, len(ra["t"]) - 11) < 0.05)
    print(f"  Granger A->B significant : {gc_significant_ab}/7 emotions")
    print(f"  Granger B->A significant : {gc_significant_ba}/7 emotions")

    # ---- Running correlation summary ----
    win_s = min(10.0, max(2.0, overlap / 3))
    print(f"\n{'='*72}")
    print(f"RUNNING CORRELATION ({win_s:.0f}s window) — happy & sad")
    print(f"{'='*72}")
    for emo in ["happy", "sad"]:
        x, y = ra["scores"][emo], rb["scores"][emo]
        t_rc, rc = running_correlation(x, y, window_s=win_s)
        if len(rc) > 0:
            print(f"  {emo}: mean={np.mean(rc):.3f}  min={np.min(rc):.3f}  "
                  f"max={np.max(rc):.3f}  std={np.std(rc):.3f}")

    # ---- Dominance agreement ----
    d_match = np.mean(ra["dominance"] == rb["dominance"])
    print(f"\nDominant emotion agreement: {d_match:.1%}")

    print(f"\n{'='*72}")
    print("LEGEND")
    print(f"  Pearson r : -1=anti-phase,  0=independent, +1=perfect sync")
    print(f"  Lag       : negative=B leads A, positive=A leads B")
    print(f"  MI        : 0=independent, 1=deterministic (nonlinear)")
    print(f"  PLV       : 0=no phase locking, 1=perfect locking")
    print(f"  GC A>B    : * = A Granger-causes B at p<.05")
    print(f"{'='*72}")

    # --- Generate HTML report ---
    html = _build_sync_report(
        a_label=a["label"], b_label=b["label"],
        a_dur=a["t"][-1], b_dur=b["t"][-1],
        align=align, overlap=overlap, n_points=len(ra["t"]),
        results=results, gc_ab=gc_significant_ab, gc_ba=gc_significant_ba,
        ra=ra, rb=rb, d_match=d_match,
    )
    out_dir = os.path.dirname(os.path.abspath(sys.argv[1]))
    # If the CSV is in output/csv/, go up one level to get the base output dir
    if os.path.basename(out_dir) == "csv":
        out_dir = os.path.dirname(out_dir)
    if "output" not in os.path.basename(out_dir):
        out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    sync_dir = os.path.join(out_dir, "sync")
    os.makedirs(sync_dir, exist_ok=True)
    report_path = os.path.join(sync_dir,
        f"sync_{os.path.splitext(os.path.basename(sys.argv[1]))[0]}_"
        f"{os.path.splitext(os.path.basename(sys.argv[2]))[0]}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nSync report saved to: {report_path}")


def _t_cdf_approx(t, df):
    x = df / (df + t**2)
    return 1 - 0.5 * np.exp(-0.5 * t**2 * (1 - 1 / (4 * df)))


def _build_sync_report(a_label, b_label, a_dur, b_dur, align, overlap,
                       n_points, results, gc_ab, gc_ba, ra, rb, d_match):
    """Generate a self-contained HTML report for synchrony analysis."""

    EMOTION_COLORS = {
        "neutral": "#aaaaaa", "happy": "#00cc00", "sad": "#4488ff",
        "angry": "#cc0000", "surprised": "#ffaa00", "fearful": "#6600cc",
        "disgusted": "#006600",
    }

    # --- Summary table rows ---
    table_rows = ""
    for emo in EMOTIONS:
        r = results[emo]
        gc_a = f"{r['gc_ab']:.2f}" + ("*" if not np.isnan(r['gc_ab']) and _gc_p(r['gc_ab'], n_points) < 0.05 else "")
        gc_b = f"{r['gc_ba']:.2f}" + ("*" if not np.isnan(r['gc_ba']) and _gc_p(r['gc_ba'], n_points) < 0.05 else "")
        table_rows += f"""
        <tr>
            <td style="color:{EMOTION_COLORS[emo]};font-weight:bold">{emo}</td>
            <td>{r['r']:.3f}</td>
            <td>{r['lag']:.1f}s</td>
            <td>{r['lag_r']:.3f}</td>
            <td>{r['mi']:.3f}</td>
            <td>{r['plv']:.3f}</td>
            <td>{gc_a}</td>
            <td>{gc_b}</td>
        </tr>"""

    # --- Composite metrics ---
    comp_r = np.mean([results[e]["r"] for e in EMOTIONS if not np.isnan(results[e]["r"])])
    comp_mi = np.mean([results[e]["mi"] for e in EMOTIONS])
    comp_plv = np.mean([results[e]["plv"] for e in EMOTIONS])

    # --- Running correlation traces ---
    rc_js = ""
    win_s_rc = min(10.0, max(2.0, overlap / 3))
    for emo in ["happy", "sad"]:
        x, y = ra["scores"][emo], rb["scores"][emo]
        t_rc, rc = running_correlation(x, y, window_s=win_s_rc)
        if len(rc) > 0:
            rc_js += json.dumps({
                "x": [round(t, 1) for t in t_rc.tolist()],
                "y": [round(v, 4) for v in rc.tolist()],
                "name": emo,
                "type": "scatter",
                "mode": "lines",
                "line": {"color": EMOTION_COLORS[emo], "width": 2},
            }) + ","

    # --- Timeline traces for both sessions ---
    tl_js = ""
    for label, src, alpha in [(a_label, ra, "A"), (b_label, rb, "B")]:
        tl_js += json.dumps({
            "x": [round(t, 1) for t in src["t"].tolist()],
            "y": [round(src["scores"]["happy"][i], 1) + round(src["scores"]["sad"][i], 1)
                  for i in range(len(src["t"]))],
            "name": f"{alpha}: happy+sad",
            "type": "scatter",
            "mode": "lines",
            "line": {"width": 1.5},
            "opacity": 0.7,
        }) + ","

    emotion_pairs_js = ""
    for emo_idx, emo in enumerate(EMOTIONS):
        y_a = [round(ra["scores"][emo][i], 1) for i in range(len(ra["t"]))]
        y_b = [round(rb["scores"][emo][i], 1) for i in range(len(rb["t"]))]
        t_arr = [round(t, 1) for t in ra["t"].tolist()]
        emotion_pairs_js += json.dumps({
            "y": y_a, "x": t_arr,
            "name": f"A: {emo}",
            "type": "scatter", "mode": "lines",
            "line": {"color": EMOTION_COLORS[emo], "width": 1.5, "dash": "solid"},
            "legendgroup": emo, "showlegend": True,
            "xaxis": f"x{emo_idx+1}", "yaxis": f"y{emo_idx+1}",
        }) + ","
        emotion_pairs_js += json.dumps({
            "y": y_b, "x": t_arr,
            "name": f"B: {emo}",
            "type": "scatter", "mode": "lines",
            "line": {"color": EMOTION_COLORS[emo], "width": 1.5, "dash": "dot"},
            "legendgroup": emo, "showlegend": False,
            "xaxis": f"x{emo_idx+1}", "yaxis": f"y{emo_idx+1}",
        }) + ","

    # --- Bar chart: per-emotion Pearson r ---
    bar_pearson = [results[e]["r"] if not np.isnan(results[e]["r"]) else 0 for e in EMOTIONS]
    bar_mi = [results[e]["mi"] for e in EMOTIONS]
    bar_plv = [results[e]["plv"] for e in EMOTIONS]

    report_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Emotion Sync Report — {a_label} vs {b_label}</title>
<script src="../plotly-2.32.0.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
.container {{ max-width: 1100px; margin: 0 auto; }}
h1 {{ color: #fff; margin-bottom: 4px; }}
.meta {{ color: #888; font-size: 14px; margin-bottom: 20px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.card {{ background: #16213e; border-radius: 10px; padding: 20px; }}
.card.full {{ grid-column: 1 / -1; }}
.card.emotion-grid {{ grid-column: 1 / -1; }}
.emotion-pairs {{ width: 100%; height: 850px; }}
.card h2 {{ font-size: 16px; color: #ccc; margin-bottom: 12px;
           border-bottom: 1px solid #2a3a5a; padding-bottom: 8px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: center; padding: 6px 8px; border-bottom: 1px solid #2a3a5a; font-size: 13px; }}
th {{ color: #aaa; font-weight: 600; }}
th:first-child, td:first-child {{ text-align: left; }}
.chart {{ width: 100%; height: 350px; }}
.timeline-chart {{ width: 100%; height: 300px; }}
.kpi {{ display: flex; gap: 15px; margin-bottom: 20px; }}
.kpi-box {{ flex: 1; background: #16213e; border-radius: 8px; padding: 14px 18px; text-align: center; }}
.kpi-box .val {{ font-size: 28px; font-weight: bold; }}
.kpi-box .lbl {{ font-size: 11px; color: #888; margin-top: 4px; }}
.footer {{ margin-top: 24px; color: #555; font-size: 12px; text-align: center; }}
</style>
</head>
<body>
<div class="container">
<h1>Emotion Synchrony Report</h1>
<div class="meta">
    A: <strong>{a_label}</strong> ({a_dur:.0f}s) &bull;
    B: <strong>{b_label}</strong> ({b_dur:.0f}s) &bull;
    Alignment: {align} &bull;
    Overlap: {overlap:.0f}s ({n_points} pts) &bull;
    Generated: {report_ts}
</div>

<div class="kpi">
    <div class="kpi-box">
        <div class="val" style="color:#ffaa00">{comp_r:.3f}</div>
        <div class="lbl">Mean Pearson r</div>
    </div>
    <div class="kpi-box">
        <div class="val" style="color:#00cc00">{comp_mi:.3f}</div>
        <div class="lbl">Mutual Information</div>
    </div>
    <div class="kpi-box">
        <div class="val" style="color:#4488ff">{comp_plv:.3f}</div>
        <div class="lbl">Phase Locking</div>
    </div>
    <div class="kpi-box">
        <div class="val" style="color:#cc0000">{d_match:.0%}</div>
        <div class="lbl">Dominance Agreement</div>
    </div>
    <div class="kpi-box">
        <div class="val" style="color:#6600cc">{gc_ab}<span style="font-size:16px">/7</span></div>
        <div class="lbl">Granger A→B</div>
    </div>
    <div class="kpi-box">
        <div class="val" style="color:#006600">{gc_ba}<span style="font-size:16px">/7</span></div>
        <div class="lbl">Granger B→A</div>
    </div>
</div>

<div class="card" style="margin-bottom:20px">
    <h2>Per-Emotion Metrics</h2>
    <table>
    <tr><th>Emotion</th><th>Pearson r</th><th>Best Lag</th><th>Lag r</th>
        <th>MI</th><th>PLV</th><th>GC A→B</th><th>GC B→A</th></tr>
    {table_rows}
    </table>
    <p style="margin-top:8px;font-size:11px;color:#888">
    * p&lt;.05 &nbsp; MI = Mutual Information (0–1) &nbsp;
    PLV = Phase Locking Value (0–1) &nbsp; GC = Granger Causality F-stat
    </p>
</div>

<div class="grid">
    <div class="card">
        <h2>Pearson r by Emotion</h2>
        <div id="bar-pearson" class="chart"></div>
    </div>
    <div class="card">
        <h2>Mutual Information &amp; PLV</h2>
        <div id="bar-mi-plv" class="chart"></div>
    </div>
    <div class="card full">
        <h2>Running Correlation (10s window)</h2>
        <div id="rc" class="timeline-chart"></div>
    </div>
    <div class="card emotion-grid">
        <h2>Emotion-by-Emotion Comparison (A= solid, B= dotted)</h2>
        <div id="emotion-pairs" class="emotion-pairs"></div>
    </div>
    <div class="card full">
        <h2>Emotion Intensity Overlay (happy + sad)</h2>
        <div id="tl" class="timeline-chart"></div>
    </div>
</div>

<div class="footer">
    HSEmotion (EfficientNet-B0, AffectNet) &bull; Pearson r, Cross-correlation,
    Mutual Information, Phase Locking, Granger Causality, Running Correlation
</div>
</div>

<script>
const emo_labels = {json.dumps(EMOTIONS)};
const emo_colors = {json.dumps([EMOTION_COLORS[e] for e in EMOTIONS])};

Plotly.newPlot('bar-pearson', [{{
    x: emo_labels, y: {json.dumps(bar_pearson)}, type: 'bar',
    marker: {{color: emo_colors}},
    text: {json.dumps([f"{v:.3f}" for v in bar_pearson])}, textposition: 'auto',
}}], {{
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{color: '#aaa', size: 12}},
    yaxis: {{title: 'Pearson r', color: '#aaa', gridcolor: '#2a3a5a', range: [-1, 1]}},
    margin: {{t: 10, b: 50, l: 50, r: 10}},
}});

Plotly.newPlot('bar-mi-plv', [
    {{ x: emo_labels, y: {json.dumps(bar_mi)}, type: 'bar',
       name: 'Mutual Information', marker: {{color: '#00cc00'}}, opacity: 0.7 }},
    {{ x: emo_labels, y: {json.dumps(bar_plv)}, type: 'bar',
       name: 'Phase Locking', marker: {{color: '#4488ff'}}, opacity: 0.7 }}
], {{
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{color: '#aaa', size: 12}}, barmode: 'group',
    yaxis: {{title: 'Value', color: '#aaa', gridcolor: '#2a3a5a', range: [0, 1]}},
    margin: {{t: 10, b: 50, l: 50, r: 10}},
    legend: {{orientation: 'h', y: 1.1}},
}});

Plotly.newPlot('rc', [{rc_js.rstrip(',')}], {{
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{color: '#aaa', size: 11}},
    xaxis: {{title: 'Time (s)', color: '#aaa', gridcolor: '#2a3a5a'}},
    yaxis: {{title: 'Pearson r', color: '#aaa', gridcolor: '#2a3a5a', range: [-1, 1]}},
    margin: {{t: 10, b: 40, l: 50, r: 10}},
    showlegend: true, legend: {{orientation: 'h', y: 1.1}},
    shapes: [{{type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 0, y1: 0,
              line: {{color: '#555', dash: 'dash'}}}}],
}});

Plotly.newPlot('tl', [{tl_js.rstrip(',')}], {{
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{color: '#aaa', size: 11}},
    xaxis: {{title: 'Time (s)', color: '#aaa', gridcolor: '#2a3a5a'}},
    yaxis: {{title: 'Score', color: '#aaa', gridcolor: '#2a3a5a'}},
    margin: {{t: 10, b: 40, l: 50, r: 10}},
    showlegend: true, legend: {{orientation: 'h', y: 1.1}},
}});

// --- Emotion-by-emotion 7-panel comparison ---
const traces_ep = [{emotion_pairs_js.rstrip(',')}];

const ep_layout = {{
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{color: '#aaa', size: 10}},
    height: 850,
    grid: {{rows: 7, columns: 1, pattern: 'independent'}},
    margin: {{t: 10, b: 30, l: 50, r: 30}},
    showlegend: true,
    legend: {{orientation: 'h', y: 1.02, font: {{size: 11}}}},
}};
""" + "".join(
    """ep_layout['xaxis""" + str(i+1) + """'] = {title: '""" + (emo if i == 6 else """""") + """', color: '#aaa', gridcolor: '#2a3a5a'};
ep_layout['yaxis""" + str(i+1) + """'] = {color: '""" + EMOTION_COLORS[emo] + """', range: [0, 100], gridcolor: '#2a3a5a', title: '""" + ("%" if i == 3 else """""") + """'};
""" for i, emo in enumerate(EMOTIONS)) + """

Plotly.newPlot('emotion-pairs', traces_ep, ep_layout);
</script>
</body>
</html>"""


def _gc_p(F, n_points):
    """Helper: approximate GC p-value from F stat."""
    return _f_cdf_approx(F, 5, n_points - 11) if not np.isnan(F) else 1.0


def _f_cdf_approx(F, df1, df2):
    """Approximate upper-tail p-value of F-distribution."""
    if F <= 0 or np.isnan(F) or np.isinf(F):
        return 1.0
    p = np.exp(-F * df1 / (2 * (1 + F * df1 / df2)))
    return min(1.0, max(0.0, float(p)))


if __name__ == "__main__":
    main()
