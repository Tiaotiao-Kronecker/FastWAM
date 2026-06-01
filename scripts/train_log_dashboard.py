import argparse
import json
import time
from pathlib import Path

from tail_train_log_to_tensorboard import iter_blocks_from_text, parse_block


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FastWAM A1 Training Dashboard</title>
  <style>
    body { margin: 0; font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; background: #f6f7f9; color: #172033; }
    header { padding: 18px 22px; background: #102033; color: white; }
    h1 { margin: 0; font-size: 20px; }
    main { max-width: 1280px; margin: 0 auto; padding: 18px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }
    .card, .chart { background: white; border: 1px solid #dfe5ee; border-radius: 8px; padding: 12px; }
    .card .label { color: #667085; font-size: 12px; }
    .card .value { font-size: 22px; font-weight: 700; margin-top: 2px; }
    .charts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .chart h2 { margin: 0 0 8px; font-size: 15px; }
    canvas { width: 100%; height: 270px; display: block; }
    .status { color: #cbd5e1; margin-top: 4px; font-size: 13px; }
    .legend { display: flex; gap: 12px; flex-wrap: wrap; color: #475467; font-size: 12px; margin-top: 6px; }
    .dot { display: inline-block; width: 9px; height: 9px; border-radius: 999px; margin-right: 5px; }
  </style>
</head>
<body>
  <header>
    <h1>FastWAM A1 Training Dashboard</h1>
    <div id="status" class="status">loading...</div>
  </header>
  <main>
    <div class="grid" id="cards"></div>
    <div class="charts">
      <div class="chart"><h2>Loss</h2><canvas id="loss"></canvas><div class="legend" id="lossLegend"></div></div>
      <div class="chart"><h2>Velocity RMS</h2><canvas id="rms"></canvas><div class="legend" id="rmsLegend"></div></div>
      <div class="chart"><h2>MeanFlow Sampling</h2><canvas id="sampling"></canvas><div class="legend" id="samplingLegend"></div></div>
      <div class="chart"><h2>Derivative / Speed</h2><canvas id="speed"></canvas><div class="legend" id="speedLegend"></div></div>
    </div>
  </main>
<script>
const COLORS = ["#1769aa", "#d92d20", "#12b76a", "#7a5af8", "#f79009"];

function fmt(v) {
  if (v === undefined || v === null || Number.isNaN(v)) return "-";
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(2);
  if (Math.abs(v) >= 1) return v.toFixed(3);
  return v.toPrecision(3);
}

function metric(record, key) {
  return record && record.metrics ? record.metrics[key] : undefined;
}

function drawLegend(el, series) {
  el.innerHTML = series.map((s, i) => `<span><span class="dot" style="background:${COLORS[i % COLORS.length]}"></span>${s.label}</span>`).join("");
}

function drawChart(canvas, records, series) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  const pad = {l: 48, r: 14, t: 12, b: 28};
  const points = [];
  for (const s of series) {
    for (const r of records) {
      const y = metric(r, s.key);
      if (typeof y === "number" && Number.isFinite(y)) points.push({x: r.step, y});
    }
  }
  if (!points.length) return;
  const xMin = Math.min(...points.map(p => p.x));
  const xMax = Math.max(...points.map(p => p.x));
  let yMin = Math.min(...points.map(p => p.y));
  let yMax = Math.max(...points.map(p => p.y));
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad; yMax += yPad;
  const xOf = x => pad.l + (x - xMin) / Math.max(1, xMax - xMin) * (w - pad.l - pad.r);
  const yOf = y => h - pad.b - (y - yMin) / Math.max(1e-12, yMax - yMin) * (h - pad.t - pad.b);

  ctx.strokeStyle = "#d0d5dd";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();

  ctx.fillStyle = "#667085";
  ctx.font = "12px system-ui";
  ctx.fillText(fmt(yMax), 4, pad.t + 4);
  ctx.fillText(fmt(yMin), 4, h - pad.b);
  ctx.fillText(String(xMin), pad.l, h - 8);
  ctx.fillText(String(xMax), w - pad.r - 44, h - 8);

  series.forEach((s, i) => {
    const values = records
      .map(r => ({x: r.step, y: metric(r, s.key)}))
      .filter(p => typeof p.y === "number" && Number.isFinite(p.y));
    if (!values.length) return;
    ctx.strokeStyle = COLORS[i % COLORS.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((p, idx) => {
      const x = xOf(p.x), y = yOf(p.y);
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

function render(data) {
  const records = data.records || [];
  const latest = records[records.length - 1] || {metrics: {}};
  const cards = [
    ["step", latest.step],
    ["loss", metric(latest, "loss")],
    ["meanflow target", metric(latest, "loss_meanflow_target")],
    ["anchor loss", metric(latest, "loss_equal_time_velocity")],
    ["lr", metric(latest, "lr")],
    ["steps/s", metric(latest, "speed")],
    ["pred rms", metric(latest, "pred_mean_velocity_rms")],
    ["target rms", metric(latest, "target_meanflow_rms")],
  ];
  document.getElementById("cards").innerHTML = cards.map(([k,v]) =>
    `<div class="card"><div class="label">${k}</div><div class="value">${fmt(v)}</div></div>`).join("");
  document.getElementById("status").textContent =
    `records=${records.length} updated=${new Date((data.updated_at || 0) * 1000).toLocaleString()} source=${data.source || ""}`;

  const configs = {
    loss: [
      {key: "loss", label: "loss"},
      {key: "loss_meanflow_target", label: "meanflow"},
      {key: "loss_equal_time_velocity", label: "anchor"},
    ],
    rms: [
      {key: "pred_mean_velocity_rms", label: "pred"},
      {key: "target_meanflow_rms", label: "target"},
    ],
    sampling: [
      {key: "meanflow_sigma_start", label: "sigma_start"},
      {key: "meanflow_sigma_end", label: "sigma_end"},
      {key: "meanflow_interval", label: "interval"},
    ],
    speed: [
      {key: "meanflow_dudt_rms", label: "dudt_rms"},
      {key: "speed", label: "steps/s"},
    ],
  };
  for (const [id, series] of Object.entries(configs)) {
    drawChart(document.getElementById(id), records, series);
    drawLegend(document.getElementById(id + "Legend"), series);
  }
}

async function refresh() {
  try {
    const res = await fetch(`metrics.json?t=${Date.now()}`);
    render(await res.json());
  } catch (err) {
    document.getElementById("status").textContent = `load failed: ${err}`;
  }
}

refresh();
setInterval(refresh, 10000);
window.addEventListener("resize", refresh);
</script>
</body>
</html>
"""


def parse_records(log_file: Path) -> list[dict]:
    text = log_file.read_text(encoding="utf-8", errors="replace")
    records = []
    seen_steps = set()
    for block in iter_blocks_from_text(text):
        step, total, metrics = parse_block(block)
        if step is None or step in seen_steps:
            continue
        seen_steps.add(step)
        records.append({"step": step, "total": total, "metrics": metrics})
    records.sort(key=lambda item: item["step"])
    return records


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os_replace(tmp, path)


def os_replace(src: Path, dst: Path) -> None:
    src.replace(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight HTML dashboard from a FastWAM training log.")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    log_file = Path(args.log_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(HTML, encoding="utf-8")
    json_path = out_dir / "metrics.json"

    while True:
        if log_file.exists():
            records = parse_records(log_file)
            write_json_atomic(
                json_path,
                {
                    "source": str(log_file),
                    "updated_at": time.time(),
                    "records": records,
                },
            )
            print(f"[dashboard] records={len(records)} last_step={records[-1]['step'] if records else None}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
