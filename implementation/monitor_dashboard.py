"""Standard-library web dashboard for Raspberry Pi monitor & model benchmarking."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IoT Edge DDoS Detector & Perception Monitor</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 24px; background: #0f172a; color: #f8fafc; }
  main { max-width: 1100px; margin: auto; }
  h1 { color: #38bdf8; margin-top: 0; display: flex; align-items: center; justify-content: space-between; }
  .badge-edge { background: #0369a1; color: #e0f2fe; font-size: 14px; padding: 4px 10px; border-radius: 9999px; font-weight: 500; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .card { background: #1e293b; padding: 18px; border: 1px solid #334155; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
  .card-title { font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 6px; }
  .value { font-size: 32px; font-weight: 800; }
  
  .act-ALLOW { color: #4ade80; }
  .act-RESTRICT { color: #facc15; }
  .act-BLOCK { color: #f87171; }
  
  .section { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 20px; margin-bottom: 24px; }
  .section h2 { margin-top: 0; color: #e2e8f0; font-size: 18px; border-bottom: 1px solid #334155; padding-bottom: 10px; }
  
  table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }
  th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #334155; }
  th { color: #94a3b8; font-weight: 600; background: #0f172a; }
  tr:hover { background: #334155; }
  
  .action-badge { padding: 4px 8px; border-radius: 4px; font-weight: 700; font-size: 12px; display: inline-block; }
  .bg-ALLOW { background: #166534; color: #86efac; }
  .bg-RESTRICT { background: #854d0e; color: #fef08a; }
  .bg-BLOCK { background: #991b1b; color: #fca5a5; }

  @media(max-width: 768px) { .grid { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>
<main>
  <h1>
    <span>IoT Edge DDoS Perception Monitor</span>
    <span class="badge-edge">Raspberry Pi Edge Defense</span>
  </h1>

  <div class="grid">
    <div class="card">
      <div class="card-title">Packets / Window</div>
      <div id="packets" class="value">0</div>
    </div>
    <div class="card">
      <div class="card-title">Model Confidence Score</div>
      <div id="score" class="value">0.000</div>
    </div>
    <div class="card">
      <div class="card-title">Fuzzy Rule Perception</div>
      <div id="fuzzy" class="value">0.000</div>
    </div>
    <div class="card">
      <div class="card-title">Edge Mitigation Action</div>
      <div id="action" class="value act-ALLOW">ALLOW</div>
    </div>
  </div>

  <div class="section">
    <h2>Model Comparison Benchmark (Raspberry Pi Resource Efficiency)</h2>
    <div id="comparison-container">Loading benchmark metrics...</div>
  </div>

  <div class="section">
    <h2>Live Packet Perception & Action Log</h2>
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Source IP</th>
          <th>SYN/ACK Ratio</th>
          <th>Half-Open</th>
          <th>SYN Density</th>
          <th>Fuzzy Score</th>
          <th>Model Prob</th>
          <th>Mitigation Action</th>
        </tr>
      </thead>
      <tbody id="rows">
        <tr><td colspan="8">Waiting for live detector events...</td></tr>
      </tbody>
    </table>
  </div>
</main>

<script>
async function loadBenchmark() {
  try {
    const res = await fetch('/benchmark');
    const data = await res.json();
    if (!data || Object.keys(data).length === 0) return;
    
    let html = '<table><thead><tr><th>Model Architecture</th><th>Accuracy</th><th>Recall</th><th>F1 Score</th><th>ROC AUC</th><th>Latency (ms / 1k pkts)</th><th>Size (KB)</th><th>Edge Feasibility</th></tr></thead><tbody>';
    for (const [model, m] of Object.entries(data)) {
      html += `<tr>
        <td><strong>${model}</strong></td>
        <td>${(m['Accuracy']*100).toFixed(2)}%</td>
        <td>${(m['Recall']*100).toFixed(2)}%</td>
        <td>${(m['F1 Score']*100).toFixed(2)}%</td>
        <td>${m['ROC AUC'].toFixed(3)}</td>
        <td>${m['Inference Latency (ms / 1k pkts)'].toFixed(2)} ms</td>
        <td>${m['Model Size (KB)']} KB</td>
        <td><span class="action-badge ${model.includes('Fuzzy') ? 'bg-ALLOW' : (model.includes('Deep') ? 'bg-RESTRICT' : 'bg-ALLOW')}">${m['Edge Compatibility']}</span></td>
      </tr>`;
    }
    html += '</tbody></table>';
    document.getElementById('comparison-container').innerHTML = html;
  } catch(e) {
    document.getElementById('comparison-container').innerHTML = 'Run <code>compare_models.py</code> to view live comparative benchmark metrics.';
  }
}

async function refreshEvents() {
  try {
    const r = await fetch('/events');
    const e = await r.json();
    const last = e[e.length - 1];
    if (last) {
      document.getElementById('packets').textContent = last.packets_in_window || 0;
      document.getElementById('score').textContent = (last.score || 0).toFixed(3);
      document.getElementById('fuzzy').textContent = (last.fuzzy_confidence || 0).toFixed(3);
      
      const actElem = document.getElementById('action');
      actElem.textContent = last.action || 'ALLOW';
      actElem.className = 'value act-' + (last.action || 'ALLOW');
    }
    
    const rowsElem = document.getElementById('rows');
    if (e.length > 0) {
      rowsElem.innerHTML = e.slice(-25).reverse().map(x => {
        const s = x.stats || {};
        return `<tr>
          <td>${new Date((x.timestamp || 0) * 1000).toLocaleTimeString()}</td>
          <td><code>${x.source}</code></td>
          <td>${s.syn_ack_ratio ?? '-'}</td>
          <td>${s.half_open_conn_count ?? '-'}</td>
          <td>${s.syn_packet_density ?? '-'} pkts/s</td>
          <td>${(x.fuzzy_confidence || 0).toFixed(3)}</td>
          <td>${(x.model_probability || 0).toFixed(3)}</td>
          <td><span class="action-badge bg-${x.action}">${x.action}</span></td>
        </tr>`;
      }).join('');
    }
  } catch(e) {}
}

loadBenchmark();
refreshEvents();
setInterval(refreshEvents, 1000);
</script>
</body>
</html>"""


def make_handler(log_path: Path, benchmark_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/events":
                events = []
                if log_path.exists():
                    lines = log_path.read_text(encoding="utf-8").splitlines()
                    events = [json.loads(line) for line in lines[-100:] if line.strip()]
                body = json.dumps(events).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            
            if self.path == "/benchmark":
                bench = {}
                if benchmark_path.exists():
                    bench = json.loads(benchmark_path.read_text(encoding="utf-8"))
                body = json.dumps(bench).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="IoT Edge Flood Monitor Dashboard")
    parser.add_argument("--log", default="reports/monitor_events.jsonl")
    parser.add_argument("--benchmark", default="reports/model_comparison_report.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    # Fallback to local root if not found in reports/
    log_p = Path(args.log) if Path(args.log).exists() else (Path("monitor_events.jsonl") if Path("monitor_events.jsonl").exists() else Path(args.log))
    bench_p = Path(args.benchmark) if Path(args.benchmark).exists() else (Path("model_comparison_report.json") if Path("model_comparison_report.json").exists() else Path(args.benchmark))

    server = ThreadingHTTPServer((args.host, args.port), make_handler(log_p, bench_p))
    print(f"IoT Edge Flood Monitor running on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()