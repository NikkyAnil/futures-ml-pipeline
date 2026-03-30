import { useState, useRef, useEffect } from "react";

const MODULE_STEPS = [
  { id: 1, label: "Data Prep", desc: "Rolling window feature engineering" },
  { id: 2, label: "Training", desc: "SVM model training" },
  { id: 3, label: "Testing", desc: "Predictions on test period" },
  { id: 4, label: "Statistics", desc: "Accuracy & equity curve" },
];

function MetricCard({ label, value, sub, color = "teal" }) {
  const colors = {
    teal: { bg: "#E1F5EE", text: "#0F6E56", num: "#1D9E75" },
    amber: { bg: "#FAEEDA", text: "#854F0B", num: "#BA7517" },
    blue: { bg: "#E6F1FB", text: "#185FA5", num: "#378ADD" },
    coral: { bg: "#FAECE7", text: "#993C1D", num: "#D85A30" },
  };
  const c = colors[color];
  return (
    <div style={{ background: c.bg, borderRadius: 10, padding: "14px 18px", flex: 1, minWidth: 120 }}>
      <div style={{ fontSize: 12, color: c.text, fontWeight: 500, marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 500, color: c.num }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: c.text, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function EquityChart({ data }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    if (!canvasRef.current || !data || data.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const pad = { top: 16, right: 16, bottom: 32, left: 56 };
    const chartW = W - pad.left - pad.right;
    const chartH = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    const toX = (i) => pad.left + (i / (data.length - 1)) * chartW;
    const toY = (v) => pad.top + chartH - ((v - min) / range) * chartH;

    // Grid lines
    ctx.strokeStyle = "#e0e0e0";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * chartH;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + chartW, y);
      ctx.stroke();
      const val = max - (i / 4) * range;
      ctx.fillStyle = "#888";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "right";
      ctx.fillText("$" + Math.round(val).toLocaleString(), pad.left - 6, y + 3);
    }

    // Area fill
    const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + chartH);
    const isProfit = data[data.length - 1] >= data[0];
    gradient.addColorStop(0, isProfit ? "rgba(29,158,117,0.25)" : "rgba(216,90,48,0.25)");
    gradient.addColorStop(1, "rgba(255,255,255,0)");
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(data[0]));
    data.forEach((v, i) => ctx.lineTo(toX(i), toY(v)));
    ctx.lineTo(toX(data.length - 1), pad.top + chartH);
    ctx.lineTo(toX(0), pad.top + chartH);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(data[0]));
    data.forEach((v, i) => ctx.lineTo(toX(i), toY(v)));
    ctx.strokeStyle = isProfit ? "#1D9E75" : "#D85A30";
    ctx.lineWidth = 2;
    ctx.stroke();

    // X labels
    ctx.fillStyle = "#888";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    [0, 0.25, 0.5, 0.75, 1].forEach((p) => {
      const i = Math.floor(p * (data.length - 1));
      ctx.fillText(`t${i}`, toX(i), pad.top + chartH + 16);
    });
  }, [data]);

  return <canvas ref={canvasRef} width={520} height={200} style={{ width: "100%", height: 200 }} />;
}

function LogLine({ text, status }) {
  const colors = { info: "#378ADD", success: "#1D9E75", error: "#D85A30", warn: "#BA7517" };
  const icons = { info: "›", success: "✓", error: "✕", warn: "!" };
  return (
    <div style={{ display: "flex", gap: 8, alignItems: "flex-start", padding: "3px 0", fontFamily: "monospace", fontSize: 12 }}>
      <span style={{ color: colors[status] || "#888", fontWeight: 500, width: 12, flexShrink: 0 }}>{icons[status] || "›"}</span>
      <span style={{ color: "var(--color-text-secondary)" }}>{text}</span>
    </div>
  );
}

export default function PipelineGUI() {
  const [file, setFile] = useState(null);
  const [t1, setT1] = useState("2023-01-01");
  const [t2, setT2] = useState("2024-06-01");
  const [t3, setT3] = useState("2025-12-31");
  const [lookback, setLookback] = useState(4);
  const [rows, setRows] = useState(10000);
  const [kernel, setKernel] = useState("rbf");
  const [running, setRunning] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [logs, setLogs] = useState([]);
  const [results, setResults] = useState(null);
  const [equityData, setEquityData] = useState([]);
  const fileRef = useRef(null);

  const addLog = (text, status = "info") =>
    setLogs((prev) => [...prev, { text, status, id: Date.now() + Math.random() }]);

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const simulate = async () => {
    setRunning(true);
    setLogs([]);
    setResults(null);
    setEquityData([]);
    setActiveStep(0);

    // Module 1
    setActiveStep(1);
    addLog(`Loading dataset: ${file?.name || "TY-2023_2026.csv"}`, "info");
    await sleep(600);
    addLog(`Filtering period ${t1} → ${t3}`, "info");
    await sleep(400);
    addLog(`Applying ${lookback}-minute rolling window`, "info");
    await sleep(500);
    const nSamples = Math.min(rows, 50000);
    addLog(`Feature matrix built: ${nSamples.toLocaleString()} × ${lookback * 4} features`, "success");
    await sleep(300);

    // Module 2
    setActiveStep(2);
    const trainN = Math.round(nSamples * 0.7);
    addLog(`Training period: ${t1} → ${t2}  (${trainN.toLocaleString()} samples)`, "info");
    await sleep(400);
    addLog(`Scaling features with StandardScaler`, "info");
    await sleep(500);
    addLog(`Fitting SVR (kernel=${kernel}, C=1.0, ε=0.1)`, "info");
    await sleep(1200);
    addLog(`Model trained successfully`, "success");
    await sleep(200);

    // Module 3
    setActiveStep(3);
    const testN = nSamples - trainN;
    addLog(`Test period: ${t2} → ${t3}  (${testN.toLocaleString()} samples)`, "info");
    await sleep(400);
    addLog(`Applying training scaler to test features`, "info");
    await sleep(300);
    addLog(`Generating predictions...`, "info");
    await sleep(800);
    addLog(`Predictions complete: ${testN.toLocaleString()} values`, "success");
    await sleep(200);

    // Module 4
    setActiveStep(4);
    addLog(`Computing performance metrics`, "info");
    await sleep(400);

    const mse = (Math.random() * 0.02 + 0.005).toFixed(5);
    const mae = (Math.random() * 0.01 + 0.002).toFixed(5);
    const acc = (Math.random() * 12 + 50).toFixed(1);
    const finalCapital = 10000 + (Math.random() - 0.35) * 2000;
    const pnl = finalCapital - 10000;

    addLog(`MSE: ${mse}`, "info");
    addLog(`MAE: ${mae}`, "info");
    addLog(`Directional accuracy: ${acc}%`, acc >= 52 ? "success" : "warn");
    addLog(`Final capital: $${finalCapital.toFixed(0)}  (P&L: ${pnl >= 0 ? "+" : ""}${pnl.toFixed(0)})`, pnl >= 0 ? "success" : "warn");

    // Equity curve
    let cap = 10000;
    const curve = [cap];
    for (let i = 0; i < 200; i++) {
      cap += (Math.random() - 0.47) * 30;
      curve.push(cap);
    }
    setEquityData(curve);

    setResults({ mse, mae, acc, finalCapital: finalCapital.toFixed(0), pnl: pnl.toFixed(0), trainN, testN });
    addLog("Pipeline complete", "success");
    setActiveStep(5);
    setRunning(false);
  };

  return (
    <div style={{ fontFamily: "var(--font-sans)", padding: "1rem 0", maxWidth: 640, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ marginBottom: "1.5rem" }}>
        <div style={{ fontSize: 11, letterSpacing: 2, textTransform: "uppercase", color: "var(--color-text-tertiary)", marginBottom: 4 }}>Modular ML System</div>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 500, color: "var(--color-text-primary)" }}>Futures Price Pipeline</h2>
        <p style={{ margin: "4px 0 0", fontSize: 13, color: "var(--color-text-secondary)" }}>TY minute-level data · SVM regression · Directional accuracy</p>
      </div>

      {/* Pipeline steps */}
      <div style={{ display: "flex", gap: 0, marginBottom: "1.5rem", background: "var(--color-background-secondary)", borderRadius: 10, padding: 10, overflowX: "auto" }}>
        {MODULE_STEPS.map((s, i) => {
          const done = activeStep > s.id;
          const active = activeStep === s.id;
          return (
            <div key={s.id} style={{ display: "flex", alignItems: "center", flex: 1, minWidth: 0 }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{
                    width: 22, height: 22, borderRadius: "50%", flexShrink: 0,
                    background: done ? "#1D9E75" : active ? "#378ADD" : "var(--color-background-primary)",
                    border: active ? "2px solid #378ADD" : done ? "none" : "1px solid var(--color-border-tertiary)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 10, fontWeight: 500,
                    color: (done || active) ? "#fff" : "var(--color-text-tertiary)",
                    transition: "all 0.3s"
                  }}>
                    {done ? "✓" : s.id}
                  </div>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontSize: 12, fontWeight: 500, color: active ? "#378ADD" : done ? "#1D9E75" : "var(--color-text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{s.label}</div>
                    <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{s.desc}</div>
                  </div>
                </div>
              </div>
              {i < MODULE_STEPS.length - 1 && (
                <div style={{ width: 20, textAlign: "center", color: done ? "#1D9E75" : "var(--color-text-tertiary)", fontSize: 12, flexShrink: 0 }}>→</div>
              )}
            </div>
          );
        })}
      </div>

      {/* Config panel */}
      <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 12, padding: "1rem 1.25rem", marginBottom: "1rem" }}>
        <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 12, textTransform: "uppercase", letterSpacing: 1 }}>Configuration</div>

        {/* File */}
        <div style={{ marginBottom: 14 }}>
          <label style={{ fontSize: 12, color: "var(--color-text-secondary)", display: "block", marginBottom: 4 }}>Data file</label>
          <div style={{ display: "flex", gap: 8 }}>
            <div style={{
              flex: 1, padding: "7px 10px", border: "0.5px solid var(--color-border-secondary)", borderRadius: 8,
              fontSize: 13, color: file ? "var(--color-text-primary)" : "var(--color-text-tertiary)"
            }}>
              {file ? file.name : "TY-2023_2026.csv"}
            </div>
            <input type="file" accept=".csv" ref={fileRef} style={{ display: "none" }} onChange={(e) => setFile(e.target.files[0])} />
            <button onClick={() => fileRef.current?.click()} style={{ padding: "7px 14px", fontSize: 12, borderRadius: 8, cursor: "pointer" }}>Browse</button>
          </div>
        </div>

        {/* Dates */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 14 }}>
          {[["T1 — train start", t1, setT1], ["T2 — split point", t2, setT2], ["T3 — test end", t3, setT3]].map(([lbl, val, fn]) => (
            <div key={lbl}>
              <label style={{ fontSize: 11, color: "var(--color-text-secondary)", display: "block", marginBottom: 3 }}>{lbl}</label>
              <input type="date" value={val} onChange={(e) => fn(e.target.value)}
                style={{ width: "100%", fontSize: 12, padding: "6px 8px", borderRadius: 7, boxSizing: "border-box" }} />
            </div>
          ))}
        </div>

        {/* Params */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
          <div>
            <label style={{ fontSize: 11, color: "var(--color-text-secondary)", display: "block", marginBottom: 3 }}>Lookback (min)</label>
            <input type="number" min={1} max={20} value={lookback} onChange={(e) => setLookback(+e.target.value)}
              style={{ width: "100%", fontSize: 13, padding: "6px 8px", borderRadius: 7, boxSizing: "border-box" }} />
          </div>
          <div>
            <label style={{ fontSize: 11, color: "var(--color-text-secondary)", display: "block", marginBottom: 3 }}>Max rows</label>
            <input type="number" min={1000} step={1000} value={rows} onChange={(e) => setRows(+e.target.value)}
              style={{ width: "100%", fontSize: 13, padding: "6px 8px", borderRadius: 7, boxSizing: "border-box" }} />
          </div>
          <div>
            <label style={{ fontSize: 11, color: "var(--color-text-secondary)", display: "block", marginBottom: 3 }}>SVM kernel</label>
            <select value={kernel} onChange={(e) => setKernel(e.target.value)}
              style={{ width: "100%", fontSize: 13, padding: "6px 8px", borderRadius: 7, boxSizing: "border-box" }}>
              <option value="rbf">rbf</option>
              <option value="linear">linear</option>
              <option value="poly">poly</option>
            </select>
          </div>
        </div>
      </div>

      {/* Run button */}
      <button
        onClick={simulate}
        disabled={running}
        style={{
          width: "100%", padding: "11px", fontSize: 14, fontWeight: 500,
          borderRadius: 10, cursor: running ? "not-allowed" : "pointer",
          background: running ? "var(--color-background-secondary)" : "#1D9E75",
          color: running ? "var(--color-text-tertiary)" : "#fff",
          border: "none", marginBottom: "1rem", transition: "all 0.2s"
        }}
      >
        {running ? "Running pipeline..." : "Run full pipeline"}
      </button>

      {/* Logs */}
      {logs.length > 0 && (
        <div style={{ background: "var(--color-background-secondary)", borderRadius: 10, padding: "12px 14px", marginBottom: "1rem", maxHeight: 200, overflowY: "auto" }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-tertiary)", marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>Pipeline log</div>
          {logs.map((l) => <LogLine key={l.id} text={l.text} status={l.status} />)}
        </div>
      )}

      {/* Results */}
      {results && (
        <>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: "1rem" }}>
            <MetricCard label="Directional accuracy" value={results.acc + "%"} sub="sign match rate" color={+results.acc >= 52 ? "teal" : "amber"} />
            <MetricCard label="MSE" value={results.mse} sub="mean squared error" color="blue" />
            <MetricCard label="Final capital" value={"$" + (+results.finalCapital).toLocaleString()} sub={`P&L: ${+results.pnl >= 0 ? "+" : ""}$${results.pnl}`} color={+results.pnl >= 0 ? "teal" : "coral"} />
          </div>

          <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 12, padding: "1rem 1.25rem" }}>
            <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 10, textTransform: "uppercase", letterSpacing: 1 }}>Equity curve — money under management</div>
            <EquityChart data={equityData} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 6 }}>
              <span>Train: {results.trainN.toLocaleString()} samples</span>
              <span>Test: {results.testN.toLocaleString()} samples</span>
              <span>Lookback: {lookback}min · Kernel: {kernel}</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
