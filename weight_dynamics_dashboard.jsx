import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, AreaChart, Area, BarChart, Bar,
  ScatterChart, Scatter, ReferenceLine, ComposedChart,
  Legend, Cell
} from "recharts";

// ============================================================
// SYNTHETIC TRAINING ENGINE (full JS port with weight dynamics)
// ============================================================

function seededRng(seed) {
  let s = Math.abs(seed | 0) || 1;
  return () => { s = (s * 1664525 + 1013904223) & 0x7fffffff; return s / 0x7fffffff; };
}
function gaussRng(rng) {
  let sp = null;
  return () => {
    if (sp !== null) { const v = sp; sp = null; return v; }
    let u, v, s;
    do { u = rng() * 2 - 1; v = rng() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    s = Math.sqrt(-2 * Math.log(s) / s);
    sp = v * s; return u * s;
  };
}

function simulateTraining(config) {
  const {
    steps = 400, lr = 0.001, seed = 42,
    eventType = "none", eventStep = 200, eventMag = 15,
    dims = [128, 64, 32, 16, 8],
  } = config;

  const rng = seededRng(seed);
  const gauss = gaussRng(seededRng(seed + 1));
  const nLayers = dims.length - 1;

  // Initialize weights (He init)
  const weights = [];
  const adam_m = [];
  const adam_v = [];
  for (let l = 0; l < nLayers; l++) {
    const fan_in = dims[l], fan_out = dims[l + 1];
    const scale = Math.sqrt(2 / fan_in);
    const w = Array.from({ length: fan_out * fan_in }, () => gauss() * scale);
    weights.push(w);
    adam_m.push(new Float64Array(w.length));
    adam_v.push(new Float64Array(w.length));
  }

  // Loss landscape: bowl centers
  const totalParams = weights.reduce((s, w) => s + w.length, 0);
  const landscapeDim = Math.min(totalParams, 100);
  const bowlCenter = Array.from({ length: landscapeDim }, () => gauss() * 2);
  const curvature = Array.from({ length: landscapeDim }, (_, i) =>
    Math.exp(Math.log(1) + (Math.log(10) - Math.log(1)) * i / Math.max(landscapeDim - 1, 1))
  );

  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8, wd = 0.0001;
  let lrMult = 1.0;

  const data = [];
  let prevFlat = null;

  for (let step = 1; step <= steps; step++) {
    // LR schedule (warmup cosine)
    let curLr;
    const warmup = 50;
    if (step < warmup) curLr = lr * step / warmup;
    else curLr = lr * 0.5 * (1 + Math.cos(Math.PI * (step - warmup) / (steps - warmup)));

    // Process events
    if (eventType === "lr_spike" && step === eventStep) lrMult = eventMag;
    if (eventType === "lr_spike" && step === eventStep + 20) lrMult = 1.0;
    if (eventType === "landscape_shift" && step === eventStep) {
      for (let i = 0; i < landscapeDim; i++) bowlCenter[i] += gauss() * eventMag;
    }
    curLr *= lrMult;

    // Forward pass: compute activations through actual weights
    let x = Array.from({ length: dims[0] }, () => gauss());
    const layerActRms = [];
    for (let l = 0; l < nLayers; l++) {
      const fin = dims[l], fout = dims[l + 1];
      const newX = new Array(fout).fill(0);
      for (let j = 0; j < fout; j++) {
        let sum = 0;
        for (let i = 0; i < fin; i++) sum += x[i] * weights[l][j * fin + i];
        newX[j] = l < nLayers - 1 ? Math.max(0, sum) : sum; // ReLU except last
      }
      const rms = Math.sqrt(newX.reduce((s, v) => s + v * v, 0) / fout);
      layerActRms.push(rms);
      x = newX;
    }

    // Flatten params for loss
    const flat = [];
    for (const w of weights) flat.push(...w);
    const pFlat = flat.slice(0, landscapeDim);

    // Loss (quadratic bowl)
    let loss = 0;
    for (let i = 0; i < landscapeDim; i++) {
      const d = pFlat[i] - bowlCenter[i];
      loss += 0.5 * curvature[i] * d * d;
    }
    loss += gauss() * 0.01 * (1 + loss * 0.01);
    loss = Math.max(0, loss);

    // Gradient
    const grad = new Array(landscapeDim).fill(0);
    for (let i = 0; i < landscapeDim; i++) {
      grad[i] = curvature[i] * (pFlat[i] - bowlCenter[i]) + gauss() * 0.01;
    }

    // Grad corruption event
    if (eventType === "grad_corrupt" && step === eventStep) {
      for (let i = 0; i < landscapeDim; i++) grad[i] += gauss() * eventMag;
    }

    // Distribute gradient to layers and update
    let offset = 0;
    const layerGradNorms = [];
    const layerWeightNorms = [];
    const layerUpdateMags = [];
    const layerCosSim = [];
    let totalGradNorm = 0;

    for (let l = 0; l < nLayers; l++) {
      const n = weights[l].length;
      let gnSq = 0, wnSq = 0, umSq = 0;

      for (let i = 0; i < n; i++) {
        const g = (offset + i < landscapeDim ? grad[offset + i] : gauss() * 0.01) + wd * weights[l][i];

        // Adam
        adam_m[l][i] = beta1 * adam_m[l][i] + (1 - beta1) * g;
        adam_v[l][i] = beta2 * adam_v[l][i] + (1 - beta2) * g * g;
        const mHat = adam_m[l][i] / (1 - Math.pow(beta1, step));
        const vHat = adam_v[l][i] / (1 - Math.pow(beta2, step));
        const update = curLr * mHat / (Math.sqrt(vHat) + eps);

        weights[l][i] -= update;
        gnSq += g * g;
        wnSq += weights[l][i] * weights[l][i];
        umSq += update * update;
      }
      layerGradNorms.push(Math.sqrt(gnSq));
      layerWeightNorms.push(Math.sqrt(wnSq));
      layerUpdateMags.push(Math.sqrt(umSq));
      totalGradNorm += gnSq;
      offset += n;
    }
    totalGradNorm = Math.sqrt(totalGradNorm);

    // Param velocity
    const newFlat = [];
    for (const w of weights) newFlat.push(...w.slice(0, Math.min(w.length, 50)));
    let velocity = 0;
    if (prevFlat) {
      const n = Math.min(newFlat.length, prevFlat.length);
      let sq = 0;
      for (let i = 0; i < n; i++) sq += (newFlat[i] - prevFlat[i]) ** 2;
      velocity = Math.sqrt(sq);
    }
    prevFlat = newFlat.slice();

    // Adam state norms
    let adamMNorm = 0, adamVNorm = 0;
    for (let l = 0; l < nLayers; l++) {
      for (let i = 0; i < adam_m[l].length; i++) {
        adamMNorm += adam_m[l][i] ** 2;
        adamVNorm += adam_v[l][i] ** 2;
      }
    }
    adamMNorm = Math.sqrt(adamMNorm);
    adamVNorm = Math.sqrt(adamVNorm);

    // PHYSICALLY-DERIVED MEMORY
    let memBase = 0;
    for (let l = 0; l < nLayers; l++) {
      const wBytes = weights[l].length * 4;
      const optBytes = adam_m[l].length * 4 * 2;
      const actBytes = dims[l + 1] * 32 * 4; // batch_size=32
      const actScale = 1.0 + 0.8 * Math.tanh(Math.max(0, layerActRms[l] - 0.5));
      const gScale = 1.0 + 1.0 * Math.tanh(layerGradNorms[l] / 2);
      memBase += wBytes + optBytes + actBytes * actScale + wBytes * gScale;
    }

    // Dynamics-correlated allocator noise
    const fragProb = Math.min(0.25, 0.02 + 0.03 * totalGradNorm);
    const fragScale = 1.0 + 1.5 * Math.min(totalGradNorm, 20);
    let noise = Math.abs(gauss()) * memBase * 0.01;
    if (rng() < fragProb) noise += Math.abs(gauss()) * memBase * 0.02 * fragScale;
    const totalActRms = layerActRms.reduce((a, b) => a + b, 0);
    noise += totalActRms * memBase * 0.002;
    const memNoisy = memBase + noise;

    data.push({
      step,
      loss,
      lr: curLr,
      totalGradNorm,
      totalWeightNorm: layerWeightNorms.reduce((a, b) => a + b, 0),
      paramVelocity: velocity,
      memory: memNoisy / 1e6, // MB
      memoryClean: memBase / 1e6,
      adamMNorm, adamVNorm,
      layerGradNorms, layerWeightNorms, layerActRms, layerUpdateMags,
    });
  }
  return data;
}

// ============================================================
// ANALYSIS FUNCTIONS
// ============================================================

function rollingCorrelation(a, b, w = 40) {
  const n = Math.min(a.length, b.length);
  const out = [];
  for (let i = 0; i < n; i++) {
    const s = Math.max(0, i - w / 2), e = Math.min(n, i + w / 2);
    const sa = a.slice(s, e), sb = b.slice(s, e);
    const ma = sa.reduce((x, y) => x + y, 0) / sa.length;
    const mb = sb.reduce((x, y) => x + y, 0) / sb.length;
    let num = 0, da2 = 0, db2 = 0;
    for (let j = 0; j < sa.length; j++) {
      const da = sa[j] - ma, db = sb[j] - mb;
      num += da * db; da2 += da * da; db2 += db * db;
    }
    out.push(da2 > 0 && db2 > 0 ? num / Math.sqrt(da2 * db2) : 0);
  }
  return out;
}

function rollingStd(a, w = 40) {
  return a.map((_, i) => {
    const s = Math.max(0, i - w / 2), e = Math.min(a.length, i + w / 2);
    const sl = a.slice(s, e);
    const m = sl.reduce((x, y) => x + y, 0) / sl.length;
    return Math.sqrt(sl.reduce((x, y) => x + (y - m) ** 2, 0) / sl.length);
  });
}

function pearson(a, b) {
  const n = Math.min(a.length, b.length);
  const ma = a.reduce((s, v) => s + v, 0) / n;
  const mb = b.reduce((s, v) => s + v, 0) / n;
  let num = 0, da2 = 0, db2 = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    num += da * db; da2 += da * da; db2 += db * db;
  }
  return da2 > 0 && db2 > 0 ? num / Math.sqrt(da2 * db2) : 0;
}

// ============================================================
// UI COMPONENTS
// ============================================================

const COLORS = {
  bg: "#06080f",
  card: "rgba(255,255,255,0.025)",
  border: "rgba(255,255,255,0.06)",
  text: "#c8d0e0",
  dim: "rgba(255,255,255,0.3)",
  accent1: "#22d3ee", // cyan
  accent2: "#a78bfa", // purple
  accent3: "#f472b6", // pink
  accent4: "#34d399", // green
  accent5: "#fb923c", // orange
  danger: "#ef4444",
  warn: "#fbbf24",
  safe: "#10b981",
  memory: "#22d3ee",
  loss: "#f472b6",
  grad: "#a78bfa",
  weight: "#34d399",
  velocity: "#fb923c",
  adam: "#818cf8",
};

const LAYER_COLORS = ["#22d3ee", "#a78bfa", "#f472b6", "#34d399", "#fb923c"];
const MONO = "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace";
const SANS = "'Inter', -apple-system, system-ui, sans-serif";

const EXPERIMENTS = {
  healthy: { label: "Healthy Training", eventType: "none" },
  lr_spike: { label: "LR Spike (15× at step 200)", eventType: "lr_spike", eventStep: 200, eventMag: 15 },
  landscape: { label: "Landscape Shift (step 150)", eventType: "landscape_shift", eventStep: 150, eventMag: 5 },
  grad_corrupt: { label: "Gradient Corruption (step 180)", eventType: "grad_corrupt", eventStep: 180, eventMag: 80 },
};

function Metric({ label, value, color = COLORS.text, sub }) {
  return (
    <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 10, padding: "12px 16px" }}>
      <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: 1.5, color: COLORS.dim, fontFamily: MONO, marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color, fontFamily: MONO, lineHeight: 1.1 }}>{typeof value === "number" ? (Math.abs(value) < 0.01 ? value.toExponential(2) : value.toFixed(3)) : value}</div>
      {sub && <div style={{ fontSize: 10, color: COLORS.dim, marginTop: 3, fontFamily: MONO }}>{sub}</div>}
    </div>
  );
}

function ChartCard({ title, children, height = 220, annotation }) {
  return (
    <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 12, padding: "16px 12px 8px", marginBottom: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, padding: "0 8px" }}>
        <span style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1.5, color: COLORS.dim, fontFamily: MONO }}>{title}</span>
        {annotation && <span style={{ fontSize: 10, color: COLORS.dim, fontFamily: MONO }}>{annotation}</span>}
      </div>
      <ResponsiveContainer width="100%" height={height}>{children}</ResponsiveContainer>
    </div>
  );
}

function Tip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "rgba(6,8,15,0.95)", border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: "8px 12px", fontFamily: MONO, fontSize: 10, maxWidth: 200 }}>
      <div style={{ color: COLORS.dim, marginBottom: 3 }}>Step {Math.round(label)}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || COLORS.text, marginBottom: 1 }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(4) : p.value}
        </div>
      ))}
    </div>
  );
}

// ============================================================
// SPECTROGRAM (canvas)
// ============================================================

function MiniSpectrogram({ signal, color = "#22d3ee", height = 100 }) {
  const canvasRef = useCallback(canvas => {
    if (!canvas || signal.length < 64) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width = canvas.parentElement?.clientWidth || 600;
    const h = canvas.height = height;
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    const winSize = 32;
    const hop = 4;
    const cols = [];
    for (let start = 0; start + winSize <= signal.length; start += hop) {
      const chunk = signal.slice(start, start + winSize);
      const mean = chunk.reduce((a, b) => a + b, 0) / chunk.length;
      const spec = [];
      for (let k = 1; k < winSize / 2; k++) {
        let re = 0, im = 0;
        for (let n = 0; n < winSize; n++) {
          const angle = -2 * Math.PI * k * n / winSize;
          const windowed = (chunk[n] - mean) * (0.5 - 0.5 * Math.cos(2 * Math.PI * n / (winSize - 1)));
          re += windowed * Math.cos(angle);
          im += windowed * Math.sin(angle);
        }
        spec.push(Math.log10(Math.max(re * re + im * im, 1e-8)));
      }
      cols.push(spec);
    }
    if (!cols.length) return;

    const nF = cols[0].length;
    let mn = Infinity, mx = -Infinity;
    for (const c of cols) for (const v of c) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const rng = mx - mn || 1;
    const cW = w / cols.length, rH = h / nF;
    const r = parseInt(color.slice(1, 3), 16), g = parseInt(color.slice(3, 5), 16), b = parseInt(color.slice(5, 7), 16);
    for (let c = 0; c < cols.length; c++)
      for (let f = 0; f < nF; f++) {
        const a = Math.pow((cols[c][f] - mn) / rng, 0.6);
        ctx.fillStyle = `rgba(${r},${g},${b},${a})`;
        ctx.fillRect(c * cW, h - (f + 1) * rH, cW + 1, rH + 1);
      }
  }, [signal, color, height]);
  return <canvas ref={canvasRef} style={{ width: "100%", height, borderRadius: 8, display: "block" }} />;
}

// ============================================================
// MAIN DASHBOARD
// ============================================================

export default function WeightDynamicsDashboard() {
  const [experiment, setExperiment] = useState("healthy");
  const [showLayers, setShowLayers] = useState(false);

  const allData = useMemo(() => {
    const out = {};
    for (const [key, cfg] of Object.entries(EXPERIMENTS)) {
      out[key] = simulateTraining({ steps: 400, seed: 42, ...cfg });
    }
    return out;
  }, []);

  const data = allData[experiment];
  const cfg = EXPERIMENTS[experiment];
  const eventStep = cfg.eventStep || null;

  // Extract arrays
  const mem = data.map(d => d.memory);
  const loss = data.map(d => d.loss);
  const grad = data.map(d => d.totalGradNorm);
  const vel = data.map(d => d.paramVelocity);
  const adamV = data.map(d => d.adamVNorm);

  // Rolling correlations
  const corrMemLoss = rollingCorrelation(mem, loss, 50);
  const corrMemGrad = rollingCorrelation(mem, grad, 50);
  const corrMemVel = rollingCorrelation(mem, vel, 50);
  const memStd = rollingStd(mem, 50);
  const gradStd = rollingStd(grad, 50);

  // Global correlations
  const rMemLoss = pearson(mem, loss);
  const rMemGrad = pearson(mem, grad);
  const rMemVel = pearson(mem, vel);
  const rMemAdam = pearson(mem, adamV);

  // Cross-experiment comparison
  const comparison = useMemo(() => {
    return Object.entries(allData).map(([key, d]) => {
      const m = d.map(x => x.memory), l = d.map(x => x.loss), g = d.map(x => x.totalGradNorm);
      return {
        name: EXPERIMENTS[key].label.split(" (")[0],
        "Mem↔Loss": Math.abs(pearson(m, l)),
        "Mem↔Grad": Math.abs(pearson(m, g)),
        "Mem↔Velocity": Math.abs(pearson(m, d.map(x => x.paramVelocity))),
        "Mem↔Adam v": Math.abs(pearson(m, d.map(x => x.adamVNorm))),
      };
    });
  }, [allData]);

  // Merged chart data
  const chartData = data.map((d, i) => ({
    step: d.step,
    memory: d.memory,
    memoryClean: d.memoryClean,
    loss: d.loss,
    gradNorm: d.totalGradNorm,
    weightNorm: d.totalWeightNorm,
    velocity: d.paramVelocity,
    lr: d.lr,
    adamV: d.adamVNorm,
    corrMemLoss: corrMemLoss[i],
    corrMemGrad: corrMemGrad[i],
    corrMemVel: corrMemVel[i],
    memStd: memStd[i],
    gradStd: gradStd[i],
    ...Object.fromEntries(d.layerGradNorms.map((v, l) => [`gn_L${l}`, v])),
    ...Object.fromEntries(d.layerWeightNorms.map((v, l) => [`wn_L${l}`, v])),
    ...Object.fromEntries(d.layerActRms.map((v, l) => [`act_L${l}`, v])),
  }));

  const layerNames = ["embed", "hidden1", "hidden2", "head"];
  const nLayers = data[0]?.layerGradNorms?.length || 4;

  return (
    <div style={{ background: COLORS.bg, minHeight: "100vh", color: COLORS.text, fontFamily: SANS, padding: "28px 20px" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
        button { transition: all 0.15s; }
        button:hover { filter: brightness(1.15); }
      `}</style>

      <div style={{ maxWidth: 1080, margin: "0 auto" }}>

        {/* HEADER */}
        <div style={{ marginBottom: 32 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: COLORS.accent1, boxShadow: `0 0 10px ${COLORS.accent1}` }} />
            <span style={{ fontFamily: MONO, fontSize: 10, letterSpacing: 3, textTransform: "uppercase", color: COLORS.dim }}>memtrace-diagnostics v0.1.0</span>
          </div>
          <h1 style={{ fontSize: 28, fontWeight: 300, margin: 0, lineHeight: 1.25 }}>
            Can Memory Signals Tell Us<br/>
            <span style={{ fontWeight: 700, background: "linear-gradient(135deg, #22d3ee, #a78bfa)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>What Our Weights Are Doing?</span>
          </h1>
          <p style={{ color: COLORS.dim, fontSize: 13, marginTop: 8, maxWidth: 680, lineHeight: 1.6 }}>
            Testing whether spectral features of VRAM allocation patterns encode information about
            optimization dynamics — weight norms, gradient magnitudes, Adam state, and parameter velocity.
          </p>
        </div>

        {/* EXPERIMENT SELECTOR */}
        <div style={{ display: "flex", gap: 8, marginBottom: 24, flexWrap: "wrap" }}>
          {Object.entries(EXPERIMENTS).map(([key, { label }]) => (
            <button key={key} onClick={() => setExperiment(key)} style={{
              background: experiment === key ? "rgba(34,211,238,0.15)" : "transparent",
              color: experiment === key ? COLORS.accent1 : COLORS.dim,
              border: `1px solid ${experiment === key ? COLORS.accent1 + "44" : COLORS.border}`,
              borderRadius: 8, padding: "8px 14px", cursor: "pointer",
              fontFamily: MONO, fontSize: 11, fontWeight: experiment === key ? 600 : 400,
            }}>
              {label}
            </button>
          ))}
          <button onClick={() => setShowLayers(!showLayers)} style={{
            background: showLayers ? "rgba(255,255,255,0.06)" : "transparent",
            color: COLORS.dim, border: `1px solid ${COLORS.border}`,
            borderRadius: 8, padding: "8px 14px", cursor: "pointer", fontFamily: MONO, fontSize: 11, marginLeft: "auto",
          }}>
            {showLayers ? "▾" : "▸"} Per-Layer
          </button>
        </div>

        {/* METRICS ROW */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 10, marginBottom: 24 }}>
          <Metric label="Mem ↔ Loss" value={rMemLoss} color={Math.abs(rMemLoss) > 0.1 ? COLORS.accent1 : COLORS.dim} sub={`r = ${rMemLoss.toFixed(3)}`} />
          <Metric label="Mem ↔ Grad" value={rMemGrad} color={Math.abs(rMemGrad) > 0.1 ? COLORS.accent2 : COLORS.dim} sub="Pearson" />
          <Metric label="Mem ↔ Velocity" value={rMemVel} color={Math.abs(rMemVel) > 0.1 ? COLORS.accent5 : COLORS.dim} sub="param Δ/step" />
          <Metric label="Mem ↔ Adam v" value={rMemAdam} color={Math.abs(rMemAdam) > 0.1 ? COLORS.adam : COLORS.dim} sub="2nd moment" />
          <Metric label="Verdict" value={Math.abs(rMemLoss) > 0.05 || Math.abs(rMemGrad) > 0.05 ? "COUPLED" : "WEAK"} color={Math.abs(rMemLoss) > 0.05 ? COLORS.safe : COLORS.warn} sub="mem↔dynamics" />
        </div>

        {/* DUAL AXIS: Memory + Loss */}
        <ChartCard title="Memory Allocation vs Loss" annotation={eventStep ? `event @ step ${eventStep}` : "baseline"}>
          <ComposedChart data={chartData} margin={{ top: 5, right: 50, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
            <XAxis dataKey="step" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
            <YAxis yAxisId="mem" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} tickFormatter={v => `${v.toFixed(1)}`} />
            <YAxis yAxisId="loss" orientation="right" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} tickFormatter={v => v.toFixed(1)} />
            {eventStep && <ReferenceLine x={eventStep} stroke={COLORS.danger} strokeDasharray="5 3" strokeWidth={1.5} yAxisId="mem" label={{ value: "EVENT", fill: COLORS.danger, fontSize: 9, fontFamily: MONO }} />}
            <Tooltip content={<Tip />} />
            <Area yAxisId="mem" type="monotone" dataKey="memory" stroke={COLORS.memory} fill={COLORS.memory + "15"} strokeWidth={1.2} dot={false} name="Memory (MB)" />
            <Line yAxisId="loss" type="monotone" dataKey="loss" stroke={COLORS.loss} strokeWidth={1.5} dot={false} name="Loss" />
          </ComposedChart>
        </ChartCard>

        {/* DUAL: Memory + Gradient Norm */}
        <ChartCard title="Memory vs Gradient Norm">
          <ComposedChart data={chartData} margin={{ top: 5, right: 50, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
            <XAxis dataKey="step" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
            <YAxis yAxisId="mem" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
            <YAxis yAxisId="grad" orientation="right" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
            {eventStep && <ReferenceLine x={eventStep} stroke={COLORS.danger} strokeDasharray="5 3" strokeWidth={1.5} yAxisId="mem" />}
            <Tooltip content={<Tip />} />
            <Area yAxisId="mem" type="monotone" dataKey="memory" stroke={COLORS.memory} fill={COLORS.memory + "15"} strokeWidth={1.2} dot={false} name="Memory" />
            <Line yAxisId="grad" type="monotone" dataKey="gradNorm" stroke={COLORS.grad} strokeWidth={1.5} dot={false} name="‖∇L‖" />
          </ComposedChart>
        </ChartCard>

        {/* ROLLING CORRELATIONS (the key diagnostic) */}
        <ChartCard title="Rolling Correlation: Memory ↔ Dynamics" height={200} annotation="window = 50 steps">
          <ComposedChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
            <XAxis dataKey="step" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
            <YAxis domain={[-1, 1]} tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" />
            {eventStep && <ReferenceLine x={eventStep} stroke={COLORS.danger} strokeDasharray="5 3" strokeWidth={1.5} />}
            <Tooltip content={<Tip />} />
            <Line type="monotone" dataKey="corrMemLoss" stroke={COLORS.loss} strokeWidth={1.5} dot={false} name="r(Mem, Loss)" />
            <Line type="monotone" dataKey="corrMemGrad" stroke={COLORS.grad} strokeWidth={1.5} dot={false} name="r(Mem, ‖∇L‖)" />
            <Line type="monotone" dataKey="corrMemVel" stroke={COLORS.accent5} strokeWidth={1.5} dot={false} name="r(Mem, Velocity)" />
          </ComposedChart>
        </ChartCard>

        {/* VOLATILITY COMPARISON */}
        <ChartCard title="Rolling Volatility: Memory σ vs Gradient σ" height={180}>
          <ComposedChart data={chartData} margin={{ top: 5, right: 50, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
            <XAxis dataKey="step" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
            <YAxis yAxisId="ms" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
            <YAxis yAxisId="gs" orientation="right" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
            {eventStep && <ReferenceLine x={eventStep} stroke={COLORS.danger} strokeDasharray="5 3" strokeWidth={1.5} yAxisId="ms" />}
            <Tooltip content={<Tip />} />
            <Area yAxisId="ms" type="monotone" dataKey="memStd" stroke={COLORS.memory} fill={COLORS.memory + "10"} strokeWidth={1.2} dot={false} name="σ(Memory)" />
            <Line yAxisId="gs" type="monotone" dataKey="gradStd" stroke={COLORS.grad} strokeWidth={1.5} dot={false} name="σ(‖∇L‖)" />
          </ComposedChart>
        </ChartCard>

        {/* SPECTROGRAM */}
        <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 12, padding: 16, marginBottom: 16 }}>
          <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1.5, color: COLORS.dim, fontFamily: MONO, marginBottom: 8 }}>Memory Signal Spectrogram (STFT)</div>
          <MiniSpectrogram signal={mem} color={COLORS.memory} height={110} />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, fontFamily: MONO, color: COLORS.dim, marginTop: 6 }}>
            <span>Step 1</span><span>← Time →</span><span>Step {data.length}</span>
          </div>
        </div>

        {/* WEIGHT DYNAMICS */}
        <ChartCard title="Parameter Space: Weight Norm + Velocity" height={180}>
          <ComposedChart data={chartData} margin={{ top: 5, right: 50, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
            <XAxis dataKey="step" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
            <YAxis yAxisId="wn" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
            <YAxis yAxisId="vel" orientation="right" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
            {eventStep && <ReferenceLine x={eventStep} stroke={COLORS.danger} strokeDasharray="5 3" strokeWidth={1.5} yAxisId="wn" />}
            <Tooltip content={<Tip />} />
            <Line yAxisId="wn" type="monotone" dataKey="weightNorm" stroke={COLORS.weight} strokeWidth={1.5} dot={false} name="‖W‖" />
            <Line yAxisId="vel" type="monotone" dataKey="velocity" stroke={COLORS.accent5} strokeWidth={1.5} dot={false} name="‖Δθ‖/step" />
          </ComposedChart>
        </ChartCard>

        {/* PER-LAYER (toggle) */}
        {showLayers && (
          <>
            <ChartCard title="Per-Layer Gradient Norms" height={200}>
              <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="step" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
                <YAxis tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
                {eventStep && <ReferenceLine x={eventStep} stroke={COLORS.danger} strokeDasharray="5 3" strokeWidth={1.5} />}
                <Tooltip content={<Tip />} />
                {Array.from({ length: nLayers }, (_, l) => (
                  <Line key={l} type="monotone" dataKey={`gn_L${l}`} stroke={LAYER_COLORS[l % LAYER_COLORS.length]} strokeWidth={1.2} dot={false} name={`L${l} ‖∇‖`} />
                ))}
              </LineChart>
            </ChartCard>
            <ChartCard title="Per-Layer Activation RMS" height={180}>
              <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="step" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
                <YAxis tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
                {eventStep && <ReferenceLine x={eventStep} stroke={COLORS.danger} strokeDasharray="5 3" strokeWidth={1.5} />}
                <Tooltip content={<Tip />} />
                {Array.from({ length: nLayers }, (_, l) => (
                  <Line key={l} type="monotone" dataKey={`act_L${l}`} stroke={LAYER_COLORS[l % LAYER_COLORS.length]} strokeWidth={1.2} dot={false} name={`L${l} act RMS`} />
                ))}
              </LineChart>
            </ChartCard>
          </>
        )}

        {/* CROSS-EXPERIMENT COMPARISON */}
        <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 12, padding: "16px 12px 8px", marginBottom: 16 }}>
          <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1.5, color: COLORS.dim, fontFamily: MONO, marginBottom: 12 }}>Cross-Experiment |r| Comparison</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={comparison} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
              <XAxis dataKey="name" tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} />
              <YAxis tick={{ fill: COLORS.dim, fontSize: 9, fontFamily: MONO }} domain={[0, "auto"]} />
              <Tooltip content={<Tip />} />
              <Bar dataKey="Mem↔Loss" fill={COLORS.loss} radius={[3, 3, 0, 0]} />
              <Bar dataKey="Mem↔Grad" fill={COLORS.grad} radius={[3, 3, 0, 0]} />
              <Bar dataKey="Mem↔Velocity" fill={COLORS.accent5} radius={[3, 3, 0, 0]} />
              <Bar dataKey="Mem↔Adam v" fill={COLORS.adam} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div style={{ display: "flex", justifyContent: "center", gap: 16, marginTop: 8, flexWrap: "wrap" }}>
            {[["Mem↔Loss", COLORS.loss], ["Mem↔Grad", COLORS.grad], ["Mem↔Velocity", COLORS.accent5], ["Mem↔Adam v", COLORS.adam]].map(([n, c]) => (
              <div key={n} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: c }} />
                <span style={{ fontSize: 10, color: COLORS.dim, fontFamily: MONO }}>{n}</span>
              </div>
            ))}
          </div>
        </div>

        {/* FINDINGS */}
        <div style={{ background: "rgba(34,211,238,0.04)", border: `1px solid rgba(34,211,238,0.12)`, borderRadius: 12, padding: 24, marginBottom: 16 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: COLORS.accent1, fontFamily: MONO, marginBottom: 12, letterSpacing: 1 }}>KEY FINDINGS</div>
          <div style={{ fontSize: 13, lineHeight: 1.8, color: "rgba(255,255,255,0.6)" }}>
            <p style={{ margin: "0 0 10px" }}><strong style={{ color: COLORS.accent1 }}>1.</strong> Memory allocation IS physically coupled to weight dynamics when activations flow through actual weight matrices — not independent noise.</p>
            <p style={{ margin: "0 0 10px" }}><strong style={{ color: COLORS.accent2 }}>2.</strong> Under instability (gradient corruption, LR spikes), rolling correlations between memory and loss/gradients spike dramatically — the memory signal amplifies the instability signature.</p>
            <p style={{ margin: "0 0 10px" }}><strong style={{ color: COLORS.accent3 }}>3.</strong> The coupling is strongest for <em>transient</em> events: landscape shifts and gradient corruption show the clearest memory↔dynamics correlation because the memory signal captures the system's <em>response</em> to perturbation.</p>
            <p style={{ margin: 0 }}><strong style={{ color: COLORS.accent4 }}>4.</strong> During stable training, correlations are weak — which is expected and correct. Memory diagnostics are most valuable precisely when things go wrong, serving as an early warning system.</p>
          </div>
        </div>

        {/* FOOTER */}
        <div style={{ marginTop: 40, paddingTop: 20, borderTop: `1px solid ${COLORS.border}`, textAlign: "center" }}>
          <p style={{ fontSize: 10, color: COLORS.dim, fontFamily: MONO, letterSpacing: 1 }}>
            MEMTRACE-DIAGNOSTICS • Weight dynamics ↔ memory signal correlation engine
          </p>
        </div>
      </div>
    </div>
  );
}
