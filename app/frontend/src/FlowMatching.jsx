import { useEffect, useRef, useState, useCallback } from "react";

const W = 680;
const H = 420;

const TARGETS = [
  { cx: 0.25, cy: 0.3, r: 0.07 },
  { cx: 0.7, cy: 0.25, r: 0.07 },
  { cx: 0.5, cy: 0.72, r: 0.07 },
];

function gaussian() {
  let u = 0,
    v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function sampleTarget() {
  const t = TARGETS[Math.floor(Math.random() * TARGETS.length)];
  const a = Math.random() * Math.PI * 2;
  const r = Math.abs((t.r * (Math.random() + Math.random())) / 2);
  return [t.cx + Math.cos(a) * r, t.cy + Math.sin(a) * r];
}

function sampleSource() {
  return [0.5 + gaussian() * 0.18, 0.5 + gaussian() * 0.18];
}

function ease(t) {
  return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function getPos(p, t) {
  const e = ease(t);
  return [lerp(p.sx, p.tx, e), lerp(p.sy, p.ty, e)];
}

function colorAt(t) {
  const r1 = 127,
    g1 = 119,
    b1 = 221;
  const r2 = 239,
    g2 = 159,
    b2 = 39;
  const r3 = 29,
    g3 = 158,
    b3 = 117;
  let r, g, b;
  if (t < 0.5) {
    const f = t * 2;
    r = r1 + (r2 - r1) * f;
    g = g1 + (g2 - g1) * f;
    b = b1 + (b2 - b1) * f;
  } else {
    const f = (t - 0.5) * 2;
    r = r2 + (r3 - r2) * f;
    g = g2 + (g3 - g2) * f;
    b = b2 + (b3 - b2) * f;
  }
  return `rgba(${Math.round(r)},${Math.round(g)},${Math.round(b)},0.85)`;
}

function makeParticles(n) {
  return Array.from({ length: n }, () => {
    const src = sampleSource();
    const tgt = sampleTarget();
    return {
      sx: Math.max(0.05, Math.min(0.95, src[0])),
      sy: Math.max(0.05, Math.min(0.95, src[1])),
      tx: Math.max(0.05, Math.min(0.95, tgt[0])),
      ty: Math.max(0.05, Math.min(0.95, tgt[1])),
      trail: [],
    };
  });
}

export default function FlowMatching() {
  const canvasRef = useRef(null);
  const stateRef = useRef({
    particles: makeParticles(80),
    T: 0,
    playing: true,
    speed: 1,
    count: 80,
    last: null,
    raf: null,
  });
  const [T, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [count, setCount] = useState(80);

  const draw = useCallback((particles, T) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const BG = dark ? "#1a1a18" : "#f8f7f3";
    const GRID = dark ? "rgba(200,200,180,0.035)" : "rgba(60,60,40,0.04)";
    const ARROW = dark ? "rgba(200,200,180,0.14)" : "rgba(80,80,60,0.11)";
    const MUTED = dark ? "#5f5e5a" : "#b4b2a9";
    const TEXT = dark ? "#888780" : "#888780";

    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      ctx.beginPath();
      ctx.moveTo((i * W) / 10, 0);
      ctx.lineTo((i * W) / 10, H);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, (i * H) / 10);
      ctx.lineTo(W, (i * H) / 10);
      ctx.stroke();
    }

    for (const tg of TARGETS) {
      const grd = ctx.createRadialGradient(
        tg.cx * W,
        tg.cy * H,
        0,
        tg.cx * W,
        tg.cy * H,
        tg.r * W * 1.8,
      );
      grd.addColorStop(0, "rgba(29,158,117,0.16)");
      grd.addColorStop(1, "rgba(29,158,117,0)");
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(tg.cx * W, tg.cy * H, tg.r * W * 1.8, 0, Math.PI * 2);
      ctx.fill();
    }

    const srcA = Math.max(0, 1 - T * 3) * 0.12;
    if (srcA > 0) {
      const sg = ctx.createRadialGradient(
        W / 2,
        H / 2,
        0,
        W / 2,
        H / 2,
        W * 0.42,
      );
      sg.addColorStop(0, `rgba(127,119,221,${srcA})`);
      sg.addColorStop(1, "rgba(127,119,221,0)");
      ctx.fillStyle = sg;
      ctx.fillRect(0, 0, W, H);
    }

    if (T > 0.05 && T < 0.95) {
      const cols = 13,
        rows = 8;
      for (let i = 0; i <= cols; i++) {
        for (let j = 0; j <= rows; j++) {
          const gx = i / cols,
            gy = j / rows;
          let vx = 0,
            vy = 0,
            cnt = 0;
          for (const p of particles) {
            const [px, py] = getPos(p, T);
            const d = Math.hypot(px - gx, py - gy);
            if (d < 0.17) {
              const w = 1 / (d + 0.025);
              vx += (p.tx - p.sx) * w;
              vy += (p.ty - p.sy) * w;
              cnt += w;
            }
          }
          if (cnt > 2) {
            vx /= cnt;
            vy /= cnt;
            const mag = Math.hypot(vx, vy);
            if (mag < 0.001) continue;
            const norm = 0.028 / (mag + 0.01);
            const x1 = gx * W,
              y1 = gy * H;
            const x2 = x1 + vx * norm * W,
              y2 = y1 + vy * norm * H;
            ctx.strokeStyle = ARROW;
            ctx.lineWidth = 0.7;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
            const ax = x2 - x1,
              ay = y2 - y1,
              len = Math.hypot(ax, ay);
            if (len > 3) {
              const ux = ax / len,
                uy = ay / len;
              ctx.beginPath();
              ctx.moveTo(x2, y2);
              ctx.lineTo(x2 - ux * 4 - uy * 2, y2 - uy * 4 + ux * 2);
              ctx.lineTo(x2 - ux * 4 + uy * 2, y2 - uy * 4 - ux * 2);
              ctx.closePath();
              ctx.fillStyle = ARROW;
              ctx.fill();
            }
          }
        }
      }
    }

    for (const p of particles) {
      const [px, py] = getPos(p, T);
      const col = colorAt(T);
      if (p.trail.length > 1) {
        ctx.beginPath();
        ctx.moveTo(p.trail[0][0] * W, p.trail[0][1] * H);
        for (let i = 1; i < p.trail.length; i++)
          ctx.lineTo(p.trail[i][0] * W, p.trail[i][1] * H);
        ctx.strokeStyle = col.replace("0.85", "0.2");
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(px * W, py * H, 3.2, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.fill();
    }

    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = MUTED;
    ctx.fillText("t = 0  noise", 14, H - 38);
    ctx.fillText("t = 1  data", 14, H - 20);
    ctx.fillStyle = dark ? "#2c2c2a" : "#e8e7e2";
    ctx.fillRect(110, H - 32, 180, 3);
    ctx.fillStyle = colorAt(T);
    ctx.fillRect(110, H - 32, 180 * T, 3);
    ctx.font = "500 12px system-ui, sans-serif";
    ctx.fillStyle = TEXT;
    ctx.fillText(`t = ${T.toFixed(2)}`, W - 70, H - 20);
  }, []);

  useEffect(() => {
    const s = stateRef.current;
    s.speed = speed;
  }, [speed]);

  useEffect(() => {
    const s = stateRef.current;
    s.playing = playing;
  }, [playing]);

  useEffect(() => {
    const s = stateRef.current;
    s.particles = makeParticles(count);
    s.T = 0;
    s.count = count;
  }, [count]);

  useEffect(() => {
    const s = stateRef.current;

    function frame(ts) {
      if (s.last === null) s.last = ts;
      const dt = Math.min((ts - s.last) / 1000, 0.05);
      s.last = ts;

      if (s.playing) {
        s.T += dt * s.speed * 0.28;
        if (s.T >= 1) {
          s.T = 0;
          s.particles = makeParticles(s.count);
        }
        for (const p of s.particles) {
          const pos = getPos(p, s.T);
          p.trail.push(pos);
          if (p.trail.length > 18) p.trail.shift();
        }
        setT(parseFloat(s.T.toFixed(2)));
      }

      draw(s.particles, s.T);
      s.raf = requestAnimationFrame(frame);
    }

    s.raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(s.raf);
  }, [draw]);

  function restart() {
    const s = stateRef.current;
    s.particles = makeParticles(s.count);
    s.T = 0;
    s.last = null;
  }

  return (
    <div style={{ fontFamily: "system-ui, sans-serif" }}>
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        style={{ width: "100%", borderRadius: 12, display: "block" }}
      />

      <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
        <span
          style={{
            fontSize: 12,
            color: "#7F77DD",
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: "#7F77DD",
              display: "inline-block",
            }}
          />
          Source (noise)
        </span>
        <span
          style={{
            fontSize: 12,
            color: "#1D9E75",
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: "#1D9E75",
              display: "inline-block",
            }}
          />
          Target (data)
        </span>
        <span
          style={{
            fontSize: 12,
            color: "#EF9F27",
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: "#EF9F27",
              display: "inline-block",
            }}
          />
          Particles in flow
        </span>
      </div>

      <div
        style={{
          display: "flex",
          gap: 20,
          marginTop: 12,
          flexWrap: "wrap",
          alignItems: "center",
          fontSize: 13,
          color: "#888",
        }}
      >
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          Speed
          <input
            type="range"
            min="0.2"
            max="3"
            step="0.1"
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
            style={{ width: 100 }}
          />
          <span style={{ minWidth: 28, fontWeight: 500, color: "#444" }}>
            {speed.toFixed(1)}×
          </span>
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          Particles
          <input
            type="range"
            min="30"
            max="200"
            step="10"
            value={count}
            onChange={(e) => setCount(parseInt(e.target.value))}
            style={{ width: 100 }}
          />
          <span style={{ minWidth: 28, fontWeight: 500, color: "#444" }}>
            {count}
          </span>
        </label>
        <span>
          t = <strong>{T.toFixed(2)}</strong>
        </span>
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
        <button onClick={() => setPlaying((p) => !p)}>
          {playing ? "Pause" : "Play"}
        </button>
        <button onClick={restart}>Restart</button>
      </div>
    </div>
  );
}
