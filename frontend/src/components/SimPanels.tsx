import { HeartPulse, Wind } from "lucide-react";
import { useEffect, useMemo, useRef, type MutableRefObject, type ReactNode } from "react";
import type { VolumeData } from "../api/client";
import { GATE_FRACTION, type SimSnapshot, type SimState } from "../sim/treatment";
import { LUTS } from "../viewer/volume";
import { cx } from "./ui";

/** Scrolling respiratory trace of the dummy patient, with the gating window and beam state. */
export function PatientMonitor({ simRef, snap, gating }: { simRef: MutableRefObject<SimState>; snap: SimSnapshot; gating: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const gatingRef = useRef(gating);
  gatingRef.current = gating;
  useEffect(() => {
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const c = canvas.current;
      if (!c) return;
      const w = c.clientWidth;
      const h = c.clientHeight;
      const dpr = Math.min(window.devicePixelRatio, 2);
      if (c.width !== Math.round(w * dpr)) {
        c.width = Math.round(w * dpr);
        c.height = Math.round(h * dpr);
      }
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const s = simRef.current;
      const pad = 6;
      const y = (b: number) => h - pad - b * (h - 2 * pad);
      // Gating window: the band near exhale where the beam may be on.
      ctx.fillStyle = gatingRef.current ? "rgba(47,208,184,0.12)" : "rgba(120,140,170,0.06)";
      ctx.fillRect(0, y(GATE_FRACTION), w, y(0) - y(GATE_FRACTION));
      ctx.strokeStyle = "rgba(120,140,170,0.18)";
      ctx.lineWidth = 1;
      for (const g of [0.25, 0.5, 0.75]) {
        ctx.beginPath();
        ctx.moveTo(0, y(g));
        ctx.lineTo(w, y(g));
        ctx.stroke();
      }
      const n = s.histCount;
      const cap = s.histBreath.length;
      if (n < 2) return;
      const dx = w / (cap - 1);
      const at = (i: number) => (s.histHead - n + i + cap) % cap;
      // Beam state strip along the bottom.
      for (let i = 0; i < n; i++) {
        const st = s.histBeam[at(i)];
        if (!st) continue;
        ctx.fillStyle = st === 1 ? "rgba(240,140,255,0.85)" : "rgba(255,176,32,0.8)";
        ctx.fillRect(w - (n - i) * dx, h - 3, dx + 0.5, 3);
      }
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const px = w - (n - 1 - i) * dx;
        const py = y(s.histBreath[at(i)]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = "#7fd6ff";
      ctx.lineWidth = 1.6;
      ctx.stroke();
      const last = s.histBreath[at(n - 1)];
      ctx.fillStyle = "#e6f6ff";
      ctx.beginPath();
      ctx.arc(w - 1.5, y(last), 2.6, 0, Math.PI * 2);
      ctx.fill();
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [simRef]);

  return (
    <div className="space-y-3">
      <div className="relative rounded-lg border border-ink-700/60 bg-ink-950 p-2">
        <canvas ref={canvas} className="block h-[96px] w-full" aria-label="Simulated breathing trace" />
        <span className="pointer-events-none absolute left-3 top-2 text-[10px] uppercase tracking-wider text-ink-500">Breathing · last 30 s</span>
        <span className="pointer-events-none absolute bottom-2 right-3 text-[10px] text-ink-500">inhale ↑ · exhale ↓</span>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <Vital icon={<Wind className="h-3.5 w-3.5" />} label="Breaths/min" value={snap.rate.toFixed(0)} />
        <Vital icon={<HeartPulse className="h-3.5 w-3.5" />} label="Pulse" value={snap.pulse.toFixed(0)} />
        <Vital label="Target shift" value={`${snap.offsetMm.toFixed(1)} mm`} warn={snap.breath > GATE_FRACTION} />
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-ink-400">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-3 rounded-sm bg-teal/25" /> Gating window (exhale)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-1 w-3 rounded-sm bg-[#f08cff]" /> Beam on
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-1 w-3 rounded-sm bg-[#ffb020]" /> Beam held
        </span>
      </div>
    </div>
  );
}

function Vital({ icon, label, value, warn }: { icon?: ReactNode; label: string; value: string; warn?: boolean }) {
  return (
    <div className="rounded-lg border border-ink-700/60 bg-ink-925/70 px-2.5 py-2">
      <div className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-ink-500">
        {icon}
        {label}
      </div>
      <div className={cx("mt-0.5 font-mono text-[17px] tabular-nums", warn ? "text-ev-uncertain" : "text-ink-100")}>{value}</div>
    </div>
  );
}

/**
 * Axial slice through the target with the illustration building up beam by beam:
 * each beam's precomputed share is weighted by how much of it has been replayed.
 */
export function DoseReplay({ image, beams, target, sliceIndex, snap, angles, fieldMm }: {
  image: VolumeData;
  beams: VolumeData;
  target: VolumeData | null;
  sliceIndex: number;
  snap: SimSnapshot;
  angles: number[];
  fieldMm: number;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [nz, ny, nx] = image.shape;
  const [sx, sy] = image.spacing;
  const z = Math.min(Math.max(sliceIndex, 0), nz - 1);

  // Grey CT slice and target outline do not change while the replay runs.
  const base = useMemo(() => {
    const px = new Uint8ClampedArray(ny * nx * 4);
    const lo = 50 - 175;
    const off = z * ny * nx;
    for (let i = 0; i < ny * nx; i++) {
      const hu = image.data[off + i] * image.scale + image.offset;
      const g = Math.min(255, Math.max(0, ((hu - lo) / 350) * 255));
      px[i * 4] = px[i * 4 + 1] = px[i * 4 + 2] = g;
      px[i * 4 + 3] = 255;
    }
    const edge = new Uint8Array(ny * nx);
    let cy = 0;
    let cxv = 0;
    let count = 0;
    if (target && target.data.length === image.data.length) {
      const t = (yy: number, xx: number) => yy >= 0 && yy < ny && xx >= 0 && xx < nx && target.data[off + yy * nx + xx] === 1;
      for (let yy = 0; yy < ny; yy++)
        for (let xx = 0; xx < nx; xx++) {
          if (!t(yy, xx)) continue;
          cy += yy;
          cxv += xx;
          count += 1;
          if (!t(yy - 1, xx) || !t(yy + 1, xx) || !t(yy, xx - 1) || !t(yy, xx + 1)) edge[yy * nx + xx] = 1;
        }
    }
    return { px, edge, iso: count ? [cxv / count, cy / count] : [nx / 2, ny / 2] };
  }, [image, target, z, ny, nx]);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    c.width = nx;
    c.height = ny;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const out = new ImageData(new Uint8ClampedArray(base.px), nx, ny);
    const nb = beams.shape[0];
    const plane = ny * nx;
    const lut = LUTS.dose;
    const vmax = 1.1;
    for (let i = 0; i < plane; i++) {
      let v = 0;
      for (let b = 0; b < nb; b++) {
        const f = snap.delivered[b] ?? 0;
        if (f) v += f * beams.data[b * plane + i];
      }
      v *= beams.scale;
      if (v > 0.05) {
        const k = Math.min(255, Math.round((v / vmax) * 255)) * 3;
        const a = 0.58;
        out.data[i * 4] = out.data[i * 4] * (1 - a) + lut[k] * a;
        out.data[i * 4 + 1] = out.data[i * 4 + 1] * (1 - a) + lut[k + 1] * a;
        out.data[i * 4 + 2] = out.data[i * 4 + 2] * (1 - a) + lut[k + 2] * a;
      }
      if (base.edge[i]) out.data[i * 4] = out.data[i * 4 + 1] = out.data[i * 4 + 2] = 255;
    }
    ctx.putImageData(out, 0, 0);
  }, [base, beams, snap.delivered, nx, ny]);

  // Beam's direction on screen. Radiological display: patient left on the right, anterior at the top.
  const theta = (snap.gantry * Math.PI) / 180;
  const dirX = Math.sin(theta) / sx;
  const dirY = -Math.cos(theta) / sy;
  const norm = Math.hypot(dirX, dirY) || 1;
  const [ix, iy] = base.iso;
  const far = Math.max(nx, ny) * 1.5;
  const halfPx = fieldMm / sx;
  const ux = dirX / norm;
  const uy = dirY / norm;
  // The field as a band through the isocentre, slightly wider on the exit side (beam divergence).
  const px = -uy * halfPx;
  const py = ux * halfPx;
  const wedge = [
    [ix + ux * far + px * 0.85, iy + uy * far + py * 0.85],
    [ix + ux * far - px * 0.85, iy + uy * far - py * 0.85],
    [ix - ux * far - px * 1.15, iy - uy * far - py * 1.15],
    [ix - ux * far + px * 1.15, iy - uy * far + py * 1.15],
  ]
    .map((p) => p.join(","))
    .join(" ");

  return (
    <div className="relative mx-auto w-full max-w-[460px]" style={{ aspectRatio: `${nx * sx} / ${ny * sy}` }}>
      <canvas ref={canvas} className="absolute inset-0 h-full w-full rounded-lg" style={{ imageRendering: "auto" }} aria-label="Dose building up on an axial slice" />
      <svg viewBox={`0 0 ${nx} ${ny}`} preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full">
        {snap.beamOn && <polygon points={wedge} fill="rgba(240,140,255,0.16)" stroke="rgba(240,140,255,0.55)" strokeWidth={0.4} />}
        {angles.map((a, i) => {
          const t = (a * Math.PI) / 180;
          const vx = Math.sin(t) / sx;
          const vy = -Math.cos(t) / sy;
          const nn = Math.hypot(vx, vy) || 1;
          const r = Math.min(nx, ny) * 0.47;
          return (
            <circle
              key={i}
              cx={nx / 2 + (vx / nn) * r}
              cy={ny / 2 + (vy / nn) * r}
              r={angles.length > 20 ? 0.8 : 1.6}
              fill={(snap.delivered[i] ?? 0) >= 1 ? "#f08cff" : (snap.delivered[i] ?? 0) > 0 ? "#ffd6ff" : "rgba(160,175,200,0.5)"}
            />
          );
        })}
      </svg>
      <span className="absolute left-1/2 top-1 -translate-x-1/2 text-[10px] font-semibold text-ink-300">A</span>
      <span className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[10px] font-semibold text-ink-300">P</span>
      <span className="absolute left-1.5 top-1/2 -translate-y-1/2 text-[10px] font-semibold text-ink-300">R</span>
      <span className="absolute right-1.5 top-1/2 -translate-y-1/2 text-[10px] font-semibold text-ink-300">L</span>
    </div>
  );
}
