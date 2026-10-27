import { useEffect, useRef, useState } from "react";

const AXIS = "#4a5d80";
const GRID = "#1b2740";
const TEXT = "#9fb0cc";

export const SERIES_COLORS = ["#2fd0b8", "#4c9bff", "#f4b740", "#ff7a8a", "#a48bff", "#e58cff", "#62d27a", "#ffe25a"];

interface Series {
  name: string;
  color: string;
  points: { x: number; y: number | null }[];
  dashed?: boolean;
}

function niceTicks(min: number, max: number, n = 5): number[] {
  if (max <= min) return [min];
  const raw = (max - min) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function LineChart({
  series,
  xLabel,
  yLabel,
  height = 220,
  xFormat = (v: number) => `${v}`,
  yFormat = (v: number) => `${v}`,
  yMin,
  yMax,
  xMax,
  markers = true,
  diagonal = false,
}: {
  series: Series[];
  xLabel?: string;
  yLabel?: string;
  height?: number;
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  yMin?: number;
  yMax?: number;
  xMax?: number;
  markers?: boolean;
  diagonal?: boolean;
}) {
  const [hover, setHover] = useState<{ x: number; y: number; label: string } | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(560);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    // Draw at the real pixel width so text and markers keep their size at any container width.
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const H = height;
  const m = { l: 48, r: 14, t: 12, b: 34 };
  const pts = series.flatMap((s) => s.points.filter((p) => p.y !== null)) as { x: number; y: number }[];
  if (!pts.length) return <p ref={wrapRef as never} className="text-[13px] text-ink-400">No data.</p>;
  const x0 = Math.min(...pts.map((p) => p.x));
  const x1 = xMax ?? Math.max(...pts.map((p) => p.x));
  const y0 = yMin ?? Math.min(0, ...pts.map((p) => p.y));
  const y1 = yMax ?? Math.max(...pts.map((p) => p.y)) * 1.08;
  const sx = (x: number) => m.l + ((x - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const sy = (y: number) => H - m.b - ((y - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const xt = niceTicks(x0, x1);
  const yt = niceTicks(y0, y1);
  return (
    <div className="relative" ref={wrapRef}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img" aria-label={yLabel ? `${yLabel} by ${xLabel}` : "chart"}>
        {yt.map((t) => (
          <g key={`y${t}`}>
            <line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} stroke={GRID} />
            <text x={m.l - 6} y={sy(t) + 3.5} textAnchor="end" fontSize="10" fill={TEXT}>
              {yFormat(t)}
            </text>
          </g>
        ))}
        {xt.map((t) => (
          <text key={`x${t}`} x={sx(t)} y={H - m.b + 14} textAnchor="middle" fontSize="10" fill={TEXT}>
            {xFormat(t)}
          </text>
        ))}
        <line x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} stroke={AXIS} />
        <line x1={m.l} x2={m.l} y1={m.t} y2={H - m.b} stroke={AXIS} />
        {diagonal && <line x1={sx(x0)} y1={sy(y0)} x2={sx(x1)} y2={sy(y1)} stroke={AXIS} strokeDasharray="4 4" />}
        {xLabel && (
          <text x={(m.l + W - m.r) / 2} y={H - 4} textAnchor="middle" fontSize="10.5" fill={TEXT}>
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text x={12} y={(m.t + H - m.b) / 2} textAnchor="middle" fontSize="10.5" fill={TEXT} transform={`rotate(-90 12 ${(m.t + H - m.b) / 2})`}>
            {yLabel}
          </text>
        )}
        {series.map((s) => {
          const segs: string[] = [];
          let open = false;
          s.points.forEach((p) => {
            if (p.y === null) {
              open = false;
              return;
            }
            segs.push(`${open ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`);
            open = true;
          });
          return (
            <g key={s.name}>
              <path d={segs.join("")} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? "5 4" : undefined} />
              {markers &&
                s.points.map((p, i) =>
                  p.y === null ? null : (
                    <circle
                      key={i}
                      cx={sx(p.x)}
                      cy={sy(p.y)}
                      r={3.5}
                      fill="#0d1421"
                      stroke={s.color}
                      strokeWidth={2}
                      onMouseEnter={() => setHover({ x: sx(p.x), y: sy(p.y!), label: `${s.name}: ${yFormat(p.y!)} @ ${xFormat(p.x)}` })}
                      onMouseLeave={() => setHover(null)}
                    />
                  ),
                )}
            </g>
          );
        })}
      </svg>
      {hover && (
        <div
          className="pointer-events-none absolute rounded-md border border-ink-600 bg-ink-850 px-2 py-1 text-[11px] text-ink-100"
          style={{ left: `${(hover.x / W) * 100}%`, top: `${(hover.y / H) * 100}%`, transform: "translate(-50%, -130%)" }}
        >
          {hover.label}
        </div>
      )}
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1">
        {series.map((s) => (
          <span key={s.name} className="flex items-center gap-1.5 text-[11px] text-ink-300">
            <span className="h-[3px] w-3 rounded" style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  );
}

/** Probability with an interval on a 0-1 axis. */
export function IntervalBar({ p, lo, hi, color = "#a48bff" }: { p: number; lo: number; hi: number; color?: string }) {
  return (
    <div className="w-full">
      <div className="relative h-3 rounded-full bg-ink-800">
        <div className="absolute inset-y-0 rounded-full opacity-35" style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%`, background: color }} />
        <div className="absolute -top-1 h-5 w-[3px] rounded" style={{ left: `calc(${p * 100}% - 1.5px)`, background: color }} />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-ink-500">
        <span>0</span>
        <span>0.25</span>
        <span>0.5</span>
        <span>0.75</span>
        <span>1</span>
      </div>
    </div>
  );
}

export function DivergingBars({ items, format = (v: number) => v.toFixed(2) }: {
  items: { label: string; value: number }[];
  format?: (v: number) => string;
}) {
  const max = Math.max(1e-6, ...items.map((i) => Math.abs(i.value)));
  return (
    <div className="space-y-1.5">
      {items.map((it) => (
        <div key={it.label} className="grid grid-cols-[minmax(0,180px)_minmax(0,1fr)_56px] items-center gap-2 text-[12px]">
          <span className="truncate text-ink-300" title={it.label}>
            {it.label}
          </span>
          <div className="relative h-2.5 rounded bg-ink-800">
            <div className="absolute inset-y-0 left-1/2 w-px bg-ink-600" />
            <div
              className="absolute inset-y-0 rounded"
              style={{
                left: it.value >= 0 ? "50%" : `${50 - (Math.abs(it.value) / max) * 50}%`,
                width: `${(Math.abs(it.value) / max) * 50}%`,
                background: it.value >= 0 ? "#ff7a8a" : "#4c9bff",
              }}
            />
          </div>
          <span className="text-right font-mono text-ink-400">{format(it.value)}</span>
        </div>
      ))}
    </div>
  );
}

/** Horizontal point estimates with CI whiskers on a shared axis. */
export function ForestPlot({ rows, min = 0, max = 1 }: {
  rows: { label: string; value: number | null; lo: number | null; hi: number | null; n?: number }[];
  min?: number;
  max?: number;
}) {
  const pos = (v: number) => `${((v - min) / (max - min)) * 100}%`;
  return (
    <div className="space-y-2">
      {rows.map((r) => (
        <div key={r.label} className="grid grid-cols-[minmax(0,160px)_minmax(0,1fr)_120px] items-center gap-3 text-[12px]">
          <span className="truncate text-ink-300">
            {r.label}
            {r.n !== undefined && <span className="text-ink-500"> · n={r.n}</span>}
          </span>
          <div className="relative h-4">
            <div className="absolute inset-x-0 top-1/2 h-px bg-ink-700" />
            <div className="absolute top-1/2 h-px bg-ink-700" style={{ left: pos(0.5), width: 1, height: 16, top: 0 }} />
            {r.lo !== null && r.hi !== null && (
              <div className="absolute top-1/2 h-[2px] -translate-y-1/2 bg-accent/70" style={{ left: pos(r.lo), width: `calc(${pos(r.hi)} - ${pos(r.lo)})` }} />
            )}
            {r.value !== null && <div className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent" style={{ left: pos(r.value) }} />}
          </div>
          <span className="text-right font-mono text-ink-400">
            {r.value === null ? "n/a" : r.value.toFixed(3)}
            {r.lo !== null && r.hi !== null && <span className="text-ink-500"> [{r.lo.toFixed(2)}-{r.hi.toFixed(2)}]</span>}
          </span>
        </div>
      ))}
      <div className="grid grid-cols-[minmax(0,160px)_minmax(0,1fr)_120px] gap-3">
        <span />
        <div className="flex justify-between font-mono text-[10px] text-ink-500">
          <span>{min}</span>
          <span>{(min + max) / 2}</span>
          <span>{max}</span>
        </div>
        <span />
      </div>
    </div>
  );
}

/** Dot strip of per-item values on 0-1. */
export function StripPlot({ values, label }: { values: number[]; label: string }) {
  return (
    <div>
      <div className="relative h-8 rounded bg-ink-925">
        {values.map((v, i) => (
          <span
            key={i}
            className="absolute h-2.5 w-2.5 -translate-x-1/2 rounded-full border border-ink-950 bg-teal"
            style={{ left: `${v * 100}%`, top: `${8 + ((i * 37) % 13)}px` }}
            title={`${label}: ${v.toFixed(3)}`}
          />
        ))}
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-ink-500">
        <span>0</span>
        <span>0.5</span>
        <span>1</span>
      </div>
    </div>
  );
}
