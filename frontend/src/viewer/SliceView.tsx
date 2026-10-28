import { Maximize2, Minus, Plus } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { VolumeData } from "../api/client";
import { cx } from "../components/ui";
import { traceContours, type Loop } from "./contours";
import { centerOn, fitView, panBy, pixelSize, planeToScreen, screenToPlane, zoomAt, type ViewState, type Viewport } from "./transform";
import {
  LESION_COLORS,
  LUTS,
  type LutName,
  type Plane,
  type RGB,
  crosshair,
  pixelToVoxel,
  planeInfo,
  sliceIndexFor,
  sliceLabels,
  sliceOffsetMm,
  voxelIndex,
  voxelValue,
} from "./volume";

export type LabelStyle = "both" | "outline" | "fill";

export interface Overlay {
  id: string;
  data: VolumeData;
  type: "label" | "map";
  opacity: number;
  colors?: Record<number, RGB>;
  style?: LabelStyle;
  only?: number[]; // label values to draw
  lut?: LutName;
  threshold?: number; // map values below this are transparent
  vmax?: number;
  /** Bump when ``data`` is edited in place so cached outlines are rebuilt. */
  revision?: number;
}

export interface Brush {
  radius: number; // voxels
  value: number; // 0 erases
  onPaint: (indices: number[], value: number) => void;
}

const ORIENT: Record<Plane, [string, string, string, string]> = {
  // left, right, top, bottom labels (radiological display)
  axial: ["R", "L", "A", "P"],
  coronal: ["R", "L", "S", "I"],
  sagittal: ["A", "P", "S", "I"],
};
const AXIS: Record<Plane, string> = { axial: "z", coronal: "y", sagittal: "x" };
const SLICE_PX = 50; // wheel distance per slice for trackpads
const rgba = (c: RGB, a: number) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

type Cache = Map<string, { deps: unknown[]; value: unknown }>;
function cached<T>(cache: Cache, key: string, deps: unknown[], make: () => T): T {
  const hit = cache.get(key);
  if (hit && hit.deps.length === deps.length && hit.deps.every((d, i) => d === deps[i])) return hit.value as T;
  const value = make();
  cache.set(key, { deps, value });
  return value;
}

type Drag = { mode: "point" | "pan" | "window" | "paint"; x: number; y: number; level: number; width: number; id: number };

export function SliceView({
  image,
  plane,
  position,
  onPosition,
  level,
  width,
  overlays = [],
  brush,
  title,
  compact = false,
  showCrosshair = true,
  smooth = true,
  resetSignal = 0,
  focusSignal,
  onWindow,
}: {
  image: VolumeData;
  plane: Plane;
  position: [number, number, number];
  onPosition: (zyx: [number, number, number]) => void;
  level: number;
  width: number;
  overlays?: Overlay[];
  brush?: Brush | null;
  title?: string;
  compact?: boolean;
  showCrosshair?: boolean;
  /** Interpolate the CT image between voxels (display only). Outlines are always exact. */
  smooth?: boolean;
  /** Changing this value fits the view again. */
  resetSignal?: number;
  /** Changing ``n`` centres a zoomed-in view on ``zyx``. */
  focusSignal?: { zyx: [number, number, number]; n: number } | null;
  onWindow?: (level: number, width: number) => void;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cache = useRef<Cache>(new Map());
  const drag = useRef<Drag | null>(null);
  const wheelAcc = useRef(0);
  const [box, setBox] = useState<Viewport>({ width: 300, height: 300 });
  const info = planeInfo(image.shape, image.spacing, plane);
  const [view, setView] = useState<ViewState>(() => fitView(info));
  const [panning, setPanning] = useState(false);
  const k = Math.min(info.count - 1, Math.max(0, sliceIndexFor(plane, position)));

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const r = entry.contentRect;
      setBox({ width: Math.max(80, r.width), height: Math.max(80, r.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // A new geometry or an explicit reset starts from the fitted view; switching between
  // scans of the same size keeps the zoom so they can be compared in place.
  useEffect(() => setView(fitView(info)), [info.width, info.height, resetSignal]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!focusSignal) return;
    const [c, r] = crosshair(image.shape, plane, focusSignal.zyx);
    setView((v) => (v.zoom > 1 ? centerOn(info, v, c + 0.5, r + 0.5) : v));
  }, [focusSignal?.n]); // eslint-disable-line react-hooks/exhaustive-deps

  // ----- cached per-slice layers (rebuilt only when their inputs change) -----
  const base = cached(cache.current, "base", [image, plane, k, level, width], () => {
    const c = document.createElement("canvas");
    c.width = info.width;
    c.height = info.height;
    const ctx = c.getContext("2d");
    if (!ctx) return c;
    const img = ctx.createImageData(info.width, info.height);
    const lo = level - width / 2;
    const inv = 255 / Math.max(width, 1e-6);
    for (let row = 0; row < info.height; row++)
      for (let col = 0; col < info.width; col++) {
        const g = Math.min(255, Math.max(0, (voxelValue(image, voxelIndex(image.shape, plane, k, col, row)) - lo) * inv));
        const p = (row * info.width + col) * 4;
        img.data[p] = img.data[p + 1] = img.data[p + 2] = g;
        img.data[p + 3] = 255;
      }
    ctx.putImageData(img, 0, 0);
    return c;
  });

  const layers = overlays.map((ov) => {
    if (ov.type === "map") {
      return {
        ov,
        bitmap: cached(cache.current, `map:${ov.id}`, [ov.data, plane, k, ov.lut, ov.threshold, ov.vmax], () => {
          const c = document.createElement("canvas");
          c.width = info.width;
          c.height = info.height;
          const ctx = c.getContext("2d");
          if (!ctx) return c;
          const img = ctx.createImageData(info.width, info.height);
          const lut = LUTS[ov.lut ?? "dose"];
          const vmax = ov.vmax ?? 1;
          const cut = (ov.threshold ?? 0.02) * vmax;
          for (let row = 0; row < info.height; row++)
            for (let col = 0; col < info.width; col++) {
              const v = ov.data.data[voxelIndex(image.shape, plane, k, col, row)] * ov.data.scale + ov.data.offset;
              if (v < cut) continue;
              const li = Math.min(255, Math.max(0, Math.round((v / vmax) * 255))) * 3;
              const p = (row * info.width + col) * 4;
              img.data[p] = lut[li];
              img.data[p + 1] = lut[li + 1];
              img.data[p + 2] = lut[li + 2];
              img.data[p + 3] = 255;
            }
          ctx.putImageData(img, 0, 0);
          return c;
        }),
        outlines: null,
      };
    }
    return {
      ov,
      bitmap: null,
      outlines: cached(cache.current, `label:${ov.id}`, [ov.data, plane, k, ov.only?.join(","), ov.revision], () => {
        const labels = sliceLabels(ov.data, plane, k);
        const values = [...new Set(labels)].filter((v) => v > 0 && (!ov.only || ov.only.includes(v)));
        return values.map((value) => ({ value, loops: traceContours(labels, info.width, info.height, value) }));
      }) as { value: number; loops: Loop[] }[],
    };
  });

  // ----- draw: cheap, runs on every view, window, crosshair or style change -----
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const bw = Math.round(box.width * dpr);
    const bh = Math.round(box.height * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, box.width, box.height);
    const [x0, y0] = planeToScreen(info, box, view, 0, 0);
    const [kx, ky] = pixelSize(info, box, view.zoom);
    ctx.imageSmoothingEnabled = smooth;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(base, x0, y0, info.width * kx, info.height * ky);

    for (const { ov, bitmap, outlines } of layers) {
      if (bitmap) {
        ctx.globalAlpha = ov.opacity;
        ctx.drawImage(bitmap, x0, y0, info.width * kx, info.height * ky);
        ctx.globalAlpha = 1;
        continue;
      }
      const style = ov.style ?? "both";
      for (const { value, loops } of outlines ?? []) {
        const c = (ov.colors ?? LESION_COLORS)[value];
        if (!c || (c[0] === 0 && c[1] === 0 && c[2] === 0)) continue;
        const path = new Path2D();
        for (const loop of loops) {
          for (let p = 0; p < loop.length; p += 2) {
            const sx = x0 + loop[p] * kx;
            const sy = y0 + loop[p + 1] * ky;
            if (p === 0) path.moveTo(sx, sy);
            else path.lineTo(sx, sy);
          }
          path.closePath();
        }
        if (style !== "outline") {
          ctx.fillStyle = rgba(c, style === "fill" ? ov.opacity : ov.opacity * 0.5);
          ctx.fill(path, "evenodd");
        }
        if (style !== "fill") {
          ctx.strokeStyle = rgba(c, ov.opacity);
          ctx.lineWidth = 1.5;
          ctx.lineJoin = "round";
          ctx.stroke(path);
        }
      }
    }

    if (showCrosshair) {
      const [cc, rr] = crosshair(image.shape, plane, position);
      const [x, y] = planeToScreen(info, box, view, cc + 0.5, rr + 0.5);
      ctx.strokeStyle = "rgba(76,155,255,0.55)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, 0);
      ctx.lineTo(Math.round(x) + 0.5, box.height);
      ctx.moveTo(0, Math.round(y) + 0.5);
      ctx.lineTo(box.width, Math.round(y) + 0.5);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  });

  // ----- interaction -----
  const setSlice = useCallback(
    (next: number) => {
      const s = Math.min(info.count - 1, Math.max(0, next));
      if (s === k) return;
      const pos: [number, number, number] = [...position];
      if (plane === "axial") pos[0] = s;
      else if (plane === "coronal") pos[1] = s;
      else pos[2] = s;
      onPosition(pos);
    },
    [info.count, k, position, plane, onPosition],
  );

  const local = (e: { clientX: number; clientY: number }): [number, number] => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  };
  const voxelAt = (x: number, y: number): [number, number] | null => {
    const [u, v] = screenToPlane(info, box, view, x, y);
    const col = Math.floor(u);
    const row = Math.floor(v);
    return !(col >= 0 && row >= 0 && col < info.width && row < info.height) ? null : [col, row];
  };

  const paintAt = (col: number, row: number) => {
    if (!brush) return;
    const idx: number[] = [];
    const r = brush.radius;
    for (let dr = -r; dr <= r; dr++)
      for (let dc = -r; dc <= r; dc++) {
        if (dc * dc + dr * dr > r * r) continue;
        const c = col + dc;
        const rr = row + dr;
        if (c < 0 || rr < 0 || c >= info.width || rr >= info.height) continue;
        idx.push(voxelIndex(image.shape, plane, k, c, rr));
      }
    brush.onPaint(idx, brush.value);
  };

  const act = (d: Drag, x: number, y: number) => {
    if (d.mode === "pan") {
      // Take the delta now: the updater may run after the next move has advanced d.
      const dx = x - d.x;
      const dy = y - d.y;
      d.x = x;
      d.y = y;
      setView((v) => panBy(info, box, v, dx, dy));
    } else if (d.mode === "window") {
      // Horizontal drag changes the width, vertical drag the level (drag up = brighter).
      onWindow?.(d.level + (y - d.y) * 2, Math.max(1, d.width + (x - d.x) * 4));
    } else {
      const p = voxelAt(x, y);
      if (!p) return;
      if (d.mode === "paint") paintAt(p[0], p[1]);
      else onPosition(pixelToVoxel(image.shape, plane, k, p[0], p[1]));
    }
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    wrapRef.current?.focus({ preventScroll: true });
    const [x, y] = local(e);
    const mode: Drag["mode"] =
      e.button === 1 || e.button === 2 || e.shiftKey ? "pan" : e.altKey && onWindow ? "window" : brush ? "paint" : "point";
    drag.current = { mode, x, y, level, width, id: e.pointerId };
    e.currentTarget.setPointerCapture?.(e.pointerId);
    if (mode === "pan") setPanning(true);
    act(drag.current, x, y);
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    if (!d || d.id !== e.pointerId) return;
    const [x, y] = local(e);
    act(d, x, y);
  };
  const endDrag = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (drag.current?.id !== e.pointerId) return;
    drag.current = null;
    setPanning(false);
  };

  // Native listener: React's wheel handler is passive, so it cannot stop the page scrolling.
  const wheelState = useRef({ info, box, view, k, setSlice });
  wheelState.current = { info, box, view, k, setSlice };
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const s = wheelState.current;
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? s.box.height : 1;
      const dy = e.deltaY * unit;
      if (e.ctrlKey || e.metaKey) {
        // Pinch gestures arrive as ctrl+wheel with small deltas; mouse notches are clamped.
        const rect = el.getBoundingClientRect();
        const factor = Math.exp(-Math.max(-50, Math.min(50, dy)) * 0.01);
        setView((v) => zoomAt(s.info, s.box, v, factor, e.clientX - rect.left, e.clientY - rect.top));
        return;
      }
      // One slice per mouse notch; trackpads accumulate small deltas, at most one slice per event.
      if (Math.abs(dy) >= SLICE_PX) {
        wheelAcc.current = 0;
        s.setSlice(s.k + Math.sign(dy));
        return;
      }
      if (Math.sign(dy) !== Math.sign(wheelAcc.current)) wheelAcc.current = 0;
      wheelAcc.current += dy;
      if (Math.abs(wheelAcc.current) >= SLICE_PX) {
        s.setSlice(s.k + Math.sign(wheelAcc.current));
        wheelAcc.current -= Math.sign(wheelAcc.current) * SLICE_PX;
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const zoomBy = (factor: number) => setView((v) => zoomAt(info, box, v, factor, box.width / 2, box.height / 2));
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.target !== e.currentTarget) return; // let the slider handle its own keys
    const page = Math.max(1, Math.round(info.count / 10));
    const handled: Record<string, () => void> = {
      ArrowUp: () => setSlice(k + 1),
      ArrowRight: () => setSlice(k + 1),
      ArrowDown: () => setSlice(k - 1),
      ArrowLeft: () => setSlice(k - 1),
      PageUp: () => setSlice(k + page),
      PageDown: () => setSlice(k - page),
      Home: () => setSlice(0),
      End: () => setSlice(info.count - 1),
      "+": () => zoomBy(1.25),
      "=": () => zoomBy(1.25),
      "-": () => zoomBy(0.8),
      "0": () => setView(fitView(info)),
    };
    const fn = handled[e.key];
    if (!fn) return;
    e.preventDefault();
    fn();
  };

  const [kx] = pixelSize(info, box, view.zoom);
  const [l, rt, t, b] = ORIENT[plane];
  const offset = sliceOffsetMm(image, plane, k);
  return (
    <div className="flex h-full min-h-0 w-full min-w-0 flex-col overflow-hidden rounded-lg border border-ink-700/70 bg-black">
      <div
        ref={wrapRef}
        tabIndex={0}
        onKeyDown={onKeyDown}
        aria-label={`${title ?? plane} view. Arrow keys change slice, plus and minus zoom, 0 resets.`}
        className="group relative min-h-0 flex-1 overflow-hidden outline-none ring-inset focus-visible:ring-1 focus-visible:ring-accent/70"
      >
        {/* Absolutely positioned so the canvas never feeds back into the container's measured size. */}
        <canvas
          ref={canvasRef}
          data-testid={`slice-${plane}`}
          aria-label={`${plane} slice ${k + 1} of ${info.count}`}
          className={cx("absolute inset-0 h-full w-full touch-none", panning ? "cursor-grabbing" : brush ? "cursor-crosshair" : "cursor-pointer")}
          style={{ width: box.width, height: box.height }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onDoubleClick={() => setView(fitView(info))}
          onContextMenu={(e) => e.preventDefault()}
        />
        {!compact && (
          <div className="pointer-events-none absolute left-2 top-1.5 flex items-center gap-2 text-[10.5px] font-semibold uppercase tracking-wider text-ink-300">
            <span>{title ?? plane}</span>
            <span className="font-mono text-ink-500">
              {k + 1}/{info.count}
            </span>
          </div>
        )}
        <div className="absolute right-1.5 top-1 flex items-center gap-1">
          <div className="flex gap-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
            <ZoomButton label="Zoom out" onClick={() => zoomBy(0.8)}>
              <Minus className="h-3 w-3" />
            </ZoomButton>
            <ZoomButton label="Zoom in" onClick={() => zoomBy(1.25)}>
              <Plus className="h-3 w-3" />
            </ZoomButton>
            <ZoomButton label="Fit to view" onClick={() => setView(fitView(info))}>
              <Maximize2 className="h-3 w-3" />
            </ZoomButton>
          </div>
        </div>
        <span className="pointer-events-none absolute left-1.5 top-1/2 -translate-y-1/2 text-[10px] font-semibold text-ink-400">{l}</span>
        <span className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 text-[10px] font-semibold text-ink-400">{rt}</span>
        {!compact && (
          <>
            <span className="pointer-events-none absolute left-1/2 top-1 -translate-x-1/2 text-[10px] font-semibold text-ink-400">{t}</span>
            <span className="pointer-events-none absolute bottom-1 left-1/2 -translate-x-1/2 text-[10px] font-semibold text-ink-400">{b}</span>
          </>
        )}
      </div>
      <div className="flex items-center gap-2 border-t border-ink-800 bg-ink-950 px-2 py-1">
        <input
          type="range"
          min={0}
          max={info.count - 1}
          value={k}
          onChange={(e) => setSlice(Number(e.target.value))}
          aria-label={`${title ?? plane} slice`}
          className="slice-slider h-1 min-w-0 flex-1"
        />
        <span className="shrink-0 font-mono text-[10px] text-ink-500">
          {AXIS[plane]} {offset === null ? "" : `${offset.toFixed(1)} mm`}
        </span>
        <span className="shrink-0 font-mono text-[10px] text-ink-400" title={`Zoom relative to fit; each voxel spans ${kx.toFixed(1)} screen pixels`}>
          {Math.round(view.zoom * 100)}%{box.width >= 300 ? ` · ${kx.toFixed(1)} px/vox` : ""}
        </span>
      </div>
    </div>
  );
}

function ZoomButton({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="flex h-5 w-5 items-center justify-center rounded border border-ink-700 bg-ink-900/90 text-ink-300 hover:text-ink-100"
    >
      {children}
    </button>
  );
}
