import { useCallback, useEffect, useRef, useState } from "react";
import type { VolumeData } from "../api/client";
import { cx } from "../components/ui";
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
  voxelIndex,
  voxelValue,
} from "./volume";

export interface Overlay {
  id: string;
  data: VolumeData;
  type: "label" | "map";
  opacity: number;
  colors?: Record<number, RGB>;
  outline?: boolean;
  only?: number[]; // label values to draw
  lut?: LutName;
  threshold?: number; // map values below this are transparent
  vmax?: number;
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
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offRef = useRef<HTMLCanvasElement | null>(null);
  const [box, setBox] = useState({ w: 300, h: 300 });
  const dragging = useRef(false);
  const info = planeInfo(image.shape, image.spacing, plane);
  const k = Math.min(info.count - 1, Math.max(0, sliceIndexFor(plane, position)));

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const r = entry.contentRect;
      setBox({ w: Math.max(80, r.width), h: Math.max(80, r.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Fit the physical aspect ratio inside the container.
  const scale = Math.min(box.w / info.widthMm, box.h / info.heightMm);
  const dispW = Math.floor(info.widthMm * scale);
  const dispH = Math.floor(info.heightMm * scale);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!offRef.current) offRef.current = document.createElement("canvas");
    const off = offRef.current;
    off.width = info.width;
    off.height = info.height;
    const octx = off.getContext("2d");
    if (!octx) return;
    const img = octx.createImageData(info.width, info.height);
    const px = img.data;
    const lo = level - width / 2;
    const inv = 255 / Math.max(width, 1e-6);
    const { shape } = image;
    for (let row = 0; row < info.height; row++) {
      for (let col = 0; col < info.width; col++) {
        const vi = voxelIndex(shape, plane, k, col, row);
        const g = Math.min(255, Math.max(0, (voxelValue(image, vi) - lo) * inv));
        let r = g,
          gg = g,
          b = g;
        for (const ov of overlays) {
          const raw = ov.data.data[vi];
          if (ov.type === "label") {
            if (!raw || (ov.only && !ov.only.includes(raw))) continue;
            const c = (ov.colors ?? LESION_COLORS)[raw];
            if (!c || (c[0] === 0 && c[1] === 0 && c[2] === 0)) continue;
            let a = ov.opacity;
            if (ov.outline) {
              const nb = (dc: number, dr: number) => {
                const cc = col + dc,
                  rr = row + dr;
                if (cc < 0 || rr < 0 || cc >= info.width || rr >= info.height) return 0;
                return ov.data.data[voxelIndex(shape, plane, k, cc, rr)];
              };
              const edge = nb(1, 0) !== raw || nb(-1, 0) !== raw || nb(0, 1) !== raw || nb(0, -1) !== raw;
              if (!edge) continue;
              a = Math.min(1, ov.opacity + 0.35);
            }
            r = r * (1 - a) + c[0] * a;
            gg = gg * (1 - a) + c[1] * a;
            b = b * (1 - a) + c[2] * a;
          } else {
            const v = raw * ov.data.scale + ov.data.offset;
            const vmax = ov.vmax ?? 1;
            if (v < (ov.threshold ?? 0.02) * vmax) continue;
            const li = Math.min(255, Math.max(0, Math.round((v / vmax) * 255))) * 3;
            const lut = LUTS[ov.lut ?? "dose"];
            const a = ov.opacity;
            r = r * (1 - a) + lut[li] * a;
            gg = gg * (1 - a) + lut[li + 1] * a;
            b = b * (1 - a) + lut[li + 2] * a;
          }
        }
        const p = (row * info.width + col) * 4;
        px[p] = r;
        px[p + 1] = gg;
        px[p + 2] = b;
        px[p + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(dispW * dpr));
    canvas.height = Math.max(1, Math.floor(dispH * dpr));
    canvas.style.width = `${dispW}px`;
    canvas.style.height = `${dispH}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, 0, 0, dispW, dispH);
    if (showCrosshair) {
      const [cc, rr] = crosshair(image.shape, plane, position);
      const x = ((cc + 0.5) / info.width) * dispW;
      const y = ((rr + 0.5) / info.height) * dispH;
      ctx.strokeStyle = "rgba(76,155,255,0.55)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, dispH);
      ctx.moveTo(0, y);
      ctx.lineTo(dispW, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [image, plane, k, level, width, overlays, dispW, dispH, position, showCrosshair, info.width, info.height]);

  const eventToPixel = useCallback(
    (e: React.MouseEvent): [number, number] | null => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return null;
      const col = Math.floor(((e.clientX - rect.left) / rect.width) * info.width);
      const row = Math.floor(((e.clientY - rect.top) / rect.height) * info.height);
      if (col < 0 || row < 0 || col >= info.width || row >= info.height) return null;
      return [col, row];
    },
    [info.width, info.height],
  );

  const paintAt = useCallback(
    (col: number, row: number) => {
      if (!brush) return;
      const idx: number[] = [];
      const r = brush.radius;
      for (let dr = -r; dr <= r; dr++)
        for (let dc = -r; dc <= r; dc++) {
          if (dc * dc + dr * dr > r * r) continue;
          const c = col + dc,
            rr = row + dr;
          if (c < 0 || rr < 0 || c >= info.width || rr >= info.height) continue;
          idx.push(voxelIndex(image.shape, plane, k, c, rr));
        }
      brush.onPaint(idx, brush.value);
    },
    [brush, image.shape, plane, k, info.width, info.height],
  );

  const handle = (e: React.MouseEvent) => {
    const p = eventToPixel(e);
    if (!p) return;
    if (brush) paintAt(p[0], p[1]);
    else onPosition(pixelToVoxel(image.shape, plane, k, p[0], p[1]));
  };

  const onWheel = (e: React.WheelEvent) => {
    const step = e.deltaY > 0 ? 1 : -1;
    const next = Math.min(info.count - 1, Math.max(0, k + step));
    const pos: [number, number, number] = [...position];
    if (plane === "axial") pos[0] = next;
    else if (plane === "coronal") pos[1] = next;
    else pos[2] = next;
    onPosition(pos);
  };

  const [l, rt, t, b] = ORIENT[plane];
  return (
    <div className="relative flex h-full min-h-0 w-full min-w-0 flex-col overflow-hidden rounded-lg border border-ink-700/70 bg-black">
      {!compact && (
        <div className="pointer-events-none absolute left-2 top-1.5 z-10 flex items-center gap-2 text-[10.5px] font-semibold uppercase tracking-wider text-ink-300">
          <span>{title ?? plane}</span>
          <span className="font-mono text-ink-500">
            {k + 1}/{info.count}
          </span>
        </div>
      )}
      <div ref={wrapRef} className="relative min-h-0 flex-1 overflow-hidden" onWheel={onWheel}>
        {/* Absolutely positioned so the canvas never feeds back into the container's measured size. */}
        <canvas
          ref={canvasRef}
          aria-label={`${plane} slice ${k + 1} of ${info.count}`}
          className={cx("absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2", brush ? "cursor-crosshair" : "cursor-pointer")}
          onMouseDown={(e) => {
            dragging.current = true;
            handle(e);
          }}
          onMouseMove={(e) => dragging.current && handle(e)}
          onMouseUp={() => (dragging.current = false)}
          onMouseLeave={() => (dragging.current = false)}
        />
        <span className="pointer-events-none absolute left-1.5 top-1/2 -translate-y-1/2 text-[10px] font-semibold text-ink-400">{l}</span>
        <span className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 text-[10px] font-semibold text-ink-400">{rt}</span>
        {!compact && (
          <>
            <span className="pointer-events-none absolute left-1/2 top-1 -translate-x-1/2 text-[10px] font-semibold text-ink-400">{t}</span>
            <span className="pointer-events-none absolute bottom-1 left-1/2 -translate-x-1/2 text-[10px] font-semibold text-ink-400">{b}</span>
          </>
        )}
      </div>
    </div>
  );
}
