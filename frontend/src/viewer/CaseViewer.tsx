import { RotateCcw } from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { api, type VolumeData } from "../api/client";
import { ErrorState, Segmented, Spinner, cx } from "../components/ui";
import { useVolume, useVolumes } from "../state/hooks";
import { SliceView, type Brush, type LabelStyle, type Overlay } from "./SliceView";
import { LUTS, type LutName, type Plane, type RGB } from "./volume";

/** One legend row: the label values it covers and their default colour. */
export interface LegendEntry {
  name: string;
  values: number[];
  color: RGB;
}

export interface OverlaySpec {
  id: string;
  /** Shown next to the overlay's controls. */
  name: string;
  url: string | null;
  type: "label" | "map";
  /** Defaults; the viewer lets the user change them. */
  opacity: number;
  style?: LabelStyle;
  legend?: LegendEntry[];
  only?: number[];
  lut?: LutName;
  vmax?: number;
  threshold?: number;
  visible?: boolean;
  /** Supply data directly (e.g. an annotation being edited) instead of fetching. */
  data?: VolumeData | null;
  revision?: number;
}

const CT_PRESETS = {
  lung: { level: -600, width: 1500, label: "Lung" },
  mediastinum: { level: 50, width: 350, label: "Mediast." },
  bone: { level: 400, width: 2000, label: "Bone" },
} as const;
type Preset = keyof typeof CT_PRESETS | "auto" | "custom";
type Win = { level: number; width: number };

function autoWindow(v: VolumeData): Win {
  const n = v.data.length;
  const sample: number[] = [];
  const step = Math.max(1, Math.floor(n / 20000));
  for (let i = 0; i < n; i += step) sample.push(v.data[i] * v.scale + v.offset);
  sample.sort((a, b) => a - b);
  const lo = sample[Math.floor(sample.length * 0.01)];
  const hi = sample[Math.floor(sample.length * 0.995)];
  return { level: (lo + hi) / 2, width: Math.max(hi - lo, 1) };
}

// ----- per-viewer display preferences (a convenience; the page works without storage) -----

interface OverlayPrefs {
  opacity?: number;
  style?: LabelStyle;
  colors?: Record<string, string>; // legend entry name -> #rrggbb
}
interface Prefs {
  smooth: boolean;
  overlays: Record<string, OverlayPrefs>;
}
const PREFS_KEY = "onconova.viewer.v1";

function loadPrefs(): Prefs {
  try {
    const raw = JSON.parse(localStorage.getItem(PREFS_KEY) ?? "null");
    if (raw && typeof raw === "object") return { smooth: raw.smooth !== false, overlays: raw.overlays ?? {} };
  } catch {
    /* storage unavailable */
  }
  return { smooth: true, overlays: {} };
}

const hex = (c: RGB) => `#${c.map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`;
const rgb = (h: string): RGB => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];

export function CaseViewer({
  caseId,
  tpId,
  imageUrl,
  modality = "CT",
  overlays = [],
  layout = "tri",
  focus,
  brush,
  toolbar,
  version = 0,
  className,
  defaultPreset = "lung",
  onPositionChange,
  syncPosition,
}: {
  caseId: string;
  tpId: string;
  imageUrl?: string;
  modality?: string;
  overlays?: OverlaySpec[];
  layout?: "tri" | "axial" | "row";
  focus?: [number, number, number] | null;
  brush?: Brush | null;
  toolbar?: ReactNode;
  version?: number;
  className?: string;
  defaultPreset?: Exclude<Preset, "custom">;
  onPositionChange?: (zyx: [number, number, number]) => void;
  syncPosition?: [number, number, number] | null;
}) {
  const { volume: image, error, loading } = useVolume(imageUrl ?? api.imageUrl(caseId, tpId), version);
  const visible = overlays.filter((o) => o.visible !== false);
  const { volumes, errors } = useVolumes(
    visible.map((o) => (o.data ? null : o.url)),
    version,
  );
  const [preset, setPreset] = useState<Preset>(modality === "CT" ? defaultPreset : "auto");
  const [custom, setCustom] = useState<Win | null>(null);
  const [pos, setPos] = useState<[number, number, number] | null>(null);
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs);
  const [resetSignal, setResetSignal] = useState(0);
  const [focusSignal, setFocusSignal] = useState<{ zyx: [number, number, number]; n: number } | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
    } catch {
      /* storage unavailable */
    }
  }, [prefs]);

  useEffect(() => {
    if (image && (!pos || pos[0] >= image.shape[0] || pos[1] >= image.shape[1] || pos[2] >= image.shape[2])) {
      setPos([Math.floor(image.shape[0] / 2), Math.floor(image.shape[1] / 2), Math.floor(image.shape[2] / 2)]);
    }
  }, [image, pos]);
  useEffect(() => {
    if (!focus) return;
    const zyx: [number, number, number] = [Math.round(focus[0]), Math.round(focus[1]), Math.round(focus[2])];
    setPos(zyx);
    setFocusSignal((s) => ({ zyx, n: (s?.n ?? 0) + 1 }));
  }, [focus]);
  useEffect(() => {
    if (syncPosition) setPos(syncPosition);
  }, [syncPosition]);

  const win = useMemo<Win>(() => {
    if (!image) return { level: 0, width: 1 };
    if (preset === "custom" && custom) return custom;
    if (preset === "auto" || preset === "custom" || modality !== "CT") return autoWindow(image);
    return CT_PRESETS[preset];
  }, [image, preset, custom, modality]);
  const setWindow = (level: number, width: number) => {
    setCustom({ level: Math.round(level), width: Math.max(1, Math.round(width)) });
    setPreset("custom");
  };

  const patchOverlay = (id: string, patch: OverlayPrefs) =>
    setPrefs((p) => ({ ...p, overlays: { ...p.overlays, [id]: { ...p.overlays[id], ...patch, colors: { ...p.overlays[id]?.colors, ...patch.colors } } } }));
  const colorOf = (o: OverlaySpec, e: LegendEntry): RGB => {
    const h = prefs.overlays[o.id]?.colors?.[e.name];
    return h ? rgb(h) : e.color;
  };

  const ready: Overlay[] = [];
  visible.forEach((o, i) => {
    const data = o.data ?? volumes[i];
    if (!data || !image || data.data.length !== image.data.length) return;
    const p = prefs.overlays[o.id] ?? {};
    const colors: Record<number, RGB> = {};
    for (const e of o.legend ?? []) for (const v of e.values) colors[v] = colorOf(o, e);
    ready.push({
      id: o.id,
      data,
      type: o.type,
      opacity: p.opacity ?? o.opacity,
      style: p.style ?? o.style,
      colors: o.legend ? colors : undefined,
      only: o.only,
      lut: o.lut,
      vmax: o.vmax,
      threshold: o.threshold,
      revision: o.revision,
    });
  });
  const overlayError = errors.find(Boolean);

  if (error) return <ErrorState error={error} />;
  if (!image || !pos)
    return (
      <div className={cx("flex h-full min-h-[320px] items-center justify-center", className)}>
        {loading ? <Spinner label="Loading volume" /> : null}
      </div>
    );

  const setPosition = (p: [number, number, number]) => {
    setPos(p);
    onPositionChange?.(p);
  };
  const view = (plane: Plane, extra?: { brush?: Brush | null; compact?: boolean }) => (
    <SliceView
      image={image}
      plane={plane}
      position={pos}
      onPosition={setPosition}
      level={win.level}
      width={win.width}
      overlays={ready}
      brush={plane === "axial" ? extra?.brush : null}
      compact={extra?.compact}
      smooth={prefs.smooth}
      resetSignal={resetSignal}
      focusSignal={focusSignal}
      onWindow={setWindow}
    />
  );
  const [nz, ny, nx] = image.shape;
  const [sx, sy, sz] = image.spacing;

  return (
    <div className={cx("flex h-full min-h-0 flex-col gap-2", className)}>
      <div className="flex flex-wrap items-center gap-2">
        {modality === "CT" ? (
          <Segmented<Preset>
            size="xs"
            value={preset}
            onChange={setPreset}
            options={[
              ...Object.entries(CT_PRESETS).map(([k, v]) => ({ value: k as Preset, label: v.label })),
              { value: "auto", label: "Auto" },
            ]}
          />
        ) : (
          <span className="label">Auto window</span>
        )}
        <NumberField label="W" value={win.width} min={1} max={6000} onCommit={(w) => setWindow(win.level, w)} />
        <NumberField label="L" value={win.level} min={-2000} max={4000} onCommit={(l) => setWindow(l, win.width)} />
        {preset === "custom" && <span className="text-[11px] text-ink-500">custom</span>}
        {toolbar}
        <div className="ml-auto flex items-center gap-2">
          <Segmented<"smooth" | "voxels">
            size="xs"
            value={prefs.smooth ? "smooth" : "voxels"}
            onChange={(v) => setPrefs((p) => ({ ...p, smooth: v === "smooth" }))}
            options={[
              { value: "smooth", label: "Smooth", title: "Interpolate the CT image between voxels (display only)" },
              { value: "voxels", label: "Voxels", title: "Show each voxel as a square, exactly as stored" },
            ]}
          />
          <button type="button" className="btn-ghost !px-2 !py-0.5 text-[11px]" onClick={() => setResetSignal((n) => n + 1)} title="Fit every view again">
            <RotateCcw className="h-3 w-3" /> Fit views
          </button>
        </div>
      </div>
      {overlayError ? <p className="text-[12px] text-ev-uncertain">An overlay could not be loaded.</p> : null}
      {layout === "tri" && (
        <div className="grid grid-cols-1 gap-2 lg:h-[clamp(460px,66vh,800px)] lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          <div className="h-[400px] lg:h-full">{view("axial", { brush })}</div>
          <div className="grid h-[520px] grid-rows-2 gap-2 lg:h-full">
            {view("coronal")}
            {view("sagittal")}
          </div>
        </div>
      )}
      {layout === "row" && (
        <div className="grid grid-cols-1 gap-2 md:h-[320px] md:grid-cols-3">
          <div className="h-[280px] md:h-full">{view("axial", { brush })}</div>
          <div className="h-[280px] md:h-full">{view("coronal")}</div>
          <div className="h-[280px] md:h-full">{view("sagittal")}</div>
        </div>
      )}
      {layout === "axial" && <div className="h-[380px]">{view("axial", { brush })}</div>}

      {visible.some((o) => o.name) && (
        <div className="flex flex-col gap-1.5 rounded-lg border border-ink-800 bg-ink-925/60 px-3 py-2">
          {visible.map((o) => (
            <OverlayControls
              key={o.id}
              spec={o}
              prefs={prefs.overlays[o.id] ?? {}}
              colorOf={(e) => colorOf(o, e)}
              onChange={(patch) => patchOverlay(o.id, patch)}
            />
          ))}
        </div>
      )}
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-[11px] text-ink-500">
        <span className="font-mono">
          Matrix {nx}×{ny}×{nz} · voxel {sx.toFixed(2)}×{sy.toFixed(2)} mm · slice spacing {sz.toFixed(2)} mm
        </span>
        <span className="font-mono">
          z{pos[0]} y{pos[1]} x{pos[2]}
        </span>
        <span className="w-full">
          Scroll, arrow keys or the slider: change slice · ⌘/Ctrl + scroll or pinch: zoom · Shift- or right-drag: pan · Alt-drag: window/level ·
          double-click: fit
        </span>
      </div>
    </div>
  );
}

function NumberField({ label, value, min, max, onCommit }: { label: string; value: number; min: number; max: number; onCommit: (v: number) => void }) {
  const [draft, setDraft] = useState(String(Math.round(value)));
  useEffect(() => setDraft(String(Math.round(value))), [value]);
  const commit = () => {
    const v = Number(draft);
    if (draft.trim() !== "" && Number.isFinite(v)) onCommit(Math.min(max, Math.max(min, v)));
    else setDraft(String(Math.round(value)));
  };
  return (
    <label className="flex items-center gap-1 text-[11px] text-ink-400">
      {label}
      <input
        inputMode="numeric"
        aria-label={label === "W" ? "Window width" : "Window level"}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && commit()}
        className="w-[58px] rounded-md border border-ink-700 bg-ink-925 px-1.5 py-0.5 font-mono text-[11px] text-ink-200 focus:border-accent focus:outline-none"
      />
    </label>
  );
}

const STYLES: { value: LabelStyle; label: string }[] = [
  { value: "both", label: "Fill + line" },
  { value: "outline", label: "Line" },
  { value: "fill", label: "Fill" },
];

function lutGradient(name: LutName): string {
  const lut = LUTS[name];
  const stops = [0, 64, 128, 192, 255].map((i) => `rgb(${lut[i * 3]},${lut[i * 3 + 1]},${lut[i * 3 + 2]}) ${Math.round((i / 255) * 100)}%`);
  return `linear-gradient(90deg, ${stops.join(", ")})`;
}

function OverlayControls({
  spec,
  prefs,
  colorOf,
  onChange,
}: {
  spec: OverlaySpec;
  prefs: OverlayPrefs;
  colorOf: (e: LegendEntry) => RGB;
  onChange: (patch: OverlayPrefs) => void;
}) {
  const opacity = prefs.opacity ?? spec.opacity;
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[11.5px] text-ink-300">
      <span className="min-w-[96px] font-medium text-ink-200">{spec.name}</span>
      {spec.type === "label" ? (
        (spec.legend ?? []).map((e) => (
          <label key={e.name} className="flex cursor-pointer items-center gap-1.5" title={`Change the colour of ${e.name}`}>
            <input type="color" className="swatch" value={hex(colorOf(e))} onChange={(ev) => onChange({ colors: { [e.name]: ev.target.value } })} />
            {e.name}
          </label>
        ))
      ) : (
        <span className="flex items-center gap-1.5 text-ink-400">
          low
          <span className="h-2 w-20 rounded-sm" style={{ background: lutGradient(spec.lut ?? "dose") }} />
          high
        </span>
      )}
      <div className="ml-auto flex items-center gap-3">
        {spec.type === "label" && (
          <Segmented<LabelStyle> size="xs" value={prefs.style ?? spec.style ?? "both"} onChange={(style) => onChange({ style })} options={STYLES} />
        )}
        <label className="flex items-center gap-1.5 text-ink-400">
          Opacity
          <input
            type="range"
            min={0.1}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => onChange({ opacity: Number(e.target.value) })}
            aria-label={`${spec.name} opacity`}
            className="slice-slider"
            style={{ width: 80 }}
          />
          <span className="w-8 text-right font-mono text-[10.5px]">{Math.round(opacity * 100)}%</span>
        </label>
      </div>
    </div>
  );
}
