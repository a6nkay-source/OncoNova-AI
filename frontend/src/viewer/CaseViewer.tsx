import { useEffect, useMemo, useState, type ReactNode } from "react";
import { api, type VolumeData } from "../api/client";
import { ErrorState, Segmented, Spinner, cx } from "../components/ui";
import { useVolume, useVolumes } from "../state/hooks";
import { SliceView, type Brush, type Overlay } from "./SliceView";
import type { LutName, Plane, RGB } from "./volume";

export interface OverlaySpec {
  id: string;
  url: string | null;
  type: "label" | "map";
  opacity: number;
  colors?: Record<number, RGB>;
  outline?: boolean;
  only?: number[];
  lut?: LutName;
  vmax?: number;
  threshold?: number;
  visible?: boolean;
  /** Supply data directly (e.g. an annotation being edited) instead of fetching. */
  data?: VolumeData | null;
}

const CT_PRESETS = {
  lung: { level: -600, width: 1500, label: "Lung" },
  mediastinum: { level: 50, width: 350, label: "Mediast." },
  bone: { level: 400, width: 2000, label: "Bone" },
} as const;
type Preset = keyof typeof CT_PRESETS | "auto";

function autoWindow(v: VolumeData): { level: number; width: number } {
  const n = v.data.length;
  const sample: number[] = [];
  const step = Math.max(1, Math.floor(n / 20000));
  for (let i = 0; i < n; i += step) sample.push(v.data[i] * v.scale + v.offset);
  sample.sort((a, b) => a - b);
  const lo = sample[Math.floor(sample.length * 0.01)];
  const hi = sample[Math.floor(sample.length * 0.995)];
  return { level: (lo + hi) / 2, width: Math.max(hi - lo, 1) };
}

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
  defaultPreset?: Preset;
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
  const [pos, setPos] = useState<[number, number, number] | null>(null);

  useEffect(() => {
    if (image && (!pos || pos[0] >= image.shape[0] || pos[1] >= image.shape[1] || pos[2] >= image.shape[2])) {
      setPos([Math.floor(image.shape[0] / 2), Math.floor(image.shape[1] / 2), Math.floor(image.shape[2] / 2)]);
    }
  }, [image, pos]);
  useEffect(() => {
    if (focus) setPos([Math.round(focus[0]), Math.round(focus[1]), Math.round(focus[2])]);
  }, [focus]);
  useEffect(() => {
    if (syncPosition) setPos(syncPosition);
  }, [syncPosition]);

  const win = useMemo(() => {
    if (!image) return { level: 0, width: 1 };
    if (preset === "auto" || modality !== "CT") return autoWindow(image);
    return CT_PRESETS[preset];
  }, [image, preset, modality]);

  const ready: Overlay[] = [];
  visible.forEach((o, i) => {
    const data = o.data ?? volumes[i];
    if (data && image && data.data.length === image.data.length) ready.push({ ...o, data } as Overlay);
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
    />
  );

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
        <span className="mono text-ink-500">
          W {Math.round(win.width)} / L {Math.round(win.level)}
        </span>
        <span className="mono ml-auto text-ink-500">
          z{pos[0]} y{pos[1]} x{pos[2]}
        </span>
        {toolbar}
      </div>
      {overlayError ? <p className="text-[12px] text-ev-uncertain">An overlay could not be loaded.</p> : null}
      {layout === "tri" && (
        <div className="grid grid-cols-1 gap-2 lg:h-[clamp(440px,64vh,760px)] lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          <div className="h-[380px] lg:h-full">{view("axial", { brush })}</div>
          <div className="grid h-[480px] grid-rows-2 gap-2 lg:h-full">
            {view("coronal")}
            {view("sagittal")}
          </div>
        </div>
      )}
      {layout === "row" && (
        <div className="grid grid-cols-1 gap-2 md:h-[300px] md:grid-cols-3">
          <div className="h-[260px] md:h-full">{view("axial", { brush })}</div>
          <div className="h-[260px] md:h-full">{view("coronal")}</div>
          <div className="h-[260px] md:h-full">{view("sagittal")}</div>
        </div>
      )}
      {layout === "axial" && <div className="h-[360px]">{view("axial", { brush })}</div>}
    </div>
  );
}
