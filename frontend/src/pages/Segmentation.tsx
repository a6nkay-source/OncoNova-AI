import { Brush as BrushIcon, Copy, Play, Save, Trash2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, fetchVolume, isUnavailable, type VolumeData } from "../api/client";
import type { ComparisonResult, SegmentationSummary, Unavailable } from "../api/types";
import { DEFAULT_ANATOMY, Volume3D, lesionMeshes } from "../components/Volume3D";
import {
  EvidenceBadge,
  ErrorState,
  FindingList,
  Note,
  Panel,
  Segmented,
  Spinner,
  Stat,
  UnavailableState,
  cx,
  fmt,
} from "../components/ui";
import { invalidateVolumes, useAsync } from "../state/hooks";
import { useApp } from "../state/AppState";
import { CaseViewer, type OverlaySpec } from "../viewer/CaseViewer";

type MapMode = "none" | "uncertainty" | "saliency";
const CLASS_LABEL: Record<string, string> = {
  primary_tumor: "Primary lesion",
  lymph_node_region_lesion: "Node-region lesion",
  other_lesion: "Other lesion",
};

export function SegmentationPage() {
  const { current, caseId, tpId, modelId, models, revision, bump } = useApp();
  const model = models.find((m) => m.id === modelId);
  const seg = useAsync(() => api.segmentation(caseId!, tpId!, modelId), [caseId, tpId, modelId, revision], !!caseId && !!tpId);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<unknown>(null);
  const [opacity, setOpacity] = useState(0.55);
  const [showRef, setShowRef] = useState(true);
  const [mapMode, setMapMode] = useState<MapMode>("none");
  const [focus, setFocus] = useState<[number, number, number] | null>(null);
  const [mcSamples, setMcSamples] = useState(8);
  const [view3d, setView3d] = useState<"prediction" | "reference">("prediction");

  const summary = seg.data && !isUnavailable(seg.data) ? (seg.data as SegmentationSummary) : null;

  const run = async (all = false) => {
    if (!caseId || !current) return;
    setRunning(true);
    setRunError(null);
    try {
      const tps = all ? current.timepoints.map((t) => t.tp_id) : [tpId!];
      for (const t of tps) await api.segment(caseId, t, modelId, mcSamples);
      invalidateVolumes(`/api/cases/${caseId}/`);
      bump();
    } catch (e) {
      setRunError(e);
    } finally {
      setRunning(false);
    }
  };

  const annotation = useAnnotation(caseId, tpId);

  if (!current || !caseId || !tpId) return <Spinner label="Loading" />;
  const overlays: OverlaySpec[] = [
    {
      id: "prediction",
      url: summary ? api.mapUrl(caseId, tpId, "prediction", { model_id: modelId }) : null,
      type: "label",
      opacity,
      visible: !!summary && mapMode === "none",
    },
    {
      id: "map",
      url: summary && mapMode !== "none" ? api.mapUrl(caseId, tpId, mapMode, { model_id: modelId }) : null,
      type: "map",
      opacity: 0.75,
      lut: mapMode === "saliency" ? "saliency" : "uncertainty",
      threshold: 0.08,
      visible: mapMode !== "none" && (mapMode !== "saliency" || summary?.has_saliency),
    },
    {
      id: "reference",
      url: current.has_reference ? api.mapUrl(caseId, tpId, "reference") : null,
      type: "label",
      opacity: 0.6,
      outline: true,
      colors: { 1: [255, 255, 255], 2: [255, 255, 255], 3: [255, 255, 255] },
      visible: showRef && current.has_reference && !annotation.editing,
    },
    {
      id: "annotation",
      url: null,
      data: annotation.volume,
      type: "label",
      opacity: 0.8,
      outline: true,
      colors: { 1: [255, 220, 90], 2: [255, 160, 60], 3: [255, 120, 220] },
      visible: !!annotation.volume && (annotation.editing || annotation.hasSaved),
    },
  ];

  return (
    <div className="space-y-4">
      <Panel bodyClassName="flex flex-wrap items-center gap-3 py-3">
        <div className="flex min-w-0 flex-1 items-center gap-3">
          <EvidenceBadge kind="model_output" />
          <div className="min-w-0">
            <div className="truncate text-[14px] font-medium text-ink-100">{model?.name ?? modelId}</div>
            <div className="truncate text-[12px] text-ink-400">
              {model?.status === "available" ? model.label : `Unavailable: ${model?.unavailable_reason ?? ""}`}
            </div>
          </div>
        </div>
        {modelId === "demo-unet" && (
          <label className="flex items-center gap-2 text-[12px] text-ink-400">
            MC-dropout passes
            <select className="input py-1" value={mcSamples} onChange={(e) => setMcSamples(+e.target.value)}>
              {[0, 4, 8, 16].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        )}
        <button className="btn-primary" disabled={running || model?.status !== "available"} onClick={() => run(false)}>
          <Play className="h-3.5 w-3.5" /> {running ? "Running..." : `Segment ${tpId}`}
        </button>
        <button className="btn-ghost" disabled={running || model?.status !== "available"} onClick={() => run(true)}>
          All timepoints
        </button>
      </Panel>
      {runError ? <ErrorState error={runError} /> : null}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_400px]">
        <Panel title="Image and segmentation" className="min-h-[620px]" bodyClassName="flex flex-col gap-3">
          <CaseViewer
            caseId={caseId}
            tpId={tpId}
            modality={current.modality}
            overlays={overlays}
            focus={focus}
            version={revision}
            brush={annotation.editing ? annotation.brush : null}
            toolbar={
              <div className="flex flex-wrap items-center gap-3">
                <Segmented<MapMode>
                  size="xs"
                  value={mapMode}
                  onChange={setMapMode}
                  options={[
                    { value: "none", label: "Mask" },
                    { value: "uncertainty", label: "Uncertainty", disabled: !summary },
                    { value: "saliency", label: "Saliency", disabled: !summary?.has_saliency, title: summary?.has_saliency ? undefined : "Only the deep model provides saliency" },
                  ]}
                />
                <label className="flex items-center gap-1.5 text-[12px] text-ink-300">
                  <input type="checkbox" checked={showRef} disabled={!current.has_reference} onChange={(e) => setShowRef(e.target.checked)} />
                  Reference outline
                </label>
                <label className="flex w-28 items-center gap-2 text-[11px] text-ink-400">
                  Opacity
                  <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(+e.target.value)} aria-label="Mask opacity" />
                </label>
              </div>
            }
          />
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11.5px] text-ink-400">
            <Legend color="#2fd0b8" label="Primary (model)" />
            <Legend color="#f4b740" label="Node-region (model)" />
            <Legend color="#e58cff" label="Other (model)" />
            <Legend color="#ffffff" label="Reference outline (synthetic ground truth)" outline />
            <Legend color="#ffdc5a" label="Manual annotation" outline />
          </div>
          {mapMode === "saliency" && (
            <Note>Saliency is the gradient of the lesion score with respect to input intensities. It shows sensitivity, not the model's reasoning or causes.</Note>
          )}
          <AnnotationTools state={annotation} caseId={caseId} tpId={tpId} hasPrediction={!!summary} modelId={modelId} />
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel title="Segmentation result" actions={summary && <span className="mono text-ink-500">{summary.runtime_s}s</span>}>
            {seg.loading && !seg.data && <Spinner label="Loading" />}
            {seg.error ? <ErrorState error={seg.error} onRetry={seg.reload} /> : null}
            {seg.data && isUnavailable(seg.data) && <UnavailableState data={seg.data as Unavailable} title="Not run yet" />}
            {summary && <SummaryView summary={summary} onFocus={setFocus} />}
          </Panel>
          {summary && <ComparisonPanel caseId={caseId} tpId={tpId} modelId={modelId} hasReference={current.has_reference} hasAnnotation={annotation.hasSaved} revision={revision} />}
          <Panel
            title="3-D view"
            actions={
              <Segmented
                size="xs"
                value={view3d}
                onChange={setView3d}
                options={[
                  { value: "prediction", label: "Model", disabled: !summary },
                  { value: "reference", label: "Reference", disabled: !current.has_reference },
                ]}
              />
            }
          >
            {summary || current.has_reference ? (
              <Volume3D
                caseId={caseId}
                tpId={tpId}
                version={revision}
                meshes={[
                  ...(current.has_structures ? DEFAULT_ANATOMY : []),
                  ...lesionMeshes(view3d === "prediction" && summary ? `pred:${modelId}` : "reference"),
                ]}
              />
            ) : (
              <Note>Run a model to see its segmentation in 3-D.</Note>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}

function Legend({ color, label, outline }: { color: string; label: string; outline?: boolean }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="h-2.5 w-2.5 rounded-sm" style={outline ? { border: `1.5px solid ${color}` } : { background: color }} />
      {label}
    </span>
  );
}

function SummaryView({ summary, onFocus }: { summary: SegmentationSummary; onFocus: (p: [number, number, number]) => void }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-2">
        {Object.entries(summary.volumes_cc).map(([k, v]) => (
          <Stat key={k} label={CLASS_LABEL[k] ?? k} value={fmt(v, 2)} unit="cc" />
        ))}
      </div>
      <div className="flex items-center gap-2 text-[12px] text-ink-400">
        <EvidenceBadge kind="measured_from_model_output" /> Volumes are measured from the model's mask (voxel count × voxel volume).
      </div>
      <div>
        <div className="label mb-1.5">Detected regions</div>
        {summary.lesions.length === 0 ? (
          <p className="text-[13px] text-ink-400">No lesion voxels predicted.</p>
        ) : (
          <table className="table-base">
            <thead>
              <tr>
                <th>Class</th>
                <th className="text-right">cc</th>
                <th className="text-right">Max ⌀ mm</th>
                <th className="text-right">p̄</th>
                <th className="text-right">Unc.</th>
              </tr>
            </thead>
            <tbody>
              {summary.lesions.map((l, i) => (
                <tr key={i} className="cursor-pointer hover:bg-ink-850" onClick={() => onFocus(l.centroid_index_zyx as [number, number, number])}>
                  <td>{CLASS_LABEL[l.class] ?? l.class}</td>
                  <td className="text-right font-mono">{fmt(l.volume_cc, 2)}</td>
                  <td className="text-right font-mono">{fmt(l.max_axial_diameter_mm, 1)}</td>
                  <td className="text-right font-mono">{fmt(l.mean_foreground_probability, 2)}</td>
                  <td className={cx("text-right font-mono", (l.mean_uncertainty ?? 0) > 0.35 && "text-ev-uncertain")}>{fmt(l.mean_uncertainty, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <div className="rounded-lg bg-ink-925 p-3">
        <div className="mb-1 flex items-center justify-between">
          <span className="label">Uncertainty</span>
          <EvidenceBadge kind="uncertain" />
        </div>
        <div className="kv">
          <span className="text-ink-400">Mean inside prediction</span>
          <span className="mono text-right">{fmt(summary.uncertainty.mean_uncertainty_in_prediction, 3)}</span>
          <span className="text-ink-400">High-uncertainty fraction</span>
          <span className="mono text-right">{fmt(summary.uncertainty.high_uncertainty_fraction_of_prediction, 3)}</span>
          <span className="text-ink-400">Method</span>
          <span className="text-right text-[12px] text-ink-300">{summary.uncertainty_method}</span>
        </div>
        {summary.unreliable_regions.length > 0 && (
          <div className="mt-2">
            <div className="label mb-1">Possibly unreliable regions (click to view)</div>
            <div className="flex flex-wrap gap-1.5">
              {summary.unreliable_regions.slice(0, 8).map((r, i) => (
                <button key={i} className="rounded-md border border-ev-uncertain/30 px-2 py-0.5 font-mono text-[11px] text-ev-uncertain hover:bg-ev-uncertain/10" onClick={() => onFocus(r.centroid_index_zyx as [number, number, number])}>
                  {fmt(r.volume_cc, 2)} cc · u {fmt(r.mean_uncertainty, 2)}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
      <div>
        <div className="label mb-1.5">Findings and warnings</div>
        <FindingList items={summary.findings} />
      </div>
    </div>
  );
}

function ComparisonPanel({ caseId, tpId, modelId, hasReference, hasAnnotation, revision }: {
  caseId: string;
  tpId: string;
  modelId: string;
  hasReference: boolean;
  hasAnnotation: boolean;
  revision: number;
}) {
  const [reference, setReference] = useState<"reference" | "annotation">(hasReference ? "reference" : "annotation");
  const cmp = useAsync(() => api.compare(caseId, tpId, modelId, reference), [caseId, tpId, modelId, reference, revision, hasAnnotation]);
  const data = cmp.data && !isUnavailable(cmp.data) ? (cmp.data as ComparisonResult) : null;
  return (
    <Panel
      title="Agreement with reference"
      actions={
        <Segmented
          size="xs"
          value={reference}
          onChange={setReference}
          options={[
            { value: "reference", label: "Reference", disabled: !hasReference },
            { value: "annotation", label: "Annotation", disabled: !hasAnnotation },
          ]}
        />
      }
    >
      {cmp.loading && !cmp.data && <Spinner label="Computing" />}
      {cmp.error ? <ErrorState error={cmp.error} /> : null}
      {cmp.data && isUnavailable(cmp.data) && <UnavailableState data={cmp.data as Unavailable} />}
      {data && (
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-2">
            <Stat label="Dice" value={fmt(data.any_lesion.dice, 3)} kind="measured" />
            <Stat label="IoU" value={fmt(data.any_lesion.iou, 3)} />
            <Stat label="HD95" value={fmt(data.any_lesion.hd95_mm, 1)} unit="mm" />
            <Stat label="Hausdorff" value={fmt(data.any_lesion.hausdorff_mm, 1)} unit="mm" />
            <Stat label="Sensitivity" value={fmt(data.any_lesion.sensitivity, 3)} />
            <Stat label="Specificity" value={fmt(data.any_lesion.specificity, 4)} sub={`within ${data.any_lesion.specificity_roi}`} />
          </div>
          {Object.keys(data.per_class).length > 0 && (
            <table className="table-base">
              <thead>
                <tr>
                  <th>Class</th>
                  <th className="text-right">Dice</th>
                  <th className="text-right">HD95</th>
                  <th className="text-right">Pred/Ref cc</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.per_class).map(([k, m]) => (
                  <tr key={k}>
                    <td>{CLASS_LABEL[k] ?? k}</td>
                    <td className="text-right font-mono">{fmt(m.dice, 3)}</td>
                    <td className="text-right font-mono">{fmt(m.hd95_mm, 1)}</td>
                    <td className="text-right font-mono">
                      {fmt(m.pred_volume_cc, 2)} / {fmt(m.ref_volume_cc, 2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <Note>
            Reference: {data.reference_kind}. Single-scan agreement only. Population results on patient-separated test data are on the Model Evaluation page.
          </Note>
        </div>
      )}
    </Panel>
  );
}

// ----- manual annotation --------------------------------------------------------

interface AnnotationState {
  volume: VolumeData | null;
  editing: boolean;
  setEditing: (v: boolean) => void;
  value: number;
  setValue: (v: number) => void;
  radius: number;
  setRadius: (v: number) => void;
  brush: { radius: number; value: number; onPaint: (idx: number[], v: number) => void };
  hasSaved: boolean;
  dirty: boolean;
  copyFrom: (v: VolumeData) => void;
  clear: () => void;
  save: () => Promise<void>;
  status: string | null;
  error: unknown;
}

function useAnnotation(caseId: string | null, tpId: string | null): AnnotationState {
  const [volume, setVolume] = useState<VolumeData | null>(null);
  const [, setTick] = useState(0);
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(1);
  const [radius, setRadius] = useState(2);
  const [hasSaved, setHasSaved] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const shapeRef = useRef<VolumeData | null>(null);

  useEffect(() => {
    if (!caseId || !tpId) return;
    let live = true;
    setEditing(false);
    setDirty(false);
    setStatus(null);
    Promise.all([fetchVolume(api.imageUrl(caseId, tpId)), fetchVolume(api.mapUrl(caseId, tpId, "annotation")).catch((e) => e)]).then(
      ([img, ann]) => {
        if (!live) return;
        shapeRef.current = img;
        if (ann instanceof ApiError || ann instanceof Error) {
          setVolume({ data: new Uint8Array(img.data.length), shape: img.shape, spacing: img.spacing, scale: 1, offset: 0 });
          setHasSaved(false);
        } else {
          setVolume({ ...ann, data: new Uint8Array(ann.data) });
          setHasSaved(true);
        }
      },
      (e) => live && setError(e),
    );
    return () => {
      live = false;
    };
  }, [caseId, tpId]);

  const onPaint = useCallback(
    (idx: number[], v: number) => {
      if (!volume) return;
      const d = volume.data as Uint8Array;
      for (const i of idx) d[i] = v;
      setDirty(true);
      setVolume({ ...volume });
      setTick((t) => t + 1);
    },
    [volume],
  );

  return {
    volume,
    editing,
    setEditing,
    value,
    setValue,
    radius,
    setRadius,
    brush: { radius, value, onPaint },
    hasSaved,
    dirty,
    status,
    error,
    copyFrom: (src) => {
      if (!volume) return;
      setVolume({ ...volume, data: new Uint8Array(src.data) });
      setDirty(true);
      setEditing(true);
    },
    clear: () => {
      if (!volume) return;
      setVolume({ ...volume, data: new Uint8Array(volume.data.length) });
      setDirty(true);
    },
    save: async () => {
      if (!caseId || !tpId || !volume) return;
      setError(null);
      try {
        const r = await api.saveAnnotation(caseId, tpId, volume.data as Uint8Array);
        setHasSaved(true);
        setDirty(false);
        setStatus(`Saved ${r.voxels} voxels (${r.volume_cc.toFixed(2)} cc).`);
      } catch (e) {
        setError(e);
      }
    },
  };
}

function AnnotationTools({ state, caseId, tpId, hasPrediction, modelId }: {
  state: AnnotationState;
  caseId: string;
  tpId: string;
  hasPrediction: boolean;
  modelId: string;
}) {
  const copy = async () => {
    const pred = await fetchVolume(api.mapUrl(caseId, tpId, "prediction", { model_id: modelId }));
    state.copyFrom(pred);
  };
  return (
    <div className="rounded-lg border border-ink-700/60 bg-ink-925/60 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="label mr-1">Manual annotation</span>
        <button className={cx(state.editing ? "btn-primary" : "btn-ghost")} onClick={() => state.setEditing(!state.editing)} aria-pressed={state.editing}>
          <BrushIcon className="h-3.5 w-3.5" /> {state.editing ? "Brush on (axial)" : "Edit"}
        </button>
        {state.editing && (
          <>
            <Segmented
              size="xs"
              value={String(state.value)}
              onChange={(v) => state.setValue(+v)}
              options={[
                { value: "1", label: "Primary" },
                { value: "2", label: "Node" },
                { value: "3", label: "Other" },
                { value: "0", label: "Erase" },
              ]}
            />
            <label className="flex w-28 items-center gap-2 text-[11px] text-ink-400">
              Size
              <input type="range" min={0} max={8} step={1} value={state.radius} onChange={(e) => state.setRadius(+e.target.value)} aria-label="Brush radius" />
            </label>
          </>
        )}
        <button className="btn-ghost" disabled={!hasPrediction} onClick={copy} title="Start from the model's mask and correct it">
          <Copy className="h-3.5 w-3.5" /> From model
        </button>
        <button className="btn-ghost" onClick={state.clear}>
          <Trash2 className="h-3.5 w-3.5" /> Clear
        </button>
        <button className="btn-primary" disabled={!state.dirty} onClick={state.save}>
          <Save className="h-3.5 w-3.5" /> Save
        </button>
        {state.status && <span className="text-[12px] text-teal">{state.status}</span>}
        {state.dirty && <span className="text-[12px] text-ev-uncertain">Unsaved changes</span>}
      </div>
      {state.error ? (
        <div className="mt-2">
          <ErrorState error={state.error} />
        </div>
      ) : null}
    </div>
  );
}
