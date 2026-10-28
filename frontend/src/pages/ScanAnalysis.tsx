import { CheckCircle2, CircleDashed, CircleSlash, FileUp, Loader2, RotateCcw, XCircle } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, isUnavailable } from "../api/client";
import type {
  CaseSummary,
  ComparisonResult,
  LongitudinalResult,
  ModelCard,
  ProgressionPrediction,
  SegmentationSummary,
  Unavailable,
} from "../api/types";
import { DivergingBars, IntervalBar, LineChart, SERIES_COLORS } from "../components/charts";
import { ResearchSummaryPanel } from "../components/ResearchSummary";
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
import { LESION_COLORS } from "../viewer/volume";

const CLASS_LABEL: Record<string, string> = {
  primary_tumor: "Primary lesion",
  lymph_node_region_lesion: "Node-region lesion",
  other_lesion: "Other lesion",
};

const FEATURE_LABEL: Record<string, string> = {
  baseline_total_volume_cc: "Baseline lesion volume",
  log_volume_ratio_per_100d: "Growth rate between scans",
  new_lesion_count: "New segmented regions",
  node_region_lesion_observed: "Node-region lesion seen",
  primary_sphericity: "Primary lesion shape (sphericity)",
  age_years: "Age",
};
const featureLabel = (f: string) =>
  f.startsWith("missingindicator_") ? `${FEATURE_LABEL[f.slice(17)] ?? f.slice(17)} missing` : (FEATURE_LABEL[f] ?? f);

// ----- automatic pipeline: draw every scan, measure change, estimate risk ------------

type StepStatus = "waiting" | "running" | "done" | "unavailable" | "error";
interface Step {
  status: StepStatus;
  detail?: string;
  error?: unknown;
}
interface PipelineState {
  seg: Step;
  change: Step;
  predict: Step;
  segs: Record<string, SegmentationSummary>;
  longi: LongitudinalResult | Unavailable | null;
  risk: ProgressionPrediction | Unavailable | null;
  /** Increments each time the pipeline finishes, so dependent views refetch. */
  finished: number;
}

const INITIAL: PipelineState = {
  seg: { status: "waiting" },
  change: { status: "waiting" },
  predict: { status: "waiting" },
  segs: {},
  longi: null,
  risk: null,
  finished: 0,
};

function useAutoPipeline(current: CaseSummary | null, model: ModelCard | undefined, onNewMasks: () => void) {
  const [state, setState] = useState<PipelineState>(INITIAL);
  const [tick, setTick] = useState(0);
  const force = useRef(false);

  useEffect(() => {
    if (!current || !model) return;
    let live = true;
    const caseId = current.case_id;
    const source = `pred:${model.id}`;
    const n = current.timepoints.length;
    const patch = (p: Partial<PipelineState>) => live && setState((s) => ({ ...s, ...p }));
    setState((s) => ({ ...INITIAL, finished: s.finished }));
    if (model.status !== "available") {
      patch({ seg: { status: "unavailable", detail: model.unavailable_reason ?? "model unavailable" } });
      return;
    }
    (async () => {
      let fresh = force.current;
      force.current = false;
      const segs: Record<string, SegmentationSummary> = {};
      patch({ seg: { status: "running", detail: `0 of ${n} scans` } });
      try {
        for (const [i, t] of current.timepoints.entries()) {
          let r = fresh ? null : await api.segmentation(caseId, t.tp_id, model.id);
          if (!r || isUnavailable(r)) {
            r = await api.segment(caseId, t.tp_id, model.id, model.id === "demo-unet" ? 8 : 0);
            fresh = true;
          }
          if (!live) return;
          segs[t.tp_id] = r as SegmentationSummary;
          patch({ segs: { ...segs }, seg: { status: "running", detail: `${i + 1} of ${n} scans` } });
        }
        if (fresh) {
          invalidateVolumes(`/api/cases/${caseId}/`);
          onNewMasks();
        }
        patch({ seg: { status: "done", detail: `${n} scan${n > 1 ? "s" : ""} · ${model.id}` } });
      } catch (e) {
        patch({ seg: { status: "error", error: e } });
        return;
      }

      patch({ change: { status: "running", detail: n > 1 ? "registering scans" : undefined } });
      let longi: LongitudinalResult | Unavailable;
      try {
        const stored = fresh ? null : await api.getLongitudinal(caseId, source);
        longi = stored && !isUnavailable(stored) ? stored : await api.longitudinal(caseId, source);
      } catch (e) {
        patch({ change: { status: "error", error: e } });
        return;
      }
      if (!live) return;
      patch({
        longi,
        change: isUnavailable(longi)
          ? { status: "unavailable", detail: n < 2 ? "needs two scans" : "not available" }
          : { status: "done", detail: `${n} scans compared` },
        predict: { status: "running" },
      });

      try {
        const risk = await api.progression(caseId, source);
        if (!live) return;
        patch({
          risk,
          predict: isUnavailable(risk)
            ? { status: "unavailable", detail: n < 2 ? "needs two scans" : "not available" }
            : { status: "done", detail: `${(risk as ProgressionPrediction).horizon_days}-day horizon` },
        });
      } catch (e) {
        patch({ predict: { status: "error", error: e } });
      }
      if (live) setState((s) => ({ ...s, finished: s.finished + 1 }));
    })();
    return () => {
      live = false;
    };
    // The case object is replaced on every case-list refresh; key the run on its id.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.case_id, model?.id, model?.status, tick]);

  const rerun = useCallback(() => {
    force.current = true;
    setTick((t) => t + 1);
  }, []);
  return { ...state, rerun };
}

// ----- page ------------------------------------------------------------------------

type ViewMode = "drawing" | "uncertainty";

export function ScanAnalysisPage() {
  const { current, caseId, tpId, setTp, models, modelId, setModelId, revision, bump } = useApp();
  const model = models.find((m) => m.id === modelId);
  const pipe = useAutoPipeline(current, model, bump);
  const [mode, setMode] = useState<ViewMode>("drawing");
  const [showTruth, setShowTruth] = useState(true);
  const [focus, setFocus] = useState<[number, number, number] | null>(null);

  const seg = tpId ? pipe.segs[tpId] : undefined;
  // Open each scan on its largest drawn lesion so the AI drawing is in view straight away.
  const largest = seg?.lesions.reduce<SegmentationSummary["lesions"][number] | null>((a, l) => (!a || l.volume_cc > a.volume_cc ? l : a), null);
  useEffect(() => {
    if (largest) setFocus(largest.centroid_index_zyx as [number, number, number]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseId, tpId, !!largest]);
  const cmp = useAsync(
    () => api.compare(caseId!, tpId!, modelId, "reference"),
    [caseId, tpId, modelId, revision, !!seg],
    !!caseId && !!tpId && !!seg && !!current?.has_reference,
  );
  const summary = useAsync(() => api.summary(caseId!), [caseId, pipe.finished], !!caseId && pipe.finished > 0);

  if (!current || !caseId || !tpId) return <Spinner label="Loading cases" />;

  const overlays: OverlaySpec[] = [
    {
      id: "drawing",
      name: "AI drawing",
      url: seg ? api.mapUrl(caseId, tpId, "prediction", { model_id: modelId }) : null,
      type: "label",
      opacity: 0.85,
      style: "both",
      legend: [
        { name: "Primary lesion", values: [1], color: LESION_COLORS[1] },
        { name: "Node-region", values: [2], color: LESION_COLORS[2] },
        { name: "Other lesion", values: [3], color: LESION_COLORS[3] },
      ],
      visible: !!seg && mode === "drawing",
    },
    {
      id: "uncertainty",
      name: "Where the AI is unsure",
      url: seg && mode === "uncertainty" ? api.mapUrl(caseId, tpId, "uncertainty", { model_id: modelId }) : null,
      type: "map",
      lut: "uncertainty",
      opacity: 0.75,
      threshold: 0.08,
      visible: !!seg && mode === "uncertainty",
    },
    {
      id: "truth",
      name: "True outline",
      url: current.has_reference ? api.mapUrl(caseId, tpId, "reference") : null,
      type: "label",
      opacity: 0.9,
      style: "outline",
      legend: [{ name: "Synthetic ground truth", values: [1, 2, 3], color: [255, 255, 255] }],
      visible: showTruth && current.has_reference,
    },
  ];

  return (
    <div className="space-y-4">
      <div className="panel flex flex-wrap items-center gap-x-5 gap-y-3 px-4 py-3">
        <div className="min-w-0">
          <div className="text-[14px] font-medium text-ink-100">{current.title}</div>
          <div className="text-[12px] text-ink-400">
            {current.cancer_type} · {current.modality} · {current.timepoints.length} scan{current.timepoints.length > 1 ? "s" : ""}
          </div>
        </div>
        <PipelineStrip current={current} pipe={pipe} />
        <div className="ml-auto flex items-center gap-2">
          <button className="btn-ghost" onClick={pipe.rerun} disabled={[pipe.seg, pipe.change, pipe.predict].some((s) => s.status === "running")}>
            <RotateCcw className="h-3.5 w-3.5" /> Re-run
          </button>
          <ImportScan />
        </div>
      </div>
      {[pipe.seg, pipe.change, pipe.predict].map((s, i) => (s.status === "error" ? <ErrorState key={i} error={s.error} onRetry={pipe.rerun} /> : null))}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
        <Panel
          title={
            <Segmented
              value={tpId}
              onChange={setTp}
              options={current.timepoints.map((t, i) => ({ value: t.tp_id, label: `Scan ${i + 1} · day ${t.days_from_baseline}` }))}
            />
          }
          actions={!seg && pipe.seg.status === "running" ? <Spinner label="AI is drawing" /> : null}
          className="self-start xl:sticky xl:top-0"
          bodyClassName="flex flex-col gap-3"
        >
          <CaseViewer
            caseId={caseId}
            tpId={tpId}
            modality={current.modality}
            overlays={overlays}
            focus={focus}
            version={revision}
            toolbar={
              <div className="flex flex-wrap items-center gap-3">
                <Segmented<ViewMode>
                  size="xs"
                  value={mode}
                  onChange={setMode}
                  options={[
                    { value: "drawing", label: "AI drawing", disabled: !seg },
                    { value: "uncertainty", label: "Where the AI is unsure", disabled: !seg },
                  ]}
                />
                {current.has_reference && (
                  <label className="flex items-center gap-1.5 text-[12px] text-ink-300">
                    <input type="checkbox" checked={showTruth} onChange={(e) => setShowTruth(e.target.checked)} />
                    True outline
                  </label>
                )}
              </div>
            }
          />
        </Panel>

        <div className="flex flex-col gap-4">
          <DrawingCard
            step={pipe.seg}
            seg={seg}
            cmp={cmp.data}
            models={models}
            modelId={modelId}
            setModelId={setModelId}
            onFocus={(zyx) => setFocus([zyx[0], zyx[1], zyx[2]])}
          />
          <ChangeCard step={pipe.change} longi={pipe.longi} />
          <PredictionCard step={pipe.predict} risk={pipe.risk} />
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Panel title="3-D view of the AI drawing">
          {seg ? (
            <Volume3D
              caseId={caseId}
              tpId={tpId}
              version={revision}
              meshes={[...(current.has_structures ? DEFAULT_ANATOMY : []), ...lesionMeshes(`pred:${modelId}`)]}
            />
          ) : (
            <Note>The 3-D view appears once the AI has drawn this scan.</Note>
          )}
        </Panel>
        <Panel title="How to read this page">
          <ul className="space-y-2.5 text-[13px] leading-relaxed text-ink-300">
            <li>
              <b className="text-ink-100">AI drawing</b> <EvidenceBadge kind="model_output" /> A demo model outlines lesions on every scan as soon as you open a
              case. It was trained only on synthetic phantoms.
            </li>
            <li>
              <b className="text-ink-100">Change over time</b> <EvidenceBadge kind="measured_from_model_output" /> Volumes are measured from the AI's masks after
              aligning each follow-up to the first scan. No prediction is involved.
            </li>
            <li>
              <b className="text-ink-100">What may happen next</b> <EvidenceBadge kind="model_output" /> A separate model estimates the chance of a defined
              outcome over a fixed horizon, with an interval. It is a hypothesis for research, not a forecast for a real person.
            </li>
            <li className="text-[12px] text-ink-400">
              New regions are flagged for expert review: a new region is not by itself evidence that cancer has spread, and nothing here predicts where it
              might spread.
            </li>
          </ul>
        </Panel>
      </div>

      <ResearchSummaryPanel summary={summary.data} loading={summary.loading} error={summary.error} />
    </div>
  );
}

// ----- header pieces -----------------------------------------------------------------

function PipelineStrip({ current, pipe }: { current: CaseSummary; pipe: PipelineState }) {
  const steps: { label: string; step: Step }[] = [
    { label: "Scans loaded", step: { status: "done", detail: `${current.timepoints.length} · ${current.modality}` } },
    { label: "AI draws lesions", step: pipe.seg },
    { label: "Change measured", step: pipe.change },
    { label: "Prediction", step: pipe.predict },
  ];
  return (
    <ol className="flex flex-wrap items-center gap-1.5" aria-label="Analysis steps">
      {steps.map(({ label, step }, i) => (
        <li key={label} className="flex items-center gap-1.5">
          {i > 0 && <span className="h-px w-4 bg-ink-700" />}
          <span
            className={cx(
              "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[12px]",
              step.status === "done" && "border-teal/40 bg-teal/10 text-ink-100",
              step.status === "running" && "border-accent/50 bg-accent/10 text-ink-100",
              step.status === "waiting" && "border-ink-700 text-ink-500",
              step.status === "unavailable" && "border-ink-600 text-ink-400",
              step.status === "error" && "border-ev-review/50 bg-ev-review/10 text-ev-review",
            )}
            title={step.detail}
          >
            <StepIcon status={step.status} />
            {label}
            {step.detail && <span className="hidden text-[11px] text-ink-400 2xl:inline">· {step.detail}</span>}
          </span>
        </li>
      ))}
    </ol>
  );
}

function StepIcon({ status }: { status: StepStatus }) {
  if (status === "done") return <CheckCircle2 className="h-3.5 w-3.5 text-teal" />;
  if (status === "running") return <Loader2 className="h-3.5 w-3.5 animate-spin text-accent" />;
  if (status === "unavailable") return <CircleSlash className="h-3.5 w-3.5" />;
  if (status === "error") return <XCircle className="h-3.5 w-3.5" />;
  return <CircleDashed className="h-3.5 w-3.5" />;
}

function ImportScan() {
  const { system, refresh, setCase } = useApp();
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const upload = async (file: File) => {
    setBusy(true);
    setError(null);
    setWarnings([]);
    try {
      const isNifti = /\.nii(\.gz)?$/i.test(file.name);
      const r = await api.upload(file, isNifti ? "CT" : undefined);
      setWarnings(r.warnings);
      refresh();
      setCase(r.case_id);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  };
  return (
    <div className="relative">
      <input
        ref={input}
        type="file"
        accept=".nii,.gz,.zip"
        className="hidden"
        aria-label="CT file"
        onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
      />
      <button
        className="btn-primary"
        disabled={busy}
        onClick={() => input.current?.click()}
        title={`A CT as .nii / .nii.gz, or a .zip of one DICOM series (up to ${system?.limits.max_upload_mb ?? 512} MB). DICOM is de-identified before it is read.`}
      >
        <FileUp className="h-3.5 w-3.5" /> {busy ? "Importing..." : "Import CT"}
      </button>
      {(error || warnings.length > 0) && (
        <div className="absolute right-0 top-full z-10 mt-2 w-[360px] space-y-2 rounded-lg border border-ink-700 bg-ink-900 p-3 shadow-xl">
          {error ? <ErrorState error={error} /> : null}
          {warnings.map((w) => (
            <Note key={w}>{w}</Note>
          ))}
          <button className="btn-ghost w-full" onClick={() => (setError(null), setWarnings([]))}>
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
}

function StepBody({ step, children, waiting }: { step: Step; children: ReactNode; waiting: string }) {
  if (step.status === "waiting") return <p className="text-[12.5px] text-ink-500">{waiting}</p>;
  if (step.status === "running" && !children) return <Spinner label={step.detail ?? "Working"} />;
  return <>{children}</>;
}

// ----- result cards ------------------------------------------------------------------

function DrawingCard({ step, seg, cmp, models, modelId, setModelId, onFocus }: {
  step: Step;
  seg: SegmentationSummary | undefined;
  cmp: ComparisonResult | Unavailable | null;
  models: ModelCard[];
  modelId: string;
  setModelId: (id: string) => void;
  onFocus: (p: [number, number, number]) => void;
}) {
  const agreement = cmp && !isUnavailable(cmp) ? (cmp as ComparisonResult) : null;
  return (
    <Panel
      title={<span className="flex items-center gap-2">1 · AI drawing <EvidenceBadge kind="model_output" /></span>}
      actions={
        <select className="input py-0.5 text-[12px]" value={modelId} onChange={(e) => setModelId(e.target.value)} aria-label="Segmentation model">
          {models.map((m) => (
            <option key={m.id} value={m.id} disabled={m.status !== "available"}>
              {m.id === "demo-unet" ? "3-D U-Net (demo)" : m.id === "demo-classical" ? "Classical baseline" : m.id}
              {m.status !== "available" ? " (unavailable)" : ""}
            </option>
          ))}
        </select>
      }
    >
      <StepBody step={step} waiting="Waiting for the model list.">
        {step.status === "unavailable" && (
          <UnavailableState data={{ status: "unavailable", kind: "unavailable", reason: step.detail ?? "Model unavailable", missing_inputs: [] }} />
        )}
        {!seg && step.status === "running" && <Spinner label={`Drawing · ${step.detail}`} />}
        {seg && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <Stat label="Primary lesion" value={fmt(seg.volumes_cc.primary_tumor, 1)} unit="cc" sub="measured from the AI mask" />
              <Stat
                label="Other regions"
                value={fmt((seg.volumes_cc.lymph_node_region_lesion ?? 0) + (seg.volumes_cc.other_lesion ?? 0), 1)}
                unit="cc"
                sub={`${seg.lesions.length} region${seg.lesions.length === 1 ? "" : "s"} found`}
              />
            </div>
            {seg.lesions.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {seg.lesions.slice(0, 6).map((l, i) => (
                  <button
                    key={i}
                    className="rounded-md border border-ink-700 px-2 py-0.5 text-[11.5px] text-ink-300 hover:border-ink-500 hover:text-ink-100"
                    onClick={() => onFocus(l.centroid_index_zyx as [number, number, number])}
                    title="Show in the viewer"
                  >
                    {CLASS_LABEL[l.class] ?? l.class} · {fmt(l.volume_cc, 1)} cc · {fmt(l.max_axial_diameter_mm, 0)} mm
                  </button>
                ))}
              </div>
            )}
            <div className="kv rounded-lg bg-ink-925 p-3">
              <span className="text-ink-400">Unsure regions</span>
              <span className={cx("mono text-right", seg.unreliable_regions.length > 0 && "text-ev-uncertain")}>{seg.unreliable_regions.length}</span>
              <span className="text-ink-400">Mean uncertainty in drawing</span>
              <span className="mono text-right">{fmt(seg.uncertainty.mean_uncertainty_in_prediction, 2)}</span>
              {agreement && (
                <>
                  <span className="text-ink-400">Agreement with true outline (Dice)</span>
                  <span className="mono text-right text-teal">{fmt(agreement.any_lesion.dice, 3)}</span>
                </>
              )}
            </div>
            {seg.warnings.length > 0 && <FindingList items={seg.warnings.map((w) => ({ kind: "uncertain", text: w }))} />}
            <p className="text-[11.5px] text-ink-500">{seg.model.label}. Uncertainty: {seg.uncertainty_method}.</p>
          </div>
        )}
      </StepBody>
    </Panel>
  );
}

function ChangeCard({ step, longi }: { step: Step; longi: LongitudinalResult | Unavailable | null }) {
  const result = longi && !isUnavailable(longi) ? (longi as LongitudinalResult) : null;
  const series = useMemo(() => {
    if (!result) return [];
    const days = result.timepoints.map((t) => t.days_from_baseline);
    return [
      { name: "All lesions", color: "#e4ebf5", points: result.timepoints.map((t) => ({ x: t.days_from_baseline, y: t.total_lesion_volume_cc })) },
      ...result.tracks.slice(0, 4).map((t, i) => ({
        name: `${t.track_id} ${CLASS_LABEL[t.class] ?? t.class}`,
        color: SERIES_COLORS[i % SERIES_COLORS.length],
        dashed: t.status === "new",
        points: t.volume_series_cc.map((v, j) => ({ x: days[j], y: v })),
      })),
    ];
  }, [result]);
  const regReview = result?.timepoints.some((t) => t.registration && t.registration.qc !== "ok");
  const notable = (result?.findings ?? []).filter((f) => f.kind !== "measured" && f.kind !== "measured_from_model_output").slice(0, 3);
  return (
    <Panel title={<span className="flex items-center gap-2">2 · Change over time <EvidenceBadge kind="measured_from_model_output" /></span>}>
      <StepBody step={step} waiting="Starts after the AI has drawn every scan.">
        {step.status === "running" && <Spinner label="Aligning scans and measuring" />}
        {longi && isUnavailable(longi) && <UnavailableState data={longi as Unavailable} title="No comparison" />}
        {result && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <Stat
                label="Total volume change"
                value={result.total_volume_change_pct === null ? "-" : `${result.total_volume_change_pct > 0 ? "+" : ""}${result.total_volume_change_pct.toFixed(0)}`}
                unit="%"
                sub={`day 0 → day ${result.timepoints[result.timepoints.length - 1].days_from_baseline}`}
              />
              <Stat
                label="New regions"
                value={result.new_lesion_count}
                kind={result.new_lesion_count ? "requires_expert_review" : undefined}
                sub={result.new_lesion_count ? "for expert review" : "none seen"}
              />
            </div>
            <LineChart height={170} series={series} xLabel="Days from first scan" yLabel="cc" yFormat={(v) => v.toFixed(0)} />
            {regReview && <FindingList items={[{ kind: "requires_expert_review", text: "Scan alignment needs review: change measurements may be off." }]} />}
            {notable.length > 0 && <FindingList items={notable} />}
          </div>
        )}
      </StepBody>
    </Panel>
  );
}

function PredictionCard({ step, risk }: { step: Step; risk: ProgressionPrediction | Unavailable | null }) {
  const p = risk && !isUnavailable(risk) ? (risk as ProgressionPrediction) : null;
  const pct = (v: number) => Math.round(v * 100);
  return (
    <Panel title={<span className="flex items-center gap-2">3 · What may happen next <EvidenceBadge kind="model_output" /></span>}>
      <StepBody step={step} waiting="Starts after the change between scans is measured.">
        {step.status === "running" && <Spinner label="Estimating" />}
        {risk && isUnavailable(risk) && <UnavailableState data={risk as Unavailable} title="No prediction" />}
        {p && (
          <div className="space-y-3">
            <div className="rounded-lg border border-ev-model/30 bg-ev-model/5 p-3.5">
              <div className="flex items-end gap-3">
                <span className="font-mono text-[38px] font-medium leading-none text-ink-100">{pct(p.probability)}%</span>
                <span className="pb-0.5 text-[12px] leading-snug text-ink-300">
                  estimated chance of progression
                  <br />
                  within {p.horizon_days} days · 95% interval {pct(p.interval95[0])}-{pct(p.interval95[1])}%
                </span>
              </div>
              <div className="mt-3">
                <IntervalBar p={p.probability} lo={p.interval95[0]} hi={p.interval95[1]} />
              </div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {p.uncertain && <EvidenceBadge kind="uncertain" />}
                <EvidenceBadge kind="requires_expert_review" />
              </div>
              <p className="mt-2 text-[11.5px] text-ink-400">Outcome the model was trained on: {p.outcome}</p>
            </div>
            <div>
              <div className="label mb-1.5">What pushed the estimate up or down</div>
              <DivergingBars
                items={p.contributions.slice(0, 4).map((c) => ({ label: featureLabel(c.feature), value: c.contribution_log_odds }))}
                format={(v) => (v > 0 ? "+" : "") + v.toFixed(2)}
              />
              <p className="mt-1 text-[11px] text-ink-500">Associations inside a linear model (log-odds), not causes.</p>
            </div>
            {p.out_of_distribution_warnings.length > 0 && (
              <FindingList items={p.out_of_distribution_warnings.map((w) => ({ kind: "uncertain", text: w }))} />
            )}
            <details className="rounded-lg bg-ink-925 p-3 text-[12px] text-ink-300">
              <summary className="cursor-pointer font-medium text-ink-200">Limitations ({p.limitations.length})</summary>
              <ul className="mt-2 list-disc space-y-1 pl-4">
                {p.limitations.map((l) => (
                  <li key={l}>{l}</li>
                ))}
              </ul>
            </details>
            <Note>This model has no labels for metastasis and does not predict whether or where disease might spread.</Note>
          </div>
        )}
      </StepBody>
    </Panel>
  );
}
