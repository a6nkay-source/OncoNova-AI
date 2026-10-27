import { Download, PlayCircle } from "lucide-react";
import { useState } from "react";
import { api, isUnavailable } from "../api/client";
import type { DoseResult, LongitudinalResult, ProgressionPrediction, ResearchSummary, SegmentationSummary } from "../api/types";
import { IntervalBar, LineChart, SERIES_COLORS } from "../components/charts";
import { DEFAULT_ANATOMY, Volume3D, lesionMeshes } from "../components/Volume3D";
import {
  DemoBadge,
  ErrorState,
  EvidenceBadge,
  FindingList,
  Note,
  Panel,
  Spinner,
  Stat,
  UnavailableState,
  downloadText,
  fmt,
} from "../components/ui";
import { invalidateVolumes, useAsync } from "../state/hooks";
import { useApp } from "../state/AppState";
import { CaseViewer } from "../viewer/CaseViewer";

const SECTION_TITLES: Record<string, string> = {
  A_imaging_observations: "A · Imaging observations",
  B_segmentation_results: "B · Segmentation results",
  C_changes_between_timepoints: "C · Changes between time points",
  D_progression_model_output: "D · Progression model output",
  E_uncertainty_and_limitations: "E · Uncertainty and known limitations",
  F_research_radiation_simulation: "F · Research-only radiation simulation",
  G_items_requiring_expert_review: "G · Items requiring qualified expert review",
};

export function OverviewPage() {
  const { current, caseId, tpId, modelId, models, revision, bump, system } = useApp();
  const source = `pred:${modelId}`;
  const seg = useAsync(() => api.segmentation(caseId!, tpId!, modelId), [caseId, tpId, modelId, revision], !!caseId && !!tpId);
  const longi = useAsync(() => api.getLongitudinal(caseId!, source), [caseId, source, revision], !!caseId);
  const dose = useAsync(() => api.getDose(caseId!, "S2", "reference"), [caseId, revision], !!caseId);
  const summary = useAsync(() => api.summary(caseId!), [caseId, revision], !!caseId);
  const [risk, setRisk] = useState<ProgressionPrediction | null>(null);
  const [steps, setSteps] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<unknown>(null);

  if (!current || !caseId || !tpId) return <Spinner label="Loading cases" />;
  const model = models.find((m) => m.id === modelId);

  const runAll = async () => {
    setRunning(true);
    setError(null);
    setSteps([]);
    const log = (s: string) => setSteps((prev) => [...prev, s]);
    try {
      for (const t of current.timepoints) {
        log(`Segmenting ${t.tp_id} with ${modelId}`);
        await api.segment(caseId, t.tp_id, modelId, 8);
      }
      invalidateVolumes(`/api/cases/${caseId}/`);
      if (current.timepoints.length > 1) {
        log("Registering follow-ups and measuring change");
        await api.longitudinal(caseId, source);
      }
      log("Estimating progression risk");
      const r = await api.progression(caseId, source);
      setRisk(isUnavailable(r) ? null : (r as ProgressionPrediction));
      if (current.has_reference) {
        log("Computing simulated dose illustration (S2)");
        await api.dose(caseId, "S2", "reference");
      }
      log("Done");
      bump();
    } catch (e) {
      setError(e);
    } finally {
      setRunning(false);
    }
  };

  const s = seg.data && !isUnavailable(seg.data) ? (seg.data as SegmentationSummary) : null;
  const l = longi.data && !isUnavailable(longi.data) ? (longi.data as LongitudinalResult) : null;
  const d = dose.data && !isUnavailable(dose.data) ? (dose.data as DoseResult) : null;

  return (
    <div className="space-y-4">
      <div className="panel flex flex-wrap items-center gap-4 bg-gradient-to-r from-ink-900 via-ink-900 to-accent/[0.07] px-5 py-4">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-[18px] font-semibold tracking-tight text-ink-100">{current.case_id}</h1>
            {current.demo && <DemoBadge />}
            <span className="text-[13px] text-ink-400">{current.title}</span>
          </div>
          <div className="mt-1 text-[12.5px] text-ink-400">
            {current.cancer_type} · {current.modality} · {current.timepoints.length} scan(s) · model {modelId} ({model?.status ?? "?"})
          </div>
        </div>
        <button className="btn-primary" onClick={runAll} disabled={running || model?.status !== "available"}>
          <PlayCircle className="h-4 w-4" /> {running ? "Running pipeline..." : "Run research pipeline"}
        </button>
      </div>
      {(running || steps.length > 0) && (
        <div className="flex flex-wrap gap-2 text-[12px]">
          {steps.map((st, i) => (
            <span key={i} className="rounded-md border border-ink-700 bg-ink-900 px-2 py-1 text-ink-300">
              {i === steps.length - 1 && running ? "… " : "✓ "}
              {st}
            </span>
          ))}
        </div>
      )}
      {error ? <ErrorState error={error} /> : null}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <Panel title="Image viewer" className="min-h-[460px]" bodyClassName="flex flex-col">
          <CaseViewer
            caseId={caseId}
            tpId={tpId}
            modality={current.modality}
            layout="row"
            version={revision}
            overlays={[
              { id: "pred", url: s ? api.mapUrl(caseId, tpId, "prediction", { model_id: modelId }) : null, type: "label", opacity: 0.5, visible: !!s },
              { id: "ref", url: current.has_reference ? api.mapUrl(caseId, tpId, "reference") : null, type: "label", opacity: 0.6, outline: true, colors: { 1: [255, 255, 255], 2: [255, 255, 255], 3: [255, 255, 255] }, visible: current.has_reference },
            ]}
          />
          <div className="mt-2 text-[11.5px] text-ink-400">Fill: {modelId} prediction · white outline: synthetic reference.</div>
        </Panel>
        <Panel title="3-D lesion and anatomy">
          {s || current.has_reference ? (
            <Volume3D caseId={caseId} tpId={tpId} version={revision} height={400} meshes={[...(current.has_structures ? DEFAULT_ANATOMY : []), ...lesionMeshes(s ? `pred:${modelId}` : "reference")]} />
          ) : (
            <Note>Run the pipeline to build surfaces.</Note>
          )}
        </Panel>
      </div>

      <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-4">
        <Panel title={<span className="flex items-center gap-2">Segmentation <EvidenceBadge kind="model_output" /></span>}>
          {s ? (
            <div className="space-y-2">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="Primary" value={fmt(s.volumes_cc.primary_tumor, 2)} unit="cc" />
                <Stat label="Other regions" value={fmt((s.volumes_cc.lymph_node_region_lesion ?? 0) + (s.volumes_cc.other_lesion ?? 0), 2)} unit="cc" />
              </div>
              <div className="text-[12px] text-ink-400">
                Mean uncertainty {fmt(s.uncertainty.mean_uncertainty_in_prediction, 3)} · {s.unreliable_regions.length} unreliable region(s)
              </div>
            </div>
          ) : (
            <Note>Not run for {tpId}.</Note>
          )}
        </Panel>
        <Panel title={<span className="flex items-center gap-2">Observed change <EvidenceBadge kind="measured_from_model_output" /></span>}>
          {l ? (
            <LineChart
              height={150}
              series={l.tracks.slice(0, 4).map((t, i) => ({
                name: t.track_id,
                color: SERIES_COLORS[i],
                dashed: t.status === "new",
                points: t.volume_series_cc.map((v, j) => ({ x: l.timepoints[j].days_from_baseline, y: v })),
              }))}
              xLabel="Days"
              yFormat={(v) => v.toFixed(0)}
            />
          ) : current.timepoints.length < 2 ? (
            <UnavailableState data={{ status: "unavailable", kind: "unavailable", reason: "Single scan: no longitudinal comparison.", missing_inputs: ["second timepoint"] }} />
          ) : (
            <Note>Not computed for source {source}.</Note>
          )}
        </Panel>
        <Panel title={<span className="flex items-center gap-2">Progression risk <EvidenceBadge kind="model_output" /></span>}>
          {risk ? (
            <div className="space-y-2">
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-[26px] text-ink-100">{risk.probability.toFixed(2)}</span>
                <span className="text-[12px] text-ink-400">
                  [{risk.interval95[0].toFixed(2)}-{risk.interval95[1].toFixed(2)}]
                </span>
                {risk.uncertain && <EvidenceBadge kind="uncertain" />}
              </div>
              <IntervalBar p={risk.probability} lo={risk.interval95[0]} hi={risk.interval95[1]} />
              <p className="text-[11.5px] text-ink-400">{risk.outcome}</p>
            </div>
          ) : (
            <ProgressionFromSummary summary={summary.data} />
          )}
        </Panel>
        <Panel title={<span className="flex items-center gap-2">Simulation · S2 <EvidenceBadge kind="simulated" /></span>}>
          {d ? (
            <div className="space-y-2">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="Target D95" value={`${(d.dvh.stats.illustrative_target.D95! * 100).toFixed(0)}`} unit="%" />
                <Stat label="Body V5" value={`${(d.dvh.stats.body.V5 * 100).toFixed(0)}`} unit="%" />
              </div>
              <p className="text-[11.5px] text-ev-simulated">Relative units · geometric illustration · non-clinical</p>
            </div>
          ) : (
            <Note>Not computed.</Note>
          )}
        </Panel>
      </div>

      <ResearchSummaryPanel summary={summary.data} loading={summary.loading} error={summary.error} />

      <Panel title="Data provenance and model versions">
        <div className="grid gap-3 text-[12.5px] md:grid-cols-3">
          <div className="kv">
            <span className="text-ink-400">Data label</span>
            <span className="text-right font-semibold text-ev-uncertain">{current.label}</span>
            <span className="text-ink-400">Software</span>
            <span className="mono text-right">OncoNova {system?.version}</span>
            <span className="text-ink-400">Device control</span>
            <span className="text-right text-ink-300">none</span>
          </div>
          {models.map((m) => (
            <div key={m.id} className="kv">
              <span className="text-ink-400">{m.id}</span>
              <span className="mono text-right">v{m.version} · {m.status}</span>
              <span className="text-ink-400">Trained on</span>
              <span className="text-right text-ink-300">{m.trained_on.slice(0, 64)}{m.trained_on.length > 64 ? "…" : ""}</span>
              <span className="text-ink-400">Checkpoint</span>
              <span className="mono text-right text-ink-400">{m.checkpoint_sha256 ? m.checkpoint_sha256.slice(0, 12) : "n/a"}</span>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function ProgressionFromSummary({ summary }: { summary: ResearchSummary | null }) {
  const items = summary?.sections.D_progression_model_output ?? [];
  if (!items.length) return <Note>Not estimated.</Note>;
  return <FindingList items={items} />;
}

function ResearchSummaryPanel({ summary, loading, error }: { summary: ResearchSummary | null; loading: boolean; error: unknown }) {
  const toMarkdown = (s: ResearchSummary) =>
    [
      `# Research summary: ${s.case_id}`,
      "",
      `> ${s.disclaimer} Data label: ${s.label}.`,
      "",
      ...Object.entries(s.sections).flatMap(([k, items]) => [`## ${SECTION_TITLES[k] ?? k}`, "", ...items.map((f) => `- **[${f.kind}]** ${f.text}`), ""]),
    ].join("\n");
  return (
    <Panel
      title="Research summary"
      actions={
        summary && (
          <>
            <button className="btn-ghost" onClick={() => downloadText(`${summary.case_id}-summary.json`, JSON.stringify(summary, null, 2))}>
              <Download className="h-3.5 w-3.5" /> JSON
            </button>
            <button className="btn-ghost" onClick={() => downloadText(`${summary.case_id}-summary.md`, toMarkdown(summary), "text/markdown")}>
              <Download className="h-3.5 w-3.5" /> Markdown
            </button>
          </>
        )
      }
    >
      {loading && !summary && <Spinner label="Assembling" />}
      {error ? <ErrorState error={error} /> : null}
      {summary && (
        <div className="space-y-4">
          <p className="text-[12px] text-ink-400">{summary.disclaimer} Each line is tagged with how it was produced.</p>
          <div className="grid gap-4 lg:grid-cols-2">
            {Object.entries(summary.sections).map(([k, items]) => (
              <div key={k} className="rounded-lg border border-ink-700/50 bg-ink-925/50 p-3">
                <h3 className="mb-2 text-[12.5px] font-semibold text-ink-100">{SECTION_TITLES[k] ?? k}</h3>
                <FindingList items={items.slice(0, 8)} />
                {items.length > 8 && <p className="mt-1 text-[11px] text-ink-500">+{items.length - 8} more in the export</p>}
              </div>
            ))}
          </div>
        </div>
      )}
    </Panel>
  );
}
