import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import type { BinaryReport, EvaluationReport, SummaryStat } from "../../api/types";
import { DivergingBars, ForestPlot, LineChart, StripPlot } from "../../components/charts";
import { DemoBadge, ErrorState, EvidenceBadge, Note, Panel, Segmented, Spinner, Stat, fmt, fmtCI } from "../../components/ui";
import { useAsync } from "../../state/hooks";

const SEG_TARGETS: Record<string, string> = {
  any_lesion: "Any lesion",
  primary_tumor: "Primary lesion",
  lymph_node_region_lesion: "Node-region lesion",
  other_lesion: "Other lesion",
};

export function EvaluationSection() {
  const list = useAsync(() => api.evaluations(), []);
  const reports = list.data?.reports ?? [];
  const latest = useMemo(() => {
    const seen = new Map<string, string>();
    for (const r of reports) {
      const key = `${r.task}:${r.model?.id}`;
      if (!seen.has(key)) seen.set(key, r.report_id);
    }
    return [...seen.values()];
  }, [reports]);

  if (list.loading && !list.data) return <Spinner label="Loading reports" />;
  if (list.error) return <ErrorState error={list.error} onRetry={list.reload} />;
  if (!reports.length)
    return (
      <Panel title="Model evaluation">
        <div className="space-y-2 text-[13px] text-ink-300">
          <p>No evaluation has been run. The prototype never shows performance numbers it has not measured.</p>
          <pre className="mono rounded-lg bg-ink-925 p-3 text-ink-200">make train-seg && make evaluate{"\n"}make train-progression</pre>
        </div>
      </Panel>
    );
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-ev-uncertain/30 bg-ev-uncertain/5 px-4 py-2.5 text-[12.5px] text-ink-200">
        <DemoBadge />
        Every number below was measured by the evaluation scripts on held-out synthetic data and loaded from a stored report. It describes
        behaviour on phantoms and simulated outcomes, not clinical performance.
      </div>
      {latest.map((id) => (
        <ReportView key={id} id={id} />
      ))}
      <Panel title="All stored reports">
        <table className="table-base">
          <thead>
            <tr>
              <th>Report</th>
              <th>Task</th>
              <th>Model</th>
              <th>Created (UTC)</th>
            </tr>
          </thead>
          <tbody>
            {reports.map((r) => (
              <tr key={r.report_id}>
                <td className="mono">{r.report_id}</td>
                <td>{r.task}</td>
                <td>{r.model?.id}</td>
                <td className="mono">{r.created_utc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}

function ReportView({ id }: { id: string }) {
  const rep = useAsync(() => api.evaluation(id), [id]);
  if (rep.loading && !rep.data) return <Spinner label={`Loading ${id}`} />;
  if (rep.error) return <ErrorState error={rep.error} />;
  if (!rep.data) return null;
  return rep.data.task === "segmentation" ? <SegmentationReport r={rep.data} /> : <ProgressionReport r={rep.data} />;
}

function Header({ r }: { r: EvaluationReport }) {
  return (
    <div className="mb-3 space-y-1">
      <div className="flex flex-wrap items-center gap-2 text-[12px] text-ink-400">
        <EvidenceBadge kind="measured" />
        <span className="mono">{r.report_id}</span>
        <span>· dataset {r.dataset.id} ({String(r.dataset.label)})</span>
        {r.model.checkpoint_sha256 && <span className="mono">· ckpt {r.model.checkpoint_sha256.slice(0, 12)}</span>}
        {r.model.artifact_sha256 && <span className="mono">· artifact {r.model.artifact_sha256.slice(0, 12)}</span>}
      </div>
      <p className="text-[12px] text-ink-400">Protocol: {r.protocol}</p>
    </div>
  );
}

function SegmentationReport({ r }: { r: EvaluationReport }) {
  const results = r.results as Record<string, Record<string, SummaryStat> & { n_scans?: number }>;
  const det = r.results.lesion_detection as { reference_lesions: number; detected: number; lesion_sensitivity: number | null; false_positive_components_per_scan: number };
  const [target, setTarget] = useState("any_lesion");
  useEffect(() => setTarget("any_lesion"), [r.report_id]);
  const t = results[target];
  return (
    <Panel title={`Segmentation · ${r.model.id} v${r.model.version} · ${String(r.dataset.n_patients)} test patients / ${String(r.dataset.n_scans)} scans`}>
      <Header r={r} />
      <Segmented
        size="xs"
        value={target}
        onChange={setTarget}
        options={Object.keys(SEG_TARGETS)
          .filter((k) => results[k])
          .map((k) => ({ value: k, label: `${SEG_TARGETS[k]} (${results[k].n_scans} scans)` }))}
      />
      {t && (
        <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6">
          <MetricStat label="Dice" s={t.dice} />
          <MetricStat label="IoU" s={t.iou} />
          <MetricStat label="HD95" s={t.hd95_mm} unit="mm" digits={1} />
          <MetricStat label="Hausdorff" s={t.hausdorff_mm} unit="mm" digits={1} />
          <MetricStat label="Sensitivity" s={t.sensitivity} />
          <MetricStat label="Specificity (body)" s={t.specificity} digits={4} />
        </div>
      )}
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div>
          <div className="label mb-2">Per-scan Dice, any lesion</div>
          <StripPlot values={(r.per_scan ?? []).map((p) => p.any_lesion.dice)} label="Dice" />
        </div>
        <div className="kv rounded-lg bg-ink-925 p-3">
          <span className="text-ink-400">Reference lesions</span>
          <span className="mono text-right">{det.reference_lesions}</span>
          <span className="text-ink-400">Detected (any overlap)</span>
          <span className="mono text-right">{det.detected}</span>
          <span className="text-ink-400">Lesion-level sensitivity</span>
          <span className="mono text-right">{fmt(det.lesion_sensitivity, 3)}</span>
          <span className="text-ink-400">False-positive components / scan</span>
          <span className="mono text-right">{fmt(det.false_positive_components_per_scan, 2)}</span>
        </div>
      </div>
      <div className="mt-3">
        <Note>{r.caveat} Means are across patients; intervals are patient-level bootstrap. A metric with n=0 could not be computed (for example, no predicted surface).</Note>
      </div>
    </Panel>
  );
}

function MetricStat({ label, s, unit, digits = 3 }: { label: string; s?: SummaryStat; unit?: string; digits?: number }) {
  return <Stat label={label} value={s?.mean === null || !s ? "n/a" : s.mean.toFixed(digits)} unit={unit} sub={s ? `${fmtCI(s.ci95, digits)} · n=${s.n_patients}` : undefined} />;
}

function ProgressionReport({ r }: { r: EvaluationReport }) {
  const results = r.results as Record<string, BinaryReport>;
  const [split, setSplit] = useState(Object.keys(results)[0]);
  const b = results[split];
  return (
    <Panel title={`Progression risk · ${r.model.id} v${r.model.version}`}>
      <Header r={r} />
      <Segmented
        size="xs"
        value={split}
        onChange={setSplit}
        options={Object.keys(results).map((k) => ({ value: k, label: k.replace(/_/g, " ") }))}
      />
      {b && b.auroc && (
        <>
          <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6">
            <Stat label="AUROC" value={fmt(b.auroc.value, 3)} sub={fmtCI(b.auroc.ci95)} />
            <Stat label="AUPRC" value={fmt(b.auprc.value, 3)} sub={`${fmtCI(b.auprc.ci95)} · prev ${fmt(b.prevalence, 2)}`} />
            <Stat label="Brier" value={fmt(b.brier.value, 3)} sub={fmtCI(b.brier.ci95)} />
            <Stat label="C-index" value={fmt(b.c_index.value, 3)} sub={fmtCI(b.c_index.ci95)} />
            <Stat label="Sensitivity" value={fmt(b.sensitivity.value, 3)} sub={`@ ${fmt(b.threshold, 3)} · ${fmtCI(b.sensitivity.ci95, 2)}`} />
            <Stat label="Specificity" value={fmt(b.specificity.value, 3)} sub={fmtCI(b.specificity.ci95, 2)} />
          </div>
          <div className="mt-1 text-[11.5px] text-ink-500">
            n={b.n} with a defined label ({b.n_events} events); {b.excluded_censored_before_horizon} censored before the horizon were excluded from binary
            metrics. Threshold: {b.threshold_source}.
          </div>
          <div className="mt-4 grid gap-4 xl:grid-cols-3">
            <div>
              <div className="label mb-1">Calibration (quantile bins)</div>
              <LineChart
                height={240}
                diagonal
                xMax={1}
                yMin={0}
                yMax={1}
                xLabel="Mean predicted"
                yLabel="Observed fraction"
                xFormat={(v) => v.toFixed(1)}
                yFormat={(v) => v.toFixed(1)}
                series={[{ name: "Model", color: "#a48bff", points: [{ x: 0, y: null }, ...b.calibration.bins.map((p) => ({ x: p.mean_predicted, y: p.observed_fraction }))] }]}
              />
              <p className="text-[11.5px] text-ink-400">
                ECE {fmt(b.calibration.expected_calibration_error, 3)} · slope {fmt(b.calibration.calibration_slope, 2)} · intercept {fmt(b.calibration.calibration_intercept, 2)}
              </p>
            </div>
            <div>
              <div className="label mb-2">Subgroups (AUROC, pooled internal + temporal test)</div>
              <ForestPlot
                min={0.4}
                max={1}
                rows={Object.entries(r.subgroups ?? {}).map(([k, v]) => ({ label: k, value: v.auroc?.value ?? null, lo: v.auroc?.ci95[0] ?? null, hi: v.auroc?.ci95[1] ?? null, n: v.n }))}
              />
            </div>
            <div>
              <div className="label mb-2">Feature importance (standardised log-odds)</div>
              <DivergingBars items={(r.feature_importance ?? []).slice(0, 8).map((f) => ({ label: f.feature.replace("missingindicator_", "missing: "), value: f.standardized_log_odds }))} />
              <div className="mt-1.5">
                <Note>Associations inside the model, not causal effects.</Note>
              </div>
            </div>
          </div>
        </>
      )}
      <div className="mt-3">
        <Note>{r.caveat}</Note>
      </div>
    </Panel>
  );
}
