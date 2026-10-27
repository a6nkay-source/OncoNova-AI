import { GitCompareArrows, TrendingUp } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, isUnavailable } from "../api/client";
import type { LongitudinalResult, ProgressionPrediction, Unavailable } from "../api/types";
import { DivergingBars, IntervalBar, LineChart, SERIES_COLORS } from "../components/charts";
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
import { useAsync } from "../state/hooks";
import { useApp } from "../state/AppState";
import { CaseViewer } from "../viewer/CaseViewer";

const FEATURE_LABEL: Record<string, string> = {
  baseline_total_volume_cc: "Baseline lesion volume (cc)",
  log_volume_ratio_per_100d: "Growth: log volume ratio / 100 d",
  new_lesion_count: "New segmented regions",
  node_region_lesion_observed: "Node-region lesion observed",
  primary_sphericity: "Primary sphericity",
  age_years: "Age (years)",
};
const featureLabel = (f: string) =>
  f.startsWith("missingindicator_") ? `${FEATURE_LABEL[f.slice(17)] ?? f.slice(17)} missing` : (FEATURE_LABEL[f] ?? f);

export function ProgressionPage() {
  const { current, caseId, models, revision, bump } = useApp();
  const sources = [
    { value: "reference", label: "Reference masks", disabled: !current?.has_reference },
    ...models.filter((m) => m.status === "available").map((m) => ({ value: `pred:${m.id}`, label: m.id })),
    { value: "annotation", label: "Annotations" },
  ];
  const [source, setSource] = useState(current?.has_reference ? "reference" : `pred:${models[0]?.id ?? "demo-classical"}`);
  const longi = useAsync(() => api.getLongitudinal(caseId!, source), [caseId, source, revision], !!caseId);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [risk, setRisk] = useState<ProgressionPrediction | Unavailable | null>(null);
  useEffect(() => {
    // Default to the reference masks whenever the selected case has them.
    setSource(current?.has_reference ? "reference" : `pred:${models.find((m) => m.status === "available")?.id ?? "demo-classical"}`);
    setRisk(null);
  }, [current?.case_id]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!current || !caseId) return <Spinner label="Loading" />;
  const single = current.timepoints.length < 2;

  const runLongitudinal = async () => {
    setBusy("longitudinal");
    setError(null);
    try {
      longi.setData(await api.longitudinal(caseId, source));
      bump();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  };
  const runRisk = async () => {
    setBusy("risk");
    setError(null);
    try {
      setRisk(await api.progression(caseId, source));
      bump();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  };
  const result = longi.data && !isUnavailable(longi.data) ? (longi.data as LongitudinalResult) : null;

  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-3">
        <TaskCard n={1} title="Observed lesion detection" kind="model_output" text="Regions visible in each scan, from a segmentation model or a reference mask. See Tumor Segmentation." />
        <TaskCard n={2} title="Longitudinal change analysis" kind="measured" text="Registers follow-ups to baseline and measures what changed between existing scans. No prediction." />
        <TaskCard n={3} title="Future progression risk" kind="model_output" text="A model's estimate of a defined outcome over a fixed horizon, with an interval. A hypothesis, not a finding." />
      </div>

      <Panel bodyClassName="flex flex-wrap items-center gap-3 py-3">
        <span className="label">Lesion source</span>
        <Segmented value={source} onChange={(v) => { setSource(v); setRisk(null); }} options={sources} />
        <div className="ml-auto flex gap-2">
          <button className="btn-primary" onClick={runLongitudinal} disabled={!!busy}>
            <GitCompareArrows className="h-3.5 w-3.5" /> {busy === "longitudinal" ? "Registering..." : "Compare scans"}
          </button>
          <button className="btn-ghost" onClick={runRisk} disabled={!!busy}>
            <TrendingUp className="h-3.5 w-3.5" /> {busy === "risk" ? "Estimating..." : "Estimate risk"}
          </button>
        </div>
      </Panel>
      {error ? <ErrorState error={error} /> : null}
      {source.startsWith("pred:") && (
        <Note>Model-mask source: segment every timepoint first. Volumes are then measurements of AI output, not of reference annotations.</Note>
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <Panel title={<span className="flex items-center gap-2">2 · Observed change <EvidenceBadge kind={source === "reference" ? "measured" : "measured_from_model_output"} /></span>}>
          {single && (
            <UnavailableState
              title="Single scan"
              data={{ status: "unavailable", kind: "unavailable", reason: "This case has one scan, so nothing can be compared over time.", missing_inputs: ["second timepoint"], how_to_fix: "Import a follow-up scan." }}
            />
          )}
          {!single && longi.loading && !longi.data && <Spinner label="Loading" />}
          {!single && longi.data && isUnavailable(longi.data) && <UnavailableState data={longi.data as Unavailable} title="Not computed" />}
          {result && <LongitudinalView result={result} />}
        </Panel>

        <Panel title={<span className="flex items-center gap-2">3 · Future progression risk <EvidenceBadge kind="model_output" /></span>}>
          {!risk && <RiskIntro />}
          {risk && isUnavailable(risk) && <UnavailableState data={risk as Unavailable} title="No estimate" />}
          {risk && !isUnavailable(risk) && <RiskView p={risk as ProgressionPrediction} />}
        </Panel>
      </div>

      {result && !single && <SideBySide caseId={caseId} result={result} source={source} />}
    </div>
  );
}

function TaskCard({ n, title, text, kind }: { n: number; title: string; text: string; kind: "measured" | "model_output" }) {
  return (
    <div className="panel p-3.5">
      <div className="flex items-center gap-2">
        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-ink-750 font-mono text-[11px] text-ink-200">{n}</span>
        <span className="text-[13px] font-semibold text-ink-100">{title}</span>
        <EvidenceBadge kind={kind} className="ml-auto" />
      </div>
      <p className="mt-1.5 text-[12px] leading-snug text-ink-400">{text}</p>
    </div>
  );
}

function LongitudinalView({ result }: { result: LongitudinalResult }) {
  const days = result.timepoints.map((t) => t.days_from_baseline);
  const series = [
    { name: "Total lesion volume", color: "#e4ebf5", points: result.timepoints.map((t) => ({ x: t.days_from_baseline, y: t.total_lesion_volume_cc })) },
    ...result.tracks.map((t, i) => ({
      name: `${t.track_id} ${t.class.replace(/_/g, " ")}`,
      color: SERIES_COLORS[i % SERIES_COLORS.length],
      dashed: t.status === "new",
      points: t.volume_series_cc.map((v, j) => ({ x: days[j], y: v })),
    })),
  ];
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-2">
        <Stat label="Total volume change" value={result.total_volume_change_pct === null ? "-" : `${result.total_volume_change_pct > 0 ? "+" : ""}${result.total_volume_change_pct.toFixed(1)}`} unit="%" />
        <Stat label="Tracked regions" value={result.tracks.length} />
        <Stat label="New" value={result.new_lesion_count} kind={result.new_lesion_count ? "requires_expert_review" : undefined} />
      </div>
      <LineChart series={series} xLabel="Days from baseline" yLabel="Volume (cc)" yFormat={(v) => v.toFixed(0)} />
      <table className="table-base">
        <thead>
          <tr>
            <th>Track</th>
            <th>Status</th>
            {result.timepoints.map((t) => (
              <th key={t.tp_id} className="text-right">
                {t.tp_id} cc
              </th>
            ))}
            <th className="text-right">Δ%</th>
            <th className="text-right">Shift mm</th>
          </tr>
        </thead>
        <tbody>
          {result.tracks.map((t) => (
            <tr key={t.track_id}>
              <td>
                <span className="font-mono">{t.track_id}</span> <span className="text-ink-400">{t.class.replace(/_/g, " ")}</span>
              </td>
              <td>
                {t.status === "new" ? <EvidenceBadge kind="requires_expert_review" /> : <span className="text-ink-300">{t.status.replace(/_/g, " ")}</span>}
              </td>
              {t.volume_series_cc.map((v, i) => (
                <td key={i} className="text-right font-mono">
                  {v === null ? "-" : v.toFixed(2)}
                </td>
              ))}
              <td className={cx("text-right font-mono", (t.volume_change_pct ?? 0) > 20 && "text-ev-review", (t.volume_change_pct ?? 0) < -20 && "text-teal")}>
                {t.volume_change_pct === undefined ? "-" : `${t.volume_change_pct > 0 ? "+" : ""}${t.volume_change_pct.toFixed(1)}`}
              </td>
              <td className="text-right font-mono">{fmt(t.centroid_shift_mm, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div>
        <div className="label mb-1.5">Registration to baseline</div>
        <table className="table-base">
          <thead>
            <tr>
              <th>Scan</th>
              <th className="text-right">Translation mm</th>
              <th className="text-right">Rotation °</th>
              <th className="text-right">Metric</th>
              <th>QC</th>
            </tr>
          </thead>
          <tbody>
            {result.timepoints.slice(1).map((t) => (
              <tr key={t.tp_id}>
                <td className="font-mono">{t.tp_id}</td>
                <td className="text-right font-mono">{t.registration?.translation_mm.map((v) => v.toFixed(1)).join(", ")}</td>
                <td className="text-right font-mono">{t.registration?.rotation_deg.map((v) => v.toFixed(1)).join(", ")}</td>
                <td className="text-right font-mono">{fmt(t.registration?.final_metric, 3)}</td>
                <td>{t.registration?.qc === "ok" ? <span className="text-teal">ok</span> : <EvidenceBadge kind="requires_expert_review" />}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <FindingList items={result.findings} />
      {result.notes.map((n) => (
        <Note key={n}>{n}</Note>
      ))}
    </div>
  );
}

function RiskIntro() {
  const card = useAsync(() => api.progressionModel(), []);
  const c = card.data as Record<string, unknown> | null;
  return (
    <div className="space-y-3 text-[13px] text-ink-300">
      <p>Estimates the probability of a defined outcome using longitudinal imaging features and optional clinical fields.</p>
      {c && !isUnavailable(c) && (
        <div className="kv rounded-lg bg-ink-925 p-3">
          <span className="text-ink-400">Outcome</span>
          <span className="text-right text-ink-200">{String(c.outcome_definition)}</span>
          <span className="text-ink-400">Horizon</span>
          <span className="text-right font-mono">{String(c.horizon_days)} days</span>
          <span className="text-ink-400">Model</span>
          <span className="text-right">{String(c.kind)}</span>
          <span className="text-ink-400">Trained on</span>
          <span className="text-right">{String(c.trained_on)}</span>
        </div>
      )}
      {c && isUnavailable(c) && <UnavailableState data={c as unknown as Unavailable} />}
      <Note>This model has no labels for metastasis sites and does not predict where disease might spread.</Note>
    </div>
  );
}

function RiskView({ p }: { p: ProgressionPrediction }) {
  const contribs = useMemo(
    () => p.contributions.slice(0, 7).map((c) => ({ label: featureLabel(c.feature), value: c.contribution_log_odds })),
    [p.contributions],
  );
  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-ev-model/30 bg-ev-model/5 p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="label">Estimated probability</div>
            <div className="mt-1 font-mono text-[34px] font-medium leading-none text-ink-100">{p.probability.toFixed(2)}</div>
            <div className="mt-1 text-[12px] text-ink-400">
              95% interval {p.interval95[0].toFixed(2)}-{p.interval95[1].toFixed(2)} · {p.training_percentile.toFixed(0)}th percentile of training predictions
            </div>
          </div>
          <div className="flex flex-col items-end gap-1">
            <EvidenceBadge kind="model_output" />
            {p.uncertain && <EvidenceBadge kind="uncertain" />}
            <EvidenceBadge kind="requires_expert_review" />
          </div>
        </div>
        <div className="mt-4">
          <IntervalBar p={p.probability} lo={p.interval95[0]} hi={p.interval95[1]} />
        </div>
        <p className="mt-3 text-[12px] text-ink-300">
          Outcome: {p.outcome} Horizon {p.horizon_days} days. {p.label}.
        </p>
      </div>
      <div>
        <div className="label mb-2">Contributions to this estimate (log-odds)</div>
        <DivergingBars items={contribs} format={(v) => (v > 0 ? "+" : "") + v.toFixed(2)} />
        <div className="mt-1.5">
          <Note>Red raises the estimate, blue lowers it. These are associations inside a linear model, not causes.</Note>
        </div>
      </div>
      <div className="kv rounded-lg bg-ink-925 p-3">
        {Object.entries(p.inputs).map(([k, v]) => (
          <div key={k} className="contents">
            <span className="text-ink-400">{FEATURE_LABEL[k] ?? k}</span>
            <span className={cx("mono text-right", v === null && "text-ev-uncertain")}>{v === null ? "missing → imputed" : Number(v).toFixed(3)}</span>
          </div>
        ))}
      </div>
      <p className="text-[11.5px] text-ink-500">Interval: {p.interval_method}.</p>
      {p.out_of_distribution_warnings.length > 0 && (
        <FindingList items={p.out_of_distribution_warnings.map((w) => ({ kind: "uncertain", text: w }))} />
      )}
      <FindingList items={p.limitations.map((l) => ({ kind: "uncertain", text: l }))} />
    </div>
  );
}

function SideBySide({ caseId, result, source }: { caseId: string; result: LongitudinalResult; source: string }) {
  const followups = result.timepoints.slice(1).map((t) => t.tp_id);
  const [tp, setTp] = useState(followups[followups.length - 1]);
  const [pos, setPos] = useState<[number, number, number] | null>(null);
  const base = result.baseline_tp;
  const baseMask =
    source === "reference" ? api.mapUrl(caseId, base, "reference") : source === "annotation" ? api.mapUrl(caseId, base, "annotation") : api.mapUrl(caseId, base, "prediction", { model_id: source.slice(5) });
  return (
    <Panel
      title="Baseline vs follow-up (registered to baseline)"
      actions={<Segmented size="xs" value={tp} onChange={setTp} options={followups.map((f) => ({ value: f, label: f }))} />}
    >
      <div className="grid gap-3 lg:grid-cols-2">
        <div className="h-[420px]">
          <div className="label mb-1">{base} · baseline</div>
          <CaseViewer caseId={caseId} tpId={base} layout="axial" overlays={[{ id: "b", url: baseMask, type: "label", opacity: 0.5, outline: true }]} onPositionChange={setPos} syncPosition={pos} />
        </div>
        <div className="h-[420px]">
          <div className="label mb-1">{tp} · resampled into baseline frame</div>
          <CaseViewer
            caseId={caseId}
            tpId={tp}
            imageUrl={api.registeredUrl(caseId, tp, "image")}
            layout="axial"
            overlays={[{ id: "f", url: api.registeredUrl(caseId, tp, "lesions", source), type: "label", opacity: 0.5, outline: true }]}
            onPositionChange={setPos}
            syncPosition={pos}
          />
        </div>
      </div>
      <div className="mt-2">
        <Note>Slices are linked: scrolling or clicking one view moves the other. Registration is rigid; residual misalignment from breathing or deformation is expected.</Note>
      </div>
    </Panel>
  );
}

