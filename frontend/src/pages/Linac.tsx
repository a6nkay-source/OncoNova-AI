import { Play, Unplug } from "lucide-react";
import { useEffect, useState } from "react";
import { api, isUnavailable } from "../api/client";
import type { DoseResult, Scenario, Unavailable } from "../api/types";
import { LineChart, SERIES_COLORS } from "../components/charts";
import { DEFAULT_ANATOMY, Volume3D, lesionMeshes } from "../components/Volume3D";
import { ErrorState, EvidenceBadge, Note, Panel, Segmented, SimulatedBanner, Spinner, UnavailableState, cx, fmt } from "../components/ui";
import { useAsync } from "../state/hooks";
import { useApp } from "../state/AppState";
import { CaseViewer } from "../viewer/CaseViewer";
import { STRUCTURE_COLORS } from "../viewer/volume";

const STRUCT_LABEL: Record<string, string> = {
  illustrative_target: "Illustrative target (GTV + margin)",
  lung_left: "Left lung",
  lung_right: "Right lung",
  heart: "Heart",
  esophagus: "Oesophagus",
  spinal_cord: "Spinal cord",
  body: "Body",
};

export function LinacPage() {
  const { current, caseId, models, revision, bump } = useApp();
  const scen = useAsync(() => api.scenarios(), []);
  const [target, setTarget] = useState("reference");
  const [results, setResults] = useState<Record<string, DoseResult | Unavailable>>({});
  const [active, setActive] = useState<string>("S2");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [opacity, setOpacity] = useState(0.5);

  useEffect(() => {
    setResults({});
    if (!caseId || !scen.data) return;
    scen.data.scenarios.forEach((s) =>
      api.getDose(caseId, s.id, target).then((r) => !isUnavailable(r) && setResults((prev) => ({ ...prev, [s.id]: r }))),
    );
  }, [caseId, target, scen.data, revision]);

  if (!current || !caseId) return <Spinner label="Loading" />;
  const tp0 = current.timepoints[0].tp_id;
  const run = async (ids: string[]) => {
    setError(null);
    try {
      for (const id of ids) {
        setBusy(id);
        const r = await api.dose(caseId, id, target);
        setResults((prev) => ({ ...prev, [id]: r }));
      }
      bump();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  };
  const activeResult = results[active];
  const dose = activeResult && !isUnavailable(activeResult) ? (activeResult as DoseResult) : null;
  const computed = Object.values(results).filter((r): r is DoseResult => !isUnavailable(r));

  return (
    <div className="space-y-4">
      <SimulatedBanner />
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-ink-700/60 bg-ink-900/70 px-4 py-2 text-[12px] text-ink-300">
        <Unplug className="h-4 w-4 text-ink-400" />
        Hardware interface: <b className="text-ink-100">none</b>. OncoNova has no device, network or plan-export interface for treatment
        machines, and it never determines or recommends a prescription.
      </div>

      <Panel bodyClassName="flex flex-wrap items-center gap-3 py-3">
        <span className="label">Target contour</span>
        <Segmented
          value={target}
          onChange={setTarget}
          options={[
            { value: "reference", label: "Reference", disabled: !current.has_reference },
            ...models.filter((m) => m.status === "available").map((m) => ({ value: `pred:${m.id}`, label: m.id })),
            { value: "annotation", label: "Annotation" },
          ]}
        />
        <div className="ml-auto flex gap-2">
          <button className="btn-primary" disabled={!!busy || !scen.data} onClick={() => run([active])}>
            <Play className="h-3.5 w-3.5" /> {busy ? `Computing ${busy}...` : `Run ${active}`}
          </button>
          <button className="btn-ghost" disabled={!!busy || !scen.data} onClick={() => run(scen.data!.scenarios.map((s) => s.id))}>
            Compare all scenarios
          </button>
        </div>
      </Panel>
      {error ? <ErrorState error={error} /> : null}

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {scen.data?.scenarios.map((s) => (
          <ScenarioCard key={s.id} s={s} active={s.id === active} computed={!!results[s.id] && !isUnavailable(results[s.id])} onClick={() => setActive(s.id)} />
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Panel
          title={<span className="flex items-center gap-2">Dose illustration · {active} <EvidenceBadge kind="simulated" /></span>}
          className="min-h-[520px]"
          bodyClassName="flex flex-col"
          actions={
            <label className="flex w-32 items-center gap-2 text-[11px] text-ink-400">
              Opacity
              <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(+e.target.value)} aria-label="Dose opacity" />
            </label>
          }
        >
          {activeResult && isUnavailable(activeResult) && <UnavailableState data={activeResult as Unavailable} />}
          <CaseViewer
            caseId={caseId}
            tpId={tp0}
            modality={current.modality}
            defaultPreset="mediastinum"
            version={revision}
            overlays={[
              { id: "dose", url: dose ? api.mapUrl(caseId, tp0, "dose", { key: dose.dose_map }) : null, type: "map", lut: "dose", vmax: 1.1, threshold: 0.05, opacity, visible: !!dose },
              { id: "oar", url: current.has_structures ? api.mapUrl(caseId, tp0, "structures") : null, type: "label", outline: true, opacity: 0.45, colors: STRUCTURE_COLORS, visible: current.has_structures },
              { id: "tgt", url: current.has_reference && target === "reference" ? api.mapUrl(caseId, tp0, "reference") : null, type: "label", outline: true, opacity: 0.7, only: [1], colors: { 1: [255, 255, 255] }, visible: target === "reference" },
            ]}
          />
          <DoseLegend />
        </Panel>
        <Panel title="Dose-volume histogram (relative units)">
          {!dose && <Note>Run a scenario to compute its DVH. Values are relative to the median dose in the illustrative target.</Note>}
          {dose && (
            <LineChart
              height={300}
              markers={false}
              xLabel="Relative dose (% of target median)"
              yLabel="Volume (%)"
              xFormat={(v) => `${Math.round(v * 100)}`}
              yFormat={(v) => `${Math.round(v * 100)}`}
              yMin={0}
              yMax={1}
              xMax={1.2}
              series={Object.entries(dose.dvh.curves).map(([name, curve], i) => ({
                name: STRUCT_LABEL[name] ?? name,
                color: name === "illustrative_target" ? "#ffffff" : SERIES_COLORS[i % SERIES_COLORS.length],
                points: dose.dvh.dose_bins.map((x, j) => ({ x, y: curve[j] })).filter((p) => p.x <= 1.2),
              }))}
            />
          )}
          {dose?.warnings.slice(1).map((w) => (
            <Note key={w}>{w}</Note>
          ))}
        </Panel>
      </div>

      {computed.length > 0 && <ComparisonTable results={computed} />}
      <ImportDosePanel caseId={caseId} target={target} />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Panel title="Target, organs at risk and anatomy">
          <Volume3D caseId={caseId} tpId={tp0} version={revision} meshes={[...(current.has_structures ? DEFAULT_ANATOMY : []), ...lesionMeshes(target).slice(0, 1)]} />
        </Panel>
        <Panel title="How planning balances coverage and healthy tissue">
          <div className="space-y-3 text-[13px] leading-relaxed text-ink-300">
            <p>
              Radiation therapy planning tries to give the <b className="text-ink-100">target</b> enough dose while keeping{" "}
              <b className="text-ink-100">organs at risk</b> (lungs, heart, spinal cord, oesophagus) below their tolerances. The two goals pull against each other.
            </p>
            <ul className="list-disc space-y-1.5 pl-5">
              <li><b className="text-ink-100">More beam directions</b> spread entry dose, so the high-dose region fits the target more tightly, but a larger volume receives some low dose. Compare V5 and V50 across scenarios A-C.</li>
              <li><b className="text-ink-100">Margins</b> around the visible tumour account for microscopic spread, motion and setup uncertainty. Larger margins cover more uncertainty and put more normal tissue in the high-dose region (scenario D).</li>
              <li><b className="text-ink-100">DVHs</b> summarise each structure as the fraction of its volume that receives at least a given dose. A good plan has a steep target curve near 100% and organ curves that fall early.</li>
            </ul>
            <p className="text-[12px] text-ink-400">
              Real planning uses commissioned beam models, inverse optimisation and clinical constraints, chosen and checked by qualified physicists and
              physicians. This page only illustrates the geometry of those trade-offs.
            </p>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function ImportDosePanel({ caseId, target }: { caseId: string; target: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [result, setResult] = useState<(DoseResult & { import: Record<string, unknown>; dose_max: number }) | null>(null);
  const submit = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.importDose(caseId, file, target));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Panel title={<span className="flex items-center gap-2">Import research dose grid <EvidenceBadge kind="imported" /></span>}>
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="file"
            accept=".dcm,.nii,.gz"
            aria-label="Dose file"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="text-[12px] text-ink-300 file:mr-3 file:rounded-md file:border-0 file:bg-ink-750 file:px-3 file:py-1.5 file:text-[12px] file:text-ink-100"
          />
          <button className="btn-ghost" disabled={!file || busy} onClick={submit}>
            {busy ? "Importing..." : "Import"}
          </button>
        </div>
        {error ? <ErrorState error={error} /> : null}
        {result && (
          <div className="space-y-2 text-[12.5px]">
            <div className="font-semibold text-accent">{result.label}</div>
            <div className="kv">
              <span className="text-ink-400">Units (as stored)</span>
              <span className="mono text-right">{String(result.import.units)}</span>
              <span className="text-ink-400">Format</span>
              <span className="text-right">{String(result.import.source_format)}</span>
              <span className="text-ink-400">Max value</span>
              <span className="mono text-right">{result.dose_max.toFixed(3)}</span>
              <span className="text-ink-400">Non-zero fraction of grid</span>
              <span className="mono text-right">{String(result.import.nonzero_fraction_of_image_grid)}</span>
            </div>
            {Object.entries(result.dvh.stats).map(([k, v]) => (
              <div key={k} className="flex justify-between text-ink-300">
                <span>{STRUCT_LABEL[k] ?? k}</span>
                <span className="mono">mean {v.mean.toFixed(2)} · D2 {v.max_D2.toFixed(2)}</span>
              </div>
            ))}
            {result.warnings.map((w) => (
              <Note key={w}>{w}</Note>
            ))}
          </div>
        )}
        <Note>DICOM RTDOSE or a NIfTI dose grid aligned to this scan. It is resampled onto the image grid and shown as stored. OncoNova does not verify it, and nothing is exported or sent anywhere.</Note>
      </div>
    </Panel>
  );
}

function ScenarioCard({ s, active, computed, onClick }: { s: Scenario; active: boolean; computed: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cx(
        "panel p-3.5 text-left transition-colors",
        active ? "border-ev-simulated/60 ring-1 ring-ev-simulated/30" : "hover:border-ink-500",
      )}
    >
      <div className="flex items-center gap-2">
        <span className="font-mono text-[12px] text-ev-simulated">{s.id}</span>
        <span className="text-[10px] font-semibold uppercase tracking-wider text-ink-500">Hypothetical</span>
        {computed && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-teal" title="computed" />}
      </div>
      <div className="mt-1 text-[13px] font-medium text-ink-100">{s.name.replace(/^Hypothetical [A-D]: /, "")}</div>
      <p className="mt-1 text-[12px] leading-snug text-ink-400">{s.description}</p>
      <BeamDiagram n={s.n_beams} />
    </button>
  );
}

function BeamDiagram({ n }: { n: number }) {
  const angles = Array.from({ length: n }, (_, i) => (i * 2 * Math.PI) / n);
  return (
    <svg viewBox="-50 -40 100 80" className="mt-2 h-16 w-full" aria-hidden>
      <ellipse cx="0" cy="0" rx="34" ry="26" fill="none" stroke="#2e3f5e" strokeWidth="1.5" />
      <circle cx="12" cy="-2" r="5" fill="#e58cff" fillOpacity="0.5" />
      {angles.map((a, i) => (
        <line key={i} x1={12 - Math.sin(a) * 40} y1={-2 - Math.cos(a) * 40} x2="12" y2="-2" stroke="#e58cff" strokeOpacity={n > 12 ? 0.35 : 0.7} strokeWidth="1" />
      ))}
    </svg>
  );
}

function DoseLegend() {
  return (
    <div className="mt-2 flex items-center gap-2 text-[11px] text-ink-400">
      <span>5%</span>
      <span className="h-2 flex-1 rounded" style={{ background: "linear-gradient(90deg,#1e3cc8,#14aae6,#28d278,#f0dc28,#fa821e,#e61e32)" }} />
      <span>110% of target median</span>
    </div>
  );
}

function ComparisonTable({ results }: { results: DoseResult[] }) {
  const rows: { key: string; label: string; get: (r: DoseResult) => number | undefined; pct?: boolean; better: "high" | "low" }[] = [
    { key: "d95", label: "Target D95", get: (r) => r.dvh.stats.illustrative_target?.D95, pct: true, better: "high" },
    { key: "hi", label: "Target homogeneity index", get: (r) => r.dvh.stats.illustrative_target?.homogeneity_index, better: "low" },
    { key: "ci", label: "Paddick conformity (95%)", get: (r) => r.dvh.paddick_conformity_index_95, better: "high" },
    { key: "v5", label: "Body V5", get: (r) => r.dvh.stats.body?.V5, pct: true, better: "low" },
    { key: "v50", label: "Body V50", get: (r) => r.dvh.stats.body?.V50, pct: true, better: "low" },
    { key: "lung", label: "Ipsilateral lung mean", get: (r) => Math.max(r.dvh.stats.lung_left?.mean ?? 0, r.dvh.stats.lung_right?.mean ?? 0), pct: true, better: "low" },
    { key: "heart", label: "Heart mean", get: (r) => r.dvh.stats.heart?.mean, pct: true, better: "low" },
    { key: "cord", label: "Spinal cord D2 (near-max)", get: (r) => r.dvh.stats.spinal_cord?.max_D2, pct: true, better: "low" },
  ];
  const sorted = [...results].sort((a, b) => a.scenario.id.localeCompare(b.scenario.id));
  return (
    <Panel title={<span className="flex items-center gap-2">Scenario comparison <EvidenceBadge kind="simulated" /></span>}>
      <div className="overflow-x-auto">
        <table className="table-base">
          <thead>
            <tr>
              <th>Metric (relative units)</th>
              {sorted.map((r) => (
                <th key={r.scenario.id} className="text-right">
                  {r.scenario.id} · {r.scenario.n_beams} beams · {r.scenario.margin_mm} mm
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const vals = sorted.map(row.get);
              const finite = vals.filter((v): v is number => v !== undefined);
              const best = row.better === "high" ? Math.max(...finite) : Math.min(...finite);
              return (
                <tr key={row.key}>
                  <td className="text-ink-300">{row.label}</td>
                  {vals.map((v, i) => (
                    <td key={i} className={cx("text-right font-mono", v === best && finite.length > 1 && "text-teal")}>
                      {v === undefined ? "-" : row.pct ? `${(v * 100).toFixed(1)}%` : fmt(v, 3)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-2">
        <Note>Teal marks the most favourable value in each row for this geometric toy. No scenario is a recommendation, and these numbers mean nothing for a real patient.</Note>
      </div>
    </Panel>
  );
}
