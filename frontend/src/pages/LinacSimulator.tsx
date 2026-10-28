import { FastForward, Pause, Play, RotateCcw, Unplug } from "lucide-react";
import { useState } from "react";
import { api, isUnavailable } from "../api/client";
import type { DoseResult, Scenario, Unavailable } from "../api/types";
import { LineChart, SERIES_COLORS } from "../components/charts";
import { DoseReplay, PatientMonitor } from "../components/SimPanels";
import { TreatmentRoom } from "../components/TreatmentRoom";
import { ErrorState, EvidenceBadge, Note, Panel, Segmented, SimulatedBanner, Spinner, Stat, UnavailableState, cx } from "../components/ui";
import { isArc, type SimSnapshot } from "../sim/treatment";
import { useTreatmentSim } from "../sim/useTreatmentSim";
import { useAsync, useVolume } from "../state/hooks";
import { useApp } from "../state/AppState";

const STRUCT_LABEL: Record<string, string> = {
  illustrative_target: "Target (lesion + margin)",
  lung_left: "Left lung",
  lung_right: "Right lung",
  heart: "Heart",
  esophagus: "Oesophagus",
  spinal_cord: "Spinal cord",
  body: "Whole body",
};

const SHORT: Record<string, string> = { S1: "3 beams", S2: "7 beams", S3: "Arc", S4: "7 beams, wide margin" };

const clock = (t: number) => `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

export function LinacSimulatorPage() {
  const { current, caseId, modelId } = useApp();
  const scen = useAsync(() => api.scenarios(), []);
  const [scenarioId, setScenarioId] = useState("S2");
  const [aim, setAim] = useState<"truth" | "ai">("truth");
  const [speed, setSpeed] = useState("2");
  const [breathing, setBreathing] = useState(true);
  const [gating, setGating] = useState(false);

  const targetSource = aim === "truth" && current?.has_reference ? "reference" : `pred:${modelId}`;
  const tp0 = current?.timepoints[0]?.tp_id ?? null;
  // Use the stored illustration when it already has a per-beam replay; otherwise compute it.
  const plan = useAsync(
    async () => {
      const stored = await api.getDose(caseId!, scenarioId, targetSource);
      if (!isUnavailable(stored) && (stored as DoseResult).beams) return stored;
      return api.dose(caseId!, scenarioId, targetSource);
    },
    [caseId, scenarioId, targetSource],
    !!caseId,
  );
  // Ignore the previous selection's result while the new one loads.
  const dose =
    plan.data && !isUnavailable(plan.data) && (plan.data as DoseResult).scenario.id === scenarioId && (plan.data as DoseResult).target_source === targetSource
      ? (plan.data as DoseResult)
      : null;
  const version = dose?.saved_utc ? Date.parse(dose.saved_utc) : 0;
  const beams = useVolume(dose ? api.doseBeamsUrl(caseId!, scenarioId, targetSource) : null, version);
  const image = useVolume(caseId && tp0 ? api.imageUrl(caseId, tp0) : null);
  const targetMask = useVolume(
    dose && caseId && tp0
      ? targetSource === "reference"
        ? api.mapUrl(caseId, tp0, "reference")
        : api.mapUrl(caseId, tp0, "prediction", { model_id: modelId })
      : null,
    version,
  );
  const angles = dose?.beams?.angles_deg ?? [];
  const sim = useTreatmentSim({ angles, speed: Number(speed), breathing, gating });
  const scenario = scen.data?.scenarios.find((s) => s.id === scenarioId);

  if (!current || !caseId || !tp0) return <Spinner label="Loading" />;
  const snap = sim.snap;
  const running = snap.status === "running";

  return (
    <div className="space-y-4">
      <SimulatedBanner text="A replay of a geometric dose illustration on a dummy (synthetic) patient, in relative units. It is not a dose calculation, a treatment plan or a prescription, and OncoNova has no connection to any treatment machine." />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Panel
          title={<span className="flex items-center gap-2">Treatment room <EvidenceBadge kind="simulated" /></span>}
          actions={<span className="text-[11px] text-ink-500">Drag to look around · scroll to zoom</span>}
          bodyClassName="p-2"
        >
          <TreatmentRoom
            caseId={caseId}
            tpId={tp0}
            targetSource={targetSource}
            hasStructures={current.has_structures}
            simRef={sim.ref}
            marginMm={scenario?.margin_mm ?? 5}
            overlay={<RoomHud snap={snap} arc={isArc(angles)} />}
          />
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel title="Simulation controls">
            <div className="space-y-3.5">
              <div>
                <div className="label mb-1.5">Hypothetical beam arrangement</div>
                <div className="grid grid-cols-2 gap-1.5">
                  {scen.data?.scenarios.map((s) => (
                    <ScenarioButton key={s.id} s={s} active={s.id === scenarioId} onClick={() => setScenarioId(s.id)} />
                  ))}
                </div>
              </div>
              {current.has_reference && (
                <div className="flex items-center justify-between gap-2">
                  <span className="label">Aim at</span>
                  <Segmented
                    size="xs"
                    value={aim}
                    onChange={setAim}
                    options={[
                      { value: "truth", label: "True outline" },
                      { value: "ai", label: "AI drawing" },
                    ]}
                  />
                </div>
              )}
              <div className="flex flex-wrap gap-2">
                {running ? (
                  <button className="btn-primary flex-1" onClick={sim.pause}>
                    <Pause className="h-3.5 w-3.5" /> Pause
                  </button>
                ) : (
                  <button className="btn-primary flex-1" onClick={sim.start} disabled={!angles.length}>
                    <Play className="h-3.5 w-3.5" /> {snap.status === "paused" ? "Resume" : snap.status === "done" ? "Replay" : "Start simulation"}
                  </button>
                )}
                <button className="btn-ghost" onClick={sim.reset} disabled={snap.status === "ready"} title="Back to the start">
                  <RotateCcw className="h-3.5 w-3.5" />
                </button>
                <button className="btn-ghost" onClick={sim.skip} disabled={!angles.length || snap.status === "done"} title="Skip to the end">
                  <FastForward className="h-3.5 w-3.5" />
                </button>
              </div>
              <div className="flex items-center justify-between gap-2">
                <span className="label">Speed</span>
                <Segmented size="xs" value={speed} onChange={setSpeed} options={["1", "2", "4"].map((v) => ({ value: v, label: `${v}×` }))} />
              </div>
              <Toggle checked={breathing} onChange={setBreathing} label="Patient breathing" hint="The dummy breathes; the lesion moves up to 8 mm with each breath." />
              <Toggle
                checked={gating}
                onChange={setGating}
                disabled={!breathing}
                label="Breath-hold gating"
                hint="Hold the beam whenever the lesion is outside a window near exhale. Takes longer, keeps the target where it was imaged."
              />
            </div>
          </Panel>

          <Panel title={<span className="flex items-center gap-2">Dummy patient monitor <EvidenceBadge kind="simulated" /></span>}>
            <PatientMonitor simRef={sim.ref} snap={snap} gating={gating && breathing} />
          </Panel>
        </div>
      </div>

      {plan.error ? <ErrorState error={plan.error} onRetry={plan.reload} /> : null}
      {plan.data && isUnavailable(plan.data) && (
        <UnavailableState
          title="Nothing to aim at yet"
          data={{
            ...(plan.data as Unavailable),
            how_to_fix: targetSource.startsWith("pred:")
              ? "Open CT Scan Analysis first: the AI draws the tumour on every scan automatically."
              : (plan.data as Unavailable).how_to_fix,
          }}
        />
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Panel
          title={<span className="flex items-center gap-2">Dose building up, beam by beam <EvidenceBadge kind="simulated" /></span>}
          actions={<span className="mono text-ink-500">{dose?.beams ? `axial slice z${dose.beams.slice_index}` : ""}</span>}
        >
          {plan.loading && !dose && <Spinner label="Preparing the illustration" />}
          {dose && beams.volume && image.volume ? (
            <div className="space-y-3">
              <DoseReplay
                image={image.volume}
                beams={beams.volume}
                target={targetMask.volume}
                sliceIndex={dose.beams!.slice_index}
                snap={snap}
                angles={angles}
                fieldMm={(scenario?.margin_mm ?? 5) + 18}
              />
              <Progress fraction={snap.fraction} />
              <div className="flex items-center gap-2 text-[11px] text-ink-400">
                <span>5%</span>
                <span className="h-2 flex-1 rounded" style={{ background: "linear-gradient(90deg,#1e3cc8,#14aae6,#28d278,#f0dc28,#fa821e,#e61e32)" }} />
                <span>110% of the target median</span>
              </div>
              <Note>White outline: the target. Dots around the edge mark the beam directions; they fill in as each beam is replayed.</Note>
            </div>
          ) : (
            dose && (beams.error ? <ErrorState error={beams.error} /> : <Spinner label="Loading the replay" />)
          )}
        </Panel>

        <Panel title={<span className="flex items-center gap-2">Result of the illustration <EvidenceBadge kind="simulated" /></span>}>
          {!dose && !plan.loading && <Note>Choose a beam arrangement to prepare an illustration.</Note>}
          {dose && snap.status !== "done" && (
            <div className="flex h-full min-h-[240px] flex-col items-center justify-center gap-2 text-center text-[13px] text-ink-400">
              <p>The dose-volume histogram appears when the replay finishes.</p>
              <button className="btn-ghost" onClick={sim.skip}>
                <FastForward className="h-3.5 w-3.5" /> Show it now
              </button>
            </div>
          )}
          {dose && snap.status === "done" && <ResultView dose={dose} snap={snap} gating={gating} breathing={breathing} />}
        </Panel>
      </div>

      <Panel title="What this simulator shows">
        <div className="grid gap-4 text-[13px] leading-relaxed text-ink-300 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
          <ul className="list-disc space-y-1.5 pl-5">
            <li>
              <b className="text-ink-100">More beam directions</b> spread the entry dose, so the high-dose region hugs the target more tightly, but more healthy
              tissue gets some low dose. Compare 3 beams, 7 beams and the arc.
            </li>
            <li>
              <b className="text-ink-100">Margins</b> around the lesion allow for motion and setup uncertainty. A wider margin covers more uncertainty and puts
              more normal tissue in the high-dose region.
            </li>
            <li>
              <b className="text-ink-100">Breathing</b> moves a lung lesion. Without gating, part of the beam-on time happens while the lesion is away from
              where it was imaged; gating waits for exhale instead.
            </li>
          </ul>
          <div className="space-y-2">
            <div className="flex items-start gap-2 rounded-lg border border-ink-700/60 bg-ink-925/60 p-3 text-[12px] text-ink-300">
              <Unplug className="mt-0.5 h-4 w-4 shrink-0 text-ink-400" />
              <span>
                Hardware interface: <b className="text-ink-100">none</b>. OncoNova cannot send commands to a treatment machine, export a plan, or determine a
                prescription. The gantry, beam and patient here exist only in your browser.
              </span>
            </div>
            <p className="text-[12px] text-ink-400">
              The dose picture is a simple geometric model (straight beams, one attenuation coefficient). Real planning uses commissioned beam models and
              optimisation, chosen and checked by qualified physicists and physicians.
            </p>
          </div>
        </div>
      </Panel>
    </div>
  );
}

function RoomHud({ snap, arc }: { snap: SimSnapshot; arc: boolean }) {
  const state = snap.status === "done"
    ? { text: "Complete", cls: "border-teal/50 bg-teal/15 text-teal" }
    : snap.beamOn
      ? { text: "Beam on", cls: "border-[#f08cff]/60 bg-[#f08cff]/20 text-[#f8c8ff]" }
      : snap.hold
        ? { text: "Beam held · waiting for exhale", cls: "border-[#ffb020]/60 bg-[#ffb020]/15 text-[#ffd27a]" }
        : snap.moving
          ? { text: "Gantry moving", cls: "border-accent/50 bg-accent/15 text-accent" }
          : { text: snap.status === "paused" ? "Paused" : "Ready", cls: "border-ink-600 bg-ink-900/80 text-ink-300" };
  return (
    <>
      <div className="pointer-events-none absolute left-3 top-3 flex flex-col gap-1.5">
        <span className="rounded-md border border-ev-simulated/50 bg-ink-950/70 px-2 py-0.5 text-[10.5px] font-bold uppercase tracking-[0.14em] text-ev-simulated">
          Simulation · dummy patient
        </span>
        <span className={cx("w-fit rounded-md border px-2 py-0.5 text-[12px] font-semibold", state.cls)}>{state.text}</span>
      </div>
      <div className="pointer-events-none absolute right-3 top-3 rounded-lg border border-ink-700/70 bg-ink-950/70 px-3 py-2 text-right">
        <div className="text-[10px] uppercase tracking-wider text-ink-500">Gantry angle</div>
        <div className="font-mono text-[22px] leading-tight tabular-nums text-ink-100">{snap.gantry.toFixed(0)}°</div>
      </div>
      <div className="pointer-events-none absolute inset-x-3 bottom-3 flex flex-wrap items-end justify-between gap-2">
        <div className="flex gap-3 rounded-lg border border-ink-700/70 bg-ink-950/70 px-3 py-1.5 font-mono text-[12px] text-ink-300">
          <span>{arc ? "Arc" : `Beam ${Math.min(snap.beam + 1, snap.nBeams)} / ${snap.nBeams}`}</span>
          <span>Time {clock(snap.t)}</span>
          <span>Delivered {Math.round(snap.fraction * 100)}%</span>
        </div>
      </div>
    </>
  );
}

function ScenarioButton({ s, active, onClick }: { s: Scenario; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      title={s.description}
      className={cx(
        "rounded-lg border px-2.5 py-2 text-left transition-colors",
        active ? "border-ev-simulated/60 bg-ev-simulated/10 ring-1 ring-ev-simulated/30" : "border-ink-700 hover:border-ink-500",
      )}
    >
      <div className="text-[12.5px] font-medium text-ink-100">{SHORT[s.id] ?? s.name}</div>
      <div className="text-[11px] text-ink-400">
        {s.n_beams > 20 ? "36 angles" : `${s.n_beams} directions`} · {s.margin_mm} mm margin
      </div>
    </button>
  );
}

function Toggle({ checked, onChange, label, hint, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint: string; disabled?: boolean }) {
  return (
    <label className={cx("flex cursor-pointer items-start gap-2.5", disabled && "cursor-not-allowed opacity-50")}>
      <input type="checkbox" className="mt-0.5" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span>
        <span className="block text-[13px] font-medium text-ink-100">{label}</span>
        <span className="block text-[11.5px] leading-snug text-ink-400">{hint}</span>
      </span>
    </label>
  );
}

function Progress({ fraction }: { fraction: number }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-[11.5px] text-ink-400">
        <span>Share of the illustration replayed</span>
        <span className="font-mono text-ink-200">{Math.round(fraction * 100)}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-ink-800">
        <div className="h-full rounded-full bg-gradient-to-r from-[#b45cff] to-[#f08cff] transition-[width] duration-100" style={{ width: `${fraction * 100}%` }} />
      </div>
    </div>
  );
}

function ResultView({ dose, snap, gating, breathing }: { dose: DoseResult; snap: SimSnapshot; gating: boolean; breathing: boolean }) {
  const st = dose.dvh.stats;
  const pct = (v: number | undefined) => (v === undefined ? "-" : `${Math.round(v * 100)}`);
  const lung = Math.max(st.lung_left?.mean ?? 0, st.lung_right?.mean ?? 0);
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat label="Target D95" value={pct(st.illustrative_target?.D95)} unit="%" sub="of target median" />
        <Stat label="Body V5" value={pct(st.body?.V5)} unit="%" sub="gets ≥ 5%" />
        <Stat label="Lung mean" value={st.lung_left ? pct(lung) : "-"} unit="%" sub="higher of two" />
        <Stat label="Heart mean" value={pct(st.heart?.mean)} unit="%" />
      </div>
      <LineChart
        height={220}
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
      {breathing && snap.beamOnTime > 0 && (
        <Note>
          {gating
            ? `Gating held the beam for ${snap.holdTime.toFixed(0)} s of breathing, so the lesion stayed within 2 mm of its imaged position while the beam was on.`
            : `The lesion was more than 2 mm from its imaged position for ${Math.round((snap.displacedTime / snap.beamOnTime) * 100)}% of beam-on time. The dose picture above ignores this motion.`}
        </Note>
      )}
      <Note>Each structure's curve is the share of its volume receiving at least that relative dose. These numbers describe a toy model and mean nothing for a real patient.</Note>
    </div>
  );
}
