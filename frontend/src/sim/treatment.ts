/**
 * Replay clock for the LINAC simulator page. SIMULATED - NON-CLINICAL.
 *
 * Drives a virtual gantry through the beam angles of a precomputed geometric dose
 * illustration and tracks how much of each beam's share has been "delivered" so the
 * page can build the dose picture up beam by beam. It also breathes a dummy patient
 * (Lujan-style cos^4 waveform, so most of each cycle is spent near exhale) and can hold
 * the beam outside an exhale gating window. Nothing here models dose physics or talks
 * to any device; it only animates numbers the backend already computed.
 */

export const GANTRY_DEG_PER_S = 12;
export const STATIC_BEAM_ON_TOTAL_S = 24; // shared across the beams of a static plan
export const ARC_BEAM_THRESHOLD = 20; // plans with this many angles replay as one continuous arc
export const BREATH_AMPLITUDE_MM = 8; // superior-inferior target motion at full inhale
export const GATE_FRACTION = 0.25; // beam allowed while the target is within 25% of the amplitude of exhale
const HISTORY = 600; // samples kept for the monitor trace
const HISTORY_HZ = 20;

export type SimStatus = "ready" | "running" | "paused" | "done";

export interface SimConfig {
  angles: number[];
  speed: number;
  breathing: boolean;
  gating: boolean;
}

export interface SimState {
  status: SimStatus;
  /** Delivery clock in simulated seconds (scaled by speed). */
  t: number;
  /** Wall clock in seconds; the dummy patient breathes in real time. */
  wall: number;
  /** Unwrapped gantry angle (degrees). Display with ``gantryDisplay``. */
  g: number;
  beam: number;
  phase: "move" | "deliver";
  delivered: number[];
  beamOn: boolean;
  hold: boolean;
  /** 0 = exhale rest position, 1 = full inhale. */
  breath: number;
  breathPhase: number;
  period: number;
  beamOnTime: number;
  holdTime: number;
  /** Beam-on time while the target was outside the gating window. */
  displacedTime: number;
  histBreath: Float32Array;
  histBeam: Uint8Array; // 0 off, 1 on, 2 held
  histHead: number;
  histCount: number;
  histAccum: number;
}

export const isArc = (angles: number[]) => angles.length >= ARC_BEAM_THRESHOLD;
const sector = (angles: number[]) => 360 / Math.max(angles.length, 1);
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

export function createSim(angles: number[]): SimState {
  return {
    status: "ready",
    t: 0,
    wall: 0,
    g: startAngle(angles),
    beam: 0,
    phase: "move",
    delivered: angles.map(() => 0),
    beamOn: false,
    hold: false,
    breath: 0,
    breathPhase: Math.PI / 2,
    period: 4,
    beamOnTime: 0,
    holdTime: 0,
    displacedTime: 0,
    histBreath: new Float32Array(HISTORY),
    histBeam: new Uint8Array(HISTORY),
    histHead: 0,
    histCount: 0,
    histAccum: 0,
  };
}

function startAngle(angles: number[]): number {
  if (!angles.length) return 0;
  return isArc(angles) ? angles[0] - sector(angles) / 2 : angles[0];
}

/** Reset delivery but keep the patient breathing (wall clock and history continue). */
export function resetDelivery(s: SimState, angles: number[]): void {
  s.status = "ready";
  s.t = 0;
  s.g = startAngle(angles);
  s.beam = 0;
  s.phase = "move";
  s.delivered = angles.map(() => 0);
  s.beamOn = false;
  s.hold = false;
  s.beamOnTime = 0;
  s.holdTime = 0;
  s.displacedTime = 0;
}

export function finishDelivery(s: SimState, angles: number[]): void {
  s.delivered = angles.map(() => 1);
  s.beam = angles.length;
  s.g = angles.length ? angles[angles.length - 1] + (isArc(angles) ? sector(angles) / 2 : 0) : 0;
  s.status = "done";
  s.beamOn = false;
  s.hold = false;
}

export const gantryDisplay = (g: number) => ((g % 360) + 360) % 360;
export const deliveredFraction = (s: SimState) => (s.delivered.length ? s.delivered.reduce((a, b) => a + b, 0) / s.delivered.length : 0);
export const targetOffsetMm = (s: SimState) => s.breath * BREATH_AMPLITUDE_MM;

function breathe(s: SimState, cfg: SimConfig, dt: number): void {
  if (!cfg.breathing) {
    s.breath = Math.max(0, s.breath - dt * 2);
    return;
  }
  // Slow, irregular drift in period and depth so the trace looks alive rather than metronomic.
  s.period = 4 + 0.35 * Math.sin(0.13 * s.wall) + 0.2 * Math.sin(0.37 * s.wall + 1);
  s.breathPhase += (Math.PI * dt) / s.period;
  const depth = 1 + 0.08 * Math.sin(0.21 * s.wall);
  s.breath = clamp01(depth * Math.cos(s.breathPhase) ** 4);
}

function record(s: SimState, dt: number): void {
  s.histAccum += dt;
  while (s.histAccum >= 1 / HISTORY_HZ) {
    s.histAccum -= 1 / HISTORY_HZ;
    s.histBreath[s.histHead] = s.breath;
    s.histBeam[s.histHead] = s.beamOn ? 1 : s.hold ? 2 : 0;
    s.histHead = (s.histHead + 1) % HISTORY;
    s.histCount = Math.min(HISTORY, s.histCount + 1);
  }
}

/** Advance by ``dt`` real seconds. */
export function step(s: SimState, cfg: SimConfig, dt: number): void {
  s.wall += dt;
  breathe(s, cfg, dt);
  const n = cfg.angles.length;
  if (s.status !== "running" || !n) {
    s.beamOn = false;
    s.hold = false;
    record(s, dt);
    return;
  }
  const d = dt * cfg.speed;
  s.t += d;
  const outsideWindow = s.breath > GATE_FRACTION;
  const held = cfg.gating && outsideWindow;
  const beamTime = () => {
    s.beamOnTime += d;
    if (outsideWindow) s.displacedTime += d;
  };
  if (isArc(cfg.angles)) {
    s.hold = held;
    s.beamOn = !held;
    if (!held) {
      beamTime();
      const w = sector(cfg.angles);
      s.g += GANTRY_DEG_PER_S * d;
      s.delivered = cfg.angles.map((a) => clamp01((s.g - (a - w / 2)) / w));
      s.beam = Math.min(n - 1, Math.max(0, Math.floor((s.g - (cfg.angles[0] - w / 2)) / w)));
      if (s.g >= cfg.angles[n - 1] + w / 2) finishDelivery(s, cfg.angles);
    }
  } else if (s.phase === "move") {
    s.beamOn = false;
    s.hold = false;
    const target = cfg.angles[s.beam];
    s.g = Math.min(target, s.g + GANTRY_DEG_PER_S * d);
    if (s.g >= target) s.phase = "deliver";
  } else {
    s.hold = held;
    s.beamOn = !held;
    if (!held) {
      beamTime();
      s.delivered[s.beam] = Math.min(1, s.delivered[s.beam] + d / (STATIC_BEAM_ON_TOTAL_S / n));
      if (s.delivered[s.beam] >= 1) {
        s.beam += 1;
        s.phase = "move";
        s.beamOn = false;
        if (s.beam >= n) finishDelivery(s, cfg.angles);
      }
    }
  }
  if (s.hold) s.holdTime += d;
  record(s, dt);
}

/** Copy of the fields the React panels show, taken a few times per second. */
export interface SimSnapshot {
  status: SimStatus;
  t: number;
  gantry: number;
  beam: number;
  nBeams: number;
  beamOn: boolean;
  hold: boolean;
  moving: boolean;
  delivered: number[];
  fraction: number;
  breath: number;
  offsetMm: number;
  rate: number;
  pulse: number;
  beamOnTime: number;
  holdTime: number;
  displacedTime: number;
}

export function snapshot(s: SimState): SimSnapshot {
  return {
    status: s.status,
    t: s.t,
    gantry: gantryDisplay(s.g),
    beam: s.beam,
    nBeams: s.delivered.length,
    beamOn: s.beamOn,
    hold: s.hold,
    moving: s.status === "running" && !s.beamOn && !s.hold,
    delivered: [...s.delivered],
    fraction: deliveredFraction(s),
    breath: s.breath,
    offsetMm: targetOffsetMm(s),
    rate: 60 / s.period,
    pulse: 70 + 4 * Math.sin(0.05 * s.wall) + 2 * Math.sin(0.23 * s.wall),
    beamOnTime: s.beamOnTime,
    holdTime: s.holdTime,
    displacedTime: s.displacedTime,
  };
}
