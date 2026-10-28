import { describe, expect, it } from "vitest";
import { GATE_FRACTION, createSim, deliveredFraction, finishDelivery, gantryDisplay, step, type SimConfig } from "../sim/treatment";

const run = (cfg: SimConfig, seconds: number, dt = 0.05) => {
  const s = createSim(cfg.angles);
  s.status = "running";
  for (let t = 0; t < seconds && s.status === "running"; t += dt) step(s, cfg, dt);
  return s;
};

describe("treatment replay clock (simulated)", () => {
  it("replays every beam of a static plan exactly once, in order", () => {
    const angles = [0, 120, 240];
    const s = run({ angles, speed: 4, breathing: false, gating: false }, 120);
    expect(s.status).toBe("done");
    expect(s.delivered).toEqual([1, 1, 1]);
    expect(deliveredFraction(s)).toBe(1);
    expect(s.holdTime).toBe(0);
  });

  it("keeps the beam off while the gantry moves between beams", () => {
    const angles = [0, 180];
    const s = createSim(angles);
    s.status = "running";
    const cfg = { angles, speed: 1, breathing: false, gating: false };
    let sawMoveWithBeamOff = false;
    for (let t = 0; t < 60 && s.status === "running"; t += 0.05) {
      step(s, cfg, 0.05);
      if (s.phase === "move" && s.beam === 1) sawMoveWithBeamOff ||= !s.beamOn;
      if (s.phase === "move") expect(s.beamOn).toBe(false);
    }
    expect(sawMoveWithBeamOff).toBe(true);
  });

  it("holds the beam outside the exhale window when gating is on", () => {
    const angles = [0, 90, 180, 270];
    const cfg = { angles, speed: 1, breathing: true, gating: true };
    const s = createSim(angles);
    s.status = "running";
    for (let t = 0; t < 200 && s.status === "running"; t += 0.02) {
      step(s, cfg, 0.02);
      if (s.beamOn) expect(s.breath).toBeLessThanOrEqual(GATE_FRACTION);
    }
    expect(s.status).toBe("done");
    expect(s.holdTime).toBeGreaterThan(0);
    expect(s.displacedTime).toBe(0);
  });

  it("without gating, some beam-on time happens with the target displaced", () => {
    const s = run({ angles: [0, 90, 180, 270], speed: 1, breathing: true, gating: false }, 200, 0.02);
    expect(s.status).toBe("done");
    expect(s.displacedTime).toBeGreaterThan(0);
    expect(s.holdTime).toBe(0);
  });

  it("replays a many-angle plan as one continuous arc with monotone delivery", () => {
    const angles = Array.from({ length: 36 }, (_, i) => i * 10);
    const cfg = { angles, speed: 4, breathing: false, gating: false };
    const s = createSim(angles);
    s.status = "running";
    let prev = 0;
    while (s.status === "running") {
      step(s, cfg, 0.05);
      const f = deliveredFraction(s);
      expect(f).toBeGreaterThanOrEqual(prev - 1e-9);
      prev = f;
    }
    expect(prev).toBe(1);
  });

  it("does nothing until started, and skip-to-end completes delivery", () => {
    const angles = [0, 120, 240];
    const s = createSim(angles);
    step(s, { angles, speed: 1, breathing: true, gating: false }, 1);
    expect(s.beamOn).toBe(false);
    expect(deliveredFraction(s)).toBe(0);
    finishDelivery(s, angles);
    expect(s.status).toBe("done");
    expect(deliveredFraction(s)).toBe(1);
  });

  it("wraps gantry angles into 0-360 for display", () => {
    expect(gantryDisplay(-5)).toBe(355);
    expect(gantryDisplay(365)).toBe(5);
  });
});
