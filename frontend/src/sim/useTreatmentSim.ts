import { useCallback, useEffect, useRef, useState, type MutableRefObject } from "react";
import { createSim, finishDelivery, resetDelivery, snapshot, step, type SimConfig, type SimSnapshot, type SimState } from "./treatment";

const SNAPSHOT_MS = 100;

/**
 * Runs the replay clock on requestAnimationFrame. Canvas and 3-D views read ``ref``
 * every frame; React panels get a ``snap`` copy ten times a second.
 */
export function useTreatmentSim(cfg: SimConfig): {
  ref: MutableRefObject<SimState>;
  snap: SimSnapshot;
  start: () => void;
  pause: () => void;
  reset: () => void;
  skip: () => void;
} {
  const ref = useRef<SimState>(createSim(cfg.angles));
  const cfgRef = useRef(cfg);
  cfgRef.current = cfg;
  const [snap, setSnap] = useState<SimSnapshot>(() => snapshot(ref.current));
  const anglesKey = cfg.angles.join(",");

  useEffect(() => {
    resetDelivery(ref.current, cfgRef.current.angles);
    setSnap(snapshot(ref.current));
  }, [anglesKey]);

  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    let lastSnap = 0;
    const loop = (now: number) => {
      // Clamp so a backgrounded tab does not jump the gantry forward on return.
      const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
      last = now;
      step(ref.current, cfgRef.current, dt);
      if (now - lastSnap > SNAPSHOT_MS) {
        lastSnap = now;
        setSnap(snapshot(ref.current));
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  const start = useCallback(() => {
    const s = ref.current;
    if (!cfgRef.current.angles.length) return;
    if (s.status === "done") resetDelivery(s, cfgRef.current.angles);
    s.status = "running";
  }, []);
  const pause = useCallback(() => {
    if (ref.current.status === "running") ref.current.status = "paused";
  }, []);
  const reset = useCallback(() => {
    resetDelivery(ref.current, cfgRef.current.angles);
    setSnap(snapshot(ref.current));
  }, []);
  const skip = useCallback(() => {
    finishDelivery(ref.current, cfgRef.current.angles);
    setSnap(snapshot(ref.current));
  }, []);
  return { ref, snap, start, pause, reset, skip };
}
