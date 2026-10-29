import { useCallback, useEffect, useRef, useState } from "react";
import { fetchVolume, type VolumeData } from "../api/client";

export interface AsyncState<T> {
  data: T | null;
  error: unknown;
  loading: boolean;
  reload: () => void;
  setData: (d: T | null) => void;
}

/** Run ``fn`` whenever ``deps`` change. Stale responses are ignored. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[], enabled = true): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [tick, setTick] = useState(0);
  const seq = useRef(0);
  useEffect(() => {
    if (!enabled) return;
    const my = ++seq.current;
    setLoading(true);
    setError(null);
    fn()
      .then((d) => my === seq.current && setData(d))
      .catch((e) => my === seq.current && setError(e))
      .finally(() => my === seq.current && setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick, enabled]);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload, setData };
}

const cache = new Map<string, Promise<VolumeData>>();

export function invalidateVolumes(prefix: string): void {
  for (const key of [...cache.keys()]) if (key.startsWith(prefix)) cache.delete(key);
}

/** Load several volumes; entries are null until loaded (or if they fail, see ``errors``). */
export function useVolumes(urls: (string | null)[], version = 0): {
  volumes: (VolumeData | null)[];
  errors: (unknown | null)[];
} {
  const key = urls.join("|");
  const [state, setState] = useState<{ key: string; volumes: (VolumeData | null)[]; errors: (unknown | null)[] }>({
    key: "",
    volumes: [],
    errors: [],
  });
  useEffect(() => {
    let live = true;
    const vols: (VolumeData | null)[] = urls.map(() => null);
    const errs: (unknown | null)[] = urls.map(() => null);
    setState({ key, volumes: [...vols], errors: [...errs] });
    urls.forEach((url, i) => {
      if (!url) return;
      const k = `${url}#${version}`;
      if (!cache.has(k)) {
        const p = fetchVolume(url);
        cache.set(k, p);
        p.catch(() => cache.delete(k));
      }
      cache
        .get(k)!
        .then((v) => {
          if (!live) return;
          vols[i] = v;
          setState({ key, volumes: [...vols], errors: [...errs] });
        })
        .catch((e) => {
          if (!live) return;
          errs[i] = e;
          setState({ key, volumes: [...vols], errors: [...errs] });
        });
    });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, version]);
  if (state.key !== key) return { volumes: urls.map(() => null), errors: urls.map(() => null) };
  return { volumes: state.volumes, errors: state.errors };
}

/**
 * Load a binary volume, sharing one request per URL across components. ``null`` url = nothing to load.
 * While a new URL loads, ``volume`` is null rather than the previous scan, so nothing loaded
 * for the new scan (overlays, positions) is ever drawn on the old one.
 */
export function useVolume(url: string | null, version = 0): { volume: VolumeData | null; error: unknown; loading: boolean } {
  const key = url ? `${url}#${version}` : "";
  const [state, setState] = useState<{ key: string; volume: VolumeData | null; error: unknown }>({ key: "", volume: null, error: null });
  useEffect(() => {
    if (!url) return;
    let live = true;
    if (!cache.has(key)) {
      const p = fetchVolume(url);
      cache.set(key, p);
      p.catch(() => cache.delete(key));
    }
    cache
      .get(key)!
      .then((volume) => live && setState({ key, volume, error: null }))
      .catch((error) => live && setState({ key, volume: null, error }));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  if (!url) return { volume: null, error: null, loading: false };
  if (state.key !== key) return { volume: null, error: null, loading: true };
  return { volume: state.volume, error: state.error, loading: false };
}
