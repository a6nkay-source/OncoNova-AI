import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { CaseSummary, ModelCard, SystemInfo } from "../api/types";

interface AppState {
  system: SystemInfo | null;
  systemError: unknown;
  cases: CaseSummary[];
  casesError: unknown;
  caseId: string | null;
  tpId: string | null;
  current: CaseSummary | null;
  models: ModelCard[];
  modelId: string;
  /** Bumped after any analysis so views refetch overlays and results. */
  revision: number;
  setCase: (id: string) => void;
  setTp: (id: string) => void;
  setModelId: (id: string) => void;
  bump: () => void;
  refresh: () => void;
}

const Ctx = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [systemError, setSystemError] = useState<unknown>(null);
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [casesError, setCasesError] = useState<unknown>(null);
  const [caseId, setCaseId] = useState<string | null>(null);
  const [tpId, setTpId] = useState<string | null>(null);
  const [models, setModels] = useState<ModelCard[]>([]);
  const [modelId, setModelId] = useState("demo-classical");
  const [revision, setRevision] = useState(0);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    api.system().then(setSystem, setSystemError);
    api.cases().then(
      (cs) => {
        setCases(cs);
        setCasesError(null);
        setCaseId((cur) => (cur && cs.some((c) => c.case_id === cur) ? cur : (cs[0]?.case_id ?? null)));
      },
      setCasesError,
    );
    api.models().then(
      (m) => {
        setModels(m.segmentation);
        const unet = m.segmentation.find((x) => x.id === "demo-unet" && x.status === "available");
        if (unet) setModelId((cur) => (cur === "demo-classical" ? "demo-unet" : cur));
      },
      () => setModels([]),
    );
  }, [tick]);

  const current = useMemo(() => cases.find((c) => c.case_id === caseId) ?? null, [cases, caseId]);
  useEffect(() => {
    if (current && !current.timepoints.some((t) => t.tp_id === tpId)) setTpId(current.timepoints[0]?.tp_id ?? null);
  }, [current, tpId]);

  const bump = useCallback(() => setRevision((r) => r + 1), []);
  const refresh = useCallback(() => setTick((t) => t + 1), []);
  const value: AppState = {
    system,
    systemError,
    cases,
    casesError,
    caseId,
    tpId,
    current,
    models,
    modelId,
    revision,
    setCase: setCaseId,
    setTp: setTpId,
    setModelId,
    bump,
    refresh,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp outside AppStateProvider");
  return v;
}
