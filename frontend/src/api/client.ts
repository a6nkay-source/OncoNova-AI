import type {
  CaseSummary,
  ComparisonResult,
  DatasetEntry,
  DoseResult,
  EvaluationReport,
  LongitudinalResult,
  ModelCard,
  ProgressionPrediction,
  ResearchSummary,
  Scenario,
  SegmentationSummary,
  SystemInfo,
  Unavailable,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public requestId?: string | null,
    public details?: unknown,
  ) {
    super(message);
  }
}

const KEY_STORAGE = "onconova.apiKey";

export function getApiKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

export function setApiKey(key: string): void {
  try {
    if (key) localStorage.setItem(KEY_STORAGE, key);
    else localStorage.removeItem(KEY_STORAGE);
  } catch {
    /* storage unavailable: key lives for this session only */
  }
}

function headers(extra?: Record<string, string>): Record<string, string> {
  const h: Record<string, string> = { ...extra };
  const key = getApiKey();
  if (key) h["X-API-Key"] = key;
  return h;
}

async function raise(res: Response): Promise<never> {
  let body: { error?: { code?: string; message?: string; details?: unknown }; request_id?: string } = {};
  try {
    body = await res.json();
  } catch {
    /* non-JSON error */
  }
  throw new ApiError(
    res.status,
    body.error?.code ?? "http_error",
    body.error?.message ?? `Request failed (${res.status})`,
    body.request_id,
    body.error?.details,
  );
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { ...init, headers: headers(init?.headers as Record<string, string>) });
  } catch {
    throw new ApiError(0, "network_error", "The OncoNova API is not reachable. Is the backend running (default port 8010)?");
  }
  if (!res.ok) await raise(res);
  return res.json() as Promise<T>;
}

const post = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body), headers: { "Content-Type": "application/json" } });

export function isUnavailable(x: unknown): x is Unavailable {
  return typeof x === "object" && x !== null && (x as { status?: string }).status === "unavailable";
}

// ----- binary volumes --------------------------------------------------------

export interface VolumeData {
  data: Int16Array | Uint8Array;
  shape: [number, number, number]; // z, y, x
  spacing: [number, number, number]; // x, y, z (mm)
  scale: number;
  offset: number;
  label?: string | null;
  /** Extra response headers some endpoints add (e.g. X-Beam-Angles). */
  meta?: Record<string, string>;
}

export async function fetchVolume(path: string, signal?: AbortSignal): Promise<VolumeData> {
  let res: Response;
  try {
    res = await fetch(path, { headers: headers(), signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "network_error", "The OncoNova API is not reachable.");
  }
  if (!res.ok) await raise(res);
  const buf = await res.arrayBuffer();
  const shape = res.headers.get("X-Shape")!.split(",").map(Number) as [number, number, number];
  const spacing = res.headers.get("X-Spacing")!.split(",").map(Number) as [number, number, number];
  const dtype = res.headers.get("X-Dtype");
  const data = dtype === "int16" ? new Int16Array(buf) : new Uint8Array(buf);
  if (data.length !== shape[0] * shape[1] * shape[2]) {
    throw new ApiError(0, "bad_volume", "Volume payload does not match its declared shape.");
  }
  return {
    data,
    shape,
    spacing,
    scale: Number(res.headers.get("X-Scale") ?? 1),
    offset: Number(res.headers.get("X-Offset") ?? 0),
    label: res.headers.get("X-Data-Label"),
    meta: Object.fromEntries(
      ["X-Beam-Angles", "X-Slice-Index"].flatMap((h) => (res.headers.get(h) !== null ? [[h, res.headers.get(h)!]] : [])),
    ),
  };
}

const tp = (caseId: string, tpId: string) => `/api/cases/${encodeURIComponent(caseId)}/timepoints/${encodeURIComponent(tpId)}`;

export const api = {
  system: () => request<SystemInfo>("/api/system"),
  cases: () => request<CaseSummary[]>("/api/cases"),
  case: (id: string) => request<CaseSummary & Record<string, unknown>>(`/api/cases/${encodeURIComponent(id)}`),
  deleteCase: (id: string) => request<{ deleted: string }>(`/api/cases/${encodeURIComponent(id)}`, { method: "DELETE" }),
  metadata: (c: string, t: string) => request<Record<string, unknown>>(`${tp(c, t)}/metadata`),
  imageUrl: (c: string, t: string) => `${tp(c, t)}/image`,
  mapUrl: (c: string, t: string, kind: string, q: Record<string, string> = {}) => {
    const qs = new URLSearchParams(q).toString();
    return `${tp(c, t)}/maps/${kind}${qs ? `?${qs}` : ""}`;
  },
  registeredUrl: (c: string, t: string, kind: "image" | "lesions", source = "reference") =>
    `${tp(c, t)}/registered?kind=${kind}&source=${encodeURIComponent(source)}`,
  mesh: (c: string, t: string, source: string, label: number) =>
    request<{ vertices: number[]; faces: number[]; n_faces: number; empty?: boolean }>(
      `${tp(c, t)}/mesh?source=${encodeURIComponent(source)}&label=${label}`,
    ),
  models: () => request<{ segmentation: ModelCard[]; progression: Record<string, unknown> }>("/api/models"),
  segment: (c: string, t: string, model_id: string, mc_samples = 8) =>
    post<SegmentationSummary>(`${tp(c, t)}/segmentation`, { model_id, mc_samples }),
  segmentation: (c: string, t: string, model_id: string) =>
    request<SegmentationSummary | Unavailable>(`${tp(c, t)}/segmentation?model_id=${model_id}`),
  compare: (c: string, t: string, model_id: string, reference: "reference" | "annotation") =>
    post<ComparisonResult | Unavailable>(`${tp(c, t)}/comparison`, { model_id, reference }),
  saveAnnotation: async (c: string, t: string, data: Uint8Array) => {
    let res: Response;
    try {
      res = await fetch(`${tp(c, t)}/annotation`, {
        method: "PUT",
        body: data,
        headers: headers({ "Content-Type": "application/octet-stream" }),
      });
    } catch {
      throw new ApiError(0, "network_error", "The OncoNova API is not reachable.");
    }
    if (!res.ok) await raise(res);
    return (await res.json()) as { saved: boolean; voxels: number; volume_cc: number };
  },
  longitudinal: (c: string, source: string) => post<LongitudinalResult | Unavailable>(`/api/cases/${c}/longitudinal`, { source }),
  getLongitudinal: (c: string, source: string) =>
    request<LongitudinalResult | Unavailable>(`/api/cases/${c}/longitudinal?source=${encodeURIComponent(source)}`),
  progression: (c: string, source: string) => post<ProgressionPrediction | Unavailable>(`/api/cases/${c}/progression`, { source }),
  progressionModel: () => request<Record<string, unknown>>("/api/progression/model"),
  scenarios: () => request<{ scenarios: Scenario[] }>("/api/dose/scenarios"),
  dose: (c: string, scenario_id: string, target_source: string) =>
    post<DoseResult | Unavailable>(`/api/cases/${c}/dose`, { scenario_id, target_source }),
  getDose: (c: string, scenario_id: string, target_source: string) =>
    request<DoseResult | Unavailable>(
      `/api/cases/${c}/dose?scenario_id=${scenario_id}&target_source=${encodeURIComponent(target_source)}`,
    ),
  doseBeamsUrl: (c: string, scenario_id: string, target_source: string) =>
    `/api/cases/${encodeURIComponent(c)}/dose/beams?scenario_id=${scenario_id}&target_source=${encodeURIComponent(target_source)}`,
  importDose: async (c: string, file: File, targetSource: string) => {
    const qs = new URLSearchParams({ filename: file.name, target_source: targetSource });
    let res: Response;
    try {
      res = await fetch(`/api/cases/${encodeURIComponent(c)}/dose/import?${qs}`, {
        method: "POST",
        body: file,
        headers: headers({ "Content-Type": "application/octet-stream" }),
      });
    } catch {
      throw new ApiError(0, "network_error", "The OncoNova API is not reachable.");
    }
    if (!res.ok) await raise(res);
    return (await res.json()) as DoseResult & { import: Record<string, unknown>; dose_max: number };
  },
  summary: (c: string) => request<ResearchSummary>(`/api/cases/${c}/summary`),
  evaluations: () => request<{ reports: { report_id: string; task: string; model: { id: string }; created_utc: string }[] }>("/api/evaluations"),
  evaluation: (id: string) => request<EvaluationReport>(`/api/evaluations/${encodeURIComponent(id)}`),
  datasets: () =>
    request<{ catalogue: DatasetEntry[]; catalogue_checked: string; local_manifests: Record<string, unknown>[] }>("/api/datasets"),
  audit: () => request<{ events: Record<string, unknown>[] }>("/api/audit"),
  deidProfile: () => request<{ source: string; profile: Record<string, unknown> }>("/api/settings/deid-profile"),
  upload: async (file: File, modality?: string) => {
    const qs = new URLSearchParams({ filename: file.name });
    if (modality) qs.set("modality", modality);
    let res: Response;
    try {
      res = await fetch(`/api/uploads?${qs}`, {
        method: "POST",
        body: file,
        headers: headers({ "Content-Type": "application/octet-stream" }),
      });
    } catch {
      throw new ApiError(0, "network_error", "The OncoNova API is not reachable.");
    }
    if (!res.ok) await raise(res);
    return (await res.json()) as CaseSummary & { warnings: string[] };
  },
};
