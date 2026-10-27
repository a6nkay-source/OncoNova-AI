import { Eye, FileUp, Lock } from "lucide-react";
import { useState } from "react";
import { api } from "../api/client";
import { EvidenceBadge, ErrorState, Note, Panel, Segmented, Spinner } from "../components/ui";
import { useApp } from "../state/AppState";
import { CaseViewer, type OverlaySpec } from "../viewer/CaseViewer";
import { STRUCTURE_COLORS } from "../viewer/volume";

export function ImagingPage() {
  const { current, caseId, tpId } = useApp();
  const [showRef, setShowRef] = useState(true);
  const [showStruct, setShowStruct] = useState(false);
  const [opacity, setOpacity] = useState(0.5);
  if (!current || !caseId || !tpId) return <Spinner label="Loading cases" />;

  const overlays: OverlaySpec[] = [
    {
      id: "structures",
      url: current.has_structures ? api.mapUrl(caseId, tpId, "structures") : null,
      type: "label",
      opacity: opacity * 0.6,
      colors: STRUCTURE_COLORS,
      visible: showStruct && current.has_structures,
    },
    {
      id: "reference",
      url: current.has_reference ? api.mapUrl(caseId, tpId, "reference") : null,
      type: "label",
      opacity,
      outline: true,
      visible: showRef && current.has_reference,
    },
  ];

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
      <Panel
        title="Medical image viewer"
        className="min-h-[640px]"
        bodyClassName="flex flex-col"
        actions={<span className="mono text-ink-500">{current.modality} · {tpId}</span>}
      >
        <CaseViewer
          caseId={caseId}
          tpId={tpId}
          modality={current.modality}
          overlays={overlays}
          toolbar={
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-1.5 text-[12px] text-ink-300">
                <input type="checkbox" checked={showRef} disabled={!current.has_reference} onChange={(e) => setShowRef(e.target.checked)} />
                Reference contour
              </label>
              <label className="flex items-center gap-1.5 text-[12px] text-ink-300">
                <input type="checkbox" checked={showStruct} disabled={!current.has_structures} onChange={(e) => setShowStruct(e.target.checked)} />
                Anatomy
              </label>
              <label className="flex w-32 items-center gap-2 text-[11px] text-ink-400">
                Opacity
                <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(+e.target.value)} aria-label="Overlay opacity" />
              </label>
            </div>
          }
        />
        <div className="mt-3">
          <Note>
            Scroll a view to move through slices; click to move the crosshair. Display is radiological (patient right on screen left).
            Volumes keep their native spacing and orientation; aspect ratio follows physical millimetres.
          </Note>
        </div>
      </Panel>
      <div className="flex flex-col gap-4">
        <CaseInfo />
        <MetadataPanel />
        <UploadPanel />
      </div>
    </div>
  );
}

function CaseInfo() {
  const { current, tpId } = useApp();
  if (!current) return null;
  return (
    <Panel title="Case">
      <div className="space-y-2 text-[13px]">
        <div className="font-medium text-ink-100">{current.title}</div>
        <div className="kv">
          <span className="text-ink-400">Data label</span>
          <span className="text-right font-semibold text-ev-uncertain">{current.label}</span>
          <span className="text-ink-400">Cancer type</span>
          <span className="text-right text-ink-200">{current.cancer_type}</span>
          <span className="text-ink-400">Region</span>
          <span className="text-right text-ink-200">{current.body_region}</span>
          <span className="text-ink-400">Timepoints</span>
          <span className="text-right text-ink-200">{current.timepoints.length}</span>
          <span className="text-ink-400">Reference mask</span>
          <span className="text-right text-ink-200">{current.has_reference ? "synthetic ground truth" : "none"}</span>
        </div>
        <div className="flex flex-wrap gap-1.5 pt-1">
          {current.timepoints.map((t) => (
            <span key={t.tp_id} className={`rounded-md border px-2 py-0.5 text-[11px] ${t.tp_id === tpId ? "border-accent/50 text-accent" : "border-ink-700 text-ink-400"}`}>
              {t.label} · d{t.days_from_baseline}
            </span>
          ))}
        </div>
      </div>
    </Panel>
  );
}

function MetadataPanel() {
  const { caseId, tpId } = useApp();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const load = () => {
    if (!caseId || !tpId) return;
    setLoading(true);
    setError(null);
    api
      .metadata(caseId, tpId)
      .then(setData, setError)
      .finally(() => setLoading(false));
  };
  const geom = data?.geometry as Record<string, unknown> | undefined;
  return (
    <Panel
      title="Metadata"
      actions={
        <button className="btn-ghost" onClick={load} disabled={loading}>
          <Eye className="h-3.5 w-3.5" /> Inspect
        </button>
      }
    >
      {loading && <Spinner label="Loading" />}
      {error ? <ErrorState error={error} /> : null}
      {!data && !loading && !error && (
        <Note>
          <Lock className="mr-1 inline h-3 w-3" />
          Needs the researcher role; each view is written to the audit log. Identifying DICOM fields are never shown, only listed by name.
        </Note>
      )}
      {data && geom && (
        <div className="space-y-3">
          <div className="kv">
            <span className="text-ink-400">Shape (z,y,x)</span>
            <span className="mono text-right">{(geom.shape_zyx as number[]).join(" × ")}</span>
            <span className="text-ink-400">Spacing (x,y,z)</span>
            <span className="mono text-right">{(geom.spacing_xyz_mm as number[]).map((v) => v.toFixed(2)).join(" / ")} mm</span>
            <span className="text-ink-400">Origin</span>
            <span className="mono text-right">{(geom.origin_xyz_mm as number[]).map((v) => v.toFixed(1)).join(", ")}</span>
            <span className="text-ink-400">Frame</span>
            <span className="mono text-right">{String(geom.coordinate_system)}{geom.oblique ? " · oblique" : ""}</span>
            <span className="text-ink-400">Intensity range</span>
            <span className="mono text-right">{(data.intensity_range as number[]).map((v) => Math.round(v)).join(" to ")}</span>
            <span className="text-ink-400">Source</span>
            <span className="text-right">{String(data.source_format ?? "-")}</span>
          </div>
          {Array.isArray(data.warnings) && data.warnings.length > 0 && (
            <ul className="space-y-1">
              {(data.warnings as string[]).map((w) => (
                <li key={w} className="flex gap-2 text-[12px] text-ink-300">
                  <EvidenceBadge kind="uncertain" /> {w}
                </li>
              ))}
            </ul>
          )}
          {data.deidentification ? (
            <div className="rounded-md bg-ink-925 p-2 text-[12px] text-ink-300">
              <div className="label mb-1">De-identification</div>
              PHI fields present in source:{" "}
              <span className="mono">{((data.deidentification as Record<string, string[]>).phi_keywords_present_in_source ?? []).join(", ") || "none"}</span>
            </div>
          ) : null}
          {data.ground_truth ? (
            <div className="rounded-md bg-ink-925 p-2 text-[12px] text-ink-300">
              <div className="label mb-1">Synthetic generator parameters</div>
              <pre className="mono whitespace-pre-wrap text-ink-400">{JSON.stringify(data.ground_truth, null, 1)}</pre>
            </div>
          ) : null}
        </div>
      )}
    </Panel>
  );
}

function UploadPanel() {
  const { refresh, setCase, system } = useApp();
  const [file, setFile] = useState<File | null>(null);
  const [modality, setModality] = useState<"CT" | "MR" | "PT">("CT");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [result, setResult] = useState<{ case_id: string; warnings: string[] } | null>(null);
  const isNifti = !!file && /\.nii(\.gz)?$/i.test(file.name);
  const submit = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.upload(file, isNifti ? modality : undefined);
      setResult(r);
      refresh();
      setCase(r.case_id);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Panel title="Import research image">
      <div className="space-y-3">
        <input
          type="file"
          accept=".nii,.gz,.zip"
          className="block w-full text-[12px] text-ink-300 file:mr-3 file:rounded-md file:border-0 file:bg-ink-750 file:px-3 file:py-1.5 file:text-[12px] file:text-ink-100"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          aria-label="Image file"
        />
        {isNifti && (
          <div className="flex items-center gap-2">
            <span className="label">Modality</span>
            <Segmented value={modality} onChange={setModality} size="xs" options={[{ value: "CT", label: "CT" }, { value: "MR", label: "MRI" }, { value: "PT", label: "PET" }]} />
          </div>
        )}
        <button className="btn-primary w-full" disabled={!file || busy} onClick={submit}>
          <FileUp className="h-4 w-4" /> {busy ? "Validating and importing..." : "Upload"}
        </button>
        {error ? <ErrorState error={error} /> : null}
        {result && (
          <div className="rounded-md border border-teal/30 bg-teal/5 p-2 text-[12px] text-ink-200">
            Imported as <span className="mono text-teal">{result.case_id}</span>.
            {result.warnings.map((w) => (
              <div key={w} className="mt-1 text-ink-400">· {w}</div>
            ))}
          </div>
        )}
        <Note>
          Accepts .nii / .nii.gz or a .zip of one DICOM series (CT, MR, PET), up to {system?.limits.max_upload_mb ?? 512} MB.
          DICOM is de-identified before parsing and only the converted volume is kept. Use only data you are authorised to process.
        </Note>
      </div>
    </Panel>
  );
}
