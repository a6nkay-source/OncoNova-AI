import { ExternalLink, Star, Trash2 } from "lucide-react";
import { useState } from "react";
import { api } from "../api/client";
import type { DatasetEntry } from "../api/types";
import { DemoBadge, ErrorState, Note, Panel, Spinner, cx } from "../components/ui";
import { useAsync } from "../state/hooks";
import { useApp } from "../state/AppState";

export function DatasetsPage() {
  const ds = useAsync(() => api.datasets(), []);
  const { cases, refresh } = useApp();
  const [error, setError] = useState<unknown>(null);
  const remove = async (id: string) => {
    if (!window.confirm(`Permanently delete ${id} and all of its derived results? This cannot be undone.`)) return;
    try {
      await api.deleteCase(id);
      refresh();
    } catch (e) {
      setError(e);
    }
  };
  return (
    <div className="space-y-4">
      <Panel title="Local datasets and split manifests">
        {ds.loading && !ds.data && <Spinner label="Loading" />}
        {ds.error ? <ErrorState error={ds.error} /> : null}
        <div className="grid gap-3 lg:grid-cols-2">
          {ds.data?.local_manifests.map((m) => (
            <div key={String(m.manifest_id)} className="rounded-lg border border-ink-700/60 bg-ink-925/60 p-3 text-[12.5px]">
              <div className="flex items-center gap-2">
                <span className="mono text-ink-100">{String(m.manifest_id)}</span>
                <DemoBadge label={String(m.label)} />
              </div>
              <p className="mt-1 text-ink-400">{String(m.description)}</p>
              <div className="kv mt-2">
                <span className="text-ink-400">Patients</span>
                <span className="mono text-right">{String(m.n_patients)}</span>
                <span className="text-ink-400">Split unit</span>
                <span className="text-right">{String(m.split_unit)}</span>
                {m.patient_counts_by_split ? (
                  <>
                    <span className="text-ink-400">Patients train / val / test</span>
                    <span className="mono text-right">
                      {Object.values(m.patient_counts_by_split as Record<string, number>).join(" / ")}
                    </span>
                    <span className="text-ink-400">Scans by split</span>
                    <span className="mono text-right">{JSON.stringify(m.scan_counts_by_split)}</span>
                    <span className="text-ink-400">Seeds (data / split)</span>
                    <span className="mono text-right">
                      {String(m.dataset_seed)} / {String(m.split_seed)}
                    </span>
                    <span className="text-ink-400">Quality errors</span>
                    <span className="mono text-right">{String(m.quality_issues)}</span>
                  </>
                ) : null}
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Cases on this server">
        {error ? <ErrorState error={error} /> : null}
        <table className="table-base">
          <thead>
            <tr>
              <th>Case</th>
              <th>Label</th>
              <th>Modality</th>
              <th>Scans</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {cases.map((c) => (
              <tr key={c.case_id}>
                <td>
                  <span className="mono">{c.case_id}</span> <span className="text-ink-400">{c.title}</span>
                </td>
                <td className={c.demo ? "text-ev-uncertain" : "text-accent"}>{c.label}</td>
                <td>{c.modality}</td>
                <td>{c.timepoints.length}</td>
                <td className="text-right">
                  {!c.demo && (
                    <button className="btn-danger py-1" onClick={() => remove(c.case_id)}>
                      <Trash2 className="h-3.5 w-3.5" /> Delete
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <Panel title={`Public dataset catalogue · checked ${ds.data?.catalogue_checked ?? ""}`}>
        <div className="mb-3">
          <Note>
            Fields come from the official collection pages and dataset papers. Anything that could not be confirmed says "not stated". Licences and
            access terms change: check them again before downloading. No dataset is downloaded automatically.
          </Note>
        </div>
        <div className="grid gap-3 xl:grid-cols-2">
          {ds.data?.catalogue.map((d) => (
            <DatasetCard key={d.id} d={d} />
          ))}
        </div>
      </Panel>
    </div>
  );
}

function DatasetCard({ d }: { d: DatasetEntry }) {
  const [open, setOpen] = useState(!!d.recommended_starter);
  const controlled = /controlled/i.test(d.access);
  return (
    <div className={cx("rounded-lg border bg-ink-925/60 p-3", d.recommended_starter ? "border-teal/50" : "border-ink-700/60")}>
      <button className="flex w-full items-start gap-2 text-left" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[13.5px] font-semibold text-ink-100">{d.name}</span>
            {d.recommended_starter && (
              <span className="inline-flex items-center gap-1 rounded-md bg-teal/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-teal">
                <Star className="h-3 w-3" /> Recommended starter
              </span>
            )}
            {controlled && <span className="rounded-md bg-ev-review/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-ev-review">Controlled access</span>}
          </div>
          <div className="mt-0.5 text-[12px] text-ink-400">
            {d.cancer_type} · {d.modality} · {d.patients} patients · {d.size}
          </div>
        </div>
      </button>
      {open && (
        <dl className="mt-3 grid grid-cols-[120px_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-[12px]">
          {(
            [
              ["Region", d.body_region],
              ["Format", d.format],
              ["Images", d.images],
              ["Annotations", d.annotations],
              ["Ground truth", d.ground_truth],
              ["Outcomes", d.outcomes],
              ["Licence", d.license],
              ["Access", d.access],
              ["Paper", d.paper],
              ["Dataset DOI", d.dataset_doi],
              ["Suitable for", d.suitable_for?.join(", ")],
              ["Notes", d.notes],
            ] as [string, string | undefined][]
          )
            .filter(([, v]) => v)
            .map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-ink-500">{k}</dt>
                <dd className={cx("text-ink-200", k === "Ground truth" && "text-ev-uncertain")}>{v}</dd>
              </div>
            ))}
          <dt className="text-ink-500">Download</dt>
          <dd>
            <a href={d.download_url.split(" ")[0]} target="_blank" rel="noreferrer noopener" className="inline-flex items-center gap-1 break-all text-accent hover:underline">
              {d.download_url} <ExternalLink className="h-3 w-3" />
            </a>
          </dd>
        </dl>
      )}
    </div>
  );
}
