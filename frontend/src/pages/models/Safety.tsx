import { KeyRound, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { api, getApiKey, setApiKey } from "../../api/client";
import { ErrorState, Note, Panel, Spinner } from "../../components/ui";
import { useAsync } from "../../state/hooks";
import { useApp } from "../../state/AppState";

export function SafetySection() {
  const { system, refresh } = useApp();
  const [key, setKey] = useState(getApiKey());
  const [auditOpen, setAuditOpen] = useState(false);
  const audit = useAsync(() => api.audit(), [auditOpen], auditOpen);
  const deid = useAsync(() => api.deidProfile(), []);
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Panel title="Access">
        <div className="space-y-3">
          <div className="kv">
            <span className="text-ink-400">Auth mode</span>
            <span className="mono text-right">{system?.auth_mode}</span>
            <span className="text-ink-400">Signed in as</span>
            <span className="mono text-right">
              {system?.principal.user} ({system?.principal.role})
            </span>
          </div>
          <label className="block">
            <span className="label">API key (api_key mode)</span>
            <div className="mt-1 flex gap-2">
              <input className="input flex-1" type="password" autoComplete="off" value={key} onChange={(e) => setKey(e.target.value)} placeholder="Not needed in dev mode" />
              <button
                className="btn-primary"
                onClick={() => {
                  setApiKey(key);
                  refresh();
                }}
              >
                <KeyRound className="h-3.5 w-3.5" /> Save
              </button>
            </div>
          </label>
          <Note>The key is kept in this browser's local storage and sent only to this OncoNova server. Dev mode accepts localhost requests only.</Note>
        </div>
      </Panel>

      <Panel title="Safety and data handling">
        {system ? (
          <ul className="space-y-2 text-[13px] text-ink-300">
            {[system.disclaimer, `Device control: ${system.device_control}.`, `Prescriptions: ${system.prescription}.`, `External transmission: ${system.external_transmission}.`, `Uploads limited to ${system.limits.max_upload_mb} MB and ${system.limits.max_zip_members} files per ZIP.`].map((t) => (
              <li key={t} className="flex gap-2">
                <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-teal" />
                {t}
              </li>
            ))}
          </ul>
        ) : (
          <Spinner />
        )}
      </Panel>

      <Panel title="DICOM de-identification profile">
        {deid.error ? <ErrorState error={deid.error} /> : null}
        {deid.data && (
          <div className="space-y-2 text-[12px]">
            <div className="text-ink-400">Source: {deid.data.source}</div>
            {Object.entries(deid.data.profile).map(([action, fields]) => (
              <div key={action}>
                <span className="label">{action.replace(/_/g, " ")}</span>
                <div className="mono mt-0.5 text-ink-300">{Array.isArray(fields) ? fields.join(", ") : String(fields)}</div>
              </div>
            ))}
            <Note>Edit configs/deid_profile.yaml to change it. This is a safeguard, not a certified de-identification tool; check pixel data for burned-in text.</Note>
          </div>
        )}
      </Panel>

      <Panel
        title="Audit log"
        actions={
          <button className="btn-ghost" onClick={() => (auditOpen ? audit.reload() : setAuditOpen(true))}>
            {auditOpen ? "Refresh" : "Load (admin)"}
          </button>
        }
      >
        {!auditOpen && <Note>Uploads, analyses, metadata views, deletions and denied requests are recorded with user, role and request ID.</Note>}
        {audit.loading && <Spinner label="Loading" />}
        {audit.error ? <ErrorState error={audit.error} /> : null}
        {audit.data && (
          <div className="max-h-[360px] overflow-y-auto">
            <table className="table-base">
              <thead>
                <tr>
                  <th>Time (UTC)</th>
                  <th>User</th>
                  <th>Action</th>
                  <th>Resource</th>
                  <th>Outcome</th>
                </tr>
              </thead>
              <tbody>
                {audit.data.events.map((e, i) => (
                  <tr key={i}>
                    <td className="mono">{String(e.ts).slice(0, 19).replace("T", " ")}</td>
                    <td>{String(e.user)}</td>
                    <td>{String(e.action)}</td>
                    <td className="mono">{String(e.resource)}</td>
                    <td>{String(e.outcome)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
