import {
  Activity,
  BarChart3,
  Database,
  LayoutDashboard,
  PanelLeftClose,
  PanelLeftOpen,
  Radiation,
  ScanLine,
  Settings as SettingsIcon,
  ShieldAlert,
  Target,
} from "lucide-react";
import { useEffect, useState, type ComponentType } from "react";
import { DemoBadge, ErrorState, cx } from "./components/ui";
import { DatasetsPage } from "./pages/Datasets";
import { EvaluationPage } from "./pages/Evaluation";
import { ImagingPage } from "./pages/Imaging";
import { LinacPage } from "./pages/Linac";
import { OverviewPage } from "./pages/Overview";
import { ProgressionPage } from "./pages/Progression";
import { SegmentationPage } from "./pages/Segmentation";
import { SettingsPage } from "./pages/Settings";
import { useApp } from "./state/AppState";

const NAV: { id: string; label: string; icon: ComponentType<{ className?: string }>; page: ComponentType }[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard, page: OverviewPage },
  { id: "imaging", label: "Medical Imaging", icon: ScanLine, page: ImagingPage },
  { id: "segmentation", label: "Tumor Segmentation", icon: Target, page: SegmentationPage },
  { id: "progression", label: "Progression Analysis", icon: Activity, page: ProgressionPage },
  { id: "linac", label: "LINAC Research Simulator", icon: Radiation, page: LinacPage },
  { id: "evaluation", label: "Model Evaluation", icon: BarChart3, page: EvaluationPage },
  { id: "datasets", label: "Dataset Management", icon: Database, page: DatasetsPage },
  { id: "settings", label: "Settings", icon: SettingsIcon, page: SettingsPage },
];

function useHashRoute(): [string, (id: string) => void] {
  const read = () => window.location.hash.replace(/^#\/?/, "") || "overview";
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    window.addEventListener("popstate", on);
    return () => {
      window.removeEventListener("hashchange", on);
      window.removeEventListener("popstate", on);
    };
  }, []);
  return [route, (id: string) => (window.location.hash = `/${id}`)];
}

function Logo({ collapsed }: { collapsed: boolean }) {
  return (
    <div className="flex items-center gap-2.5 px-1">
      <svg viewBox="0 0 32 32" className="h-7 w-7 shrink-0" aria-hidden>
        <defs>
          <linearGradient id="lg" x1="0" x2="1" y1="0" y2="1">
            <stop offset="0" stopColor="#2fd0b8" />
            <stop offset="1" stopColor="#4c9bff" />
          </linearGradient>
        </defs>
        <circle cx="16" cy="16" r="13" fill="none" stroke="url(#lg)" strokeWidth="2.5" />
        <circle cx="16" cy="16" r="7.5" fill="none" stroke="#4c9bff" strokeOpacity="0.45" strokeWidth="1.5" strokeDasharray="2 2.5" />
        <circle cx="16" cy="16" r="3.2" fill="#2fd0b8" />
      </svg>
      {!collapsed && (
        <div className="leading-tight">
          <div className="text-[14px] font-semibold tracking-tight text-ink-100">
            OncoNova <span className="text-teal">AI</span>
          </div>
          <div className="text-[10px] uppercase tracking-[0.16em] text-ink-500">Research workstation</div>
        </div>
      )}
    </div>
  );
}

function TopBar() {
  const { cases, caseId, setCase, current, tpId, setTp, models, modelId, setModelId, system } = useApp();
  return (
    <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-ink-800 bg-ink-925/90 px-5 py-2.5 backdrop-blur">
      <label className="flex items-center gap-2">
        <span className="label">Case</span>
        <select className="input max-w-[320px] py-1" value={caseId ?? ""} onChange={(e) => setCase(e.target.value)} aria-label="Case">
          {cases.map((c) => (
            <option key={c.case_id} value={c.case_id}>
              {c.case_id} · {c.title}
            </option>
          ))}
        </select>
      </label>
      {current && (
        <div className="flex items-center gap-1" role="tablist" aria-label="Timepoint">
          {current.timepoints.map((t) => (
            <button
              key={t.tp_id}
              role="tab"
              aria-selected={t.tp_id === tpId}
              onClick={() => setTp(t.tp_id)}
              className={cx(
                "rounded-md px-2.5 py-1 text-[12px] font-medium transition-colors",
                t.tp_id === tpId ? "bg-accent/15 text-accent" : "text-ink-400 hover:bg-ink-800 hover:text-ink-200",
              )}
            >
              {t.tp_id} <span className="text-ink-500">· day {t.days_from_baseline}</span>
            </button>
          ))}
        </div>
      )}
      {current?.demo && <DemoBadge />}
      {current && !current.demo && <DemoBadge label={current.label} />}
      <div className="ml-auto flex items-center gap-3">
        <label className="flex items-center gap-2">
          <span className="label">Model</span>
          <select className="input py-1" value={modelId} onChange={(e) => setModelId(e.target.value)} aria-label="Segmentation model">
            {models.map((m) => (
              <option key={m.id} value={m.id} disabled={m.status !== "available"}>
                {m.id}
                {m.status !== "available" ? " (unavailable)" : ""}
              </option>
            ))}
          </select>
        </label>
        <div className="flex items-center gap-1.5" title="Model status">
          {models.map((m) => (
            <span
              key={m.id}
              title={`${m.id}: ${m.status}${m.unavailable_reason ? ` - ${m.unavailable_reason}` : ""}`}
              className={cx("h-2 w-2 rounded-full", m.status === "available" ? "bg-teal shadow-[0_0_8px_#2fd0b8]" : "bg-ink-600")}
            />
          ))}
        </div>
        {system && (
          <span className="mono text-ink-500" title={`Auth mode: ${system.auth_mode}`}>
            {system.principal.user} · {system.principal.role}
          </span>
        )}
      </div>
    </header>
  );
}

export default function App() {
  const [route, go] = useHashRoute();
  const [collapsed, setCollapsed] = useState(false);
  const { systemError, casesError, refresh } = useApp();
  const active = NAV.find((n) => n.id === route) ?? NAV[0];
  const Page = active.page;
  return (
    <div className="flex h-full min-h-0">
      <aside
        className={cx(
          "flex shrink-0 flex-col border-r border-ink-800 bg-ink-925 transition-[width] duration-200",
          collapsed ? "w-[60px]" : "w-[232px]",
        )}
      >
        <div className="flex h-[52px] items-center justify-between px-3">
          <Logo collapsed={collapsed} />
        </div>
        <nav className="mt-2 flex-1 space-y-0.5 px-2" aria-label="Main">
          {NAV.map((n) => {
            const Icon = n.icon;
            const on = n.id === active.id;
            return (
              <a
                key={n.id}
                href={`#/${n.id}`}
                onClick={(e) => {
                  e.preventDefault();
                  go(n.id);
                }}
                title={collapsed ? n.label : undefined}
                aria-current={on ? "page" : undefined}
                className={cx(
                  "group flex items-center gap-3 rounded-lg px-2.5 py-2 text-[13px] font-medium transition-colors",
                  on ? "bg-gradient-to-r from-accent/15 to-transparent text-ink-100" : "text-ink-400 hover:bg-ink-850 hover:text-ink-200",
                )}
              >
                <Icon className={cx("h-[18px] w-[18px] shrink-0", on ? "text-accent" : "text-ink-500 group-hover:text-ink-300")} />
                {!collapsed && <span className="truncate">{n.label}</span>}
              </a>
            );
          })}
        </nav>
        <div className="space-y-2 p-3">
          {!collapsed && (
            <div className="rounded-lg border border-ev-review/25 bg-ev-review/5 p-2.5 text-[11px] leading-snug text-ink-300">
              <div className="mb-1 flex items-center gap-1.5 font-semibold text-ev-review">
                <ShieldAlert className="h-3.5 w-3.5" /> Research use only
              </div>
              Not a medical device. No diagnosis, prescription or device control. Outputs need qualified expert review.
            </div>
          )}
          <button className="btn-ghost w-full" onClick={() => setCollapsed((c) => !c)} aria-label="Toggle sidebar">
            {collapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
            {!collapsed && "Collapse"}
          </button>
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-center gap-2 border-b border-ev-review/20 bg-ev-review/[0.06] px-4 py-1 text-[11px] font-medium tracking-wide text-ev-review">
          <ShieldAlert className="h-3.5 w-3.5" />
          RESEARCH PROTOTYPE · NOT FOR CLINICAL USE · SIMULATIONS ARE NON-CLINICAL · NO CONNECTION TO MEDICAL EQUIPMENT
        </div>
        <TopBar />
        <main className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
          {systemError || casesError ? (
            <div className="mx-auto max-w-xl pt-10">
              <ErrorState error={systemError || casesError} onRetry={refresh} />
            </div>
          ) : (
            <Page />
          )}
        </main>
      </div>
    </div>
  );
}
