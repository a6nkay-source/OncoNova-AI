import { AlertTriangle, CircleSlash, Info, Loader2, RefreshCw } from "lucide-react";
import type { ReactNode } from "react";
import { ApiError } from "../api/client";
import type { EvidenceKind, Finding, Unavailable } from "../api/types";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

export function Panel({
  title,
  actions,
  children,
  className,
  bodyClassName,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={cx("panel flex min-w-0 flex-col", className)}>
      {(title || actions) && (
        <header className="panel-header">
          <h2 className="panel-title">{title}</h2>
          <div className="flex items-center gap-2">{actions}</div>
        </header>
      )}
      <div className={cx("min-h-0 flex-1 p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

const EVIDENCE: Record<EvidenceKind, { label: string; cls: string }> = {
  measured: { label: "Measured", cls: "text-ev-measured border-ev-measured/40 bg-ev-measured/10" },
  measured_from_model_output: {
    label: "Measured · AI mask",
    cls: "text-ev-fromModel border-ev-fromModel/40 bg-ev-fromModel/10",
  },
  model_output: { label: "AI prediction", cls: "text-ev-model border-ev-model/40 bg-ev-model/10" },
  uncertain: { label: "Uncertain", cls: "text-ev-uncertain border-ev-uncertain/40 bg-ev-uncertain/10" },
  requires_expert_review: { label: "Expert review", cls: "text-ev-review border-ev-review/40 bg-ev-review/10" },
  simulated: { label: "Simulated", cls: "text-ev-simulated border-ev-simulated/40 bg-ev-simulated/10" },
  unavailable: { label: "Unavailable", cls: "text-ev-unavailable border-ev-unavailable/40 bg-ev-unavailable/10" },
  imported: { label: "Imported", cls: "text-accent border-accent/40 bg-accent/10" },
};

export function EvidenceBadge({ kind, className }: { kind: EvidenceKind; className?: string }) {
  const e = EVIDENCE[kind] ?? EVIDENCE.unavailable;
  return (
    <span
      className={cx(
        "inline-flex shrink-0 items-center whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
        e.cls,
        className,
      )}
      title={`Evidence type: ${e.label}`}
    >
      {e.label}
    </span>
  );
}

export function FindingList({ items, empty }: { items: Finding[]; empty?: string }) {
  if (!items.length) return <p className="text-[13px] text-ink-400">{empty ?? "Nothing to show."}</p>;
  return (
    <ul className="space-y-2">
      {items.map((f, i) => (
        <li key={i} className="flex items-start gap-2.5 text-[13px] leading-snug text-ink-200">
          <EvidenceBadge kind={f.kind} className="mt-[1px] w-[118px] justify-center" />
          <span className="min-w-0">{f.text}</span>
        </li>
      ))}
    </ul>
  );
}

export function DemoBadge({ label = "DEMO DATA" }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-ev-uncertain/50 bg-ev-uncertain/10 px-2 py-0.5 text-[10.5px] font-bold uppercase tracking-[0.12em] text-ev-uncertain">
      {label}
    </span>
  );
}

export function SimulatedBanner({ text }: { text?: string }) {
  return (
    <div className="sim-stripes flex items-center gap-3 rounded-lg border border-ev-simulated/40 px-4 py-2.5 text-[12.5px] text-ev-simulated">
      <AlertTriangle className="h-4 w-4 shrink-0" />
      <span>
        <b className="font-semibold tracking-wide">SIMULATED · NON-CLINICAL.</b>{" "}
        {text ??
          "Geometric illustration in relative units. Not a dose calculation, not a treatment plan, no prescription. This software has no connection to any treatment machine."}
      </span>
    </div>
  );
}

export function UnavailableState({ data, title }: { data: Unavailable; title?: string }) {
  return (
    <div role="status" className="rounded-lg border border-dashed border-ink-600 bg-ink-925/60 p-4">
      <div className="flex items-center gap-2 text-ink-200">
        <CircleSlash className="h-4 w-4 text-ev-unavailable" />
        <span className="text-[13px] font-semibold">{title ?? "Unavailable"}</span>
        <EvidenceBadge kind="unavailable" />
      </div>
      <p className="mt-2 text-[13px] text-ink-300">{data.reason}</p>
      {data.upstream_reason && <p className="mt-1 text-[12px] text-ink-400">Upstream: {data.upstream_reason}</p>}
      {data.missing_inputs?.length > 0 && (
        <p className="mt-1 text-[12px] text-ink-400">
          Missing: <span className="mono text-ink-300">{data.missing_inputs.join(", ")}</span>
        </p>
      )}
      {data.how_to_fix && <p className="mt-1 text-[12px] text-ink-400">To enable: {data.how_to_fix}</p>}
      <p className="mt-2 text-[11px] text-ink-500">No estimate was produced. Nothing has been inferred from missing inputs.</p>
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const e = error instanceof ApiError ? error : null;
  const message = e?.message ?? (error instanceof Error ? error.message : "Something went wrong.");
  return (
    <div role="alert" className="rounded-lg border border-ev-review/40 bg-ev-review/5 p-4">
      <div className="flex items-center gap-2 text-ev-review">
        <AlertTriangle className="h-4 w-4" />
        <span className="text-[13px] font-semibold">{e?.status === 0 ? "Backend unreachable" : "Request failed"}</span>
        {e && <span className="mono text-ink-400">{e.code}</span>}
      </div>
      <p className="mt-1.5 text-[13px] text-ink-200">{message}</p>
      {e?.requestId && <p className="mt-1 text-[11px] text-ink-500">Request ID {e.requestId}</p>}
      {onRetry && (
        <button className="btn-ghost mt-3" onClick={onRetry}>
          <RefreshCw className="h-3.5 w-3.5" /> Retry
        </button>
      )}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-[13px] text-ink-300">
      <Loader2 className="h-4 w-4 animate-spin text-accent" />
      {label}
    </span>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <p className="flex items-start gap-2 text-[12px] leading-snug text-ink-400">
      <Info className="mt-[2px] h-3.5 w-3.5 shrink-0 text-ink-500" />
      <span>{children}</span>
    </p>
  );
}

export function Stat({
  label,
  value,
  unit,
  kind,
  sub,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  kind?: EvidenceKind;
  sub?: ReactNode;
}) {
  return (
    <div className="min-w-0 rounded-lg border border-ink-700/60 bg-ink-925/70 px-3 py-2.5">
      <div className="flex items-center justify-between gap-2">
        <span className="label truncate">{label}</span>
        {kind && <EvidenceBadge kind={kind} />}
      </div>
      <div className="mt-1 flex items-baseline gap-1">
        <span className="font-mono text-[20px] font-medium tabular-nums text-ink-100">{value}</span>
        {unit && <span className="text-[12px] text-ink-400">{unit}</span>}
      </div>
      {sub && <div className="mt-0.5 text-[11.5px] text-ink-400">{sub}</div>}
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  size = "sm",
}: {
  value: T;
  options: { value: T; label: ReactNode; disabled?: boolean; title?: string }[];
  onChange: (v: T) => void;
  size?: "sm" | "xs";
}) {
  return (
    <div className="inline-flex rounded-lg border border-ink-700 bg-ink-925 p-0.5" role="radiogroup">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          disabled={o.disabled}
          title={o.title}
          onClick={() => onChange(o.value)}
          className={cx(
            "rounded-md font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40",
            size === "sm" ? "px-2.5 py-1 text-[12px]" : "px-2 py-0.5 text-[11px]",
            value === o.value ? "bg-ink-700 text-ink-100" : "text-ink-400 hover:text-ink-200",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function fmt(v: number | null | undefined, digits = 2): string {
  return v === null || v === undefined || Number.isNaN(v) ? "-" : v.toFixed(digits);
}

export function fmtCI(ci: [number | null, number | null] | undefined, digits = 3): string {
  if (!ci || ci[0] === null || ci[1] === null) return "CI n/a";
  return `95% CI ${ci[0].toFixed(digits)}-${ci[1].toFixed(digits)}`;
}

export function downloadText(filename: string, text: string, mime = "application/json") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
