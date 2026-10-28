import { ChevronDown, ChevronRight, Download } from "lucide-react";
import { useState } from "react";
import type { ResearchSummary } from "../api/types";
import { ErrorState, FindingList, Panel, Spinner, downloadText } from "./ui";

const SECTION_TITLES: Record<string, string> = {
  A_imaging_observations: "A · Imaging observations",
  B_segmentation_results: "B · Segmentation results",
  C_changes_between_timepoints: "C · Changes between time points",
  D_progression_model_output: "D · Progression model output",
  E_uncertainty_and_limitations: "E · Uncertainty and known limitations",
  F_research_radiation_simulation: "F · Research-only radiation simulation",
  G_items_requiring_expert_review: "G · Items requiring qualified expert review",
};

function toMarkdown(s: ResearchSummary): string {
  return [
    `# Research summary: ${s.case_id}`,
    "",
    `> ${s.disclaimer} Data label: ${s.label}.`,
    "",
    ...Object.entries(s.sections).flatMap(([k, items]) => [`## ${SECTION_TITLES[k] ?? k}`, "", ...items.map((f) => `- **[${f.kind}]** ${f.text}`), ""]),
  ].join("\n");
}

/** The A-G research summary: every line tagged with how it was produced (measured, AI prediction, simulated...). */
export function ResearchSummaryPanel({ summary, loading, error }: { summary: ResearchSummary | null; loading: boolean; error: unknown }) {
  const [open, setOpen] = useState(false);
  return (
    <Panel
      title={
        <button className="flex items-center gap-1.5" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
          {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          Full research report (sections A-G)
        </button>
      }
      bodyClassName={open ? undefined : "hidden"}
      actions={
        summary && (
          <>
            <button className="btn-ghost" onClick={() => downloadText(`${summary.case_id}-summary.json`, JSON.stringify(summary, null, 2))}>
              <Download className="h-3.5 w-3.5" /> JSON
            </button>
            <button className="btn-ghost" onClick={() => downloadText(`${summary.case_id}-summary.md`, toMarkdown(summary), "text/markdown")}>
              <Download className="h-3.5 w-3.5" /> Markdown
            </button>
          </>
        )
      }
    >
      {loading && !summary && <Spinner label="Assembling" />}
      {error ? <ErrorState error={error} /> : null}
      {summary && (
        <div className="space-y-4">
          <p className="text-[12px] text-ink-400">{summary.disclaimer} Each line is tagged with how it was produced.</p>
          <div className="grid gap-4 lg:grid-cols-2">
            {Object.entries(summary.sections).map(([k, items]) => (
              <div key={k} className="rounded-lg border border-ink-700/50 bg-ink-925/50 p-3">
                <h3 className="mb-2 text-[12.5px] font-semibold text-ink-100">{SECTION_TITLES[k] ?? k}</h3>
                <FindingList items={items.slice(0, 8)} />
                {items.length > 8 && <p className="mt-1 text-[11px] text-ink-500">+{items.length - 8} more in the export</p>}
              </div>
            ))}
          </div>
        </div>
      )}
    </Panel>
  );
}
