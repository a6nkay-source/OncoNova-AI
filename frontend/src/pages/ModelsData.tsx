import { useEffect, useState } from "react";
import { Segmented } from "../components/ui";
import { DatasetsSection } from "./models/Datasets";
import { EvaluationSection } from "./models/Evaluation";
import { SafetySection } from "./models/Safety";

type Tab = "performance" | "datasets" | "safety";
const TABS: { value: Tab; label: string }[] = [
  { value: "performance", label: "Model performance" },
  { value: "datasets", label: "Datasets and cases" },
  { value: "safety", label: "Safety and access" },
];
const KEY = "onconova.modelsTab";

function initialTab(): Tab {
  try {
    const v = localStorage.getItem(KEY) as Tab | null;
    return v && TABS.some((t) => t.value === v) ? v : "performance";
  } catch {
    return "performance";
  }
}

export function ModelsDataPage() {
  const [tab, setTab] = useState<Tab>(initialTab);
  useEffect(() => {
    try {
      localStorage.setItem(KEY, tab);
    } catch {
      /* storage unavailable */
    }
  }, [tab]);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Segmented value={tab} onChange={setTab} options={TABS} />
        <span className="text-[12px] text-ink-400">
          {tab === "performance" && "Measured on held-out synthetic data and loaded from stored reports. Nothing here is clinical performance."}
          {tab === "datasets" && "Public datasets checked against their official pages, plus the cases on this server."}
          {tab === "safety" && "What the prototype will not do, how data is handled, and access settings."}
        </span>
      </div>
      {tab === "performance" && <EvaluationSection />}
      {tab === "datasets" && <DatasetsSection />}
      {tab === "safety" && <SafetySection />}
    </div>
  );
}
