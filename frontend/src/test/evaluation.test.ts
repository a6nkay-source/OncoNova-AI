import { describe, expect, it } from "vitest";
import { latestReports } from "../pages/models/Evaluation";

const r = (report_id: string, model: string, dataset: [string, string], created_utc: string, task = "segmentation") => ({
  report_id,
  task,
  model: { id: model },
  dataset: { id: dataset[0], label: dataset[1] },
  created_utc,
});
const SYN: [string, string] = ["synthetic-seg-v1", "DEMO DATA"];
const MSD: [string, string] = ["msd-task06-lung", "PUBLIC RESEARCH DATA"];

describe("which evaluation reports the Models page shows", () => {
  it("keeps synthetic and real-CT results apart, newest per model and dataset", () => {
    const out = latestReports([
      r("seg-unet-msd", "demo-unet", MSD, "2026-10-29T06:21:35"),
      r("seg-unet-syn-new", "demo-unet", SYN, "2026-10-27T07:04:29"),
      r("seg-unet-syn-old", "demo-unet", SYN, "2026-10-26T07:04:29"),
      r("seg-classical-syn", "demo-classical", SYN, "2026-10-27T07:04:17"),
      r("prog", "progression-demo", ["synthetic-progression", "DEMO DATA"], "2026-10-27T06:21:28", "future_progression_risk"),
    ]);
    expect(out.real).toEqual(["seg-unet-msd"]);
    expect(out.synthetic).toEqual(["seg-unet-syn-new", "seg-classical-syn", "prog"]);
  });

  it("does not render report types it has no view for", () => {
    const out = latestReports([r("cal", "msd-unet-ens", MSD, "2026-10-29T12:00:00", "segmentation_calibration")]);
    expect(out).toEqual({ synthetic: [], real: [] });
  });
});
