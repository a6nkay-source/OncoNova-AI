export type EvidenceKind =
  | "measured"
  | "measured_from_model_output"
  | "model_output"
  | "uncertain"
  | "requires_expert_review"
  | "simulated"
  | "unavailable"
  | "imported";

export interface Finding {
  kind: EvidenceKind;
  text: string;
  value?: number;
  unit?: string;
  track_id?: string;
}

export interface Unavailable {
  status: "unavailable";
  kind: "unavailable";
  reason: string;
  missing_inputs: string[];
  how_to_fix?: string | null;
  upstream_reason?: string;
}

export interface TimepointSummary {
  tp_id: string;
  label: string;
  days_from_baseline: number;
  modality: string;
}

export interface CaseSummary {
  case_id: string;
  title: string;
  demo: boolean;
  label: string;
  modality: string;
  cancer_type: string;
  body_region: string;
  timepoints: TimepointSummary[];
  has_reference: boolean;
  has_structures: boolean;
}

export interface Geometry {
  shape_zyx: [number, number, number];
  spacing_xyz_mm: [number, number, number];
  origin_xyz_mm: [number, number, number];
  direction: number[];
  coordinate_system: string;
  oblique: boolean;
}

export interface ModelCard {
  id: string;
  name: string;
  version: string;
  kind: string;
  modalities: string[];
  classes: Record<string, string>;
  trained_on: string;
  intended_use: string;
  limitations: string[];
  label: string;
  validated_for_clinical_use: boolean;
  status: "available" | "unavailable";
  unavailable_reason: string | null;
  uncertainty_method: string;
  checkpoint_sha256: string | null;
  training_manifest: string | null;
}

export interface LesionComponent {
  volume_cc: number;
  centroid_mm: number[];
  centroid_index_zyx: number[];
  bbox_index_zyx: number[][];
  max_axial_diameter_mm: number;
  sphericity: number | null;
  class: string;
  class_id?: number;
  mean_foreground_probability?: number;
  mean_uncertainty?: number;
}

export interface UnreliableRegion {
  volume_cc: number;
  mean_uncertainty: number;
  centroid_index_zyx: number[];
  centroid_mm: number[];
}

export interface SegmentationSummary {
  case_id: string;
  tp_id: string;
  model_id: string;
  kind: EvidenceKind;
  label: string;
  model: ModelCard;
  volumes_cc: Record<string, number>;
  volume_kind: EvidenceKind;
  lesions: LesionComponent[];
  uncertainty: {
    mean_uncertainty_in_prediction: number | null;
    high_uncertainty_fraction_of_prediction: number | null;
    threshold: number;
  };
  uncertainty_method: string;
  unreliable_regions: UnreliableRegion[];
  assessed_fraction_of_body: number;
  has_saliency: boolean;
  warnings: string[];
  findings: Finding[];
  runtime_s: number;
  provenance: Record<string, unknown>;
  saved_utc?: string;
}

export interface PairMetrics {
  dice: number;
  iou: number;
  hausdorff_mm: number | null;
  hd95_mm: number | null;
  mean_surface_distance_mm: number | null;
  sensitivity: number | null;
  specificity: number | null;
  specificity_roi: string;
  pred_volume_cc: number;
  ref_volume_cc: number;
}

export interface ComparisonResult {
  case_id: string;
  tp_id: string;
  model_id: string;
  reference: string;
  reference_kind: string;
  any_lesion: PairMetrics;
  per_class: Record<string, PairMetrics>;
  note: string;
}

export interface Registration {
  method: string;
  final_metric: number;
  iterations: number;
  translation_mm: number[];
  rotation_deg: number[];
  stop_condition: string;
  qc: "ok" | "review";
}

export interface LongitudinalTrack {
  track_id: string;
  class: string;
  first_seen: string;
  status: "persisting" | "new" | "not_segmented_at_last_scan";
  volume_series_cc: (number | null)[];
  volume_change_pct?: number;
  centroid_shift_mm?: number | null;
  measurements: Record<string, LesionComponent & { centroid_in_baseline_frame_mm?: number[] | null }>;
}

export interface LongitudinalResult {
  status: "ok";
  source: string;
  measurement_kind: EvidenceKind;
  baseline_tp: string;
  timepoints: {
    tp_id: string;
    days_from_baseline: number;
    total_lesion_volume_cc: number;
    lesion_count: number;
    registration: Registration | null;
  }[];
  tracks: LongitudinalTrack[];
  total_volume_change_pct: number | null;
  new_lesion_count: number;
  findings: Finding[];
  notes: string[];
}

export interface ProgressionPrediction {
  status: "ok";
  kind: EvidenceKind;
  label: string;
  outcome: string;
  horizon_days: number;
  probability: number;
  interval95: [number, number];
  interval_method: string;
  uncertain: boolean;
  training_percentile: number;
  contributions: { feature: string; contribution_log_odds: number }[];
  imputed_features: string[];
  out_of_distribution_warnings: string[];
  inputs: Record<string, number | null>;
  requires_expert_review: boolean;
  limitations: string[];
  source: string;
}

export interface Scenario {
  id: string;
  name: string;
  n_beams: number;
  margin_mm: number;
  description: string;
  label: string;
  hypothetical: boolean;
}

export interface StructureStats {
  volume_cc: number;
  mean: number;
  max_D2: number;
  D50: number;
  V5: number;
  V20: number;
  V50: number;
  D95?: number;
  D98?: number;
  V95?: number;
  homogeneity_index?: number;
}

export interface DVH {
  dose_bins: number[];
  curves: Record<string, number[]>;
  stats: Record<string, StructureStats>;
  skipped_empty: string[];
  paddick_conformity_index_95?: number;
}

export interface DoseResult {
  status: "ok";
  kind: EvidenceKind;
  label: string;
  case_id: string;
  tp_id: string;
  scenario: Scenario;
  target_source: string;
  dose_units: string;
  dose_map: string;
  dvh: DVH;
  warnings: string[];
  /** Per-beam replay metadata (absent in results computed before replay support). */
  beams?: { angles_deg: number[]; slice_index: number; angle_convention: string; note: string };
  saved_utc?: string;
}

export interface ResearchSummary {
  case_id: string;
  label: string;
  demo: boolean;
  sections: Record<string, Finding[]>;
  disclaimer: string;
  provenance: Record<string, unknown>;
}

export interface SystemInfo {
  name: string;
  version: string;
  disclaimer: string;
  demo_label: string;
  simulation_label: string;
  device_control: string;
  prescription: string;
  external_transmission: string;
  auth_mode: string;
  principal: { user: string; role: string };
  limits: { max_upload_mb: number; max_zip_members: number };
  models: { id: string; status: string; reason: string | null }[];
  cases: { total: number; demo: number };
}

export interface MetricCI {
  value: number | null;
  ci95: [number | null, number | null];
  metric: string;
}

export interface BinaryReport {
  n: number;
  n_events: number;
  prevalence: number;
  auroc: MetricCI;
  auprc: MetricCI;
  brier: MetricCI;
  sensitivity: MetricCI;
  specificity: MetricCI;
  threshold: number;
  threshold_source: string;
  calibration: {
    bins: { mean_predicted: number; observed_fraction: number; n: number }[];
    expected_calibration_error: number;
    calibration_slope: number | null;
    calibration_intercept: number | null;
  };
  c_index: MetricCI & { n: number };
  excluded_censored_before_horizon: number;
  status?: string;
}

export interface SummaryStat {
  n_patients: number;
  mean: number | null;
  median: number | null;
  std?: number;
  ci95: [number | null, number | null];
}

export interface EvaluationReport {
  report_id: string;
  task: "segmentation" | "future_progression_risk";
  model: { id: string; version: string; kind?: string; checkpoint_sha256?: string | null; artifact_sha256?: string };
  dataset: Record<string, unknown> & { id: string; label: string };
  protocol: string;
  results: Record<string, unknown>;
  subgroups?: Record<string, BinaryReport>;
  feature_importance?: { feature: string; standardized_log_odds: number }[];
  per_scan?: { patient_id: string; scan_id: string; any_lesion: PairMetrics }[];
  caveat: string;
  created_utc: string;
}

export interface DatasetEntry {
  id: string;
  name: string;
  cancer_type: string;
  body_region: string;
  modality: string;
  format: string;
  patients: string;
  images?: string;
  annotations: string;
  ground_truth: string;
  outcomes?: string;
  download_url: string;
  landing_page?: string;
  paper: string;
  dataset_doi?: string;
  license: string;
  access: string;
  size: string;
  suitable_for: string[];
  notes?: string;
  recommended_starter?: boolean;
}
