// Shapes returned by the FastAPI backend (see collection_builder/backend/main.py).

export type TargetMeta = { label: string; color: string };

export type Signal = { type: string; badge: string; primary: boolean };

export type Dataset = {
  id: string;
  title: string;
  study_class: string;
  event_classes: string[];
  decode_target: string;
  region_lca: string | null;
  region_label: string;
  region_short: string;
  channels: number[];
  rates: number[];
  signals: Signal[];
  participants: string[];
  participants_short: string[];
  subjects: string[];
  has_sbp: boolean;
  has_primary: boolean;
  platform: string | null;
  under: string[];
};

export type RegionNode = {
  id: string;
  label: string;
  short: string;
  parent: string | null;
  children: string[];
  counts: Record<string, number>;
  n_datasets: number;
};

export type RegionTree = {
  display_roots: string[];
  nodes: Record<string, RegionNode>;
};

export type Suggestion = { title: string; ids: string[]; note: string | null };

/** JSON-Schema fragment for one editable field of a step (from its pydantic model). */
export type ParamSchema = {
  title?: string;
  type?: string;
  default?: unknown;
  anyOf?: { type: string }[];
};

/** A step the builder can place on a lane: a Study or an EventsTransform. */
export type StepSpec = {
  name: string;
  description: string;
  params: Record<string, ParamSchema>;
};

export type SavedCollection = {
  id: number;
  name: string;
  n_datasets: number;
  datasets: string[];
  /** The composed step chain, so loading the collection restores it. Null for pre-Compose rows. */
  chain: Record<string, unknown> | null;
  /** Declared intent: neuroai event class names this collection reads and decodes. */
  input_event_types: string[];
  target_event_types: string[];
  /** Derived from a recipe digest, never stored. */
  manifest: "not_built" | "stale" | "built";
};

/** An event class the catalog knows, with the kind its neuroai base class implies. */
export type EventClass = {
  name: string;
  kind: "data" | "text" | "categorical" | null;
};

/** Which members can supply the declared streams, answered from the catalog alone. */
export type StreamReport = {
  verdict: "ok" | "warn" | "block";
  available: Record<string, string[]>;
  inputs: Record<string, string[]>;
  missing_inputs: string[];
  missing_targets: string[];
};

export type Bootstrap = {
  target_order: string[];
  target_meta: Record<string, TargetMeta>;
  datasets: Dataset[];
  regions: RegionTree;
  suggestions: Suggestion[];
  saved: SavedCollection[];
  transforms: StepSpec[];
  study_params: Record<string, ParamSchema>;
  event_classes: EventClass[];
};

export type CompatRow = {
  status: "ok" | "info" | "warn" | "block";
  key: string;
  value: string | string[];
};

export type Analysis = {
  verdict: "ok" | "block" | "empty";
  rows: CompatRow[];
  targets: string[];
  rates: number[];
  parts: string[];
  parts_short: string[];
  adapters: number;
  n_subjects: number;
};
