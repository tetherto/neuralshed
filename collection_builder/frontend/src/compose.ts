// Builds the neuroai Step config the Compose tab emits.
//
// The shape mirrors neuralshed/transforms.py: a Chain whose head is CombineStudies. Each
// member is a list of step dicts (a Study, then that study's own transforms); steps after the
// fan-in apply to the pooled frame. Keys are derived from the step class names, never typed by
// the user, so identical collections serialize identically.

import type { Dataset, ParamSchema, StepSpec } from "./types";

/** One placed step: which class, and the params the user filled in. */
export type StepInstance = {
  uid: string; // React key only; never serialized
  name: string;
  params: Record<string, unknown>;
};

/** A member of the collection: one study plus its own transform chain. */
export type Lane = {
  dataset: Dataset;
  study: StepInstance;
  transforms: StepInstance[];
};

export type ComposeState = {
  /** Per-dataset study params + transforms, keyed by dataset id. */
  lanes: Record<string, { study: Record<string, unknown>; transforms: StepInstance[] }>;
  /** Steps applied to the pooled frame, after the fan-in. */
  collection: StepInstance[];
};

export const emptyCompose: ComposeState = { lanes: {}, collection: [] };

/** Drop lanes for datasets no longer selected: leaving the selection leaves the collection. */
export function pruneLanes(state: ComposeState, selectedIds: Set<string>): ComposeState {
  const kept = Object.keys(state.lanes).filter((id) => selectedIds.has(id));
  if (kept.length === Object.keys(state.lanes).length) return state;
  return {
    ...state,
    lanes: Object.fromEntries(kept.map((id) => [id, state.lanes[id]])),
  };
}

let _uid = 0;
export const newUid = () => `s${++_uid}`;

export function newStep(spec: StepSpec): StepInstance {
  return { uid: newUid(), name: spec.name, params: {} };
}

/**
 * Suffix repeated names with `_0`, `_1`, … in first-seen order; unique names unchanged.
 * Only the pooled steps can repeat: lane keys start with a study class, unique per dataset.
 */
export function uniqueKeys(names: string[]): string[] {
  const counts = new Map<string, number>();
  for (const n of names) counts.set(n, (counts.get(n) ?? 0) + 1);
  const seen = new Map<string, number>();
  return names.map((n) => {
    if (counts.get(n) === 1) return n;
    const i = seen.get(n) ?? 0;
    seen.set(n, i + 1);
    return `${n}_${i}`;
  });
}

/** A member's key: its step class names joined by `_` (study first). */
export function laneKey(lane: Lane): string {
  return [lane.study.name, ...lane.transforms.map((t) => t.name)].join("_");
}

/** The default a pydantic field declares, used to drop no-op params from the config. */
function isDefault(value: unknown, schema: ParamSchema | undefined): boolean {
  if (value === undefined || value === null || value === "") return true;
  return schema !== undefined && value === schema.default;
}

/**
 * `{name, ...params}`, omitting anything left at its default so the blob stays minimal.
 * Params are sorted, so the bytes depend on the config and not on the order fields were typed.
 */
function stepDict(
  step: StepInstance,
  schema: Record<string, ParamSchema>,
): Record<string, unknown> {
  const out: Record<string, unknown> = { name: step.name };
  for (const k of Object.keys(step.params).sort()) {
    if (!isDefault(step.params[k], schema[k])) out[k] = step.params[k];
  }
  return out;
}

/**
 * Members execute in canonical order (by study class name), matching `CombineStudies._run`,
 * so what the UI lists top-to-bottom is what the backend concatenates.
 */
export function orderedLanes(lanes: Lane[]): Lane[] {
  return [...lanes].sort((a, b) => a.study.name.localeCompare(b.study.name));
}

export function buildLanes(state: ComposeState, selected: Dataset[]): Lane[] {
  const lanes = selected.map((dataset) => {
    const entry = state.lanes[dataset.id];
    return {
      dataset,
      study: {
        uid: `study:${dataset.id}`,
        name: dataset.study_class,
        params: entry?.study ?? {},
      },
      transforms: entry?.transforms ?? [],
    };
  });
  return orderedLanes(lanes);
}

/** The stored blob. No `path` anywhere: it is environment, injected when the config is loaded. */
export function buildConfig(
  lanes: Lane[],
  collection: StepInstance[],
  studyParams: Record<string, ParamSchema>,
  transformSpecs: StepSpec[],
): Record<string, unknown> | null {
  if (lanes.length === 0) return null;
  const schemaOf = (name: string) =>
    transformSpecs.find((t) => t.name === name)?.params ?? {};

  const chains: Record<string, unknown> = {};
  for (const [key, lane] of zip(lanes.map(laneKey), lanes)) {
    chains[key] = [
      stepDict(lane.study, studyParams),
      ...lane.transforms.map((t) => stepDict(t, schemaOf(t.name))),
    ];
  }

  const steps: Record<string, unknown> = {
    CombineStudies: { name: "CombineStudies", chains },
  };
  for (const [key, step] of zip(uniqueKeys(collection.map((s) => s.name)), collection)) {
    steps[key] = stepDict(step, schemaOf(step.name));
  }
  return { name: "Chain", steps };
}

function zip<A, B>(a: A[], b: B[]): [A, B][] {
  return a.map((x, i) => [x, b[i]]);
}

// --- reading a stored config back ------------------------------------------

type StepDict = { name: string } & Record<string, unknown>;

function asRecord(v: unknown): Record<string, unknown> | null {
  return typeof v === "object" && v !== null && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : null;
}

function asStep(v: unknown): StepDict | null {
  const r = asRecord(v);
  return r && typeof r.name === "string" ? (r as StepDict) : null;
}

function stepParams(s: StepDict): Record<string, unknown> {
  const { name: _name, ...params } = s;
  return params;
}

function stepInstance(s: StepDict): StepInstance {
  return { uid: newUid(), name: s.name, params: stepParams(s) };
}

/** A member is a list of steps (what `buildConfig` writes) or a bare Study dict. */
function memberSteps(v: unknown): StepDict[] {
  const list = Array.isArray(v) ? v : [v];
  return list.map(asStep).filter((s): s is StepDict => s !== null);
}

/**
 * Inverse of `buildConfig`. A member names its study class, never its dataset, so the class is
 * the link back. An unknown class drops its lane rather than failing the load.
 */
export function parseConfig(
  config: Record<string, unknown> | null | undefined,
  byId: Record<string, Dataset>,
): ComposeState {
  const steps = asRecord(config?.steps);
  if (!steps) return emptyCompose;

  const datasetOf = new Map(Object.values(byId).map((d) => [d.study_class, d.id]));
  const chains = asRecord(asRecord(steps.CombineStudies)?.chains) ?? {};

  const lanes: ComposeState["lanes"] = {};
  for (const member of Object.values(chains)) {
    const [study, ...transforms] = memberSteps(member);
    const id = study && datasetOf.get(study.name);
    if (!id) continue;
    lanes[id] = { study: stepParams(study), transforms: transforms.map(stepInstance) };
  }

  const collection = Object.entries(steps)
    .filter(([key]) => key !== "CombineStudies")
    .map(([, v]) => asStep(v))
    .filter((s): s is StepDict => s !== null)
    .map(stepInstance);

  return { lanes, collection };
}
