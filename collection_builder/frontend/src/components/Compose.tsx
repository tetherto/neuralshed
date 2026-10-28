import { useMemo, useState } from "react";

import type { Analysis, Dataset, ParamSchema, StepSpec } from "../types";
import {
  laneKey,
  newStep,
  uniqueKeys,
  type ComposeState,
  type Lane,
  type StepInstance,
} from "../compose";
import { famStyle } from "../util";
import { StepCard } from "./StepCard";

type Props = {
  selected: Dataset[];
  /** Built in App, so the panel below and the Save button ship the same config. */
  lanes: Lane[];
  config: Record<string, unknown> | null;
  analysis: Analysis | null;
  transforms: StepSpec[];
  studyParams: Record<string, ParamSchema>;
  state: ComposeState;
  onChange: (next: ComposeState) => void;
};

function AddTransform({
  transforms,
  onAdd,
}: {
  transforms: StepSpec[];
  onAdd: (spec: StepSpec) => void;
}) {
  const [open, setOpen] = useState(false);
  if (transforms.length === 0) return null;
  return (
    <div className="add-step">
      <button type="button" className="add-step-btn" onClick={() => setOpen((v) => !v)}>
        + add transform
      </button>
      {open && (
        <div className="add-step-menu">
          {transforms.map((t) => (
            <button
              key={t.name}
              type="button"
              onClick={() => {
                onAdd(t);
                setOpen(false);
              }}
            >
              <code>{t.name}</code>
              <span>{t.description}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** Reorder helpers over a step list; returns a new array. */
function move<T>(items: T[], from: number, to: number): T[] {
  const out = [...items];
  const [x] = out.splice(from, 1);
  out.splice(to, 0, x);
  return out;
}

export function Compose({
  selected,
  lanes,
  config,
  analysis,
  transforms,
  studyParams,
  state,
  onChange,
}: Props) {
  const laneKeys = useMemo(() => lanes.map(laneKey), [lanes]);
  const collectionKeys = useMemo(
    () => uniqueKeys(state.collection.map((s) => s.name)),
    [state.collection],
  );
  const [showJson, setShowJson] = useState(true);

  // -- lane mutations ------------------------------------------------------
  const laneEntry = (id: string) => state.lanes[id] ?? { study: {}, transforms: [] };

  const setLane = (id: string, patch: Partial<ComposeState["lanes"][string]>) =>
    onChange({ ...state, lanes: { ...state.lanes, [id]: { ...laneEntry(id), ...patch } } });

  const setStudyParam = (id: string, key: string, value: unknown) =>
    setLane(id, { study: { ...laneEntry(id).study, [key]: value } });

  const setLaneSteps = (id: string, steps: StepInstance[]) => setLane(id, { transforms: steps });

  const setStepParam = (steps: StepInstance[], uid: string, key: string, value: unknown) =>
    steps.map((s) => (s.uid === uid ? { ...s, params: { ...s.params, [key]: value } } : s));

  const schemaOf = (name: string) => transforms.find((t) => t.name === name)?.params ?? {};

  if (selected.length === 0) {
    return (
      <section className="panel compose-empty">
        <h2>Compose</h2>
        <p>
          Pick datasets in <strong>Selection</strong> first. Each one becomes a member of the
          collection, with its own chain of transforms.
        </p>
      </section>
    );
  }

  if (analysis && analysis.verdict === "block") {
    return (
      <section className="panel compose-empty">
        <h2>Compose</h2>
        <p className="compose-blocked">
          This selection is not poolable: it mixes decode targets (
          {analysis.targets.join(", ")}). One collection trains one decoder head, so its members
          must share a target. Fix the selection to compose a chain.
        </p>
      </section>
    );
  }

  return (
    <section className="compose">
      <div className="compose-head">
        <h2>Compose</h2>
        <p>
          Each lane is one member: its study, then the transforms applied to that study alone.
          Steps below the fan-in run once, on the pooled events. Lanes are shown in the order
          <code>CombineStudies</code> concatenates them.
        </p>
      </div>

      <div className="lanes">
        {lanes.map((lane, li) => (
          <div className="lane" key={lane.dataset.id} style={famStyle(lane.dataset.decode_target)}>
            <div className="lane-head">
              <span className="lane-dataset">{lane.dataset.id}</span>
              <code className="lane-key">{laneKeys[li]}</code>
            </div>

            <StepCard
              step={lane.study}
              schema={studyParams}
              kind="study"
              subtitle={lane.dataset.title}
              badges={lane.dataset.event_classes}
              onChangeParam={(k, v) => setStudyParam(lane.dataset.id, k, v)}
            />

            {lane.transforms.map((step, i) => (
              <StepCard
                key={step.uid}
                step={step}
                schema={schemaOf(step.name)}
                kind="transform"
                onChangeParam={(k, v) =>
                  setLaneSteps(
                    lane.dataset.id,
                    setStepParam(lane.transforms, step.uid, k, v),
                  )
                }
                onRemove={() =>
                  setLaneSteps(
                    lane.dataset.id,
                    lane.transforms.filter((s) => s.uid !== step.uid),
                  )
                }
                onMoveUp={
                  i > 0
                    ? () => setLaneSteps(lane.dataset.id, move(lane.transforms, i, i - 1))
                    : undefined
                }
                onMoveDown={
                  i < lane.transforms.length - 1
                    ? () => setLaneSteps(lane.dataset.id, move(lane.transforms, i, i + 1))
                    : undefined
                }
              />
            ))}

            <AddTransform
              transforms={transforms}
              onAdd={(spec) =>
                setLaneSteps(lane.dataset.id, [...lane.transforms, newStep(spec)])
              }
            />
          </div>
        ))}
      </div>

      <div className="fan-in" aria-hidden="true">
        <span />
      </div>

      <div className="collection-lane">
        <div className="lane-head">
          <span className="lane-dataset">pooled events</span>
          <code className="lane-key">CombineStudies</code>
        </div>

        {state.collection.map((step, i) => (
          <StepCard
            key={step.uid}
            step={step}
            schema={schemaOf(step.name)}
            kind="transform"
            subtitle={collectionKeys[i]}
            onChangeParam={(k, v) =>
              onChange({
                ...state,
                collection: setStepParam(state.collection, step.uid, k, v),
              })
            }
            onRemove={() =>
              onChange({
                ...state,
                collection: state.collection.filter((s) => s.uid !== step.uid),
              })
            }
            onMoveUp={
              i > 0
                ? () => onChange({ ...state, collection: move(state.collection, i, i - 1) })
                : undefined
            }
            onMoveDown={
              i < state.collection.length - 1
                ? () => onChange({ ...state, collection: move(state.collection, i, i + 1) })
                : undefined
            }
          />
        ))}

        <AddTransform
          transforms={transforms}
          onAdd={(spec) => onChange({ ...state, collection: [...state.collection, newStep(spec)] })}
        />
      </div>

      <div className="panel json-panel">
        <button
          type="button"
          className="json-toggle"
          onClick={() => setShowJson((v) => !v)}
          aria-expanded={showJson}
        >
          <span>{showJson ? "▾" : "▸"}</span> config
          <em>Data.study — no paths; they are injected at load</em>
        </button>
        {showJson && (
          <div className="code">
            <pre>{JSON.stringify(config, null, 2)}</pre>
          </div>
        )}
      </div>
    </section>
  );
}
