import type { Dataset, EventClass, StreamReport } from "../types";

type Props = {
  items: Dataset[];
  eventClasses: EventClass[];
  inputs: Set<string>;
  targets: Set<string>;
  report: StreamReport | null;
  onToggleInput: (name: string) => void;
  onToggleTarget: (name: string) => void;
};

type Candidate = { name: string; kind: string; n: number };

/** Every class the selection carries, with its kind and how many members have it. */
function candidates(items: Dataset[], eventClasses: EventClass[]): Candidate[] {
  const kindOf = new Map(eventClasses.map((e) => [e.name, e.kind ?? "other"]));
  const counts = new Map<string, number>();
  for (const d of items) {
    for (const c of d.event_classes) counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([name, n]) => ({ name, kind: kindOf.get(name) ?? "other", n }))
    .sort((a, b) => b.n - a.n || a.name.localeCompare(b.name));
}

// Kind orders the chips and never removes one. Kind and role are independent axes: the manifest
// assigns a role from what the collection declares and never consults the kind. Filtering by kind
// here would make whole decoder families unreachable, since `Audio` is kind `data` and so is every
// behavioral series.
const READ_ORDER = ["data", "text", "categorical", "other"];
const DECODE_ORDER: Record<string, string[]> = {
  text: ["text", "categorical", "data", "other"],
  voice: ["data", "text", "categorical", "other"],
  movement: ["data", "categorical", "text", "other"],
};
const DECODE_ORDER_MIXED = ["text", "categorical", "data", "other"];

/** The selection's decode target, when the members agree on one. */
function decodeTarget(items: Dataset[]): string | null {
  const all = new Set(items.map((d) => d.decode_target));
  return all.size === 1 ? [...all][0] : null;
}

/**
 * The target to preselect, or nothing.
 *
 * Only fires when it cannot be wrong: the members agree on a decode target, and exactly one class
 * of the kind that target implies is carried by every member. Anything less stays unset, because a
 * target is required and gates the save, so a wrong guess would be saved rather than questioned.
 */
export function defaultTargets(items: Dataset[], eventClasses: EventClass[]): string[] {
  const target = decodeTarget(items);
  if (!target || items.length === 0) return [];
  const wantedKind = (DECODE_ORDER[target] ?? DECODE_ORDER_MIXED)[0];
  const shared = candidates(items, eventClasses).filter(
    (c) => c.kind === wantedKind && c.n === items.length,
  );
  return shared.length === 1 ? [shared[0].name] : [];
}

function Row({
  label,
  hint,
  options,
  order,
  chosen,
  total,
  onToggle,
}: {
  label: string;
  hint: string;
  options: Candidate[];
  order: string[];
  chosen: Set<string>;
  total: number;
  onToggle: (name: string) => void;
}) {
  // Kind sorts the chips and the trailing slice catches any kind the order forgot, so no class
  // can fall out of the list.
  const ordered = [
    ...order.flatMap((kind) => options.filter((o) => o.kind === kind)),
    ...options.filter((o) => !order.includes(o.kind)),
  ];

  return (
    <div className="dc-row">
      <div className="dc-label">
        {label} <span className="dc-hint">{hint}</span>
      </div>
      <div className="dc-chips">
        {ordered.length === 0 ? (
          <span className="dc-none">nothing in the selection</span>
        ) : (
          ordered.map((o) => (
            <button
              key={o.name}
              type="button"
              className={`chip ${chosen.has(o.name) ? "on" : ""}`}
              aria-pressed={chosen.has(o.name)}
              onClick={() => onToggle(o.name)}
            >
              {o.name}
              <span className="dc-count">
                {o.n}/{total}
              </span>
            </button>
          ))
        )}
      </div>
    </div>
  );
}

export function Declare({
  items,
  eventClasses,
  inputs,
  targets,
  report,
  onToggleInput,
  onToggleTarget,
}: Props) {
  const total = items.length;
  const opts = candidates(items, eventClasses);
  const target = decodeTarget(items);
  const decodeOrder = (target && DECODE_ORDER[target]) || DECODE_ORDER_MIXED;

  // Partial inputs are the normal case, not a fault: a member simply contributes fewer streams.
  const partial = report
    ? Object.entries(report.inputs)
        .filter(([, found]) => found.length > 0 && found.length < inputs.size)
        .map(([id, found]) => `${id} supplies ${found.join(", ")} only`)
    : [];

  return (
    <div className="declare">
      <h3>Modalities</h3>
      <Row
        label="Input"
        hint="wanted if present"
        options={opts}
        order={READ_ORDER}
        chosen={inputs}
        total={total}
        onToggle={onToggleInput}
      />
      <Row
        label="Target"
        hint="required of every member"
        options={opts}
        order={decodeOrder}
        chosen={targets}
        total={total}
        onToggle={onToggleTarget}
      />

      {report?.missing_targets.length ? (
        <p className="dc-msg block">
          {report.missing_targets.join(", ")} cannot supply {[...targets].join(" or ")}.
        </p>
      ) : null}
      {report?.missing_inputs.length ? (
        <p className="dc-msg block">
          {report.missing_inputs.join(", ")} carry none of the declared inputs.
        </p>
      ) : null}
      {partial.length > 0 && <p className="dc-msg warn">{partial.join(" · ")}</p>}
    </div>
  );
}
