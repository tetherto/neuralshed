import type { TargetMeta } from "../types";

type RegionChip = { id: string; short: string };

type Props = {
  targetOrder: string[];
  targetMeta: Record<string, TargetMeta>;
  fTargets: Set<string>;
  regionChips: RegionChip[];
  onToggleTarget: (t: string) => void;
  onRemoveRegion: (id: string) => void;
  onClearAll: () => void;
};

export function Filters({
  targetOrder,
  targetMeta,
  fTargets,
  regionChips,
  onToggleTarget,
  onRemoveRegion,
  onClearAll,
}: Props) {
  const any = fTargets.size > 0 || regionChips.length > 0;
  return (
    <div className="filters">
      <span className="fl-label">Target</span>
      {targetOrder.map((t) => (
        <button
          key={t}
          className={`chip${fTargets.has(t) ? " on" : ""}`}
          type="button"
          onClick={() => onToggleTarget(t)}
        >
          {targetMeta[t]?.label ?? t}
        </button>
      ))}
      {regionChips.length > 0 && <span className="fl-label" style={{ marginLeft: 6 }}>Region</span>}
      {regionChips.map((rc) => (
        <button
          key={rc.id}
          className="chip region-on"
          type="button"
          onClick={() => onRemoveRegion(rc.id)}
          title="remove region filter"
        >
          {rc.short} ✕
        </button>
      ))}
      {any && (
        <button className="chip clear" type="button" onClick={onClearAll}>
          clear filters
        </button>
      )}
    </div>
  );
}
