import type { RegionTree, TargetMeta } from "../types";
import { famVar } from "../util";

type Props = {
  tree: RegionTree;
  targetOrder: string[];
  targetMeta: Record<string, TargetMeta>;
  expanded: Set<string>;
  selectedRegions: Set<string>;
  onToggleExpand: (nid: string) => void;
  onToggleRegion: (nid: string) => void;
};

type Row = { nid: string; depth: number; hasKids: boolean };

function visibleRows(tree: RegionTree, expanded: Set<string>): Row[] {
  const rows: Row[] = [];
  const walk = (nid: string, depth: number) => {
    const node = tree.nodes[nid];
    if (!node) return;
    const hasKids = node.children.length > 0;
    rows.push({ nid, depth, hasKids });
    if (hasKids && expanded.has(nid)) {
      for (const k of node.children) walk(k, depth + 1);
    }
  };
  for (const r of tree.display_roots) walk(r, 0);
  return rows;
}

export function CoverageTree({
  tree,
  targetOrder,
  targetMeta,
  expanded,
  selectedRegions,
  onToggleExpand,
  onToggleRegion,
}: Props) {
  const rows = visibleRows(tree, expanded);
  return (
    <section className="panel coverage">
      <div className="cov-head">
        <h2>Coverage</h2>
      </div>
      <div className="covgrid">
        <div className="cov-colhead">
          <span className="ch-region">brain region (HOMBA)</span>
          {targetOrder.map((t) => (
            <span className="ch-t" key={t} style={{ color: famVar(t) }}>
              <span className="swatch" style={{ background: famVar(t) }} />
              {targetMeta[t]?.label ?? t}
            </span>
          ))}
        </div>
        <div className="covtree">
          {rows.map(({ nid, depth, hasKids }) => {
            const node = tree.nodes[nid];
            const active = selectedRegions.has(nid);
            return (
              <div className={`tree-row${active ? " active" : ""}`} key={nid}>
                <span className="tree-label" style={{ paddingLeft: depth * 14 }}>
                  <button
                    className={`tree-toggle${hasKids ? "" : " leaf"}`}
                    type="button"
                    onClick={() => hasKids && onToggleExpand(nid)}
                    aria-label={expanded.has(nid) ? "collapse" : "expand"}
                    tabIndex={hasKids ? 0 : -1}
                  >
                    {hasKids ? (expanded.has(nid) ? "▼" : "▶") : "•"}
                  </button>
                  <label className="tree-name" title={node.label}>
                    <input
                      type="checkbox"
                      className="tree-check"
                      checked={active}
                      onChange={() => onToggleRegion(nid)}
                    />
                    <span className="tree-name-text">{node.short}</span>
                  </label>
                </span>
                {targetOrder.map((t) => {
                  const n = node.counts[t] ?? 0;
                  return (
                    <span
                      className={`tree-count${n ? "" : " zero"}`}
                      key={t}
                      style={n ? { color: famVar(t) } : undefined}
                    >
                      {n || "·"}
                    </span>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
