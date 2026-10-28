import type { Dataset, TargetMeta } from "../types";
import { famStyle, fmtChannels, fmtRates } from "../util";

type Props = {
  datasets: Dataset[];
  selected: Set<string>;
  targetMeta: Record<string, TargetMeta>;
  onToggle: (id: string) => void;
  onOpenReadme: (id: string) => void;
};

export function Catalog({ datasets, selected, targetMeta, onToggle, onOpenReadme }: Props) {
  return (
    <div className="cards">
      {datasets.length === 0 && <div className="cards-empty">No datasets match these filters.</div>}
      {datasets.map((d) => {
        const sel = selected.has(d.id);
        return (
          <div className={`card${sel ? " sel" : ""}`} key={d.id} style={famStyle(d.decode_target)}>
            <div className="card-top">
              <div>
                <div className="card-id">{d.id}</div>
                <div className="card-title">{d.title}</div>
              </div>
              <span className="fam-tag">{targetMeta[d.decode_target]?.label ?? d.decode_target}</span>
            </div>
            <dl className="card-grid">
              <dt>Subject</dt>
              <dd>{d.participants_short.join(", ") || "—"}</dd>
              <dt>Region</dt>
              <dd>{d.region_short}</dd>
              <dt>Channels</dt>
              <dd>{fmtChannels(d.channels)} ch</dd>
              <dt>Rate</dt>
              <dd>{fmtRates(d.rates)} Hz</dd>
            </dl>
            <div className="sigs">
              {d.signals.map((s) => (
                <span className={`sig${s.primary ? " pri" : ""}`} key={s.type}>
                  {s.badge}
                </span>
              ))}
            </div>
            <div className="card-actions">
              <button className="add-btn" type="button" onClick={() => onToggle(d.id)}>
                {sel ? "Remove" : "Add to collection"}
              </button>
              <button
                className="card-readme"
                type="button"
                title="Open README"
                aria-label="Open README"
                onClick={() => onOpenReadme(d.id)}
              >
                <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path
                    d="M3.5 2.25A.75.75 0 0 1 4.25 1.5h4.9L12.5 4.85v8.9a.75.75 0 0 1-.75.75h-7.5a.75.75 0 0 1-.75-.75V2.25Z"
                    stroke="currentColor"
                    strokeWidth="1.2"
                    strokeLinejoin="round"
                  />
                  <path d="M9 1.5v3.6h3.6" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
                </svg>
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
