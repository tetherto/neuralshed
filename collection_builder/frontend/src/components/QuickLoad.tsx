import { useEffect, useRef, useState, type CSSProperties } from "react";

import type { Dataset, SavedCollection, Suggestion } from "../types";
import { famVar } from "../util";

type Props = {
  suggestions: Suggestion[];
  saved: SavedCollection[];
  byId: Record<string, Dataset>;
  onLoadSuggestion: (ids: string[]) => void;
  onLoadSaved: (collection: SavedCollection) => void;
  onDeleteSaved: (name: string) => void;
};

function dotColor(ids: string[], byId: Record<string, Dataset>): string {
  const first = ids.map((i) => byId[i]).find(Boolean);
  return first ? famVar(first.decode_target) : "var(--ink-faint)";
}

function dot(color: string) {
  return <span className="sg-dot" style={{ ["--fam"]: color } as CSSProperties} />;
}

function plural(n: number, word: string) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

export function QuickLoad({
  suggestions,
  saved,
  byId,
  onLoadSuggestion,
  onLoadSaved,
  onDeleteSaved,
}: Props) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  // A dropdown that stays open behind a click elsewhere reads as stuck.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const loadSuggestion = (ids: string[]) => {
    onLoadSuggestion(ids);
    setOpen(false);
  };
  const loadSaved = (c: SavedCollection) => {
    onLoadSaved(c);
    setOpen(false);
  };

  return (
    <div className="quickload" ref={root}>
      <button
        type="button"
        className="ql-btn"
        aria-expanded={open}
        aria-haspopup="menu"
        onClick={() => setOpen((v) => !v)}
      >
        <span className="ql-btn-label">Quick load collection</span>
        <span className="ql-caret" aria-hidden="true">
          {open ? "▴" : "▾"}
        </span>
      </button>

      {open && (
        <div className="ql-menu" role="menu">
          <div className="ql-group">Suggested</div>
          {suggestions.map((s, i) => {
            const nParts = new Set(s.ids.flatMap((id) => byId[id]?.participants_short ?? [])).size;
            const meta = s.note
              ? `${plural(s.ids.length, "dataset")} · ${s.note}`
              : `${plural(s.ids.length, "dataset")} · ${plural(nParts, "participant")}`;
            return (
              <div className="ql-row" key={i}>
                <button
                  className="ql-item"
                  type="button"
                  role="menuitem"
                  onClick={() => loadSuggestion(s.ids)}
                >
                  {dot(dotColor(s.ids, byId))}
                  <span className="sg-main">
                    <span className="sg-title">{s.title}</span>
                    <span className="sg-meta">{meta}</span>
                  </span>
                </button>
              </div>
            );
          })}

          <div className="ql-group">Saved</div>
          {saved.length === 0 ? (
            <p className="ql-empty">None yet. Compose a selection and save it.</p>
          ) : (
            saved.map((c) => (
              <div className="ql-row" key={c.name}>
                <button
                  className="ql-item"
                  type="button"
                  role="menuitem"
                  onClick={() => loadSaved(c)}
                >
                  {dot(dotColor(c.datasets, byId))}
                  <span className="sg-main">
                    <span className="sg-title">
                      {c.name}
                      <span className={`sg-badge ${c.manifest}`}>{c.manifest.replace("_", " ")}</span>
                    </span>
                    <span className="sg-meta">
                      {plural(c.n_datasets, "dataset")}
                      {c.target_event_types.length > 0 &&
                        ` · decodes ${c.target_event_types.join(", ")}`}
                    </span>
                  </span>
                </button>
                <button
                  className="ql-del"
                  type="button"
                  aria-label={`delete ${c.name}`}
                  onClick={() => onDeleteSaved(c.name)}
                >
                  ×
                </button>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
