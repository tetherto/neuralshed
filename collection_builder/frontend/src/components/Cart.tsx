import type { ReactNode } from "react";

import type { Analysis, Dataset } from "../types";
import { famStyle, fmtChannels, fmtRates, isValidCollectionName } from "../util";

const ICON: Record<string, string> = { ok: "✓", info: "i", warn: "!", block: "✕" };

type Props = {
  items: Dataset[];
  analysis: Analysis | null;
  name: string;
  saving: boolean;
  savedMsg: string | null;
  /** The declared-streams picker, owned by App and rendered above the save footer. */
  declare: ReactNode;
  /** Why saving is blocked, or null when it is allowed. */
  blocked: string | null;
  onRemove: (id: string) => void;
  onClear: () => void;
  onChangeName: (v: string) => void;
  onSave: () => void;
};

export function Cart({
  items,
  analysis,
  name,
  saving,
  savedMsg,
  declare,
  blocked,
  onRemove,
  onClear,
  onChangeName,
  onSave,
}: Props) {
  const nameOk = isValidCollectionName(name);
  const nameDirty = name.trim().length > 0;
  const hasAnalysis = items.length > 0 && analysis && analysis.verdict !== "empty";
  return (
    <aside className="panel cart">
      <div className="cart-head">
        <h2>Current Selection</h2>
        {items.length > 0 && (
          <button className="cart-clear" type="button" onClick={onClear}>
            clear
          </button>
        )}
      </div>

      <div className="cart-body">
        {items.length === 0 ? (
          <div className="cart-empty">
          Empty. Click a coverage region, a collection, or <b>Add</b> on any dataset.
        </div>
      ) : (
        <div className="cart-items">
          {items.map((d) => (
            <div className="ci" key={d.id} style={famStyle(d.decode_target)}>
              <span className="ci-dot" />
              <span className="ci-name">{d.id}</span>
              <span className="ci-sub">
                {fmtChannels(d.channels)}ch · {fmtRates(d.rates)}Hz
              </span>
              <button className="ci-x" type="button" aria-label={`remove ${d.id}`} onClick={() => onRemove(d.id)}>
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {hasAnalysis && (
        <>
          <div className="compat">
            <h3>Compatibility</h3>
            {analysis!.rows.map((r, i) => (
              <div className={`cx-row cx-${r.status}`} key={i}>
                <span className="cx-ic">{ICON[r.status]}</span>
                <div className="cx-body">
                  <span className="cx-k">{r.key}</span>
                  {Array.isArray(r.value) ? (
                    <ul className="cx-list">
                      {r.value.map((v, j) => (
                        <li key={j}>{v}</li>
                      ))}
                    </ul>
                  ) : (
                    <span className="cx-v">{r.value}</span>
                  )}
                </div>
              </div>
            ))}
          </div>

          {analysis!.verdict === "ok" ? (
            <div className="verdict ok">
              <span className="v-ic">✓</span>
              <div>
                <div className="v-title">Ready to train</div>
                <div className="v-body">
                  1 shared decoder · {analysis!.adapters} adapters · {analysis!.n_subjects} subjects
                </div>
              </div>
            </div>
          ) : (
            <div className="verdict block">
              <span className="v-ic">✕</span>
              <div>
                <div className="v-title">Not poolable yet</div>
                <div className="v-body">Pick a single decode target — a collection trains one decoder head.</div>
              </div>
            </div>
          )}

          {analysis!.verdict === "ok" && declare}

          {analysis!.verdict === "ok" && (
            <div className="cart-foot">
              <div className="field">
                <label htmlFor="cname">Collection name</label>
                <input
                  id="cname"
                  type="text"
                  placeholder="e.g. speech-text"
                  value={name}
                  aria-invalid={nameDirty && !nameOk}
                  aria-describedby="cname-hint"
                  onChange={(e) => onChangeName(e.target.value)}
                />
                <p id="cname-hint" className={`field-hint ${nameDirty && !nameOk ? "bad" : ""}`}>
                  Letters, digits, <code>-</code> and <code>_</code> only. No spaces.
                </p>
              </div>
              <button
                className="export-btn"
                type="button"
                disabled={!nameOk || saving || blocked !== null}
                onClick={onSave}
              >
                {saving ? "Saving…" : "Save collection"}
              </button>
              {blocked && <p className="field-hint bad">{blocked}</p>}
              {savedMsg && <div className="saved-toast">{savedMsg}</div>}
            </div>
          )}
        </>
      )}
      </div>
    </aside>
  );
}
