import { useEffect, useMemo, useRef, useState } from "react";

import { analyze, checkStreams, deleteCollection, getBootstrap, saveCollection } from "./api";
import {
  buildConfig,
  buildLanes,
  emptyCompose,
  parseConfig,
  pruneLanes,
  type ComposeState,
} from "./compose";
import { Cart } from "./components/Cart";
import { Declare, defaultTargets } from "./components/Declare";
import { Catalog } from "./components/Catalog";
import { Compose } from "./components/Compose";
import { CoverageTree } from "./components/CoverageTree";
import { Filters } from "./components/Filters";
import { Header } from "./components/Header";
import { QuickLoad } from "./components/QuickLoad";
import { ReadmeDrawer } from "./components/ReadmeDrawer";
import type { Analysis, Bootstrap, Dataset, SavedCollection, StreamReport } from "./types";
import { isValidCollectionName } from "./util";

const DEFAULT_SELECTED = ["card_2024", "willet_2023", "fan_2023"];

type Tab = "selection" | "compose";

export default function App() {
  const [boot, setBoot] = useState<Bootstrap | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [selected, setSelected] = useState<Set<string>>(new Set(DEFAULT_SELECTED));
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [fTargets, setFTargets] = useState<Set<string>>(new Set());
  const [fRegions, setFRegions] = useState<Set<string>>(new Set());

  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [inputs, setInputs] = useState<Set<string>>(new Set());
  const [targets, setTargets] = useState<Set<string>>(new Set());
  const [streams, setStreams] = useState<StreamReport | null>(null);
  const [saved, setSaved] = useState<SavedCollection[]>([]);
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);
  const [readmeId, setReadmeId] = useState<string | null>(null);

  const [tab, setTab] = useState<Tab>("selection");
  const [compose, setCompose] = useState<ComposeState>(emptyCompose);

  // --- load ---------------------------------------------------------------
  useEffect(() => {
    getBootstrap()
      .then((b) => {
        setBoot(b);
        setSaved(b.saved);
        setExpanded(new Set(Object.values(b.regions.nodes).filter((n) => n.children.length).map((n) => n.id)));
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  // --- derived ------------------------------------------------------------
  const byId = useMemo<Record<string, Dataset>>(
    () => Object.fromEntries((boot?.datasets ?? []).map((d) => [d.id, d])),
    [boot],
  );
  const selKey = useMemo(() => [...selected].sort().join(","), [selected]);

  const filtered = useMemo(
    () =>
      (boot?.datasets ?? []).filter(
        (d) =>
          (fTargets.size === 0 || fTargets.has(d.decode_target)) &&
          (fRegions.size === 0 || [...fRegions].some((r) => d.under.includes(r))),
      ),
    [boot, fTargets, fRegions],
  );

  const cartItems = useMemo(
    () =>
      [...selected]
        .map((id) => byId[id])
        .filter(Boolean)
        .sort((a, b) => a.id.localeCompare(b.id)),
    [selKey, byId], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const regionChips = [...fRegions].map((id) => ({ id, short: boot?.regions.nodes[id]?.short ?? id }));

  // Removing a dataset from the selection removes it from the collection: drop its lane.
  useEffect(() => {
    setCompose((c) => pruneLanes(c, selected));
  }, [selKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // The composed step chain, built once here: Compose renders it, Cart saves it.
  const lanes = useMemo(() => buildLanes(compose, cartItems), [compose, cartItems]);
  const config = useMemo(
    () =>
      boot ? buildConfig(lanes, compose.collection, boot.study_params, boot.transforms) : null,
    [lanes, compose.collection, boot],
  );

  // --- compatibility (debounced) ------------------------------------------
  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!boot) return;
    const ids = selKey ? selKey.split(",") : [];
    if (ids.length === 0) {
      setAnalysis(null);
      return;
    }
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      analyze(ids).then(setAnalysis).catch(() => setAnalysis(null));
    }, 180);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [selKey, boot]);

  // Offer a target only while none is chosen, so this never overwrites a hand-picked one or the
  // declaration a saved collection was loaded with. Clearing them all is left alone on purpose.
  useEffect(() => {
    if (!boot || targets.size > 0 || cartItems.length === 0) return;
    const guess = defaultTargets(cartItems, boot.event_classes);
    if (guess.length) setTargets(new Set(guess));
  }, [selKey, boot]); // eslint-disable-line react-hooks/exhaustive-deps

  // The declared streams are checked against the catalog alone, so this is cheap and can run
  // on every toggle. It is what turns "I picked three datasets" into "and this is what for".
  const declKey = useMemo(
    () => `${[...inputs].sort()}|${[...targets].sort()}`,
    [inputs, targets],
  );
  useEffect(() => {
    const ids = selKey ? selKey.split(",") : [];
    if (ids.length === 0 || (inputs.size === 0 && targets.size === 0)) {
      setStreams(null);
      return;
    }
    checkStreams(ids, [...inputs], [...targets])
      .then(setStreams)
      .catch(() => setStreams(null));
  }, [selKey, declKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // A collection with no declared streams saves fine but cannot be turned into a manifest, so
  // the builder asks for them up front rather than letting that surface much later.
  const blocked = useMemo(() => {
    if (targets.size === 0) return "Pick what this collection decodes.";
    if (inputs.size === 0) return "Pick at least one input stream.";
    if (streams?.verdict === "block") return "Some members cannot supply the declared target.";
    return null;
  }, [inputs, targets, streams]);

  // --- handlers -----------------------------------------------------------
  const toggleSelect = (id: string) =>
    setSelected((s) => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  const toggleExpand = (nid: string) =>
    setExpanded((s) => {
      const n = new Set(s);
      n.has(nid) ? n.delete(nid) : n.add(nid);
      return n;
    });
  const toggleTarget = (t: string) =>
    setFTargets((s) => {
      const n = new Set(s);
      n.has(t) ? n.delete(t) : n.add(t);
      return n;
    });
  const toggleIn = (name: string) =>
    setInputs((s) => {
      const n = new Set(s);
      n.has(name) ? n.delete(name) : n.add(name);
      return n;
    });
  const toggleOut = (name: string) =>
    setTargets((s) => {
      const n = new Set(s);
      n.has(name) ? n.delete(name) : n.add(name);
      return n;
    });
  const toggleRegion = (nid: string) =>
    setFRegions((s) => {
      const n = new Set(s);
      n.has(nid) ? n.delete(nid) : n.add(nid);
      return n;
    });

  async function handleSave() {
    if (!isValidCollectionName(name)) return;
    setSaving(true);
    try {
      const res = await saveCollection(name.trim(), [...selected], config, [...inputs], [...targets]);
      setSaved(res.collections);
      setSavedMsg(`Saved '${name.trim()}'`);
      window.setTimeout(() => setSavedMsg(null), 2500);
    } catch (e) {
      setSavedMsg(e instanceof Error ? e.message : "save failed");
    } finally {
      setSaving(false);
    }
  }

  // A suggestion carries no chain, so it starts one from scratch.
  function loadSuggestion(ids: string[]) {
    setSelected(new Set(ids));
    setCompose(emptyCompose);
    setInputs(new Set());
    setTargets(new Set());
  }

  // Restoring the chain and the name means re-saving overwrites what was stored, not defaults.
  function loadSaved(c: SavedCollection) {
    setSelected(new Set(c.datasets));
    setCompose(parseConfig(c.chain, byId));
    setInputs(new Set(c.input_event_types));
    setTargets(new Set(c.target_event_types));
    setName(c.name);
  }

  async function handleDelete(n: string) {
    try {
      const res = await deleteCollection(n);
      setSaved(res.collections);
    } catch {
      /* ignore */
    }
  }

  // --- render -------------------------------------------------------------
  if (error) return <div className="wrap status-line error">Failed to load: {error}</div>;
  if (!boot) return <div className="wrap status-line">Loading catalog…</div>;

  return (
    <div className="wrap">
      <Header />

      <div className="layout">
        <aside className="sidebar">
          <QuickLoad
            suggestions={boot.suggestions}
            saved={saved}
            byId={byId}
            onLoadSuggestion={loadSuggestion}
            onLoadSaved={loadSaved}
            onDeleteSaved={handleDelete}
          />
          <Cart
            items={cartItems}
            analysis={analysis}
            name={name}
            saving={saving}
            savedMsg={savedMsg}
            blocked={blocked}
            declare={
              <Declare
                items={cartItems}
                eventClasses={boot.event_classes}
                inputs={inputs}
                targets={targets}
                report={streams}
                onToggleInput={toggleIn}
                onToggleTarget={toggleOut}
              />
            }
            onRemove={toggleSelect}
            onClear={() => setSelected(new Set())}
            onChangeName={setName}
            onSave={handleSave}
          />
        </aside>

        <main className="main">
          <nav className="tabs" role="tablist">
            {(["selection", "compose"] as Tab[]).map((t) => (
              <button
                key={t}
                role="tab"
                type="button"
                aria-selected={tab === t}
                className={`tab ${tab === t ? "on" : ""}`}
                onClick={() => setTab(t)}
              >
                {t === "selection" ? "Selection" : "Compose"}
              </button>
            ))}
          </nav>

          {tab === "selection" ? (
            <>
              <CoverageTree
                tree={boot.regions}
                targetOrder={boot.target_order}
                targetMeta={boot.target_meta}
                expanded={expanded}
                selectedRegions={fRegions}
                onToggleExpand={toggleExpand}
                onToggleRegion={toggleRegion}
              />
              <div className="toolbar">
                <Filters
                  targetOrder={boot.target_order}
                  targetMeta={boot.target_meta}
                  fTargets={fTargets}
                  regionChips={regionChips}
                  onToggleTarget={toggleTarget}
                  onRemoveRegion={toggleRegion}
                  onClearAll={() => {
                    setFTargets(new Set());
                    setFRegions(new Set());
                  }}
                />
              </div>
              <Catalog
                datasets={filtered}
                selected={selected}
                targetMeta={boot.target_meta}
                onToggle={toggleSelect}
                onOpenReadme={setReadmeId}
              />
            </>
          ) : (
            <Compose
              selected={cartItems}
              lanes={lanes}
              config={config}
              analysis={analysis}
              transforms={boot.transforms}
              studyParams={boot.study_params}
              state={compose}
              onChange={setCompose}
            />
          )}
        </main>
      </div>

      <ReadmeDrawer datasetId={readmeId} onClose={() => setReadmeId(null)} />
    </div>
  );
}
