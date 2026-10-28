"""Data assembly for the Collections app.
Turns the metadata database into what the UI needs.
"""

from neuralshed import regions
from neuralshed.database import query

# --- display vocab ---------------------------------------------------------
TARGET_META = {
    "text": {"label": "Text", "color": "#3b7dd8"},
    "voice": {"label": "Voice", "color": "#e08331"},
    "movement": {"label": "Movement", "color": "#2f9d6b"},
}
TARGET_ORDER = ["text", "voice", "movement"]

SIGNAL_META = {
    "threshold_crossings": ("TX", True),
    "sorted_spikes": ("Units", True),
    "spike_band_power": ("SBP", False),
    "kinematics": ("Kin", False),
    "audio": ("Audio", False),
    "acoustic_features": ("Acoustic", False),
    "lfp": ("LFP", False),
}
NEURAL_SIGNALS = {"threshold_crossings", "spike_band_power", "sorted_spikes"}
SPIKE_SIGNALS = {"threshold_crossings", "sorted_spikes"}

# Neural read-in modalities the decoder consumes, in display order (primary first).
INPUT_SIGNAL_LABELS = {
    "threshold_crossings": "spike counts",
    "sorted_spikes": "sorted units",
    "spike_band_power": "spike band power",
    "lfp": "LFP",
}


def signal_badge(sig: str) -> str:
    return SIGNAL_META.get(sig, (sig, False))[0]


# --- HOMBA region tree -----------------------------------------------------
REGION_GRAPH = regions.load("homba")
_NODES = REGION_GRAPH["nodes"]
REGION_ROOTS = REGION_GRAPH["roots"]

# node -> ordered children (built by inverting the parent pointers)
REGION_CHILDREN: dict[str, list[str]] = {}
for _nid, _n in _NODES.items():
    if _n["parent"]:
        REGION_CHILDREN.setdefault(_n["parent"], []).append(_nid)


def _display_roots() -> list[str]:
    """Where the coverage tree starts: descend through the non-branching spine so it opens at
    the first real split (frontal cortex today), not the far-off CNS root above it."""
    roots = []
    for r in REGION_ROOTS:
        nid, kids = r, REGION_CHILDREN.get(r, [])
        while len(kids) == 1:
            nid = kids[0]
            kids = REGION_CHILDREN.get(nid, [])
        roots.append(nid)
    return roots


DISPLAY_ROOTS = _display_roots()


def region_label(node_id: str | None) -> str:
    if not node_id:
        return "—"
    return _NODES.get(node_id, {}).get("label", node_id)


def region_short_label(node_id: str | None) -> str:
    """region_label with the HOMBA synonym parenthetical dropped (full name goes in a tooltip)."""
    return region_label(node_id).split(" (")[0]


def region_ancestors(node_id: str) -> list[str]:
    """node_id's ancestors, immediate parent first up to the root (node itself excluded)."""
    return regions.ancestors(node_id, REGION_GRAPH)


def _under(leaf: str) -> list[str]:
    """Every node a leaf sits under: the leaf plus all its ancestors."""
    return [leaf, *region_ancestors(leaf)]


def region_lca(node_ids) -> str | None:
    """Lowest common ancestor of a set of nodes (their single shared region)."""
    node_ids = list(node_ids)
    if not node_ids:
        return None
    paths = [
        list(reversed([n, *region_ancestors(n)])) for n in node_ids
    ]  # each root..node
    common = None
    for tier in zip(*paths):
        if len(set(tier)) == 1:
            common = tier[0]
        else:
            break
    return common


# --- catalog ---------------------------------------------------------------
def load_catalog() -> list[dict]:
    """One summary dict per dataset, assembled from the metadata DB + region graph."""
    ds = query.datasets().sort_values("id")
    subj = query.read_sql("SELECT dataset_id, label, participant_id FROM subjects")
    arr = query.read_sql(
        "SELECT su.dataset_id AS dataset_id, a.subject_id AS subject_id, a.n_channels AS n "
        "FROM arrays a JOIN subjects su ON su.id = a.subject_id"
    )
    ev = query.read_sql(
        "SELECT dataset_id, signal_type, rate_hz FROM event_types WHERE category = 'data'"
    )
    # every Event subclass a dataset emits — what a transform in its chain has to work with
    ecls = query.read_sql(
        "SELECT dataset_id, event_class FROM event_types WHERE event_class IS NOT NULL"
    )
    reg = query.read_sql(
        "SELECT su.dataset_id AS dataset_id, ba.homba_id AS homba_id "
        "FROM brain_areas ba JOIN subjects su ON su.id = ba.subject_id "
        "WHERE ba.homba_id IS NOT NULL"
    )

    ch_by_ds: dict[str, set] = {}
    for (dsid, _sid), n in arr.groupby(["dataset_id", "subject_id"])["n"].sum().items():
        ch_by_ds.setdefault(dsid, set()).add(int(n))
    homba_by_ds: dict[str, set] = {}
    for _, r in reg.iterrows():
        homba_by_ds.setdefault(r.dataset_id, set()).add(r.homba_id)

    out = []
    for _, r in ds.iterrows():
        did = r["id"]
        leaves = sorted(homba_by_ds.get(did, set()))
        under = set()
        for lf in leaves:
            under.update(_under(lf))
        lca = region_lca(leaves)
        streams = ev[ev.dataset_id == did]
        sigs = list(dict.fromkeys(streams.signal_type.dropna()))
        rates = sorted(
            {
                float(x)
                for x in streams.loc[
                    streams.signal_type.isin(NEURAL_SIGNALS), "rate_hz"
                ].dropna()
            }
        )
        sdf = subj[subj.dataset_id == did]
        out.append(
            {
                "id": did,
                "title": r.get("title") or did,
                "decode_target": r.get("decode_target"),
                "macro_task": r.get("macro_task"),
                "status": r.get("status"),
                "platform": r.get("source_data_platform"),
                "study_class": r.get("study_class"),  # `name` in a collection config
                "event_classes": sorted(
                    set(ecls.loc[ecls.dataset_id == did, "event_class"])
                ),
                "homba_leaves": leaves,
                "under": under,  # all region nodes this dataset has data under
                "region_lca": lca,
                "region_label": region_label(lca),
                "channels": sorted(ch_by_ds.get(did, set())),
                "rates": rates,
                "signals": sigs,
                "has_sbp": "spike_band_power" in sigs,
                "has_primary": bool(SPIKE_SIGNALS & set(sigs)),
                "subjects": sorted(set(sdf.label.dropna())),
                "participants": sorted(set(sdf.participant_id.dropna())),
            }
        )
    return out


def datasets_under(node_id: str, catalog: list[dict]) -> list[dict]:
    """Datasets with at least one array in ``node_id`` or any descendant of it."""
    return [d for d in catalog if node_id in d["under"]]


def target_counts(node_id: str, catalog: list[dict]) -> dict[str, int]:
    counts = {t: 0 for t in TARGET_ORDER}
    for d in datasets_under(node_id, catalog):
        if d["decode_target"] in counts:
            counts[d["decode_target"]] += 1
    return counts


def region_rows(catalog: list[dict], expanded: set[str]) -> list[tuple[str, int, bool]]:
    """Visible ``(node_id, depth, has_children)`` rows for the coverage tree, given which nodes
    are expanded. Children are ordered by dataset count (busiest first)."""
    rows: list[tuple[str, int, bool]] = []

    def ordered(nid):
        kids = REGION_CHILDREN.get(nid, [])
        return sorted(
            kids, key=lambda k: (-len(datasets_under(k, catalog)), region_label(k))
        )

    def walk(nid, depth):
        kids = ordered(nid)
        rows.append((nid, depth, bool(kids)))
        if nid in expanded:
            for k in kids:
                walk(k, depth + 1)

    for root in DISPLAY_ROOTS:
        walk(root, 0)
    return rows


def suggestions(catalog: list[dict]) -> list[dict]:
    """Ready-made collections: one per decode target, plus the busiest participant."""
    by_t: dict[str, list[str]] = {}
    for d in catalog:
        by_t.setdefault(d["decode_target"], []).append(d["id"])
    out = []
    for t in TARGET_ORDER:
        ids = sorted(by_t.get(t, []))
        if ids:
            title = (
                "Movement decoding"
                if t == "movement"
                else f"Brain-to-{TARGET_META[t]['label'].lower()}"
            )
            out.append({"title": title, "ids": ids, "note": None})

    part_ds: dict[str, list[str]] = {}
    for d in catalog:
        for p in d["participants"]:
            part_ds.setdefault(p, []).append(d["id"])
    if part_ds:
        p, ids = max(part_ds.items(), key=lambda kv: len(kv[1]))
        if len(ids) > 1:
            out.append(
                {
                    "title": f"Everything {p.split('/')[-1]} did",
                    "ids": sorted(ids),
                    "note": "same participant",
                }
            )
    return out


def analyze(selected_ids, by_id: dict[str, dict]) -> dict:
    """Compatibility of a chosen set, phrased in model consequences."""
    sel = [by_id[i] for i in selected_ids if i in by_id]
    if not sel:
        return {
            "rows": [],
            "verdict": "empty",
            "sel": [],
            "targets": [],
            "rates": [],
            "parts": [],
        }

    targets = sorted({d["decode_target"] for d in sel})
    widths = sorted({c for d in sel for c in d["channels"]})
    rates = sorted({r for d in sel for r in d["rates"]})
    lcas = sorted({d["region_lca"] for d in sel if d["region_lca"]})
    part_counts: dict[str, int] = {}
    for d in sel:
        for p in d["participants"]:
            part_counts[p] = part_counts.get(p, 0) + 1
    parts = sorted(part_counts)
    shared = [p for p, n in part_counts.items() if n > 1]

    def _lbl(p):
        return p.split("/")[-1]

    rows = []
    if len(targets) == 1:
        rows.append(
            (
                "ok",
                "Decode target",
                f"All decode {TARGET_META[targets[0]]['label'].lower()} — one shared decoder head",
            )
        )
    else:
        rows.append(
            (
                "block",
                "Decode target",
                f"Mixed: {', '.join(TARGET_META[t]['label'] for t in targets)} — these can't share one decoder",
            )
        )

    n = len(sel)
    inputs = [
        f"{label} ({'all' if c == n else f'{c}x'})"
        for sig, label in INPUT_SIGNAL_LABELS.items()
        if (c := sum(sig in d["signals"] for d in sel))
    ]
    rows.append(("info", "Inputs", inputs))

    rows.append(
        (
            "info",
            "Channels",
            f"{len(sel)} read-in adapters — widths {', '.join(map(str, widths))} ch",
        )
    )

    rows.append(
        (
            "ok" if len(lcas) == 1 else "info",
            "Brain region",
            f"{len(lcas)} region{'s' if len(lcas) != 1 else ''}: {', '.join(region_label(x) for x in lcas)}",
        )
    )

    pv = f"{len(parts)} subject embedding{'s' if len(parts) != 1 else ''}: {', '.join(_lbl(p) for p in parts)}"
    if shared:
        pv += f" · {', '.join(_lbl(p) for p in shared)} span multiple datasets (within-subject transfer)"
    rows.append(("info", "Participants", pv))

    return {
        "rows": rows,
        "verdict": "ok" if len(targets) == 1 else "block",
        "sel": sel,
        "targets": targets,
        "rates": rates,
        "parts": parts,
    }
