"""Region ontology reference (in-repo, ontology-swappable).

Holds a small *induced subtree* of a brain-region ontology covering only the areas our
datasets actually use -- HOMBA today (``homba.json``); a UBERON sibling (``uberon.json``)
could drop in with the same shape. Each node maps an id to its official label and its
``part_of`` parent, so the collection builder / model can render a collapsible region tree and walk it
to any granularity. The DB stays ontology-agnostic (it stores only ``homba_id`` /
``uberon_id`` on ``brain_areas``); this file supplies the names and hierarchy.

Regenerate with ``python -m neuralshed.regions.build`` after harmonizing new datasets.
"""

import json
from pathlib import Path

_DIR = Path(__file__).parent


def load(ontology: str = "homba") -> dict:
    """Return the region graph: ``{ontology, version, roots, nodes: {id: {label, parent}}}``."""
    return json.loads((_DIR / f"{ontology}.json").read_text())


def ancestors(node_id: str, graph: dict) -> list[str]:
    """``node_id``'s ancestors, immediate parent first up to the root (node itself excluded)."""
    out, nodes = [], graph["nodes"]
    cur = nodes.get(node_id, {}).get("parent")
    while cur:
        out.append(cur)
        cur = nodes.get(cur, {}).get("parent")
    return out


__all__ = ["load", "ancestors"]
