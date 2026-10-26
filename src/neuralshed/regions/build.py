"""Generate the in-repo region graph from the official HOMBA release.

Emits only the ontology nodes on a ``part_of`` path from one of our datasets' brain areas up
to the root -- a small induced subtree, not the full 2349-term ontology. Leaf ids are read
from the metadata database (``brain_areas.homba_id``), so re-running after harmonizing new
datasets extends the tree automatically.

    python -m neuralshed.regions.build                       # download HOMBA, query DB, write homba.json
    python -m neuralshed.regions.build --obo homba-simple.obo   # use a local OBO
"""

import argparse
import json
import urllib.request
from pathlib import Path

HOMBA_VERSION = "v2026-04-28"
HOMBA_OBO_URL = (
    "https://github.com/brain-bican/harmonized_ontology_of_mammalian_brain_anatomy_ontology"
    f"/releases/download/{HOMBA_VERSION}/homba-simple.obo"
)
PART_OF = "BFO:0000050"  # HOMBA encodes its hierarchy as part_of, not is_a
OUT = Path(__file__).with_name("homba.json")


def parse_obo(text: str) -> dict:
    """id -> {name, parent} using the first part_of relationship as the parent."""
    terms, cur = {}, None
    rel = f"relationship: {PART_OF} "
    for line in text.splitlines():
        if line == "[Term]":
            cur = {"id": None, "name": None, "parent": None}
        elif line.startswith("[") and line.endswith("]"):
            cur = None
        elif cur is not None:
            if line.startswith("id: ") and cur["id"] is None:
                cur["id"] = line[4:].strip()
                terms[cur["id"]] = cur
            elif line.startswith("name: "):
                cur["name"] = line[6:].strip()
            elif line.startswith(rel) and cur["parent"] is None:
                cur["parent"] = line[len(rel) :].split("!")[0].strip()
    return terms


def db_leaf_ids() -> list[str]:
    from neuralshed.database import query

    df = query.read_sql(
        "SELECT DISTINCT homba_id FROM brain_areas WHERE homba_id IS NOT NULL"
    )
    return sorted(df["homba_id"])  # underscore form, e.g. HOMBA_10170


def _curie(u):  # HOMBA_10170 -> HOMBA:10170
    return u.replace("_", ":", 1)


def _uscore(c):  # HOMBA:10170 -> HOMBA_10170
    return c.replace(":", "_", 1)


def induced(leaf_ids: list[str], terms: dict) -> dict:
    """Union of each leaf's part_of chain to the root, keyed by the DB id form."""
    nodes = {}
    for leaf in leaf_ids:
        cid = _curie(leaf)
        while cid in terms and _uscore(cid) not in nodes:
            t = terms[cid]
            nodes[_uscore(cid)] = {
                "label": t["name"],
                "parent": _uscore(t["parent"]) if t["parent"] else None,
            }
            cid = t["parent"]
    return nodes


def depth(nid, nodes):
    d, cur = 0, nodes[nid]["parent"]
    while cur:
        d, cur = d + 1, nodes[cur]["parent"]
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--obo", help="local homba OBO (default: download the pinned release)"
    )
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    if args.obo:
        text = Path(args.obo).read_text()
    else:
        print(f"downloading {HOMBA_OBO_URL}")
        text = urllib.request.urlopen(HOMBA_OBO_URL, timeout=120).read().decode()

    terms = parse_obo(text)
    leaves = db_leaf_ids()
    nodes = induced(leaves, terms)
    nodes = dict(sorted(nodes.items(), key=lambda kv: (depth(kv[0], nodes), kv[0])))
    roots = [k for k, v in nodes.items() if v["parent"] is None]

    data = {
        "ontology": "HOMBA",
        "id_property": "homba_id",
        "source": HOMBA_OBO_URL,
        "version": HOMBA_VERSION,
        "relation": "part_of (BFO:0000050)",
        "roots": roots,
        "leaves": leaves,
        "nodes": nodes,
    }
    Path(args.out).write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {args.out}: {len(nodes)} nodes, {len(leaves)} leaves, roots={roots}")


if __name__ == "__main__":
    main()
