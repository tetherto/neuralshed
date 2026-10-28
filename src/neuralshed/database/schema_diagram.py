"""Generate an entity-relationship diagram of the metadata DB schema.

Introspects ``schema.Base.metadata`` (so the diagram never drifts from the ORM)
and emits a Mermaid ER diagram as a ``.mmd`` source, a standalone self-contained
``.html`` (Mermaid loaded from CDN; open in any browser), and the fenced ``mermaid``
block in ``README.md``.

Output is deterministic: rerunning without a schema change rewrites the same bytes,
so ``git diff`` answers "is the checked-in diagram current?".

Usage::

    python -m neuralshed.database.schema_diagram             # schema.mmd + schema.html + README
    python -m neuralshed.database.schema_diagram --no-readme # skip the README block
    python -m neuralshed.database.schema_diagram --out docs  # custom output dir/prefix
"""

import argparse
import re
from pathlib import Path

from .schema import Base

HERE = Path(__file__).resolve().parent

# Mermaid scalar type buckets (purely cosmetic labels in the diagram).
_TYPE_LABEL = {
    "INTEGER": "int",
    "VARCHAR": "string",
    "TEXT": "text",
    "FLOAT": "float",
    "NUMERIC": "float",
}


def _type_label(col) -> str:
    name = type(col.type).__name__.upper()
    return _TYPE_LABEL.get(name, name.lower())


def build_mermaid() -> str:
    """Return a Mermaid ``erDiagram`` string built from the ORM metadata."""
    md = Base.metadata
    lines: list[str] = ["erDiagram"]

    # Relationships first (parent ||--o{ child for each FK). ``table.foreign_keys`` is an
    # unordered set, so sort to keep the output byte-stable across runs.
    rels: list[str] = []
    for table in md.sorted_tables:
        for fk in table.foreign_keys:
            parent = fk.column.table.name
            child = table.name
            # parent-first: "one parent has zero-or-many children".
            # required FK -> child relates to exactly one parent (||);
            # nullable FK -> child relates to zero-or-one parent (o|).
            card = "||--o{" if not fk.parent.nullable else "|o--o{"
            verb = fk.parent.name.replace("_id", "")
            rels.append(f"    {parent.upper()} {card} {child.upper()} : {verb}")
    lines.extend(sorted(rels))
    lines.append("")

    # Entities with columns, PK/FK markers.
    for table in md.sorted_tables:
        fk_cols = {fk.parent.name for fk in table.foreign_keys}
        lines.append(f"    {table.name.upper()} {{")
        for col in table.columns:
            markers = []
            if col.primary_key:
                markers.append("PK")
            if col.name in fk_cols:
                markers.append("FK")
            marker = " ".join(markers)
            label = _type_label(col)
            suffix = f" {marker}" if marker else ""
            lines.append(f"        {label} {col.name}{suffix}")
        lines.append("    }")
    return "\n".join(lines)


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>neuralshed Metadata Database — Schema</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 1.5rem 2rem;
          background: #fafafa; color: #1a1a1a; }}
  @media (prefers-color-scheme: dark) {{ body {{ background:#16181d; color:#e6e6e6; }} }}
  header {{ margin-bottom: 1rem; }}
  h1 {{ font-size: 1.25rem; margin: 0 0 .25rem; }}
  p.sub {{ margin: 0; opacity: .7; font-size: .9rem; }}
  .diagram {{ overflow: auto; padding: 1rem; border-radius: 10px;
              background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  @media (prefers-color-scheme: dark) {{ .diagram {{ background:#1f2228; box-shadow:none; }} }}
  footer {{ margin-top: 1rem; font-size: .8rem; opacity: .6; }}
</style>
</head>
<body>
<header>
  <h1>neuralshed Metadata Database — Schema (SOW §3.2)</h1>
  <p class="sub">{n_tables} tables · generated from <code>neuralshed/database/schema.py</code> (SQLAlchemy metadata). PK = primary key, FK = foreign key.</p>
</header>
<div class="diagram">
<pre class="mermaid">
{diagram}
</pre>
</div>
<footer>Regenerate: <code>python -m neuralshed.database.schema_diagram</code></footer>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  mermaid.initialize({{ startOnLoad: true, theme: dark ? "dark" : "default" }});
</script>
</body>
</html>
"""


#: The one fenced ``mermaid`` block in README.md, whose body this module owns.
_README_BLOCK = re.compile(r"(?<=^```mermaid\n).*?(?=^```$)", re.DOTALL | re.MULTILINE)


def sync_readme(readme_path: Path, diagram: str) -> bool:
    """Replace the body of README's fenced ``mermaid`` block with ``diagram``.

    Returns whether the file changed. Raises if the block isn't found exactly once,
    so a renamed/removed section fails loudly instead of silently going stale.
    """
    text = readme_path.read_text()
    if len(_README_BLOCK.findall(text)) != 1:
        raise ValueError(f"{readme_path}: expected exactly one ```mermaid block")
    new = _README_BLOCK.sub(lambda _: diagram + "\n", text)
    if new == text:
        return False
    readme_path.write_text(new)
    return True


def write(out_prefix: Path, readme: Path | None = None) -> list[Path]:
    diagram = build_mermaid()
    mmd_path = out_prefix.with_suffix(".mmd")
    html_path = out_prefix.with_suffix(".html")
    mmd_path.write_text(diagram + "\n")
    html_path.write_text(
        _HTML_TEMPLATE.format(
            diagram=diagram, n_tables=len(Base.metadata.sorted_tables)
        )
    )
    written = [mmd_path, html_path]
    if readme is not None:
        sync_readme(readme, diagram)
        written.append(readme)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out",
        default=str(HERE / "schema"),
        help="output path prefix (default: neuralshed/database/schema -> schema.mmd + schema.html)",
    )
    p.add_argument(
        "--no-readme", action="store_true", help="don't sync README.md's mermaid block"
    )
    args = p.parse_args(argv)
    for path in write(
        Path(args.out), readme=None if args.no_readme else HERE / "README.md"
    ):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
