"""Render the README figures for a built-in world from the package's own ``to_dot`` renderer.

Dogfoods ``to_dot`` and lays it out with Graphviz ``dot`` (clean layered routing). Produces two
figures, each = exactly what a user gets from ``causal-worlds viz coffee --format dot | dot -Tpng``:

  1. coffee_world.png        — the declared SCM (the hero), with a colored role legend.
  2. coffee_do_footfall.png  — Rung 2: the same world after ``do(footfall)`` graph surgery.

Requires Graphviz (`brew install graphviz` / `apt-get install graphviz`):

    uv run python docs/figures/render_world_dag.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from causal_worlds import to_dot, worlds

WORLD = "coffee"
HERE = Path(__file__).resolve().parent

# Title + colored role legend as an HTML-like graph label (labelloc=t ⇒ above the graph body, so it
# never tangles node layout). Swatch colors match the node fills in to_dot / to_mermaid.
_HERO_LABEL = (
    '<<table border="0" cellborder="0" cellspacing="1" cellpadding="3">'
    f'<tr><td colspan="5"><b>causal-worlds — the declared SCM for "{WORLD}"</b></td></tr>'
    '<tr><td colspan="5"><font point-size="10" color="#6b7280">'
    "the hidden confounder (dashed + red) is never in the data — that is what makes recovery hard"
    "</font></td></tr><tr>"
    '<td bgcolor="#dbeafe"> controllable (lever) </td>'
    '<td bgcolor="#dcfce7"> outcome (KPI) </td>'
    '<td bgcolor="#f1f5f9"> observable </td>'
    '<td bgcolor="#fef3c7"> disturbance </td>'
    '<td bgcolor="#fee2e2"> hidden confounder </td>'
    "</tr></table>>"
)
_SURGERY_LABEL = (
    '<<table border="0" cellborder="0" cellspacing="1" cellpadding="3">'
    "<tr><td><b>Rung 2 — do(footfall = 1): graph surgery</b></td></tr>"
    '<tr><td><font point-size="10" color="#6b7280">'
    "the arrows INTO footfall are cut — its causes (incl. the hidden confounder) no longer apply; "
    "its effects still flow"
    "</font></td></tr></table>>"
)


def _render(dot: str, dot_src: str, label: str, out: Path) -> None:
    header = f'  labelloc="t"; labeljust="c"; fontname="Helvetica"; label={label};\n  nodesep=0.45; ranksep=1.05;\n'  # noqa: E501
    src = dot_src.replace("{\n", "{\n" + header, 1)
    subprocess.run(  # noqa: S603 - args fixed here, src is our own generated DOT
        [dot, "-Tpng", "-Gdpi=160", "-o", str(out)], input=src, text=True, check=True
    )
    print(f"wrote {out}")


def main() -> None:
    dot = shutil.which("dot")
    if dot is None:
        sys.exit("graphviz 'dot' not found — install it (e.g. `brew install graphviz`).")
    spec = worlds.get(WORLD)
    _render(dot, to_dot(spec), _HERO_LABEL, HERE / f"{WORLD}_world.png")
    _render(dot, to_dot(spec, do={"footfall": 1.0}), _SURGERY_LABEL, HERE / f"{WORLD}_do_footfall.png")


if __name__ == "__main__":
    main()
