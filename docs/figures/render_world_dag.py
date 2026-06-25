"""Render a built-in world's declared SCM to a PNG for the README (renders on GitHub AND PyPI).

This is the one-off image generator for the docs. It dogfoods the package's own ``to_dot`` renderer
and lays it out with Graphviz ``dot`` (proper layered routing — a clean, single-glance DAG), then
injects a title + a colored role legend (as the graph's top label, so it never tangles the layout).
The graph body is *exactly* what a user gets from ``causal-worlds viz coffee --format dot | dot -Tpng``.

Requires Graphviz (`brew install graphviz` / `apt-get install graphviz`):

    uv run python docs/figures/render_world_dag.py

Outputs docs/figures/coffee_world.png. Re-run after changing the world; commit the PNG.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from causal_worlds import to_dot, worlds

WORLD = "coffee"
OUT = Path(__file__).resolve().parent / f"{WORLD}_world.png"

# Title + colored role legend as one HTML-like graph label (labelloc=t ⇒ sits above the graph body,
# so it can't interfere with node layout). Swatch colors match the node fills in to_dot / to_mermaid.
_LEGEND = (
    '<<table border="0" cellborder="0" cellspacing="1" cellpadding="3">'
    f'<tr><td colspan="5"><b>causal-worlds — the declared SCM for "{WORLD}"</b></td></tr>'
    '<tr><td colspan="5"><font point-size="10" color="#6b7280">'
    "the hidden confounder (dashed + red) is never in the data — that is what makes recovery hard"
    "</font></td></tr>"
    '<tr>'
    '<td bgcolor="#dbeafe"> controllable (lever) </td>'
    '<td bgcolor="#dcfce7"> outcome (KPI) </td>'
    '<td bgcolor="#f1f5f9"> observable </td>'
    '<td bgcolor="#fef3c7"> disturbance </td>'
    '<td bgcolor="#fee2e2"> hidden confounder </td>'
    "</tr></table>>"
)
_HEADER = (
    f'  labelloc="t"; labeljust="c"; fontname="Helvetica"; label={_LEGEND};\n'
    "  nodesep=0.45; ranksep=1.05;\n"
)


def main() -> None:
    dot = shutil.which("dot")
    if dot is None:
        sys.exit("graphviz 'dot' not found — install it (e.g. `brew install graphviz`).")
    src = to_dot(worlds.get(WORLD)).replace("{\n", "{\n" + _HEADER, 1)
    subprocess.run(  # noqa: S603 - args fixed here, src is our own generated DOT
        [dot, "-Tpng", "-Gdpi=160", "-o", str(OUT)],
        input=src,
        text=True,
        check=True,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
