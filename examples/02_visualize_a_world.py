"""Visualize a world's declared SCM — an SCM is a DAG, so look at it (no API key).

`to_mermaid` and `to_dot` are pure strings (no extra dependencies). Paste the Mermaid into any
Markdown file (GitHub renders it live) or the DOT into a Graphviz viewer. The hidden confounder is
drawn dashed — it is exactly the latent structure a discovery method never gets to see.

    uv run python examples/02_visualize_a_world.py

Expected output:

    # Mermaid (paste into a ```mermaid block on GitHub):
    graph LR
        weekend{{"weekend"}}:::disturbance
        price(["price"]):::controllable
        local_buzz(("local_buzz")):::hidden
        footfall["footfall"]:::observable
        overtime["overtime"]:::observable
        demand["demand"]:::observable
        sales[["sales"]]:::outcome
        local_buzz -.-> footfall
        local_buzz -.-> overtime
        footfall --> overtime
        price --> demand
        footfall --> demand
        weekend --> demand
        demand --> sales
        footfall --> sales
        local_buzz -.-> sales
        classDef controllable fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a;
        ...
    answer key: 6 causal edges, 1 hidden-confounded pair(s)
    confounded (no direct edge, shared hidden cause): overtime ~ sales
"""

from causal_worlds import answer_key, to_mermaid, worlds

spec = worlds.get("coffee")

print("# Mermaid (paste into a ```mermaid block on GitHub):")
print(to_mermaid(spec))

key = answer_key(spec)
print(f"\nanswer key: {len(key.edges)} causal edges, {len(key.confounded)} hidden-confounded pair(s)")
for pair in key.confounded:
    a, b = sorted(pair)
    print(f"confounded (no direct edge, shared hidden cause): {a} ~ {b}")
# Tip: `causal-worlds viz coffee` prints the same Mermaid; `--format dot` prints Graphviz DOT.
# For a generated world on disk: `to_mermaid(load_bundle("./my-world").spec)`.
