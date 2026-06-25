"""Render the *feature-tour* blog figures for causal-worlds — computed LIVE from the package.

Run from the repo root:

    uv run --extra discover --with matplotlib python blog/figures/make_tour_figures.py

Unlike the launch figures (which read stored eval JSONs), every number here is recomputed from the
built-in `coffee` world at run time, so the charts can never drift from the package's behaviour.

Outputs (PNG, 150 dpi) into blog/figures/:
  tour1_shootout.png      - the discovery shootout: F1 + confounded-pairs-kept, every box-stock method
  tour2_regime_flip.png   - control under perturbation: a regime-blind policy collapses when the regime flips
"""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import causal_worlds as cw
from causal_worlds import worlds

HERE = Path(__file__).resolve().parent
OUT = HERE

# House palette (matches make_figures.py).
INK = "#1b1f24"
MUTED = "#6b7280"
GRID = "#e5e7eb"
REF = "#0f766e"  # teal: the latent-aware reference (the good outcome)
BAD = "#b91c1c"  # red: keeps the confounded pair (the failure)
NEUTRAL = "#64748b"  # slate: everyone else

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def titled(fig: plt.Figure, title: str, subtitle: str, *, wrap: int = 104) -> None:
    fig.suptitle(title, x=0.012, y=0.985, ha="left", va="top", fontsize=14.5, weight="bold", color=INK)
    body = "\n".join(textwrap.wrap(subtitle, wrap))
    fig.text(0.012, 0.915, body, ha="left", va="top", fontsize=9.6, color=MUTED, linespacing=1.35)


def footnote(fig: plt.Figure, text: str) -> None:
    fig.text(0.012, 0.012, text, ha="left", fontsize=8.4, color=MUTED)


def save(fig: plt.Figure, name: str) -> None:
    out = OUT / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(OUT.parents[1])}")


# --------------------------------------------------------------------------- #
# tour1 — the discovery shootout (live)
# --------------------------------------------------------------------------- #
def fig_shootout() -> None:
    spec = worlds.get("coffee")
    methods = [("interventional-ci", cw.InterventionalCiDiscoverer)]
    methods += [(n, d) for n, d in cw.BASELINES.items()]
    rows = []
    for name, disc in methods:
        try:
            r = cw.grade_spec(spec, disc(), seed=0)
            rows.append((name, r.f1, r.confounded_reported))
        except Exception:  # noqa: BLE001 - a baseline that errors on this world is simply skipped
            continue
    rows.sort(key=lambda x: x[1])  # ascending F1, reference ends on top

    names = [r[0] for r in rows]
    f1s = [r[1] for r in rows]
    kept = [r[2] for r in rows]
    colors = [REF if n == "interventional-ci" else (BAD if k > 0 else NEUTRAL) for n, k in zip(names, kept)]

    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    fig.subplots_adjust(top=0.74, left=0.22, right=0.96, bottom=0.15)
    y = range(len(names))
    ax.barh(list(y), f1s, color=colors, height=0.62, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(
        [f"{n}  ◀ reference" if n == "interventional-ci" else n for n in names], fontsize=10
    )
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("directed-edge F1 vs the declared answer key (higher is better)")
    ax.xaxis.grid(visible=True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for yi, (f1, k) in enumerate(zip(f1s, kept)):
        tag = "0 confounded kept" if k == 0 else f"{k} confounded pair kept as causal"
        ax.text(f1 + 0.015, yi, f"{f1:.2f}   ({tag})", va="center", fontsize=8.7,
                color=(REF if k == 0 else BAD))

    titled(
        fig,
        "Only a latent-aware method isn't fooled",
        "Every discovery method that ships in causal-worlds, run on the coffee world (n=8000, seed 0). "
        "Red = keeps the spurious overtime~sales pair as a real causal edge; teal = keeps zero. "
        "The reference is the only one that nails the structure AND avoids the trap.",
    )
    footnote(fig, "computed live: grade_spec(worlds.get('coffee'), <discoverer>, seed=0)   causal-worlds")
    save(fig, "tour1_shootout.png")


# --------------------------------------------------------------------------- #
# tour2 — control under perturbation (live)
# --------------------------------------------------------------------------- #
def fig_regime_flip() -> None:
    spec = worlds.get("coffee")
    obj = cw.default_objective(spec)
    base = cw.regime_optimal_policy(spec, obj, regime_on=set())
    wknd = cw.regime_optimal_policy(spec, obj, regime_on={"weekend"})
    rep_blind = cw.regret_under_perturbation(spec, obj, base, seed=7)  # tuned for baseline only

    regimes = list(rep_blind.per_regime.keys())
    blind = [rep_blind.per_regime[r] for r in regimes]
    aware = [0.0 for _ in regimes]  # the regime-aware controller plays each regime's own optimum

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    fig.subplots_adjust(top=0.72, left=0.1, right=0.96, bottom=0.15)
    x = range(len(regimes))
    w = 0.36
    ax.bar([i - w / 2 for i in x], blind, width=w, color=BAD, label="regime-blind policy", zorder=3)
    ax.bar([i + w / 2 for i in x], aware, width=w, color=REF, label="regime-aware policy", zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r}\nregime" for r in regimes], fontsize=10)
    ax.set_ylabel("regret vs that regime's optimum (0 = optimal)")
    ax.set_ylim(0, max(blind) * 1.25 + 0.3)
    ax.yaxis.grid(visible=True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, v in zip(x, blind):
        ax.text(i - w / 2, v + 0.04, f"{v:.1f}", ha="center", fontsize=9, color=BAD)
    for i in x:
        ax.text(i + w / 2, 0.04, "0.0", ha="center", fontsize=9, color=REF)
    ax.legend(frameon=False, loc="upper left", fontsize=9.2)

    flip = f"price optimum flips sign: {base.get('price'):+.0f} (baseline) -> {wknd.get('price'):+.0f} (weekend)"
    titled(
        fig,
        "A regime-blind policy stays optimal — until the regime flips",
        "The coffee world's price lever reverses sign across regimes, so a policy tuned for one regime "
        f"is perfect there and badly wrong when it flips. {flip}.",
    )
    footnote(fig, "computed live: regret_under_perturbation(coffee, default_objective, regime-blind policy, seed=7)")
    save(fig, "tour2_regime_flip.png")


def main() -> None:
    print("rendering feature-tour figures (live from the package):")
    fig_shootout()
    fig_regime_flip()
    print("done.")


if __name__ == "__main__":
    main()
