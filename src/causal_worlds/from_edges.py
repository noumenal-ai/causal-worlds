"""Compile a flat causal graph (variables + weighted edges) into a runnable :class:`WorldSpec`.

The inverse of discovery. A learned causal model — e.g. an Operation Model's causal claims, a flat
list of ``parent -> target`` edges each carrying a slope — is *decompressed* into an executable
:class:`WorldSpec` by grouping the edges that share a target into one additive :class:`Mechanism`.
The resulting world runs on the ordinary :class:`~causal_worlds.sample.ScmSubstrate` and control
surface, so a model recovered *from data* can be rolled forward, intervened on with ``do(x)``, and
counterfactually replayed exactly like an authored world.

The mapping is faithful by construction: each edge ``(parent, target, coeff)`` becomes the term
``coeff * parent`` in ``target``'s mechanism, so :func:`~causal_worlds.schema.answer_key` returns
the input edge set and — for a linear world — :func:`~causal_worlds.control.lever_effects` returns
the input slopes' path-sums. What the source model does *not* carry (functional forms beyond the
linear default, the residual noise scale, regime coefficients) is supplied here as an explicit,
declared default, never silently invented (the lossy-model gap).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from causal_worlds.schema import Mechanism, Term, Transform, Variable, WorldSpec, validate


@dataclass(frozen=True, slots=True)
class WeightedEdge:
    """A single directed, weighted causal edge — the flat form a learned model emits.

    The edge contributes the term ``coeff * transform(parent)`` to ``target``'s mechanism (see
    :class:`~causal_worlds.schema.Term`). ``lag`` is how many timesteps back the parent is read
    (``0`` is contemporaneous, the default); ``transform`` defaults to identity — the linear edge a
    bare slope describes.
    """

    parent: str
    target: str
    coeff: float
    lag: int = 0
    transform: Transform = Transform.IDENTITY


def world_from_edges(
    variables: Sequence[Variable],
    edges: Sequence[WeightedEdge],
    *,
    noise_scale: float = 0.3,
) -> WorldSpec:
    """Compile variables + flat weighted edges into a validated, runnable :class:`WorldSpec`.

    Edges sharing a ``target`` are grouped into one additive :class:`Mechanism`
    (``Sigma coeff_i * transform(parent_i) + noise``); a variable that is no edge's ``target`` stays
    an exogenous root. ``noise_scale`` is the residual std given each synthesised mechanism — the
    information the source model abstracted away, declared once and explicitly rather than invented
    per edge.

    Args:
        variables: The world's variables with their roles. The compiled spec needs at least one
            observed controllable and one observed outcome (enforced by
            :func:`~causal_worlds.schema.validate`).
        edges: The flat directed, weighted edges; each becomes one term in its target's mechanism.
        noise_scale: Residual Gaussian std for every synthesised mechanism.

    Returns:
        A :class:`WorldSpec` that has passed :func:`~causal_worlds.schema.validate`.

    Raises:
        ValueError: ``noise_scale`` is not > 0, or two edges share the same
            ``(parent, target, lag, transform)`` — which would silently double that effect while the
            answer-key still shows one edge, breaking the round-trip guarantee.
        DanglingReferenceError: An edge references a variable not declared in ``variables``.
        SpecError: The compiled spec is otherwise invalid (cyclic, explosive, missing a role).
    """
    if noise_scale <= 0:
        msg = f"noise_scale must be > 0, got {noise_scale}"
        raise ValueError(msg)
    terms_by_target: dict[str, list[Term]] = {}
    seen: set[tuple[str, str, int, Transform]] = set()
    for edge in edges:
        key = (edge.parent, edge.target, edge.lag, edge.transform)
        if key in seen:
            msg = (
                f"duplicate edge {edge.parent!r}->{edge.target!r} (lag {edge.lag}, "
                f"{edge.transform.value}); declare each (parent, target, lag, transform) once — "
                f"silently summing them would double the effect the answer-key still shows once"
            )
            raise ValueError(msg)
        seen.add(key)
        terms_by_target.setdefault(edge.target, []).append(
            Term(parent=edge.parent, coeff=edge.coeff, lag=edge.lag, transform=edge.transform),
        )
    mechanisms = tuple(
        Mechanism(target=target, terms=tuple(terms), noise_scale=noise_scale)
        for target, terms in terms_by_target.items()
    )
    spec = WorldSpec(variables=tuple(variables), mechanisms=mechanisms)
    validate(spec)
    return spec
