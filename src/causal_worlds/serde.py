"""The boundary model: a pydantic mirror of :class:`WorldSpec` for LLM I/O and persistence.

The core IR (:mod:`causal_worlds.schema`) is frozen dataclasses — small, hashable, side-effect-free.
At the boundary we need validation with bounded re-ask (the LLM author) and stable JSON (the on-disk
artifact). That is pydantic's job, so we keep *one* boundary model here and convert both ways:
parse-don't-validate inbound (:meth:`WorldSpecModel.to_spec`), serialize outbound
(:meth:`WorldSpecModel.from_spec`).

Field descriptions are written for the *author* LLM: they are the contract the model fills in.
"""

from pydantic import BaseModel, Field

from causal_worlds.schema import Claim, Mechanism, Role, Term, Transform, Variable, WorldSpec


class TermModel(BaseModel):
    """A single term ``coeff * transform(parent)`` feeding a mechanism."""

    parent: str = Field(description="Name of the parent variable that drives the target.")
    coeff: float = Field(description="Coefficient; sign encodes the direction of effect.")
    lag: int = Field(
        default=0,
        ge=0,
        description="Timesteps back the parent is read: 0 = contemporaneous, >=1 = a lagged edge "
        "(a lagged self-reference is autoregression). Use 0 unless the world is temporal.",
    )
    transform: Transform = Field(
        default=Transform.IDENTITY,
        description="Elementwise nonlinearity applied to the parent before scaling: identity "
        "(linear; the default), square, cube, tanh, relu, or abs. Use identity unless the "
        "mechanism is genuinely nonlinear.",
    )

    def to_term(self) -> Term:
        """Convert to the frozen core :class:`Term`."""
        return Term(parent=self.parent, coeff=self.coeff, lag=self.lag, transform=self.transform)


class VariableModel(BaseModel):
    """A variable in the world; hidden variables drive sampling but are not emitted as data."""

    name: str = Field(description="Unique short identifier, e.g. 'price' or 'demand'.")
    role: Role = Field(
        description=(
            "controllable (a lever an operator sets), observable (measured, not set), "
            "disturbance (exogenous shock), or outcome (the KPI). A world needs at least one "
            "observed controllable and one observed outcome."
        )
    )
    hidden: bool = Field(
        default=False,
        description="True for a latent common cause (confounder) that is NOT emitted in the data.",
    )

    def to_variable(self) -> Variable:
        """Convert to the frozen core :class:`Variable`."""
        return Variable(name=self.name, role=self.role, hidden=self.hidden)


class MechanismModel(BaseModel):
    """How one variable is generated: a linear function of parents plus Gaussian noise."""

    target: str = Field(description="The variable this mechanism generates.")
    terms: list[TermModel] = Field(
        default_factory=list,
        description="Parent terms in the default regime; empty means an exogenous root (noise).",
    )
    noise_scale: float = Field(
        default=0.3, description="Standard deviation of the additive Gaussian noise (> 0)."
    )
    regime: str | None = Field(
        default=None,
        description=(
            "Optional name of a binary variable; when set, 'regime_terms' apply on rows where it "
            "is truthy and 'terms' on the rest — this is how an effect flips or rescales by regime."
        ),
    )
    regime_terms: list[TermModel] | None = Field(
        default=None,
        description="Parent terms used when 'regime' is truthy; required iff 'regime' is set.",
    )

    def to_mechanism(self) -> Mechanism:
        """Convert to the frozen core :class:`Mechanism`."""
        regime_terms = (
            tuple(term.to_term() for term in self.regime_terms)
            if self.regime_terms is not None
            else None
        )
        return Mechanism(
            target=self.target,
            terms=tuple(term.to_term() for term in self.terms),
            noise_scale=self.noise_scale,
            regime=self.regime,
            regime_terms=regime_terms,
        )


class ClaimModel(BaseModel):
    """Which observed variables embody the prompt's causal question (the user's cause X and Y)."""

    cause: str = Field(
        description="The observed variable that embodies the prompt's cause (X) — what the user "
        "asked about as the driver. Must be an observed (non-hidden) variable name in this spec."
    )
    outcome: str = Field(
        description="The observed variable that embodies the prompt's outcome (Y) — the effect the "
        "user cares about (usually the OUTCOME-role KPI). Must be a different observed variable."
    )

    def to_claim(self) -> Claim:
        """Build the frozen core :class:`Claim`."""
        return Claim(cause=self.cause, outcome=self.outcome)


class WorldSpecModel(BaseModel):
    """The boundary form of a :class:`WorldSpec`: validated LLM output and the persisted shape."""

    variables: list[VariableModel] = Field(description="Every variable, including hidden ones.")
    mechanisms: list[MechanismModel] = Field(
        description="One mechanism per non-root variable (roots may be omitted)."
    )
    claim: ClaimModel | None = Field(
        default=None,
        description="If the operation was described as a causal question (does X affect Y?), name "
        "the observed variables embodying the prompt's cause and outcome. Omit if no focus pair.",
    )

    def to_spec(self) -> WorldSpec:
        """Build the frozen core :class:`WorldSpec` (does not run semantic validation)."""
        return WorldSpec(
            variables=tuple(variable.to_variable() for variable in self.variables),
            mechanisms=tuple(mechanism.to_mechanism() for mechanism in self.mechanisms),
            claim=self.claim.to_claim() if self.claim is not None else None,
        )

    @classmethod
    def from_spec(cls, spec: WorldSpec) -> "WorldSpecModel":
        """Build the boundary model from a frozen core :class:`WorldSpec`."""
        return cls(
            variables=[
                VariableModel(name=variable.name, role=variable.role, hidden=variable.hidden)
                for variable in spec.variables
            ],
            mechanisms=[
                MechanismModel(
                    target=mechanism.target,
                    terms=[
                        TermModel(parent=t.parent, coeff=t.coeff, lag=t.lag, transform=t.transform)
                        for t in mechanism.terms
                    ],
                    noise_scale=mechanism.noise_scale,
                    regime=mechanism.regime,
                    regime_terms=(
                        [
                            TermModel(
                                parent=t.parent, coeff=t.coeff, lag=t.lag, transform=t.transform
                            )
                            for t in mechanism.regime_terms
                        ]
                        if mechanism.regime_terms is not None
                        else None
                    ),
                )
                for mechanism in spec.mechanisms
            ],
            claim=(
                ClaimModel(cause=spec.claim.cause, outcome=spec.claim.outcome)
                if spec.claim is not None
                else None
            ),
        )


def spec_to_json(spec: WorldSpec) -> str:
    """Serialize a spec to stable, indented JSON (the persisted ``spec.json``)."""
    return WorldSpecModel.from_spec(spec).model_dump_json(indent=2)


def spec_from_json(text: str) -> WorldSpec:
    """Parse a spec from JSON; raises ``pydantic.ValidationError`` on a malformed payload."""
    return WorldSpecModel.model_validate_json(text).to_spec()
