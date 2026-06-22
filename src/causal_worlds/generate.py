"""The author->gate->admit loop: turn a natural-language prompt into an admitted world.

An :class:`Author` proposes a :class:`WorldSpec`; the gate pipeline judges it; on failure the gate's
reason is turned into actionable feedback and the author is re-asked, up to ``max_attempts`` times.
A world that never passes raises :class:`NotAdmittedError` — we never silently ship an unjustified
world. This is the imperative shell; the gates and the substrate it calls are the functional core.
"""

from dataclasses import dataclass

from causal_worlds.errors import CausalWorldsError
from causal_worlds.gates import GateReport, run_gates
from causal_worlds.protocols import Author, Discoverer, Judge
from causal_worlds.schema import WorldSpec

_DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class AdmittedWorld:
    """A world that passed every gate, with the report that admitted it."""

    prompt: str
    spec: WorldSpec
    report: GateReport
    attempts: int


class NotAdmittedError(CausalWorldsError):
    """No proposed spec passed the gates within the attempt budget."""

    def __init__(self, prompt: str, attempts: int, last: GateReport | None) -> None:
        """Record the prompt, the attempts spent, and the final gate report."""
        self.prompt = prompt
        self.attempts = attempts
        self.last = last
        reason = last.reason if last is not None else "no attempts made"
        super().__init__(f"not admitted after {attempts} attempt(s): {reason}")


def _feedback(report: GateReport) -> str:
    """Turn a failing gate report into a concrete instruction for the next author attempt."""
    hints = {
        "T1": "Make the graph acyclic, declare every referenced variable, and include at least "
        "one observed controllable lever and one observed outcome.",
        "T2": "Give every variable real variance — avoid mechanisms that collapse to a constant.",
        "T3": "Strengthen the causal effects relative to the noise so the structure is "
        "recoverable, and keep the controllable lever genuinely influential.",
        "T4 unfaithful": "Represent the described operation more faithfully — keep the variables "
        "and effects the prompt implies.",
        "T4 cliché": "Make the structure less guessable from the variable names: add a hidden "
        "confounder, or a regime that flips or rescales an effect.",
    }
    for prefix, hint in hints.items():
        if report.reason.startswith(prefix):
            return f"Your previous attempt failed: {report.reason}. {hint}"
    return f"Your previous attempt failed: {report.reason}. Please revise the world."


def generate(  # noqa: PLR0913 — a public entrypoint; the extra params are all keyword-only knobs
    prompt: str,
    *,
    author: Author,
    discoverer: Discoverer | None = None,
    judge: Judge | None = None,
    seed: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> AdmittedWorld:
    """Author a world from ``prompt`` and admit it through the gates, re-asking on failure.

    Args:
        prompt: The natural-language operation description.
        author: The world author (an LLM behind an adapter, or a fake).
        discoverer: The reference grader (defaults to the interventional-CI discoverer).
        judge: An independent judge; enables the T4 anti-cliché gate when supplied.
        seed: Seeds sampling, grading, and the random-graph null.
        max_attempts: How many times to (re-)ask the author before giving up.

    Returns:
        The admitted world and the gate report that admitted it.

    Raises:
        NotAdmittedError: No proposed spec passed the gates within ``max_attempts``.
    """
    feedback: str | None = None
    last: GateReport | None = None
    for attempt in range(1, max_attempts + 1):
        spec = author.author(prompt, feedback=feedback)
        report = run_gates(spec, discoverer=discoverer, seed=seed, judge=judge, prose=prompt)
        if report.admitted:
            return AdmittedWorld(prompt=prompt, spec=spec, report=report, attempts=attempt)
        last = report
        feedback = _feedback(report)
    raise NotAdmittedError(prompt, max_attempts, last)
