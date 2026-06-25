"""Grade a controller by regret against the by-construction optimal policy (Stage 2, no API key).

The same worlds are a *control* benchmark: choose lever (controllable) values to maximise an
objective. Because the mechanisms are declared, the best the levers can do is computable — so a
policy is scored by **regret** against that declared optimum, with no external data needed.

    uv run python examples/03_benchmark_a_controller.py

Expected output:

    optimal policy (declared): {'price': 0.0}
    your policy {'price': 3.0}  -> regret 4.502   (0 = optimal play)
    optimal policy itself       -> regret 0.000
"""

from causal_worlds import default_objective, grade_control, optimal_policy, worlds

SEED = 7
spec = worlds.get("coffee")
objective = default_objective(spec)  # controllables raise the outcome KPI under a quadratic cost

best = optimal_policy(spec, objective)
print(f"optimal policy (declared): { {k: round(v, 2) for k, v in best.items()} }")

your_policy = {"price": 3.0}  # push the lever hard — overshoots the quadratic-cost optimum
report = grade_control(spec, objective, your_policy, seed=SEED)
print(f"your policy {your_policy}  -> regret {report.regret:.3f}   (0 = optimal play)")

# Playing the declared optimum scores ~zero regret (the answer-key is correct by construction):
optimal_report = grade_control(spec, objective, best, seed=SEED)
print(f"optimal policy itself       -> regret {optimal_report.regret:.3f}")
