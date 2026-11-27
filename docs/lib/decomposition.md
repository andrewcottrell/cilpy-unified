# Cooperative Decomposition

Most population-based algorithms, such as PSO, do not scale well to problems
with hundreds or thousands of decision variables. The cooperative solvers in
`cilpy.solver.decomposition` split the variables into groups and give every
group its own sub-solver, which only searches over its own variables.

## How it works

- **Context vector** (`ContextVector`): a full solution that holds the best
  value found so far for every variable.
- **Sub-problem** (`SubSpaceProblem`): looks like an ordinary small problem to
  the sub-solver. To evaluate a partial solution, it inserts the values into a
  copy of the context vector and evaluates the real problem. Because of this,
  any `cilpy` solver (PSO, GA or DE) can be used as a sub-solver without changes.
- **Coordinator** (`CooperativeDecompositionSolver`): runs one iteration as follows:
    1. Evaluate the current context vector.
    2. For every group in turn:
        - re-evaluate the sub-solver's stored fitness values, because the
          context may have changed since its last step (`refresh_evaluations`);
        - offer the context's own values for the group as a known good
          solution (`accept_best`), so the sub-solver's best is never worse
          than the context;
        - let the sub-solver take one step;
        - write the sub-solver's best into the context, so the next group
          already benefits from it.
    3. Call `_post_step_hook`, where the four solvers differ.

The coordinator looks like a plain `Solver` to the `ExperimentRunner`, and its
result is the context vector.

## The four solvers

| Solver | Starts with | What happens during the run |
|---|---|---|
| `StandardCooperativeSolver` (CPSO) | `num_groups` groups | Nothing, the groups never change. |
| `DynamicRandomGroupingSolver` (DRG) | `num_groups` groups | When the trigger fires, all variables are regrouped (randomly by default). |
| `DecompositionCooperativeSolver` (DCPSO) | 1 group | When the trigger fires, every group is split, until each group holds one variable. |
| `MergingCooperativeSolver` (MCPSO) | 1 group per variable | When the trigger fires, groups are merged, until one group holds all variables. |

DRG, DCPSO and MCPSO only write a group's new best into the context when it does
not make the context worse (`only_accept_improvements`).

When the groups change, the positions, velocities and personal bests are saved
per variable and copied into the new sub-solvers, so the search continues
instead of restarting. Building the new sub-solvers costs no evaluations. This
needs every sub-solver to have the same population size.

After a split, DCPSO gives PSO particles new random velocities, and GA or DE
individuals (except the first) new random positions.

## Strategies

Every solver is built from interchangeable strategies.

**Groupings** decide how the variables are put into groups:

- `StaticEvenGrouping`: in order, the last group takes the remainder.
- `RandomGrouping`: shuffled, then split into equal groups.
- `RandomGroupSizeGrouping(group_sizes)`: picks a random group size at every
  call, so the number of groups can change. It ignores `num_groups`.

**Triggers** decide when the groups change:

- `FixedIntervalTrigger(interval)`: every `interval` iterations.
- `EvaluationIntervalTrigger(interval)`: every `interval` fitness evaluations.
- `StagnationTrigger(patience, tolerance)`: after `patience` iterations without
  an improvement of more than `tolerance` times the best fitness.
- `NeverTrigger`: never.

**Splitters** (DCPSO) decide how a group is split:

- `EvenSplitter(split_factor)`: cuts each group in order.
- `RandomSplitter(split_factor)`: shuffles each group first.

**Mergers** (MCPSO) decide which groups are merged:

- `EvenMerger(merge_factor)`: merges neighbouring groups.
- `RandomMerger(merge_factor)`: merges randomly chosen groups.

If DCPSO or MCPSO get no trigger, they build one from `max_iterations` or
`max_evaluations` that spaces the splits or merges evenly over the budget.
`restructure_interval(budget, dimension, factor)` gives the interval:

```
interval = budget / (1 + log(dimension) / log(factor))
```

## Example

```python
from cilpy.problem.unconstrained import Rastrigin
from cilpy.runner import ExperimentRunner
from cilpy.solver.decomposition import DynamicRandomGroupingSolver, FixedIntervalTrigger
from cilpy.solver.ga import GA

runner = ExperimentRunner(
    problems=[Rastrigin(dimension=500, domain=(-5.12, 5.12))],
    solver_configurations=[{
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "DRG_GA",
            "num_groups": 10,
            "sub_solver_class": GA,
            "sub_solver_params": {
                "name": "sub_ga",
                "population_size": 10,
                "crossover_rate": 0.9,
                "mutation_rate": 0.1,
            },
            "regroup_trigger": FixedIntervalTrigger(interval=50),
        },
    }],
    num_runs=5,
    max_evaluations=500000,
)
runner.run_experiments()
```

## Evaluation budget

One iteration of a cooperative solver uses many more fitness evaluations than
one iteration of a plain solver. With PSO sub-solvers of `s` particles and `K`
groups, an iteration costs `1 + 2 * s * K` evaluations. Comparing solvers on an
iteration budget is therefore unfair, so `ExperimentRunner` also accepts
`max_evaluations`. Every evaluation goes through a `CountingProblem`, and the
runner's own measurements are made inside `not_counting()` so they do not use up
the budget.

See `examples/cpso_examples.py`, `drg_examples.py`, `dcpso_examples.py` and
`mcpso_examples.py` for every grouping, trigger, splitter and merger in use.
