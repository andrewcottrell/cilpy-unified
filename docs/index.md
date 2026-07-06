# Welcome to `cilpy`

`cilpy` (which stands for "computational intelligence library for Python") is an
extensible Python library, designed to help research and experiment with
nature-inspired algorithms (NIAs).

## The Guiding Mission for `cilpy`
The goal for `cilpy` is to be a useful tool for people implementing and studying
NIAs. It should

The primary goal of `cilpy` is to provide a simple and versatile framework for
studying a wide range of optimization problems and algorithms. The library
should:

1. Allow users to implement a wide variety of optimization problems, including:
    - Single- and multi-objective problems  
    - Constrained and unconstrained problems  
    - Problems with static and dynamic objectives and constraints  
2. Allow users to implement a wide variety of nature-inspired algorithms.
3. Make it easy to specify and repeat experiments.

## An Overview of `cilpy`
`cilpy` serves its goal by clearly separating the concerns of defining a
problem, defining a solving algorithm, and specifying an experiment. For each of
these components, we find which abstractions strike the right balance between
simplicity and utility through iteration. The components of this library are
thus:

1. **`cilpy.problem`**: This component is used to specify optimization problems,
   including objective function(s), search space, and constraints.
2. **`cilpy.solver`**: This component is used to specify solvers (NIAs), that
   search for solutions to a given problem.
3. **`cilpy.compare`**: This component will provide the tools for statistical
   analysis and visualization to compare the performance of different solvers on
   various problems.
4. **`cilpy.runner`**: This component is used to orchestrate the interaction
   between components, run experiments, and log results.

By enforcing separation between these parts through well-defined interfaces,
`cilpy` allows users to seamlessly swap out components.

- To test a new algorithm: Implement the `Solver` interface, and you can
  immediately benchmark it against all existing problems.
- To try a new benchmark problem: Implement the `Problem` interface, and any
  existing solver can be used to tackle it.

## How `cilpy` is Used
The `ExperimentRunner` in `cilpy.runner` orchestrates the entire experimental
process. After being provided a list of problems to solve, solvers and their
configurations to test, and comparisons to perform on the algorithm, the runner
orchestrates all further action, and performs the experiment.

The typical workflow looks like this:

1. **Define Problems**: Instantiate or create custom classes for the
   optimization problems you want to investigate.
2. **Configure Solvers**: Create a list of dictionaries, where each dictionary
   specifies a solver's class (e.g., `GA`, `PSO`) and its parameters (e.g.,
   `population_size`, `mutation_rate`).
3. **Specify Comparisons**: TODO
4. **Configure the Runner**: Initialize the `ExperimentRunner` with your
   problems, solver configurations, comparisons to perform, and experiment
   parameters (like the number of runs and iterations).
5. **Execute**: Call the `run_experiments()` method. The runner will
   systematically pair each solver with each problem, execute the specified
   number of independent runs, and save the results for later analysis.

This declarative approach makes experiments easy to define, reproduce, and
modify.

## How to Use This Documentation

This documentation is structured to guide you, whether you are using the library
for the first time or developing new components for it.

* **[Quickstart](quickstart.md)**: A quick guide on how to implement the core
  interfaces and run an experiment.
* **[Included](lib/index.md)**: Discusses the problems, solvers, and analysis
  tools that have been incorporated into `cilpy`.
* **[API Reference](api/index.md)**: A detailed, technical reference for all the
  core classes and functions in the library.
* **[Developer Guide](dev/index.md)**: For those who want to contribute to
  `cilpy`, this contains information on our coding standards, testing
  procedures, and how to get involved.
