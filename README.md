[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Actions status](https://github.com/NicholasBehr/hcnn/actions/workflows/ci.yml/badge.svg)](https://github.com/NicholasBehr/hcnn/actions)

# Hard-Constrained Neural Networks in JAX

Research code accompanying **Non-Convex Projection Layers via Sequential Quadratic Programming**, Nicholas Behr's master's thesis at the Automatic Control Laboratory, ETH Zürich (September 2025).

**[Read the thesis](docs/thesis.pdf)**

Advisors: Panagiotis D. Grontas, Antonio Terpin, Efe C. Balta and John Lygeros.

## Overview

How can a neural network's output satisfy nonlinear, nonconvex constraints while remaining differentiable for learning?

This project implements an optimisation-based projection layer in JAX. The forward pass uses sequential quadratic programming (SQP), with operator-splitting quadratic subproblems, to map a proposed output towards a feasible solution. The backward pass uses implicit differentiation of the optimality conditions rather than differentiating through every solver iteration.

For control applications, the layer jointly adjusts proposed states and actions to satisfy system dynamics and constraints.

## Demonstrations and evaluation

The thesis studies projection problems ranging from linear-system rollouts to nonlinear kinematic bicycle trajectories with nonconvex exclusion zones. It evaluates solver residuals and compares vector-Jacobian products with finite-difference estimates, including a two-dimensional nonconvex clover-set experiment.

These experiments demonstrate projection and differentiation behaviour. They do not establish global optimality or safety for arbitrary initialisations and solver settings.

## Installation

The project requires **Python 3.13 or newer** and uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/NicholasBehr/hcnn.git
cd hcnn
uv sync
```

The current project configuration includes JAX CUDA 12 dependencies on Linux. GPU execution requires a compatible NVIDIA driver and hardware.

Launch the notebooks with:

```bash
uv run jupyter lab
```

## Where to start

- [`notebooks/linear_rollout.ipynb`](notebooks/linear_rollout.ipynb): linear-system rollout experiments.
- [`notebooks/linear_rollout_ncvx.ipynb`](notebooks/linear_rollout_ncvx.ipynb): rollouts with nonconvex constraints.
- [`notebooks/slalom.ipynb`](notebooks/slalom.ipynb): slalom example.
- [`notebooks/backwards.ipynb`](notebooks/backwards.ipynb): backward-pass experiments.

The notebooks are research examples; consult their configuration and the thesis for experimental settings. Some contain machine-specific GPU selection (for example, `CUDA_VISIBLE_DEVICES` in `slalom.ipynb`); adjust this for your machine before running.

## Code structure

- [`src/hcnn/solver/`](src/hcnn/solver/): SQP and operator-splitting QP solvers.
- [`src/hcnn/optimization_layer/`](src/hcnn/optimization_layer/): projection and control optimisation layers.
- [`src/hcnn/system_dynamics/`](src/hcnn/system_dynamics/): dynamical-system models.
- [`src/hcnn/system_constraints/`](src/hcnn/system_constraints/): constraint definitions.
- [`src/hcnn/system_costs/`](src/hcnn/system_costs/): cost functions.
- [`src/hcnn/training/`](src/hcnn/training/): learning utilities.
- [`tests/`](tests/): component tests.

Run the tests with:

```bash
uv run pytest
```

## Scope and limitations

This is research software. Nonconvex projections can have multiple local solutions and depend on initialisation. Implicit derivatives require suitable local regularity; changes between solution branches can produce discontinuities. Finite solver tolerances and regularisation affect feasibility and gradient accuracy. The thesis discusses observed discrepancies and convergence limitations.

## Citation

If you use this work, please cite the thesis:

```bibtex
@mastersthesis{behr2025nonconvex,
  author = {Behr, Nicholas},
  title = {Non-Convex Projection Layers via Sequential Quadratic Programming},
  school = {ETH Zurich},
  year = {2025},
  month = sep
}
```

## License

See the repository's [LICENSE](LICENSE) for the applicable terms.
