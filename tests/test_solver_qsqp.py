import jax
import jax.numpy as jnp
import pytest
from flax import struct

from hcnn.optimization.optimization import OptimizationProblem
from hcnn.optimization.solver_qsqp import SolverQSQP

# ruff: noqa: N806, matrix uppercase reference cases


@struct.dataclass
class Problem(OptimizationProblem):
    _x0: jax.Array

    @property
    def x0(self) -> jax.Array:
        return self._x0

    @staticmethod
    def factory(x0: jax.Array) -> "Problem":
        return Problem(
            cost_linear=jnp.array([0.0, 0.0]),
            lower_bound=jnp.array([1.0, -jnp.inf]),
            upper_bound=jnp.array([jnp.inf, 2.0]),
            _x0=x0,
        )

    ## cost
    def cost_qvp(self, x: jax.Array) -> jax.Array:
        # Q = jnp.eye(2)
        return x

    ## constraints
    def constraints(self, x: jax.Array) -> jax.Array:
        test_constraints = jnp.dot(x, x)
        return jnp.tile(test_constraints, 2)


@pytest.fixture
def problem():
    return Problem.factory(jnp.array([2.0, 2.0]))


@pytest.fixture
def variables(problem):
    return SolverQSQP.Variables.factory(problem)


def randomize_variables(rng, variables) -> SolverQSQP.Variables:
    # Flatten the structure into leaves
    leaves, treedef = jax.tree_util.tree_flatten(variables)

    # Split RNG for each leaf
    keys = jax.random.split(rng, len(leaves))

    # Randomize each leaf, preserving shape
    new_leaves = [jax.random.normal(k, v.shape) for k, v in zip(keys, leaves, strict=False)]

    # Rebuild the struct with randomized leaves
    new_variables = jax.tree_util.tree_unflatten(treedef, new_leaves)
    return new_variables


def check_shapes_match(tree_a, tree_b):
    leaves_a, treedef_a = jax.tree_util.tree_flatten(tree_a)
    leaves_b, treedef_b = jax.tree_util.tree_flatten(tree_b)

    if treedef_a != treedef_b:
        return False

    for _, (a, b) in enumerate(zip(leaves_a, leaves_b, strict=False)):
        if a.shape != b.shape:
            return False

    return True


def test_step_variables(problem: Problem, variables: SolverQSQP.Variables):
    absolute_tolerance = 1e-5

    ## jit compilable
    jit_step = jax.jit(SolverQSQP.Variables.update_primal_dual)

    for i in range(25):
        # radomize variables
        rng = jax.random.PRNGKey(i)
        variables = randomize_variables(rng, variables)

        # ensure valid variables
        variables.replace(
            alpha=jnp.clip(variables.alpha, 1e-8, 2 - 1e-8),  # (0, 2)
            sigma=jnp.abs(variables.sigma) + 1e-8,  # >0
            rho_vec=jnp.abs(variables.rho_vec) + 1e-8,  # > 0
        )
        new_variables = jit_step(self=variables, problem=problem, atol=absolute_tolerance)

        ## check shapes and types
        assert check_shapes_match(variables, new_variables)

        Q = jnp.eye(2)
        R = jnp.diag(variables.rho_vec)
        h = problem.constraints(variables.x_tilde)
        J = jax.jacobian(problem.constraints)(variables.x_tilde)

        ## check x_tilde
        lhs = (Q + variables.sigma * jnp.eye(2) + J.T @ R @ J) @ new_variables.x_tilde
        rhs = (
            variables.sigma * variables.x
            - problem.cost_linear
            + J.T @ (R @ (variables.z - h + J @ variables.x_tilde) - variables.y)
        )
        assert jnp.allclose(lhs, rhs, atol=absolute_tolerance)

        ## check z_tilde
        lhs = new_variables.z_tilde
        rhs = h + J @ (new_variables.x_tilde - variables.x_tilde)
        assert jnp.allclose(lhs, rhs, atol=absolute_tolerance)

        ## check x
        lhs = new_variables.x
        rhs = variables.alpha * new_variables.x_tilde + (1 - variables.alpha) * variables.x
        assert jnp.allclose(lhs, rhs, atol=absolute_tolerance)

        ## check z
        R_inv = jnp.linalg.inv(R)
        lhs = new_variables.z
        rhs = jnp.clip(
            (
                variables.alpha * new_variables.z_tilde
                + (1 - variables.alpha) * variables.z
                + R_inv @ variables.y
            ),
            problem.lower_bound,
            problem.upper_bound,
        )
        assert jnp.allclose(lhs, rhs, atol=absolute_tolerance)

        ## check y
        lhs = new_variables.y
        rhs = variables.y + R @ (
            variables.alpha * new_variables.z_tilde
            + (1 - variables.alpha) * variables.z
            - new_variables.z
        )
        assert jnp.allclose(lhs, rhs, atol=absolute_tolerance)

        ## check primal residuals ##
        lhs = new_variables.primal_residuals
        rhs = h + J @ (new_variables.x - variables.x_tilde) - new_variables.z
        assert jnp.allclose(lhs, rhs, atol=absolute_tolerance)

        ## check dual residuals ##
        lhs = new_variables.dual_residuals
        rhs = problem.cost_qvp(new_variables.x) + problem.cost_linear + J.T @ new_variables.y
        assert jnp.allclose(lhs, rhs, atol=absolute_tolerance)


def test_solve_start_feasible():
    problem = Problem.factory(jnp.array([2.0, 2.0]))
    var = SolverQSQP(atol=1e-6)._solve_full(problem)
    x = var.x
    constraints = problem.constraints(x)

    assert jnp.all(problem.lower_bound - 1e-6 <= constraints)
    assert jnp.all(constraints <= problem.upper_bound + 1e-6)


def test_solve_start_infeasible():
    problem = Problem.factory(jnp.array([0.5, 0.5]))
    var = SolverQSQP(atol=1e-5)._solve_full(problem)
    x = var.x

    assert jnp.all(problem.lower_bound - 1e-5 <= problem.constraints(x))
    assert jnp.all(problem.constraints(x) <= problem.upper_bound + 1e-5)
