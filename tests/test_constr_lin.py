import jax
import jax.numpy as jnp
from flax import struct

from hcnn.qcnc.qcnc import OptimizationProblem, OptimizationVariable
from hcnn.qcnc.qcnc_constr_lin import ConstraintLinearization


@struct.dataclass
class OptimizationVariableTest(OptimizationVariable):
    x: jax.Array


@struct.dataclass
class OptimizationProblemTest(OptimizationProblem):
    def warm_start(self) -> OptimizationVariableTest:
        return OptimizationVariableTest(
            max_ineq_relaxation=0.0, ineq_relaxations=jnp.zeros((1)), x=jnp.array([5.0, 5.0])
        )

    def cost_matvec_quadratic(self, x: OptimizationVariableTest) -> OptimizationVariableTest:
        return OptimizationVariableTest(
            max_ineq_relaxation=0.0,
            ineq_relaxations=jnp.zeros_like(x.ineq_relaxations),
            x=x.x,
        )

    cost_linear: jax.Array = struct.field(
        default_factory=lambda: OptimizationVariableTest(
            max_ineq_relaxation=1.0,
            ineq_relaxations=jnp.zeros((1)),
            x=jnp.zeros((2)),
        )
    )

    def constraints_inequality(self, x: OptimizationVariableTest) -> jax.Array:
        base_inequality = super().constraints_inequality(x)
        test_inequality = jnp.dot(x.x, x.x) + x.ineq_relaxations
        return jnp.concatenate((base_inequality, test_inequality))

    def bounds_inequality(self, x: OptimizationVariableTest) -> tuple[jax.Array, jax.Array]:
        base_lower, base_upper = super().bounds_inequality(x)
        return (
            jnp.concatenate((base_lower, jnp.array([3.0]))),
            jnp.concatenate((base_upper, jnp.array([5.0]))),
        )


def test_solve():
    problem = OptimizationProblemTest()
    optimizer = ConstraintLinearization(problem)

    x = optimizer.solve(problem)

    lower, upper = optimizer.bounds_full(x)

    assert jnp.all(lower - 1e-6 <= optimizer.constraints_full(x))
    assert jnp.all(optimizer.constraints_full(x) <= upper + 1e-6)
