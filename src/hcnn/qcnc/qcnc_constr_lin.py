import jax
import jax.numpy as jnp
from flax import struct
from jaxopt import BoxOSQP

from hcnn.qcnc.qcnc import QCNC, OptimizationProblem, OptimizationVariable


@struct.dataclass
class ConstraintLinearization(QCNC):
    problem: OptimizationProblem

    def constraints_full(self, x: OptimizationVariable) -> jax.Array:
        return jnp.concatenate(
            (
                self.problem.constraints_equality(x),
                self.problem.constraints_inequality(x),
            )
        )

    def bounds_full(self, x: OptimizationVariable) -> tuple[jax.Array, jax.Array]:
        bound_equality = self.problem.bound_equality(x)
        l_inequality, u_inequality = self.problem.bounds_inequality(x)

        return (
            # lower
            jnp.concatenate((bound_equality, l_inequality)),
            # upper
            jnp.concatenate((bound_equality, u_inequality)),
        )

    def solve(self, problem: OptimizationProblem) -> OptimizationVariable:
        x = self.problem.warm_start()

        lower, upper = self.bounds_full(x)

        for _ in range(5):

            def matvec_q(params_opt, x):
                return params_opt.problem.cost_matvec_quadratic(x)

            hx, grad_hx = jax.linearize(self.constraints_full, x)

            def matvec_a(params_eq, d):
                grad_hx = params_eq
                return grad_hx(d)

            bound_shift = grad_hx(x) - hx
            hyper_params = dict(
                params_obj=(self, self.problem.cost_linear),
                params_eq=grad_hx,
                params_ineq=(lower + bound_shift, upper + bound_shift),
            )
            osqp = BoxOSQP(matvec_Q=matvec_q, matvec_A=matvec_a, tol=1e-6)
            sol = osqp.run(None, **hyper_params).params
            x = sol.primal[0]
        return x
