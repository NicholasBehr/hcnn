import jax
from flax import struct
from jaxopt import BoxOSQP

from hcnn.optimization.optimization import (
    OptimizationProblem,
    OptimizationSolver,
)


@struct.dataclass
class SolverLinearize(OptimizationSolver):
    @staticmethod
    def solve(problem: OptimizationProblem, verbose: bool = False) -> jax.Array:
        def body_fun(i, x):
            # linearize constraints at current x
            hx, grad_hx = jax.linearize(problem.constraints, x)

            # shift bounds, such that constraint function is linear (not affine)
            bound_shift = grad_hx(x) - hx

            # setup QP
            hyper_params = dict(
                params_obj=(problem, problem.cost_linear),
                params_eq=grad_hx,
                params_ineq=(
                    problem.lower_bound + bound_shift,
                    problem.upper_bound + bound_shift,
                ),
            )
            osqp = BoxOSQP(
                matvec_Q=lambda problem, x: problem.cost_qvp(x),
                matvec_A=lambda grad_hx, x: grad_hx(x),
                tol=1e-3,
            )

            # solve QP
            sol = osqp.run(None, **hyper_params).params

            return sol.primal[0]

        return jax.lax.fori_loop(
            init_val=problem.x0,
            body_fun=body_fun,
            lower=0,
            upper=25,
        )
