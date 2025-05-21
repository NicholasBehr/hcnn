import jax
from flax import struct
from jaxopt import BoxOSQP

from .optimization import OptimizationProblemBase, OptimizationSolverBase, PhiType


@struct.dataclass
class OptimizationSolverLinearize(OptimizationSolverBase[PhiType]):
    num_iter: int = 10

    def solve(self, problem: OptimizationProblemBase, phi: PhiType) -> jax.Array:
        def body_fun(i, x):
            hx, grad_hx = jax.linearize(lambda x: problem.constraint_fn(x, phi), x)
            bound_shift = grad_hx(x) - hx

            lower = problem.lower_bound + bound_shift
            upper = problem.upper_bound + bound_shift

            osqp = BoxOSQP(
                matvec_Q=lambda problem, x: problem.cost_qvp(x),
                matvec_A=lambda grad_hx, x: grad_hx(x),
                tol=1e-4,
            )

            sol = osqp.run(
                init_params=None,
                params_obj=(problem, problem.cost_linear),
                params_eq=grad_hx,
                params_ineq=(lower, upper),
            ).params

            return sol.primal[0]

        return jax.lax.fori_loop(
            init_val=problem.get_x0,
            body_fun=body_fun,
            lower=0,
            upper=self.num_iter,
        )
