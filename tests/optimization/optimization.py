import jax
import jax.numpy as jnp
from flax import struct

from hcnn.optimization.optimization import OptimizationProblemBase, OptimizationSolverBase

# @struct.dataclass
# class ConvexProblem(OptimizationProblemBase[jax.Array]):
#     def get_x0(self, phi: jax.Array):
#         return jnp.zeros((2 * dim,))

#     def constraint_fn(self, x: jax.Array, phi: jax.Array) -> jax.Array:
#         return jnp.hstack([jnp.zeros((dim, dim)), -jnp.eye(dim)]) @ x

#     @classmethod
#     def factory(cls, jax_key):
#         cost_quad_diag = jnp.ones(2 * dim)
#         cost_linear = jnp.zeros(2 * dim)
#         phi = jax.random.normal(jax.random.fold_in(jax_key, 1), (dim, dim))
#         lower_bound = -jnp.inf * jnp.ones((dim,))
#         upper_bound = jax.random.uniform(jax.random.fold_in(jax_key, 2), (dim,), minval=-10, maxval=10)

#         return ConvexProblem(
#             cost_quad_diag=cost_quad_diag,
#             cost_linear=cost_linear,
#             lower_bound=lower_bound,
#             upper_bound=upper_bound,
#         ), phi


## unconstrained problem
@struct.dataclass
class UnconstrainedProblem(OptimizationProblemBase[jax.Array]):
    """
    x_star = argmin 1/2 x.T diag(q) x + c.T x
    """

    def get_x0(self, phi: jax.Array):
        return jnp.zeros_like(self.cost_linear)

    def constraint_fn(self, x: jax.Array, phi: jax.Array) -> jax.Array:
        return jnp.zeros_like(x)

    @classmethod
    def factory(cls, cost_quad_diag, cost_linear):
        phi = jnp.zeros((1,))
        lower_bound = jnp.zeros_like(cost_linear)
        upper_bound = jnp.zeros_like(cost_linear)

        return UnconstrainedProblem(
            cost_quad_diag=cost_quad_diag,
            cost_linear=cost_linear,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ), phi


def test_solve_unconstrained(solver: OptimizationSolverBase):
    dimension = 5
    for i in range(30):
        # create
        keys = jax.random.split(jax.random.key(i), 2)
        cost_quad_diag = jax.random.normal(keys[0], (dimension,))
        cost_linear = jax.random.normal(keys[1], (dimension,))
        problem, phi = UnconstrainedProblem.factory(cost_quad_diag, cost_linear)

        # solve
        x_star = solver.solve(problem, phi)

        # verify
        x_star_analytic = -cost_linear / cost_quad_diag
        assert jnp.allclose(
            x_star,
            x_star_analytic,
            atol=1e-4,
        )


def test_vjp_unconstrained(solver: OptimizationSolverBase):
    dimension = 5
    for i in range(30):
        # create
        keys = jax.random.split(jax.random.key(i), 2)
        cost_quad_diag = jax.random.normal(keys[0], (dimension,))
        cost_linear = jax.random.normal(keys[1], (dimension,))
        problem, phi = UnconstrainedProblem.factory(cost_quad_diag, cost_linear)

        # solve
        def optimization(problem, phi):
            return solver.solve(problem, phi)

        jac_prob, jac_phi = jax.jacrev(optimization, argnums=(0, 1))(problem, phi)

        # verify
        assert jnp.allclose(
            jac_prob.cost_quad_diag,
            jnp.diag(cost_linear / (cost_quad_diag**2)),
            atol=1e-4,
        )
        assert jnp.allclose(
            jac_prob.cost_linear,
            jnp.diag(-1 / (cost_quad_diag)),
            atol=1e-4,
        )
        assert jnp.allclose(
            jac_prob.lower_bound,
            jnp.zeros((dimension, dimension)),
            atol=1e-4,
        )
        assert jnp.allclose(
            jac_prob.upper_bound,
            jnp.zeros((dimension, dimension)),
            atol=1e-4,
        )
        assert jnp.allclose(
            jac_phi,
            jnp.zeros((1, 1)),
            atol=1e-4,
        )


## box projection
@struct.dataclass
class BoxProjectionProblem(OptimizationProblemBase[jax.Array]):
    """
    x_star = argmin |x - origin|_2^2
            s.t. l <= x <= u
    """

    def get_x0(self, phi: jax.Array):
        return jnp.zeros_like(self.cost_linear)

    def constraint_fn(self, x: jax.Array, phi: jax.Array) -> jax.Array:
        return x

    @classmethod
    def factory(cls, origin, lower_bound, upper_bound):
        cost_quad_diag = jnp.ones_like(origin)
        cost_linear = -origin
        phi = jnp.zeros((1,))

        return BoxProjectionProblem(
            cost_quad_diag=cost_quad_diag,
            cost_linear=cost_linear,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ), phi


def test_solve_box_projection(solver: OptimizationSolverBase):
    dimension = 5
    for i in range(30):
        # create
        keys = jax.random.split(jax.random.key(i), 3)
        origin = jax.random.normal(keys[0], (dimension,))
        lower_bound = jax.random.uniform(keys[1], (dimension,), minval=-5, maxval=5)
        upper_bound = jax.random.uniform(keys[2], (dimension,), minval=lower_bound, maxval=10)
        problem, phi = BoxProjectionProblem.factory(origin, lower_bound, upper_bound)

        # solve
        x_star = solver.solve(problem, phi)

        # verify
        x_star_analytic = jnp.clip(origin, lower_bound, upper_bound)
        assert jnp.allclose(
            x_star,
            x_star_analytic,
            atol=1e-4,
        )


def test_vjp_box_projection(solver: OptimizationSolverBase):
    dimension = 2
    for i in range(30):
        # create
        keys = jax.random.split(jax.random.key(i), 3)
        origin = jax.random.normal(keys[0], (dimension,))
        lower_bound = jax.random.uniform(keys[1], (dimension,), minval=-5, maxval=5)
        upper_bound = jax.random.uniform(keys[2], (dimension,), minval=lower_bound, maxval=10)

        # solve
        def box_projection(origin, lower_bound, upper_bound):
            problem, phi = BoxProjectionProblem.factory(origin, lower_bound, upper_bound)
            return solver.solve(problem, phi)

        jac_o, jac_lb, jac_ub = jax.jacrev(box_projection, argnums=(0, 1, 2))(origin, lower_bound, upper_bound)

        # verify
        def box_projection_reference(origin, lower_bound, upper_bound):
            return jnp.clip(origin, lower_bound, upper_bound)

        jac_o_ref, jac_lb_ref, jac_ub_ref = jax.jacrev(box_projection_reference, argnums=(0, 1, 2))(
            origin, lower_bound, upper_bound
        )

        assert jnp.allclose(
            jac_o,
            jac_o_ref,
            atol=1e-4,
        ), f"was {jnp.diagonal(jac_o)} should be {jnp.diagonal(jac_o_ref)}"
        assert jnp.allclose(
            jac_lb,
            jac_lb_ref,
            atol=1e-4,
        ), f"was {jnp.diagonal(jac_lb)} should be {jnp.diagonal(jac_lb_ref)}"
        assert jnp.allclose(
            jac_ub,
            jac_ub_ref,
            atol=1e-4,
        ), f"was {jnp.diagonal(jac_ub)} should be {jnp.diagonal(jac_lb_ref)}"
