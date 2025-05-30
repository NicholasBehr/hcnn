from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jax import test_util

from hcnn.optimization.optimization import OptimizationProblemBase, OptimizationSolverBase


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
    dimension = 15
    for i in range(15):
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
            atol=1e-5,
        )


def test_vjp_unconstrained(solver: OptimizationSolverBase):
    dimension = 15
    for i in range(15):
        # create
        keys = jax.random.split(jax.random.key(i), 2)
        cost_quad_diag = jax.random.normal(keys[0], (dimension,))
        cost_linear = jax.random.normal(keys[1], (dimension,))
        problem, phi = UnconstrainedProblem.factory(cost_quad_diag, cost_linear)

        # solve
        jac_prob, jac_phi = jax.jacrev(solver.solve, argnums=(0, 1))(problem, phi)

        # jac_prob_unroll = jax.jacrev(solver._solve_for_unrolling, argnums=(0,))(problem, phi)

        # verify
        assert jnp.allclose(
            jac_prob.cost_quad_diag,
            jnp.diag(cost_linear / (cost_quad_diag**2)),
            atol=1e-5,
        )
        assert jnp.allclose(
            jac_prob.cost_linear,
            jnp.diag(-1 / (cost_quad_diag)),
            atol=1e-5,
        )
        assert jnp.allclose(
            jac_prob.lower_bound,
            jnp.zeros((dimension, dimension)),
            atol=1e-5,
        )
        assert jnp.allclose(
            jac_prob.upper_bound,
            jnp.zeros((dimension, dimension)),
            atol=1e-5,
        )
        assert jnp.allclose(
            jac_phi,
            jnp.zeros((1, 1)),
            atol=1e-5,
        )


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
    dimension = 15
    for i in range(15):
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
            atol=1e-5,
        )


def test_vjp_box_projection(solver: OptimizationSolverBase):
    dimension = 3
    for i in range(1):
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
        print(f"run {i}")

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


@struct.dataclass
class TriangleProjectionProblem(OptimizationProblemBase[jax.Array]):
    def get_x0(self, phi: jax.Array):
        return jnp.zeros_like(self.cost_linear)

    def constraint_fn(self, x: jax.Array, phi: jax.Array) -> jax.Array:
        return phi @ x

    @classmethod
    def factory(cls, origin, point_a, point_b, point_c):
        cost_quad_diag = jnp.ones_like(origin)
        cost_linear = -origin

        # Stack points into a (3, 2) array
        points = jnp.stack([point_a, point_b, point_c])

        # Get the edges in order
        edges = jnp.roll(points, -1, axis=0) - points  # shape (3, 2)

        # Compute outward normals to each edge (2D perp vector: (x, y) -> (y, -x))
        normals = jnp.stack([edges[:, 1], -edges[:, 0]], axis=1)

        # Normalize the normals
        norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms

        # Compute b_i = n_i · p_i
        b = jnp.sum(normals * points, axis=1)

        phi = normals
        lower_bound = -50 * jnp.ones((3,))
        upper_bound = b

        return TriangleProjectionProblem(
            cost_quad_diag=cost_quad_diag,
            cost_linear=cost_linear,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ), phi


def test_solve_triangle_projection(solver: OptimizationSolverBase):
    point_a = jnp.array([0.0, 0.0])
    point_b = jnp.array([1.0, 0.0])
    point_c = jnp.array([0.0, 1.0])

    origins = jnp.array(
        [
            [1.0, 1.0],
            [-0.5, 0.5],
            [0.5, -0.5],
            [0.25, 0.25],
        ]
    )

    x_stars_analytic = jnp.array(
        [
            [0.5, 0.5],
            [0.0, 0.5],
            [0.5, 0.0],
            [0.25, 0.25],
        ]
    )

    for origin, x_star_analytic in zip(origins, x_stars_analytic, strict=False):
        problem, phi = TriangleProjectionProblem.factory(origin, point_a, point_b, point_c)

        # solve
        x_star = solver.solve(problem, phi)

        # verify
        assert jnp.allclose(
            x_star,
            x_star_analytic,
            atol=1e-5,
        )


def test_vjp_triangle_projection(solver: OptimizationSolverBase):
    point_a = jnp.array([0.0, 0.0])
    point_b = jnp.array([1.0, 0.0])
    point_c = jnp.array([0.0, 1.0])

    origins = jnp.array(
        [
            [1.0, 1.0],
            [-0.5, 0.5],
            [0.5, -0.5],
            [0.25, 0.25],
        ]
    )

    for origin in origins:
        problem, phi = TriangleProjectionProblem.factory(origin, point_a, point_b, point_c)

        test_util.check_vjp(
            f=solver.solve,
            f_vjp=partial(jax.vjp, solver.solve),
            args=(problem, phi),
        )


@struct.dataclass
class Context:
    a: jax.Array
    b: jax.Array
    jax_key: jax.Array = struct.field(pytree_node=False)


@struct.dataclass
class CircleProjectionProblem(OptimizationProblemBase[Context]):
    """
    x_star = argmin |x - origin|_2^2
            s.t. l <= x.T @ x <= u
    """

    def get_x0(self, phi: Context):
        return jax.random.normal(jax.random.fold_in(phi.jax_key, 0), (3,))

    def constraint_fn(self, x: jax.Array, phi: Context) -> jax.Array:
        radius_sqr = jnp.dot(x, x) * phi.a * phi.b
        return jnp.tile(radius_sqr, 5)

    @classmethod
    def factory(cls, jax_key):
        key_a = jax.random.fold_in(jax_key, 1)
        a = jax.random.uniform(key_a, (1,), minval=0.9, maxval=1.1)

        key_b = jax.random.fold_in(jax_key, 2)
        b = jax.random.uniform(key_b, (1,), minval=0.9, maxval=1.1)

        cost_quad_diag = jnp.ones(3)
        cost_linear = jnp.ones(3)
        key_l = jax.random.fold_in(jax_key, 3)
        lower_bound = jax.random.uniform(key_l, (5,), minval=3, maxval=4)
        key_u = jax.random.fold_in(jax_key, 4)
        upper_bound = jax.random.uniform(key_u, (5,), minval=6, maxval=7)

        return CircleProjectionProblem(
            cost_quad_diag=cost_quad_diag,
            cost_linear=cost_linear,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ), Context(a, b, jax_key)


def test_vjp_circle_projection(solver: OptimizationSolverBase):
    # create
    jax_key = jax.random.key(3)
    problem, phi = CircleProjectionProblem.factory(jax_key)

    test_util.check_vjp(
        f=solver.solve,
        f_vjp=partial(jax.vjp, solver.solve),
        args=(problem, phi),
        rtol=1 / 100,
    )
