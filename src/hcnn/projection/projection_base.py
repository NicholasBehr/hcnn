from abc import abstractmethod

import jax
from flax import nnx, struct

from hcnn.optimization import OptimizationProblem
from hcnn.policy import PolicyBase
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase


@struct.dataclass
class RolloutPlan:
    states: jax.Array
    actions: jax.Array
    interventions: jax.Array
    relaxations: jax.Array


@struct.dataclass
class ProjectionBase(OptimizationProblem):
    sys_dynamics: SystemDynamicsBase
    sys_constraints: SystemConstraintsBase
    policy: PolicyBase

    @classmethod
    @abstractmethod
    def factory(
        cls,
        sys_dynamics: SystemDynamicsBase,
        sys_constraints: SystemConstraintsBase,
        policy: PolicyBase,
        horizon: int,
    ):
        pass

    @abstractmethod
    def get_x0(
        self,
        horizon: int,
        initial_state: jax.Array,
        policy_params: nnx.State,
        *,
        jax_key: jax.Array,
    ) -> jax.Array:
        pass

    @abstractmethod
    def get_h(
        self,
        initial_state: jax.Array,
        x: jax.Array,
        policy_params: nnx.State,
        *,
        jax_key: jax.Array,
    ) -> jax.Array:
        pass

    @jax.jit
    def get_problem(
        self,
        initial_state: jax.Array,
        policy_params: nnx.State,
        *,
        jax_key: jax.Array,
    ):
        def constraint_fn(x, policy_params):
            return self.get_h(initial_state, x, policy_params, jax_key=jax_key)

        return OptimizationProblem(
            x0=self.get_x0(
                horizon=self.horizon,
                initial_state=initial_state,
                policy_params=policy_params,
                jax_key=jax_key,
            ),
            cost_quad_diag=self.cost_quad_diag,
            cost_linear=self.cost_linear,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            constraint_params=policy_params,
            constraint_fn=constraint_fn,
        )

    @abstractmethod
    def get_rollout(
        self,
        x_star: jax.Array,
    ) -> RolloutPlan:
        pass
