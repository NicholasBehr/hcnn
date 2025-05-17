from abc import ABC, abstractmethod

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
class ProjectionBase(ABC):
    sys_dynamics: SystemDynamicsBase
    sys_constraints: SystemConstraintsBase
    horizon: int  # N rollout horizon length
    relaxation_penalty: float  # \rho relaxation (exact) penalty term

    policy_graph: nnx.GraphDef
    policy_other: nnx.State

    # constant OptimizationProblem params
    cost_quad_diag: jax.Array
    cost_linear: jax.Array
    lower_bound: jax.Array
    upper_bound: jax.Array

    @classmethod
    @abstractmethod
    def factory(
        cls,
        sys_dynamics: SystemDynamicsBase,
        sys_constraints: SystemConstraintsBase,
        policy: PolicyBase,
        horizon: int,
        relaxation_penalty: float,
    ):
        pass

    @abstractmethod
    def get_x0(
        self,
        initial_state: jax.Array,
        policy_params: nnx.State,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        pass

    @abstractmethod
    def get_h(
        self,
        initial_state: jax.Array,
        x: jax.Array,
        policy_params: nnx.State,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        pass

    def get_problem(
        self,
        initial_state: jax.Array,
        policy_params: nnx.State,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> OptimizationProblem:
        def constraint_fn(x, policy_params):
            return self.get_h(initial_state, x, policy_params, rngs=rngs)

        return OptimizationProblem(
            x0=self.get_x0(
                initial_state=initial_state,
                policy_params=policy_params,
                rngs=rngs,
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
