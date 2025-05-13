from abc import ABC, abstractmethod

import jax
from flax import nnx, struct

from hcnn.environment import Environment
from hcnn.optimization.optimization import OptimizationSolver


@struct.dataclass
class RolloutPlan:
    actions: jax.Array
    policy_actions: jax.Array
    states: jax.Array
    relaxations: jax.Array


class RolloutProjection(ABC):
    ## Use in conjunction with nnx.cached_partial
    @abstractmethod
    @nnx.custom_vjp
    @classmethod
    def sample(
        cls,
        environment: Environment,
        optimizer: OptimizationSolver,
        initial_state: jax.Array,
    ) -> RolloutPlan:
        pass

    @abstractmethod
    @classmethod
    def sample_fwd(
        cls: "RolloutProjection",
        environment: Environment,
        optimizer: OptimizationSolver,
        initial_state: jax.Array,
    ) -> RolloutPlan:
        pass

    @abstractmethod
    @staticmethod
    def sample_bwd(res, g):
        input_updates_g, out_g = g
        cos_x, sin_x, m = res

    sample.defvjp(sample_fwd, sample_bwd)
