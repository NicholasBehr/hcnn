from abc import abstractmethod

import jax
from flax import nnx, struct

from hcnn.optimization import OptimizationProblemBase
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase


@struct.dataclass
class RolloutPlan:
    states: jax.Array
    actions: jax.Array
    interventions: jax.Array
    relaxations: jax.Array


@struct.dataclass
class Context:
    initial_state: jax.Array
    policy_params: nnx.State
    jax_key: jax.Array


@struct.dataclass
class ProjectionBase(OptimizationProblemBase[Context]):
    sys_dynamics: SystemDynamicsBase
    sys_constraints: SystemConstraintsBase
    policy_graphdef: nnx.GraphDef
    policy_non_param: nnx.GraphState | nnx.VariableState
    horizon: int = struct.field(pytree_node=False)

    @abstractmethod
    def get_rollout(self, x_star: jax.Array) -> RolloutPlan:
        pass
