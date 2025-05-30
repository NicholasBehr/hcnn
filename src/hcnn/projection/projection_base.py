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
class ProjectionPhi:
    initial_state: jax.Array
    policy_params: nnx.State
    jax_key: jax.Array


@struct.dataclass
class ProjectionBase(OptimizationProblemBase[ProjectionPhi]):
    sys_dynamics: SystemDynamicsBase
    sys_constraints: SystemConstraintsBase
    policy_graphdef: nnx.GraphDef
    policy_non_param: nnx.GraphState | nnx.VariableState
    horizon: int = struct.field(pytree_node=False)

    @abstractmethod
    def get_x0(self, phi: ProjectionPhi) -> jax.Array:
        """
        Returns an initial guess for the optimization variable x, given the projection parameters.
        This is obtained by rolling out the policy without interventions.

        Args:
            phi (ProjectionPhi): The projection parameters, including initial state, policy parameters, and random key.

        Returns:
            jax.Array: A flat 1D array representing the initial guess for the optimization variable.
        """
        pass

    @abstractmethod
    def constraint_fn(self, x: jax.Array, phi: ProjectionPhi) -> jax.Array:
        """
        Computes the nonlinear constraint values for the optimization problem.

        Args:
            x (jax.Array): The flat 1D optimization variable.
            phi (ProjectionPhi): The projection parameters.

        Returns:
            jax.Array: A flat 1D array of constraint values, to be used with lower and upper bounds.
        """
        pass

    @abstractmethod
    def get_rollout(self, x_star: jax.Array) -> RolloutPlan:
        """
        Reconstructs a RolloutPlan from the optimized variable x_star.

        Args:
            x_star (jax.Array): The optimized flat 1D variable from the solver.

        Returns:
            RolloutPlan: Structured rollout containing states, actions, interventions, and relaxations.
        """
        pass
