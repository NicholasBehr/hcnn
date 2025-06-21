import jax
import jax.numpy as jnp
from flax import nnx, struct

from hcnn.optimization.optimization_base import OptimizationSolverBase
from hcnn.projection.projection_base import ProjectionBase, ProjectionPhi
from hcnn.system_cost.system_cost_base import SystemCostBase
from hcnn.training.dropout_mlp import DropoutMLP
from hcnn.training.generalized_advantage import (
    generalized_advantage_1,
    generalized_advantage_t,
    td_residuals,
)


@struct.dataclass
class LossAndGrad:
    projection: ProjectionBase
    solver: OptimizationSolverBase
    sys_cost: SystemCostBase
    value_to_go_graphdef: nnx.GraphDef
    relax_to_go_graphdef: nnx.GraphDef

    @classmethod
    def factory(
        cls,
        projection: ProjectionBase,
        sys_cost: SystemCostBase,
        solver: OptimizationSolverBase,
        value_to_go: DropoutMLP,
        relax_to_go: DropoutMLP,
    ):
        return cls(
            projection=projection,
            solver=solver,
            sys_cost=sys_cost,
            value_to_go_graphdef=nnx.split(value_to_go)[0],
            relax_to_go_graphdef=nnx.split(relax_to_go)[0],
        )

    def policy_loss(
        self,
        value_residuals: jax.Array,
        rollout_interventions: jax.Array,
        td_gamma: float,
        gae_lambda: float,
    ):
        """
        Evaluates the loss function used to train policy weights.

        Args:
            value_residuals (jax.Array): shape (..., horizon-1).
            rollout_interventions (jax.Array): shape (..., horizon).
            td_gamma (float): Temporal difference discount factor.
            gae_lambda (float): Generalized advantage estimation lambda.

        Returns:
            jax.Array: Loss value, shape ().
        """
        advantage_1 = generalized_advantage_1(
            td_residuals=value_residuals,
            td_gamma=td_gamma,
            gae_lambda=gae_lambda,
        )
        discounted_interventions = jnp.sum(
            rollout_interventions
            * (td_gamma ** jnp.arange(rollout_interventions.shape[-1])),
            axis=-1,
        )

        # minimize discounted interventions and maximize advantage
        # (i.e., minimize the negative of the advantage)
        return jnp.mean(discounted_interventions**2 - advantage_1)

    def value_to_go_loss(
        self,
        value_residuals: jax.Array,
        td_gamma,
        gae_lambda,
    ):
        """
        Evaluates the loss function used to train the value-to-go weights.

        Args:
            value_residuals (jax.Array): shape (..., horizon-1).
            td_gamma (float): Temporal difference discount factor.
            gae_lambda (float): Generalized advantage estimation lambda.

        Returns:
            jax.Array: Loss value, shape ().
        """
        advantage_t = generalized_advantage_t(
            td_residuals=value_residuals,
            td_gamma=td_gamma,
            gae_lambda=gae_lambda,
        )

        # eliminate the value_residuals
        # (i.e., minimize the squared advantage)
        return jnp.mean(advantage_t**2)

    def relax_to_go_loss(
        self,
        relax_residuals: jax.Array,
        td_gamma: float,
        gae_lambda: float,
    ):
        """
        Evaluates the loss function used to train the relax-to-go weights.

        Args:
            relax_residuals (jax.Array): shape (..., horizon-1).
            td_gamma (float): Temporal difference discount factor.
            gae_lambda (float): Generalized advantage estimation lambda.

        Returns:
            jax.Array: Loss value, shape ().
        """
        advantage_t = generalized_advantage_t(
            td_residuals=relax_residuals,
            td_gamma=td_gamma,
            gae_lambda=gae_lambda,
        )

        # eliminate the relax_residuals
        # (i.e., minimize the squared advantage)
        return jnp.mean(advantage_t**2)

    def value_residuals(
        self,
        states: jax.Array,
        actions: jax.Array,
        value_to_go_state: nnx.State,
        td_gamma: float,
    ) -> tuple[jax.Array, nnx.State]:
        """
        Calculates the value residuals for the given states and actions.

        Args:
            states (jax.Array): shape (..., horizon, state_dim).
            actions (jax.Array): shape (..., horizon, action_dim).
            value_to_go_state (nnx.State):
                The state of the value-to-go model before.
            td_gamma (float): Temporal difference discount factor.

        Returns:
            tuple: A pair containing:
            - jax.Array: The calculated value residuals.
                shape (..., horizon-1).
            - nnx.State: The updated state of the value-to-go model.
        """
        # Compute the costs and rewards
        costs = self.sys_cost(states, actions)

        # Shape of stage_value is (..., horizon)
        stage_value = -costs

        # Hydrate value_to_go model
        value_to_go = nnx.merge(self.value_to_go_graphdef, value_to_go_state)

        # Current states shape is (..., horizon, state_dim)
        original_shape = states.shape
        flat_states = states.reshape(-1, original_shape[-1])

        # Apply the value_to_go model to the reshaped states
        flat_value_to_go_eval = value_to_go(flat_states)

        # Reshape values to (..., horizon)
        value_to_go_eval = flat_value_to_go_eval.reshape(original_shape[:-1])

        # Update the value_to_go_state with the new values
        value_to_go_state = nnx.state(value_to_go)

        # Compute the temporal difference (TD) errors
        value_residuals = td_residuals(
            rewards=stage_value, values=value_to_go_eval, td_gamma=td_gamma
        )
        return value_residuals, value_to_go_state

    def relax_residuals(
        self,
        states: jax.Array,
        relaxations_plus: jax.Array,
        relaxations_minus: jax.Array,
        relax_to_go_state: nnx.State,
        td_gamma: float,
    ) -> tuple[jax.Array, nnx.State]:
        """
        Calculates the relaxation residuals for the given states and relaxations.

        Args:
            states (jax.Array): shape (..., horizon, state_dim).
            relaxations_plus (jax.Array): shape (..., horizon, constraint_dim).
            relaxations_minus (jax.Array): shape (..., horizon, constraint_dim).
            relax_to_go_state (nnx.State):
                The state of the relax-to-go model before.
            td_gamma (float): Temporal difference discount factor.

        Returns:
            tuple: A pair containing:
            - jax.Array: The calculated relaxation residuals.
                shape (..., horizon-1).
            - nnx.State: The updated state of the relax-to-go model.
        """
        # Calculate max relaxations at each time step
        relaxations_plus_max = jnp.max(relaxations_plus, axis=-1)
        relaxations_minus_max = jnp.max(relaxations_minus, axis=-1)

        # Shape of stage_relax is (..., horizon)
        stage_relax = jnp.maximum(relaxations_plus_max, relaxations_minus_max)

        # Hydrate the relax_to_go model
        relax_to_go = nnx.merge(self.relax_to_go_graphdef, relax_to_go_state)

        # Current states shape is (..., horizon, state_dim)
        original_shape = states.shape
        flat_states = states.reshape(-1, original_shape[-1])

        # Apply the relax_to_go model to the reshaped states
        flat_relax_to_go_eval = relax_to_go(flat_states)

        # Reshape "values" to (..., horizon)
        relax_to_go_eval = flat_relax_to_go_eval.reshape(original_shape[:-1])

        # Update the value_to_go_state with the new values
        relax_to_go_state = nnx.state(relax_to_go)

        # Compute the temporal difference (TD) errors
        relax_residuals = td_residuals(
            rewards=stage_relax, values=relax_to_go_eval, td_gamma=td_gamma
        )
        return relax_residuals, relax_to_go_state

    def calculate_rollout(
        self,
        policy_state: nnx.State,
        relax_to_go_state: nnx.State,  # Kept for API compatibility
        initial_states: jax.Array,
        jax_keys: jax.Array,
    ):
        """
        Calculates the rollout for the given initial state and policy
        parameters.

        Args:
            policy_state (nnx.State):
                The state of the policy model.
            relax_to_go_state (nnx.State):
                The state of the relax-to-go model.
            initial_states (jax.Array): The initial state of the system.
                shape (batch_size, state_dim).
            jax_keys (jax.Array): JAX random keys for stochastic policies.
                shape (batch_size,).

        Returns:
            Rollout: The calculated rollout.
        """

        def sample_rollout(initial_state, policy_state, jax_key):
            phi = ProjectionPhi(
                initial_state=initial_state,
                policy_state=policy_state,
                jax_key=jax_key,
            )
            x_star = self.solver.solve(self.projection, phi)
            return self.projection.x_to_rollout(x_star)

        return jax.vmap(
            sample_rollout,
            in_axes=(0, None, 0),
            out_axes=0,
        )(
            initial_states,
            policy_state,
            jax_keys,
        )
