import jax
import jax.numpy as jnp
from flax import nnx, struct

from hcnn.projection.projection_base import ProjectionBase
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
    value_to_go_graphdef: nnx.GraphDef
    relax_to_go_graphdef: nnx.GraphDef
    sys_cost: SystemCostBase

    @classmethod
    def factory(
        cls,
        projection: ProjectionBase,
        sys_cost: SystemCostBase,
        value_to_go: DropoutMLP,
        relax_to_go: DropoutMLP,
    ):
        return cls(
            projection=projection,
            value_to_go_graphdef=nnx.split(value_to_go)[0],
            relax_to_go_graphdef=nnx.split(relax_to_go)[0],
            sys_cost=sys_cost,
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
            jax.Array: Loss values, shape (...,).
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
            value_residuals (jax.Array):
                shape (horizon-1,) or (batch_size, horizon-1).
            td_gamma (float): Temporal difference discount factor.
            gae_lambda (float): Generalized advantage estimation lambda.

        Returns:
            jax.Array: Loss values,
                shape () or (batch_size,).
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
            relax_residuals (jax.Array):
                shape (horizon-1,) or (batch_size, horizon-1).
            td_gamma (float): Temporal difference discount factor.
            gae_lambda (float): Generalized advantage estimation lambda.

        Returns:
            jax.Array: Loss values,
                shape () or (batch_size,).
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
            states (jax.Array):
                shape (horizon, state_dim) or (batch_size, horizon, state_dim).
            actions (jax.Array):
                shape (horizon, action_dim) or (batch_size, horizon, action_dim).
            value_to_go_state (nnx.State):
                The state of the value-to-go model before.
            td_gamma (float): Temporal difference discount factor.

        Returns:
            tuple: A pair containing:
            - jax.Array: The calculated value residuals.
                shape (horizon-1,) or (batch_size, horizon-1).
            - nnx.State: The updated state of the value-to-go model.
        """
        # Compute the costs and rewards
        costs = self.sys_cost(states, actions)
        rewards = -costs

        # Hydrate value_to_go model
        value_to_go = nnx.merge(self.value_to_go_graphdef, value_to_go_state)

        # Evaluate the value_to_go model with the current states
        values = nnx.vmap(value_to_go, in_axes=-1)(states)

        # Update the value_to_go_state with the new values
        value_to_go_state = nnx.state(value_to_go)

        # Compute the temporal difference (TD) errors
        value_residuals = td_residuals(
            rewards=rewards, values=values, td_gamma=td_gamma
        )
        return value_residuals, value_to_go_state

    def relax_residuals(
        self,
        states: jax.Array,
        relaxations: jax.Array,
        relax_to_go_state: nnx.State,
        td_gamma: float,
    ) -> tuple[jax.Array, nnx.State]:
        """
        Calculates the relaxation residuals for the given states and relaxations.

        Args:
            states (jax.Array):
                shape (state_dim,) or (batch_size, state_dim).
            relaxations (jax.Array):
                shape (horizon,) or (batch_size, horizon).
            relax_to_go_state (nnx.State):
                The state of the relax-to-go model before.
            td_gamma (float): Temporal difference discount factor.

        Returns:
            tuple: A pair containing:
            - jax.Array: The calculated relaxation residuals.
                shape (horizon-1,) or (batch_size, horizon-1).
            - nnx.State: The updated state of the relax-to-go model.
        """
        # Compute the rewards
        rewards = relaxations

        # Hydrate the relax_to_go model
        relax_to_go = nnx.merge(self.relax_to_go_graphdef, relax_to_go_state)

        # Evaluate the relax_to_go model with the current states
        values = nnx.vmap(relax_to_go, in_axes=-1)(states)

        # Update the relax_to_go_state with the new values
        relax_to_go_state = nnx.state(relax_to_go)

        # Compute the temporal difference (TD) errors
        relax_residuals = td_residuals(
            rewards=rewards, values=values, td_gamma=td_gamma
        )
        return relax_residuals, relax_to_go_state

    def calculate_rollout(
        self,
        policy_state: nnx.State,
        value_to_go_state: nnx.State,
        relax_to_go_state: nnx.State,
        initial_states: jax.Array,
        jax_keys: jax.Array,
    ):
        """
        Calculates the rollout for the given initial state and policy parameters.

        Args:
            policy_params (nnx.State): The parameters of the policy.
            initial_states (jax.Array): The initial state of the system.
                shape (state_dim,) or (batch_size, state_dim).
            jax_key (jax.Array): JAX random keys for stochastic policies.
                shape () or (batch_size,).

        Returns:
            Rollout: The calculated rollout.
        """
        pass
