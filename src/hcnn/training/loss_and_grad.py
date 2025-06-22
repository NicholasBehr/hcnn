import jax
import jax.numpy as jnp
from flax import nnx, struct

from hcnn.optimization.optimization_base import OptimizationSolverBase
from hcnn.projection.projection_base import (
    ProjectionBase,
    ProjectionPhi,
    Rollout,
)
from hcnn.system_costs.system_costs_base import SystemCostsBase
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
    sys_cost: SystemCostsBase
    value_to_go_graphdef: nnx.GraphDef
    relax_to_go_graphdef: nnx.GraphDef

    @classmethod
    def factory(
        cls,
        projection: ProjectionBase,
        sys_cost: SystemCostsBase,
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
        return jnp.mean(discounted_interventions**2) - jnp.mean(advantage_1)

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
        Calculates the relaxation residuals for the given states and relaxations

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
        relax_to_go_state: nnx.State,
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

        def sample_rollout(
            initial_state, policy_state, relax_to_go_state, jax_key
        ):
            phi = ProjectionPhi(
                initial_state=initial_state,
                policy_state=policy_state,
                relax_to_go_state=relax_to_go_state,
                jax_key=jax_key,
            )
            x_star = self.solver.solve(self.projection, phi)
            return self.projection.x_to_rollout(x_star)

        return jax.vmap(
            sample_rollout,
            in_axes=(0, None, None, 0),
            out_axes=0,
        )(
            initial_states,
            policy_state,
            relax_to_go_state,
            jax_keys,
        )

    def __call__(  # noqa: PLR0913
        self,
        policy_state: nnx.State,
        relax_to_go_state: nnx.State,
        value_to_go_state: nnx.State,
        initial_states: jax.Array,
        jax_keys: jax.Array,
        td_gamma: float,
        gae_lambda: float,
    ):
        """
        Computes the loss and gradients for the policy, value-to-go, and
        relax-to-go models.

        Args:
            policy_state (nnx.State): The state of the policy model.
            relax_to_go_state (nnx.State): The state of the relax-to-go model.
            value_to_go_state (nnx.State): The state of the value-to-go model.
            initial_states (jax.Array): Initial states for the rollout.
                shape (batch_size, state_dim).
            jax_keys (jax.Array): JAX random keys for stochastic policies.
                shape (batch_size,).
            td_gamma (float): Temporal difference discount factor.
            gae_lambda (float): Generalized advantage estimation lambda.

        Returns:
            tuple: A tuple containing:
                - tuple: Loss values for policy, relax_to_go, and value_to_go.
                - tuple: Gradients for policy_state, relax_to_go_state,
                    and value_to_go_state.
                - tuple: Updated states for policy, relax_to_go, value_to_go.
        """
        # Calculate the rollout
        rollout, rollout_vjp = jax.vjp(
            lambda policy_state: self.calculate_rollout(
                policy_state=policy_state,
                relax_to_go_state=relax_to_go_state,
                initial_states=initial_states,
                jax_keys=jax_keys,
            ),
            policy_state,
        )

        (value_residuals, value_to_go_state), value_res_vjp = jax.vjp(
            lambda states, actions, value_to_go_state: self.value_residuals(
                states=states,
                actions=actions,
                value_to_go_state=value_to_go_state,
                td_gamma=td_gamma,
            ),
            rollout.states,
            rollout.actions,
            value_to_go_state,
        )

        (relax_residuals, relax_to_go_state), relax_res_vjp = jax.vjp(
            lambda states,
            relax_plus,
            relax_minus,
            relax_to_go_state: self.relax_residuals(
                states=states,
                relaxations_plus=relax_plus,
                relaxations_minus=relax_minus,
                relax_to_go_state=relax_to_go_state,
                td_gamma=td_gamma,
            ),
            rollout.states,
            rollout.relaxations_plus,
            rollout.relaxations_minus,
            relax_to_go_state,
        )

        policy_loss, (policy_val_res_grad, policy_interv_grad) = (
            jax.value_and_grad(
                self.policy_loss,
                argnums=(0, 1),
            )(
                value_residuals,
                rollout.interventions,
                td_gamma=td_gamma,
                gae_lambda=gae_lambda,
            )
        )

        value_to_go_loss, value_val_res_grad = jax.value_and_grad(
            self.value_to_go_loss,
        )(
            value_residuals,
            td_gamma=td_gamma,
            gae_lambda=gae_lambda,
        )

        relax_to_go_loss, relax_rel_res_grad = jax.value_and_grad(
            self.relax_to_go_loss,
        )(
            relax_residuals,
            td_gamma=td_gamma,
            gae_lambda=gae_lambda,
        )

        # Calculate the gradients for the policy state
        policy_value_to_go_state_grad = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x), value_to_go_state
        )  # the outgoing value_to_go_state does not affect the policy loss
        (policy_states_grad, policy_actions_grad, _) = value_res_vjp(
            (policy_val_res_grad, policy_value_to_go_state_grad)
        )
        policy_rollout_grad = Rollout(
            states=policy_states_grad,
            actions=policy_actions_grad,
            interventions=policy_interv_grad,
            relaxations_plus=jnp.zeros_like(rollout.relaxations_plus),
            relaxations_minus=jnp.zeros_like(rollout.relaxations_minus),
        )
        policy_policy_state_grad = rollout_vjp(policy_rollout_grad)

        # Calculate the gradients for the value_to_go state
        value_value_to_go_state_grad = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x), value_to_go_state
        )  # the outgoing value_to_go_state does not affect the value loss
        (_, _, value_value_to_go_state_grad) = value_res_vjp(
            (value_val_res_grad, value_value_to_go_state_grad)
        )

        # Calculate the gradients for the relax_to_go state
        relax_relax_to_go_state_grad = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x), relax_to_go_state
        )  # the outgoing relax_to_go_state does not affect the relax loss
        (_, _, _, relax_relax_to_go_state_grad) = relax_res_vjp(
            (relax_rel_res_grad, relax_relax_to_go_state_grad)
        )
        # We purposfully do not propagate gradients through the rollout here!
        # We dont want to change the rollout to improve the relax to go loss.

        return (
            (policy_loss, relax_to_go_loss, value_to_go_loss),
            (
                policy_policy_state_grad,
                relax_relax_to_go_state_grad,
                value_value_to_go_state_grad,
            ),
            (policy_state, relax_to_go_state, value_to_go_state),
        )
