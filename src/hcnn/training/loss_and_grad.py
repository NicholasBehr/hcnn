import jax
import jax.numpy as jnp
from flax import nnx, struct

from hcnn.optimization import OptimizationSolverBase
from hcnn.projection import (
    ProjectionBase,
    ProjectionPhi,
    Rollout,
)
from hcnn.system_costs import SystemCostsBase

from .dropout_mlp import DropoutMLP
from .generalized_advantage import (
    generalized_advantage_1,
    generalized_advantage_t,
    td_residuals,
)


@struct.dataclass
class LossAndGrad:
    projection: ProjectionBase
    solver: OptimizationSolverBase
    sys_cost: SystemCostsBase
    cost_to_go_graphdef: nnx.GraphDef
    relax_to_go_graphdef: nnx.GraphDef

    @classmethod
    def factory(
        cls,
        projection: ProjectionBase,
        sys_cost: SystemCostsBase,
        solver: OptimizationSolverBase,
        cost_to_go: DropoutMLP,
        relax_to_go: DropoutMLP,
    ):
        return cls(
            projection=projection,
            solver=solver,
            sys_cost=sys_cost,
            cost_to_go_graphdef=nnx.split(cost_to_go)[0],
            relax_to_go_graphdef=nnx.split(relax_to_go)[0],
        )

    def policy_loss(
        self,
        cost_residuals: jax.Array,
        rollout_interventions: jax.Array,
        td_gamma: float,
        gae_lambda: float,
    ):
        """
        Evaluates the loss function used to train policy weights.

        Args:
            cost_residuals (jax.Array): shape (..., horizon-1).
            rollout_interventions (jax.Array): shape (..., horizon).
            td_gamma (float): Temporal difference discount factor.
            gae_lambda (float): Generalized advantage estimation lambda.

        Returns:
            jax.Array: Policy loss value, shape ().
        """
        advantage_1 = generalized_advantage_1(
            td_residuals=cost_residuals,
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

    def cost_to_go_loss(
        self,
        cost_residuals: jax.Array,
        td_gamma,
        gae_lambda,
    ):
        """
        Evaluates the loss function used to train the cost-to-go weights.

        Args:
            cost_residuals (jax.Array): shape (..., horizon-1).
            td_gamma (float): Temporal difference discount factor.
            gae_lambda (float): Generalized advantage estimation lambda.

        Returns:
            jax.Array: Loss, shape ().
        """
        advantage_t = generalized_advantage_t(
            td_residuals=cost_residuals,
            td_gamma=td_gamma,
            gae_lambda=gae_lambda,
        )

        # eliminate the cost_residuals
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
            jax.Array: Relax-to-go loss value, shape ().
        """
        advantage_t = generalized_advantage_t(
            td_residuals=relax_residuals,
            td_gamma=td_gamma,
            gae_lambda=gae_lambda,
        )

        # eliminate the relax_residuals
        # (i.e., minimize the squared advantage)
        return jnp.mean(advantage_t**2)

    def cost_residuals(
        self,
        states: jax.Array,
        actions: jax.Array,
        cost_to_go_state: nnx.State,
        td_gamma: float,
    ) -> tuple[jax.Array, nnx.State]:
        """
        Calculates the cost residuals for the given states and actions.

        Args:
            states (jax.Array): shape (..., horizon, state_dim).
            actions (jax.Array): shape (..., horizon, action_dim).
            cost_to_go_state (nnx.State):
                The state of the cost-to-go model before.
            td_gamma (float): Temporal difference discount factor.

        Returns:
            tuple: A pair containing:
            - jax.Array: The calculated cost residuals.
                shape (..., horizon-1).
            - nnx.State: The updated state of the cost-to-go model.
        """
        # Shape of stage_cost is (..., horizon)
        stage_cost = self.sys_cost(states, actions)

        # Hydrate cost_to_go model
        cost_to_go = nnx.merge(self.cost_to_go_graphdef, cost_to_go_state)

        # Current states shape is (..., horizon, state_dim)
        original_shape = states.shape
        flat_states = states.reshape(-1, original_shape[-1])

        # Apply the cost_to_go model to the reshaped states
        flat_cost_to_go_eval = cost_to_go(flat_states)

        # Reshape costs to (..., horizon)
        cost_to_go_eval = flat_cost_to_go_eval.reshape(original_shape[:-1])

        # Update the cost_to_go_state with the new costs
        cost_to_go_state = nnx.state(cost_to_go)

        # Compute the temporal difference (TD) errors
        cost_residuals = td_residuals(
            rewards=stage_cost, values=cost_to_go_eval, td_gamma=td_gamma
        )
        return cost_residuals, cost_to_go_state

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

        # Reshape "costs" to (..., horizon)
        relax_to_go_eval = flat_relax_to_go_eval.reshape(original_shape[:-1])

        # Update the cost_to_go_state with the new costs
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
        cost_to_go_state: nnx.State,
        initial_states: jax.Array,
        jax_keys: jax.Array,
        td_gamma: float,
        gae_lambda: float,
    ):
        """
        Computes the loss and gradients for the policy, cost-to-go, and
        relax-to-go models.

        Args:
            policy_state (nnx.State): The state of the policy model.
            relax_to_go_state (nnx.State): The state of the relax-to-go model.
            cost_to_go_state (nnx.State): The state of the cost-to-go model.
            initial_states (jax.Array): Initial states for the rollout.
                shape (batch_size, state_dim).
            jax_keys (jax.Array): JAX random keys for stochastic policies.
                shape (batch_size,).
            td_gamma (float): Temporal difference discount factor.
            gae_lambda (float): Generalized advantage estimation lambda.

        Returns:
            tuple: A tuple containing:
                - tuple: Loss values for policy, relax_to_go, and cost_to_go.
                - tuple: Gradients for policy_state, relax_to_go_state,
                    and cost_to_go_state.
                - tuple: Updated states for policy, relax_to_go, cost_to_go.
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

        (cost_residuals, cost_to_go_state), cost_res_vjp = jax.vjp(
            lambda states, actions, cost_to_go_state: self.cost_residuals(
                states=states,
                actions=actions,
                cost_to_go_state=cost_to_go_state,
                td_gamma=td_gamma,
            ),
            rollout.states,
            rollout.actions,
            cost_to_go_state,
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
                cost_residuals,
                rollout.interventions,
                td_gamma=td_gamma,
                gae_lambda=gae_lambda,
            )
        )

        cost_to_go_loss, cost_val_res_grad = jax.value_and_grad(
            self.cost_to_go_loss,
        )(
            cost_residuals,
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

        ## Calculate the gradients for the policy state ##
        policy_cost_to_go_state_grad = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x), cost_to_go_state
        )  # the outgoing cost_to_go_state does not affect the policy loss
        (policy_states_grad, policy_actions_grad, _) = cost_res_vjp(
            (policy_val_res_grad, policy_cost_to_go_state_grad)
        )
        policy_rollout_grad = Rollout(
            states=policy_states_grad,
            actions=policy_actions_grad,
            interventions=policy_interv_grad,
            relaxations_plus=jnp.zeros_like(rollout.relaxations_plus),
            relaxations_minus=jnp.zeros_like(rollout.relaxations_minus),
        )
        policy_policy_state_grad = rollout_vjp(policy_rollout_grad)

        ## Calculate the gradients for the cost_to_go state ##
        cost_cost_to_go_state_grad = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x), cost_to_go_state
        )  # the outgoing cost_to_go_state does not affect the cost loss
        (_, _, cost_cost_to_go_state_grad) = cost_res_vjp(
            (cost_val_res_grad, cost_cost_to_go_state_grad)
        )

        ## Calculate the gradients for the relax_to_go state ##
        relax_relax_to_go_state_grad = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x), relax_to_go_state
        )  # the outgoing relax_to_go_state does not affect the relax loss
        (_, _, _, relax_relax_to_go_state_grad) = relax_res_vjp(
            (relax_rel_res_grad, relax_relax_to_go_state_grad)
        )
        # We purposfully do not propagate gradients through the rollout here!
        # We dont want to change the rollout to improve the relax to go loss.

        return (
            (policy_loss, relax_to_go_loss, cost_to_go_loss),
            (
                policy_policy_state_grad,
                relax_relax_to_go_state_grad,
                cost_cost_to_go_state_grad,
            ),
            (policy_state, relax_to_go_state, cost_to_go_state),
        )
