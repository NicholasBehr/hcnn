import jax
import jax.numpy as jnp
from flax import nnx, struct

from hcnn.optimization import OptimizationInfo, OptimizationSolverBase
from hcnn.projection import (
    ProjectionBase,
    ProjectionPhiD,
    ProjectionPhiND,
    Rollout,
)
from hcnn.system_costs import SystemCostsBase

from .dropout_mlp import DropoutMLP
from .generalized_advantage import (
    generalized_advantage_t,
    td_residuals,
)


@struct.dataclass
class ModelStates:
    policy: nnx.State
    relax_to_go: nnx.State
    cost_to_go: nnx.State


@struct.dataclass
class ModelLoss:
    policy: jax.Array
    cost_to_go: jax.Array
    cost_encountered: jax.Array
    relax_to_go: jax.Array


@struct.dataclass
class ModelGrad:
    policy: nnx.State
    cost_to_go: nnx.State
    relax_to_go: nnx.State


@struct.dataclass
class LossAndGrad:
    projection: ProjectionBase
    solver: OptimizationSolverBase
    sys_costs: SystemCostsBase
    cost_to_go_graphdef: nnx.GraphDef
    relax_to_go_graphdef: nnx.GraphDef
    td_gamma: float = 0.9
    policy_gae_lambda: float = 0.8
    x_to_go_gae_lambda: float = 0.8
    policy_intervention_cost: float = 0.5

    @classmethod
    def factory(
        cls,
        projection: ProjectionBase,
        solver: OptimizationSolverBase,
        sys_costs: SystemCostsBase,
        cost_to_go: DropoutMLP,
        relax_to_go: DropoutMLP,
        td_gamma: float = 0.9,
        policy_gae_lambda: float = 0.8,
        x_to_go_gae_lambda: float = 0.8,
        policy_intervention_cost: float = 0.5,
    ):
        return LossAndGrad(
            projection=projection,
            solver=solver,
            sys_costs=sys_costs,
            cost_to_go_graphdef=nnx.split(cost_to_go)[0],
            relax_to_go_graphdef=nnx.split(relax_to_go)[0],
            td_gamma=td_gamma,
            policy_gae_lambda=policy_gae_lambda,
            x_to_go_gae_lambda=x_to_go_gae_lambda,
            policy_intervention_cost=policy_intervention_cost,
        )

    def policy_loss(
        self,
        cost_residuals: jax.Array,
        rollout_interventions: jax.Array,
    ):
        """
        Evaluates the loss function used to train policy weights.

        Args:
            cost_residuals (jax.Array): shape (..., horizon-1).
            rollout_interventions (jax.Array): shape (..., horizon).

        Returns:
            jax.Array: Policy loss value, shape ().
        """
        advantage_t = generalized_advantage_t(
            td_residuals=cost_residuals,
            td_gamma=self.td_gamma,
            gae_lambda=self.policy_gae_lambda,
        )
        interventions = jnp.sum(rollout_interventions, axis=-1)

        # minimize discounted interventions and maximize advantage
        # (i.e., minimize the negative of the advantage)
        return (
            -jnp.mean(advantage_t) * (1.0 - self.policy_intervention_cost)
            + jnp.mean(interventions) * self.policy_intervention_cost
        )

    def cost_to_go_loss(
        self,
        cost_residuals: jax.Array,
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
            td_gamma=self.td_gamma,
            gae_lambda=self.x_to_go_gae_lambda,
        )

        # eliminate the cost_residuals
        # (i.e., minimize the squared advantage)
        return jnp.mean(advantage_t**2)

    def relax_to_go_loss(
        self,
        relax_residuals: jax.Array,
    ):
        """
        Evaluates the loss function used to train the relax-to-go weights.

        Args:
            relax_residuals (jax.Array): shape (..., horizon-1).

        Returns:
            jax.Array: Relax-to-go loss value, shape ().
        """
        advantage_t = generalized_advantage_t(
            td_residuals=relax_residuals,
            td_gamma=self.td_gamma,
            gae_lambda=self.x_to_go_gae_lambda,
        )

        # eliminate the relax_residuals
        # (i.e., minimize the squared advantage)
        return jnp.mean(advantage_t**2)

    def cost_residuals(
        self,
        states: jax.Array,
        actions: jax.Array,
        cost_to_go_state: nnx.State,
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
        stage_cost = self.sys_costs(states, actions)

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
            rewards=stage_cost, values=cost_to_go_eval, td_gamma=self.td_gamma
        )
        return cost_residuals, cost_to_go_state

    def relax_residuals(
        self,
        states: jax.Array,
        relaxations_plus: jax.Array,
        relaxations_minus: jax.Array,
        relax_to_go_state: nnx.State,
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
        relaxations_plus_max = jnp.sum(jnp.abs(relaxations_plus), axis=-1)
        relaxations_minus_max = jnp.sum(jnp.abs(relaxations_minus), axis=-1)

        # Shape of stage_relax is (..., horizon)
        stage_relax = relaxations_plus_max + relaxations_minus_max

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
            rewards=stage_relax, values=relax_to_go_eval, td_gamma=self.td_gamma
        )
        return relax_residuals, relax_to_go_state

    def calculate_rollout(
        self,
        policy_params: nnx.State,
        relax_to_go_params: nnx.State,
        policy_non_params: nnx.State,
        relax_to_go_non_params: nnx.State,
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
            initial_state,
            jax_key,
        ):
            phi_d = ProjectionPhiD(
                policy_params=policy_params,
                relax_to_go_params=relax_to_go_params,
            )
            phi_nd = ProjectionPhiND(
                policy_non_params=policy_non_params,
                relax_to_go_non_params=relax_to_go_non_params,
                initial_state=initial_state,
                jax_key=jax_key,
            )
            x_star, info = self.solver.solve(self.projection, phi_d, phi_nd)
            rollout = self.projection.x_to_rollout(x_star)
            return rollout, info

        return jax.vmap(sample_rollout)(
            initial_states,
            jax_keys,
        )

    def __call__(
        self,
        model_states: ModelStates,
        initial_states: jax.Array,
        jax_keys: jax.Array,
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
        in_policy_state = model_states.policy
        in_relax_to_go_state = model_states.relax_to_go
        in_cost_to_go_state = model_states.cost_to_go

        # Calculate the rollout
        policy_params, policy_non_params = nnx.filter_state(
            in_policy_state, nnx.Param, ...
        )
        relax_to_go_params, relax_to_go_non_params = nnx.filter_state(
            in_relax_to_go_state, nnx.Param, ...
        )
        (rollout, info), rollout_vjp = jax.vjp(
            lambda policy_params, relax_to_go_params: self.calculate_rollout(
                policy_params=policy_params,
                relax_to_go_params=relax_to_go_params,
                policy_non_params=policy_non_params,
                relax_to_go_non_params=relax_to_go_non_params,
                initial_states=initial_states,
                jax_keys=jax_keys,
            ),
            policy_params,
            relax_to_go_params,
        )

        # Calculate cost and relaxation residuals
        (cost_residuals, out_cost_to_go_state), cost_res_vjp = jax.vjp(
            lambda states, actions, cost_to_go_state: self.cost_residuals(
                states=states,
                actions=actions,
                cost_to_go_state=cost_to_go_state,
            ),
            rollout.states,
            rollout.actions,
            in_cost_to_go_state,
        )

        (relax_residuals, out_relax_to_go_state), relax_res_vjp = jax.vjp(
            lambda relax_to_go_state: self.relax_residuals(
                states=rollout.states,
                relaxations_plus=rollout.relaxations_plus,
                relaxations_minus=rollout.relaxations_minus,
                relax_to_go_state=relax_to_go_state,
            ),
            in_relax_to_go_state,
        )

        # Create the model states for the output
        # the policy state is not updated, since the projection needs the
        # policy to be reproducible for convergence
        # the relax_to_go and cost_to_go states are updated with the new values
        model_states = ModelStates(
            policy=in_policy_state,
            relax_to_go=out_relax_to_go_state,
            cost_to_go=out_cost_to_go_state,
        )

        # Calculate the losses
        policy_loss, (policy_val_res_grad, policy_interv_grad) = (
            jax.value_and_grad(
                self.policy_loss,
                argnums=(0, 1),
            )(cost_residuals, rollout.interventions)
        )

        cost_to_go_loss, cost_val_res_grad = jax.value_and_grad(
            self.cost_to_go_loss,
        )(cost_residuals)

        cost_encountered = jnp.mean(
            self.sys_costs(rollout.states, rollout.actions)
        )

        relax_to_go_loss, relax_rel_res_grad = jax.value_and_grad(
            self.relax_to_go_loss
        )(relax_residuals)

        loss = ModelLoss(
            policy=policy_loss,
            cost_to_go=cost_to_go_loss,
            cost_encountered=cost_encountered,
            relax_to_go=relax_to_go_loss,
        )

        ## Calculate the gradients for the policy state ##
        grad = dict()
        policy_cost_to_go_state_grad = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x), in_cost_to_go_state
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
            relaxation_terminal=jnp.zeros_like(rollout.relaxation_terminal),
        )
        policy_info_grad = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x), info
        )  # the outgoing info does not affect the policy loss
        policy_policy_params_grad, _ = rollout_vjp(
            (policy_rollout_grad, policy_info_grad)
        )

        ## Calculate the gradients for the cost_to_go state ##
        cost_cost_to_go_state_grad = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x), in_cost_to_go_state
        )  # the outgoing cost_to_go_state does not affect the cost loss
        (_, _, cost_cost_to_go_state_grad) = cost_res_vjp(
            (cost_val_res_grad, cost_cost_to_go_state_grad)
        )
        cost_cost_to_go_params_grad = nnx.filter_state(
            cost_cost_to_go_state_grad, nnx.Param
        )

        ## Calculate the gradients for the relax_to_go state ##
        relax_relax_to_go_state_grad = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x), in_relax_to_go_state
        )  # the outgoing relax_to_go_state does not affect the relax loss
        (relax_relax_to_go_state_grad,) = relax_res_vjp(
            (relax_rel_res_grad, relax_relax_to_go_state_grad)
        )
        # We purposfully do not propagate gradients through the rollout here!
        # We dont want to change the rollout to improve the relax to go loss.
        relax_relax_to_go_params_grad = nnx.filter_state(
            relax_relax_to_go_state_grad, nnx.Param
        )

        grad = ModelGrad(
            policy=policy_policy_params_grad,
            cost_to_go=cost_cost_to_go_params_grad,
            relax_to_go=relax_relax_to_go_params_grad,
        )

        # Compute the mean of each jax array in info
        info = OptimizationInfo(
            primal_residual=jnp.mean(info.primal_residual),
            dual_residual=jnp.mean(info.dual_residual),
            iterations_lin=jnp.mean(info.iterations_lin),
            iterations_nonlin=jnp.mean(info.iterations_nonlin),
        )

        return loss, grad, model_states, info
