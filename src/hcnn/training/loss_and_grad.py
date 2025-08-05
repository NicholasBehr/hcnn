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
    gae_lambda: float = 0.8
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
        gae_lambda: float = 0.8,
        policy_intervention_cost: float = 0.5,
    ):
        return LossAndGrad(
            projection=projection,
            solver=solver,
            sys_costs=sys_costs,
            cost_to_go_graphdef=nnx.split(cost_to_go)[0],
            relax_to_go_graphdef=nnx.split(relax_to_go)[0],
            td_gamma=td_gamma,
            gae_lambda=gae_lambda,
            policy_intervention_cost=policy_intervention_cost,
        )

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
            OptimizationInfo: Information about the optimization process.
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

    def policy_costs(
        self,
        rollout: Rollout,
    ):
        """
        Computes the costs for the entire horizon based on the states,
        actions, and interventions.

        Args:
            states (jax.Array):  shape (..., horizon, state_dim).
            actions (jax.Array): shape (..., horizon, action_dim).
            interventions (jax.Array): shape (..., horizon, action_dim).

        Returns:
            jax.Array: The computed costs for the entire horizon.
                shape (..., horizon).
        """
        # sys_costs returns (..., horizon)
        cost_encountered = self.sys_costs(rollout.states, rollout.actions)

        # Sum interventions over action_dim to get (..., horizon)
        intervention_cost = jnp.sum(rollout.interventions, axis=-1)
        return (
            cost_encountered * (1.0 - self.policy_intervention_cost)
            + intervention_cost * self.policy_intervention_cost
        )

    def projection_costs(
        self,
        rollout: Rollout,
    ):
        """
        Computes the projection costs based on the interventions and
        relaxations.

        Args:
            interventions (jax.Array): shape (..., horizon, action_dim).
            relaxations_plus (jax.Array): shape (..., horizon, constraint_dim).
            relaxations_minus (jax.Array): shape (..., horizon, constraint_dim).

        Returns:
            jax.Array: shape (..., horizon).
        """
        # Sum over action_dim for quadratic cost
        quad = 2 * jnp.sum(rollout.interventions**2, axis=-1)

        # Sum over constraint_dim for linear cost
        linear = self.projection.relaxation_penalty * jnp.sum(
            (
                jnp.abs(rollout.relaxations_plus)
                + jnp.abs(rollout.relaxations_minus)
            ),
            axis=-1,
        )

        return quad + linear

    def policy_loss(
        self,
        rollout: Rollout,
        cost_to_go_graphdef: nnx.GraphDef,
        cost_to_go_state: nnx.State,
    ):
        """
        Evaluates the loss function used to train the policy.
        As the minimization of the n-step TD difference of discounted costs
        and cost-to-go value function.

        Args:
            rollout (Rollout):
                The rollout containing states, actions, interventions.
            cost_to_go_graphdef (nnx.GraphDef):
                Graph definition of cost-to-go model.
            cost_to_go_state (nnx.State):
                Incoming state.

        Returns:
            jax.Array: Loss, shape ().
            nnx.State: Outgoing state of the cost_to_go model.
        """
        horizon = self.projection.horizon

        # Compute policy costs from rollout
        policy_costs = self.policy_costs(rollout)

        # Create discount factors: shape (horizon,)
        discount_factors = self.td_gamma ** jnp.arange(horizon)

        # Compute discounted cumulative costs from each timestep
        # policy_costs shape: (..., horizon)
        # discount_factors shape: (horizon,)
        discounted_costs = policy_costs * discount_factors
        # Sum over horizon to get n-step discounted return
        discounted_sum = jnp.sum(discounted_costs, axis=-1)  # shape: (...)

        # Get cost-to-go model
        cost_to_go_model = nnx.merge(cost_to_go_graphdef, cost_to_go_state)

        # Extract initial and terminal states from rollout
        initial_states = rollout.states[..., 0, :]  # shape: (..., state_dim)
        terminal_states = rollout.states[..., -1, :]  # shape: (..., state_dim)

        # Flatten states for model evaluation
        original_shape = initial_states.shape
        batch_dims = original_shape[:-1]
        state_dim = original_shape[-1]

        # Evaluate cost-to-go for both initial and terminal states
        all_states = jnp.concatenate(
            [
                initial_states.reshape(-1, state_dim),
                terminal_states.reshape(-1, state_dim),
            ],
            axis=0,
        )
        all_costs = cost_to_go_model(all_states)
        cost_to_go_state = nnx.state(cost_to_go_model)

        # Split back into initial and terminal costs
        batch_size = initial_states.reshape(-1, state_dim).shape[0]
        initial_costs = all_costs[:batch_size].reshape(*batch_dims)
        terminal_costs = all_costs[batch_size:].reshape(*batch_dims)

        # Compute TD difference:
        # discounted sum + discounted terminal value - initial value
        td_difference = (
            discounted_sum
            + self.td_gamma**horizon * terminal_costs
            - jax.lax.stop_gradient(initial_costs)
        )

        return jnp.mean(td_difference), cost_to_go_state

    def advantage_loss(
        self,
        to_go_params: nnx.State,
        to_go_non_params: nnx.State,
        to_go_graphdef: nnx.GraphDef,
        states: jax.Array,
        stage_costs: jax.Array,
        converged_flags: jax.Array,
    ):
        """
        Evaluates the loss function used to train the x-to-go weights.
        Args:
            to_go_params (nnx.State): Parameters of the to-go model.
            to_go_non_params (nnx.State): Non-parameters of the to-go model.
            to_go_graphdef (nnx.GraphDef): Graph definition of the to-go model.
            states (jax.Array): shape (..., horizon, state_dim).
            stage_costs (jax.Array): shape (..., horizon).
        Returns:
            jax.Array: Loss, shape ().
            nnx.State: Outgoing state of the to_go model.
        """
        # Hydrate model
        model = nnx.merge(to_go_graphdef, to_go_params, to_go_non_params)

        # Current states shape is (..., horizon, state_dim)
        original_shape = states.shape
        batch_dims = original_shape[:-2]
        horizon, state_dim = original_shape[-2:]

        # Reshape to (batch_size * horizon, state_dim)
        flat_states = states.reshape(-1, state_dim)

        # Apply the model to the reshaped states
        flat_model_eval = model(flat_states)

        # Update the to_go_state with the new model state
        to_go_state = nnx.state(model)

        # Reshape evaluation back to (..., horizon)
        model_eval = flat_model_eval.reshape(*batch_dims, horizon)

        # Compute the temporal difference (TD) errors
        residuals = td_residuals(
            rewards=stage_costs, values=model_eval, td_gamma=self.td_gamma
        )

        # Calculate the generalized advantage estimation (GAE)
        advantage = generalized_advantage_t(
            td_residuals=residuals,
            td_gamma=self.td_gamma,
            gae_lambda=self.gae_lambda,
        )

        # eliminate the residuals of converged rollouts
        return jnp.mean(
            jnp.mean(advantage**2, -1) * converged_flags
        ), to_go_state

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
        # (0) Unpack the model states
        policy_state = model_states.policy
        cost_to_go_state = model_states.cost_to_go
        relax_to_go_state = model_states.relax_to_go

        # (1) Calculate the rollout (does not modify the model states!)
        policy_params, policy_non_params = nnx.filter_state(
            policy_state, nnx.Param, ...
        )
        relax_to_go_params, relax_to_go_non_params = nnx.filter_state(
            relax_to_go_state, nnx.Param, ...
        )
        rollout, rollout_vjp, info = jax.vjp(
            lambda policy_params: self.calculate_rollout(
                policy_params=policy_params,
                relax_to_go_params=relax_to_go_params,
                policy_non_params=policy_non_params,
                relax_to_go_non_params=relax_to_go_non_params,
                initial_states=initial_states,
                jax_keys=jax_keys,
            ),
            policy_params,
            has_aux=True,
        )

        # (2.1) Calculate the policy costs
        policy_costs, policy_costs_vjp = jax.vjp(self.policy_costs, rollout)

        # (2.2) Calculate the projection costs
        projection_costs = self.projection_costs(rollout)

        # (3.1) calculate the policy loss (modifies the cost_to_go state)
        (policy_loss, cost_to_go_state), dp_drollout = jax.value_and_grad(
            self.policy_loss, has_aux=True
        )(
            rollout,
            self.cost_to_go_graphdef,
            cost_to_go_state,
        )

        # (3.2) calculate the cost_to_go loss (modifies the cost_to_go state)
        cost_to_go_params, cost_to_go_non_params = nnx.filter_state(
            cost_to_go_state, nnx.Param, ...
        )
        (cost_to_go_loss, cost_to_go_state), dc_din_cost_to_go_params = (
            jax.value_and_grad(self.advantage_loss, has_aux=True)(
                cost_to_go_params,
                cost_to_go_non_params,
                self.cost_to_go_graphdef,
                rollout.states,
                policy_costs,
                info.converged_flag,
            )
        )

        # (3.3) calculate the relax_to_go loss (modifies the relax_to_go state)
        relax_to_go_params, relax_to_go_non_params = nnx.filter_state(
            relax_to_go_state, nnx.Param, ...
        )
        (relax_to_go_loss, relax_to_go_state), dr_din_relax_to_go_params = (
            jax.value_and_grad(self.advantage_loss, has_aux=True)(
                relax_to_go_params,
                relax_to_go_non_params,
                self.relax_to_go_graphdef,
                rollout.states,
                projection_costs,
                info.converged_flag,
            )
        )

        # (4.1) Calculate the gradients for the policy params
        dp_dpolicy_params = rollout_vjp(dp_drollout)[0]

        # (5.1) Construct loss output
        loss = ModelLoss(
            policy=policy_loss,
            cost_to_go=cost_to_go_loss,
            relax_to_go=relax_to_go_loss,
        )

        # (5.2) Construct grad output
        grad = ModelGrad(
            policy=dp_dpolicy_params,
            cost_to_go=dc_din_cost_to_go_params,
            relax_to_go=dr_din_relax_to_go_params,
        )

        # (5.3) Construct model states output
        model_states = ModelStates(
            policy=policy_state,  # the policy state is not altered!
            relax_to_go=relax_to_go_state,
            cost_to_go=cost_to_go_state,
        )

        # (5.4) Construct info output
        info = OptimizationInfo(
            primal_residual=jnp.mean(info.primal_residual),
            dual_residual=jnp.mean(info.dual_residual),
            iterations_lin=jnp.mean(info.iterations_lin),
            iterations_nonlin=jnp.mean(info.iterations_nonlin),
            converged_flag=jnp.mean(info.converged_flag),
        )

        return loss, grad, model_states, info
