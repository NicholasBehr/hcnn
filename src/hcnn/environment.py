import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct

from hcnn.optimization import OptimizationSolverBase
from hcnn.policy import PolicyBase
from hcnn.projection import ProjectionBase, ProjectionPhi, RolloutPlan
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_cost import SystemCostBase
from hcnn.system_dynamics import SystemDynamicsBase
from hcnn.training.dropout_mlp import DropoutMLP
from hcnn.training.generalized_advantage import generalized_advantage_1, generalized_advantage_t, td_residuals


@struct.dataclass
class Environment:
    ## SYSTEM DEFINITION
    sys_constraints: SystemConstraintsBase
    sys_cost: SystemCostBase
    sys_dynamics: SystemDynamicsBase

    ## PROJECTION
    projection: ProjectionBase
    solver: OptimizationSolverBase

    # PARAMETERS
    td_gamma: float = 0.9
    gae_lambda: float = 0.99

    def init(
        self,
        policy: PolicyBase,
        value_to_go: DropoutMLP,
        relax_to_go: DropoutMLP,
        rngs: nnx.Rngs,
    ):
        # optimizers
        policy_optimizer = nnx.Optimizer(policy, optax.nadamw(learning_rate=1e-3))
        value_to_go_optimizer = nnx.Optimizer(value_to_go, optax.nadamw(learning_rate=1e-2))
        relax_to_go_optimizer = nnx.Optimizer(relax_to_go, optax.nadamw(learning_rate=1e-2))

        # metrics
        metrics = nnx.MultiMetric(
            policy_loss=nnx.metrics.Average("policy_loss"),
            value_to_go_loss=nnx.metrics.Average("value_to_go_loss"),
            relax_to_go_loss=nnx.metrics.Average("relax_to_go_loss"),
            cost=nnx.metrics.Average("cost"),
        )

        graphdef, state = nnx.split(
            (
                self,
                policy,
                policy_optimizer,
                value_to_go,
                value_to_go_optimizer,
                relax_to_go,
                relax_to_go_optimizer,
                metrics,
                rngs,
            )
        )

        return graphdef, state

    @staticmethod
    @jax.jit
    def train_epoch(graphdef, state):
        def scan_fn(carry, _):
            return Environment.train_step(graphdef, carry), None

        state, _ = jax.lax.scan(scan_fn, state, None, length=50)

        (
            self,
            policy,
            policy_optimizer,
            value_to_go,
            value_to_go_optimizer,
            relax_to_go,
            relax_to_go_optimizer,
            metrics,
            rngs,
        ) = nnx.merge(graphdef, state)

        jax.debug.print("{}", metrics.compute())
        metrics.reset()

        # Return the updated state
        state = nnx.state(
            (
                self,
                policy,
                policy_optimizer,
                value_to_go,
                value_to_go_optimizer,
                relax_to_go,
                relax_to_go_optimizer,
                metrics,
                rngs,
            )
        )
        return state

    @staticmethod
    @jax.jit
    def train_step(graphdef, state):
        (
            self,
            policy,
            policy_optimizer,
            value_to_go,
            value_to_go_optimizer,
            relax_to_go,
            relax_to_go_optimizer,
            metrics,
            rngs,
        ) = nnx.merge(graphdef, state)

        # Determine training 'location'
        initial_state = jax.random.uniform(rngs(), (self.sys_dynamics.state_dim,), minval=-6, maxval=6)
        jax_key = rngs()

        # Sample a rollout
        rollout, rollout_vjp = jax.vjp(
            lambda policy_params: self.calculate_rollout(
                initial_state=initial_state,
                policy_params=policy_params,
                jax_key=jax_key,
            ),
            nnx.state(policy, nnx.Param),
        )

        # Calculate value residuals
        value_to_go_graphdef, value_to_go_param, value_to_go_non_param = nnx.split(value_to_go, nnx.Param, ...)
        value_residuals, value_residuals_vjp = jax.vjp(
            lambda value_to_go_params, rollout: self.calculate_value_residuals(
                value_to_go=nnx.merge(value_to_go_graphdef, value_to_go_params, value_to_go_non_param),
                rollout=rollout,
            ),
            value_to_go_param,
            rollout,
        )

        # Calculate relax residuals
        relax_to_go_graphdef, relax_to_go_param, relax_to_go_non_param = nnx.split(relax_to_go, nnx.Param, ...)
        relax_residuals, relax_residuals_vjp = jax.vjp(
            lambda relax_to_go_params: self.calculate_relax_residuals(
                relax_to_go=nnx.merge(relax_to_go_graphdef, relax_to_go_params, relax_to_go_non_param),
                rollout=rollout,
            ),
            relax_to_go_param,
        )

        def value_to_go_loss_fn(value_residuals: jax.Array):
            advantages_t = generalized_advantage_t(
                deltas=value_residuals,
                td_gamma=self.td_gamma,
                gae_lambda=self.gae_lambda,
            )
            return jnp.mean(advantages_t**2)

        def relax_to_go_loss_fn(relax_residuals: jax.Array):
            advantages_t = generalized_advantage_t(
                deltas=relax_residuals,
                td_gamma=self.td_gamma,
                gae_lambda=self.gae_lambda,
            )
            return jnp.mean(advantages_t**2)

        def policy_loss_fn(value_residuals: jax.Array, rollout: RolloutPlan):
            advantage_1 = generalized_advantage_1(
                deltas=value_residuals,
                td_gamma=self.td_gamma,
                gae_lambda=self.gae_lambda,
            )

            loss = jnp.mean(rollout.interventions**2) - advantage_1
            return loss

        # Calculate the value_to_go loss
        value_to_go_loss, dval_dres = jax.value_and_grad(value_to_go_loss_fn)(value_residuals)
        dval_dparam = value_residuals_vjp(dval_dres)[0]

        # Calculate the relax_to_go loss
        relax_to_go_loss, drelax_dres = jax.value_and_grad(relax_to_go_loss_fn)(relax_residuals)
        drelax_dparam = relax_residuals_vjp(drelax_dres)[0]

        # Calculate the policy loss
        policy_loss, (dpol_dres, dpol_droll1) = jax.value_and_grad(
            policy_loss_fn,
            argnums=(0, 1),
        )(value_residuals, rollout)
        dpol_droll2 = value_residuals_vjp(dpol_dres)[1]
        dpol_droll = jax.tree_util.tree_map(lambda x, y: x + y, dpol_droll1, dpol_droll2)
        dpol_dparam = rollout_vjp(dpol_droll)[0]

        # Update the value_to_go and policy parameters
        value_to_go_optimizer.update(dval_dparam)
        policy_optimizer.update(dpol_dparam)
        relax_to_go_optimizer.update(drelax_dparam)

        metrics.update(
            policy_loss=policy_loss,
            value_to_go_loss=value_to_go_loss,
            relax_to_go_loss=relax_to_go_loss,
            cost=jnp.mean(jax.vmap(self.sys_cost)(rollout.states, rollout.actions)),
        )

        return nnx.state(
            (
                self,
                policy,
                policy_optimizer,
                value_to_go,
                value_to_go_optimizer,
                relax_to_go,
                relax_to_go_optimizer,
                metrics,
                rngs,
            )
        )

    def calculate_rollout(
        self,
        initial_state: jax.Array,
        policy_params: nnx.State,
        jax_key: jax.Array,
    ):
        phi = ProjectionPhi(
            initial_state=initial_state,
            policy_params=policy_params,
            jax_key=jax_key,
        )
        x_star = self.solver.solve(self.projection, phi)
        rollout = self.projection.get_rollout(x_star)
        return rollout

    def calculate_value_residuals(self, value_to_go: DropoutMLP, rollout: RolloutPlan):
        costs = jax.vmap(self.sys_cost)(rollout.states, rollout.actions)
        rewards = -costs
        values = value_to_go(rollout.states).squeeze()

        # Compute the temporal difference (TD) errors
        value_residuals = td_residuals(rewards=rewards, values=values, td_gamma=self.td_gamma)
        return value_residuals

    def calculate_relax_residuals(self, relax_to_go: DropoutMLP, rollout: RolloutPlan):
        rewards = rollout.relaxations
        values = relax_to_go(rollout.states).squeeze()

        # Compute the temporal difference (TD) errors
        value_residuals = td_residuals(rewards=rewards, values=values, td_gamma=self.td_gamma)
        return value_residuals
