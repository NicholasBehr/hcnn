import jax.numpy as jnp
import optax
from flax import nnx, struct

from hcnn.optimization import (
    OptimizationSolverBase,
    OptimizationSolverOSQP,
    OptimizationSolverShallow,
)
from hcnn.policy import PolicyBase, PolicyMLP
from hcnn.projection import (
    ProjectionBase,
    ProjectionNoSubsNoTerm,
    ProjectionNoSubsTerm,
)
from hcnn.system_constraints import (
    SystemConstraintsBase,
    SystemConstraintsBox,
)
from hcnn.system_costs import (
    SystemCostsBase,
    SystemCostsDoublePendulum,
    SystemCostsQuadratic,
)
from hcnn.system_dynamics import (
    SystemDynamicsBase,
    SystemDynamicsDoublePendulum,
    SystemDynamicsLinear,
)
from hcnn.system_dynamics.system_dynamics_bike import SystemDynamicsBike
from hcnn.training.dropout_mlp import DropoutMLP
from hcnn.training.loss_and_grad import LossAndGrad
from wandb.sdk.wandb_config import Config as WandbConfig

OPTIMIZERS = {
    "adabelief": optax.adabelief,
    "adagrad": optax.adagrad,
    "adam": optax.adam,
    "nadam": optax.nadam,
    "adamw": optax.adamw,
    "nadamw": optax.nadamw,
}


@struct.dataclass
class TrainBundle(nnx.Module):
    policy: PolicyBase
    cost_to_go: DropoutMLP
    relax_to_go: DropoutMLP
    loss_and_grad: LossAndGrad
    opt_policy: nnx.Optimizer
    opt_cost: nnx.Optimizer
    opt_relax: nnx.Optimizer
    metrics: nnx.MultiMetric
    rngs: nnx.Rngs

    @staticmethod
    def build_x_to_go(
        config: WandbConfig,
        state_dim: int,
        rngs: nnx.Rngs,
        periodic_states: frozenset[int] = frozenset(),
    ):
        return DropoutMLP(
            in_features=state_dim,
            out_features=1,
            hidden_features=[config.x_to_go_hidden_dim]
            * config.x_to_go_hidden_depth,
            rngs=rngs,
            activation=config.x_to_go_activation,
            dropout_rate=config.x_to_go_dropout_rate,
            periodic_in_features=periodic_states,
        )

    @staticmethod
    def build_double_integrator(config: WandbConfig):
        sys_dynamics = SystemDynamicsLinear.factory(
            state_matrix=jnp.array([[1, 1], [0, 1]]),
            action_matrix=jnp.array([[0.5], [1]]),
        )
        sys_costs = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[5, 0], [0, 5]]),
            action_cost_matrix=jnp.array([[1]]),
        )
        action_min = jnp.array([-1.0])
        action_max = jnp.array([1.0])
        sys_constraints = SystemConstraintsBox.factory(
            state_l=jnp.array([-10, -10]),
            state_u=jnp.array([+10, +10]),
            action_l=action_min,
            action_u=action_max,
        )
        rngs = nnx.Rngs(config.seed)
        policy = PolicyMLP.factory(
            state_dim=2,
            action_dim=1,
            hidden_features=[config.policy_hidden_dim]
            * config.policy_hidden_depth,
            rngs=rngs,
            activation=config.policy_activation,
            dropout_rate=config.policy_dropout_rate,
            action_min=action_min,
            action_max=action_max,
        )
        cost_to_go = TrainBundle.build_x_to_go(
            config=config,
            state_dim=2,
            rngs=rngs,
        )
        relax_to_go = nnx.clone(cost_to_go)

        return (
            sys_dynamics,
            sys_costs,
            sys_constraints,
            rngs,
            policy,
            cost_to_go,
            relax_to_go,
        )

    @staticmethod
    def build_double_pendulum(config: WandbConfig):
        sys_dynamics = SystemDynamicsDoublePendulum.factory(dt=0.05)
        sys_costs = SystemCostsDoublePendulum.factory(
            dynamics=sys_dynamics,
            reference_state=jnp.array([jnp.pi, 0.0, 0.0, 0.0]),
            position_weights=jnp.array([0.0, 0.0, 5.0, 5.0]),
            action_weights=jnp.zeros(1),
        )
        action_min = jnp.array([-1.0])
        action_max = jnp.array([1.0])
        sys_constraints = SystemConstraintsBox.factory(
            state_l=jnp.array([-jnp.inf, -jnp.inf, -2 * jnp.pi, -2 * jnp.pi]),
            state_u=jnp.array([+jnp.inf, +jnp.inf, +2 * jnp.pi, +2 * jnp.pi]),
            action_l=action_min,
            action_u=action_max,
        )
        rngs = nnx.Rngs(config.seed)
        periodic_states = frozenset({0, 1})  # Pendulum angles
        policy = PolicyMLP.factory(
            state_dim=4,
            action_dim=1,
            hidden_features=[config.policy_hidden_dim]
            * config.policy_hidden_depth,
            rngs=rngs,
            activation=config.policy_activation,
            dropout_rate=config.policy_dropout_rate,
            periodic_states=periodic_states,
            action_min=action_min,
            action_max=action_max,
        )
        cost_to_go = TrainBundle.build_x_to_go(
            config=config,
            state_dim=4,
            rngs=rngs,
            periodic_states=periodic_states,
        )
        relax_to_go = nnx.clone(cost_to_go)

        return (
            sys_dynamics,
            sys_costs,
            sys_constraints,
            rngs,
            policy,
            cost_to_go,
            relax_to_go,
        )

    @staticmethod
    def build_bike(config: WandbConfig):
        sys_dynamics = SystemDynamicsBike.factory(
            wheelbase=0.5,
            max_steering=0.6,
            dt=0.1,
        )
        sys_costs = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.diag(jnp.array([10.0, 10.0, 0.0, 0.0])),
            action_cost_matrix=jnp.diag(jnp.array([1.0, 1.0])),
            reference_state=jnp.array([0.0, 0.0, 0.0, 0.0]),
            reference_action=jnp.array([0.0, 0.0]),
        )
        action_min = jnp.array([-1.0, -0.5])  # [acceleration, steering]
        action_max = jnp.array([1.0, 0.5])  # [acceleration, steering]
        sys_constraints = SystemConstraintsBox.factory(
            state_l=jnp.array([-jnp.inf, -jnp.inf, -jnp.pi, 0]),
            state_u=jnp.array([+jnp.inf, +jnp.inf, +jnp.pi, +jnp.inf]),
            action_l=action_min,
            action_u=action_max,
        )
        rngs = nnx.Rngs(config.seed)
        periodic_states = frozenset({2})  # Bike heading angle
        policy = PolicyMLP.factory(
            state_dim=sys_dynamics.state_dim,
            action_dim=sys_dynamics.action_dim,
            hidden_features=[config.policy_hidden_dim]
            * config.policy_hidden_depth,
            rngs=rngs,
            activation=config.policy_activation,
            dropout_rate=config.policy_dropout_rate,
            periodic_states=periodic_states,
            action_min=action_min,
            action_max=action_max,
        )
        cost_to_go = TrainBundle.build_x_to_go(
            config=config,
            state_dim=sys_dynamics.state_dim,
            rngs=rngs,
            periodic_states=periodic_states,
        )
        relax_to_go = nnx.clone(cost_to_go)

        return (
            sys_dynamics,
            sys_costs,
            sys_constraints,
            rngs,
            policy,
            cost_to_go,
            relax_to_go,
        )

    @staticmethod
    def build_projection_operator(
        config: WandbConfig,
        policy: PolicyBase,
        relax_to_go: DropoutMLP,
        sys_dynamics: SystemDynamicsBase,
        sys_constraints: SystemConstraintsBase,
    ) -> ProjectionBase:
        if config.projection_name == "no_substitution_no_terminal":
            return ProjectionNoSubsNoTerm.factory(
                policy=policy,
                relax_to_go=relax_to_go,
                sys_dynamics=sys_dynamics,
                sys_constraints=sys_constraints,
                horizon=config.projection_horizon,
                td_gamma=config.td_gamma,
                relaxation_penalty=config.projection_relaxation_penalty,
            )
        elif config.projection_name == "no_substitution_terminal":
            return ProjectionNoSubsTerm.factory(
                policy=policy,
                relax_to_go=relax_to_go,
                sys_dynamics=sys_dynamics,
                sys_constraints=sys_constraints,
                horizon=config.projection_horizon,
                td_gamma=config.td_gamma,
                relaxation_penalty=config.projection_relaxation_penalty,
            )
        else:
            raise ValueError(f"Unknown projection: {config.projection_name}")

    @staticmethod
    def build_solver(config: WandbConfig) -> OptimizationSolverBase:
        if config.solver_name == "osqp":
            solver = OptimizationSolverOSQP
        elif config.solver_name == "shallow":
            solver = OptimizationSolverShallow
        else:
            raise ValueError(f"Unknown system: {config.system_name}")

        return solver(
            alpha=config.solver_alpha,
            sigma=config.solver_sigma,
            rho_equality=config.solver_rho_equality,
            rho_inequality=config.solver_rho_inequality,
            max_iter_nonlin=config.solver_max_iter_nonlin,
            max_iter_lin=config.solver_max_iter_lin,
            tol=config.solver_tol,
        )

    @staticmethod
    def build_loss_and_grad(
        config: WandbConfig,
        projection: ProjectionBase,
        solver: OptimizationSolverBase,
        sys_costs: SystemCostsBase,
        cost_to_go: DropoutMLP,
        relax_to_go: DropoutMLP,
    ) -> LossAndGrad:
        return LossAndGrad.factory(
            projection=projection,
            solver=solver,
            sys_costs=sys_costs,
            cost_to_go=cost_to_go,
            relax_to_go=relax_to_go,
            td_gamma=config.td_gamma,
            gae_lambda=config.gae_lambda,
            policy_intervention_cost=config.policy_intervention_cost,
        )

    @staticmethod
    def build_optimizer(model, kind: str, learning_rate):
        return nnx.Optimizer(
            model, OPTIMIZERS[kind](learning_rate=learning_rate)
        )

    @staticmethod
    def factory_from_config(
        config: WandbConfig,
    ) -> "TrainBundle":
        # Build the system components
        system_builders = {
            "double_integrator": TrainBundle.build_double_integrator,
            "double_pendulum": TrainBundle.build_double_pendulum,
            "bike": TrainBundle.build_bike,
        }
        builder = system_builders.get(config.system_name)

        if not builder:
            raise ValueError(f"Unknown system: {config.system_name}")

        (
            sys_dynamics,
            sys_costs,
            sys_constraints,
            rngs,
            policy,
            cost_to_go,
            relax_to_go,
        ) = builder(config)

        # Build the projection operator
        projection = TrainBundle.build_projection_operator(
            config,
            policy=policy,
            relax_to_go=relax_to_go,
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
        )

        # Build the solver
        solver = TrainBundle.build_solver(config)

        # Build the loss and gradient computation
        loss_and_grad = TrainBundle.build_loss_and_grad(
            config=config,
            projection=projection,
            solver=solver,
            sys_costs=sys_costs,
            cost_to_go=cost_to_go,
            relax_to_go=relax_to_go,
        )

        # Build optimizers
        opt_policy = TrainBundle.build_optimizer(
            policy,
            config.optimizer_policy_name,
            config.optimizer_policy_learning_rate,
        )
        opt_cost = TrainBundle.build_optimizer(
            cost_to_go,
            config.optimizer_cost_name,
            config.optimizer_cost_learning_rate,
        )
        opt_relax = TrainBundle.build_optimizer(
            relax_to_go,
            config.optimizer_relax_name,
            config.optimizer_relax_learning_rate,
        )

        # Build optax metrics
        metrics = nnx.MultiMetric(
            # model evaluation metrics
            policy_loss=nnx.metrics.Average("policy_loss"),
            cost_to_go_loss=nnx.metrics.Average("cost_to_go_loss"),
            relax_to_go_loss=nnx.metrics.Average("relax_to_go_loss"),
            # solver metrics
            primal_residual=nnx.metrics.Average("primal_residual"),
            dual_residual=nnx.metrics.Average("dual_residual"),
            residual=nnx.metrics.Average("residual"),
            solver_iter_lin=nnx.metrics.Average("solver_iter_lin"),
            solver_iter_nonlin=nnx.metrics.Average("solver_iter_nonlin"),
            solver_converged=nnx.metrics.Average("solver_converged"),
            # policy training metrics
            mean_policy_gradient=nnx.metrics.Average("mean_policy_gradient"),
            max_policy_gradient=nnx.metrics.Average("max_policy_gradient"),
        )

        return TrainBundle(
            policy=policy,
            cost_to_go=cost_to_go,
            relax_to_go=relax_to_go,
            loss_and_grad=loss_and_grad,
            opt_policy=opt_policy,
            opt_cost=opt_cost,
            opt_relax=opt_relax,
            metrics=metrics,
            rngs=rngs,
        )
