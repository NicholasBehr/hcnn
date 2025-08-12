import jax.numpy as jnp
import optax
from flax import nnx, struct

from hcnn.optimization import (
    OptimizationSolverBase,
    OptimizationSolverOSQP,
    OptimizationSolverShallow,
)
from hcnn.projection import ProjectionLayer
from hcnn.sampler import SamplerUniform
from hcnn.state_to_rollout import StateToRolloutActions, StateToRolloutBase
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
from wandb.sdk.wandb_config import Config as WandbConfig


@struct.dataclass
class System:
    sys_dynamics: SystemDynamicsBase
    sys_costs: SystemCostsBase
    sys_constraints: SystemConstraintsBase
    state_min: jnp.ndarray
    state_max: jnp.ndarray
    periodic_states: frozenset[int] | None = None
    action_min: jnp.ndarray | None = None
    action_max: jnp.ndarray | None = None


@struct.dataclass
class TrainBundle(nnx.Module):
    system: System
    sampler: SamplerUniform
    state_to_rollout: StateToRolloutBase
    optimizer: nnx.Optimizer
    projection_layer: ProjectionLayer
    metrics: nnx.MultiMetric

    @staticmethod
    def build_double_integrator(
        config: WandbConfig,
    ) -> System:
        sys_dynamics = SystemDynamicsLinear.factory(
            state_matrix=jnp.array([[1, 1], [0, 1]]),
            action_matrix=jnp.array([[0.5], [1]]),
        )
        sys_costs = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[10, 0], [0, 10]]),
            action_cost_matrix=jnp.array([[1]]),
        )
        state_min = jnp.array([-10.0, -10.0])
        state_max = jnp.array([+10.0, +10.0])
        action_min = jnp.array([-1.0])
        action_max = jnp.array([+1.0])
        sys_constraints = SystemConstraintsBox.factory(
            state_l=state_min,
            state_u=state_max,
            action_l=action_min,
            action_u=action_max,
        )

        return System(
            sys_dynamics=sys_dynamics,
            sys_costs=sys_costs,
            sys_constraints=sys_constraints,
            state_min=state_min,
            state_max=state_max,
            action_min=action_min,
            action_max=action_max,
        )

    @staticmethod
    def build_double_pendulum(config: WandbConfig) -> System:
        # State: [theta1, theta2, dtheta1, dtheta2]
        # Action: [torque1]
        sys_dynamics = SystemDynamicsDoublePendulum.factory(dt=0.05)  # 50 ms
        sys_costs = SystemCostsDoublePendulum.factory(
            dynamics=sys_dynamics,
            reference_state=jnp.array([jnp.pi, 0.0, 0.0, 0.0]),
            position_weights=jnp.array([0.0, 0.0, 5.0, 5.0]),
            action_weights=jnp.zeros(1),
        )
        state_min = jnp.array([-jnp.pi, -jnp.pi, -2 * jnp.pi, -2 * jnp.pi])
        state_max = jnp.array([+jnp.pi, +jnp.pi, +2 * jnp.pi, +2 * jnp.pi])
        periodic_states = frozenset({0, 1})  # Pendulum angles
        action_min = jnp.array([-1.0])
        action_max = jnp.array([1.0])
        sys_constraints = SystemConstraintsBox.factory(
            state_l=jnp.array([-jnp.inf, -jnp.inf, -2 * jnp.pi, -2 * jnp.pi]),
            state_u=jnp.array([+jnp.inf, +jnp.inf, +2 * jnp.pi, +2 * jnp.pi]),
            action_l=action_min,
            action_u=action_max,
        )

        return System(
            sys_dynamics=sys_dynamics,
            sys_costs=sys_costs,
            sys_constraints=sys_constraints,
            state_min=state_min,
            state_max=state_max,
            periodic_states=periodic_states,
            action_min=action_min,
            action_max=action_max,
        )

    @staticmethod
    def build_bike(config: WandbConfig) -> System:
        # State: [x, y, heading, velocity]
        # Action: [acceleration, steering]
        sys_dynamics = SystemDynamicsBike.factory(
            wheelbase=0.5,
            max_steering=0.6,
            dt=0.25,  # 250 ms
        )
        sys_costs = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.diag(jnp.array([10.0, 10.0, 0.0, 0.0])),
            action_cost_matrix=jnp.diag(jnp.array([1.0, 1.0])),
        )
        state_min = jnp.array([-10, -10, -jnp.inf, -1.0])
        state_max = jnp.array([+10, +10, +jnp.inf, +5.0])
        periodic_states = frozenset({2})  # Bike heading angle
        action_min = jnp.array([-2.5, -0.5])
        action_max = jnp.array([+2.5, +0.5])
        sys_constraints = SystemConstraintsBox.factory(
            state_l=state_min,
            state_u=state_max,
            action_l=action_min,
            action_u=action_max,
        )

        return System(
            sys_dynamics=sys_dynamics,
            sys_costs=sys_costs,
            sys_constraints=sys_constraints,
            state_min=state_min,
            state_max=state_max,
            periodic_states=periodic_states,
            action_min=action_min,
            action_max=action_max,
        )

    @staticmethod
    def build_sampler(config: WandbConfig, system: System) -> SamplerUniform:
        lower_bound = system.state_min
        upper_bound = system.state_max
        if system.periodic_states:
            # for periodic states we use the range [-pi, pi]
            for i in system.periodic_states:
                lower_bound = lower_bound.at[i].set(-jnp.pi)
                upper_bound = upper_bound.at[i].set(+jnp.pi)

        return SamplerUniform.factory(
            batch_size=config.batch_size,
            lower_bound=lower_bound * 1.1,  # 10% margin
            upper_bound=upper_bound * 1.1,  # 10% margin
            rngs=nnx.Rngs(config.seed),
        )

    @staticmethod
    def build_state_to_rollout(
        config: WandbConfig, system: System
    ) -> StateToRolloutBase:
        if config.nn_name == "action":
            return StateToRolloutActions.factory(
                state_dim=system.sys_dynamics.state_dim,
                action_dim=system.sys_dynamics.action_dim,
                hidden_features=[config.nn_hidden_dim] * config.nn_hidden_depth,
                horizon=config.projection_horizon,
                sys_dynamics=system.sys_dynamics,
                rngs=nnx.Rngs(config.seed),
                activation=config.nn_activation,
                dropout_rate=config.nn_dropout_rate,
                periodic_states=system.periodic_states,
                action_min=system.action_min,
                action_max=system.action_max,
            )
        else:
            raise ValueError(
                f"Unknown state to rollout type: {config.state_to_rollout_name}"
            )

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
    def build_projection_layer(
        config: WandbConfig,
        system: System,
        solver: OptimizationSolverBase,
    ) -> ProjectionLayer:
        return ProjectionLayer.factory(
            sys_dynamics=system.sys_dynamics,
            sys_constraints=system.sys_constraints,
            horizon=config.projection_horizon,
            solver=solver,
        )

    @staticmethod
    def build_optimizer(
        config: WandbConfig, state_to_rollout: StateToRolloutBase
    ) -> nnx.Optimizer:
        return nnx.Optimizer(
            state_to_rollout,
            optax.nadamw(
                learning_rate=config.optimizer_learning_rate,
                b1=config.optimizer_b1,
                b2=config.optimizer_b2,
                weight_decay=config.optimizer_weight_decay,
            ),
        )

    @staticmethod
    def factory_from_config(
        config: WandbConfig,
    ) -> "TrainBundle":
        # Build the system components
        if config.system_name == "double_integrator":
            system = TrainBundle.build_double_integrator(config)
        elif config.system_name == "double_pendulum":
            system = TrainBundle.build_double_pendulum(config)
        elif config.system_name == "bike":
            system = TrainBundle.build_bike(config)
        else:
            raise ValueError(f"Unknown system: {config.system_name}")

        # Build the sampler
        sampler = TrainBundle.build_sampler(config, system)

        # Build the state to rollout module
        state_to_rollout = TrainBundle.build_state_to_rollout(
            config=config,
            system=system,
        )

        # Build optimizer
        optimizer = TrainBundle.build_optimizer(
            config=config,
            state_to_rollout=state_to_rollout,
        )

        # Build the solver
        solver = TrainBundle.build_solver(config)

        # Build the projection layer
        projection_layer = TrainBundle.build_projection_layer(
            config=config,
            system=system,
            solver=solver,
        )

        # Build optax metrics
        metrics = nnx.MultiMetric(
            # rollout metrics
            cost=nnx.metrics.Average("cost"),
            difference=nnx.metrics.Average("difference"),
            loss=nnx.metrics.Average("loss"),
            # solver metrics
            primal_residual=nnx.metrics.Average("primal_residual"),
            dual_residual=nnx.metrics.Average("dual_residual"),
            solver_iter_lin=nnx.metrics.Average("solver_iter_lin"),
            solver_iter_nonlin=nnx.metrics.Average("solver_iter_nonlin"),
            solver_converged=nnx.metrics.Average("solver_converged"),
        )

        return TrainBundle(
            system=system,
            sampler=sampler,
            state_to_rollout=state_to_rollout,
            optimizer=optimizer,
            projection_layer=projection_layer,
            metrics=metrics,
        )
