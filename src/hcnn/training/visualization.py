import jax
import jax.numpy as jnp
import plotly.graph_objects as go
from flax import nnx
from plotly.subplots import make_subplots

import wandb
from hcnn.projection import (
    ProjectionPhiD,
    ProjectionPhiND,
)

from .train_bundle import TrainBundle


def sample_rollout(train_bundle: TrainBundle, initial_state: jax.Array):
    policy_params, policy_non_params = nnx.state(
        train_bundle.policy, nnx.Param, ...
    )
    relax_to_go_params, relax_to_go_non_params = nnx.state(
        train_bundle.relax_to_go, nnx.Param, ...
    )

    phi_d = ProjectionPhiD(
        policy_params=policy_params,
        relax_to_go_params=relax_to_go_params,
    )
    phi_nd = ProjectionPhiND(
        policy_non_params=policy_non_params,
        relax_to_go_non_params=relax_to_go_non_params,
        initial_state=initial_state,
        jax_key=jax.random.key(0),
    )
    x_star, _ = train_bundle.loss_and_grad.solver.solve(
        train_bundle.loss_and_grad.projection, phi_d, phi_nd
    )
    rollout = train_bundle.loss_and_grad.projection.x_to_rollout(x_star)

    return rollout.states, rollout.actions


def visualize_receding_horizon(
    train_bundle: TrainBundle,
    steps: int = 50,
    initial_state: jax.Array | None = None,
):
    """
    Visualize the receding horizon evolution.
    At each step, show the full planned rollout and the actual step taken.
    """
    # make everything in bundle deterministic
    train_bundle.eval()

    sys_dynamics = train_bundle.loss_and_grad.projection.sys_dynamics
    state_dim = sys_dynamics.state_dim
    horizon = train_bundle.loss_and_grad.projection.horizon

    # If no initial state is provided, use zero state
    if initial_state is None:
        initial_state = jnp.zeros(sys_dynamics.state_dim)

    # Create subplots - one for each state dimension
    fig = make_subplots(
        rows=state_dim,
        cols=1,
        subplot_titles=[f"State {i + 1}" for i in range(state_dim)],
        vertical_spacing=0.1,
    )

    # Track the actual trajectory taken
    actual_states = [initial_state]
    current_state = initial_state

    # Colors for visualization
    light_color = "rgba(100, 149, 237, 0.3)"  # Light blue for planned
    dark_color = "rgba(0, 0, 139, 1.0)"  # Dark blue for actual

    # Simulate receding horizon for the specified number of steps
    for step in range(steps):
        # Get rollout from current state
        planned_states, planned_actions = sample_rollout(
            train_bundle, initial_state=current_state
        )

        # Time steps for this rollout (offset by current step)
        rollout_time_steps = jnp.arange(horizon) + step

        # Add planned trajectory (light color)
        for state_idx in range(state_dim):
            fig.add_trace(
                go.Scatter(
                    x=rollout_time_steps,
                    y=planned_states[:, state_idx],
                    mode="lines",
                    line=dict(color=light_color, width=1),
                    showlegend=False,
                ),
                row=state_idx + 1,
                col=1,
            )

        # Take the first action and advance the state
        first_action = planned_actions[0]
        next_state = sys_dynamics(current_state, first_action)

        # Add the actual transition (dark color)
        for state_idx in range(state_dim):
            fig.add_trace(
                go.Scatter(
                    x=[step, step + 1],
                    y=[current_state[state_idx], next_state[state_idx]],
                    mode="lines+markers",
                    line=dict(color=dark_color, width=3),
                    marker=dict(size=6, color=dark_color),
                    showlegend=False,
                ),
                row=state_idx + 1,
                col=1,
            )

        # Update current state and add to actual trajectory
        current_state = next_state
        actual_states.append(current_state)

    # Update layout
    fig.update_layout(
        title="Receding Horizon State Trajectory Visualization",
        height=300 * state_dim,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    # Update x-axis labels for all subplots
    for i in range(state_dim):
        fig.update_xaxes(title_text="Time Step", row=i + 1, col=1)
        fig.update_yaxes(title_text=f"State {i + 1} Value", row=i + 1, col=1)

    # Log to wandb
    wandb.log({"receding_horizon_visualization": fig})

    print(f"Simulated {len(actual_states)} steps with {state_dim} states")
    print("Receding horizon visualization logged to wandb")

    # restore non deterministic training mode
    train_bundle.train()
