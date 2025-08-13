import jax
import jax.numpy as jnp
import plotly.graph_objects as go
from flax import nnx
from plotly.subplots import make_subplots

import wandb

from .train_bundle import TrainBundle


@nnx.jit
def sample_rollout(train_bundle: TrainBundle, initial_states: jax.Array):
    rollout = train_bundle.state_to_rollout(initial_states)
    rollout_proj, _ = train_bundle.projection_layer(rollout, initial_states)
    return rollout_proj


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

    sys_dynamics = train_bundle.system.sys_dynamics
    state_dim = sys_dynamics.state_dim
    action_dim = sys_dynamics.action_dim
    horizon = train_bundle.projection_layer.horizon

    # If no initial state is provided, use zero state
    if initial_state is None:
        initial_state = jnp.zeros(sys_dynamics.state_dim)

    # Create subplots - one for each state dimension, action dimension,
    # and max relaxation
    total_plots = state_dim + action_dim + 1  # +1 for max relaxation
    subplot_titles = (
        [f"State {i + 1}" for i in range(state_dim)]
        + [f"Action {i + 1}" for i in range(action_dim)]
        + ["Max Relaxation"]
    )

    fig = make_subplots(
        rows=total_plots,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
    )

    # Track the actual trajectory taken
    actual_states = [initial_state]
    actual_actions = []
    actual_max_relaxations = []  # Track actual max relaxations
    current_state = initial_state

    # Colors for visualization
    light_color = "rgba(100, 149, 237, 0.3)"  # Light blue for planned
    dark_color = "rgba(0, 0, 139, 1.0)"  # Dark blue for actual
    action_light_color = (
        "rgba(255, 165, 0, 0.3)"  # Light orange for planned actions
    )
    action_dark_color = (
        "rgba(255, 140, 0, 1.0)"  # Dark orange for actual actions
    )

    # Simulate receding horizon for the specified number of steps
    for step in range(steps):
        # Get rollout from current state
        rollout = sample_rollout(train_bundle, initial_states=current_state)
        planned_states = rollout.states
        planned_actions = rollout.actions
        relaxations = rollout.relax

        # Time steps for this rollout (offset by current step)
        rollout_time_steps = jnp.arange(horizon) + step
        action_time_steps = (
            jnp.arange(horizon - 1) + step
        )  # Actions are one step shorter

        # Calculate max relaxation over all constraints for this rollout
        max_relaxation = jnp.max(
            jnp.abs(relaxations), axis=-1
        )  # Max over constraint dim

        # Add planned state trajectory (light color)
        for state_idx in range(state_dim):
            fig.add_trace(
                go.Scatter(
                    x=rollout_time_steps,
                    y=planned_states[:, state_idx],
                    mode="lines",
                    line=dict(color=light_color, width=1),
                    name="States" if step == 0 and state_idx == 0 else None,
                    showlegend=True if step == 0 and state_idx == 0 else False,
                ),
                row=state_idx + 1,
                col=1,
            )

        # Add max relaxation trajectory (light purple)
        fig.add_trace(
            go.Scatter(
                x=rollout_time_steps,
                y=max_relaxation,
                mode="lines",
                line=dict(color="rgba(138, 43, 226, 0.3)", width=1),
                name="Planned Max Relaxation" if step == 0 else None,
                showlegend=True if step == 0 else False,
            ),
            row=total_plots,  # Last subplot for max relaxation
            col=1,
        )

        # Add planned action trajectory (light orange)
        for action_idx in range(action_dim):
            fig.add_trace(
                go.Scatter(
                    x=action_time_steps,
                    y=planned_actions[
                        :-1, action_idx
                    ],  # Exclude last action as it's not used
                    mode="lines",
                    line=dict(color=action_light_color, width=1),
                    name="Planned Actions"
                    if step == 0 and action_idx == 0
                    else None,
                    showlegend=True if step == 0 and action_idx == 0 else False,
                ),
                row=state_dim + action_idx + 1,
                col=1,
            )

        # Take the first action and advance the state
        first_action = planned_actions[0]
        next_state = sys_dynamics(current_state, first_action)

        # Calculate the actual max relaxation for the current step
        current_max_relaxation = jnp.max(jnp.abs(relaxations[0]))

        # Add the actual state transition (dark color)
        for state_idx in range(state_dim):
            fig.add_trace(
                go.Scatter(
                    x=[step, step + 1],
                    y=[current_state[state_idx], next_state[state_idx]],
                    mode="lines+markers",
                    line=dict(color=dark_color, width=3),
                    marker=dict(size=6, color=dark_color),
                    name="Actual States"
                    if step == 0 and state_idx == 0
                    else None,
                    showlegend=True if step == 0 and state_idx == 0 else False,
                ),
                row=state_idx + 1,
                col=1,
            )

        # Add the actual max relaxation (dark purple)
        fig.add_trace(
            go.Scatter(
                x=[step],
                y=[current_max_relaxation],
                mode="markers",
                marker=dict(size=6, color="rgba(75, 0, 130, 1.0)"),
                name="Actual Max Relaxation" if step == 0 else None,
                showlegend=True if step == 0 else False,
            ),
            row=total_plots,  # Last subplot for max relaxation
            col=1,
        )

        # Add the actual action taken (dark orange)
        for action_idx in range(action_dim):
            fig.add_trace(
                go.Scatter(
                    x=[step],
                    y=[first_action[action_idx]],
                    mode="markers",
                    marker=dict(size=6, color=action_dark_color),
                    name="Actual Actions"
                    if step == 0 and action_idx == 0
                    else None,
                    showlegend=True if step == 0 and action_idx == 0 else False,
                ),
                row=state_dim + action_idx + 1,
                col=1,
            )

        # Update current state and add to actual trajectory
        current_state = next_state
        actual_states.append(current_state)
        actual_actions.append(first_action)
        actual_max_relaxations.append(current_max_relaxation)

    # Update layout with padding
    fig.update_layout(
        title="Receding Horizon State, Action, and Max Relaxation "
        "Visualization",
        height=500 * total_plots,
        margin=dict(l=80, r=80, t=100, b=80),
    )

    # Update x-axis labels for all subplots with padding
    for i in range(state_dim):
        fig.update_xaxes(title_text="Time Step", row=i + 1, col=1)
        fig.update_yaxes(
            title_text=f"State {i + 1} Value", row=i + 1, col=1, automargin=True
        )

    for i in range(action_dim):
        fig.update_xaxes(title_text="Time Step", row=state_dim + i + 1, col=1)
        fig.update_yaxes(
            title_text=f"Action {i + 1} Value",
            row=state_dim + i + 1,
            col=1,
            automargin=True,
        )

    # Update max relaxation subplot axes
    fig.update_xaxes(title_text="Time Step", row=total_plots, col=1)
    fig.update_yaxes(
        title_text="Max Relaxation",
        row=total_plots,
        col=1,
        automargin=True,
    )

    # Log to wandb
    wandb.log({"receding_horizon_visualization": fig})

    print(
        f"Simulated {len(actual_states)} steps with"
        f" {state_dim} states, {action_dim} actions, and max relaxation"
    )
    print("Receding horizon visualization with max relaxation logged to wandb")

    # restore non deterministic training mode
    train_bundle.train()
