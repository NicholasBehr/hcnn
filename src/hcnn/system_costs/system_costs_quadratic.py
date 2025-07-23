import jax
from flax import struct
from jax import numpy as jnp

from hcnn.system_costs.system_costs_base import SystemCostsBase


@struct.dataclass
class SystemCostsQuadratic(SystemCostsBase):
    state_cost_matrix: jax.Array  # float 2d matrix
    action_cost_matrix: jax.Array  # float 2d matrix
    reference_state: jax.Array  # float 1d vector
    reference_action: jax.Array  # float 1d vector
    periodic_states: frozenset[int] | None  # indices of 2π periodic states

    @classmethod
    def factory(
        cls,
        state_cost_matrix: jax.Array,
        action_cost_matrix: jax.Array,
        reference_state: jax.Array | None = None,
        reference_action: jax.Array | None = None,
        periodic_states: frozenset[int] | None = None,
    ):
        # Get JAX's default float type to respect 32/64 bit config
        float_dtype = jnp.ones(1).dtype

        # Cast to floating respecting JAX config
        state_cost_matrix = jnp.asarray(state_cost_matrix, dtype=float_dtype)
        action_cost_matrix = jnp.asarray(action_cost_matrix, dtype=float_dtype)

        # ensure 2d
        state_cost_matrix = jnp.atleast_2d(state_cost_matrix)
        action_cost_matrix = jnp.atleast_2d(action_cost_matrix)

        # verify exacly 2d
        two_dim = 2
        assert jnp.ndim(state_cost_matrix) == two_dim
        assert jnp.ndim(action_cost_matrix) == two_dim

        state_dim = state_cost_matrix.shape[1]
        action_dim = action_cost_matrix.shape[1]

        # verify square
        assert state_cost_matrix.shape == (state_dim, state_dim)
        assert action_cost_matrix.shape == (action_dim, action_dim)

        # Handle reference state and action
        if reference_state is None:
            reference_state = jnp.zeros(state_dim, dtype=float_dtype)
        else:
            reference_state = jnp.asarray(reference_state, dtype=float_dtype)
            assert reference_state.shape == (state_dim,)

        if reference_action is None:
            reference_action = jnp.zeros(action_dim, dtype=float_dtype)
        else:
            reference_action = jnp.asarray(reference_action, dtype=float_dtype)
            assert reference_action.shape == (action_dim,)

        # Validate periodic_states indices
        if periodic_states is not None:
            for idx in periodic_states:
                assert 0 <= idx < state_dim, (
                    f"Periodic state index {idx} out of range [0, {state_dim})"
                )

        return SystemCostsQuadratic(
            state_dim=state_dim,
            action_dim=action_dim,
            state_cost_matrix=state_cost_matrix,
            action_cost_matrix=action_cost_matrix,
            reference_state=reference_state,
            reference_action=reference_action,
            periodic_states=periodic_states,
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        # Compute deviations from reference
        state_deviation = state - self.reference_state

        # Handle periodic states: wrap deviations to [-π, π]
        if self.periodic_states is not None:
            for idx in self.periodic_states:
                # Use jax.numpy for JAX compatibility
                state_deviation = state_deviation.at[..., idx].set(
                    jnp.arctan2(
                        jnp.sin(state_deviation[..., idx]),
                        jnp.cos(state_deviation[..., idx]),
                    )
                )

        action_deviation = action - self.reference_action

        # Compute quadratic costs: x^T Q x for both state and action deviations
        state_cost = jnp.sum(
            (state_deviation @ self.state_cost_matrix) * state_deviation,
            axis=-1,
        )
        action_cost = jnp.sum(
            (action_deviation @ self.action_cost_matrix) * action_deviation,
            axis=-1,
        )
        return state_cost + action_cost
