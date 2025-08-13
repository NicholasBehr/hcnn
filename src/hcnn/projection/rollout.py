import jax
import jax.numpy as jnp
from flax import struct

from hcnn.system_dynamics import SystemDynamicsBase


@struct.dataclass
class Rollout:
    states: jax.Array  # shape (..., horizon, state_dim)
    actions: jax.Array  # shape (..., horizon, action_dim)
    relax: jax.Array  # shape (..., horizon, constr_dim)

    @property
    def state_dim(self) -> int:
        """Get the state dimension from the states array."""
        return self.states.shape[-1]

    @property
    def action_dim(self) -> int:
        """Get the action dimension from the actions array."""
        return self.actions.shape[-1]

    @property
    def relax_dim(self) -> int:
        """Get the relaxation dimension from the relax array."""
        return self.relax.shape[-1]

    @staticmethod
    def from_actions(
        actions: jax.Array,
        sys_dynamics: SystemDynamicsBase,
        initial_state: jax.Array,
    ) -> "Rollout":
        """Compute states from actions using system dynamics.

        Args:
            actions: Action sequence, shape (..., horizon, action_dim)
            initial_state: Initial state, shape (..., state_dim)
            sys_dynamics: System dynamics model

        Returns:
            Rollout with states and actions
                states: Computed states, shape (..., horizon, state_dim)
                actions: Provided actions, shape (..., horizon, action_dim)
        Note:
            The returned states sequence contains the state at each time prior
            to applying the corresponding action (i.e. states[t] evolves to
            states[t+1] under actions[t]).
        """

        def step(carry, action):
            next_state = sys_dynamics(carry, action)
            return next_state, carry  # emit current state, carry next

        actions_shape = actions.shape
        batch_shape = actions_shape[:-2]  # leading batch dims (maybe empty)
        t = actions_shape[-2]

        if batch_shape:  # batched rollout
            # Flatten batch dims → single vmap for speed & clarity
            flat_batch = 1
            for d in batch_shape:
                flat_batch *= d
            action_dim = actions_shape[-1]
            state_dim = initial_state.shape[-1]

            actions_flat = actions.reshape((flat_batch, t, action_dim))
            init_flat = initial_state.reshape((flat_batch, state_dim))

            def rollout(init_s, act_seq):
                _, states_seq = jax.lax.scan(step, init_s, act_seq)
                return states_seq  # (t, state_dim)

            states_flat = jax.vmap(rollout)(
                init_flat, actions_flat
            )  # (B, t, state_dim)
            states = states_flat.reshape(batch_shape + (t, state_dim))
        else:  # single trajectory
            _, states = jax.lax.scan(
                step, initial_state, actions
            )  # (t, state_dim)

        # TODO
        return Rollout(
            states=states,
            actions=actions,
            relax=jnp.zeros_like(states),
        )
