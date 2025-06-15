from typing import Callable, Tuple

import jax
from flax import struct

from hcnn.optimization import OptimizationProblemBase


@struct.dataclass
class Rollout:
    states: jax.Array
    actions: jax.Array
    interventions: jax.Array
    relaxations_plus: jax.Array
    relaxations_minus: jax.Array


@struct.dataclass
class ProjectionPhi:
    initial_state: jax.Array
    split_policy: Tuple
    split_dynamics: Tuple
    split_constraints: Tuple
    jax_key: jax.Array = struct.field(pytree_node=False)
    horizon: int = struct.field(pytree_node=False)
    x_to_rollout: Callable[[jax.Array], Rollout] = struct.field(pytree_node=False)


@struct.dataclass
class ProjectionBase(OptimizationProblemBase[ProjectionPhi]):
    pass
