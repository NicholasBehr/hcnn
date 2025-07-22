from enum import Enum

import optax


class Optimizer(Enum):
    adamw = optax.adamw


print("Testing enum:")
print(f"optax.adamw: {optax.adamw}")
print(f"Optimizer members: {list(Optimizer)}")
print(f"Optimizer._member_map_: {Optimizer._member_map_}")
try:
    print(f"Optimizer['adamw']: {Optimizer['adamw']}")
except KeyError as e:
    print(f"KeyError: {e}")

try:
    print(f"Optimizer.adamw: {Optimizer.adamw}")
except AttributeError as e:
    print(f"AttributeError: {e}")
