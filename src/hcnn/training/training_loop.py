import os

import wandb

from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_costs import SystemCostsBase
from hcnn.system_dynamics import SystemDynamicsBase


def build_system(
    system_name: str,
) -> tuple[SystemDynamicsBase, SystemCostsBase, SystemConstraintsBase]:
    """
    Build the system based on the provided system name.
    """
    if system_name == "double_integrator":
        # Example: return DoubleIntegratorSystem()
        pass
    else:
        raise ValueError(f"Unknown system: {system_name}")


def train(project=None, config=None):
    with wandb.init(project=project, config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        system_name = config.system_name

        wandb.log({"test_out": 42})


if __name__ == "__main__":
    print("This is a training loop with Weights & Biases integration.")

    # restrict access to one GPU (check availability nvidia-smi)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Specify the project name and configuration
    # This can be overridden by wandb.agent when running a sweep
    project = "double_integrator"
    config = {
        "system_name": "double_integrator",
    }

    # Start the training loop
    train(project=project, config=config)
