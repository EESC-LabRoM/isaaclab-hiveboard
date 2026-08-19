# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available HiveBoard environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="List Isaac Lab environments.")
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import isaaclab_hiveboard.tasks  # noqa: F401


def main():
    """Print all environments registered in `isaaclab_hiveboard` extension."""
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available HiveBoard Environments in Isaac Lab"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        is_hiveboard = (
            "HiveBoard" in task_spec.id
            or "Spot-Manipulation" in task_spec.id
            or "Franka-Manipulation" in task_spec.id
            or "Template-" in task_spec.id
        )
        if is_hiveboard and (args_cli.keyword is None or args_cli.keyword.lower() in task_spec.id.lower()):
            config_entry = task_spec.kwargs.get("env_cfg_entry_point", "N/A") if task_spec.kwargs else "N/A"
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, config_entry])
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        simulation_app.close()
