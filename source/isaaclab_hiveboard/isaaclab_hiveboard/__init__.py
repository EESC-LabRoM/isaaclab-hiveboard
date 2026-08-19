# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module for the HiveBoard multi-robot manipulation extension.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions if Omniverse Kit is running.
try:
    from .ui_extension_example import *
except (ImportError, ModuleNotFoundError):
    pass
