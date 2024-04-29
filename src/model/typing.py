# Copyright (c) 2024, Juanwu Lu.
# Released under the BSD 3-Clause License.
# Please see the LICENSE file that should have been included as part of this
# project source code.
"""Common type definitions for the model module."""
from __future__ import annotations

from typing import Dict

from torch import Tensor

StateDict = Dict[str, Tensor]
