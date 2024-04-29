# Copyright (c) 2024, Juanwu Lu.
# Released under the BSD 3-Clause License.
# Please see the LICENSE file that should have been included as part of this
# project source code.
import pyrootutils

from . import _metadata

_ = pyrootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=True
)
__author__ = _metadata.author
__version__ = _metadata.version
__docformat__ = "google"
__doc__ = """
Learning Stochastic Driver Behavior: A variational inference approach
=====================================================================
This repository is the official implementation of the course project
for Purdue University ECE69500: Inference & Learning in Generative Models.
"""
