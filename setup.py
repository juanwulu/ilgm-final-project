# !/usr/bin/env python
# Copyright (c) 2024, Juanwu Lu.
# Released under the BSD 3-Clause License.
# Please see the LICENSE file that should have been included as part of this
# project source code.
from src._metadata import version
from setuptools import find_packages, setup

setup(name="src", version=version, packages=find_packages())
