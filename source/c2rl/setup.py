# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'c2rl' python package."""

import os

import toml
from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "psutil",
    "torch",
    "skrl>=2.1.0",
    "wandb",
    "gymnasium",
    "scipy",
    # convex (SDP) solve for C2RL's CMG synthesis (cmg_method="cvstem"; agents/skrl/ncm_synthesis.py).
    # Pinned <1.7: cvxpy 1.7+ hard-requires numpy>=2.0 and osqp>=1.0, which conflict with the
    # Isaac Sim/Lab pins (numpy<2, osqp==0.6.7.post3). 1.6.x supports numpy 1.26 + osqp 0.6.x and
    # still ships the SCS solver C2RL uses.
    "cvxpy<1.7",
    "matplotlib",
    "tensorboard",
    # Optional interior-point SDP solver for C2RL's contraction-metric SDP
    # (cm_solver: MOSEK in the yaml's `cm:` block; default is cm_solver: SCS,
    # which needs no license). The `mosek` package installs fine on its own,
    # but actually solving with it needs a license file — see README.md's
    # Installation section for how to get and install one.
    "mosek",
]

# Installation operation
setup(
    name="c2rl",
    packages=["c2rl"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
