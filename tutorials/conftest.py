# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""pytest configuration file for tutorials tests."""

import os
import typing

import _pytest.config
import _pytest.main
import _pytest.nodes
import nbvalx.pytest_hooks_notebooks


def pytest_addoption(parser: _pytest.main.Parser) -> None:
    """Add options to set the number of processes and whether to run mesh generation notebooks or tutorials."""
    nbvalx.pytest_hooks_notebooks.addoption(parser)
    parser.addoption("--meshgen", action="store_true", help="run mesh generation notebooks")


def pytest_collection_modifyitems(
    session: _pytest.main.Session, config: _pytest.config.Config, items: typing.List[_pytest.nodes.Item]
) -> None:
    """Deselect notebooks based on the value of the --meshgen option."""
    mesh_generation_items = list()
    tutorial_items = list()
    for item in items:
        if "generate_mesh" in item.parent.name:
            mesh_generation_items.append(item)
        else:
            tutorial_items.append(item)
    if config.getoption("--meshgen"):
        config.hook.pytest_deselected(items=tutorial_items)
        items[:] = mesh_generation_items
    else:
        config.hook.pytest_deselected(items=mesh_generation_items)
        items[:] = tutorial_items


def pytest_runtest_setup(item: _pytest.nodes.Item) -> None:
    """Insert skips on cell failure."""
    nbvalx.pytest_hooks_notebooks.runtest_setup(item)
    # Create a symbolic link to the data folder when running on the temporary ipyparallel copy of the notebook
    if item.config.option.np > 1 and item.name == "Cell 0":
        dest_data = str(item.parent.fspath.new(basename="data"))
        src_data = dest_data.replace(os.sep + ".ipynb_mpi" + os.sep, os.sep)
        if os.path.exists(dest_data):
            os.unlink(dest_data)
        if os.path.exists(src_data):
            os.symlink(src_data, dest_data)


pytest_sessionstart = nbvalx.pytest_hooks_notebooks.sessionstart
pytest_collect_file = nbvalx.pytest_hooks_notebooks.collect_file
pytest_runtest_makereport = nbvalx.pytest_hooks_notebooks.runtest_makereport
pytest_runtest_teardown = nbvalx.pytest_hooks_notebooks.runtest_teardown
