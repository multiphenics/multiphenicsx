# Copyright (C) 2016-2021 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import re
import importlib
import pytest
import pytest_flake8
import matplotlib.pyplot as plt
from nbconvert.exporters import PythonExporter
import nbconvert.filters
from mpi4py import MPI
plt.switch_backend("Agg")


def pytest_ignore_collect(path, config):
    if path.ext == ".py" and path.new(ext=".ipynb").exists():  # ignore .py files obtained from previous runs
        return True
    else:
        return False


def pytest_collect_file(path, parent):
    """
    Collect tutorial files.
    """
    if path.ext == ".ipynb":
        # Convert .ipynb notebooks to plain .py files
        def comment_lines(text, prefix="# "):
            regex = re.compile(r".{1,80}(?:\s+|$)")
            input_lines = text.split("\n")
            output_lines = [split_line.rstrip() for line in input_lines for split_line in regex.findall(line)]
            output = prefix + ("\n" + prefix).join(output_lines)
            return output.replace(prefix + "\n", prefix.rstrip(" ") + "\n")

        def ipython2python(code):
            return nbconvert.filters.ipython2python(code).rstrip("\n") + "\n"

        filters = {
            "comment_lines": comment_lines,
            "ipython2python": ipython2python
        }
        exporter = PythonExporter(filters=filters)
        exporter.exclude_input_prompt = True
        code, _ = exporter.from_filename(path)
        code = code.rstrip("\n") + "\n"
        if MPI.COMM_WORLD.rank == 0:
            with open(path.new(ext=".py"), "w", encoding="utf-8") as f:
                f.write(code)
        # Collect the corresponding .py file
        config = parent.config
        if config.getoption("--flake8"):
            return pytest_flake8.pytest_collect_file(path.new(ext=".py"), parent)
        else:
            if not path.basename.startswith("x"):
                return TutorialFile.from_parent(parent=parent, fspath=path.new(ext=".py"))
            else:
                return DoNothingFile.from_parent(parent=parent, fspath=path.new(ext=".py"))
    elif path.ext == ".py":
        assert not path.new(ext=".ipynb").exists(), "Please run pytest on jupyter notebooks, not plain python files."
        return DoNothingFile.from_parent(parent=parent, fspath=path)


def pytest_pycollect_makemodule(path, parent):
    """
    Disable running .py files produced by previous runs, as they may get out of sync with the corresponding .ipynb file.
    """
    if path.ext == ".py":
        assert not path.new(ext=".ipynb").exists(), "Please run pytest on jupyter notebooks, not plain python files."
        return DoNothingFile.from_parent(parent=parent, fspath=path)


def pytest_addoption(parser):
    parser.addoption("--meshgen", action="store_true", help="run mesh generation notebooks")


def pytest_collection_modifyitems(session, config, items):
    """
    Collect mesh generation notebooks first.
    """
    mesh_generation_items = list()
    tutorial_items = list()
    for item in items:
        if "generate_mesh" in item.name:
            mesh_generation_items.append(item)
        else:
            tutorial_items.append(item)
    if config.getoption("--flake8") or (config.getoption("--meshgen") and MPI.COMM_WORLD.size == 1):
        items[:] = mesh_generation_items + tutorial_items
    else:
        config.hook.pytest_deselected(items=mesh_generation_items)
        items[:] = tutorial_items


def pytest_runtest_teardown(item, nextitem):
    # Do the normal teardown
    item.teardown()
    # Add a MPI barrier in parallel
    MPI.COMM_WORLD.Barrier()


class TutorialFile(pytest.File):
    """
    Custom file handler for tutorial files.
    """

    def collect(self):
        yield TutorialItem.from_parent(
            parent=self, name=os.path.relpath(str(self.fspath), str(self.parent.fspath)))


class TutorialItem(pytest.Item):
    """
    Handle the execution of the tutorial.
    """

    def __init__(self, name, parent):
        super(TutorialItem, self).__init__(name, parent)

    def runtest(self):
        os.chdir(self.parent.fspath.dirname)
        spec = importlib.util.spec_from_file_location(self.name, str(self.parent.fspath))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plt.close("all")  # do not trigger matplotlib max_open_warning

    def reportinfo(self):
        return self.fspath, 0, self.name


class DoNothingFile(pytest.File):
    """
    Custom file handler to avoid running twice python files explicitly provided on the command line.
    """

    def collect(self):
        return []
