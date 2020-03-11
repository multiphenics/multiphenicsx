# Copyright (C) 2016-2020 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

import os
import re
import importlib
import pytest
import pytest_flake8
import matplotlib.pyplot as plt  # TODO remove after transition to ipynb is complete?
from nbconvert.exporters import PythonExporter
import nbconvert.filters
from dolfinx import MPI
plt.switch_backend("Agg")  # TODO remove after transition to ipynb is complete?


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
        if MPI.rank(MPI.comm_world) == 0:
            with open(path.new(ext=".py"), "w", encoding="utf-8") as f:
                f.write(code)
        # Collect the corresponding .py file
        config = parent.config
        if config.getoption("--flake8"):
            return pytest_flake8.pytest_collect_file(path.new(ext=".py"), parent)
        else:
            if "data" not in path.dirname:  # skip running mesh generation notebooks
                return TutorialFile(path.new(ext=".py"), parent)
    elif path.ext == ".py":  # TODO remove after transition to ipynb is complete? assert never py files?
        if (path.basename not in "conftest.py"  # do not run pytest configuration file
                or "data" not in path.dirname):  # skip running mesh generation notebooks
            return TutorialFile(path, parent)


def pytest_pycollect_makemodule(path, parent):
    """
    Disable running .py files produced by previous runs, as they may get out of sync with the corresponding .ipynb file.
    """
    if path.ext == ".py":
        assert not path.new(ext=".ipynb").exists(), "Please run pytest on jupyter notebooks, not plain python files."
        return DoNothingFile(path, parent)  # TODO remove after transition to ipynb is complete?


def pytest_runtest_teardown(item, nextitem):
    # Do the normal teardown
    item.teardown()
    # Add a MPI barrier in parallel
    MPI.barrier(MPI.comm_world)


class TutorialFile(pytest.File):
    """
    Custom file handler for tutorial files.
    """

    def collect(self):
        yield TutorialItem("run_tutorial -> " + os.path.relpath(str(self.fspath), str(self.parent.fspath)), self)


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
