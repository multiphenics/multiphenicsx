# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Utility functions to be used in pytest configuration file for notebooks tests.

Such functions are mainly for starting a ipyparallel Cluster when running notebooks tests in parallel.

See also: https://github.com/pytest-dev/pytest/blob/main/src/_pytest/hookspec.py for type hints.
"""

import os
import typing

import nbformat
import nbval.plugin

try:
    import _pytest.config
    import _pytest.main
    import _pytest.nodes
    import _pytest.runner
    import py
    import pytest
except ImportError:  # pragma: no cover
    addoption = None
    collect_file = None
    runtest_setup = None
    runtest_makereport = None
    runtest_teardown = None
else:
    def addoption(parser: _pytest.main.Parser) -> None:
        """Add option to set the number of processes."""
        parser.addoption("--np", action="store", type=int, default=1, help="Number of MPI processes to use")
        assert (
            not ("OMPI_COMM_WORLD_SIZE" in os.environ  # OpenMPI
                 or "MPI_LOCALNRANKS" in os.environ)), (  # MPICH
            "Please do not start pytest under mpirun. Use the --np pytest option.")

    def collect_file(path: py.path.local, parent: _pytest.nodes.Collector) -> None:
        """Collect IPython notebooks using a custom pytest nbval hook."""
        opt = parent.config.option
        assert not opt.nbval, "--nbval is implicitly enabled, do not provide it on the command line"
        if path.fnmatch("**/*.ipynb") and not path.fnmatch("**/.ipynb_mpi/*.ipynb"):
            if opt.np > 1:
                # Read in notebook
                with open(path) as f:
                    nb = nbformat.read(f, as_version=4)
                # Add the %%px magic to every existing cell
                for cell in nb.cells:
                    if cell.cell_type == "code":
                        cell.source = "%%px\n" + cell.source
                # Add a cell on top to start a new ipyparallel cluster
                cluster_start_code = f"""import ipyparallel as ipp
cluster = ipp.Cluster(engines="MPI", profile="mpi", n={opt.np})
cluster.start_and_connect_sync()"""
                cluster_start_cell = nbformat.v4.new_code_cell(cluster_start_code)
                cluster_start_cell.id = "cluster_start"
                nb.cells.insert(0, cluster_start_cell)
                # Add a further cell on top to disable garbage collection
                gc_disable_code = """%%px
import gc
gc.disable()"""
                gc_disable_cell = nbformat.v4.new_code_cell(gc_disable_code)
                gc_disable_cell.id = "gc_disable"
                nb.cells.insert(1, gc_disable_cell)
                # Add a cell at the end to re-enable garbage collection
                gc_enable_code = """%%px
gc.enable()
gc.collect()"""
                gc_enable_cell = nbformat.v4.new_code_cell(gc_enable_code)
                gc_enable_cell.id = "gc_enable"
                nb.cells.append(gc_enable_cell)
                # Add a cell at the end to stop the ipyparallel cluster
                cluster_stop_code = """cluster.stop_cluster_sync()"""
                cluster_stop_cell = nbformat.v4.new_code_cell(cluster_stop_code)
                cluster_stop_cell.id = "cluster_stop"
                nb.cells.append(cluster_stop_cell)
                # Write modified notebook to a temporary file
                mpi_dir = os.path.join(path.dirname, ".ipynb_mpi")
                os.makedirs(mpi_dir, exist_ok=True)
                ipynb_path = path.new(dirname=mpi_dir)
                with open(ipynb_path, "w") as f:
                    nbformat.write(nb, str(ipynb_path))
            else:  # pragma: no cover
                ipynb_path = path
            return nbval.plugin.IPyNbFile.from_parent(parent, fspath=ipynb_path)

    def runtest_setup(item: _pytest.nodes.Item) -> None:
        """Insert skips on cell failure."""
        # Do the normal setup
        item.setup()
        # If previous cells in a notebook failed skip the rest of the notebook
        if hasattr(item, "_force_skip"):
            if not hasattr(item.cell, "id") or item.cell.id not in ("gc_enable", "cluster_stop"):
                pytest.skip("A previous cell failed")

    def runtest_makereport(item: _pytest.nodes.Item, call: _pytest.runner.CallInfo[None]) -> None:
        """Determine whether the current cell failed or not."""
        if call.when == "call":
            if call.excinfo:
                np = item.config.option.np
                source = item.cell.source
                if np > 1 and source.startswith("%%px"):
                    source = source.replace("%%px\n", "")
                if source.startswith("# PYTEST_XFAIL"):
                    xfail_line = source.splitlines()[0]
                    xfail_comment = xfail_line.replace("# ", "")
                    xfail_marker, xfail_reason = xfail_comment.split(": ")
                    assert xfail_marker in (
                        "PYTEST_XFAIL", "PYTEST_XFAIL_IN_PARALLEL",
                        "PYTEST_XFAIL_AND_SKIP_NEXT", "PYTEST_XFAIL_IN_PARALLEL_AND_SKIP_NEXT")
                    if (xfail_marker in ("PYTEST_XFAIL", "PYTEST_XFAIL_AND_SKIP_NEXT")
                        or (xfail_marker in ("PYTEST_XFAIL_IN_PARALLEL", "PYTEST_XFAIL_IN_PARALLEL_AND_SKIP_NEXT")
                            and np > 1)):
                        # This failure was expected: report the reason of xfail.
                        call.excinfo._excinfo = (
                            call.excinfo._excinfo[0],
                            pytest.xfail.Exception(xfail_reason.capitalize()),
                            call.excinfo._excinfo[2])
                    if xfail_marker in ("PYTEST_XFAIL_AND_SKIP_NEXT", "PYTEST_XFAIL_IN_PARALLEL_AND_SKIP_NEXT"):
                        # The failure, even though expected, forces the rest of the notebook to be skipped.
                        item._force_skip = True
                else:  # pragma: no cover
                    # An unexpected error forces the rest of the notebook to be skipped.
                    item._force_skip = True

    def runtest_teardown(item: _pytest.nodes.Item, nextitem: typing.Optional[_pytest.nodes.Item]) -> None:
        """Propagate cell failure."""
        # Do the normal teardown
        item.teardown()
        # Inform next cell of the notebook of failure of any previous cells
        if hasattr(item, "_force_skip"):
            if nextitem is not None and nextitem.name != "Cell 0":
                nextitem._force_skip = True
