# Copyright (C) 2016-2021 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import gc
from mpi4py import MPI


def pytest_runtest_setup(item):
    # Do the normal setup
    item.setup()
    # Disable garbage collection
    gc.disable()


def pytest_runtest_teardown(item, nextitem):
    # Do the normal teardown
    item.teardown()
    # Re-enable garbage collection
    gc.enable()
    # Run garbage gollection
    del item
    gc.collect()
    # Add a MPI barrier in parallel
    MPI.COMM_WORLD.Barrier()
