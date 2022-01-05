# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""pytest configuration file for notebooks tests."""

import multiphenicsx.test.notebooks

pytest_addoption = multiphenicsx.test.notebooks.addoption
pytest_collect_file = multiphenicsx.test.notebooks.collect_file
pytest_runtest_setup = multiphenicsx.test.notebooks.runtest_setup
pytest_runtest_makereport = multiphenicsx.test.notebooks.runtest_makereport
pytest_runtest_teardown = multiphenicsx.test.notebooks.runtest_teardown
