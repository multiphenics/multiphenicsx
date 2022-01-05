# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""pytest configuration file for unit tests."""

import multiphenicsx.test.unit_tests

pytest_runtest_setup = multiphenicsx.test.unit_tests.runtest_setup
pytest_runtest_teardown = multiphenicsx.test.unit_tests.runtest_teardown
