# Copyright (C) 2016-2021 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

FROM dolfinx/dolfinx
MAINTAINER Francesco Ballarin <francesco.ballarin@unicatt.it>

RUN pip3 -q install --upgrade itkwidgets matplotlib nbconvert Pygments pytest pytest-flake8 pytest-xdist pyvista sympy

COPY . /tmp/multiphenicsx

WORKDIR /tmp/multiphenicsx
RUN python3 setup.py -q install

WORKDIR $HOME
RUN mkdir multiphenicsx && \
    mv /tmp/multiphenicsx/tests multiphenicsx/ && \
    mv /tmp/multiphenicsx/tutorials multiphenicsx/ && \
    rm -rf /tmp/multiphenicsx
