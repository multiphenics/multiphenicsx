# Copyright (C) 2016-2023 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

FROM dolfinx/dolfinx:nightly AS multiphenicsx
MAINTAINER Francesco Ballarin <francesco.ballarin@unicatt.it>

COPY . /tmp/multiphenicsx

WORKDIR /tmp/multiphenicsx
RUN python3 -m pip install --upgrade pip && python3 -m pip -q install .[tests]

WORKDIR /root
RUN mkdir multiphenicsx && \
    mv /tmp/multiphenicsx/tests multiphenicsx/ && \
    rm -rf /tmp/multiphenicsx

########################################

FROM dolfinx/lab:nightly AS jupyter
MAINTAINER Francesco Ballarin <francesco.ballarin@unicatt.it>

COPY . /tmp/multiphenicsx

WORKDIR /tmp/multiphenicsx
RUN python3 -m pip install --upgrade pip && python3 -m pip -q install .[tests,tutorials]

WORKDIR /root
RUN mkdir multiphenicsx && \
    mv /tmp/multiphenicsx/tests multiphenicsx/ && \
    mv /tmp/multiphenicsx/tutorials multiphenicsx/ && \
    rm -rf /tmp/multiphenicsx
