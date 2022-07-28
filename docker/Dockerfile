# Copyright (C) 2016-2022 by the multiphenicsx authors
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
RUN python3 -m pip install --upgrade pip && python3 -m pip install --upgrade https://dmf.unicatt.it/ballarin/public/zstandard-0.19.0.dev0-cp310-cp310-linux_x86_64.whl && python3 -m pip install --upgrade https://dmf.unicatt.it/ballarin/public/itkwidgets-0.33.0.dev0-py2.py3-none-any.whl && python3 -m pip -q install .[tests,tutorials]

WORKDIR /root
RUN mkdir multiphenicsx && \
    mv /tmp/multiphenicsx/tests multiphenicsx/ && \
    mv /tmp/multiphenicsx/tutorials multiphenicsx/ && \
    rm -rf /tmp/multiphenicsx

# We use jupyter notebook rather than jupyter lab because itkwidgets are not yet supported in jupyter lab 3.
ENTRYPOINT ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
